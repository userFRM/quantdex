#!/usr/bin/env python3
"""
Experiment 05: Block Summary Table Pruning
============================================
Benchmarks the block summary table approach where keys are grouped into
blocks, each block stores min/max per coordinate, and queries prune entire
blocks before scoring individual keys.

Tests:
  1. Summary table size (should be ~640KB for n=1M)
  2. Pruning effectiveness (fraction of blocks pruned)
  3. End-to-end speedup (build -> prune -> score)
  4. Memory reads (bytes from HBM in each phase)
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_LIB_AVAILABLE = False
try:
    from quantdex.turbo_quant import TurboQuant
    from quantdex.sub_linear_attention import BlockPruning as _LibBlockPruning
    from quantdex.metrics import recall_at_k
    _LIB_AVAILABLE = True
except ImportError as exc:
    warnings.warn(
        f"Could not import quantdex library: {exc}\n"
        "Falling back to self-contained reference implementations.\n"
        "To use the real library, ensure ~/quantdex/src/ contains:\n"
        "  - turbo_quant.py          (TurboQuant class)\n"
        "  - sub_linear_attention.py (BlockPruning class)\n"
        "  - metrics.py              (recall_at_k function)\n"
    )

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'font.size': 12, 'figure.figsize': (8, 5), 'figure.dpi': 150})

FIGURES_DIR = _PROJECT_ROOT / 'figures'
RESULTS_DIR = Path(__file__).resolve().parent / 'results'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def _random_orthogonal(d: int, rng: np.random.RandomState) -> np.ndarray:
    H = rng.randn(d, d)
    Q, R = np.linalg.qr(H)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q


def _random_unit_vectors(n: int, d: int, rng: np.random.RandomState) -> np.ndarray:
    X = rng.randn(n, d)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _uniform_scalar_quantize(x: np.ndarray, b: int) -> np.ndarray:
    L = 2 ** b
    x_clipped = np.clip(x, -1.0, 1.0)
    x_01 = (x_clipped + 1.0) / 2.0
    idx = np.floor(x_01 * L).astype(np.int32)
    idx = np.clip(idx, 0, L - 1)
    return (idx + 0.5) / L * 2.0 - 1.0


def _sample_vmf(mu, kappa, n, rng):
    d = len(mu)
    mu = mu / np.linalg.norm(mu)
    noise = rng.randn(n, d) * (1.0 / np.sqrt(max(kappa, 1.0)))
    vecs = mu[None, :] + noise
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _generate_clustered_keys(n, d, n_clusters, kappa, rng):
    centers = _random_unit_vectors(n_clusters, d, rng)
    per_cluster = n // n_clusters
    remainder = n - per_cluster * n_clusters
    all_vecs = []
    for i in range(n_clusters):
        n_i = per_cluster + (1 if i < remainder else 0)
        vecs = _sample_vmf(centers[i], kappa, n_i, rng)
        all_vecs.append(vecs)
    result = np.concatenate(all_vecs, axis=0)
    result = result / np.linalg.norm(result, axis=1, keepdims=True)
    return result[rng.permutation(n)]


class _ReferenceBlockSummaryTable:
    """
    Block summary table for pruning.

    Keys are divided into blocks of size B. For each block, we store:
    - min and max of each coordinate (after rotation + quantization)

    At query time, for the top-m query coordinates, we compute an upper
    bound on the dot product for each block:
      upper_bound = sum_j max(q_j * block_max_j, q_j * block_min_j)
    Blocks whose upper bound is below the k-th best lower bound are pruned.
    """

    def __init__(self, K_rot: np.ndarray, block_size: int = 256):
        """
        K_rot: (n, d) rotated (and optionally quantized) key vectors
        block_size: number of keys per block
        """
        self.n, self.d = K_rot.shape
        self.block_size = block_size
        self.n_blocks = (self.n + block_size - 1) // block_size
        self.K_rot = K_rot

        # Build summary: per-block min and max for each coordinate
        self.block_min = np.zeros((self.n_blocks, self.d), dtype=np.float32)
        self.block_max = np.zeros((self.n_blocks, self.d), dtype=np.float32)

        self._build_time = 0.0
        self._build()

    def _build(self):
        t0 = time.perf_counter()
        for bi in range(self.n_blocks):
            start = bi * self.block_size
            end = min(start + self.block_size, self.n)
            block = self.K_rot[start:end]
            self.block_min[bi] = block.min(axis=0)
            self.block_max[bi] = block.max(axis=0)
        self._build_time = time.perf_counter() - t0

    def summary_size_bytes(self) -> int:
        """Size of the summary table in bytes."""
        # 2 * n_blocks * d * 4 bytes (float32 min + max)
        return 2 * self.n_blocks * self.d * 4

    def prune(self, q_rot: np.ndarray, k: int, m: int = None):
        """
        Prune blocks using upper-bound scoring.

        q_rot: (d,) rotated query vector
        k: number of top results desired
        m: number of top coordinates for upper bound (None = all)

        Returns:
            surviving_blocks: list of block indices that survive pruning
            stats: dict with pruning statistics
        """
        if m is None:
            m = self.d

        # Top-m coordinates by query magnitude
        coord_order = np.argsort(np.abs(q_rot))[::-1]
        top_m = coord_order[:m]

        # Upper bound for each block using top-m coords
        q_m = q_rot[top_m]  # (m,)
        bmin_m = self.block_min[:, top_m]  # (n_blocks, m)
        bmax_m = self.block_max[:, top_m]  # (n_blocks, m)

        # For each coordinate j and block b:
        # contribution upper bound = max(q_j * min_j, q_j * max_j)
        contrib_a = bmin_m * q_m[None, :]  # (n_blocks, m)
        contrib_b = bmax_m * q_m[None, :]  # (n_blocks, m)
        upper_contribs = np.maximum(contrib_a, contrib_b)
        upper_bounds = upper_contribs.sum(axis=1)  # (n_blocks,)

        # Also add remaining coordinates' maximum possible contribution
        if m < self.d:
            remaining = coord_order[m:]
            q_rem = np.abs(q_rot[remaining])
            # max possible contribution from each remaining coord: |q_j| * 1.0
            remaining_budget = np.sum(q_rem)
            upper_bounds += remaining_budget

        # To get threshold: we need an estimate of the k-th best score.
        # Quick lower bound: score a sample of blocks to get a rough threshold.
        # Score the first few blocks fully to establish a baseline.
        n_sample = min(10, self.n_blocks)
        sample_scores = []
        for bi in range(n_sample):
            start = bi * self.block_size
            end = min(start + self.block_size, self.n)
            scores = self.K_rot[start:end] @ q_rot
            sample_scores.extend(scores.tolist())

        if len(sample_scores) >= k:
            sample_scores.sort(reverse=True)
            threshold = sample_scores[k - 1]
        else:
            threshold = -np.inf

        # Prune blocks whose upper bound is below threshold
        surviving = np.where(upper_bounds >= threshold)[0]

        # Refine: actually score surviving blocks and find top-k
        all_scores = []
        all_indices = []
        for bi in surviving:
            start = bi * self.block_size
            end = min(start + self.block_size, self.n)
            scores = self.K_rot[start:end] @ q_rot
            all_scores.append(scores)
            all_indices.append(np.arange(start, end))

        if all_scores:
            all_scores = np.concatenate(all_scores)
            all_indices = np.concatenate(all_indices)
            # Update threshold with actual scores
            if len(all_scores) >= k:
                kth_score = np.partition(all_scores, -k)[-k]
                # Re-prune with better threshold
                surviving_refined = np.where(upper_bounds >= kth_score)[0]
            else:
                surviving_refined = surviving
        else:
            surviving_refined = surviving

        stats = {
            'n_blocks': self.n_blocks,
            'surviving_initial': len(surviving),
            'surviving_refined': len(surviving_refined),
            'pruned_fraction': 1.0 - len(surviving_refined) / self.n_blocks,
        }

        return surviving_refined.tolist(), stats

    def search(self, q_rot: np.ndarray, k: int, m: int = None):
        """
        Full search: prune blocks, then score surviving keys, return top-k.

        Returns: (top_k_indices, top_k_scores, stats)
        """
        surviving_blocks, prune_stats = self.prune(q_rot, k, m)

        # Score all keys in surviving blocks
        all_scores = []
        all_indices = []
        for bi in surviving_blocks:
            start = bi * self.block_size
            end = min(start + self.block_size, self.n)
            scores = self.K_rot[start:end] @ q_rot
            all_scores.append(scores)
            all_indices.append(np.arange(start, end))

        if all_scores:
            all_scores = np.concatenate(all_scores)
            all_indices = np.concatenate(all_indices)
            top_k_local = np.argsort(all_scores)[-k:]
            top_k_indices = all_indices[top_k_local]
            top_k_scores = all_scores[top_k_local]
        else:
            top_k_indices = np.array([], dtype=int)
            top_k_scores = np.array([])

        keys_scored = sum(
            min((bi + 1) * self.block_size, self.n) - bi * self.block_size
            for bi in surviving_blocks
        )
        prune_stats['keys_scored'] = keys_scored

        return top_k_indices, top_k_scores, prune_stats


# ---------------------------------------------------------------------------
# Test 1: Summary table size
# ---------------------------------------------------------------------------
def test_summary_size(d: int, block_size: int):
    """Verify the summary table size formula."""
    print("\n[Test 1] Summary table size")

    ns = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    results = {}

    for n in ns:
        n_blocks = (n + block_size - 1) // block_size
        # 2 (min + max) * n_blocks * d * 4 bytes (float32)
        size_bytes = 2 * n_blocks * d * 4
        size_kb = size_bytes / 1024
        size_mb = size_bytes / (1024 * 1024)
        print(f"  n={n:>10,}: {n_blocks:>5} blocks, "
              f"summary = {size_kb:.1f}KB ({size_mb:.3f}MB)")
        results[n] = {
            'n_blocks': n_blocks,
            'size_bytes': size_bytes,
            'size_kb': float(size_kb),
        }

    # Plot
    fig, ax = plt.subplots()
    ns_arr = np.array(ns, dtype=float)
    sizes_kb = [results[n]['size_kb'] for n in ns]
    ax.plot(ns_arr, sizes_kb, 'o-', lw=2)
    ax.set_xscale('log')
    ax.set_xlabel('Number of keys $n$')
    ax.set_ylabel('Summary table size (KB)')
    ax.set_title(f'Block Summary Table Size (d={d}, block_size={block_size})')
    ax.grid(True, alpha=0.3)
    # Annotate the 1M point
    ax.annotate(f"{sizes_kb[-1]:.0f}KB", xy=(ns[-1], sizes_kb[-1]),
                fontsize=10, ha='right', va='bottom',
                xytext=(-10, 5), textcoords='offset points')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp05_summary_size.png')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'exp05_summary_size.png'}")

    return {str(n): v for n, v in results.items()}


# ---------------------------------------------------------------------------
# Test 2: Pruning effectiveness
# ---------------------------------------------------------------------------
def test_pruning_effectiveness(n: int, d: int, b: int, block_size: int,
                                k: int, n_repeats: int, rng: np.random.RandomState):
    """How many blocks survive for random vs structured keys?"""
    print(f"\n[Test 2] Pruning effectiveness (n={n:,})")

    ms_to_test = [8, 16, 32, 64, 128]  # number of top coords for pruning
    ms_to_test = [m for m in ms_to_test if m <= d]
    results = {}

    # Random keys
    print("  Random keys:")
    random_pruned = {m: [] for m in ms_to_test}
    for rep in range(n_repeats):
        R = _random_orthogonal(d, rng)
        K = _random_unit_vectors(n, d, rng)
        K_rot = K @ R.T
        K_quant = _uniform_scalar_quantize(K_rot, b)

        bst = _ReferenceBlockSummaryTable(K_quant, block_size)

        q = _random_unit_vectors(1, d, rng).ravel()
        q_rot = R @ q

        for m in ms_to_test:
            _, stats = bst.prune(q_rot, k, m)
            random_pruned[m].append(stats['pruned_fraction'])

    for m in ms_to_test:
        mean_pf = np.mean(random_pruned[m])
        print(f"    m={m:3d}: {mean_pf:.1%} pruned")
    results['random'] = {str(m): float(np.mean(random_pruned[m])) for m in ms_to_test}

    # Structured keys
    for kappa in [50, 200]:
        print(f"  Clustered keys (kappa={kappa}):")
        clustered_pruned = {m: [] for m in ms_to_test}
        for rep in range(n_repeats):
            R = _random_orthogonal(d, rng)
            K = _generate_clustered_keys(n, d, 50, kappa, rng)
            K_rot = K @ R.T
            K_quant = _uniform_scalar_quantize(K_rot, b)

            bst = _ReferenceBlockSummaryTable(K_quant, block_size)

            # Query near a key (focused attention)
            qi = rng.randint(n)
            q = K[qi] + rng.randn(d) * 0.01
            q = q / np.linalg.norm(q)
            q_rot = R @ q

            for m in ms_to_test:
                _, stats = bst.prune(q_rot, k, m)
                clustered_pruned[m].append(stats['pruned_fraction'])

        for m in ms_to_test:
            mean_pf = np.mean(clustered_pruned[m])
            print(f"    m={m:3d}: {mean_pf:.1%} pruned")
        results[f'kappa_{kappa}'] = {
            str(m): float(np.mean(clustered_pruned[m])) for m in ms_to_test
        }

    # Plot
    fig, ax = plt.subplots()
    ms_arr = np.array(ms_to_test, dtype=float)
    for label, color in [('random', 'C0'), ('kappa_50', 'C1'), ('kappa_200', 'C2')]:
        vals = [results[label][str(m)] for m in ms_to_test]
        display = label.replace('_', '=').replace('kappa', '$\\kappa$')
        ax.plot(ms_arr, vals, 'o-', lw=2, color=color, label=display)

    ax.set_xlabel('Pruning coordinates $m$')
    ax.set_ylabel('Fraction of blocks pruned')
    ax.set_title(f'Block Pruning Effectiveness (n={n:,}, d={d})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp05_pruning_effectiveness.png')
    plt.close(fig)
    print(f"\n  Saved: {FIGURES_DIR / 'exp05_pruning_effectiveness.png'}")

    return results


# ---------------------------------------------------------------------------
# Test 3: End-to-end speedup
# ---------------------------------------------------------------------------
def test_end_to_end(n: int, d: int, b: int, block_size: int, k: int,
                    n_repeats: int, rng: np.random.RandomState):
    """Full pipeline timing: build summary -> prune -> score survivors."""
    print(f"\n[Test 3] End-to-end speedup (n={n:,})")

    ns = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    results_by_n = {}

    for n_test in ns:
        print(f"  n = {n_test:,}")
        bf_times = []
        block_times = []
        recalls = []
        keys_scored_fracs = []

        for rep in range(n_repeats):
            K = _random_unit_vectors(n_test, d, rng)
            q = _random_unit_vectors(1, d, rng).ravel()

            if _LIB_AVAILABLE:
                # Use real TurboQuant + BlockPruning library
                tq = TurboQuant(d=d, bits=b, seed=rng.randint(0, 2**31))
                codes, norms = tq.quantize_batch(K)

                # Brute force (quantized)
                t0 = time.perf_counter()
                bf_scores = tq.batch_dot_products(codes, norms, q)
                bf_topk = set(np.argsort(bf_scores)[-k:])
                t_bf = time.perf_counter() - t0

                # Block pruning via library
                t0 = time.perf_counter()
                bp = _LibBlockPruning(tq, codes, norms, block_size=block_size)
                bp_idx, _, bp_stats = bp.query(q, k=k)
                t_block = time.perf_counter() - t0

                recall = len(set(bp_idx) & bf_topk) / k
                keys_frac = bp_stats['keys_scored'] / n_test
            else:
                R = _random_orthogonal(d, rng)
                K_rot = K @ R.T
                K_quant = _uniform_scalar_quantize(K_rot, b)
                q_rot = R @ q

                # Brute force
                t0 = time.perf_counter()
                bf_scores = K_quant @ q_rot
                bf_topk = set(np.argsort(bf_scores)[-k:])
                t_bf = time.perf_counter() - t0

                # Block pruning (reference)
                t0 = time.perf_counter()
                bst = _ReferenceBlockSummaryTable(K_quant, block_size)
                topk_idx, _, stats = bst.search(q_rot, k, m=32)
                t_block = time.perf_counter() - t0

                recall = len(set(topk_idx) & bf_topk) / k
                keys_frac = stats['keys_scored'] / n_test

            bf_times.append(t_bf)
            block_times.append(t_block)
            recalls.append(recall)
            keys_scored_fracs.append(keys_frac)

        results_by_n[n_test] = {
            'bf_time_ms': float(np.mean(bf_times) * 1000),
            'block_time_ms': float(np.mean(block_times) * 1000),
            'speedup': float(np.mean(bf_times) / np.mean(block_times)),
            'recall': float(np.mean(recalls)),
            'keys_scored_frac': float(np.mean(keys_scored_fracs)),
        }
        r = results_by_n[n_test]
        print(f"    BF: {r['bf_time_ms']:.1f}ms, Block: {r['block_time_ms']:.1f}ms, "
              f"speedup: {r['speedup']:.2f}x, recall: {r['recall']:.4f}, "
              f"keys scored: {r['keys_scored_frac']:.1%}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ns_arr = np.array(ns, dtype=float)
    bf_t = [results_by_n[n_test]['bf_time_ms'] for n_test in ns]
    bl_t = [results_by_n[n_test]['block_time_ms'] for n_test in ns]
    speedups = [results_by_n[n_test]['speedup'] for n_test in ns]

    ax1.plot(ns_arr, bf_t, 'o-', lw=2, label='Brute force')
    ax1.plot(ns_arr, bl_t, 's-', lw=2, label='Block pruning')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of keys $n$')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('End-to-End Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(range(len(ns)), speedups, tick_label=[f'{n_test//1000}K' for n_test in ns],
            color='C2')
    ax2.set_xlabel('Number of keys $n$')
    ax2.set_ylabel('Speedup factor')
    ax2.set_title('Block Pruning Speedup')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp05_end_to_end.png')
    plt.close(fig)
    print(f"\n  Saved: {FIGURES_DIR / 'exp05_end_to_end.png'}")

    return {str(n_test): v for n_test, v in results_by_n.items()}


# ---------------------------------------------------------------------------
# Test 4: Memory reads
# ---------------------------------------------------------------------------
def test_memory_reads(n: int, d: int, b: int, block_size: int, k: int,
                      n_repeats: int, rng: np.random.RandomState):
    """Count bytes read from HBM in each phase."""
    print(f"\n[Test 4] Memory reads analysis (n={n:,})")

    n_blocks = (n + block_size - 1) // block_size
    bytes_per_element = 4  # float32

    # Phase 1: Read summary table (min + max)
    summary_read_bytes = 2 * n_blocks * d * bytes_per_element
    # Phase 2: Read surviving blocks
    # Phase 3: Score surviving keys (already in cache from phase 2)

    # Brute force: read all keys
    bf_bytes = n * d * bytes_per_element

    phase_bytes = {
        'summary_table': [],
        'surviving_blocks': [],
        'total_block_pruning': [],
        'brute_force': bf_bytes,
    }

    for rep in range(n_repeats):
        R = _random_orthogonal(d, rng)
        K = _random_unit_vectors(n, d, rng)
        K_rot = K @ R.T
        K_quant = _uniform_scalar_quantize(K_rot, b)

        bst = _ReferenceBlockSummaryTable(K_quant, block_size)

        q = _random_unit_vectors(1, d, rng).ravel()
        q_rot = R @ q

        _, _, stats = bst.search(q_rot, k, m=32)
        n_surviving = stats['surviving_refined'] if 'surviving_refined' in stats else n_blocks
        surviving_bytes = n_surviving * block_size * d * bytes_per_element

        phase_bytes['summary_table'].append(summary_read_bytes)
        phase_bytes['surviving_blocks'].append(surviving_bytes)
        phase_bytes['total_block_pruning'].append(summary_read_bytes + surviving_bytes)

    mean_summary = np.mean(phase_bytes['summary_table'])
    mean_surviving = np.mean(phase_bytes['surviving_blocks'])
    mean_total = np.mean(phase_bytes['total_block_pruning'])

    print(f"  Brute force reads:    {bf_bytes / 1e6:.1f}MB")
    print(f"  Summary table reads:  {mean_summary / 1e6:.1f}MB")
    print(f"  Surviving block reads: {mean_surviving / 1e6:.1f}MB")
    print(f"  Total block pruning:  {mean_total / 1e6:.1f}MB")
    print(f"  Memory bandwidth savings: {(1 - mean_total / bf_bytes) * 100:.1f}%")

    # Plot
    fig, ax = plt.subplots()
    methods = ['Brute\nforce', 'Summary\ntable', 'Surviving\nblocks', 'Block pruning\ntotal']
    values_mb = [bf_bytes / 1e6, mean_summary / 1e6, mean_surviving / 1e6, mean_total / 1e6]
    colors = ['C0', 'C1', 'C2', 'C3']

    bars = ax.bar(methods, values_mb, color=colors, alpha=0.8)
    ax.set_ylabel('Memory reads (MB)')
    ax.set_title(f'HBM Memory Reads (n={n:,}, d={d})')

    # Annotate
    for bar, val in zip(bars, values_mb):
        ax.annotate(f'{val:.1f}MB', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10)

    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp05_memory_reads.png')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'exp05_memory_reads.png'}")

    return {
        'brute_force_bytes': float(bf_bytes),
        'summary_table_bytes': float(mean_summary),
        'surviving_blocks_bytes': float(mean_surviving),
        'total_block_pruning_bytes': float(mean_total),
        'bandwidth_savings_pct': float((1 - mean_total / bf_bytes) * 100),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Experiment 05: Block Summary Table Pruning")
    print("=" * 70)
    if not _LIB_AVAILABLE:
        print("WARNING: Using reference implementations (library not yet available)")
    print()

    np.random.seed(42)
    rng = np.random.RandomState(42)

    D = 128
    B = 2
    K = 100
    BLOCK_SIZE = 256
    N_REPEATS = 5

    results = {}
    t0 = time.time()

    results['test1_summary_size'] = test_summary_size(D, BLOCK_SIZE)
    results['test2_pruning_effectiveness'] = test_pruning_effectiveness(
        100_000, D, B, BLOCK_SIZE, K, N_REPEATS, rng)
    results['test3_end_to_end'] = test_end_to_end(
        100_000, D, B, BLOCK_SIZE, K, N_REPEATS, rng)
    results['test4_memory_reads'] = test_memory_reads(
        100_000, D, B, BLOCK_SIZE, K, N_REPEATS, rng)

    elapsed = time.time() - t0
    results['total_time_seconds'] = float(elapsed)
    results['parameters'] = {
        'd': D, 'b': B, 'k': K, 'block_size': BLOCK_SIZE,
        'n_repeats': N_REPEATS, 'library_available': _LIB_AVAILABLE,
    }

    summary_path = RESULTS_DIR / 'exp05_results.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
