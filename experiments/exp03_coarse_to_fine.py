#!/usr/bin/env python3
"""
Experiment 03: Coarse-to-Fine Algorithm Benchmark
===================================================
Benchmarks the multi-round coarse-to-fine scoring algorithm that uses
partial dot products (top-m coordinates) to prune candidates before
computing full scores.

Tests:
  1. Recall vs speedup for 1-round, 2-round, and 3-round strategies
  2. Structured vs random keys (von Mises-Fisher clusters)
  3. Scaling with n (10K to 1M)
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from collections import defaultdict

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
    from quantdex.sub_linear_attention import CoarseToFine
    from quantdex.metrics import recall_at_k
    _LIB_AVAILABLE = True
except ImportError as exc:
    warnings.warn(
        f"Could not import quantdex library: {exc}\n"
        "Falling back to self-contained reference implementations.\n"
        "To use the real library, ensure ~/quantdex/src/ contains:\n"
        "  - turbo_quant.py          (TurboQuant class)\n"
        "  - sub_linear_attention.py (CoarseToFine class)\n"
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


def _sample_vmf(mu: np.ndarray, kappa: float, n: int,
                rng: np.random.RandomState) -> np.ndarray:
    """
    Sample n vectors from von Mises-Fisher distribution with mean direction mu
    and concentration kappa, in d dimensions.

    Uses the rejection sampling method of Wood (1994).
    """
    d = len(mu)
    # Normalize mu
    mu = mu / np.linalg.norm(mu)

    # For large d, use approximate sampling:
    # Sample w ~ some distribution on [-1, 1], then x = w*mu + sqrt(1-w^2)*v
    # where v is uniform on the (d-1)-sphere orthogonal to mu.

    # Build orthonormal basis with mu as first vector
    # We only need to generate the tangent component
    samples = []
    b_param = (-2.0 * kappa + np.sqrt(4.0 * kappa**2 + (d - 1)**2)) / (d - 1)
    x0 = (1.0 - b_param) / (1.0 + b_param)
    c = kappa * x0 + (d - 1) * np.log(1.0 - x0**2)

    n_accepted = 0
    while n_accepted < n:
        batch = min(n * 3, n - n_accepted + 1000)  # oversample
        z = rng.beta((d - 1) / 2.0, (d - 1) / 2.0, size=batch)
        w = (1.0 - (1.0 + b_param) * z) / (1.0 - (1.0 - b_param) * z)
        u = rng.uniform(size=batch)
        accept = kappa * w + (d - 1) * np.log(1.0 - x0 * w) - c >= np.log(u)
        w_accepted = w[accept][:n - n_accepted]

        if len(w_accepted) == 0:
            # Fallback: just use the mean direction with noise
            remaining = n - n_accepted
            noise = rng.randn(remaining, d) * (1.0 / np.sqrt(kappa))
            vecs = mu[None, :] + noise
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            samples.append(vecs)
            n_accepted += remaining
            continue

        for w_val in w_accepted:
            # Sample tangent direction
            v = rng.randn(d)
            v = v - np.dot(v, mu) * mu
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-10:
                v = np.zeros(d)
                v[0 if abs(mu[0]) < 0.9 else 1] = 1.0
                v = v - np.dot(v, mu) * mu
                v_norm = np.linalg.norm(v)
            v = v / v_norm
            sample = w_val * mu + np.sqrt(max(0, 1.0 - w_val**2)) * v
            samples.append(sample[None, :])
            n_accepted += 1
            if n_accepted >= n:
                break

    return np.concatenate(samples, axis=0)[:n]


def _generate_clustered_keys(n: int, d: int, n_clusters: int, kappa: float,
                              rng: np.random.RandomState) -> np.ndarray:
    """Generate n vectors from n_clusters von Mises-Fisher distributions."""
    # Random cluster centers
    centers = _random_unit_vectors(n_clusters, d, rng)
    per_cluster = n // n_clusters
    remainder = n - per_cluster * n_clusters

    all_vecs = []
    for i in range(n_clusters):
        n_i = per_cluster + (1 if i < remainder else 0)
        vecs = _sample_vmf(centers[i], kappa, n_i, rng)
        all_vecs.append(vecs)

    result = np.concatenate(all_vecs, axis=0)
    # Normalize
    result = result / np.linalg.norm(result, axis=1, keepdims=True)
    # Shuffle
    perm = rng.permutation(n)
    return result[perm]


def _coarse_to_fine(q_rot: np.ndarray, K_rot: np.ndarray, rounds: list,
                    k: int, candidate_multiplier: float = 5.0):
    """
    Multi-round coarse-to-fine search.

    rounds: list of m values (number of coords to use in each round)
            e.g., [8, 24, 128] for 3-round with m=8 -> m=24 -> m=128
    k: number of results to return
    candidate_multiplier: how many candidates to keep between rounds

    Returns:
        top_k_indices: indices of top-k keys
        ops_count: total coordinate-level multiply-add operations
    """
    n, d = K_rot.shape
    top_idx = np.argsort(np.abs(q_rot))[::-1]  # sorted by |q_i|

    candidates = np.arange(n)  # start with all keys
    total_ops = 0

    for round_i, m in enumerate(rounds):
        coord_idx = top_idx[:m]

        # Score candidates using m coordinates
        if round_i == 0:
            # First round: score all n keys
            scores = K_rot[candidates][:, coord_idx] @ q_rot[coord_idx]
            total_ops += len(candidates) * m
        else:
            # Subsequent rounds: only score surviving candidates
            scores = K_rot[candidates][:, coord_idx] @ q_rot[coord_idx]
            total_ops += len(candidates) * m

        # Keep top candidates for next round (or return if last round)
        if round_i < len(rounds) - 1:
            # Keep top C*k candidates
            n_keep = min(int(candidate_multiplier * k), len(candidates))
            top_cand_idx = np.argsort(scores)[-n_keep:]
            candidates = candidates[top_cand_idx]
        else:
            # Last round: return top-k
            top_cand_idx = np.argsort(scores)[-k:]
            candidates = candidates[top_cand_idx]

    return candidates, total_ops


def _lib_coarse_to_fine_search(K: np.ndarray, q: np.ndarray, k: int,
                                rounds_config: list, bits: int = 2,
                                seed: int = 42):
    """
    Run coarse-to-fine using the real TurboQuant + CoarseToFine library.

    Returns: (found_indices, recall_vs_brute, stats)
    """
    n, d = K.shape
    tq = TurboQuant(d=d, bits=bits, seed=seed)
    codes, norms = tq.quantize_batch(K)

    # Build round config as (m, keep) tuples for CoarseToFine
    ctf_rounds = []
    for i, m in enumerate(rounds_config):
        if i < len(rounds_config) - 1:
            ctf_rounds.append((m, min(5 * k, n)))
        else:
            ctf_rounds.append((m, k))

    ctf = CoarseToFine(tq, codes, norms)
    top_k_indices, top_k_scores, stats = ctf.query(q, k=k, rounds=ctf_rounds)

    # Brute force for recall
    brute_scores = tq.batch_dot_products(codes, norms, q)
    true_topk = set(np.argsort(brute_scores)[-k:])
    recall = len(set(top_k_indices) & true_topk) / k

    return top_k_indices, recall, stats


# ---------------------------------------------------------------------------
# Test 1: Recall vs speedup
# ---------------------------------------------------------------------------
def test_recall_vs_speedup(d: int, b: int, k: int, n_repeats: int,
                            rng: np.random.RandomState):
    """Test recall vs speedup for different n and round configurations."""
    print("\n[Test 1] Recall vs speedup")

    ns = [100_000, 500_000, 1_000_000]
    # Round configurations: (label, [m_values])
    configs_1round = [
        ('1R m=8', [8]),
        ('1R m=16', [16]),
        ('1R m=24', [24]),
        ('1R m=32', [32]),
        ('1R m=48', [48]),
        ('1R m=64', [64]),
    ]
    configs_2round = [
        ('2R 8->128', [8, 128]),
        ('2R 16->128', [16, 128]),
        ('2R 24->128', [24, 128]),
    ]
    configs_3round = [
        ('3R 8->24->128', [8, 24, 128]),
        ('3R 8->32->128', [8, 32, 128]),
        ('3R 16->48->128', [16, 48, 128]),
    ]
    all_configs = configs_1round + configs_2round + configs_3round

    results_by_n = {}

    for n in ns:
        print(f"\n  n = {n:,}")
        brute_force_ops = n * d  # full dot product for all keys

        config_results = {}
        for label, rounds in all_configs:
            recalls = []
            speedups = []
            for rep in range(n_repeats):
                R = _random_orthogonal(d, rng)
                K = _random_unit_vectors(n, d, rng)
                q = _random_unit_vectors(1, d, rng).ravel()
                K_rot = K @ R.T
                q_rot = R @ q

                # Brute force top-k
                full_scores = K_rot @ q_rot
                true_topk = set(np.argsort(full_scores)[-k:])

                # Coarse-to-fine
                found_idx, ops = _coarse_to_fine(q_rot, K_rot, rounds, k,
                                                  candidate_multiplier=5.0)
                recall = len(set(found_idx) & true_topk) / k
                speedup = brute_force_ops / max(ops, 1)

                recalls.append(recall)
                speedups.append(speedup)

            mean_recall = np.mean(recalls)
            mean_speedup = np.mean(speedups)
            config_results[label] = {
                'recall': float(mean_recall),
                'speedup': float(mean_speedup),
                'rounds': rounds,
            }
            print(f"    {label:20s}: recall={mean_recall:.4f}, speedup={mean_speedup:.2f}x")

        results_by_n[n] = config_results

    # Plot for largest n
    n_plot = ns[-1]
    res = results_by_n[n_plot]

    fig, ax = plt.subplots()
    markers = {'1R': 'o', '2R': 's', '3R': 'D'}
    colors = {'1R': 'C0', '2R': 'C1', '3R': 'C3'}

    for label, data in res.items():
        prefix = label[:2]
        ax.scatter(data['speedup'], data['recall'],
                   marker=markers[prefix], color=colors[prefix], s=80, zorder=3)
        ax.annotate(label.split(' ')[1], (data['speedup'], data['recall']),
                    fontsize=7, ha='left', va='bottom',
                    xytext=(4, 2), textcoords='offset points')

    # Connect same-type configs
    for prefix in ['1R', '2R', '3R']:
        pts = [(res[l]['speedup'], res[l]['recall']) for l in res if l.startswith(prefix)]
        if len(pts) > 1:
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], '--',
                    color=colors[prefix], alpha=0.5, lw=1)

    ax.set_xlabel('Speedup factor')
    ax.set_ylabel(f'Recall@{k}')
    ax.set_title(f'Coarse-to-Fine: Recall vs Speedup (n={n_plot:,}, d={d})')
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='C0', lw=0, label='1-round', markersize=8),
        Line2D([0], [0], marker='s', color='C1', lw=0, label='2-round', markersize=8),
        Line2D([0], [0], marker='D', color='C3', lw=0, label='3-round', markersize=8),
    ]
    ax.legend(handles=legend_elements)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp03_recall_vs_speedup.png')
    plt.close(fig)
    print(f"\n  Saved: {FIGURES_DIR / 'exp03_recall_vs_speedup.png'}")

    return results_by_n


# ---------------------------------------------------------------------------
# Test 2: Structured vs random keys
# ---------------------------------------------------------------------------
def test_structured_vs_random(n: int, d: int, b: int, k: int, n_repeats: int,
                               rng: np.random.RandomState):
    """Compare performance on random vs clustered keys."""
    print("\n[Test 2] Structured vs random keys")

    kappas = [10, 50, 200]
    n_clusters = 50
    rounds_config = [8, 24, 128]  # 3-round
    brute_force_ops = n * d

    results = {}

    # Random keys
    print("  Random keys:")
    random_recalls = []
    random_speedups = []
    for rep in range(n_repeats):
        R = _random_orthogonal(d, rng)
        K = _random_unit_vectors(n, d, rng)
        q = _random_unit_vectors(1, d, rng).ravel()
        K_rot = K @ R.T
        q_rot = R @ q

        full_scores = K_rot @ q_rot
        true_topk = set(np.argsort(full_scores)[-k:])

        found_idx, ops = _coarse_to_fine(q_rot, K_rot, rounds_config, k,
                                          candidate_multiplier=5.0)
        recall = len(set(found_idx) & true_topk) / k
        speedup = brute_force_ops / max(ops, 1)
        random_recalls.append(recall)
        random_speedups.append(speedup)

    results['random'] = {
        'recall': float(np.mean(random_recalls)),
        'speedup': float(np.mean(random_speedups)),
    }
    print(f"    recall={results['random']['recall']:.4f}, "
          f"speedup={results['random']['speedup']:.2f}x")

    # Structured keys
    for kappa in kappas:
        print(f"  Clustered keys (kappa={kappa}):")
        recalls = []
        speedups = []
        for rep in range(n_repeats):
            R = _random_orthogonal(d, rng)
            K = _generate_clustered_keys(n, d, n_clusters, kappa, rng)
            # Query near one of the clusters
            q = K[rng.randint(n)] + rng.randn(d) * 0.01
            q = q / np.linalg.norm(q)

            K_rot = K @ R.T
            q_rot = R @ q

            full_scores = K_rot @ q_rot
            true_topk = set(np.argsort(full_scores)[-k:])

            found_idx, ops = _coarse_to_fine(q_rot, K_rot, rounds_config, k,
                                              candidate_multiplier=5.0)
            recall = len(set(found_idx) & true_topk) / k
            speedup = brute_force_ops / max(ops, 1)
            recalls.append(recall)
            speedups.append(speedup)

        results[f'kappa_{kappa}'] = {
            'recall': float(np.mean(recalls)),
            'speedup': float(np.mean(speedups)),
        }
        print(f"    recall={results[f'kappa_{kappa}']['recall']:.4f}, "
              f"speedup={results[f'kappa_{kappa}']['speedup']:.2f}x")

    # Plot
    fig, ax = plt.subplots()
    labels = ['Random'] + [f'$\\kappa$={k}' for k in kappas]
    recalls = [results['random']['recall']] + [results[f'kappa_{k}']['recall'] for k in kappas]
    speedups = [results['random']['speedup']] + [results[f'kappa_{k}']['speedup'] for k in kappas]

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, recalls, width, label='Recall@100', color='C0')
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, speedups, width, label='Speedup', color='C1', alpha=0.7)

    ax.set_xlabel('Key distribution')
    ax.set_ylabel('Recall@100', color='C0')
    ax2.set_ylabel('Speedup factor', color='C1')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_title(f'Structured vs Random Keys (n={n:,}, d={d}, 3-round)')

    # Merge legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp03_structured_vs_random.png')
    plt.close(fig)
    print(f"\n  Saved: {FIGURES_DIR / 'exp03_structured_vs_random.png'}")

    return results


# ---------------------------------------------------------------------------
# Test 3: Scaling with n
# ---------------------------------------------------------------------------
def test_scaling_with_n(d: int, b: int, k: int, n_repeats: int,
                        rng: np.random.RandomState):
    """Measure brute force and coarse-to-fine time as n grows."""
    print("\n[Test 3] Scaling with n")

    ns = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    rounds_config = [8, 24, 128]

    scaling_results = {}

    for n in ns:
        print(f"  n = {n:,}")
        bf_times = []
        c2f_times = []
        recalls = []
        speedup_factors = []

        for rep in range(n_repeats):
            K = _random_unit_vectors(n, d, rng)
            q = _random_unit_vectors(1, d, rng).ravel()

            if _LIB_AVAILABLE:
                # Use real TurboQuant + CoarseToFine library
                tq = TurboQuant(d=d, bits=b, seed=rng.randint(0, 2**31))
                codes, norms = tq.quantize_batch(K)

                # Brute force (quantized)
                t0 = time.perf_counter()
                brute_scores = tq.batch_dot_products(codes, norms, q)
                true_topk_idx = np.argsort(brute_scores)[-k:]
                t_bf = time.perf_counter() - t0
                true_topk = set(true_topk_idx)

                # Coarse-to-fine via library
                ctf_rounds = [
                    (8, min(100 * k, n)),
                    (24, min(20 * k, n)),
                    (d, k),
                ]
                ctf = CoarseToFine(tq, codes, norms)
                t0 = time.perf_counter()
                found_idx, _, stats = ctf.query(q, k=k, rounds=ctf_rounds)
                t_c2f = time.perf_counter() - t0

                recall = len(set(found_idx) & true_topk) / k
                bf_ops = n * d
                ops = stats['coords_read']
                speedup = bf_ops / max(ops, 1)
            else:
                R = _random_orthogonal(d, rng)
                K_rot = K @ R.T
                q_rot = R @ q

                # Brute force
                t0 = time.perf_counter()
                full_scores = K_rot @ q_rot
                true_topk_idx = np.argsort(full_scores)[-k:]
                t_bf = time.perf_counter() - t0
                true_topk = set(true_topk_idx)

                # Coarse-to-fine (reference)
                t0 = time.perf_counter()
                found_idx, ops = _coarse_to_fine(q_rot, K_rot, rounds_config, k,
                                                  candidate_multiplier=5.0)
                t_c2f = time.perf_counter() - t0

                recall = len(set(found_idx) & true_topk) / k
                bf_ops = n * d
                speedup = bf_ops / max(ops, 1)

            bf_times.append(t_bf)
            c2f_times.append(t_c2f)
            recalls.append(recall)
            speedup_factors.append(speedup)

        scaling_results[n] = {
            'brute_force_time_ms': float(np.mean(bf_times) * 1000),
            'coarse_to_fine_time_ms': float(np.mean(c2f_times) * 1000),
            'wall_speedup': float(np.mean(bf_times) / np.mean(c2f_times)),
            'ops_speedup': float(np.mean(speedup_factors)),
            'recall': float(np.mean(recalls)),
        }
        r = scaling_results[n]
        print(f"    BF: {r['brute_force_time_ms']:.1f}ms, "
              f"C2F: {r['coarse_to_fine_time_ms']:.1f}ms, "
              f"wall speedup: {r['wall_speedup']:.2f}x, "
              f"ops speedup: {r['ops_speedup']:.2f}x, "
              f"recall: {r['recall']:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ns_arr = np.array(ns, dtype=float)
    bf_t = [scaling_results[n]['brute_force_time_ms'] for n in ns]
    c2f_t = [scaling_results[n]['coarse_to_fine_time_ms'] for n in ns]
    speedups = [scaling_results[n]['wall_speedup'] for n in ns]
    recalls = [scaling_results[n]['recall'] for n in ns]

    ax1.plot(ns_arr, bf_t, 'o-', lw=2, label='Brute force')
    ax1.plot(ns_arr, c2f_t, 's-', lw=2, label='Coarse-to-fine (3-round)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of keys $n$')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Wall-Clock Time vs n')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ns_arr, speedups, 'D-', lw=2, color='C2', label='Wall speedup')
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of keys $n$')
    ax2.set_ylabel('Speedup factor', color='C2')
    ax2.set_title('Speedup and Recall vs n')
    ax2.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.plot(ns_arr, recalls, 'o--', lw=2, color='C3', label='Recall@100')
    ax2b.set_ylabel('Recall@100', color='C3')
    ax2b.set_ylim(0, 1.05)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp03_scaling_with_n.png')
    plt.close(fig)
    print(f"\n  Saved: {FIGURES_DIR / 'exp03_scaling_with_n.png'}")

    return {str(n): v for n, v in scaling_results.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Experiment 03: Coarse-to-Fine Algorithm Benchmark")
    print("=" * 70)
    if not _LIB_AVAILABLE:
        print("WARNING: Using reference implementations (library not yet available)")
    print()

    np.random.seed(42)
    rng = np.random.RandomState(42)

    D = 128
    B = 2
    K = 100
    N_REPEATS = 5

    results = {}
    t0 = time.time()

    results['test1_recall_vs_speedup'] = test_recall_vs_speedup(D, B, K, N_REPEATS, rng)
    # Use smaller n for structured test (vMF sampling is slow)
    results['test2_structured_vs_random'] = test_structured_vs_random(
        n=100_000, d=D, b=B, k=K, n_repeats=N_REPEATS, rng=rng)
    results['test3_scaling_with_n'] = test_scaling_with_n(D, B, K, N_REPEATS, rng)

    elapsed = time.time() - t0
    results['total_time_seconds'] = float(elapsed)
    results['parameters'] = {
        'd': D, 'b': B, 'k': K, 'n_repeats': N_REPEATS,
        'library_available': _LIB_AVAILABLE,
    }

    # Serialize: convert int keys to str for JSON
    def _fixkeys(obj):
        if isinstance(obj, dict):
            return {str(k): _fixkeys(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_fixkeys(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    summary_path = RESULTS_DIR / 'exp03_results.json'
    with open(summary_path, 'w') as f:
        json.dump(_fixkeys(results), f, indent=2)
    print(f"\nResults saved to {summary_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
