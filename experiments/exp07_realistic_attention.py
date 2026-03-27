#!/usr/bin/env python3
"""
Experiment 07: Realistic LLM Attention Patterns
=================================================
The definitive validation experiment -- tests all three sub-linear attention
algorithms (CoarseToFine, CodeTrie, BlockPruning) against realistic attention
patterns based on empirical observations from H2O, SnapKV, and StreamingLLM.

Tests:
  1. Recall@100 vs keys read (the money plot)
  2. L2 attention output error vs k for each algorithm
  3. Performance scaling with sequence length n
  4. Visualization of the generated attention pattern

Generates publication-quality figures:
  - exp07_recall_vs_keys_read.png
  - exp07_attention_quality.png
  - exp07_scaling.png
  - exp07_attention_pattern.png
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

from src.turbo_quant import TurboQuant
from src.sub_linear_attention import CoarseToFine, CodeTrie, BlockPruning
from src.metrics import recall_at_k, softmax_mass_captured, attention_output_error
from src.attention_patterns import RealisticAttentionGenerator, compute_ground_truth

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
})

FIGURES_DIR = _PROJECT_ROOT / 'figures'
RESULTS_DIR = Path(__file__).resolve().parent / 'results'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: stable softmax
# ---------------------------------------------------------------------------
def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp_l = np.exp(logits)
    return exp_l / np.sum(exp_l)


# ---------------------------------------------------------------------------
# Helper: compute attention output error
# ---------------------------------------------------------------------------
def _attention_l2_error(true_scores, values, found_indices, k):
    """Compute L2 error of approximate attention output vs exact."""
    n = len(true_scores)

    # Exact attention output
    weights = _stable_softmax(true_scores)
    exact_output = weights @ values  # (d_v,)

    # Approximate: use found keys with their exact weights, renormalized
    if len(found_indices) == 0:
        return float(np.linalg.norm(exact_output))

    found_idx = found_indices[:k]
    approx_weights = np.zeros(n)
    approx_weights[found_idx] = weights[found_idx]
    mass = np.sum(approx_weights)
    if mass > 1e-30:
        approx_weights /= mass
    approx_output = approx_weights @ values

    return float(np.linalg.norm(exact_output - approx_output) /
                 max(np.linalg.norm(exact_output), 1e-30))


# ---------------------------------------------------------------------------
# Test 1: Recall vs Keys Read (the money plot)
# ---------------------------------------------------------------------------
def test_recall_vs_keys_read(n: int, d: int, bits: int, k: int, seed: int):
    """Run all three algorithms and measure recall@k vs bandwidth."""
    print(f"\n[Test 1] Recall vs Keys Read (n={n:,}, d={d}, k={k})")

    gen = RealisticAttentionGenerator(
        n=n, d=d, n_sinks=4, local_window=256,
        n_clusters=20, cluster_concentration=50.0,
        sparsity=0.05, seed=seed,
    )
    keys, values, query, metadata = gen.generate()

    # Scale keys to have varying norms (realistic)
    key_scales = 0.5 + np.abs(gen.rng.standard_normal(n))
    keys_scaled = keys * key_scales[:, np.newaxis]

    # Quantize
    tq = TurboQuant(d, bits=bits, seed=42)
    codes, norms = tq.quantize_batch(keys_scaled)

    # Brute force baseline
    true_scores = tq.batch_dot_products(codes, norms, query)
    true_topk = np.argsort(true_scores)[::-1][:k]

    # Ground truth metrics
    gt = compute_ground_truth(keys, query)

    results = {"n": n, "d": d, "k": k, "bits": bits}
    results["ground_truth"] = {
        "effective_sparsity_90": gt["effective_sparsity_90"],
        "effective_sparsity_95": gt["effective_sparsity_95"],
        "entropy_bits": gt["entropy_bits"],
        "heavy_hitter_fraction": gt["heavy_hitter_fraction"],
    }

    # --- CoarseToFine with different round configurations ---
    print("  Running CoarseToFine...")
    c2f_results = []
    c2f_configs = [
        [(8, min(200 * k, n)), (48, min(20 * k, n)), (d, k)],
        [(16, min(200 * k, n)), (48, min(20 * k, n)), (d, k)],
        [(8, min(100 * k, n)), (32, min(10 * k, n)), (d, k)],
        [(16, min(50 * k, n)), (d, k)],
    ]
    for ci, rounds in enumerate(c2f_configs):
        ctf = CoarseToFine(tq, codes, norms)
        t0 = time.perf_counter()
        ctf_idx, ctf_scores, ctf_stats = ctf.query(query, k=k, rounds=rounds)
        elapsed = time.perf_counter() - t0

        recall = recall_at_k(true_topk, ctf_idx)
        mass = softmax_mass_captured(true_scores, ctf_idx)
        keys_read = ctf_stats["coords_read"] / d  # normalize to "equivalent full keys"

        c2f_results.append({
            "config": str(rounds),
            "recall": float(recall),
            "mass": float(mass),
            "keys_read": float(keys_read),
            "keys_read_fraction": ctf_stats["coords_read_fraction"],
            "time_ms": elapsed * 1000,
        })
        print(f"    Config {ci}: recall={recall:.4f}, mass={mass:.4f}, "
              f"keys_read={keys_read:.0f}, time={elapsed*1000:.1f}ms")

    results["c2f"] = c2f_results

    # --- BlockPruning ---
    print("  Running BlockPruning...")
    bp_results = []
    for block_size in [256, 512, 1024]:
        if block_size > n:
            continue
        bp = BlockPruning(tq, codes, norms, block_size=block_size)
        t0 = time.perf_counter()
        bp_idx, bp_scores, bp_stats = bp.query(query, k=k)
        elapsed = time.perf_counter() - t0

        recall = recall_at_k(true_topk, bp_idx)
        mass = softmax_mass_captured(true_scores, bp_idx)

        bp_results.append({
            "block_size": block_size,
            "recall": float(recall),
            "mass": float(mass),
            "keys_scored": bp_stats["keys_scored"],
            "keys_scored_fraction": bp_stats["keys_scored_fraction"],
            "blocks_pruned": bp_stats["blocks_pruned"],
            "prune_fraction": bp_stats["prune_fraction"],
            "time_ms": elapsed * 1000,
        })
        print(f"    BS={block_size}: recall={recall:.4f}, mass={mass:.4f}, "
              f"pruned={bp_stats['prune_fraction']:.1%}, "
              f"keys_scored={bp_stats['keys_scored_fraction']:.1%}, "
              f"time={elapsed*1000:.1f}ms")

    results["block_pruning"] = bp_results

    # --- CodeTrie (smaller n due to build cost) ---
    print("  Running CodeTrie...")
    ct_n = min(n, 5000)  # CodeTrie is O(n) build with large constant
    ct_results = []
    for block_size_trie in [8, 16]:
        ct = CodeTrie(tq, codes[:ct_n], norms[:ct_n],
                      values=values[:ct_n].astype(np.float64),
                      block_size=block_size_trie)
        ct_k = min(k, ct_n // 2)
        true_topk_ct = np.argsort(true_scores[:ct_n])[::-1][:ct_k]

        t0 = time.perf_counter()
        ct_idx, ct_scores, ct_attn, ct_stats = ct.query(query, k=ct_k)
        elapsed = time.perf_counter() - t0

        recall = recall_at_k(true_topk_ct, ct_idx)

        ct_results.append({
            "block_size": block_size_trie,
            "n_used": ct_n,
            "k_used": ct_k,
            "recall": float(recall),
            "keys_scored": ct_stats["keys_scored"],
            "keys_scored_fraction": ct_stats["keys_scored_fraction"],
            "nodes_visited": ct_stats["nodes_visited"],
            "nodes_pruned": ct_stats["nodes_pruned"],
            "time_ms": elapsed * 1000,
        })
        print(f"    BS={block_size_trie}: recall={recall:.4f}, "
              f"keys_scored={ct_stats['keys_scored_fraction']:.1%}, "
              f"pruned_nodes={ct_stats['nodes_pruned']}, "
              f"time={elapsed*1000:.1f}ms")

    results["code_trie"] = ct_results

    return results


# ---------------------------------------------------------------------------
# Test 2: Attention Quality (L2 error vs k)
# ---------------------------------------------------------------------------
def test_attention_quality(n: int, d: int, bits: int, seed: int):
    """L2 error of attention output for varying k."""
    print(f"\n[Test 2] Attention Quality (n={n:,}, d={d})")

    gen = RealisticAttentionGenerator(
        n=n, d=d, n_sinks=4, local_window=256,
        n_clusters=20, cluster_concentration=50.0,
        seed=seed,
    )
    keys, values, query, metadata = gen.generate()

    key_scales = 0.5 + np.abs(gen.rng.standard_normal(n))
    keys_scaled = keys * key_scales[:, np.newaxis]

    tq = TurboQuant(d, bits=bits, seed=42)
    codes, norms = tq.quantize_batch(keys_scaled)
    true_scores = tq.batch_dot_products(codes, norms, query)

    ks = [10, 50, 100, 200, 500]
    ks = [kk for kk in ks if kk < n // 2]

    results = {"ks": ks}

    for algo_name in ["c2f", "block_pruning", "ideal"]:
        errors = []
        masses = []
        for kk in ks:
            if algo_name == "c2f":
                ctf = CoarseToFine(tq, codes, norms)
                rounds = [
                    (min(16, d), min(200 * kk, n)),
                    (min(48, d), min(20 * kk, n)),
                    (d, kk),
                ]
                idx, _, _ = ctf.query(query, k=kk, rounds=rounds)
            elif algo_name == "block_pruning":
                bp = BlockPruning(tq, codes, norms, block_size=256)
                idx, _, _ = bp.query(query, k=kk)
            else:  # ideal
                idx = np.argsort(true_scores)[::-1][:kk]

            err = _attention_l2_error(true_scores, values, idx, kk)
            mass = softmax_mass_captured(true_scores, idx)
            errors.append(float(err))
            masses.append(float(mass))

        results[algo_name] = {"errors": errors, "masses": masses}
        print(f"  {algo_name:15s}: errors={[f'{e:.4f}' for e in errors]}")

    return results


# ---------------------------------------------------------------------------
# Test 3: Scaling with n
# ---------------------------------------------------------------------------
def test_scaling(d: int, bits: int, k: int, seed: int):
    """How do algorithms scale with sequence length?"""
    print(f"\n[Test 3] Scaling with n (d={d}, k={k})")

    ns = [10_000, 50_000, 100_000]
    # Only go to 500K if we have time
    try:
        # Quick check if 500K is feasible (memory)
        test_size = 500_000 * d * 8  # bytes for float64
        if test_size < 4e9:  # under 4GB
            ns.append(500_000)
    except Exception:
        pass

    results = {"ns": ns, "d": d, "k": k, "bits": bits}

    for n_test in ns:
        print(f"\n  n = {n_test:,}")
        gen = RealisticAttentionGenerator(
            n=n_test, d=d, n_sinks=4, local_window=256,
            n_clusters=20, cluster_concentration=50.0,
            seed=seed,
        )
        keys, values, query, metadata = gen.generate()
        key_scales = 0.5 + np.abs(gen.rng.standard_normal(n_test))
        keys_scaled = keys * key_scales[:, np.newaxis]

        tq = TurboQuant(d, bits=bits, seed=42)
        codes, norms = tq.quantize_batch(keys_scaled)

        # Brute force
        t0 = time.perf_counter()
        true_scores = tq.batch_dot_products(codes, norms, query)
        t_bf = time.perf_counter() - t0
        true_topk = np.argsort(true_scores)[::-1][:k]

        # C2F
        rounds = [
            (min(16, d), min(200 * k, n_test)),
            (min(48, d), min(20 * k, n_test)),
            (d, k),
        ]
        ctf = CoarseToFine(tq, codes, norms)
        t0 = time.perf_counter()
        c2f_idx, _, c2f_stats = ctf.query(query, k=k, rounds=rounds)
        t_c2f = time.perf_counter() - t0
        c2f_recall = recall_at_k(true_topk, c2f_idx)

        # BlockPruning
        bp = BlockPruning(tq, codes, norms, block_size=512)
        t0 = time.perf_counter()
        bp_idx, _, bp_stats = bp.query(query, k=k)
        t_bp = time.perf_counter() - t0
        bp_recall = recall_at_k(true_topk, bp_idx)

        entry = {
            "brute_force_ms": t_bf * 1000,
            "c2f_ms": t_c2f * 1000,
            "c2f_recall": float(c2f_recall),
            "c2f_keys_fraction": c2f_stats["coords_read_fraction"],
            "bp_ms": t_bp * 1000,
            "bp_recall": float(bp_recall),
            "bp_prune_fraction": bp_stats["prune_fraction"],
            "bp_keys_fraction": bp_stats["keys_scored_fraction"],
        }
        results[str(n_test)] = entry

        print(f"    BF: {t_bf*1000:.1f}ms")
        print(f"    C2F: {t_c2f*1000:.1f}ms, recall={c2f_recall:.4f}, "
              f"keys_read={c2f_stats['coords_read_fraction']:.1%}")
        print(f"    BP: {t_bp*1000:.1f}ms, recall={bp_recall:.4f}, "
              f"pruned={bp_stats['prune_fraction']:.1%}, "
              f"keys_scored={bp_stats['keys_scored_fraction']:.1%}")

    return results


# ---------------------------------------------------------------------------
# Test 4: Attention Pattern Visualization
# ---------------------------------------------------------------------------
def test_attention_pattern_viz(n: int, d: int, seed: int):
    """Visualize the generated attention pattern."""
    print(f"\n[Test 4] Attention Pattern Visualization (n={n:,})")

    gen = RealisticAttentionGenerator(
        n=n, d=d, n_sinks=4, local_window=256,
        n_clusters=20, cluster_concentration=50.0,
        seed=seed,
    )
    keys, values, query, metadata = gen.generate()
    gt = compute_ground_truth(keys, query)

    scores = gt["scores"]
    weights = gt["attention_weights"]
    sorted_idx = np.argsort(weights)[::-1]

    return {
        "scores": scores,
        "weights": weights,
        "sorted_idx": sorted_idx,
        "metadata": {
            "n_sinks": metadata["n_sinks"],
            "sink_indices": metadata["sink_indices"].tolist(),
            "local_window_start": metadata["local_window_start"],
            "cluster_assignments": metadata["cluster_assignments"].tolist(),
            "target_cluster": metadata["target_cluster"],
            "effective_sparsity_90": gt["effective_sparsity_90"],
            "effective_sparsity_95": gt["effective_sparsity_95"],
            "effective_sparsity_99": gt["effective_sparsity_99"],
            "entropy_bits": gt["entropy_bits"],
            "heavy_hitter_fraction": gt["heavy_hitter_fraction"],
            "gini_coefficient": gt["gini_coefficient"],
            "mass_in_top": {str(k): v for k, v in gt["mass_in_top"].items()},
        },
    }


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------
def plot_recall_vs_keys_read(results, ns_tested):
    """Plot 1: The money plot -- recall@k vs bandwidth for each algorithm."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect data points from the best config at each n
    for n_key, res in results.items():
        if not isinstance(res, dict) or "c2f" not in res:
            continue
        n = res["n"]

        # C2F points
        for cr in res["c2f"]:
            ax.scatter(cr["keys_read_fraction"] * 100, cr["recall"],
                       marker='o', s=80, c='C0', zorder=5, alpha=0.7)

        # BP points
        for br in res["block_pruning"]:
            ax.scatter(br["keys_scored_fraction"] * 100, br["recall"],
                       marker='s', s=80, c='C1', zorder=5, alpha=0.7)

        # CodeTrie points
        for tr in res.get("code_trie", []):
            ax.scatter(tr["keys_scored_fraction"] * 100, tr["recall"],
                       marker='^', s=80, c='C2', zorder=5, alpha=0.7)

    # Add legend handles
    ax.scatter([], [], marker='o', c='C0', s=80, label='CoarseToFine')
    ax.scatter([], [], marker='s', c='C1', s=80, label='BlockPruning')
    ax.scatter([], [], marker='^', c='C2', s=80, label='CodeTrie')

    ax.axhline(y=1.0, color='gray', ls=':', alpha=0.5, label='Perfect recall')
    ax.axhline(y=0.95, color='red', ls=':', alpha=0.3, label='95% recall')

    ax.set_xlabel('Keys read (%)')
    ax.set_ylabel('Recall@$k$')
    ax.set_title('Recall vs Bandwidth (Realistic LLM Attention)')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp07_recall_vs_keys_read.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'exp07_recall_vs_keys_read.png'}")


def plot_attention_quality(quality_results):
    """Plot 2: L2 error vs k for each algorithm."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ks = quality_results["ks"]
    ks_arr = np.array(ks, dtype=float)

    colors = {"c2f": "C0", "block_pruning": "C1", "ideal": "C3"}
    labels = {"c2f": "CoarseToFine", "block_pruning": "BlockPruning", "ideal": "Ideal top-$k$"}
    markers = {"c2f": "o", "block_pruning": "s", "ideal": "D"}

    for algo in ["c2f", "block_pruning", "ideal"]:
        res = quality_results[algo]

        ax1.plot(ks_arr, res["errors"], f'{markers[algo]}-', lw=2,
                 color=colors[algo], label=labels[algo], markersize=7)

        ax2.plot(ks_arr, res["masses"], f'{markers[algo]}-', lw=2,
                 color=colors[algo], label=labels[algo], markersize=7)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Keys found $k$')
    ax1.set_ylabel('Relative $L_2$ error')
    ax1.set_title('Attention Output Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xscale('log')
    ax2.set_xlabel('Keys found $k$')
    ax2.set_ylabel('Softmax mass captured')
    ax2.set_title('Softmax Mass Captured')
    ax2.axhline(y=0.99, color='red', ls=':', alpha=0.5, label='99% mass')
    ax2.axhline(y=0.95, color='gray', ls=':', alpha=0.5, label='95% mass')
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Attention Quality with Realistic LLM Patterns', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp07_attention_quality.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'exp07_attention_quality.png'}")


def plot_scaling(scaling_results):
    """Plot 3: Performance vs n."""
    ns = scaling_results["ns"]
    ns_arr = np.array(ns, dtype=float)

    bf_times = [scaling_results[str(n)]["brute_force_ms"] for n in ns]
    c2f_times = [scaling_results[str(n)]["c2f_ms"] for n in ns]
    bp_times = [scaling_results[str(n)]["bp_ms"] for n in ns]

    c2f_recalls = [scaling_results[str(n)]["c2f_recall"] for n in ns]
    bp_recalls = [scaling_results[str(n)]["bp_recall"] for n in ns]
    bp_prune = [scaling_results[str(n)]["bp_prune_fraction"] for n in ns]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Time scaling
    ax1.plot(ns_arr, bf_times, 'D-', lw=2, label='Brute force', color='gray')
    ax1.plot(ns_arr, c2f_times, 'o-', lw=2, label='CoarseToFine', color='C0')
    ax1.plot(ns_arr, bp_times, 's-', lw=2, label='BlockPruning', color='C1')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Sequence length $n$')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Latency Scaling')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Recall scaling
    ax2.plot(ns_arr, c2f_recalls, 'o-', lw=2, label='CoarseToFine', color='C0')
    ax2.plot(ns_arr, bp_recalls, 's-', lw=2, label='BlockPruning', color='C1')
    ax2.axhline(y=1.0, color='gray', ls=':', alpha=0.5)
    ax2.axhline(y=0.95, color='red', ls=':', alpha=0.3, label='95% target')
    ax2.set_xscale('log')
    ax2.set_xlabel('Sequence length $n$')
    ax2.set_ylabel('Recall@$k$')
    ax2.set_title('Recall Scaling')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Block pruning fraction
    ax3.bar(range(len(ns)), [pf * 100 for pf in bp_prune],
            tick_label=[f'{n//1000}K' for n in ns], color='C1', alpha=0.8)
    ax3.set_xlabel('Sequence length $n$')
    ax3.set_ylabel('Blocks pruned (%)')
    ax3.set_title('BlockPruning Effectiveness')
    ax3.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Performance Scaling with Realistic Attention', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp07_scaling.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'exp07_scaling.png'}")


def plot_attention_pattern(viz_results):
    """Plot 4: Visualize the generated attention pattern."""
    weights = viz_results["weights"]
    scores = viz_results["scores"]
    meta = viz_results["metadata"]
    n = len(weights)

    cluster_assignments = np.array(meta["cluster_assignments"])
    target_cluster = meta["target_cluster"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Attention weights by position
    ax = axes[0, 0]
    positions = np.arange(n)
    ax.scatter(positions, weights, s=0.5, alpha=0.3, c='C0', rasterized=True)

    # Highlight sinks
    sink_idx = meta["sink_indices"]
    if len(sink_idx) > 0:
        ax.scatter(sink_idx, weights[sink_idx], s=40, c='red',
                   marker='*', zorder=5, label='Sinks')

    # Highlight local window
    lw_start = meta["local_window_start"]
    lw_mask = positions >= lw_start
    ax.scatter(positions[lw_mask], weights[lw_mask], s=1, alpha=0.5,
               c='C2', rasterized=True, label='Local window')

    # Highlight target cluster
    tc_mask = cluster_assignments == target_cluster
    ax.scatter(positions[tc_mask], weights[tc_mask], s=2, alpha=0.5,
               c='C1', rasterized=True, label=f'Target cluster')

    ax.set_xlabel('Token position')
    ax.set_ylabel('Attention weight')
    ax.set_title('Attention Distribution by Position')
    ax.set_yscale('log')
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(True, alpha=0.3)

    # (b) Sorted attention weights (log scale)
    ax = axes[0, 1]
    sorted_weights = np.sort(weights)[::-1]
    ax.plot(np.arange(1, n + 1), sorted_weights, lw=1.5, color='C0')
    ax.axvline(x=meta["effective_sparsity_90"], color='C1', ls='--',
               label=f'90% mass: k={meta["effective_sparsity_90"]}')
    ax.axvline(x=meta["effective_sparsity_95"], color='C2', ls='--',
               label=f'95% mass: k={meta["effective_sparsity_95"]}')
    ax.axvline(x=meta["effective_sparsity_99"], color='C3', ls='--',
               label=f'99% mass: k={meta["effective_sparsity_99"]}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Attention weight')
    ax.set_title('Sorted Attention Weights')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) Cumulative mass
    ax = axes[1, 0]
    cumsum = np.cumsum(sorted_weights)
    ax.plot(np.arange(1, n + 1), cumsum, lw=2, color='C0')
    ax.axhline(y=0.90, color='C1', ls=':', alpha=0.5, label='90%')
    ax.axhline(y=0.95, color='C2', ls=':', alpha=0.5, label='95%')
    ax.axhline(y=0.99, color='C3', ls=':', alpha=0.5, label='99%')
    ax.set_xscale('log')
    ax.set_xlabel('Number of tokens $k$')
    ax.set_ylabel('Cumulative softmax mass')
    ax.set_title('Cumulative Mass Concentration')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # (d) Score histogram by category
    ax = axes[1, 1]
    # Categorize tokens
    is_sink = np.zeros(n, dtype=bool)
    if len(sink_idx) > 0:
        is_sink[sink_idx] = True
    is_local = (positions >= lw_start) & ~is_sink
    is_target = (cluster_assignments == target_cluster) & ~is_sink & ~is_local
    is_other = ~is_sink & ~is_local & ~is_target

    for mask, label, color in [
        (is_other, 'Other tokens', 'C0'),
        (is_local, 'Local window', 'C2'),
        (is_target, 'Target cluster', 'C1'),
        (is_sink, 'Sinks', 'red'),
    ]:
        if np.any(mask):
            ax.hist(scores[mask], bins=50, alpha=0.5, color=color,
                    label=f'{label} (n={np.sum(mask)})', density=True)

    ax.set_xlabel('Dot-product score')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution by Token Category')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add text summary
    fig.text(0.5, -0.02,
             f"n={n:,} | Entropy={meta['entropy_bits']:.1f} bits | "
             f"Gini={meta['gini_coefficient']:.3f} | "
             f"Heavy hitters={meta['heavy_hitter_fraction']:.1%}",
             ha='center', fontsize=11, style='italic')

    fig.suptitle('Realistic LLM Attention Pattern', fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp07_attention_pattern.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'exp07_attention_pattern.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Experiment 07: Realistic LLM Attention Patterns")
    print("=" * 70)

    D = 128
    BITS = 2
    K = 100
    SEED = 42

    all_results = {}
    t0_total = time.time()

    # -----------------------------------------------------------------------
    # Test 1: Recall vs Keys Read at multiple scales
    # -----------------------------------------------------------------------
    ns_test1 = [10_000, 50_000, 100_000]
    recall_results = {}
    for n in ns_test1:
        key = f"n={n}"
        recall_results[key] = test_recall_vs_keys_read(n, D, BITS, K, SEED)
    all_results["test1_recall_vs_keys_read"] = recall_results

    # -----------------------------------------------------------------------
    # Test 2: Attention quality
    # -----------------------------------------------------------------------
    quality_results = test_attention_quality(50_000, D, BITS, SEED)
    all_results["test2_attention_quality"] = quality_results

    # -----------------------------------------------------------------------
    # Test 3: Scaling
    # -----------------------------------------------------------------------
    scaling_results = test_scaling(D, BITS, K, SEED)
    all_results["test3_scaling"] = scaling_results

    # -----------------------------------------------------------------------
    # Test 4: Attention pattern visualization
    # -----------------------------------------------------------------------
    viz_results = test_attention_pattern_viz(50_000, D, SEED)
    all_results["test4_attention_pattern"] = {
        "metadata": viz_results["metadata"]
    }

    # -----------------------------------------------------------------------
    # Generate figures
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Generating figures...")
    print("-" * 70)

    plot_recall_vs_keys_read(recall_results, ns_test1)
    plot_attention_quality(quality_results)
    plot_scaling(scaling_results)
    plot_attention_pattern(viz_results)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    elapsed = time.time() - t0_total
    all_results["parameters"] = {
        "d": D, "bits": BITS, "k": K, "seed": SEED,
        "ns_tested": ns_test1,
    }
    all_results["total_time_seconds"] = float(elapsed)

    # Make JSON-serializable (remove numpy arrays)
    def _make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serializable(v) for v in obj]
        return obj

    serializable = _make_serializable(all_results)

    results_path = RESULTS_DIR / 'exp07_results.json'
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"Total time: {elapsed:.1f}s")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Attention pattern properties:")
    viz_meta = viz_results["metadata"]
    print(f"    Effective sparsity (90% mass): {viz_meta['effective_sparsity_90']} tokens")
    print(f"    Effective sparsity (95% mass): {viz_meta['effective_sparsity_95']} tokens")
    print(f"    Heavy hitter fraction: {viz_meta['heavy_hitter_fraction']:.1%}")
    print(f"    Entropy: {viz_meta['entropy_bits']:.1f} bits")
    print(f"    Gini coefficient: {viz_meta['gini_coefficient']:.3f}")

    print(f"\n  Algorithm performance (n=100K, k=100):")
    if "n=100000" in recall_results:
        r100k = recall_results["n=100000"]
        if r100k.get("c2f"):
            best_c2f = max(r100k["c2f"], key=lambda x: x["recall"])
            print(f"    C2F best:  recall={best_c2f['recall']:.4f}, "
                  f"keys_read={best_c2f['keys_read_fraction']:.1%}")
        if r100k.get("block_pruning"):
            best_bp = max(r100k["block_pruning"], key=lambda x: x["recall"])
            print(f"    BP best:   recall={best_bp['recall']:.4f}, "
                  f"pruned={best_bp['prune_fraction']:.1%}, "
                  f"keys_scored={best_bp['keys_scored_fraction']:.1%}")

    print(f"\n  BlockPruning prune fractions across scales:")
    for n in ns_test1:
        key = f"n={n}"
        if key in recall_results and recall_results[key].get("block_pruning"):
            for bp_r in recall_results[key]["block_pruning"]:
                print(f"    n={n:>7,}, BS={bp_r['block_size']:>5}: "
                      f"pruned={bp_r['prune_fraction']:.1%}")


if __name__ == '__main__':
    main()
