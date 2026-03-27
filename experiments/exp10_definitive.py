"""Experiment 10: The Definitive Result — Fused CUDA on Realistic Attention Data.

This is THE headline experiment for the QuantDex paper. It combines:
  - Realistic LLM attention patterns (exp07) with clustered keys, attention sinks,
    and local windows that produce sparse softmax distributions
  - Fused CUDA kernels (exp09) that eliminate Python overhead and achieve
    bandwidth-optimal coarse-to-fine search

The key hypothesis: on realistic (structured) data, the coarse coordinates
strongly separate heavy-hitter keys from the background, yielding BOTH
high speedup AND high recall simultaneously — unlike random data where
recall degrades to 63-89%.

Expected: 5-9x speedup with 95-100% recall on realistic attention data.
"""
import os, sys, json, time, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cupy as cp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6), 'figure.dpi': 150})

from exp09_fused_cuda import (
    _partial_score_kernel, _full_score_kernel, _survivor_score_kernel,
    lloyd_max_levels_2bit, lloyd_max_boundaries_2bit,
    quantize_gpu, gpu_time_ms,
    bench_brute_force_gemm, bench_brute_force_fused, bench_c2f_fused,
)
from src.attention_patterns import RealisticAttentionGenerator, compute_ground_truth

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def rotate_query_gpu(q_gpu, signs_gpu, d):
    """Apply RHT rotation to query on GPU. Returns rotated query (d,)."""
    q_rot = (q_gpu * signs_gpu).reshape(1, -1)
    h = 1
    while h < d:
        r = q_rot.reshape(1, -1, 2 * h)
        a = r[:, :, :h].copy()
        b = r[:, :, h:].copy()
        r[:, :, :h] = a + b
        r[:, :, h:] = a - b
        h *= 2
    q_rot = (q_rot / cp.sqrt(cp.float32(d))).ravel()
    return q_rot


def compute_attention_quality(bf_scores_gpu, c2f_scores_gpu, bf_idx_gpu, c2f_idx_gpu, n, d):
    """Compute attention quality metrics: softmax L2 error between brute force and C2F.

    Uses the top-k indices from each method to compute softmax-weighted outputs,
    then measures the L2 error between them.
    """
    # Get CPU arrays
    bf_idx = cp.asnumpy(bf_idx_gpu)
    bf_scores = cp.asnumpy(bf_scores_gpu).astype(np.float64)
    c2f_idx = cp.asnumpy(c2f_idx_gpu)
    c2f_scores = cp.asnumpy(c2f_scores_gpu).astype(np.float64)

    # Compute softmax weights for brute force top-k
    bf_max = np.max(bf_scores)
    bf_exp = np.exp(bf_scores - bf_max)
    bf_weights = bf_exp / np.sum(bf_exp)

    # Compute softmax weights for C2F top-k
    c2f_max = np.max(c2f_scores)
    c2f_exp = np.exp(c2f_scores - c2f_max)
    c2f_weights = c2f_exp / np.sum(c2f_exp)

    # For attention quality: compare softmax mass distribution
    # Build a unified index set and compare weight vectors
    all_idx = np.union1d(bf_idx, c2f_idx)
    bf_weight_map = dict(zip(bf_idx, bf_weights))
    c2f_weight_map = dict(zip(c2f_idx, c2f_weights))

    bf_vec = np.array([bf_weight_map.get(i, 0.0) for i in all_idx])
    c2f_vec = np.array([c2f_weight_map.get(i, 0.0) for i in all_idx])

    # L2 error of softmax weight vectors
    l2_error = np.linalg.norm(bf_vec - c2f_vec)

    # Also compute softmax mass captured: sum of BF softmax weights for C2F survivors
    mass_captured = sum(bf_weight_map.get(i, 0.0) for i in c2f_idx)

    return float(l2_error), float(mass_captured)


def main():
    print("=" * 72)
    print("Experiment 10: DEFINITIVE — Fused CUDA on Realistic Attention Data")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
    free_gb = cp.cuda.runtime.memGetInfo()[0] / 1e9
    print(f"VRAM: {free_gb:.1f} GB free")
    print("=" * 72)

    d, b, k = 128, 2, 100
    levels_np = lloyd_max_levels_2bit(d)
    bounds_np = lloyd_max_boundaries_2bit(d)
    levels_gpu = cp.array(levels_np)

    # Fixed RHT signs (must be consistent between keys and query)
    rng_signs = np.random.RandomState(42)
    signs_np = rng_signs.choice([-1.0, 1.0], size=d).astype(np.float32)
    signs_gpu = cp.array(signs_np)

    n_values = [50_000, 100_000, 500_000, 1_000_000, 2_000_000]

    # C2F configs: (name, m, keep_mult_func)
    # keep_mult scales with n to keep a constant survival FRACTION, ensuring
    # all heavy hitters (~5-10% of n) survive the coarse round.
    # At n=1M, k=100: keep_mult=2000 means 200K survivors = 20% of n.
    # Speedup comes from the fine round touching only 20% of keys at full d.
    def make_configs(n_val):
        """Adaptive keep_mult: target survival fractions of ~40%, ~20%, ~10%."""
        return [
            ("C2F m=8",  8,  max(200, int(0.40 * n_val / k))),
            ("C2F m=16", 16, max(100, int(0.20 * n_val / k))),
            ("C2F m=32", 32, max(50,  int(0.10 * n_val / k))),
        ]

    results = {}

    for n in n_values:
        # VRAM budget check: codes (n*d) + codes_T (n*d) + norms (4*n) + misc
        vram_need = n * d * 2 + n * 4 * 3  # codes + codes_T + norms + scores buffers
        free = cp.cuda.runtime.memGetInfo()[0]
        if vram_need > free * 0.85:
            print(f"\nn={n:>9,}: skipping (need {vram_need/1e9:.2f} GB, have {free/1e9:.2f} GB)")
            continue

        print(f"\n{'='*72}")
        print(f"n = {n:,}   |   d = {d}   |   b = {b}-bit   |   k = {k}")
        print(f"{'='*72}")

        # ── Step 1: Generate realistic attention data on CPU ──
        print("  Generating realistic attention data...")
        t0 = time.perf_counter()
        gen = RealisticAttentionGenerator(
            n=n, d=d,
            n_sinks=4,
            local_window=min(256, n),
            n_clusters=20,
            cluster_concentration=200.0,
            sparsity=0.05,
            seed=42,
        )
        keys, values, query, metadata = gen.generate()
        t_gen = time.perf_counter() - t0
        print(f"  Generated in {t_gen:.2f}s")

        # Attention statistics (on a subsample for large n)
        if n <= 500_000:
            gt = compute_ground_truth(keys, query.astype(np.float64))
            print(f"  Attention stats: eff_sparsity_90={gt['effective_sparsity_90']}, "
                  f"eff_sparsity_95={gt['effective_sparsity_95']}, "
                  f"gini={gt['gini_coefficient']:.4f}, "
                  f"heavy_hitters={gt['heavy_hitter_fraction']:.4f}")
            attn_stats = {
                "eff_sparsity_90": int(gt['effective_sparsity_90']),
                "eff_sparsity_95": int(gt['effective_sparsity_95']),
                "gini": float(gt['gini_coefficient']),
                "heavy_hitter_fraction": float(gt['heavy_hitter_fraction']),
                "entropy_bits": float(gt['entropy_bits']),
            }
        else:
            attn_stats = {"note": "skipped for large n"}

        # ── Step 2: Transfer to GPU and quantize ──
        print("  Transferring to GPU and quantizing...")
        keys_f32 = keys.astype(np.float32)
        query_f32 = query.astype(np.float32)

        keys_gpu = cp.array(keys_f32)
        query_gpu = cp.array(query_f32)

        t0 = time.perf_counter()
        codes_gpu, norms_gpu = quantize_gpu(keys_gpu, signs_gpu, bounds_np)
        cp.cuda.Stream.null.synchronize()
        t_quant = time.perf_counter() - t0
        print(f"  Quantized {n:,} keys in {t_quant:.3f}s")

        # Columnar layout for fused kernels
        codes_T_gpu = cp.ascontiguousarray(codes_gpu.T)  # (d, n)

        # Rotate query
        q_rot = rotate_query_gpu(query_gpu, signs_gpu, d)
        coord_order = cp.argsort(cp.abs(q_rot))[::-1].astype(cp.int32)

        # Free raw keys from GPU
        del keys_gpu, query_gpu
        cp.get_default_memory_pool().free_all_blocks()

        # ── Step 3: Brute force GEMM baseline ──
        print("  Running brute force (GEMM)...")
        bf_gemm_ms, (bf_gemm_idx, bf_gemm_scores) = gpu_time_ms(
            lambda: bench_brute_force_gemm(codes_gpu, norms_gpu, q_rot, levels_gpu, k),
            warmup=5, runs=20,
        )
        bf_bw_gemm = n * d * 4 / (bf_gemm_ms / 1000) / 1e9
        print(f"  BF (GEMM):   {bf_gemm_ms:8.3f} ms   ({bf_bw_gemm:.0f} GB/s eff)")

        # ── Step 4: Brute force FUSED baseline ──
        print("  Running brute force (fused uint8)...")
        bf_fused_ms, (bf_fused_idx, bf_fused_scores) = gpu_time_ms(
            lambda: bench_brute_force_fused(codes_T_gpu, norms_gpu, q_rot, levels_gpu, k, d),
            warmup=5, runs=20,
        )
        bf_bw_fused = n * d * 1 / (bf_fused_ms / 1000) / 1e9
        print(f"  BF (fused):  {bf_fused_ms:8.3f} ms   ({bf_bw_fused:.0f} GB/s eff)")

        # Ground truth: use GEMM brute force as reference
        bf_set = set(cp.asnumpy(bf_gemm_idx).tolist())

        entry = {
            "n": n,
            "d": d,
            "b": b,
            "k": k,
            "bf_gemm_ms": float(bf_gemm_ms),
            "bf_fused_ms": float(bf_fused_ms),
            "bf_bw_gemm_GBs": float(bf_bw_gemm),
            "bf_bw_fused_GBs": float(bf_bw_fused),
            "attention_stats": attn_stats,
            "generation_time_s": float(t_gen),
            "quantization_time_s": float(t_quant),
        }

        # ── Step 5: C2F at various m values ──
        print(f"\n  {'Method':<22s} {'Time (ms)':>10s} {'vs GEMM':>8s} {'vs Fused':>9s} "
              f"{'Recall':>7s} {'Surv%':>7s} {'Reads%':>7s} {'L2 Err':>8s} {'Mass':>6s}")
        print("  " + "-" * 90)

        c2f_configs = make_configs(n)
        for name, m1, keep_mult in c2f_configs:
            c2f_ms, (c2f_idx, c2f_scores, n_surv) = gpu_time_ms(
                lambda m1=m1, km=keep_mult: bench_c2f_fused(
                    codes_T_gpu, norms_gpu, q_rot, levels_gpu, coord_order,
                    k, m1=m1, keep_mult=km),
                warmup=5, runs=20,
            )

            c2f_set = set(cp.asnumpy(c2f_idx).tolist())
            recall = len(bf_set & c2f_set) / k if k > 0 else 0
            speedup_gemm = bf_gemm_ms / c2f_ms
            speedup_fused = bf_fused_ms / c2f_ms
            surv_pct = n_surv / n * 100
            reads_pct = (m1 * n + (d - m1) * n_surv) / (d * n) * 100

            # Attention quality
            l2_err, mass_captured = compute_attention_quality(
                bf_gemm_scores, c2f_scores, bf_gemm_idx, c2f_idx, n, d
            )

            print(f"  {name:<22s} {c2f_ms:>10.3f} {speedup_gemm:>7.2f}x {speedup_fused:>8.2f}x "
                  f"{recall:>7.2%} {surv_pct:>6.1f}% {reads_pct:>6.0f}% "
                  f"{l2_err:>8.4f} {mass_captured:>5.2%}")

            entry[name] = {
                "ms": float(c2f_ms),
                "speedup_vs_gemm": float(speedup_gemm),
                "speedup_vs_fused": float(speedup_fused),
                "recall": float(recall),
                "survivors_pct": float(surv_pct),
                "reads_pct": float(reads_pct),
                "n_survivors": int(n_surv),
                "l2_error": float(l2_err),
                "softmax_mass_captured": float(mass_captured),
            }

        results[str(n)] = entry

        # Clean up VRAM
        del codes_gpu, codes_T_gpu, norms_gpu
        del bf_gemm_idx, bf_gemm_scores, bf_fused_idx, bf_fused_scores
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    # ── Save results ──
    with open(os.path.join(RESULTS_DIR, 'exp10_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {os.path.join(RESULTS_DIR, 'exp10_results.json')}")

    # ── Generate figures ──
    print("\nGenerating figures...")
    ns = sorted([int(k) for k in results.keys()])

    if len(ns) < 2:
        print("  Not enough data points for figures.")
        return results

    # ── Figure 1: THE HEADLINE — Speedup vs Recall ──
    fig, ax = plt.subplots(figsize=(10, 7))

    # Config names/styles for figures (consistent across all n)
    fig_configs = [("C2F m=8", 8), ("C2F m=16", 16), ("C2F m=32", 32)]
    colors = {'C2F m=8': '#2ca02c', 'C2F m=16': '#1f77b4', 'C2F m=32': '#ff7f0e'}
    markers = {'C2F m=8': 'o', 'C2F m=16': 's', 'C2F m=32': '^'}

    for name, _ in fig_configs:
        recalls = []
        speedups = []
        ns_for_label = []
        for n_val in ns:
            r = results[str(n_val)]
            if name in r:
                recalls.append(r[name]["recall"])
                speedups.append(r[name]["speedup_vs_gemm"])
                ns_for_label.append(n_val)

        if recalls:
            ax.scatter(recalls, speedups, c=colors[name], marker=markers[name],
                       s=120, zorder=5, edgecolors='black', linewidth=0.5,
                       label=f'{name}')
            # Annotate each point with n
            for i, n_val in enumerate(ns_for_label):
                if n_val >= 1_000_000:
                    label_txt = f'{n_val/1e6:.0f}M'
                else:
                    label_txt = f'{n_val/1e3:.0f}K'
                ax.annotate(label_txt, (recalls[i], speedups[i]),
                            textcoords="offset points", xytext=(8, 5),
                            fontsize=9, color=colors[name])

    # Draw the "ideal" region
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.4, label='Break-even')
    ax.axvspan(0.95, 1.01, alpha=0.08, color='green', label='High recall (>95%)')

    ax.set_xlabel('Recall@100 (fraction of true top-100 found)', fontsize=13)
    ax.set_ylabel('Speedup over brute force (GEMM)', fontsize=13)
    ax.set_title('QuantDex: Speedup vs Recall on Realistic LLM Attention\n'
                 '(RTX 3070, d=128, 2-bit TurboQuant, structured KV cache)',
                 fontsize=13)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 1.02)
    # Set y range dynamically
    all_speedups = [results[str(n)][name]["speedup_vs_gemm"]
                    for n in ns for name, _ in fig_configs if name in results[str(n)]]
    if all_speedups:
        ax.set_ylim(0, max(all_speedups) * 1.2)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'exp10_headline.png')
    fig.savefig(path, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

    # ── Figure 2: Wall-clock time vs n ──
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ns, [results[str(n)]["bf_gemm_ms"] for n in ns],
            'o-', label='Brute force (cuBLAS GEMM)', linewidth=2.5,
            markersize=8, color='#d62728')
    ax.plot(ns, [results[str(n)]["bf_fused_ms"] for n in ns],
            's-', label='Brute force (fused uint8)', linewidth=2,
            markersize=7, color='#ff7f0e')
    for name, _ in fig_configs:
        times = [results[str(n)].get(name, {}).get("ms", np.nan) for n in ns]
        ax.plot(ns, times, '^--', label=f'{name}', linewidth=1.5, markersize=6,
                color=colors.get(name, None))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of keys $n$', fontsize=13)
    ax.set_ylabel('Wall-clock time (ms)', fontsize=13)
    ax.set_title('GPU Attention Latency on Realistic Data\n'
                 '(RTX 3070, d=128, 2-bit TurboQuant)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'exp10_wallclock_realistic.png')
    fig.savefig(path, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

    # ── Figure 3: Speedup factor vs n ──
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, _ in fig_configs:
        su_gemm = [results[str(n)].get(name, {}).get("speedup_vs_gemm", np.nan) for n in ns]
        ax.plot(ns, su_gemm, 'o-', label=f'{name} vs GEMM', linewidth=2,
                markersize=8, color=colors.get(name, None))

    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Break-even')
    ax.set_xscale('log')
    ax.set_xlabel('Number of keys $n$', fontsize=13)
    ax.set_ylabel('Speedup factor', fontsize=13)
    ax.set_title('C2F Speedup vs Brute Force on Realistic Data\n'
                 '(should increase monotonically with $n$)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'exp10_scaling.png')
    fig.savefig(path, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

    # ── Summary Table ──
    print(f"\n{'='*90}")
    print("HEADLINE RESULT: Fused C2F on Realistic LLM Attention Data")
    print(f"{'='*90}")
    print(f"{'n':>12s}  {'BF GEMM':>9s}  {'BF fused':>9s}  "
          f"{'C2F m=8':>9s}  {'C2F m=16':>9s}  {'C2F m=32':>9s}  "
          f"{'R@8':>5s}  {'R@16':>5s}  {'R@32':>5s}")
    print("-" * 90)
    for n_val in ns:
        r = results[str(n_val)]
        c8 = r.get("C2F m=8", {})
        c16 = r.get("C2F m=16", {})
        c32 = r.get("C2F m=32", {})
        print(f"{n_val:>12,}  {r['bf_gemm_ms']:>8.2f}ms  {r['bf_fused_ms']:>8.2f}ms  "
              f"{c8.get('ms', 0):>8.2f}ms  {c16.get('ms', 0):>8.2f}ms  {c32.get('ms', 0):>8.2f}ms  "
              f"{c8.get('recall', 0):>5.1%}  {c16.get('recall', 0):>5.1%}  {c32.get('recall', 0):>5.1%}")

    print(f"\n{'Speedup vs GEMM:':<20s}")
    for n_val in ns:
        r = results[str(n_val)]
        c8 = r.get("C2F m=8", {})
        c16 = r.get("C2F m=16", {})
        c32 = r.get("C2F m=32", {})
        print(f"{n_val:>12,}  {'':>9s}  {'':>9s}  "
              f"{c8.get('speedup_vs_gemm', 0):>8.2f}x  "
              f"{c16.get('speedup_vs_gemm', 0):>8.2f}x  "
              f"{c32.get('speedup_vs_gemm', 0):>8.2f}x")

    print(f"\n{'Attention Quality (L2 error / mass captured):'}")
    for n_val in ns:
        r = results[str(n_val)]
        parts = []
        for name, _ in fig_configs:
            c = r.get(name, {})
            parts.append(f"{c.get('l2_error', 0):.4f}/{c.get('softmax_mass_captured', 0):.2%}")
        print(f"  n={n_val:>10,}: " + "  |  ".join(parts))

    print(f"\n{'='*90}")
    return results


if __name__ == "__main__":
    main()
