"""Experiment 11: Ablation Studies — Mapping the Parameter Space.

Systematically varies m (coarse coordinates), d (dimension), and b (bit-width)
to characterize the full performance landscape of QuantDex C2F search.

Uses fused CUDA kernels from exp09, realistic attention data from exp07.
"""
import os, sys, json, time, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cupy as cp
from scipy.stats import norm as sp_norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6), 'figure.dpi': 150})

from exp09_fused_cuda import (
    _partial_score_kernel, _full_score_kernel, _survivor_score_kernel,
    quantize_gpu, gpu_time_ms,
    bench_brute_force_gemm, bench_brute_force_fused, bench_c2f_fused,
)
from quantdex.attention_patterns import RealisticAttentionGenerator

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Peak memory bandwidth of RTX 3070
PEAK_BW_GBs = 448.0


# ── Lloyd-Max for arbitrary bit widths ──────────────────────────────

def lloyd_max_gaussian(bits, sigma, max_iter=200, tol=1e-12):
    """Compute optimal Lloyd-Max levels and boundaries for N(0, sigma^2)."""
    n_levels = 1 << bits

    if bits == 1:
        c = sigma * np.sqrt(2.0 / np.pi)
        return np.array([0.0], dtype=np.float32), np.array([-c, c], dtype=np.float32)

    levels = np.linspace(-3.0 * sigma, 3.0 * sigma, n_levels)

    for _ in range(max_iter):
        thresholds = 0.5 * (levels[:-1] + levels[1:])
        boundaries = np.concatenate([[-np.inf], thresholds, [np.inf]])
        new_levels = np.empty(n_levels)
        for j in range(n_levels):
            lo = boundaries[j]
            hi = boundaries[j + 1]
            lo_n = lo / sigma if np.isfinite(lo) else -np.inf
            hi_n = hi / sigma if np.isfinite(hi) else np.inf
            num = sigma * (sp_norm.pdf(lo_n) - sp_norm.pdf(hi_n))
            den = sp_norm.cdf(hi_n) - sp_norm.cdf(lo_n)
            if den < 1e-30:
                new_levels[j] = 0.5 * (lo + hi) if np.isfinite(lo + hi) else 0.0
            else:
                new_levels[j] = num / den
        if np.max(np.abs(new_levels - levels)) < tol:
            levels = new_levels
            break
        levels = new_levels

    thresholds = 0.5 * (levels[:-1] + levels[1:])
    return thresholds.astype(np.float32), levels.astype(np.float32)


def quantize_gpu_nbits(X_gpu, signs_gpu, boundaries):
    """Quantize on GPU with arbitrary boundaries (1,2,3,4-bit)."""
    n, d = X_gpu.shape
    norms = cp.linalg.norm(X_gpu, axis=1)
    safe_norms = cp.where(norms > 1e-30, norms, cp.float32(1.0))
    X_unit = X_gpu / safe_norms[:, cp.newaxis]

    # RHT: sign flip + FWHT
    Y = X_unit * signs_gpu[cp.newaxis, :]
    h = 1
    while h < d:
        Y_r = Y.reshape(n, -1, 2 * h)
        a = Y_r[:, :, :h].copy()
        b = Y_r[:, :, h:].copy()
        Y_r[:, :, :h] = a + b
        Y_r[:, :, h:] = a - b
        h *= 2
    Y /= cp.sqrt(cp.float32(d))

    # Scalar quantize
    codes = cp.zeros((n, d), dtype=cp.uint8)
    for bnd in boundaries:
        codes += (Y >= bnd).astype(cp.uint8)

    return codes, norms


def rotate_query_gpu(q_gpu, signs_gpu, d):
    """Apply RHT rotation to query on GPU."""
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


def free_gpu():
    """Free all CuPy memory."""
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()


def generate_realistic_data(n, d, seed=42):
    """Generate realistic attention data, return keys and query as float32 numpy."""
    gen = RealisticAttentionGenerator(
        n=n, d=d, n_sinks=4,
        local_window=min(256, n),
        n_clusters=20,
        cluster_concentration=200.0,
        sparsity=0.05, seed=seed,
    )
    keys, values, query, metadata = gen.generate()
    return keys.astype(np.float32), query.astype(np.float32)


# =====================================================================
# Ablation 1: Vary m (coarse coordinates)
# =====================================================================

def ablation_m(n=1_000_000, d=128, b=2, k=100):
    """Sweep m from 4 to 128, measuring speedup and recall."""
    print("\n" + "=" * 72)
    print(f"ABLATION 1: Vary m (coarse coordinates), n={n:,}, d={d}, b={b}-bit")
    print("=" * 72)

    sigma = 1.0 / np.sqrt(d)
    bounds_np, levels_np = lloyd_max_gaussian(b, sigma)
    levels_gpu = cp.array(levels_np)

    rng_signs = np.random.RandomState(42)
    signs_np = rng_signs.choice([-1.0, 1.0], size=d).astype(np.float32)
    signs_gpu = cp.array(signs_np)

    # Generate data
    print("  Generating realistic data...")
    keys_np, query_np = generate_realistic_data(n, d)
    keys_gpu = cp.array(keys_np)
    query_gpu = cp.array(query_np)

    codes_gpu, norms_gpu = quantize_gpu_nbits(keys_gpu, signs_gpu, bounds_np)
    codes_T_gpu = cp.ascontiguousarray(codes_gpu.T)
    q_rot = rotate_query_gpu(query_gpu, signs_gpu, d)
    coord_order = cp.argsort(cp.abs(q_rot))[::-1].astype(cp.int32)

    del keys_gpu, query_gpu
    free_gpu()

    # Brute force baseline
    print("  Running brute force baseline...")
    bf_ms, (bf_idx, bf_scores) = gpu_time_ms(
        lambda: bench_brute_force_gemm(codes_gpu, norms_gpu, q_rot, levels_gpu, k))
    bf_set = set(cp.asnumpy(bf_idx).tolist())
    print(f"  BF (GEMM): {bf_ms:.3f} ms")

    bf_fused_ms, _ = gpu_time_ms(
        lambda: bench_brute_force_fused(codes_T_gpu, norms_gpu, q_rot, levels_gpu, k, d))
    print(f"  BF (fused): {bf_fused_ms:.3f} ms")

    m_values = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    m_results = []

    print(f"\n  {'m':>5s}  {'Time (ms)':>10s}  {'vs GEMM':>8s}  {'vs Fused':>9s}  "
          f"{'Recall':>7s}  {'Surv%':>7s}  {'Reads%':>7s}  {'BW (GB/s)':>10s}  {'BW%':>5s}")
    print("  " + "-" * 85)

    for m in m_values:
        if m >= d:
            # m == d is just brute force
            c2f_ms = bf_fused_ms
            recall = 1.0
            n_surv = n
        else:
            # Use a generous, constant keep_mult so that the survival count
            # depends on m's discrimination power, not on artificial caps.
            # Target ~20% survivors: keep_mult = 0.20 * n / k
            keep_mult = int(0.20 * n / k)

            c2f_ms, (c2f_idx, c2f_scores, n_surv) = gpu_time_ms(
                lambda _m=m, _km=keep_mult: bench_c2f_fused(
                    codes_T_gpu, norms_gpu, q_rot, levels_gpu, coord_order,
                    k, m1=_m, keep_mult=_km),
                warmup=5, runs=20,
            )
            c2f_set = set(cp.asnumpy(c2f_idx).tolist())
            recall = len(bf_set & c2f_set) / k

        speedup_gemm = bf_ms / c2f_ms
        speedup_fused = bf_fused_ms / c2f_ms
        surv_pct = n_surv / n * 100
        reads_pct = (m * n + (d - m) * n_surv) / (d * n) * 100
        # Bandwidth: round 1 reads m*n bytes (uint8), round 2 reads (d-m)*n_surv bytes
        bytes_read = m * n + (d - m) * n_surv + n * 4  # codes + norms(float32)
        eff_bw = bytes_read / (c2f_ms / 1000) / 1e9
        bw_frac = eff_bw / PEAK_BW_GBs * 100

        print(f"  {m:>5d}  {c2f_ms:>10.3f}  {speedup_gemm:>7.2f}x  {speedup_fused:>8.2f}x  "
              f"{recall:>7.2%}  {surv_pct:>6.1f}%  {reads_pct:>6.0f}%  "
              f"{eff_bw:>10.1f}  {bw_frac:>4.0f}%")

        m_results.append({
            "m": m,
            "ms": float(c2f_ms),
            "speedup_vs_gemm": float(speedup_gemm),
            "speedup_vs_fused": float(speedup_fused),
            "recall": float(recall),
            "survivors_pct": float(surv_pct),
            "reads_pct": float(reads_pct),
            "bytes_read": int(bytes_read),
            "eff_bw_GBs": float(eff_bw),
            "bw_fraction": float(bw_frac / 100),
        })

    del codes_gpu, codes_T_gpu, norms_gpu, bf_idx, bf_scores
    free_gpu()

    return {
        "bf_gemm_ms": float(bf_ms),
        "bf_fused_ms": float(bf_fused_ms),
        "n": n, "d": d, "b": b, "k": k,
        "sweep": m_results,
    }


# =====================================================================
# Ablation 2: Vary d (dimension)
# =====================================================================

def ablation_d(n=500_000, b=2, k=100, m_frac=0.125):
    """Sweep d in {64, 128, 256, 512}, measure speedup."""
    print("\n" + "=" * 72)
    print(f"ABLATION 2: Vary d (dimension), n={n:,}, b={b}-bit, m=d/8")
    print("=" * 72)

    d_values = [64, 128, 256, 512]
    d_results = []

    for d in d_values:
        m = max(4, int(d * m_frac))
        keep_mult = int(0.20 * n / k)  # ~20% survivors

        sigma = 1.0 / np.sqrt(d)
        bounds_np, levels_np = lloyd_max_gaussian(b, sigma)
        levels_gpu = cp.array(levels_np)

        rng_signs = np.random.RandomState(42)
        signs_np = rng_signs.choice([-1.0, 1.0], size=d).astype(np.float32)
        signs_gpu = cp.array(signs_np)

        # Check VRAM
        vram_need = n * d * 3 / 1e9  # codes + codes_T + misc
        free = cp.cuda.runtime.memGetInfo()[0] / 1e9
        if vram_need > free * 0.85:
            print(f"\n  d={d}: skipping (need {vram_need:.1f} GB, have {free:.1f} GB)")
            continue

        print(f"\n  d={d}, m={m}")
        keys_np, query_np = generate_realistic_data(n, d)
        keys_gpu = cp.array(keys_np)
        query_gpu = cp.array(query_np)

        codes_gpu, norms_gpu = quantize_gpu_nbits(keys_gpu, signs_gpu, bounds_np)
        codes_T_gpu = cp.ascontiguousarray(codes_gpu.T)
        q_rot = rotate_query_gpu(query_gpu, signs_gpu, d)
        coord_order = cp.argsort(cp.abs(q_rot))[::-1].astype(cp.int32)

        del keys_gpu, query_gpu
        free_gpu()

        # Brute force
        bf_ms, (bf_idx, bf_scores) = gpu_time_ms(
            lambda: bench_brute_force_gemm(codes_gpu, norms_gpu, q_rot, levels_gpu, k))
        bf_set = set(cp.asnumpy(bf_idx).tolist())

        bf_fused_ms, _ = gpu_time_ms(
            lambda: bench_brute_force_fused(codes_T_gpu, norms_gpu, q_rot, levels_gpu, k, d))

        # C2F
        c2f_ms, (c2f_idx, c2f_scores, n_surv) = gpu_time_ms(
            lambda: bench_c2f_fused(
                codes_T_gpu, norms_gpu, q_rot, levels_gpu, coord_order,
                k, m1=m, keep_mult=keep_mult))
        c2f_set = set(cp.asnumpy(c2f_idx).tolist())
        recall = len(bf_set & c2f_set) / k

        speedup_gemm = bf_ms / c2f_ms
        speedup_fused = bf_fused_ms / c2f_ms
        surv_pct = n_surv / n * 100
        bytes_read = m * n + (d - m) * n_surv + n * 4
        eff_bw = bytes_read / (c2f_ms / 1000) / 1e9

        print(f"    BF GEMM: {bf_ms:.3f} ms  |  BF fused: {bf_fused_ms:.3f} ms  |  "
              f"C2F: {c2f_ms:.3f} ms")
        print(f"    Speedup: {speedup_gemm:.2f}x (GEMM)  {speedup_fused:.2f}x (fused)  "
              f"Recall: {recall:.2%}  Survivors: {surv_pct:.1f}%")

        d_results.append({
            "d": d, "m": m,
            "bf_gemm_ms": float(bf_ms),
            "bf_fused_ms": float(bf_fused_ms),
            "c2f_ms": float(c2f_ms),
            "speedup_vs_gemm": float(speedup_gemm),
            "speedup_vs_fused": float(speedup_fused),
            "recall": float(recall),
            "survivors_pct": float(surv_pct),
            "eff_bw_GBs": float(eff_bw),
        })

        del codes_gpu, codes_T_gpu, norms_gpu, bf_idx, bf_scores
        free_gpu()

    return {"n": n, "b": b, "k": k, "m_frac": m_frac, "sweep": d_results}


# =====================================================================
# Ablation 3: Vary b (bit-width)
# =====================================================================

def ablation_b(n=500_000, d=128, k=100, m=16):
    """Sweep b in {1, 2, 3, 4}, measure speedup."""
    print("\n" + "=" * 72)
    print(f"ABLATION 3: Vary b (bit-width), n={n:,}, d={d}, m={m}")
    print("=" * 72)

    b_values = [1, 2, 3, 4]
    b_results = []

    for b in b_values:
        n_levels = 1 << b
        sigma = 1.0 / np.sqrt(d)
        bounds_np, levels_np = lloyd_max_gaussian(b, sigma)
        levels_gpu = cp.array(levels_np)

        # Verify levels fit in uint8 (max code = n_levels - 1)
        assert n_levels <= 256, f"Too many levels for uint8: {n_levels}"

        rng_signs = np.random.RandomState(42)
        signs_np = rng_signs.choice([-1.0, 1.0], size=d).astype(np.float32)
        signs_gpu = cp.array(signs_np)

        print(f"\n  b={b}, n_levels={n_levels}")
        print(f"    Levels: {levels_np}")

        keys_np, query_np = generate_realistic_data(n, d)
        keys_gpu = cp.array(keys_np)
        query_gpu = cp.array(query_np)

        codes_gpu, norms_gpu = quantize_gpu_nbits(keys_gpu, signs_gpu, bounds_np)
        codes_T_gpu = cp.ascontiguousarray(codes_gpu.T)
        q_rot = rotate_query_gpu(query_gpu, signs_gpu, d)
        coord_order = cp.argsort(cp.abs(q_rot))[::-1].astype(cp.int32)

        del keys_gpu, query_gpu
        free_gpu()

        # Brute force (GEMM uses dequant -> float32 matmul)
        bf_ms, (bf_idx, bf_scores) = gpu_time_ms(
            lambda: bench_brute_force_gemm(codes_gpu, norms_gpu, q_rot, levels_gpu, k))
        bf_set = set(cp.asnumpy(bf_idx).tolist())

        bf_fused_ms, _ = gpu_time_ms(
            lambda: bench_brute_force_fused(codes_T_gpu, norms_gpu, q_rot, levels_gpu, k, d))

        # C2F — use generous keep_mult for ~20% survivors
        keep_mult = int(0.20 * n / k)
        c2f_ms, (c2f_idx, c2f_scores, n_surv) = gpu_time_ms(
            lambda: bench_c2f_fused(
                codes_T_gpu, norms_gpu, q_rot, levels_gpu, coord_order,
                k, m1=m, keep_mult=keep_mult))
        c2f_set = set(cp.asnumpy(c2f_idx).tolist())
        recall = len(bf_set & c2f_set) / k

        speedup_gemm = bf_ms / c2f_ms
        speedup_fused = bf_fused_ms / c2f_ms
        surv_pct = n_surv / n * 100

        # Bandwidth analysis
        # For all bit widths, we still store 1 byte per code (uint8 container)
        # The actual information is b bits, but the kernel reads full bytes
        bytes_read_bf = n * d * 1 + n * 4  # uint8 codes + float32 norms
        bytes_read_c2f = m * n + (d - m) * n_surv + n * 4
        eff_bw_bf = bytes_read_bf / (bf_fused_ms / 1000) / 1e9
        eff_bw_c2f = bytes_read_c2f / (c2f_ms / 1000) / 1e9

        print(f"    BF GEMM: {bf_ms:.3f} ms  |  BF fused: {bf_fused_ms:.3f} ms  |  "
              f"C2F: {c2f_ms:.3f} ms")
        print(f"    Speedup: {speedup_gemm:.2f}x (GEMM)  {speedup_fused:.2f}x (fused)  "
              f"Recall: {recall:.2%}")

        b_results.append({
            "b": b, "n_levels": n_levels,
            "levels": levels_np.tolist(),
            "bf_gemm_ms": float(bf_ms),
            "bf_fused_ms": float(bf_fused_ms),
            "c2f_ms": float(c2f_ms),
            "speedup_vs_gemm": float(speedup_gemm),
            "speedup_vs_fused": float(speedup_fused),
            "recall": float(recall),
            "survivors_pct": float(surv_pct),
            "eff_bw_bf_GBs": float(eff_bw_bf),
            "eff_bw_c2f_GBs": float(eff_bw_c2f),
            "bw_fraction_bf": float(eff_bw_bf / PEAK_BW_GBs),
            "bw_fraction_c2f": float(eff_bw_c2f / PEAK_BW_GBs),
        })

        del codes_gpu, codes_T_gpu, norms_gpu, bf_idx, bf_scores
        free_gpu()

    return {"n": n, "d": d, "k": k, "m": m, "sweep": b_results}


# =====================================================================
# Figures
# =====================================================================

def generate_figures(m_data, d_data, b_data):
    """Generate all ablation figures."""
    print("\nGenerating ablation figures...")

    # ── Figure 1: Ablation m — dual y-axis (speedup + recall) ──
    if m_data and m_data["sweep"]:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        ms = [r["m"] for r in m_data["sweep"]]
        speedups = [r["speedup_vs_gemm"] for r in m_data["sweep"]]
        recalls = [r["recall"] for r in m_data["sweep"]]

        l1, = ax1.plot(ms, speedups, 'o-', color='#1f77b4', linewidth=2,
                       markersize=8, label='Speedup vs GEMM')
        l2, = ax2.plot(ms, recalls, 's--', color='#d62728', linewidth=2,
                       markersize=8, label='Recall@100')

        ax1.set_xlabel('$m$ (number of coarse coordinates)', fontsize=13)
        ax1.set_ylabel('Speedup over brute force', fontsize=13, color='#1f77b4')
        ax2.set_ylabel('Recall@100', fontsize=13, color='#d62728')

        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        ax2.tick_params(axis='y', labelcolor='#d62728')

        # Mark Pareto-optimal region
        ax2.axhline(y=0.95, color='#d62728', linestyle=':', alpha=0.3)
        ax1.axhline(y=1.0, color='#1f77b4', linestyle=':', alpha=0.3)

        ax1.set_title(f'Ablation: Coarse Coordinates $m$\n'
                      f'(n={m_data["n"]:,}, d={m_data["d"]}, {m_data["b"]}-bit)',
                      fontsize=13)
        lines = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, fontsize=11, loc='center right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(ms)

        fig.tight_layout()
        path = os.path.join(FIGURES_DIR, 'exp11_ablation_m.png')
        fig.savefig(path, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close()

    # ── Figure 2: Ablation d — speedup vs dimension ──
    if d_data and d_data["sweep"]:
        fig, ax = plt.subplots(figsize=(10, 6))

        ds = [r["d"] for r in d_data["sweep"]]
        su_gemm = [r["speedup_vs_gemm"] for r in d_data["sweep"]]
        su_fused = [r["speedup_vs_fused"] for r in d_data["sweep"]]
        recalls = [r["recall"] for r in d_data["sweep"]]

        ax.bar([d - 8 for d in ds], su_gemm, width=16, color='#1f77b4',
               alpha=0.8, label='vs GEMM')
        ax.bar([d + 8 for d in ds], su_fused, width=16, color='#ff7f0e',
               alpha=0.8, label='vs Fused BF')

        # Add recall annotations
        for i, d_val in enumerate(ds):
            ax.annotate(f'R={recalls[i]:.0%}', (d_val, max(su_gemm[i], su_fused[i])),
                        textcoords="offset points", xytext=(0, 8),
                        ha='center', fontsize=10, fontweight='bold')

        ax.set_xlabel('Dimension $d$', fontsize=13)
        ax.set_ylabel('Speedup', fontsize=13)
        ax.set_title(f'C2F Speedup vs Dimension (m=d/8)\n'
                     f'(n={d_data["n"]:,}, {d_data["b"]}-bit)',
                     fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(ds)
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.4)

        fig.tight_layout()
        path = os.path.join(FIGURES_DIR, 'exp11_ablation_d.png')
        fig.savefig(path, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close()

    # ── Figure 3: Bandwidth utilization ──
    if m_data and m_data["sweep"]:
        fig, ax = plt.subplots(figsize=(10, 6))

        ms = [r["m"] for r in m_data["sweep"]]
        eff_bws = [r["eff_bw_GBs"] for r in m_data["sweep"]]
        bw_fracs = [r["bw_fraction"] * 100 for r in m_data["sweep"]]

        ax.bar(ms, eff_bws, width=3, color='#2ca02c', alpha=0.8, label='Effective bandwidth')
        ax.axhline(y=PEAK_BW_GBs, color='#d62728', linestyle='--', linewidth=2,
                   label=f'RTX 3070 peak ({PEAK_BW_GBs:.0f} GB/s)')

        # Add percentage annotations
        for i, m_val in enumerate(ms):
            ax.annotate(f'{bw_fracs[i]:.0f}%', (m_val, eff_bws[i]),
                        textcoords="offset points", xytext=(0, 5),
                        ha='center', fontsize=9)

        ax.set_xlabel('$m$ (coarse coordinates)', fontsize=13)
        ax.set_ylabel('Effective bandwidth (GB/s)', fontsize=13)
        ax.set_title(f'Memory Bandwidth Utilization\n'
                     f'(n={m_data["n"]:,}, d={m_data["d"]}, {m_data["b"]}-bit)',
                     fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(ms)

        fig.tight_layout()
        path = os.path.join(FIGURES_DIR, 'exp11_bandwidth.png')
        fig.savefig(path, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close()


def main():
    print("=" * 72)
    print("Experiment 11: Ablation Studies")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[0]/1e9:.1f} GB free")
    print("=" * 72)

    all_results = {}

    # Ablation 1: Vary m
    m_data = ablation_m(n=1_000_000, d=128, b=2, k=100)
    all_results["ablation_m"] = m_data
    free_gpu()

    # Ablation 2: Vary d
    d_data = ablation_d(n=500_000, b=2, k=100, m_frac=0.125)
    all_results["ablation_d"] = d_data
    free_gpu()

    # Ablation 3: Vary b
    b_data = ablation_b(n=500_000, d=128, k=100, m=16)
    all_results["ablation_b"] = b_data
    free_gpu()

    # Save results
    with open(os.path.join(RESULTS_DIR, 'exp11_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {os.path.join(RESULTS_DIR, 'exp11_results.json')}")

    # Generate figures
    generate_figures(m_data, d_data, b_data)

    # ── Summary ──
    print(f"\n{'='*72}")
    print("ABLATION SUMMARY")
    print(f"{'='*72}")

    if m_data and m_data["sweep"]:
        print(f"\nAblation m (n={m_data['n']:,}, d={m_data['d']}):")
        print(f"  {'m':>5s}  {'Speedup':>8s}  {'Recall':>7s}  {'BW util':>8s}")
        for r in m_data["sweep"]:
            print(f"  {r['m']:>5d}  {r['speedup_vs_gemm']:>7.2f}x  "
                  f"{r['recall']:>7.2%}  {r['bw_fraction']*100:>7.0f}%")

        # Find Pareto-optimal m (highest speedup with recall >= 95%)
        good = [r for r in m_data["sweep"] if r["recall"] >= 0.95]
        if good:
            best = max(good, key=lambda r: r["speedup_vs_gemm"])
            print(f"\n  ** Pareto optimal (recall >= 95%): m={best['m']}, "
                  f"speedup={best['speedup_vs_gemm']:.2f}x, recall={best['recall']:.2%}")
        # Also report highest speedup with recall >= 90%
        good90 = [r for r in m_data["sweep"] if r["recall"] >= 0.90]
        if good90:
            best90 = max(good90, key=lambda r: r["speedup_vs_gemm"])
            print(f"  ** Pareto optimal (recall >= 90%): m={best90['m']}, "
                  f"speedup={best90['speedup_vs_gemm']:.2f}x, recall={best90['recall']:.2%}")

    if d_data and d_data["sweep"]:
        print(f"\nAblation d (n={d_data['n']:,}):")
        for r in d_data["sweep"]:
            print(f"  d={r['d']:>4d}, m={r['m']:>3d}: "
                  f"speedup={r['speedup_vs_gemm']:.2f}x, recall={r['recall']:.2%}")

    if b_data and b_data["sweep"]:
        print(f"\nAblation b (n={b_data['n']:,}, d={b_data['d']}):")
        for r in b_data["sweep"]:
            print(f"  b={r['b']}: speedup={r['speedup_vs_gemm']:.2f}x, "
                  f"recall={r['recall']:.2%}, BW={r['eff_bw_c2f_GBs']:.0f} GB/s "
                  f"({r['bw_fraction_c2f']*100:.0f}%)")

    print(f"\n{'='*72}")


if __name__ == "__main__":
    main()
