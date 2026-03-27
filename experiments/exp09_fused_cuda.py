"""Experiment 09: Fused CUDA kernel — eliminate Python loop overhead.

The CuPy benchmark (exp08) showed only 1.8x speedup because each coordinate
was a separate kernel launch. This experiment fuses the coordinate loop into
a SINGLE CUDA kernel, eliminating all Python overhead.

Expected: 3-5x speedup from bandwidth reduction alone.
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cupy as cp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 5), 'figure.dpi': 150})

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Fused CUDA Kernels ──────────────────────────────────────────────

# Kernel 1: Compute partial dot product over m coordinates in ONE launch.
# Each thread handles one key. Reads m bytes from columnar layout.
# NOTE: The index `j * n + i` uses int32 arithmetic. This is safe for
# d*n < 2^31 (~2.1B), i.e., d=128 and n up to ~16M. For larger n,
# the addressing would need to be cast to int64.
_partial_score_kernel = cp.RawKernel(r'''
extern "C" __global__ void partial_score(
    const unsigned char* __restrict__ codes_T,   // (d, n) column-major
    const float* __restrict__ q_rot,             // (d,)
    const float* __restrict__ levels,            // (n_levels,)
    const int* __restrict__ coord_order,         // (m,) sorted coords
    const float* __restrict__ norms,             // (n,)
    float* __restrict__ scores,                  // (n,) output
    const int n,
    const int m
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float s = 0.0f;
    for (int jj = 0; jj < m; jj++) {
        int j = coord_order[jj];
        unsigned char code = codes_T[j * n + i];
        s += levels[code] * q_rot[j];
    }
    scores[i] = s * norms[i];
}
''', 'partial_score')

# Kernel 2: Compute FULL dot product (brute force) in one launch.
_full_score_kernel = cp.RawKernel(r'''
extern "C" __global__ void full_score(
    const unsigned char* __restrict__ codes_T,   // (d, n) column-major
    const float* __restrict__ q_rot,             // (d,)
    const float* __restrict__ levels,            // (n_levels,)
    const float* __restrict__ norms,             // (n,)
    float* __restrict__ scores,                  // (n,) output
    const int n,
    const int d
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float s = 0.0f;
    for (int j = 0; j < d; j++) {
        unsigned char code = codes_T[j * n + i];
        s += levels[code] * q_rot[j];
    }
    scores[i] = s * norms[i];
}
''', 'full_score')

# Kernel 3: Score only survivors (indexed access)
_survivor_score_kernel = cp.RawKernel(r'''
extern "C" __global__ void survivor_score(
    const unsigned char* __restrict__ codes_T,
    const float* __restrict__ q_rot,
    const float* __restrict__ levels,
    const float* __restrict__ norms,
    const int* __restrict__ survivor_idx,
    const int* __restrict__ coord_order,
    const float* __restrict__ partial_in,
    float* __restrict__ scores_out,
    const int n_survivors,
    const int n,
    const int m_start,
    const int m_end
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_survivors) return;

    int i = survivor_idx[tid];
    float s = partial_in[tid];
    for (int jj = m_start; jj < m_end; jj++) {
        int j = coord_order[jj];
        unsigned char code = codes_T[j * n + i];
        s += levels[code] * q_rot[j];
    }
    scores_out[tid] = s * norms[i];
}
''', 'survivor_score')


# ── Lloyd-Max levels ────────────────────────────────────────────────
def lloyd_max_levels_2bit(d):
    sigma = 1.0 / np.sqrt(d)
    return np.array([-1.510*sigma, -0.4528*sigma, 0.4528*sigma, 1.510*sigma],
                    dtype=np.float32)

def lloyd_max_boundaries_2bit(d):
    sigma = 1.0 / np.sqrt(d)
    return np.array([-0.9816*sigma, 0.0, 0.9816*sigma], dtype=np.float32)


# ── Quantize on GPU ─────────────────────────────────────────────────
def quantize_gpu(X_gpu, signs_gpu, boundaries):
    n, d = X_gpu.shape
    norms = cp.linalg.norm(X_gpu, axis=1)
    X_unit = X_gpu / norms[:, cp.newaxis]

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
    for i, b in enumerate(boundaries):
        codes += (Y >= b).astype(cp.uint8)

    return codes, norms


# ── Timing helper ───────────────────────────────────────────────────
def gpu_time_ms(fn, warmup=5, runs=20):
    for _ in range(warmup):
        fn()
    cp.cuda.Stream.null.synchronize()
    times = []
    for _ in range(runs):
        s = cp.cuda.Event(); e = cp.cuda.Event()
        s.record()
        result = fn()
        e.record(); e.synchronize()
        times.append(cp.cuda.get_elapsed_time(s, e))
    return np.median(times), result


# ── Benchmark functions ─────────────────────────────────────────────

def bench_brute_force_fused(codes_T, norms, q_rot, levels, k, d):
    """Brute force with fused CUDA kernel (not cuBLAS GEMM)."""
    n = codes_T.shape[1]
    scores = cp.empty(n, dtype=cp.float32)
    block = 256
    grid = (n + block - 1) // block
    _full_score_kernel((grid,), (block,),
                       (codes_T, q_rot, levels, norms, scores, np.int32(n), np.int32(d)))
    actual_k = min(k, n)
    if actual_k == 0:
        return cp.array([], dtype=cp.int64), cp.array([], dtype=cp.float32)
    topk_idx = cp.argpartition(scores, -actual_k)[-actual_k:]
    topk_scores = scores[topk_idx]
    order = cp.argsort(topk_scores)[::-1]
    return topk_idx[order], topk_scores[order]


def bench_brute_force_gemm(codes, norms, q_rot, levels, k):
    """Brute force with dequant + cuBLAS GEMM (the exp08 baseline)."""
    Y = levels[codes]  # (n, d) float32
    scores = Y @ q_rot * norms
    topk_idx = cp.argpartition(scores, -k)[-k:]
    topk_scores = scores[topk_idx]
    order = cp.argsort(topk_scores)[::-1]
    return topk_idx[order], topk_scores[order]


def bench_c2f_fused(codes_T, norms, q_rot, levels, coord_order, k, m1, keep_mult):
    """Coarse-to-fine with fused CUDA kernels. 2 kernel launches total."""
    n = codes_T.shape[1]
    d = codes_T.shape[0]

    # ── Round 1: ONE kernel launch, m1 coordinates, all n keys ──
    scores_r1 = cp.empty(n, dtype=cp.float32)
    block = 256
    grid = (n + block - 1) // block
    _partial_score_kernel((grid,), (block,),
                          (codes_T, q_rot, levels, coord_order,
                           norms, scores_r1, np.int32(n), np.int32(m1)))

    # Keep top candidates
    keep = min(keep_mult * k, n)
    threshold = cp.partition(scores_r1, -keep)[-keep]
    survivor_mask = scores_r1 >= threshold
    survivor_idx = cp.where(survivor_mask)[0].astype(cp.int32)
    n_surv = int(survivor_idx.shape[0])

    if n_surv == 0:
        return cp.array([], dtype=cp.int64), cp.array([], dtype=cp.float32), 0

    # ── Round 2: ONE kernel launch, remaining coordinates, survivors only ──
    partial_in = scores_r1[survivor_idx] / norms[survivor_idx]  # un-scale for re-accumulation
    scores_r2 = cp.empty(n_surv, dtype=cp.float32)
    grid2 = (n_surv + block - 1) // block
    _survivor_score_kernel((grid2,), (block,),
                           (codes_T, q_rot, levels, norms, survivor_idx, coord_order,
                            partial_in, scores_r2,
                            np.int32(n_surv), np.int32(n), np.int32(m1), np.int32(d)))

    # Top-k from survivors
    actual_k = min(k, n_surv)
    topk_local = cp.argpartition(scores_r2, -actual_k)[-actual_k:]
    topk_scores = scores_r2[topk_local]
    order = cp.argsort(topk_scores)[::-1]
    topk_idx = survivor_idx[topk_local[order]]
    return topk_idx, topk_scores[order], n_surv


# ── Main ────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Experiment 09: Fused CUDA Kernels — Eliminating Python Overhead")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[0]/1e9:.1f} GB free")
    print("=" * 70)

    d, b, k = 128, 2, 100
    levels_np = lloyd_max_levels_2bit(d)
    bounds_np = lloyd_max_boundaries_2bit(d)
    levels_gpu = cp.array(levels_np)
    rng = np.random.RandomState(42)
    signs_np = rng.choice([-1.0, 1.0], size=d).astype(np.float32)
    signs_gpu = cp.array(signs_np)

    # Pre-compute coordinate order for a reference query
    q_np = rng.randn(d).astype(np.float32)
    q_np /= np.linalg.norm(q_np)
    q_gpu = cp.array(q_np)

    # Rotate query
    q_rot = (q_gpu * signs_gpu).reshape(1, -1)
    h = 1
    while h < d:
        r = q_rot.reshape(1, -1, 2*h)
        a = r[:,:,:h].copy(); b2 = r[:,:,h:].copy()
        r[:,:,:h] = a + b2; r[:,:,h:] = a - b2
        h *= 2
    q_rot = (q_rot / cp.sqrt(cp.float32(d))).ravel()

    coord_order = cp.argsort(cp.abs(q_rot))[::-1].astype(cp.int32)

    results = {}
    n_values = [10_000, 50_000, 100_000, 500_000, 1_000_000, 2_000_000, 4_000_000]

    c2f_configs = [
        ("C2F m=8",  8,  200),
        ("C2F m=16", 16, 100),
        ("C2F m=32", 32, 50),
    ]

    for n in n_values:
        vram_need = n * d * 5 / 1e9  # codes + codes_T + norms + scores
        free = cp.cuda.runtime.memGetInfo()[0] / 1e9
        if vram_need > free * 0.9:
            print(f"\nn={n:>9,}: skipping (need {vram_need:.1f} GB, have {free:.1f} GB)")
            continue

        print(f"\n{'─'*70}")
        print(f"n = {n:,}   |   codes: {n*d/1e6:.0f} MB   |   codes_T: {n*d/1e6:.0f} MB")
        print(f"{'─'*70}")

        # Generate random unit vectors and quantize
        X_gpu = cp.random.randn(n, d, dtype=cp.float32)
        X_gpu /= cp.linalg.norm(X_gpu, axis=1, keepdims=True)
        codes_gpu, norms_gpu = quantize_gpu(X_gpu, signs_gpu, bounds_np)
        codes_T_gpu = cp.ascontiguousarray(codes_gpu.T)  # (d, n) contiguous
        del X_gpu
        cp.get_default_memory_pool().free_all_blocks()

        # ── Brute force GEMM (exp08 baseline) ──
        bf_gemm_ms, (bf_idx, bf_scores) = gpu_time_ms(
            lambda: bench_brute_force_gemm(codes_gpu, norms_gpu, q_rot, levels_gpu, k))
        bf_bw = n * d * 4 / (bf_gemm_ms / 1000) / 1e9  # dequant is float32
        print(f"  BF (GEMM):          {bf_gemm_ms:8.3f} ms   ({bf_bw:.0f} GB/s eff)")

        # ── Brute force FUSED (same algo, our kernel) ──
        bf_fused_ms, (bf_fused_idx, bf_fused_scores) = gpu_time_ms(
            lambda: bench_brute_force_fused(codes_T_gpu, norms_gpu, q_rot, levels_gpu, k, d))
        bf_fused_bw = n * d * 1 / (bf_fused_ms / 1000) / 1e9  # reads uint8
        print(f"  BF (fused uint8):   {bf_fused_ms:8.3f} ms   ({bf_fused_bw:.0f} GB/s eff)")

        # ── Validate fused BF kernel output against GEMM BF ──
        bf_gemm_set = set(cp.asnumpy(bf_idx).tolist())
        bf_fused_set = set(cp.asnumpy(bf_fused_idx).tolist())
        fused_vs_gemm_recall = len(bf_gemm_set & bf_fused_set) / k if k > 0 else 1.0
        print(f"  Fused BF recall vs GEMM BF: {fused_vs_gemm_recall:.2%}")

        entry = {
            "bf_gemm_ms": float(bf_gemm_ms),
            "bf_fused_ms": float(bf_fused_ms),
            "fused_vs_gemm_recall": float(fused_vs_gemm_recall),
        }

        # ── C2F configs ──
        bf_set = set(cp.asnumpy(bf_idx).tolist())
        for name, m1, keep_mult in c2f_configs:
            c2f_ms, (c2f_idx, c2f_scores, n_surv) = gpu_time_ms(
                lambda m1=m1, km=keep_mult: bench_c2f_fused(
                    codes_T_gpu, norms_gpu, q_rot, levels_gpu, coord_order,
                    k, m1=m1, keep_mult=km))

            c2f_set = set(cp.asnumpy(c2f_idx).tolist())
            recall = len(bf_set & c2f_set) / k if k > 0 else 0
            speedup_gemm = bf_gemm_ms / c2f_ms
            speedup_fused = bf_fused_ms / c2f_ms
            surv_pct = n_surv / n * 100
            reads_pct = (m1 * n + (d - m1) * n_surv) / (d * n) * 100

            print(f"  {name:20s}: {c2f_ms:8.3f} ms   "
                  f"vs_GEMM={speedup_gemm:5.2f}x  vs_fused={speedup_fused:5.2f}x  "
                  f"recall={recall:.2f}  survivors={surv_pct:.1f}%  reads={reads_pct:.0f}%")

            entry[name] = {
                "ms": float(c2f_ms),
                "speedup_vs_gemm": float(speedup_gemm),
                "speedup_vs_fused": float(speedup_fused),
                "recall": float(recall),
                "survivors_pct": float(surv_pct),
                "reads_pct": float(reads_pct),
            }

        results[str(n)] = entry
        del codes_gpu, codes_T_gpu, norms_gpu
        cp.get_default_memory_pool().free_all_blocks()

    # ── Figures ──────────────────────────────────────────────────────
    print("\nGenerating figures...")
    ns = sorted([int(k) for k in results.keys()])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Wall-clock time
    ax = axes[0]
    ax.plot(ns, [results[str(n)]["bf_gemm_ms"] for n in ns],
            'o-', label='Brute force (cuBLAS GEMM)', linewidth=2, markersize=8, color='#d62728')
    ax.plot(ns, [results[str(n)]["bf_fused_ms"] for n in ns],
            's-', label='Brute force (fused uint8)', linewidth=2, markersize=7, color='#ff7f0e')
    for name, m1, km in c2f_configs:
        times = [results[str(n)].get(name, {}).get("ms", np.nan) for n in ns]
        ax.plot(ns, times, '^--', label=f'{name}', linewidth=1.5, markersize=6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Number of keys $n$')
    ax.set_ylabel('Wall-clock time (ms)')
    ax.set_title('GPU Attention Latency\n(RTX 3070, d=128, 2-bit)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: Speedup
    ax = axes[1]
    for name, m1, km in c2f_configs:
        su = [results[str(n)].get(name, {}).get("speedup_vs_gemm", np.nan) for n in ns]
        ax.plot(ns, su, 'o-', label=f'{name} vs GEMM', linewidth=2, markersize=7)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Number of keys $n$')
    ax.set_ylabel('Speedup over brute force')
    ax.set_title('Fused C2F Speedup\n(single kernel launch per round)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'exp09_fused_cuda.png')
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close()

    with open(os.path.join(RESULTS_DIR, 'exp09_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Fused CUDA vs Python-loop CuPy")
    print(f"{'='*70}")
    print(f"{'n':>12s}  {'BF GEMM':>10s}  {'BF fused':>10s}  {'C2F m=16':>10s}  {'Speedup':>8s}  {'Recall':>7s}")
    print("-" * 70)
    for n in ns:
        r = results[str(n)]
        c = r.get("C2F m=16", {})
        print(f"{n:>12,}  {r['bf_gemm_ms']:>10.3f}  {r['bf_fused_ms']:>10.3f}  "
              f"{c.get('ms', 0):>10.3f}  {c.get('speedup_vs_gemm', 0):>7.2f}x  "
              f"{c.get('recall', 0):>7.2f}")


if __name__ == "__main__":
    main()
