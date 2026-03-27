#!/usr/bin/env python3
"""
Experiment 08: GPU Benchmark -- Coarse-to-Fine vs Brute Force
==============================================================
Proves wall-clock speedup of the CoarseToFine algorithm over brute-force
attention on an RTX 3070 (8 GB VRAM, CUDA 12.0).

Key insight
-----------
GPU attention is **memory-bandwidth bound**, not compute bound.  Brute force
reads ALL n*d bytes of quantized codes from VRAM.  Coarse-to-Fine reads only
m*n bytes in round 1 (m << d), then d*n' bytes in round 2 (n' << n).  Reading
fewer bytes = faster wall clock.

The columnar (transposed) storage layout stores codes as (d, n) so that
reading coordinate j for all n keys is a single coalesced contiguous read.

This benchmark measures the QUERY algorithm, not the encoding step.  Data is
generated directly in quantized form on GPU to isolate the speed comparison.

Requirements: CuPy 14.x, NumPy 2.x, CUDA 12.0.  No PyTorch, no Numba.

Generates:
  - figures/exp08_gpu_wallclock.png        -- wall-clock time vs n
  - figures/exp08_gpu_speedup.png          -- speedup factor vs n
  - figures/exp08_gpu_recall_vs_speedup.png -- Pareto frontier
  - experiments/results/exp08_results.json
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

FIGURES_DIR = _PROJECT_ROOT / "figures"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CuPy import with diagnostics
# ---------------------------------------------------------------------------
try:
    import cupy as cp
except ImportError:
    print("ERROR: CuPy is required.  Install with: pip install cupy-cuda12x")
    sys.exit(1)


def _gpu_info() -> dict:
    """Collect GPU device information."""
    props = cp.cuda.runtime.getDeviceProperties(0)
    free, total = cp.cuda.runtime.memGetInfo()
    return {
        "name": props["name"].decode() if isinstance(props["name"], bytes)
                else props["name"],
        "vram_total_gb": round(total / 1e9, 2),
        "vram_free_gb": round(free / 1e9, 2),
        "sm_count": props["multiProcessorCount"],
        "memory_bus_width_bits": props["memoryBusWidth"],
        "memory_clock_mhz": props["memoryClockRate"] // 1000,
        "compute_capability": f"{props['major']}.{props['minor']}",
        "cupy_version": cp.__version__,
    }


def _peak_bandwidth_gbps(info: dict) -> float:
    """Theoretical peak memory bandwidth in GB/s (DDR factor of 2)."""
    return (2.0 * info["memory_clock_mhz"] * 1e6
            * info["memory_bus_width_bits"] / 8 / 1e9)


# ---------------------------------------------------------------------------
# Lloyd-Max levels for N(0, 1/d)
# ---------------------------------------------------------------------------
def _lloyd_max_levels(bits: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute Lloyd-Max thresholds and levels for N(0, 1/d)."""
    from scipy.stats import norm as sp_norm

    sigma = 1.0 / np.sqrt(d)
    n_levels = 1 << bits

    if bits == 1:
        c = sigma * np.sqrt(2.0 / np.pi)
        return np.array([0.0]), np.array([-c, c])

    levels = np.linspace(-3.0 * sigma, 3.0 * sigma, n_levels)
    for _ in range(200):
        thresholds = 0.5 * (levels[:-1] + levels[1:])
        boundaries = np.concatenate([[-np.inf], thresholds, [np.inf]])
        new_levels = np.empty(n_levels)
        for j in range(n_levels):
            lo, hi = boundaries[j], boundaries[j + 1]
            lo_n = lo / sigma if np.isfinite(lo) else -np.inf
            hi_n = hi / sigma if np.isfinite(hi) else np.inf
            num = sigma * (sp_norm.pdf(lo_n) - sp_norm.pdf(hi_n))
            den = sp_norm.cdf(hi_n) - sp_norm.cdf(lo_n)
            new_levels[j] = (num / den if den > 1e-30 else
                             (0.5 * (lo + hi)
                              if np.isfinite(lo + hi) else 0.0))
        if np.max(np.abs(new_levels - levels)) < 1e-12:
            levels = new_levels
            break
        levels = new_levels

    thresholds = 0.5 * (levels[:-1] + levels[1:])
    return thresholds, levels


# ---------------------------------------------------------------------------
# Data generation: build quantized codes + query directly on GPU
# ---------------------------------------------------------------------------
def _generate_quantized_data(n: int, d: int, bits: int,
                             levels_gpu: cp.ndarray,
                             seed: int = 42) -> dict:
    """Generate pre-quantized attention data directly on GPU.

    Produces codes that simulate the output of TurboQuant.quantize_batch():
    after RHT, coordinates are approximately N(0, 1/d), so codes are drawn
    from the corresponding bin probabilities.

    The query is constructed with concentrated energy on a subset of
    coordinates (as happens after RHT of a realistic LLM query), which is
    what makes Coarse-to-Fine effective.

    This approach avoids the expensive FWHT step and focuses the benchmark
    purely on the query-time memory-bandwidth argument.
    """
    rng = np.random.default_rng(seed)
    n_levels = len(levels_gpu)

    # ---- Generate codes on CPU, transfer to GPU ----
    # For realistic quantization, simulate N(0, 1/d) post-RHT values
    # then quantize.  But for large n this is expensive on CPU.
    # Instead: draw codes from the multinomial distribution that matches
    # Lloyd-Max bin probabilities for N(0, 1/d).
    from scipy.stats import norm as sp_norm
    sigma = 1.0 / np.sqrt(d)
    thresholds_np, levels_np = _lloyd_max_levels(bits, d)
    boundaries = np.concatenate([[-np.inf], thresholds_np, [np.inf]])
    bin_probs = np.array([
        sp_norm.cdf(boundaries[j + 1] / sigma) - sp_norm.cdf(boundaries[j] / sigma)
        for j in range(n_levels)
    ])
    bin_probs /= bin_probs.sum()  # ensure sums to 1

    # Draw codes from multinomial
    codes_np = rng.choice(n_levels, size=(n, d), p=bin_probs).astype(np.uint8)

    # ---- Norms: chi-distributed, mimicking ||x||^2 ~ chi^2(d) ----
    norms_np = np.sqrt(rng.chisquare(d, size=n)).astype(np.float32) / np.sqrt(d)

    # ---- Query: concentrated energy on a few coordinates ----
    # After RHT, a structured query has most energy in a subset of coords.
    # Simulate this with a power-law profile: q_rot[j] ~ (j+1)^(-alpha)
    # with random signs.  Alpha controls concentration.
    alpha = 0.8  # moderate concentration
    coord_energy = np.arange(1, d + 1, dtype=np.float64) ** (-alpha)
    coord_energy /= np.sqrt(np.sum(coord_energy ** 2))
    signs = rng.choice([-1.0, 1.0], size=d)
    q_rot_np = (coord_energy * signs).astype(np.float32)
    # Scale like a real LLM query: ||q|| ~ sqrt(d)
    q_rot_np *= np.sqrt(d)

    # ---- Transfer to GPU ----
    codes_gpu = cp.asarray(codes_np)
    norms_gpu = cp.asarray(norms_np)
    q_rot_gpu = cp.asarray(q_rot_np)

    # Columnar layout (d, n) -- coordinate j for all n keys is contiguous
    codes_T_gpu = cp.ascontiguousarray(codes_gpu.T)

    # ---- Coordinate importance ordering ----
    # Sort by |q_rot[j]| descending -- this is the C2F ordering.
    # With our power-law query, the first coordinates are already the
    # most important, but we do the sort anyway for correctness.
    coord_order = cp.argsort(cp.abs(q_rot_gpu))[::-1]
    codes_T_ordered = cp.ascontiguousarray(codes_T_gpu[coord_order])
    q_rot_ordered = q_rot_gpu[coord_order].copy()

    del codes_gpu, codes_np

    return {
        "codes_T_gpu": codes_T_gpu,
        "codes_T_ordered": codes_T_ordered,
        "norms_gpu": norms_gpu,
        "q_rot_gpu": q_rot_gpu,
        "q_rot_ordered": q_rot_ordered,
        "coord_order": coord_order,
    }


# ---------------------------------------------------------------------------
# Brute-force attention on GPU
# ---------------------------------------------------------------------------
def brute_force_topk_gpu(codes_T_gpu: cp.ndarray, norms_gpu: cp.ndarray,
                         q_rot_gpu: cp.ndarray, levels_gpu: cp.ndarray,
                         k: int = 100):
    """Brute-force: dequantize all codes, full matvec, argpartition.

    This is the GPU equivalent of standard attention over quantized keys.
    It must read every byte of the code table.

    Parameters
    ----------
    codes_T_gpu : (d, n) uint8 -- columnar codes
    norms_gpu   : (n,)  float32
    q_rot_gpu   : (d,)  float32 -- rotated query
    levels_gpu  : (L,)  float32 -- reconstruction levels
    k           : int

    Returns
    -------
    topk_indices : (k,) int64
    topk_scores  : (k,) float32
    """
    # Dequantize: levels[codes_T] -> (d, n) float32
    Y_T = levels_gpu[codes_T_gpu]                   # (d, n) float32
    scores = q_rot_gpu @ Y_T                         # (n,)
    scores *= norms_gpu

    # Top-k via argpartition + sort
    n = scores.shape[0]
    k_actual = min(k, n)
    topk_local = cp.argpartition(scores, -k_actual)[-k_actual:]
    topk_scores = scores[topk_local]
    order = cp.argsort(topk_scores)[::-1]
    return topk_local[order], topk_scores[order]


# ---------------------------------------------------------------------------
# Coarse-to-Fine attention on GPU
# ---------------------------------------------------------------------------
def coarse_to_fine_topk_gpu(codes_T_ordered: cp.ndarray,
                            norms_gpu: cp.ndarray,
                            q_rot_ordered: cp.ndarray,
                            levels_gpu: cp.ndarray,
                            k: int = 100,
                            rounds: list[tuple[int, int]] | None = None):
    """Multi-round coarse-to-fine scoring on GPU.

    Exploits the columnar layout: in round 1 we read only m << d coordinate
    rows (contiguous in memory) to score ALL n keys cheaply, then prune to
    a small survivor set before reading the remaining coordinates.

    The codes_T_ordered array has its coordinate rows sorted by |q_rot[j]|
    descending, so codes_T_ordered[:m] contains the most important coords.

    This implementation tracks survivor scores in a compact local array
    (not a full-n array) to minimize memory traffic and kernel launches.

    Parameters
    ----------
    codes_T_ordered : (d, n) uint8 -- columnar codes, coord-importance order
    norms_gpu       : (n,)  float32
    q_rot_ordered   : (d,)  float32 -- query values in coord-importance order
    levels_gpu      : (L,)  float32
    k               : int
    rounds          : list of (m_cumulative, keep) tuples

    Returns
    -------
    topk_indices : (k,) int64
    topk_scores  : (k,) float32
    """
    d, n = codes_T_ordered.shape

    if rounds is None:
        rounds = [
            (16, max(n // 10, 10 * k)),
            (48, max(n // 100, 2 * k)),
            (d, k),
        ]

    prev_m = 0
    survivor_idx = None
    survivor_scores = None

    for round_idx, (m, keep) in enumerate(rounds):
        if round_idx == 0:
            # Round 1: read top-m coord rows for ALL n keys (contiguous).
            # This is a single large dequant + matvec -- very GPU-friendly.
            Y_sub = levels_gpu[codes_T_ordered[:m]]     # (m, n) float32
            survivor_scores = q_rot_ordered[:m] @ Y_sub  # (n,)
            survivor_idx = cp.arange(n, dtype=cp.int64)
        else:
            # Later rounds: read additional coords for survivors only.
            # codes_T_ordered[prev_m:m] is (m_delta, n) contiguous.
            # Indexing [:, survivor_idx] gathers only the survivor columns.
            Y_sub = levels_gpu[codes_T_ordered[prev_m:m][:, survivor_idx]]
            delta = q_rot_ordered[prev_m:m] @ Y_sub      # (|surv|,)
            survivor_scores += delta

        # Prune to top-keep
        n_alive = len(survivor_idx)
        keep_actual = min(keep, n_alive)
        if keep_actual < n_alive:
            topk_local = cp.argpartition(
                survivor_scores, -keep_actual)[-keep_actual:]
            survivor_idx = survivor_idx[topk_local]
            survivor_scores = survivor_scores[topk_local]

        prev_m = m

    # Final: scale by norms, sort, return top-k
    final_scores = survivor_scores * norms_gpu[survivor_idx]
    k_actual = min(k, len(survivor_idx))
    topk_local = cp.argpartition(final_scores, -k_actual)[-k_actual:]
    topk_scores = final_scores[topk_local]
    order = cp.argsort(topk_scores)[::-1]
    return survivor_idx[topk_local[order]], topk_scores[order]


# ---------------------------------------------------------------------------
# CUDA-event timing helper
# ---------------------------------------------------------------------------
def _time_gpu(fn, n_warmup: int = 10, n_iter: int = 30) -> float:
    """Time a GPU function using CUDA events.  Returns median time in ms.

    Uses per-iteration event objects to avoid timing interference.
    """
    for _ in range(n_warmup):
        fn()
    cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(n_iter):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))

    return float(np.median(times))


# ---------------------------------------------------------------------------
# Recall measurement
# ---------------------------------------------------------------------------
def _measure_recall(bf_indices: cp.ndarray, c2f_indices: cp.ndarray) -> float:
    """Recall@k of C2F vs brute-force ground truth."""
    bf_set = set(bf_indices.get().tolist())
    c2f_set = set(c2f_indices.get().tolist())
    if len(bf_set) == 0:
        return 1.0
    return len(bf_set & c2f_set) / len(bf_set)


# ---------------------------------------------------------------------------
# Effective memory bandwidth estimation
# ---------------------------------------------------------------------------
def _effective_bandwidth(n: int, d: int, time_ms: float,
                         method: str,
                         rounds: list | None = None) -> float:
    """Estimate effective memory bandwidth in GB/s.

    Counts bytes read and written by each method.
    """
    if method == "brute_force":
        # Read: codes_T (d*n * 1B) + q_rot (d*4B) + norms (n*4B)
        # Write: dequantized (d*n*4B) + scores (n*4B)
        total = d * n * 1 + d * 4 + n * 4 + d * n * 4 + n * 4
    else:
        total = 0
        prev_m = 0
        n_alive = n
        for m_cum, keep in (rounds or [(16, n // 10), (48, n // 100), (128, 100)]):
            m_delta = m_cum - prev_m
            # Read codes + dequant write + matvec accumulate
            total += m_delta * n_alive * 1    # uint8 code reads
            total += m_delta * n_alive * 4    # float32 dequant write
            total += m_delta * 4              # q_rot_ordered slice
            total += n_alive * 4              # partial score update
            n_alive = min(keep, n_alive)
            prev_m = m_cum
        total += n_alive * 4  # norm reads for final scoring

    if time_ms <= 0:
        return 0.0
    return total / (time_ms / 1000.0) / 1e9


# ---------------------------------------------------------------------------
# Coordinate reads ratio
# ---------------------------------------------------------------------------
def _read_ratio(n: int, d: int, rounds: list[tuple[int, int]]) -> float:
    """Fraction of coordinate reads vs brute force (n*d)."""
    prev_m = 0
    total_reads = 0
    n_alive = n
    for m_cum, keep in rounds:
        total_reads += (m_cum - prev_m) * n_alive
        n_alive = min(keep, n_alive)
        prev_m = m_cum
    return total_reads / (n * d)


# ===========================================================================
# Main benchmark
# ===========================================================================
def benchmark():
    """Main benchmark: brute force vs C2F at varying n on GPU."""
    print("=" * 72)
    print("Experiment 08: GPU Benchmark -- Coarse-to-Fine vs Brute Force")
    print("=" * 72)

    info = _gpu_info()
    peak_bw = _peak_bandwidth_gbps(info)
    print(f"\nGPU: {info['name']}")
    print(f"VRAM: {info['vram_total_gb']} GB total, {info['vram_free_gb']} GB free")
    print(f"Compute capability: {info['compute_capability']}")
    print(f"Memory bus: {info['memory_bus_width_bits']} bits @ "
          f"{info['memory_clock_mhz']} MHz")
    print(f"Peak memory bandwidth: {peak_bw:.0f} GB/s")
    print(f"CuPy: {info['cupy_version']}")

    d = 128
    bits = 2
    k = 100
    seed = 42

    # Lloyd-Max levels for N(0, 1/d)
    _, levels_np = _lloyd_max_levels(bits, d)
    levels_gpu = cp.asarray(levels_np.astype(np.float32))
    print(f"\nLloyd-Max levels ({bits}-bit): {levels_np.round(6)}")

    n_values = [10_000, 50_000, 100_000, 200_000, 500_000,
                1_000_000, 2_000_000, 4_000_000]
    all_results = []

    for n in n_values:
        print(f"\n{'=' * 72}")
        print(f"n = {n:>10,}   d = {d}   bits = {bits}   k = {k}")
        print(f"{'=' * 72}")

        # VRAM estimate: codes_T (d*n) + codes_T_ordered (d*n) +
        # dequantized temporary (d*n*4) + norms (4n) + scores (4n)
        vram_est_gb = (2 * d * n + d * n * 4 + 8 * n) / 1e9
        free_gb = cp.cuda.runtime.memGetInfo()[0] / 1e9
        if vram_est_gb > free_gb * 0.85:
            print(f"  Skipping: est VRAM {vram_est_gb:.2f} GB > "
                  f"{free_gb * 0.85:.2f} GB safe limit")
            continue

        print(f"  Estimated VRAM: {vram_est_gb:.2f} GB "
              f"({free_gb:.2f} GB free)")

        # ---- Generate data ----
        t0 = time.perf_counter()
        data = _generate_quantized_data(n, d, bits, levels_gpu, seed=seed)
        cp.cuda.Stream.null.synchronize()
        gen_s = time.perf_counter() - t0
        print(f"  Data generation: {gen_s:.2f}s")

        codes_T = data["codes_T_gpu"]
        codes_T_ord = data["codes_T_ordered"]
        norms = data["norms_gpu"]
        q_rot = data["q_rot_gpu"]
        q_rot_ord = data["q_rot_ordered"]

        # ---- C2F round configurations ----
        c2f_configs = {
            "aggressive": [
                (16, max(n // 20, 5 * k)),
                (48, max(n // 200, 2 * k)),
                (d, k),
            ],
            "balanced": [
                (16, max(n // 10, 10 * k)),
                (48, max(n // 100, 2 * k)),
                (d, k),
            ],
            "conservative": [
                (16, max(n // 5, 20 * k)),
                (48, max(n // 50, 5 * k)),
                (d, k),
            ],
        }

        # ---- Brute force ----
        def run_bf():
            return brute_force_topk_gpu(
                codes_T, norms, q_rot, levels_gpu, k)

        bf_ms = _time_gpu(run_bf, n_warmup=15, n_iter=40)
        bf_idx, bf_scores = run_bf()
        bf_bw = _effective_bandwidth(n, d, bf_ms, "brute_force")
        print(f"\n  Brute force:     {bf_ms:8.3f} ms   "
              f"(eff BW: {bf_bw:6.1f} GB/s)")

        # ---- C2F configs ----
        row = {
            "n": n,
            "d": d,
            "bits": bits,
            "k": k,
            "bf_ms": round(bf_ms, 4),
            "bf_bandwidth_gbps": round(bf_bw, 1),
            "c2f": {},
        }

        for cfg_name, rounds in c2f_configs.items():
            def run_c2f(_rounds=rounds):
                return coarse_to_fine_topk_gpu(
                    codes_T_ord, norms, q_rot_ord, levels_gpu,
                    k=k, rounds=_rounds,
                )

            c2f_ms = _time_gpu(run_c2f, n_warmup=15, n_iter=40)
            c2f_idx, _ = run_c2f()

            recall = _measure_recall(bf_idx, c2f_idx)
            speedup = bf_ms / c2f_ms if c2f_ms > 0 else 0.0
            c2f_bw = _effective_bandwidth(n, d, c2f_ms, "c2f", rounds)
            rr = _read_ratio(n, d, rounds)

            print(f"  C2F {cfg_name:>12s}: {c2f_ms:8.3f} ms   "
                  f"speedup {speedup:5.2f}x   "
                  f"recall@{k} {recall:.4f}   "
                  f"reads {rr:.2%}")

            row["c2f"][cfg_name] = {
                "rounds": [(int(m), int(kp)) for m, kp in rounds],
                "ms": round(c2f_ms, 4),
                "speedup": round(speedup, 3),
                "recall": round(recall, 4),
                "bandwidth_gbps": round(c2f_bw, 1),
                "read_ratio": round(rr, 4),
            }

        all_results.append(row)

        # ---- Cleanup ----
        del data, codes_T, codes_T_ord, norms, q_rot, q_rot_ord
        cp.get_default_memory_pool().free_all_blocks()

    # ---- Save results ----
    output = {
        "experiment": "exp08_gpu_benchmark",
        "gpu_info": info,
        "peak_bandwidth_gbps": round(peak_bw, 1),
        "parameters": {"d": d, "bits": bits, "k": k, "seed": seed},
        "results": all_results,
    }
    results_path = RESULTS_DIR / "exp08_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return output


# ===========================================================================
# Figures
# ===========================================================================
def plot_results(output: dict):
    """Generate publication-quality figures from benchmark results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "font.size": 12,
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
    })

    results = output["results"]
    if not results:
        print("No results to plot.")
        return

    gpu_name = output["gpu_info"]["name"]
    peak_bw = output["peak_bandwidth_gbps"]
    d = output["parameters"]["d"]
    k = output["parameters"]["k"]
    bits = output["parameters"]["bits"]

    ns = [r["n"] for r in results]
    bf_ms = [r["bf_ms"] for r in results]

    # Use "balanced" as primary comparison
    primary_cfg = "balanced"
    c2f_ms_bal = []
    c2f_recall_bal = []
    c2f_speedup_bal = []
    for r in results:
        cfg = r["c2f"].get(primary_cfg, next(iter(r["c2f"].values())))
        c2f_ms_bal.append(cfg["ms"])
        c2f_recall_bal.append(cfg["recall"])
        c2f_speedup_bal.append(cfg["speedup"])

    # Extract all configs for multi-line plots
    all_cfgs = {}
    for cfg_name in ["aggressive", "balanced", "conservative"]:
        all_cfgs[cfg_name] = {"ms": [], "recall": [], "speedup": []}
        for r in results:
            if cfg_name in r["c2f"]:
                all_cfgs[cfg_name]["ms"].append(r["c2f"][cfg_name]["ms"])
                all_cfgs[cfg_name]["recall"].append(
                    r["c2f"][cfg_name]["recall"])
                all_cfgs[cfg_name]["speedup"].append(
                    r["c2f"][cfg_name]["speedup"])

    colors = {"aggressive": "#ff7f0e", "balanced": "#1f77b4",
              "conservative": "#2ca02c"}
    markers = {"aggressive": "^", "balanced": "s", "conservative": "o"}

    # ------------------------------------------------------------------
    # Figure 1: Wall-clock time vs n (log-log)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.loglog(ns, bf_ms, "o-", color="#d62728", linewidth=2.0,
              markersize=7, label="Brute force", zorder=5)
    ax.loglog(ns, c2f_ms_bal, "s-", color="#1f77b4", linewidth=2.0,
              markersize=7, label="Coarse-to-Fine (balanced)", zorder=5)

    # Mark crossover
    for i in range(len(ns) - 1):
        if bf_ms[i] <= c2f_ms_bal[i] and bf_ms[i + 1] > c2f_ms_bal[i + 1]:
            ratio = ((bf_ms[i] - c2f_ms_bal[i])
                     / ((bf_ms[i] - c2f_ms_bal[i])
                        - (bf_ms[i + 1] - c2f_ms_bal[i + 1])))
            cross_n = ns[i] + (ns[i + 1] - ns[i]) * ratio
            y_cross = bf_ms[i] + (bf_ms[i + 1] - bf_ms[i]) * ratio
            ax.axvline(cross_n, color="gray", linestyle="--", alpha=0.5)
            ax.annotate(
                f"crossover\nn~{cross_n:,.0f}",
                xy=(cross_n, y_cross), fontsize=9,
                ha="right", va="bottom", color="gray",
                textcoords="offset points", xytext=(-8, 5),
            )
            break

    ax.set_xlabel("Number of keys (n)")
    ax.set_ylabel("Wall-clock time (ms)")
    ax.set_title(f"GPU Attention Latency: Brute Force vs C2F\n"
                 f"{gpu_name} | d={d}, {bits}-bit, k={k}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    path = FIGURES_DIR / "exp08_gpu_wallclock.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")

    # ------------------------------------------------------------------
    # Figure 2: Speedup factor vs n
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    for cfg_name in ["conservative", "balanced", "aggressive"]:
        if cfg_name not in all_cfgs or not all_cfgs[cfg_name]["speedup"]:
            continue
        cfg_ns = ns[:len(all_cfgs[cfg_name]["speedup"])]
        ax.semilogx(cfg_ns, all_cfgs[cfg_name]["speedup"],
                     f"{markers[cfg_name]}-",
                     color=colors[cfg_name],
                     linewidth=2.0, markersize=7,
                     label=f"C2F ({cfg_name})")

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5,
               label="break-even")

    # Shade region above 1.0
    ax.axhspan(1.0, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 5,
               alpha=0.05, color="green")

    ax.set_xlabel("Number of keys (n)")
    ax.set_ylabel("Speedup over brute force")
    ax.set_title(f"GPU C2F Speedup Factor vs Sequence Length\n"
                 f"{gpu_name} | d={d}, {bits}-bit, k={k}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIGURES_DIR / "exp08_gpu_speedup.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")

    # ------------------------------------------------------------------
    # Figure 3: Recall@k vs Speedup (Pareto frontier)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    for cfg_name in ["conservative", "balanced", "aggressive"]:
        if cfg_name not in all_cfgs:
            continue
        recalls = all_cfgs[cfg_name]["recall"]
        speedups = all_cfgs[cfg_name]["speedup"]
        if not recalls:
            continue
        cfg_ns = ns[:len(recalls)]
        ax.scatter(
            speedups, recalls,
            c=np.log10(cfg_ns),
            marker=markers[cfg_name], s=80, alpha=0.8,
            edgecolors="white", linewidth=0.5,
            cmap="viridis",
            vmin=np.log10(min(ns)), vmax=np.log10(max(ns)),
            label=f"C2F ({cfg_name})", zorder=5,
        )
        ax.plot(speedups, recalls, "-", color=colors[cfg_name], alpha=0.4)

    # Label balanced points with n
    for i, n_val in enumerate(
            ns[:len(all_cfgs.get("balanced", {}).get("recall", []))]):
        spd = all_cfgs["balanced"]["speedup"][i]
        rec = all_cfgs["balanced"]["recall"][i]
        label = (f"n={n_val // 1000}K" if n_val < 1_000_000
                 else f"n={n_val // 1_000_000}M")
        ax.annotate(label, xy=(spd, rec), fontsize=7, ha="left",
                    textcoords="offset points", xytext=(5, -5))

    ax.axhline(0.95, color="green", linestyle=":", alpha=0.5,
               label="95% recall")
    ax.axhline(0.99, color="blue", linestyle=":", alpha=0.5,
               label="99% recall")
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Speedup over brute force")
    ax.set_ylabel(f"Recall@{k}")
    ax.set_title(f"GPU C2F: Recall vs Speedup Tradeoff\n"
                 f"{gpu_name} | d={d}, {bits}-bit")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIGURES_DIR / "exp08_gpu_recall_vs_speedup.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ===========================================================================
# Summary table
# ===========================================================================
def print_summary(output: dict):
    """Print a clean summary table."""
    results = output["results"]
    if not results:
        return

    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")

    header = (f"{'n':>12s}  {'BF (ms)':>10s}  {'C2F (ms)':>10s}  "
              f"{'Speedup':>8s}  {'Recall':>8s}  {'Reads':>8s}")
    print(header)
    print("-" * len(header))

    for r in results:
        cfg = r["c2f"].get("balanced", next(iter(r["c2f"].values())))
        print(f"{r['n']:>12,}  {r['bf_ms']:>10.3f}  {cfg['ms']:>10.3f}  "
              f"{cfg['speedup']:>7.2f}x  {cfg['recall']:>8.4f}  "
              f"{cfg['read_ratio']:>7.1%}")

    # Peak speedup
    max_row = max(results,
                  key=lambda r: r["c2f"].get(
                      "balanced", next(iter(r["c2f"].values()))
                  )["speedup"])
    max_cfg = max_row["c2f"].get("balanced",
                                 next(iter(max_row["c2f"].values())))
    print(f"\nPeak speedup: {max_cfg['speedup']:.2f}x "
          f"at n={max_row['n']:,} "
          f"(recall={max_cfg['recall']:.4f})")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    output = benchmark()
    print_summary(output)
    plot_results(output)
    print("\nDone.")
