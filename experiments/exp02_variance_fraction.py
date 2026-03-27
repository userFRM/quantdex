#!/usr/bin/env python3
"""
Experiment 02: Variance Fraction V(m)
======================================
Validates the V(m) formula -- the fraction of dot-product variance captured
by the top-m query coordinates after rotation.

Tests:
  1. Empirical V(m) vs theory for d=128
  2. V(m) across dimensions (d=64, 128, 256, 512)
  3. Score correlation: full score vs partial score using top-m coords
  4. Pruning simulation: recall of true top-100 using partial scores
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
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_LIB_AVAILABLE = False
try:
    from quantdex.turbo_quant import TurboQuant
    from quantdex.metrics import variance_fraction as _lib_variance_fraction
    _LIB_AVAILABLE = True
except ImportError as exc:
    warnings.warn(
        f"Could not import quantdex library: {exc}\n"
        "Falling back to self-contained reference implementations.\n"
        "To use the real library, ensure ~/quantdex/src/ contains:\n"
        "  - turbo_quant.py  (TurboQuant class)\n"
        "  - metrics.py      (variance_fraction function)\n"
    )
    _lib_variance_fraction = None

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


def _variance_fraction_theory(m: int, d: int) -> float:
    """
    Theoretical variance fraction V(m):

    For the top-m coordinates of a rotated query vector q, the fraction of
    dot-product variance captured is approximately:

      V(m) ~ (2m/d) * (ln(d/m) + 1)   for m << d

    This comes from the order statistics of |q_i|^2 being approximately
    exponential(d) distributed (for large d on the unit sphere), so the
    sum of the top-m order statistics has a known formula.

    Clamped to [0, 1].
    """
    # The library's variance_fraction(q_rotated, m) is empirical (for a specific query),
    # not theoretical. Always use the local theoretical formula here.
    if m >= d:
        return 1.0
    if m == 0:
        return 0.0
    ratio = d / m
    v = (2.0 * m / d) * (np.log(ratio) + 1.0)
    return min(v, 1.0)


def _empirical_variance_fraction(q_rot: np.ndarray, K_rot: np.ndarray, m: int) -> float:
    """
    Compute empirical V(m): fraction of dot-product variance explained by top-m coords.

    q_rot: (d,) rotated query
    K_rot: (n, d) rotated keys
    m: number of top coordinates to use
    """
    d = q_rot.shape[0]
    # Select top-m coordinates by |q_i|
    top_m_idx = np.argsort(np.abs(q_rot))[::-1][:m]

    # Full dot products
    full_scores = K_rot @ q_rot  # (n,)
    # Partial dot products using top-m coords
    partial_scores = K_rot[:, top_m_idx] @ q_rot[top_m_idx]  # (n,)

    var_full = np.var(full_scores)
    if var_full < 1e-15:
        return 1.0
    var_partial = np.var(partial_scores)
    return float(var_partial / var_full)


# ---------------------------------------------------------------------------
# Test 1: Empirical V(m) vs theory
# ---------------------------------------------------------------------------
def test_vm_vs_theory(n: int, d: int, n_repeats: int, rng: np.random.RandomState):
    """Compare empirical V(m) against theoretical formula for d=128."""
    print(f"\n[Test 1] Empirical V(m) vs theory (d={d})")

    ms = list(range(1, d + 1))
    # Theoretical curve
    v_theory = [_variance_fraction_theory(m, d) for m in ms]

    # Empirical
    v_empirical_all = {m: [] for m in ms}
    for rep in range(n_repeats):
        R = _random_orthogonal(d, rng)
        K = _random_unit_vectors(n, d, rng)
        q = _random_unit_vectors(1, d, rng).ravel()

        K_rot = K @ R.T
        q_rot = R @ q  # same rotation

        for m in ms:
            v = _empirical_variance_fraction(q_rot, K_rot, m)
            v_empirical_all[m].append(v)

    v_mean = [np.mean(v_empirical_all[m]) for m in ms]
    v_std = [np.std(v_empirical_all[m]) for m in ms]

    # Plot
    fig, ax = plt.subplots()
    ms_arr = np.array(ms)
    ax.fill_between(ms_arr,
                     np.array(v_mean) - np.array(v_std),
                     np.array(v_mean) + np.array(v_std),
                     alpha=0.2, color='C0')
    ax.plot(ms_arr, v_mean, '-', lw=2, color='C0', label='Empirical $V(m)$')
    ax.plot(ms_arr, v_theory, '--', lw=2, color='C1',
            label=r'Theory: $\frac{2m}{d}(\ln\frac{d}{m}+1)$')
    ax.axhline(y=0.5, color='gray', ls=':', alpha=0.5)
    ax.axhline(y=0.9, color='gray', ls=':', alpha=0.5)
    # Mark where V(m) crosses 0.5 and 0.9
    for threshold in [0.5, 0.9]:
        for i, v in enumerate(v_mean):
            if v >= threshold:
                ax.axvline(x=ms[i], color='gray', ls=':', alpha=0.3)
                ax.annotate(f'm={ms[i]}', xy=(ms[i], threshold),
                            fontsize=9, ha='left', va='bottom')
                break

    ax.set_xlabel('Number of top coordinates $m$')
    ax.set_ylabel('Variance fraction $V(m)$')
    ax.set_title(f'Variance Fraction: Empirical vs Theory (d={d})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp02_vm_vs_theory.png')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'exp02_vm_vs_theory.png'}")

    # Find m for 50% and 90% variance
    m_50 = next((m for m, v in zip(ms, v_mean) if v >= 0.5), d)
    m_90 = next((m for m, v in zip(ms, v_mean) if v >= 0.9), d)
    print(f"  m for 50% variance: {m_50}")
    print(f"  m for 90% variance: {m_90}")

    return {
        'v_mean': {str(m): float(v) for m, v in zip(ms, v_mean)},
        'm_50pct': int(m_50),
        'm_90pct': int(m_90),
    }


# ---------------------------------------------------------------------------
# Test 2: V(m) across dimensions
# ---------------------------------------------------------------------------
def test_vm_across_dimensions(n: int, n_repeats: int, rng: np.random.RandomState):
    """Plot V(m) for d=64, 128, 256, 512."""
    print("\n[Test 2] V(m) across dimensions")
    dims = [64, 128, 256, 512]
    colors = ['C0', 'C1', 'C2', 'C3']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    results_by_d = {}

    for d, color in zip(dims, colors):
        ms = list(range(1, d + 1))
        # Use fractional m/d on x-axis for comparison
        fracs = [m / d for m in ms]
        v_theory = [_variance_fraction_theory(m, d) for m in ms]

        # Empirical (use fewer repeats for large dims)
        reps = max(1, n_repeats // (d // 64))
        v_emp_all = {m: [] for m in ms}
        n_use = min(n, 50000)
        for rep in range(reps):
            R = _random_orthogonal(d, rng)
            K = _random_unit_vectors(n_use, d, rng)
            q = _random_unit_vectors(1, d, rng).ravel()
            K_rot = K @ R.T
            q_rot = R @ q
            for m in ms:
                v_emp_all[m].append(_empirical_variance_fraction(q_rot, K_rot, m))

        v_mean = [np.mean(v_emp_all[m]) for m in ms]

        ax1.plot(ms, v_mean, '-', lw=1.5, color=color, label=f'd={d} (emp)')
        ax1.plot(ms, v_theory, '--', lw=1, color=color, alpha=0.6)

        ax2.plot(fracs, v_mean, '-', lw=1.5, color=color, label=f'd={d} (emp)')
        ax2.plot(fracs, v_theory, '--', lw=1, color=color, alpha=0.6)

        results_by_d[d] = {'v_mean': {str(m): float(v) for m, v in zip(ms, v_mean)}}

    ax1.set_xlabel('$m$ (number of top coordinates)')
    ax1.set_ylabel('$V(m)$')
    ax1.set_title('Variance fraction vs m')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('$m/d$ (fraction of coordinates)')
    ax2.set_ylabel('$V(m)$')
    ax2.set_title('Variance fraction vs m/d (normalized)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp02_vm_across_dims.png')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'exp02_vm_across_dims.png'}")

    return results_by_d


# ---------------------------------------------------------------------------
# Test 3: Score correlation
# ---------------------------------------------------------------------------
def test_score_correlation(n: int, d: int, n_repeats: int, rng: np.random.RandomState):
    """
    Compute correlation between full score and partial score (top-m coords)
    as a function of m. Should approach 1 rapidly.
    """
    print(f"\n[Test 3] Score correlation (d={d})")
    ms = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    ms = [m for m in ms if m <= d]

    corr_all = {m: [] for m in ms}

    for rep in range(n_repeats):
        R = _random_orthogonal(d, rng)
        K = _random_unit_vectors(n, d, rng)
        q = _random_unit_vectors(1, d, rng).ravel()
        K_rot = K @ R.T
        q_rot = R @ q

        full_scores = K_rot @ q_rot
        top_idx_full = np.argsort(np.abs(q_rot))[::-1]

        for m in ms:
            idx = top_idx_full[:m]
            partial_scores = K_rot[:, idx] @ q_rot[idx]
            corr = np.corrcoef(full_scores, partial_scores)[0, 1]
            corr_all[m].append(corr)

    corr_mean = {m: np.mean(corr_all[m]) for m in ms}
    corr_std = {m: np.std(corr_all[m]) for m in ms}

    for m in ms:
        print(f"  m={m:4d}: correlation = {corr_mean[m]:.6f} +/- {corr_std[m]:.6f}")

    # Plot
    fig, ax = plt.subplots()
    ms_arr = np.array(ms, dtype=float)
    means = np.array([corr_mean[m] for m in ms])
    stds = np.array([corr_std[m] for m in ms])
    ax.errorbar(ms_arr, means, yerr=stds, fmt='o-', capsize=4, lw=2, zorder=3)
    ax.axhline(y=1.0, color='gray', ls='--', alpha=0.5)
    ax.axhline(y=0.95, color='red', ls=':', alpha=0.5, label='0.95 threshold')
    ax.set_xlabel('Number of top coordinates $m$')
    ax.set_ylabel('Pearson correlation')
    ax.set_title(f'Score correlation: full vs partial (d={d})')
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp02_score_correlation.png')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'exp02_score_correlation.png'}")

    return {
        'correlation_mean': {str(m): float(corr_mean[m]) for m in ms},
        'correlation_std': {str(m): float(corr_std[m]) for m in ms},
    }


# ---------------------------------------------------------------------------
# Test 4: Pruning simulation
# ---------------------------------------------------------------------------
def test_pruning_simulation(n: int, d: int, n_repeats: int, rng: np.random.RandomState):
    """
    For varying m and candidate multiplier C, measure recall of the true top-100
    keys when selecting top-C*100 by partial score.
    """
    print(f"\n[Test 4] Pruning simulation (d={d})")
    k = 100  # true top-k to find
    ms = [4, 8, 16, 24, 32, 48, 64]
    ms = [m for m in ms if m <= d]
    Cs = [2, 5, 10, 20, 50]

    recall_data = {}  # (m, C) -> list of recalls

    for m in ms:
        for C in Cs:
            recall_data[(m, C)] = []

    for rep in range(n_repeats):
        R = _random_orthogonal(d, rng)
        K = _random_unit_vectors(n, d, rng)
        q = _random_unit_vectors(1, d, rng).ravel()
        K_rot = K @ R.T
        q_rot = R @ q

        full_scores = K_rot @ q_rot
        true_topk = set(np.argsort(full_scores)[-k:])

        top_idx = np.argsort(np.abs(q_rot))[::-1]

        for m in ms:
            idx = top_idx[:m]
            partial_scores = K_rot[:, idx] @ q_rot[idx]

            for C in Cs:
                n_candidates = min(C * k, n)
                candidate_set = set(np.argsort(partial_scores)[-n_candidates:])
                recall = len(true_topk & candidate_set) / k
                recall_data[(m, C)].append(recall)

    # Compute means
    recall_mean = {key: np.mean(vals) for key, vals in recall_data.items()}

    # Print table
    header = f"{'m':>4s}" + "".join(f"  C={C:3d}" for C in Cs)
    print(f"  {header}")
    for m in ms:
        row = f"{m:4d}"
        for C in Cs:
            row += f"  {recall_mean[(m, C)]:6.3f}"
        print(f"  {row}")

    # Plot: heatmap-style for each m, recall vs C
    fig, ax = plt.subplots()
    for i, m in enumerate(ms):
        recalls = [recall_mean[(m, C)] for C in Cs]
        ax.plot(Cs, recalls, 'o-', lw=1.5, label=f'm={m}')

    ax.set_xlabel('Candidate multiplier $C$')
    ax.set_ylabel(f'Recall@{k}')
    ax.set_title(f'Pruning recall by partial score (d={d}, n={n:,})')
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp02_pruning_recall.png')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'exp02_pruning_recall.png'}")

    return {
        'recall_mean': {f"m={m}_C={C}": float(recall_mean[(m, C)]) for m in ms for C in Cs},
        'ms': ms,
        'Cs': Cs,
        'k': k,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Experiment 02: Variance Fraction V(m)")
    print("=" * 70)
    if not _LIB_AVAILABLE:
        print("WARNING: Using reference implementations (library not yet available)")
    print()

    np.random.seed(42)
    rng = np.random.RandomState(42)

    N = 100_000
    D = 128
    N_REPEATS = 5

    results = {}
    t0 = time.time()

    results['test1_vm_vs_theory'] = test_vm_vs_theory(N, D, N_REPEATS, rng)
    results['test2_vm_across_dims'] = test_vm_across_dimensions(N, N_REPEATS, rng)
    results['test3_score_correlation'] = test_score_correlation(N, D, N_REPEATS, rng)
    results['test4_pruning_simulation'] = test_pruning_simulation(N, D, N_REPEATS, rng)

    elapsed = time.time() - t0
    results['total_time_seconds'] = float(elapsed)
    results['parameters'] = {'n': N, 'd': D, 'n_repeats': N_REPEATS, 'library_available': _LIB_AVAILABLE}

    summary_path = RESULTS_DIR / 'exp02_results.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
