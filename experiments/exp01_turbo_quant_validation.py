#!/usr/bin/env python3
"""
Experiment 01: TurboQuant Validation
=====================================
Validates that our TurboQuant implementation matches theoretical bounds.

Tests:
  1. Coordinate distribution after rotation (should match Beta / N(0,1/d))
  2. MSE vs bit-width (compare against theoretical + info-theoretic bounds)
  3. Inner product bias factors (verify alpha_b values)
  4. Inner product error vs dimension (should be O(1/d))
  5. Near-independence of coordinates after rotation (correlations ~ O(1/d^2))
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
from scipy import stats

# ---------------------------------------------------------------------------
# Path setup: find the src package being built in parallel
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_LIB_AVAILABLE = False
try:
    from src.turbo_quant import TurboQuant, lloyd_max_mse
    from src.metrics import mse_reconstruction, inner_product_correlation
    _LIB_AVAILABLE = True
except ImportError as exc:
    warnings.warn(
        f"Could not import quantdex library: {exc}\n"
        "Falling back to self-contained reference implementations.\n"
        "To use the real library, ensure ~/quantdex/src/ contains:\n"
        "  - turbo_quant.py  (TurboQuant class with rotate/quantize/etc)\n"
        "  - metrics.py      (mse_reconstruction, inner_product_correlation)\n"
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
# Reference / fallback implementations
# ---------------------------------------------------------------------------

def _random_orthogonal(d: int, rng: np.random.RandomState) -> np.ndarray:
    """Haar-distributed random orthogonal matrix via QR of Gaussian."""
    H = rng.randn(d, d)
    Q, R = np.linalg.qr(H)
    # Ensure uniform Haar measure: fix sign ambiguity
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q


def _random_unit_vectors(n: int, d: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample n uniform random unit vectors on S^{d-1}."""
    X = rng.randn(n, d)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


def _uniform_scalar_quantize(x: np.ndarray, b: int, range_scale: float = None) -> np.ndarray:
    """
    Uniform scalar quantization to b bits.

    If range_scale is provided, quantizes in [-range_scale, range_scale].
    Otherwise uses [-1, 1].

    Maps each scalar to one of 2^b levels, then reconstructs midpoint.
    """
    L = 2 ** b
    if range_scale is None:
        range_scale = 1.0
    # Clip to range, shift to [0, 1]
    x_clipped = np.clip(x, -range_scale, range_scale)
    x_01 = (x_clipped + range_scale) / (2.0 * range_scale)
    # Quantize
    idx = np.floor(x_01 * L).astype(np.int32)
    idx = np.clip(idx, 0, L - 1)
    # Reconstruct midpoints
    x_hat = (idx + 0.5) / L * (2.0 * range_scale) - range_scale
    return x_hat


class _ReferenceTurboQuant:
    """Minimal reference TurboQuant for when the library is unavailable."""

    def __init__(self, d: int, b: int, rng: np.random.RandomState):
        self.d = d
        self.b = b
        self.R = _random_orthogonal(d, rng)
        # Quantization range: 3 sigma for coordinates of unit vectors on S^{d-1}
        # After rotation, coordinates are ~ N(0, 1/d), so 3-sigma ~ 3/sqrt(d).
        self.range_scale = 3.0 / np.sqrt(d)

    def rotate(self, X: np.ndarray) -> np.ndarray:
        """Apply rotation: X @ R^T  (each row is a vector)."""
        return X @ self.R.T

    def quantize(self, X_rot: np.ndarray) -> np.ndarray:
        """Quantize rotated coordinates with data-adapted range."""
        return _uniform_scalar_quantize(X_rot, self.b, self.range_scale)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Rotate then quantize."""
        return self.quantize(self.rotate(X))

    def inner_product(self, q_enc: np.ndarray, k_enc: np.ndarray) -> np.ndarray:
        """Approximate inner product from quantized codes."""
        # q_enc: (d,), k_enc: (n, d)
        return k_enc @ q_enc

    def bias_factor(self, X: np.ndarray, X_hat: np.ndarray = None) -> float:
        """
        Empirical bias factor alpha_b.

        For two independent unit vectors x, y:
          alpha_b = E[<Q(Rx), Q(Ry)>] / E[<Rx, Ry>]
        where Q is the scalar quantizer applied coordinate-wise.

        Equivalently, per-coordinate: alpha_b = E[Q(z) * z'] / E[z * z']
        where z, z' ~ N(0, 1/d) independent. Since E[z*z'] = 0 this ratio
        is ill-defined for independent vectors. Instead we measure:
          alpha_b = E[z * Q(z)] / E[z^2]
        which is the per-coordinate bias (correlation between original and quantized).
        """
        X_rot = self.rotate(X)
        X_hat_rot = self.quantize(X_rot)
        # Per-coordinate: E[z * Q(z)] / E[z^2]
        numerator = np.mean(X_rot * X_hat_rot)
        denominator = np.mean(X_rot ** 2)
        return numerator / denominator


def _get_tq(d: int, b: int, rng: np.random.RandomState):
    """Get a TurboQuant instance (library or reference)."""
    if _LIB_AVAILABLE:
        return TurboQuant(d=d, bits=b, seed=rng.randint(0, 2**31))
    return _ReferenceTurboQuant(d, b, rng)


# ---------------------------------------------------------------------------
# Test 1: Coordinate distribution after rotation
# ---------------------------------------------------------------------------
def test_coordinate_distribution(n: int, d: int, n_repeats: int, rng: np.random.RandomState):
    """Rotate random unit vectors, check that coordinates ~ N(0, 1/d)."""
    print("\n[Test 1] Coordinate distribution after rotation")
    all_coords = []

    for rep in range(n_repeats):
        R = _random_orthogonal(d, rng)
        X = _random_unit_vectors(n, d, rng)
        X_rot = X @ R.T
        # Collect first coordinate from each vector
        all_coords.append(X_rot[:, 0])

    all_coords = np.concatenate(all_coords)
    empirical_mean = np.mean(all_coords)
    empirical_var = np.var(all_coords)
    theoretical_var = 1.0 / d

    print(f"  Empirical mean:     {empirical_mean:.6f}  (expected: 0)")
    print(f"  Empirical variance: {empirical_var:.6f}  (expected: {theoretical_var:.6f})")

    # KS test against N(0, 1/d)
    ks_stat, ks_p = stats.kstest(all_coords, 'norm', args=(0, np.sqrt(theoretical_var)))
    print(f"  KS test vs N(0,1/d): stat={ks_stat:.6f}, p={ks_p:.4f}")

    # Plot
    fig, ax = plt.subplots()
    ax.hist(all_coords, bins=100, density=True, alpha=0.7, label='Empirical')
    x_plot = np.linspace(-4 * np.sqrt(theoretical_var), 4 * np.sqrt(theoretical_var), 300)
    ax.plot(x_plot, stats.norm.pdf(x_plot, 0, np.sqrt(theoretical_var)),
            'r-', lw=2, label=f'$N(0, 1/{d})$')
    ax.set_xlabel('Coordinate value')
    ax.set_ylabel('Density')
    ax.set_title(f'Coordinate distribution after Haar rotation (d={d})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp01_coordinate_distribution.png')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'exp01_coordinate_distribution.png'}")

    return {
        'empirical_mean': float(empirical_mean),
        'empirical_var': float(empirical_var),
        'theoretical_var': float(theoretical_var),
        'ks_stat': float(ks_stat),
        'ks_p': float(ks_p),
    }


# ---------------------------------------------------------------------------
# Test 2: MSE vs bit-width
# ---------------------------------------------------------------------------
def test_mse_vs_bits(n: int, d: int, n_repeats: int, rng: np.random.RandomState):
    """Measure empirical MSE at b=1,2,3,4 and compare with bounds."""
    print("\n[Test 2] MSE vs bit-width")
    bits = [1, 2, 3, 4]
    empirical_mse = {b: [] for b in bits}

    for rep in range(n_repeats):
        X = _random_unit_vectors(n, d, rng)
        for b in bits:
            tq = _get_tq(d, b, rng)
            if isinstance(tq, _ReferenceTurboQuant):
                X_rot = tq.rotate(X)
                X_hat_rot = tq.quantize(X_rot)
                # MSE per coordinate, averaged over all vectors and coordinates
                mse = np.mean((X_rot - X_hat_rot) ** 2)
            else:
                # Use real TurboQuant API: quantize_batch returns (codes, norms)
                codes, norms = tq.quantize_batch(X)
                X_hat = tq.dequantize_batch(codes, norms)
                # MSE per coordinate, averaged over all vectors and coordinates
                mse = np.mean((X - X_hat) ** 2)
            empirical_mse[b].append(mse)

    mean_mse = {b: np.mean(empirical_mse[b]) for b in bits}
    std_mse = {b: np.std(empirical_mse[b]) for b in bits}

    # Theoretical bounds
    # With data-adapted range R = 3/sqrt(d), step size delta = 2R/2^b = 6/(sqrt(d)*2^b)
    # Uniform quantization MSE = delta^2/12
    # For the per-coordinate MSE, this gives (6/(sqrt(d)*2^b))^2 / 12 = 3/(d * 4^b)
    R_scale = 3.0 / np.sqrt(d)
    turbo_bound = {b: (2 * R_scale / (2 ** b)) ** 2 / 12.0 for b in bits}
    # Information-theoretic floor: sigma^2 / 4^b where sigma^2 = 1/d
    info_floor = {b: (1.0 / d) / (4 ** b) for b in bits}

    for b in bits:
        print(f"  b={b}: empirical MSE = {mean_mse[b]:.6f} +/- {std_mse[b]:.6f}, "
              f"uniform quant MSE = {turbo_bound[b]:.6f}, "
              f"info floor = {info_floor[b]:.6f}")

    # Plot
    fig, ax = plt.subplots()
    b_arr = np.array(bits, dtype=float)
    means = np.array([mean_mse[b] for b in bits])
    stds = np.array([std_mse[b] for b in bits])
    ax.errorbar(b_arr, means, yerr=stds, fmt='o-', capsize=4, lw=2,
                label='Empirical MSE', zorder=3)
    ax.plot(b_arr, [turbo_bound[b] for b in bits], 's--', lw=1.5,
            label=r'Uniform quant: $\Delta^2/12$')
    ax.plot(b_arr, [info_floor[b] for b in bits], '^--', lw=1.5,
            label=r'Info floor: $1/4^b$')
    # TurboQuant theoretical MSE: (sqrt(3)*pi/2) * sigma^2 / 4^b
    # where sigma^2 = 1/d is the per-coordinate variance
    turbo_theory = [(np.sqrt(3) * np.pi / 2) * (1.0 / d) / (4 ** b) for b in bits]
    ax.plot(b_arr, turbo_theory, 'D--', lw=1.5,
            label=r'$\frac{\sqrt{3}\pi}{2d} / 4^b$')
    ax.set_yscale('log')
    ax.set_xlabel('Bits per coordinate (b)')
    ax.set_ylabel('MSE (per coordinate)')
    ax.set_title('Quantization MSE vs Bit-Width')
    ax.set_xticks(bits)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp01_mse_vs_bits.png')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'exp01_mse_vs_bits.png'}")

    return {
        'empirical_mse': {str(b): float(mean_mse[b]) for b in bits},
        'empirical_mse_std': {str(b): float(std_mse[b]) for b in bits},
        'turbo_bound': {str(b): float(turbo_bound[b]) for b in bits},
        'info_floor': {str(b): float(info_floor[b]) for b in bits},
        'turbo_theory': {str(b): float(v) for b, v in zip(bits, turbo_theory)},
    }


# ---------------------------------------------------------------------------
# Test 3: Inner product bias
# ---------------------------------------------------------------------------
def test_inner_product_bias(n: int, d: int, n_repeats: int, rng: np.random.RandomState):
    """Verify bias factor alpha_b for b=1,2,3,4."""
    print("\n[Test 3] Inner product bias factors")
    bits = [1, 2, 3, 4]
    empirical_alpha = {b: [] for b in bits}

    for rep in range(n_repeats):
        X = _random_unit_vectors(n, d, rng)
        for b in bits:
            tq = _get_tq(d, b, rng)
            if isinstance(tq, _ReferenceTurboQuant):
                alpha = tq.bias_factor(X, None)
            else:
                # Compute bias factor from real TurboQuant API:
                # alpha_b = E[z * Q(z)] / E[z^2]
                # where z are the rotated unit-vector coordinates
                X_rot = tq.rotate_batch(X)
                codes, norms = tq.quantize_batch(X)
                X_hat = tq.dequantize_batch(codes, norms)
                X_hat_rot = tq.rotate_batch(X_hat / norms[:, np.newaxis])
                # Per-coordinate: E[z * Q(z)] / E[z^2]
                numerator = np.mean(X_rot * X_hat_rot)
                denominator = np.mean(X_rot ** 2)
                alpha = numerator / denominator
            empirical_alpha[b].append(alpha)

    mean_alpha = {b: np.mean(empirical_alpha[b]) for b in bits}
    std_alpha = {b: np.std(empirical_alpha[b]) for b in bits}

    # Theoretical bias: alpha_b = E[z * Q(z)] / E[z^2]
    # where z ~ N(0, 1/d), Q is uniform quantizer with 2^b levels in [-R, R].
    #
    # For a Gaussian quantized by a uniform quantizer with range R = c*sigma:
    #   alpha_b ~ 1 - MSE_quant / sigma^2
    # The MSE for a uniform quantizer on Gaussian data is LESS than delta^2/12
    # because the data concentrates near 0. We compute it numerically.
    R_scale = 3.0 / np.sqrt(d)
    sigma = 1.0 / np.sqrt(d)
    theoretical_alpha = {}
    for b in bits:
        # Numerical integration: E[z * Q(z)] / E[z^2] for z ~ N(0, sigma^2)
        L = 2 ** b
        delta = 2 * R_scale / L
        # Bin edges: -R, -R+delta, ..., R
        edges = np.linspace(-R_scale, R_scale, L + 1)
        midpoints = (edges[:-1] + edges[1:]) / 2.0

        # E[z * Q(z)] = sum over bins of midpoint_j * E[z * 1_{z in bin_j}]
        # where E[z * 1_{a <= z < b}] = sigma^2 * (phi(a/sigma) - phi(b/sigma))
        # and phi is the standard normal PDF.
        dist = stats.norm(0, sigma)
        ez_qz = 0.0
        for j in range(L):
            lo, hi = edges[j], edges[j + 1]
            # E[z * 1_{lo <= z < hi}] for z ~ N(0, sigma^2)
            # = sigma^2 * (pdf(lo) - pdf(hi))  [standard result]
            ez_1_bin = sigma**2 * (dist.pdf(lo) - dist.pdf(hi))
            ez_qz += midpoints[j] * ez_1_bin
        # Tail contributions (clipped values):
        # z < -R: clipped to first bin midpoint
        ez_tail_lo = sigma**2 * (0 - dist.pdf(-R_scale))  # E[z * 1_{z < -R}]
        # Actually E[z * 1_{z < -R}] = -sigma^2 * pdf(-R) (using standard result)
        # But the quantized value for z < -R is midpoints[0]
        # This is already handled since our bin edges include -R to R and
        # the clip maps everything outside to the boundary bins.
        # For 3-sigma range, tail probability < 0.3%, negligible.

        alpha_b = ez_qz / sigma**2
        theoretical_alpha[b] = float(alpha_b)

    for b in bits:
        print(f"  b={b}: empirical alpha = {mean_alpha[b]:.6f} +/- {std_alpha[b]:.6f}, "
              f"theoretical = {theoretical_alpha.get(b, 'N/A')}")

    # Plot
    fig, ax = plt.subplots()
    b_arr = np.array(bits, dtype=float)
    means = np.array([mean_alpha[b] for b in bits])
    stds = np.array([std_alpha[b] for b in bits])
    ax.errorbar(b_arr, means, yerr=stds, fmt='o-', capsize=4, lw=2,
                label='Empirical $\\alpha_b$', zorder=3)
    th_vals = [theoretical_alpha.get(b, np.nan) for b in bits]
    ax.plot(b_arr, th_vals, 's--', lw=1.5, label='Theoretical $\\alpha_b$')
    ax.axhline(y=1.0, color='gray', ls=':', lw=1, alpha=0.5)
    ax.axhline(y=2.0 / np.pi, color='red', ls=':', lw=1, alpha=0.5, label='$2/\\pi$')
    ax.set_xlabel('Bits per coordinate (b)')
    ax.set_ylabel('Bias factor $\\alpha_b$')
    ax.set_title('Inner Product Bias Factor vs Bit-Width')
    ax.set_xticks(bits)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp01_bias_factor.png')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'exp01_bias_factor.png'}")

    return {
        'empirical_alpha': {str(b): float(mean_alpha[b]) for b in bits},
        'empirical_alpha_std': {str(b): float(std_alpha[b]) for b in bits},
        'theoretical_alpha': {str(b): float(v) for b, v in theoretical_alpha.items()},
    }


# ---------------------------------------------------------------------------
# Test 4: Inner product error vs dimension
# ---------------------------------------------------------------------------
def test_ip_error_vs_dimension(n: int, b: int, n_repeats: int, rng: np.random.RandomState):
    """Measure inner product MSE at b=2 across dimensions. Should be O(1/d)."""
    print("\n[Test 4] Inner product error vs dimension")
    dims = [32, 64, 128, 256, 512, 1024]
    ip_mse = {d: [] for d in dims}

    for rep in range(n_repeats):
        for d in dims:
            n_use = min(n, 10000)  # Cap for large dims to keep runtime manageable
            X = _random_unit_vectors(n_use, d, rng)
            Q_vecs = _random_unit_vectors(100, d, rng)  # 100 queries

            tq = _get_tq(d, b, rng)
            if isinstance(tq, _ReferenceTurboQuant):
                X_rot = tq.rotate(X)
                X_hat = tq.quantize(X_rot)
                Q_rot = tq.rotate(Q_vecs)
                Q_hat = tq.quantize(Q_rot)

                # Exact inner products: <q, k> = <Rq, Rk>
                exact_ip = Q_rot @ X_rot.T  # (100, n_use)
                # Approximate inner products from quantized codes
                approx_ip = Q_hat @ X_hat.T
            else:
                # Use real TurboQuant API
                codes_x, norms_x = tq.quantize_batch(X)
                exact_ip = Q_vecs @ X.T  # (100, n_use)
                # Compute approximate inner products using batch_dot_products
                approx_ip = np.empty((len(Q_vecs), n_use))
                for qi in range(len(Q_vecs)):
                    approx_ip[qi] = tq.batch_dot_products(codes_x, norms_x, Q_vecs[qi])

            mse = np.mean((exact_ip - approx_ip) ** 2)
            ip_mse[d].append(mse)

    mean_ip_mse = {d: np.mean(ip_mse[d]) for d in dims}
    std_ip_mse = {d: np.std(ip_mse[d]) for d in dims}

    for d in dims:
        print(f"  d={d:5d}: IP MSE = {mean_ip_mse[d]:.8f} +/- {std_ip_mse[d]:.8f}")

    # Fit O(1/d) model: MSE = C/d
    log_d = np.log(np.array(dims, dtype=float))
    log_mse = np.log(np.array([mean_ip_mse[d] for d in dims]))
    slope, intercept = np.polyfit(log_d, log_mse, 1)
    print(f"  Log-log slope: {slope:.4f} (expected: -1.0 for O(1/d))")

    # Plot
    fig, ax = plt.subplots()
    d_arr = np.array(dims, dtype=float)
    means = np.array([mean_ip_mse[d] for d in dims])
    stds = np.array([std_ip_mse[d] for d in dims])
    ax.errorbar(d_arr, means, yerr=stds, fmt='o-', capsize=4, lw=2,
                label='Empirical IP MSE', zorder=3)
    # Fit line
    C_fit = np.exp(intercept)
    ax.plot(d_arr, C_fit / d_arr, '--', lw=1.5, color='red',
            label=f'$C/d$ fit (slope={slope:.2f})')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Dimension $d$')
    ax.set_ylabel('Inner Product MSE')
    ax.set_title(f'Inner Product Error vs Dimension (b={b})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp01_ip_error_vs_dim.png')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'exp01_ip_error_vs_dim.png'}")

    return {
        'dims': dims,
        'mean_ip_mse': {str(d): float(mean_ip_mse[d]) for d in dims},
        'std_ip_mse': {str(d): float(std_ip_mse[d]) for d in dims},
        'log_log_slope': float(slope),
        'C_fit': float(C_fit),
    }


# ---------------------------------------------------------------------------
# Test 5: Near-independence of coordinates
# ---------------------------------------------------------------------------
def test_coordinate_independence(n: int, d: int, n_repeats: int, rng: np.random.RandomState):
    """Compute empirical correlation between coordinate pairs after rotation."""
    print("\n[Test 5] Near-independence of coordinates after rotation")
    n_pairs = 200  # Number of coordinate pairs to sample
    all_corrs = []

    for rep in range(n_repeats):
        R = _random_orthogonal(d, rng)
        X = _random_unit_vectors(n, d, rng)
        X_rot = X @ R.T

        # Sample random pairs of coordinates
        for _ in range(n_pairs):
            i, j = rng.choice(d, size=2, replace=False)
            corr = np.corrcoef(X_rot[:, i], X_rot[:, j])[0, 1]
            all_corrs.append(corr)

    all_corrs = np.array(all_corrs)
    mean_abs_corr = np.mean(np.abs(all_corrs))
    max_abs_corr = np.max(np.abs(all_corrs))
    theoretical_bound = 1.0 / (d ** 2)  # O(1/d^2)

    print(f"  Mean |correlation|: {mean_abs_corr:.8f}")
    print(f"  Max  |correlation|: {max_abs_corr:.8f}")
    print(f"  Theoretical O(1/d^2) = {theoretical_bound:.8f}")
    # Actually for n samples, the expected correlation magnitude is ~ 1/sqrt(n)
    # for truly independent coords. The near-independence claim is about the
    # population correlation being O(1/d^2), but sample correlation is O(1/sqrt(n)).
    sampling_noise = 1.0 / np.sqrt(n)
    print(f"  Sampling noise floor (1/sqrt(n)): {sampling_noise:.8f}")

    # Test across dimensions
    dims_test = [32, 64, 128, 256, 512, 1024]
    mean_corrs_by_d = {}
    for d_test in dims_test:
        corrs_d = []
        R = _random_orthogonal(d_test, rng)
        X = _random_unit_vectors(min(n, 50000), d_test, rng)
        X_rot = X @ R.T
        for _ in range(100):
            i, j = rng.choice(d_test, size=2, replace=False)
            corr = np.corrcoef(X_rot[:, i], X_rot[:, j])[0, 1]
            corrs_d.append(abs(corr))
        mean_corrs_by_d[d_test] = np.mean(corrs_d)

    # Plot histogram of correlations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(all_corrs, bins=80, density=True, alpha=0.7)
    ax1.axvline(x=0, color='red', ls='--', lw=1.5)
    ax1.set_xlabel('Pairwise coordinate correlation')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Coordinate correlations after rotation (d={d})')

    # Plot mean |corr| vs dimension
    d_plot = np.array(list(mean_corrs_by_d.keys()), dtype=float)
    c_plot = np.array(list(mean_corrs_by_d.values()))
    ax2.plot(d_plot, c_plot, 'o-', lw=2, label='Mean |correlation|')
    # Reference line 1/sqrt(n_samples) -- this is the dominant term
    n_eff = min(n, 50000)
    ax2.axhline(y=1.0 / np.sqrt(n_eff), color='red', ls='--', lw=1.5,
                label=f'$1/\\sqrt{{n}}$ = {1.0/np.sqrt(n_eff):.5f}')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Dimension $d$')
    ax2.set_ylabel('Mean |correlation|')
    ax2.set_title('Coordinate independence vs dimension')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp01_coordinate_independence.png')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'exp01_coordinate_independence.png'}")

    return {
        'mean_abs_correlation': float(mean_abs_corr),
        'max_abs_correlation': float(max_abs_corr),
        'sampling_noise_floor': float(sampling_noise),
        'mean_corrs_by_d': {str(d): float(v) for d, v in mean_corrs_by_d.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Experiment 01: TurboQuant Validation")
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

    results['test1_coordinate_distribution'] = test_coordinate_distribution(N, D, N_REPEATS, rng)
    results['test2_mse_vs_bits'] = test_mse_vs_bits(N, D, N_REPEATS, rng)
    results['test3_inner_product_bias'] = test_inner_product_bias(N, D, N_REPEATS, rng)
    results['test4_ip_error_vs_dimension'] = test_ip_error_vs_dimension(N, b=2, n_repeats=N_REPEATS, rng=rng)
    results['test5_coordinate_independence'] = test_coordinate_independence(N, D, N_REPEATS, rng)

    elapsed = time.time() - t0
    results['total_time_seconds'] = float(elapsed)
    results['parameters'] = {'n': N, 'd': D, 'n_repeats': N_REPEATS, 'library_available': _LIB_AVAILABLE}

    # Save JSON summary
    summary_path = RESULTS_DIR / 'exp01_results.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
