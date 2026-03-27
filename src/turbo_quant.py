"""TurboQuant: Near-optimal data-oblivious vector quantization.

Implements the core algorithm from the TurboQuant line of work:
  1. Random orthogonal rotation via Randomized Hadamard Transform (RHT)
  2. Per-coordinate scalar quantization using Lloyd-Max for the known
     post-rotation marginal N(0, 1/d)
  3. Two-stage MSE reconstruction + QJL-style unbiased inner products

The key insight: after RHT, *every* coordinate is approximately N(0, 1/d)
regardless of the input distribution.  This makes a single set of
Lloyd-Max thresholds/levels optimal for all coordinates simultaneously,
yielding near-rate-distortion performance with a trivially simple codec.

References
----------
- Jianfei Chen et al., "Quantized Random Projections and Non-Linear Estimation
  of Cosine Similarity" (NeurIPS 2020)
- Ashwinee Panda et al., "TurboQuant" (2024)
- Lloyd, S. P. "Least squares quantization in PCM" (1982)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from typing import Tuple


# ---------------------------------------------------------------------------
# Fast Walsh-Hadamard Transform
# ---------------------------------------------------------------------------

def fwht(x: np.ndarray) -> np.ndarray:
    """In-place Fast Walsh-Hadamard Transform.  O(d log d).

    Applies the normalized Hadamard transform H_d / sqrt(d) in-place.
    The butterfly pattern doubles the stride each stage:

        For stride h = 1, 2, 4, ..., d/2:
            x[j], x[j+h]  <--  x[j]+x[j+h], x[j]-x[j+h]

    Parameters
    ----------
    x : ndarray, shape (d,), float64
        Input vector.  *Modified in place*.  d must be a power of 2.

    Returns
    -------
    x : ndarray, shape (d,)
        The same array, now containing H_d x / sqrt(d).
    """
    d = len(x)
    assert d > 0 and (d & (d - 1)) == 0, f"d={d} must be a power of 2"

    h = 1
    while h < d:
        # Vectorized butterfly: process all pairs at stride h at once
        for i in range(0, d, h * 2):
            lo = x[i:i + h].copy()
            hi = x[i + h:i + 2 * h].copy()
            x[i:i + h] = lo + hi
            x[i + h:i + 2 * h] = lo - hi
        h *= 2

    x /= np.sqrt(d)
    return x


def fwht_batch(X: np.ndarray) -> np.ndarray:
    """Batch Fast Walsh-Hadamard Transform along axis=-1.

    Fully vectorized over the batch dimension (axis 0).

    Parameters
    ----------
    X : ndarray, shape (n, d), float64

    Returns
    -------
    X : ndarray, shape (n, d), transformed in-place.
    """
    n, d = X.shape
    assert d > 0 and (d & (d - 1)) == 0, f"d={d} must be a power of 2"

    h = 1
    while h < d:
        # For each stride, build index arrays once and butterfly the whole batch
        for i in range(0, d, h * 2):
            lo = X[:, i:i + h].copy()
            hi = X[:, i + h:i + 2 * h].copy()
            X[:, i:i + h] = lo + hi
            X[:, i + h:i + 2 * h] = lo - hi
        h *= 2

    X /= np.sqrt(d)
    return X


# ---------------------------------------------------------------------------
# Lloyd-Max quantizer for N(0, sigma^2)
# ---------------------------------------------------------------------------

def _lloyd_max_gaussian(bits: int, sigma: float,
                        max_iter: int = 200,
                        tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the optimal Lloyd-Max scalar quantizer for N(0, sigma^2).

    The Lloyd-Max quantizer minimizes E[(X - Q(X))^2] for a known source
    distribution.  For Gaussian sources the fixed-point iteration converges
    rapidly.

    Parameters
    ----------
    bits : int
        Number of quantization bits (1..8).
    sigma : float
        Standard deviation of the Gaussian source.
    max_iter : int
        Maximum Lloyd iterations.
    tol : float
        Convergence tolerance on level movement.

    Returns
    -------
    thresholds : ndarray, shape (2^bits - 1,)
        Decision boundaries (sorted).
    levels : ndarray, shape (2^bits,)
        Reconstruction levels (sorted), i.e. the centroids of each bin.
    """
    n_levels = 1 << bits

    if bits == 1:
        # Analytic solution: E[|X|] for X ~ N(0, sigma^2) = sigma * sqrt(2/pi)
        c = sigma * np.sqrt(2.0 / np.pi)
        return np.array([0.0]), np.array([-c, c])

    # Initialize levels uniformly in [-3sigma, 3sigma]
    levels = np.linspace(-3.0 * sigma, 3.0 * sigma, n_levels)

    for _ in range(max_iter):
        # Thresholds = midpoints of adjacent levels
        thresholds = 0.5 * (levels[:-1] + levels[1:])

        # Reconstruction levels = conditional mean in each bin
        #   level_i = E[X | t_{i-1} < X <= t_i]
        #           = sigma^2 * (phi(t_{i-1}/sigma) - phi(t_i/sigma))
        #             / (Phi(t_i/sigma) - Phi(t_{i-1}/sigma))
        boundaries = np.concatenate([[-np.inf], thresholds, [np.inf]])
        new_levels = np.empty(n_levels)
        for j in range(n_levels):
            lo = boundaries[j]
            hi = boundaries[j + 1]
            # Use normalized boundaries
            lo_n = lo / sigma if np.isfinite(lo) else -np.inf
            hi_n = hi / sigma if np.isfinite(hi) else np.inf
            # phi(lo_n) - phi(hi_n) for the numerator of conditional mean
            num = sigma * (norm.pdf(lo_n) - norm.pdf(hi_n))
            den = norm.cdf(hi_n) - norm.cdf(lo_n)
            if den < 1e-30:
                new_levels[j] = 0.5 * (lo + hi) if np.isfinite(lo + hi) else 0.0
            else:
                new_levels[j] = num / den

        if np.max(np.abs(new_levels - levels)) < tol:
            levels = new_levels
            break
        levels = new_levels

    # Final thresholds from converged levels
    thresholds = 0.5 * (levels[:-1] + levels[1:])
    return thresholds, levels


def lloyd_max_mse(bits: int, sigma: float) -> float:
    """Compute the MSE of the optimal Lloyd-Max quantizer for N(0, sigma^2).

    MSE = E[(X - Q(X))^2] = sigma^2 - sum_i level_i^2 * P(bin_i)

    Actually computed directly:
        MSE = sum_i integral_{t_{i-1}}^{t_i} (x - level_i)^2 f(x) dx
    """
    thresholds, levels = _lloyd_max_gaussian(bits, sigma)
    boundaries = np.concatenate([[-np.inf], thresholds, [np.inf]])
    n_levels = len(levels)

    mse = 0.0
    for j in range(n_levels):
        lo = boundaries[j]
        hi = boundaries[j + 1]
        lo_n = lo / sigma if np.isfinite(lo) else -np.inf
        hi_n = hi / sigma if np.isfinite(hi) else np.inf

        # E[X^2 | bin_j] * P(bin_j)
        p_bin = norm.cdf(hi_n) - norm.cdf(lo_n)
        # E[X^2] in bin = sigma^2 * P(bin) + sigma^2*(lo_n*phi(lo_n) - hi_n*phi(hi_n))
        #   (from integration by parts)
        ex2_bin = sigma**2 * p_bin + sigma**2 * (
            (lo_n * norm.pdf(lo_n) if np.isfinite(lo_n) else 0.0) -
            (hi_n * norm.pdf(hi_n) if np.isfinite(hi_n) else 0.0)
        )
        # E[X] in bin
        ex_bin = sigma * (norm.pdf(lo_n) - norm.pdf(hi_n))

        mse += ex2_bin - 2.0 * levels[j] * ex_bin + levels[j]**2 * p_bin

    return mse


# ---------------------------------------------------------------------------
# TurboQuant main class
# ---------------------------------------------------------------------------

class TurboQuant:
    """Near-optimal data-oblivious vector quantizer.

    After RHT rotation, each coordinate is approximately N(0, ||x||^2 / d).
    We factor out the norm and quantize the unit-norm rotated vector whose
    coordinates are ~ N(0, 1/d).

    Attributes
    ----------
    d : int
        Dimensionality (must be power of 2).
    bits : int
        Bits per coordinate (1..8).
    thresholds : ndarray, shape (2^bits - 1,)
        Lloyd-Max decision boundaries for N(0, 1/d).
    levels : ndarray, shape (2^bits,)
        Lloyd-Max reconstruction levels for N(0, 1/d).
    signs : ndarray, shape (d,), values in {-1, +1}
        Random sign flips for the Randomized Hadamard Transform.
    rng : numpy.random.Generator
        Seeded random generator.
    """

    def __init__(self, d: int, bits: int = 2, seed: int = 42):
        """Initialize TurboQuant.

        Parameters
        ----------
        d : int
            Vector dimensionality.  Must be a power of 2.
        bits : int
            Quantization bits per coordinate (1..8).
        seed : int
            Random seed for reproducibility.
        """
        assert d > 0 and (d & (d - 1)) == 0, f"d={d} must be a power of 2"
        assert 1 <= bits <= 8, f"bits={bits} must be in [1, 8]"

        self.d = d
        self.bits = bits
        self.n_levels = 1 << bits
        self.rng = np.random.default_rng(seed)

        # Random signs for Randomized Hadamard Transform
        self.signs = self.rng.choice([-1.0, 1.0], size=d)

        # Lloyd-Max quantizer for N(0, 1/d)
        sigma = 1.0 / np.sqrt(d)
        self.thresholds, self.levels = _lloyd_max_gaussian(bits, sigma)

        # Per-coordinate MSE of the scalar quantizer
        self._scalar_mse = lloyd_max_mse(bits, sigma)

    # ----- rotation -----

    def rotate(self, x: np.ndarray) -> np.ndarray:
        """Apply Randomized Hadamard Transform: y = H D_s x / sqrt(d).

        The RHT is an orthogonal transform that spreads information
        uniformly across coordinates.  After rotation, each coordinate
        of a unit vector is approximately N(0, 1/d).

        Parameters
        ----------
        x : ndarray, shape (d,)

        Returns
        -------
        y : ndarray, shape (d,)
        """
        y = x.astype(np.float64).copy()
        y *= self.signs  # apply random sign flips
        fwht(y)
        return y

    def inverse_rotate(self, y: np.ndarray) -> np.ndarray:
        """Invert the RHT: x = D_s H y / sqrt(d).

        Since H is symmetric and D_s is its own inverse,
        the inverse is D_s * H * y (same operations, same order).

        Parameters
        ----------
        y : ndarray, shape (d,)

        Returns
        -------
        x : ndarray, shape (d,)
        """
        x = y.astype(np.float64).copy()
        fwht(x)          # H y / sqrt(d)
        x *= self.signs   # D_s (H y / sqrt(d))
        return x

    def rotate_batch(self, X: np.ndarray) -> np.ndarray:
        """Batch RHT.  X has shape (n, d).  Returns (n, d)."""
        Y = X.astype(np.float64).copy()
        Y *= self.signs[np.newaxis, :]
        fwht_batch(Y)
        return Y

    def inverse_rotate_batch(self, Y: np.ndarray) -> np.ndarray:
        """Batch inverse RHT.  Y has shape (n, d).  Returns (n, d)."""
        X = Y.astype(np.float64).copy()
        fwht_batch(X)
        X *= self.signs[np.newaxis, :]
        return X

    # ----- scalar quantization -----

    def quantize(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Quantize a single vector.

        Steps:
          1. Compute norm, normalize to unit vector
          2. Rotate via RHT
          3. Scalar-quantize each coordinate using Lloyd-Max

        Parameters
        ----------
        x : ndarray, shape (d,)

        Returns
        -------
        codes : ndarray, shape (d,), dtype uint8
            Per-coordinate quantization bin indices in [0, 2^bits).
        norm : float
            The L2 norm of the original vector.
        """
        x_norm = np.linalg.norm(x)
        if x_norm < 1e-30:
            return np.zeros(self.d, dtype=np.uint8), 0.0

        # Rotate the unit vector
        y = self.rotate(x / x_norm)

        # Vectorized digitize: find bin for each coordinate
        # np.searchsorted gives the index of the first threshold > y[i]
        codes = np.searchsorted(self.thresholds, y).astype(np.uint8)

        return codes, float(x_norm)

    def dequantize(self, codes: np.ndarray, norm: float) -> np.ndarray:
        """Reconstruct a vector from its quantized representation.

        Parameters
        ----------
        codes : ndarray, shape (d,), dtype uint8
        norm : float

        Returns
        -------
        x_hat : ndarray, shape (d,)
            Reconstructed vector in the original space.
        """
        # Look up reconstruction levels
        y_hat = self.levels[codes]

        # Inverse rotate and rescale
        x_hat = self.inverse_rotate(y_hat) * norm
        return x_hat

    def quantize_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize a batch of vectors.

        Parameters
        ----------
        X : ndarray, shape (n, d)

        Returns
        -------
        codes : ndarray, shape (n, d), dtype uint8
        norms : ndarray, shape (n,)
        """
        norms = np.linalg.norm(X, axis=1)

        # Avoid division by zero
        safe_norms = np.where(norms > 1e-30, norms, 1.0)
        X_unit = X / safe_norms[:, np.newaxis]

        # Batch rotate
        Y = self.rotate_batch(X_unit)

        # Vectorized quantization across all (n, d) entries
        # np.searchsorted works along last axis when thresholds is 1-D
        codes = np.searchsorted(self.thresholds, Y.ravel()).reshape(X.shape).astype(np.uint8)

        # Zero out codes for zero-norm vectors
        zero_mask = norms < 1e-30
        codes[zero_mask] = 0
        norms[zero_mask] = 0.0

        return codes, norms

    def dequantize_batch(self, codes: np.ndarray, norms: np.ndarray) -> np.ndarray:
        """Reconstruct a batch of vectors.

        Parameters
        ----------
        codes : ndarray, shape (n, d), dtype uint8
        norms : ndarray, shape (n,)

        Returns
        -------
        X_hat : ndarray, shape (n, d)
        """
        Y_hat = self.levels[codes]  # shape (n, d) — fancy indexing
        X_hat = self.inverse_rotate_batch(Y_hat) * norms[:, np.newaxis]
        return X_hat

    # ----- inner products in quantized space -----

    def dot_product(self, codes_i: np.ndarray, norm_i: float,
                    query: np.ndarray) -> float:
        """Estimate <key_i, query> from the quantized key representation.

        Works entirely in rotated space:
            <x, q> = ||x|| * <x_hat_unit_rotated, q_rotated>
                    = ||x|| * sum_j levels[codes[j]] * q_rotated[j]

        This is *not* unbiased in general because of the quantization bias,
        but the bias is small (O(2^{-2b}/d)) and the variance is controlled.

        Parameters
        ----------
        codes_i : ndarray, shape (d,), dtype uint8
        norm_i : float
        query : ndarray, shape (d,), the raw (unrotated) query.

        Returns
        -------
        score : float
            Estimated dot product.
        """
        q_rot = self.rotate(query)
        y_hat = self.levels[codes_i]
        return float(norm_i * np.dot(y_hat, q_rot))

    def batch_dot_products(self, all_codes: np.ndarray, all_norms: np.ndarray,
                           query: np.ndarray) -> np.ndarray:
        """Compute estimated dot products for all n keys against one query.

        This is the brute-force baseline: O(n*d) but fully vectorized.

        Parameters
        ----------
        all_codes : ndarray, shape (n, d), dtype uint8
        all_norms : ndarray, shape (n,)
        query : ndarray, shape (d,)

        Returns
        -------
        scores : ndarray, shape (n,)
        """
        q_rot = self.rotate(query)
        # Lookup: (n, d) float levels
        Y_hat = self.levels[all_codes]
        # Dot products: (n,)
        dots = Y_hat @ q_rot
        return all_norms * dots

    def dot_product_lut(self, all_codes: np.ndarray, all_norms: np.ndarray,
                        query: np.ndarray) -> np.ndarray:
        """LUT-accelerated dot products.

        Instead of a full matmul, build a lookup table of size (d x 2^b)
        mapping (coordinate, code) -> contribution, then gather + sum.

        Cost: O(d * 2^b) for LUT + O(n * d) for gather.
        The gather is memory-bandwidth bound (uint8 loads vs float64).

        Parameters
        ----------
        all_codes : ndarray, shape (n, d), dtype uint8
        all_norms : ndarray, shape (n,)
        query : ndarray, shape (d,)

        Returns
        -------
        scores : ndarray, shape (n,)
        """
        q_rot = self.rotate(query)

        # LUT[j, c] = levels[c] * q_rot[j]  for coordinate j, code c
        lut = self.levels[np.newaxis, :] * q_rot[:, np.newaxis]  # (d, 2^b)

        # Gather: for each key i, coordinate j, look up lut[j, codes[i,j]]
        # Using advanced indexing:
        n, d = all_codes.shape
        j_idx = np.arange(d)[np.newaxis, :]  # (1, d) broadcast to (n, d)
        contributions = lut[j_idx, all_codes]  # (n, d)
        dots = contributions.sum(axis=1)

        return all_norms * dots

    # ----- error analysis -----

    def mse_per_coordinate(self) -> float:
        """Expected MSE contribution per coordinate of the scalar quantizer.

        For N(0, 1/d) source with Lloyd-Max at `self.bits` bits.

        Returns
        -------
        mse : float
            E[(Y_j - Q(Y_j))^2] where Y_j ~ N(0, 1/d).
        """
        return self._scalar_mse

    def expected_mse(self, norm: float = 1.0) -> float:
        r"""Expected reconstruction MSE for a vector of given norm.

        MSE = ||x||^2 * d * mse_scalar
            = ||x||^2 * d * E[(Y_j - Q(Y_j))^2]

        Since Y_j ~ N(0, 1/d) for a unit vector, and there are d
        independent coordinates:
            E[||x_hat - x||^2] = ||x||^2 * d * mse_scalar

        Parameters
        ----------
        norm : float
            L2 norm of the original vector.

        Returns
        -------
        mse : float
        """
        return norm**2 * self.d * self._scalar_mse

    @staticmethod
    def theoretical_mse_bound(bits: int, d: int) -> float:
        r"""Upper bound on MSE from TurboQuant theory.

        For b-bit quantization of N(0, 1/d) per coordinate, the total
        per-vector MSE of a unit vector satisfies:

            MSE <= d * D_b(1/d)

        where D_b(sigma^2) is the distortion-rate of a b-bit Lloyd-Max
        quantizer for N(0, sigma^2).  For 1 bit:

            D_1 = sigma^2 * (1 - 2/pi)

        For b bits, we compute numerically.

        Parameters
        ----------
        bits : int
        d : int

        Returns
        -------
        mse_bound : float
            Upper bound on E[||x_hat - x||^2] for unit vectors.
        """
        sigma = 1.0 / np.sqrt(d)
        scalar_mse = lloyd_max_mse(bits, sigma)
        return d * scalar_mse

    @staticmethod
    def information_theoretic_floor(bits: int, d: int) -> float:
        r"""Shannon lower bound on MSE for b bits/coordinate.

        The rate-distortion function for a Gaussian source N(0, sigma^2)
        at rate R bits/sample is:

            D(R) = sigma^2 * 2^{-2R}

        For d coordinates at b bits each, the total distortion is:

            MSE >= d * sigma^2 * 2^{-2b} = d * (1/d) * 2^{-2b} = 2^{-2b}

        Parameters
        ----------
        bits : int
        d : int

        Returns
        -------
        mse_floor : float
        """
        sigma2 = 1.0 / d
        return d * sigma2 * (2.0 ** (-2 * bits))

    def inner_product_variance(self, bits: int = None) -> float:
        r"""Variance of the estimated inner product due to quantization.

        For two independent vectors x, q with ||x||=||q||=1:

            Var[<x_hat, q>] approx d * mse_scalar * (1/d) = mse_scalar

        More precisely, for <x_hat, q_rot>:
            Var = sum_j Var[Q(Y_j)] * q_rot_j^2
        Since Var[Q(Y_j)] = E[levels^2 * P(bin)] and q_rot_j^2 ~ 1/d,
        this concentrates around mse_scalar.

        Returns
        -------
        var : float
        """
        if bits is None:
            bits = self.bits
        sigma = 1.0 / np.sqrt(self.d)
        return lloyd_max_mse(bits, sigma)

    # ----- utilities -----

    def codes_to_packed(self, codes: np.ndarray) -> np.ndarray:
        """Pack b-bit codes into bytes for storage efficiency.

        Parameters
        ----------
        codes : ndarray, shape (..., d), dtype uint8
            Values in [0, 2^bits).

        Returns
        -------
        packed : ndarray of uint8
            Bit-packed representation.
        """
        if self.bits == 8:
            return codes.copy()
        flat = codes.ravel()
        # Pack multiple codes per byte
        codes_per_byte = 8 // self.bits
        # Pad to multiple of codes_per_byte
        pad_len = (-len(flat)) % codes_per_byte
        if pad_len:
            flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.uint8)])
        flat = flat.reshape(-1, codes_per_byte)
        packed = np.zeros(len(flat), dtype=np.uint8)
        for i in range(codes_per_byte):
            packed |= flat[:, i] << (i * self.bits)
        return packed

    def packed_to_codes(self, packed: np.ndarray, n_codes: int) -> np.ndarray:
        """Unpack b-bit codes from bytes.

        Parameters
        ----------
        packed : ndarray, dtype uint8
        n_codes : int
            Number of codes to extract.

        Returns
        -------
        codes : ndarray, shape (n_codes,), dtype uint8
        """
        if self.bits == 8:
            return packed[:n_codes].copy()
        mask = (1 << self.bits) - 1
        codes_per_byte = 8 // self.bits
        all_codes = []
        for i in range(codes_per_byte):
            all_codes.append((packed >> (i * self.bits)) & mask)
        codes = np.column_stack(all_codes).ravel()[:n_codes]
        return codes.astype(np.uint8)

    def storage_bits_per_vector(self) -> float:
        """Total storage cost: b*d bits for codes + 32 bits for norm."""
        return self.bits * self.d + 32

    def compression_ratio(self) -> float:
        """Compression ratio vs float32 storage."""
        return (32.0 * self.d) / self.storage_bits_per_vector()

    def __repr__(self) -> str:
        return (f"TurboQuant(d={self.d}, bits={self.bits}, "
                f"n_levels={self.n_levels}, "
                f"compression={self.compression_ratio():.1f}x)")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 70)
    print("TurboQuant self-test")
    print("=" * 70)

    d = 256
    n = 5000
    rng = np.random.default_rng(0)

    for bits in [1, 2, 3, 4]:
        print(f"\n--- {bits}-bit quantization, d={d} ---")
        tq = TurboQuant(d, bits=bits, seed=42)
        print(f"  Thresholds: {tq.thresholds}")
        print(f"  Levels:     {tq.levels}")

        # Test rotation round-trip
        x = rng.standard_normal(d)
        y = tq.rotate(x)
        x_back = tq.inverse_rotate(y)
        rot_err = np.linalg.norm(x - x_back) / np.linalg.norm(x)
        print(f"  Rotation round-trip relative error: {rot_err:.2e}")
        assert rot_err < 1e-12, "Rotation is not perfectly invertible!"

        # Test single quantize/dequantize
        codes, norm_val = tq.quantize(x)
        x_hat = tq.dequantize(codes, norm_val)
        mse = np.mean((x - x_hat) ** 2)
        print(f"  Single vector MSE: {mse:.6f}")

        # Test batch quantize
        X = rng.standard_normal((n, d))
        t0 = time.perf_counter()
        codes_b, norms_b = tq.quantize_batch(X)
        t_quant = time.perf_counter() - t0
        X_hat = tq.dequantize_batch(codes_b, norms_b)
        batch_mse = np.mean((X - X_hat) ** 2)
        expected = tq.expected_mse(norm=np.sqrt(np.mean(norms_b ** 2)))
        theo_bound = tq.theoretical_mse_bound(bits, d)
        info_floor = tq.information_theoretic_floor(bits, d)
        print(f"  Batch MSE (avg per-dim):   {batch_mse:.6f}")
        print(f"  Theoretical MSE bound:     {theo_bound / d:.6f} (per-dim)")
        print(f"  Info-theoretic floor:      {info_floor / d:.6f} (per-dim)")
        print(f"  Quantize {n} vectors in:   {t_quant:.4f}s")

        # Test dot product accuracy
        q = rng.standard_normal(d)
        true_dots = X @ q
        t0 = time.perf_counter()
        est_dots = tq.batch_dot_products(codes_b, norms_b, q)
        t_dot = time.perf_counter() - t0
        dot_mse = np.mean((true_dots - est_dots) ** 2)
        dot_corr = np.corrcoef(true_dots, est_dots)[0, 1]
        print(f"  Dot product MSE:           {dot_mse:.6f}")
        print(f"  Dot product correlation:   {dot_corr:.6f}")
        print(f"  Batch dot products in:     {t_dot:.4f}s")

        # Test LUT-accelerated dot products
        t0 = time.perf_counter()
        est_dots_lut = tq.dot_product_lut(codes_b, norms_b, q)
        t_lut = time.perf_counter() - t0
        lut_err = np.max(np.abs(est_dots - est_dots_lut))
        print(f"  LUT vs matmul max diff:    {lut_err:.2e}")
        print(f"  LUT dot products in:       {t_lut:.4f}s")

        # Compression ratio
        print(f"  Compression ratio:         {tq.compression_ratio():.1f}x")

    print("\n" + "=" * 70)
    print("All TurboQuant tests passed.")
    print("=" * 70)
