"""Metrics for evaluating sub-linear attention quality.

These metrics quantify how well a sub-linear attention algorithm
approximates exact (brute-force) attention:

- **Recall@k**: What fraction of the true top-k keys were found?
  This is the standard information-retrieval metric.

- **Attention output error**: L2 distance between exact and approximate
  weighted-sum outputs.  Directly measures task-level impact.

- **Softmax mass captured**: What fraction of the softmax distribution's
  probability mass is concentrated on the found keys?  This is the most
  meaningful metric for attention: if we capture 99% of softmax mass,
  the output error is bounded by 1% of the value norm.

- **Variance fraction V(m)**: What fraction of the dot-product variance
  is captured by the top-m coordinates of the rotated query?  This
  predicts the effectiveness of CoarseToFine's early rounds.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def recall_at_k(true_topk: np.ndarray, predicted_topk: np.ndarray) -> float:
    """Fraction of the true top-k keys that appear in the predicted set.

    recall@k = |true_topk intersection predicted_topk| / k

    Parameters
    ----------
    true_topk : ndarray, shape (k,)
        Indices of the true top-k keys (from brute-force).
    predicted_topk : ndarray, shape (k',)
        Indices of the predicted top-k keys.  k' may differ from k.

    Returns
    -------
    recall : float in [0, 1]
    """
    if len(true_topk) == 0:
        return 1.0 if len(predicted_topk) == 0 else 0.0
    true_set = set(true_topk.ravel())
    pred_set = set(predicted_topk.ravel())
    return len(true_set & pred_set) / len(true_set)


def attention_output_error(true_output: np.ndarray,
                           approx_output: np.ndarray,
                           relative: bool = True) -> float:
    """L2 error between exact and approximate attention outputs.

    If relative=True, returns ||true - approx||_2 / ||true||_2.
    Otherwise returns the absolute L2 distance.

    Parameters
    ----------
    true_output : ndarray, shape (d_v,)
        Exact attention output: softmax(scores) @ values.
    approx_output : ndarray, shape (d_v,)
        Approximate output from the sub-linear algorithm.
    relative : bool
        If True, normalize by the true output norm.

    Returns
    -------
    error : float
    """
    diff = np.linalg.norm(true_output - approx_output)
    if relative:
        true_norm = np.linalg.norm(true_output)
        return diff / true_norm if true_norm > 1e-30 else diff
    return float(diff)


def softmax_mass_captured(true_scores: np.ndarray,
                          found_indices: np.ndarray,
                          temperature: float = 1.0) -> float:
    """Fraction of softmax probability mass on the found keys.

    softmax_mass = sum_{i in found} exp(s_i / T) / sum_{j=1}^{n} exp(s_j / T)

    This is the most important metric for attention quality: if the found
    keys capture fraction f of the softmax mass, then the attention output
    error is bounded by (1 - f) * max_j ||v_j||.

    Parameters
    ----------
    true_scores : ndarray, shape (n,)
        Dot-product scores for all n keys (from brute force).
    found_indices : ndarray, shape (k,)
        Indices of keys found by the sub-linear algorithm.
    temperature : float
        Softmax temperature.  Lower temperature concentrates mass on
        the top keys (making sub-linear attention easier).

    Returns
    -------
    mass : float in [0, 1]
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    if len(true_scores) == 0:
        return 1.0

    # Numerically stable softmax
    scaled = true_scores / temperature
    max_s = np.max(scaled)
    exp_scores = np.exp(scaled - max_s)
    total_mass = np.sum(exp_scores)

    if total_mass < 1e-30:
        return 0.0

    found_mask = np.zeros(len(true_scores), dtype=bool)
    valid_indices = found_indices[found_indices < len(true_scores)]
    found_mask[valid_indices] = True

    found_mass = np.sum(exp_scores[found_mask])
    return float(found_mass / total_mass)


def variance_fraction(q_rotated: np.ndarray, m: int) -> float:
    r"""Fraction of dot-product variance captured by top-m coordinates.

    After RHT, the inner product <x, q> = <x_rot, q_rot>.  The variance
    of the estimator using only coordinates S is:

        V(S) = sum_{j in S} q_rot[j]^2 / ||q_rot||^2

    When S = top-m coordinates by |q_rot[j]|, this is maximized.

    For a random unit query, V(m) concentrates around m/d (since the
    RHT spreads energy uniformly).  But for structured queries, the
    top coordinates can capture much more variance.

    Parameters
    ----------
    q_rotated : ndarray, shape (d,)
        The rotated query vector.
    m : int
        Number of top coordinates to use.

    Returns
    -------
    vf : float in [0, 1]
        Fraction of variance captured.
    """
    q_sq = q_rotated ** 2
    total = np.sum(q_sq)
    if total < 1e-30:
        return 0.0
    # Top-m by magnitude
    top_m_idx = np.argsort(q_sq)[::-1][:m]
    captured = np.sum(q_sq[top_m_idx])
    return float(captured / total)


def precision_at_k(true_topk: np.ndarray, predicted_topk: np.ndarray) -> float:
    """Fraction of predicted top-k keys that are in the true top-k.

    precision@k = |true_topk intersection predicted_topk| / |predicted_topk|

    Parameters
    ----------
    true_topk : ndarray, shape (k,)
    predicted_topk : ndarray, shape (k',)

    Returns
    -------
    precision : float in [0, 1]
    """
    if len(predicted_topk) == 0:
        return 1.0 if len(true_topk) == 0 else 0.0
    true_set = set(true_topk.ravel())
    pred_set = set(predicted_topk.ravel())
    return len(true_set & pred_set) / len(pred_set)


def ndcg_at_k(true_scores: np.ndarray, predicted_topk: np.ndarray,
              k: int) -> float:
    """Normalized Discounted Cumulative Gain at k.

    NDCG@k = DCG@k(predicted) / DCG@k(ideal)

    where DCG@k = sum_{i=1}^{k} score(rank_i) / log2(i + 1)

    This accounts for the *ranking* of retrieved keys, not just which
    keys were found.

    Parameters
    ----------
    true_scores : ndarray, shape (n,)
        True dot-product scores for all keys.
    predicted_topk : ndarray, shape (k',)
        Indices of predicted top-k keys, in rank order.
    k : int
        Evaluation depth.

    Returns
    -------
    ndcg : float in [0, 1]
    """
    if len(predicted_topk) == 0 or k == 0:
        return 0.0

    # Ideal ranking
    ideal_order = np.argsort(true_scores)[::-1][:k]
    ideal_scores = true_scores[ideal_order]

    # DCG of ideal ranking
    positions = np.arange(1, k + 1)
    discounts = 1.0 / np.log2(positions + 1)
    ideal_dcg = np.sum(ideal_scores * discounts)

    if ideal_dcg < 1e-30:
        return 0.0

    # DCG of predicted ranking
    pred_k = min(k, len(predicted_topk))
    pred_scores = true_scores[predicted_topk[:pred_k]]
    pred_dcg = np.sum(pred_scores * discounts[:pred_k])

    return float(pred_dcg / ideal_dcg)


def mse_reconstruction(X_original: np.ndarray,
                       X_reconstructed: np.ndarray,
                       per_vector: bool = False) -> np.ndarray | float:
    """Mean squared error of vector reconstruction.

    Parameters
    ----------
    X_original : ndarray, shape (n, d) or (d,)
    X_reconstructed : ndarray, shape (n, d) or (d,)
    per_vector : bool
        If True, return per-vector MSE. Otherwise return scalar mean.

    Returns
    -------
    mse : float or ndarray
    """
    diff_sq = (X_original - X_reconstructed) ** 2
    if X_original.ndim == 1:
        return float(np.mean(diff_sq))
    if per_vector:
        return np.mean(diff_sq, axis=1)
    return float(np.mean(diff_sq))


def cosine_similarity_preserved(X_original: np.ndarray,
                                X_reconstructed: np.ndarray) -> float:
    """Average cosine similarity between original and reconstructed vectors.

    A value of 1.0 means perfect directional preservation.

    Parameters
    ----------
    X_original : ndarray, shape (n, d)
    X_reconstructed : ndarray, shape (n, d)

    Returns
    -------
    avg_cosine : float in [-1, 1]
    """
    # Normalize rows
    norms_orig = np.linalg.norm(X_original, axis=1, keepdims=True)
    norms_recon = np.linalg.norm(X_reconstructed, axis=1, keepdims=True)

    # Avoid division by zero
    safe_orig = np.where(norms_orig > 1e-30, norms_orig, 1.0)
    safe_recon = np.where(norms_recon > 1e-30, norms_recon, 1.0)

    X_orig_unit = X_original / safe_orig
    X_recon_unit = X_reconstructed / safe_recon

    cosines = np.sum(X_orig_unit * X_recon_unit, axis=1)
    return float(np.mean(cosines))


def inner_product_correlation(true_dots: np.ndarray,
                              estimated_dots: np.ndarray) -> float:
    """Pearson correlation between true and estimated inner products.

    This measures how well the quantized representation preserves the
    *ranking* of inner products (which is what attention cares about).

    Parameters
    ----------
    true_dots : ndarray, shape (n,)
    estimated_dots : ndarray, shape (n,)

    Returns
    -------
    correlation : float in [-1, 1]
    """
    if len(true_dots) < 2:
        return 1.0
    return float(np.corrcoef(true_dots, estimated_dots)[0, 1])


def speedup_ratio(brute_force_ops: int, sublinear_ops: int) -> float:
    """Computational speedup of the sub-linear algorithm.

    Parameters
    ----------
    brute_force_ops : int
        Number of coordinate reads in brute-force (= n * d).
    sublinear_ops : int
        Number of coordinate reads in the sub-linear algorithm.

    Returns
    -------
    speedup : float
        brute_force_ops / sublinear_ops
    """
    if sublinear_ops == 0:
        return float("inf")
    return brute_force_ops / sublinear_ops


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Metrics self-test")
    print("=" * 70)

    rng = np.random.default_rng(42)

    # recall_at_k
    true_topk = np.array([0, 1, 2, 3, 4])
    pred_perfect = np.array([0, 1, 2, 3, 4])
    pred_half = np.array([0, 1, 5, 6, 7])
    pred_none = np.array([5, 6, 7, 8, 9])
    assert recall_at_k(true_topk, pred_perfect) == 1.0
    assert abs(recall_at_k(true_topk, pred_half) - 0.4) < 1e-10
    assert recall_at_k(true_topk, pred_none) == 0.0
    print("  recall_at_k:              PASS")

    # softmax_mass_captured
    scores = np.array([10.0, 5.0, 1.0, 0.5, 0.1])
    found_top1 = np.array([0])
    mass = softmax_mass_captured(scores, found_top1, temperature=1.0)
    assert mass > 0.99, f"Top score should dominate: got {mass}"
    print(f"  softmax_mass (top-1):     {mass:.6f}  PASS")

    found_all = np.arange(5)
    mass_all = softmax_mass_captured(scores, found_all, temperature=1.0)
    assert abs(mass_all - 1.0) < 1e-10
    print(f"  softmax_mass (all):       {mass_all:.6f}  PASS")

    # variance_fraction
    q = rng.standard_normal(128)
    vf_all = variance_fraction(q, 128)
    assert abs(vf_all - 1.0) < 1e-10
    vf_half = variance_fraction(q, 64)
    assert 0.0 < vf_half < 1.0
    vf_few = variance_fraction(q, 8)
    assert vf_few <= vf_half
    print(f"  variance_fraction(8):     {vf_few:.4f}")
    print(f"  variance_fraction(64):    {vf_half:.4f}")
    print(f"  variance_fraction(128):   {vf_all:.4f}  PASS")

    # attention_output_error
    true_out = rng.standard_normal(64)
    err_zero = attention_output_error(true_out, true_out)
    assert err_zero < 1e-10
    noisy_out = true_out + 0.01 * rng.standard_normal(64)
    err_small = attention_output_error(true_out, noisy_out)
    assert err_small < 0.05
    print(f"  attention_output_error:   {err_small:.6f}  PASS")

    # ndcg_at_k
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1])
    perfect_pred = np.array([0, 1, 2, 3, 4])
    ndcg_perfect = ndcg_at_k(scores, perfect_pred, k=5)
    assert abs(ndcg_perfect - 1.0) < 1e-10
    reversed_pred = np.array([4, 3, 2, 1, 0])
    ndcg_reversed = ndcg_at_k(scores, reversed_pred, k=5)
    assert ndcg_reversed < 1.0
    print(f"  ndcg@5 (perfect):         {ndcg_perfect:.4f}")
    print(f"  ndcg@5 (reversed):        {ndcg_reversed:.4f}  PASS")

    # cosine_similarity_preserved
    X = rng.standard_normal((100, 64))
    cos_perf = cosine_similarity_preserved(X, X)
    assert abs(cos_perf - 1.0) < 1e-10
    X_noisy = X + 0.1 * rng.standard_normal((100, 64))
    cos_noisy = cosine_similarity_preserved(X, X_noisy)
    assert cos_noisy > 0.9
    print(f"  cosine_preserved (noisy): {cos_noisy:.4f}  PASS")

    # inner_product_correlation
    true_d = rng.standard_normal(1000)
    est_d = true_d + 0.1 * rng.standard_normal(1000)
    corr = inner_product_correlation(true_d, est_d)
    assert corr > 0.99
    print(f"  dot_correlation:          {corr:.6f}  PASS")

    print("\n" + "=" * 70)
    print("All metrics tests passed.")
    print("=" * 70)
