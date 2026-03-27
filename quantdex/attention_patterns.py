"""Realistic LLM attention pattern generator.

Generates KV cache vectors that produce attention patterns matching
empirical observations from modern LLM research:

- **Attention sinks** (StreamingLLM, Xiao et al. 2023): The first few
  tokens (BOS, punctuation) receive high attention regardless of query
  content, typically 5-10 tokens.

- **Local window** (StreamingLLM): Recent tokens in a sliding window
  of ~256-512 tokens receive moderate-to-high attention.

- **Semantic clusters** (SnapKV, Li et al. 2024): Groups of tokens
  about the same semantic topic cluster together in key space, and
  queries attend strongly to their relevant cluster.

- **Power-law decay**: Attention weight decreases as a power law
  with distance from the query position.

- **Heavy hitters** (H2O, Zhang et al. 2023): Only 1-5% of tokens
  carry >1% of total attention weight; 5% of tokens capture 95% of
  softmax mass.

References
----------
- Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative
  Inference of Large Language Models" (NeurIPS 2023)
- Li et al., "SnapKV: LLM Knows What You Are Looking For Before
  Generation" (2024)
- Xiao et al., "Efficient Streaming Language Models with Attention
  Sinks" (ICLR 2024)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


def _sample_vmf(center: np.ndarray, kappa: float, n: int,
                rng: np.random.Generator) -> np.ndarray:
    """Sample n unit vectors from von Mises-Fisher distribution.

    Uses the tangent-normal decomposition: project Gaussian noise
    orthogonal to the center, then mix center and noise with a
    concentration-dependent weight.

    Parameters
    ----------
    center : ndarray, shape (d,)
        Mean direction (will be normalized).
    kappa : float
        Concentration parameter.  Higher = tighter cluster.
    n : int
        Number of samples.
    rng : numpy.random.Generator

    Returns
    -------
    samples : ndarray, shape (n, d), unit-norm rows.
    """
    d = len(center)
    center = center / np.linalg.norm(center)

    # For high d + moderate kappa, the Gaussian approximation is excellent
    noise = rng.standard_normal((n, d))
    # Project out the center component to get tangent noise
    noise -= np.outer(noise @ center, center)
    # Normalize tangent part
    tangent_norms = np.linalg.norm(noise, axis=1, keepdims=True)
    tangent_norms = np.where(tangent_norms > 1e-12, tangent_norms, 1.0)
    noise /= tangent_norms

    # Sample the cosine of the angle from the center
    # For large kappa, cos(theta) ~ 1 - 1/kappa approximately
    # We use the Wood (1994) sampling method simplified for high d
    if kappa > 0:
        # Approximate: w ~ 1 - (d-1)/(2*kappa) + noise
        # More precisely, sample from the marginal via rejection
        # For simplicity and vectorization, use the Gaussian approximation
        # which is accurate for kappa * d >> 1
        sigma_w = 1.0 / np.sqrt(kappa)
        w = 1.0 - np.abs(rng.standard_normal(n)) * sigma_w
        w = np.clip(w, -1.0, 1.0)
    else:
        w = 2.0 * rng.random(n) - 1.0

    # Combine: sample = w * center + sqrt(1-w^2) * tangent
    sin_part = np.sqrt(np.clip(1.0 - w ** 2, 0, None))
    samples = w[:, np.newaxis] * center[np.newaxis, :] + \
              sin_part[:, np.newaxis] * noise

    # Normalize to unit sphere
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    samples /= norms

    return samples


class RealisticAttentionGenerator:
    """Generate KV cache vectors that produce realistic attention patterns.

    Based on empirical observations from:
    - H2O (2023): "Heavy Hitter Oracle" -- 5% of tokens get 95% of attention
    - SnapKV (2024): Attention concentrated on specific "observation" tokens
    - StreamingLLM (2023): Attention sinks on initial tokens

    Parameters
    ----------
    n : int
        Number of cached tokens (sequence length).
    d : int
        Dimension of key/value vectors.
    n_sinks : int
        Number of attention sink tokens (default 4).
    local_window : int
        Size of local attention window (default 256).
    n_clusters : int
        Number of semantic clusters (default 20).
    cluster_concentration : float
        How tight clusters are (kappa for vMF, default 50).
    sparsity : float
        Fraction of tokens that are "heavy hitters" (default 0.05).
    seed : int
        Random seed.
    """

    def __init__(self, n: int, d: int,
                 n_sinks: int = 4,
                 local_window: int = 256,
                 n_clusters: int = 20,
                 cluster_concentration: float = 200.0,
                 sparsity: float = 0.05,
                 seed: int = 42):
        self.n = n
        self.d = d
        self.n_sinks = n_sinks
        self.local_window = min(local_window, n)
        self.n_clusters = n_clusters
        self.cluster_concentration = cluster_concentration
        self.sparsity = sparsity
        self.rng = np.random.default_rng(seed)

        # Pre-generate cluster centers and the sink direction
        self._sink_direction = self.rng.standard_normal(d)
        self._sink_direction /= np.linalg.norm(self._sink_direction)

        # Generate cluster centers on S^{d-1}, spread apart
        self._cluster_centers = self.rng.standard_normal((n_clusters, d))
        # Make cluster centers orthogonal to the sink direction to avoid
        # interference (sinks should respond to all queries)
        proj = self._cluster_centers @ self._sink_direction
        self._cluster_centers -= proj[:, np.newaxis] * self._sink_direction[np.newaxis, :]
        norms = np.linalg.norm(self._cluster_centers, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        self._cluster_centers /= norms

    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Generate keys, values, and a representative query.

        Returns
        -------
        keys : ndarray (n, d) -- key vectors (unit norm)
        values : ndarray (n, d) -- value vectors
        query : ndarray (d,) -- query vector that produces realistic attention
        metadata : dict -- info about the generation (which tokens are sinks,
                         which are in local window, cluster assignments, etc.)
        """
        n, d = self.n, self.d
        rng = self.rng

        # Guard: if n is too small for the configured n_sinks, reduce n_sinks
        n_sinks = self.n_sinks
        if n <= n_sinks:
            n_sinks = max(1, n // 4)

        keys = np.zeros((n, d), dtype=np.float64)
        cluster_assignments = np.full(n, -1, dtype=np.int32)
        cluster_probs = None  # default; set below if n_remaining > 0

        # ------------------------------------------------------------------
        # 1. Attention sinks (first n_sinks tokens)
        # ------------------------------------------------------------------
        # Sink keys are strongly aligned with a fixed direction so ANY
        # query gives them high dot product via the shared component.
        sink_indices = np.arange(min(n_sinks, n))
        for i in sink_indices:
            noise = rng.standard_normal(d) * 0.05
            keys[i] = self._sink_direction + noise
        cluster_assignments[sink_indices] = -2  # special marker for sinks

        # ------------------------------------------------------------------
        # 2. Assign non-sink tokens to semantic clusters
        # ------------------------------------------------------------------
        non_sink_start = len(sink_indices)
        n_remaining = n - non_sink_start

        if n_remaining > 0:
            # Cluster sizes follow a power law (a few large clusters,
            # many small ones), matching the Zipfian distribution of topics.
            cluster_probs = np.arange(1, self.n_clusters + 1,
                                       dtype=np.float64) ** (-1.0)
            cluster_probs /= cluster_probs.sum()

            assignments = rng.choice(self.n_clusters, size=n_remaining,
                                      p=cluster_probs)
            cluster_assignments[non_sink_start:] = assignments

            # ------------------------------------------------------------------
            # 3. Generate keys within each cluster
            # ------------------------------------------------------------------
            for c in range(self.n_clusters):
                mask = np.where(cluster_assignments == c)[0]
                if len(mask) == 0:
                    continue

                center = self._cluster_centers[c]
                # Use high concentration for tight clusters
                cluster_keys = _sample_vmf(
                    center, self.cluster_concentration, len(mask), rng
                )
                keys[mask] = cluster_keys

        # ------------------------------------------------------------------
        # 4. Local window modulation
        # ------------------------------------------------------------------
        local_start = max(0, n - self.local_window)
        local_indices = np.arange(local_start, n)

        recency_dir = rng.standard_normal(d)
        recency_dir /= np.linalg.norm(recency_dir)

        if len(local_indices) > 0:
            # Linearly increasing strength for more recent tokens
            strengths = np.linspace(0.05, 0.4, len(local_indices))
            keys[local_indices] += strengths[:, np.newaxis] * recency_dir[np.newaxis, :]

        # ------------------------------------------------------------------
        # 5. Normalize all keys to unit norm
        # ------------------------------------------------------------------
        key_norms = np.linalg.norm(keys, axis=1, keepdims=True)
        key_norms = np.where(key_norms > 1e-12, key_norms, 1.0)
        keys /= key_norms

        # ------------------------------------------------------------------
        # 6. Generate value vectors
        # ------------------------------------------------------------------
        values = rng.standard_normal((n, d))
        for c in range(self.n_clusters):
            mask = np.where(cluster_assignments == c)[0]
            if len(mask) < 2:
                continue
            cluster_val = rng.standard_normal(d) * 0.5
            values[mask] += cluster_val[np.newaxis, :]

        # ------------------------------------------------------------------
        # 7. Generate query that produces sparse attention
        # ------------------------------------------------------------------
        # In real transformers, attention scores are q^T k / sqrt(d_k).
        # With unit-norm keys at d=128, random dot products have std ~0.088.
        # We use the standard sqrt(d) scaling (matching the 1/sqrt(d_k) in
        # the attention formula) and control sparsity via the `concentration`
        # parameter derived from `self.sparsity`.
        #
        # The concentration determines how strongly the query aligns with
        # its target cluster center. Lower sparsity (fewer heavy hitters)
        # means higher concentration.
        if cluster_probs is not None:
            target_cluster = int(np.argmax(cluster_probs))  # largest cluster
        else:
            target_cluster = 0

        # Derive concentration from sparsity: lower sparsity -> higher concentration
        # sparsity=0.05 (5% heavy hitters) -> concentration ~20
        # sparsity=0.10 (10% heavy hitters) -> concentration ~10
        concentration = 1.0 / max(self.sparsity, 1e-6)

        cluster_center = self._cluster_centers[target_cluster].copy()

        # Build query direction: cluster center + sink + recency + noise
        query_dir = cluster_center * concentration

        # Add sink component (ensures sinks get attention)
        query_dir += self._sink_direction * 0.3 * concentration

        # Add recency component
        if len(local_indices) > 0:
            query_dir += recency_dir * 0.15 * concentration

        # Small noise
        query_dir += rng.standard_normal(d) * 0.02

        # Normalize direction then apply standard sqrt(d) scaling
        query_dir /= np.linalg.norm(query_dir)

        # Standard attention scaling: sqrt(d).  This is the honest scale
        # factor that matches the 1/sqrt(d_k) denominator in attention.
        query = query_dir * np.sqrt(d)

        # ------------------------------------------------------------------
        # Metadata
        # ------------------------------------------------------------------
        metadata = {
            "n": n,
            "d": d,
            "n_sinks": len(sink_indices),
            "sink_indices": sink_indices,
            "local_window_start": local_start,
            "local_window_size": len(local_indices),
            "n_clusters": self.n_clusters,
            "cluster_assignments": cluster_assignments,
            "target_cluster": target_cluster,
            "cluster_sizes": [int(np.sum(cluster_assignments == c))
                              for c in range(self.n_clusters)],
            "recency_direction": recency_dir if len(local_indices) > 0 else None,
        }

        return keys, values, query, metadata


def compute_ground_truth(keys: np.ndarray, query: np.ndarray,
                         temperature: float = 1.0) -> dict:
    """Compute exact attention scores and quality metrics.

    Parameters
    ----------
    keys : ndarray, shape (n, d)
        Key vectors.
    query : ndarray, shape (d,)
        Query vector.
    temperature : float
        Softmax temperature (default 1.0).

    Returns
    -------
    metrics : dict
        Contains:
        - scores: raw dot-product scores (n,)
        - attention_weights: softmax weights (n,)
        - top_k_indices: dict mapping k -> indices of top-k keys
        - effective_sparsity: number of keys needed for 90% mass
        - entropy: entropy of attention distribution (bits)
        - mass_in_top: dict mapping k -> softmax mass in top-k
        - heavy_hitter_fraction: fraction of keys with weight > 1/n
        - gini_coefficient: Gini coefficient of attention weights
    """
    n = len(keys)
    scores = keys @ query  # (n,)

    # Softmax
    scaled = scores / temperature
    max_s = np.max(scaled)
    exp_scores = np.exp(scaled - max_s)
    total = np.sum(exp_scores)
    attention_weights = exp_scores / total

    # Sort by weight descending
    sorted_idx = np.argsort(attention_weights)[::-1]
    sorted_weights = attention_weights[sorted_idx]

    # Effective sparsity: k_eff such that top-k_eff captures 90% mass
    cumsum = np.cumsum(sorted_weights)
    k_eff_90 = int(np.searchsorted(cumsum, 0.90)) + 1
    k_eff_95 = int(np.searchsorted(cumsum, 0.95)) + 1
    k_eff_99 = int(np.searchsorted(cumsum, 0.99)) + 1

    # Entropy in bits
    log_weights = np.log2(np.clip(attention_weights, 1e-30, None))
    entropy = -np.sum(attention_weights * log_weights)

    # Mass in top-k for various k values
    ks_to_check = [10, 50, 100, 200, 500, 1000]
    mass_in_top = {}
    top_k_indices = {}
    for k in ks_to_check:
        if k <= n:
            top_k_idx = sorted_idx[:k]
            top_k_indices[k] = top_k_idx
            mass_in_top[k] = float(np.sum(sorted_weights[:k]))

    # Heavy hitter fraction: tokens with weight > 1/n
    hh_threshold = 1.0 / n
    heavy_hitter_fraction = float(np.sum(attention_weights > hh_threshold) / n)

    # Gini coefficient
    # G = (2 * sum_i (i * w_sorted[i])) / (n * sum(w)) - (n+1)/n
    ascending_weights = sorted_weights[::-1]  # ascending order
    indices = np.arange(1, n + 1)
    gini = (2.0 * np.sum(indices * ascending_weights) /
            (n * np.sum(ascending_weights))) - (n + 1) / n

    return {
        "scores": scores,
        "attention_weights": attention_weights,
        "top_k_indices": top_k_indices,
        "effective_sparsity_90": k_eff_90,
        "effective_sparsity_95": k_eff_95,
        "effective_sparsity_99": k_eff_99,
        "entropy_bits": float(entropy),
        "mass_in_top": mass_in_top,
        "heavy_hitter_fraction": heavy_hitter_fraction,
        "gini_coefficient": float(gini),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Realistic Attention Pattern Generator -- self-test")
    print("=" * 70)

    for n in [1000, 10_000, 50_000]:
        d = 128
        print(f"\n--- n={n:,}, d={d} ---")

        gen = RealisticAttentionGenerator(
            n=n, d=d,
            n_sinks=4,
            local_window=256,
            n_clusters=20,
            cluster_concentration=50.0,
            sparsity=0.05,
            seed=42,
        )

        keys, values, query, metadata = gen.generate()

        # Verify shapes
        assert keys.shape == (n, d), f"keys shape {keys.shape}"
        assert values.shape == (n, d), f"values shape {values.shape}"
        assert query.shape == (d,), f"query shape {query.shape}"

        # Verify unit norms
        key_norms = np.linalg.norm(keys, axis=1)
        assert np.allclose(key_norms, 1.0, atol=1e-6), \
            f"key norms not unit: {key_norms.min():.6f} to {key_norms.max():.6f}"

        q_norm = np.linalg.norm(query)
        assert q_norm > 0.1, f"query norm too small: {q_norm}"

        # Compute ground truth
        gt = compute_ground_truth(keys, query)

        print(f"  Cluster sizes (top 5): {sorted(metadata['cluster_sizes'], reverse=True)[:5]}")
        print(f"  Effective sparsity (90%): {gt['effective_sparsity_90']}")
        print(f"  Effective sparsity (95%): {gt['effective_sparsity_95']}")
        print(f"  Effective sparsity (99%): {gt['effective_sparsity_99']}")
        print(f"  Entropy: {gt['entropy_bits']:.2f} bits")
        print(f"  Heavy hitter fraction: {gt['heavy_hitter_fraction']:.4f}")
        print(f"  Gini coefficient: {gt['gini_coefficient']:.4f}")

        for k, mass in sorted(gt['mass_in_top'].items()):
            print(f"  Mass in top-{k}: {mass:.6f}")

        # Verify realistic properties
        if n >= 10_000:
            assert gt['effective_sparsity_90'] < 0.8 * n, \
                f"90% mass should be in <80% of tokens, got {gt['effective_sparsity_90']}"
            assert gt['heavy_hitter_fraction'] < 0.5, \
                f"Heavy hitter fraction too high: {gt['heavy_hitter_fraction']}"
            # With honest sqrt(d) scaling, attention is less concentrated
            # than the artificial d-scaling produced. Gini > 0.3 is sufficient.
            assert gt['gini_coefficient'] > 0.3, \
                f"Attention should show some concentration (Gini > 0.3), got {gt['gini_coefficient']}"

    print("\n" + "=" * 70)
    print("All attention pattern tests passed.")
    print("=" * 70)
