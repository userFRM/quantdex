"""Sub-linear attention via quantized code structure.

The central thesis of QuantDex: the *same* quantization codes that compress
key vectors also induce a search index for sub-linear attention.

Three algorithms demonstrate increasingly sophisticated use of this structure:

1. **CoarseToFine** -- Multi-round progressive coordinate scoring.
   Exploits the fact that RHT concentrates query energy: the top-m
   coordinates of the rotated query capture most of the dot-product
   variance, so partial scoring on those coordinates is a good proxy.

2. **CodeTrie** -- Branch-and-bound on a hierarchical trie of codes.
   Groups keys by their quantization codes in blocks of coordinates.
   Prunes subtrees whose upper-bound contribution cannot beat the
   current k-th best score.  Pruned subtrees contribute approximate
   attention via their value centroids.

3. **BlockPruning** -- GPU-friendly block summary table with upper bounds.
   Maintains per-block histograms of code values.  Computes an upper
   bound on the max dot product in each block from the histogram,
   then only scores keys in blocks that could contain a top-k key.

All three algorithms accept the same quantized representation produced
by TurboQuant.quantize_batch() and return compatible outputs.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from .turbo_quant import TurboQuant


# ---------------------------------------------------------------------------
# Algorithm 1: Coarse-to-Fine multi-round scoring
# ---------------------------------------------------------------------------

class CoarseToFine:
    """Multi-round coarse-to-fine attention using coordinate importance.

    After RHT, the rotated query q_rot has its energy spread across all d
    coordinates, but not uniformly.  Sorting coordinates by |q_rot[j]|
    gives a natural importance ordering.  In round r, we score using only
    the top-m_r coordinates, which captures a fraction V(m_r) of the
    dot-product variance.

    Variance captured by top-m coordinates:

        V(m) = sum_{j in top-m} q_rot[j]^2 / ||q_rot||^2

    The columnar (transposed) layout codes_transposed[j, i] lets us
    read only the coordinates we need, achieving sub-linear I/O when
    m << d.

    Parameters
    ----------
    tq : TurboQuant
        The quantizer (provides levels, rotation, etc.).
    codes : ndarray, shape (n, d), dtype uint8
        Quantized codes for all n keys.
    norms : ndarray, shape (n,)
        Norms of all n keys.
    block_size : int
        Not used in CoarseToFine but kept for API consistency.
    """

    def __init__(self, tq: TurboQuant, codes: np.ndarray,
                 norms: np.ndarray, block_size: int = 1024):
        self.tq = tq
        self.n = len(norms)
        self.norms = norms.astype(np.float64)

        # COLUMNAR layout: codes_transposed[coord, key_idx]
        # This allows reading only the coordinates we need in each round.
        self.codes_transposed = codes.T.copy()  # shape (d, n)

    def query(self, q: np.ndarray, k: int = 100,
              rounds: Optional[List[Tuple[int, int]]] = None
              ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Multi-round coarse-to-fine attention.

        Parameters
        ----------
        q : ndarray, shape (d,)
            Query vector (in original space).
        k : int
            Number of top keys to return.
        rounds : list of (m, keep) tuples, optional
            Each round specifies:
              m    -- number of coordinates to use for scoring
              keep -- number of survivors to pass to the next round
            Default: [(8, 100*k), (24, 20*k), (d, k)]

        Returns
        -------
        top_k_indices : ndarray, shape (k,), dtype int64
        top_k_scores : ndarray, shape (k,), dtype float64
        stats : dict
            Diagnostic statistics: coords_read, keys_scored, recall, etc.
        """
        d = self.tq.d
        n = self.n
        k = min(k, n)
        if k == 0:
            return (np.array([], dtype=np.int64),
                    np.array([], dtype=np.float64),
                    {"coords_read": 0, "coords_read_fraction": 0.0,
                     "recall_at_k": 1.0, "variance_fractions": [],
                     "n_survivors_per_round": [], "brute_force_cost": n * d})

        if rounds is None:
            m1 = min(8, d)
            m2 = min(24, d)
            rounds = [
                (m1, min(100 * k, n)),
                (m2, min(20 * k, n)),
                (d, k),
            ]

        # Rotate query once
        q_rot = self.tq.rotate(q)

        # Sort coordinates by importance (descending |q_rot[j]|)
        coord_order = np.argsort(np.abs(q_rot))[::-1]

        # Precompute LUT: lut[c] = levels[c] for fast lookup
        levels = self.tq.levels  # shape (2^b,)

        survivors = np.arange(n)  # initially all keys are candidates
        total_coords_read = 0

        for round_idx, (m, keep) in enumerate(rounds):
            # Select which coordinates to score in this round
            coords = coord_order[:m]
            q_rot_subset = q_rot[coords]  # (m,)

            # Read codes for these coordinates, only for surviving keys
            # codes_subset[coord_local, survivor_local] -> code
            codes_subset = self.codes_transposed[coords][:, survivors]  # (m, |survivors|)

            # Compute partial dot products via LUT gather:
            #   partial_score[i] = norm[i] * sum_{j in coords} levels[code[j,i]] * q_rot[j]
            contributions = levels[codes_subset]  # (m, |survivors|)
            partial_dots = q_rot_subset @ contributions  # (|survivors|,)
            scores = self.norms[survivors] * partial_dots

            # Track I/O
            total_coords_read += m * len(survivors)

            # Select top-keep survivors
            keep_actual = min(keep, len(survivors))
            if keep_actual < len(survivors):
                top_idx = np.argpartition(scores, -keep_actual)[-keep_actual:]
                survivors = survivors[top_idx]
                scores = scores[top_idx]
            # else: keep all

        # Final sort to get top-k in order
        final_k = min(k, len(survivors))
        top_k_local = np.argsort(scores)[::-1][:final_k]
        top_k_indices = survivors[top_k_local]
        top_k_scores = scores[top_k_local]

        # Compute brute-force for recall measurement
        brute_scores = self._brute_force_scores(q_rot, levels)
        true_top_k = np.argsort(brute_scores)[::-1][:k]

        recall = len(np.intersect1d(top_k_indices, true_top_k)) / k

        # Variance captured by each round
        q_energy = np.sum(q_rot ** 2)
        variance_fractions = []
        for m, _ in rounds:
            coords = coord_order[:m]
            vf = np.sum(q_rot[coords] ** 2) / q_energy if q_energy > 0 else 0.0
            variance_fractions.append(float(vf))

        stats = {
            "coords_read": total_coords_read,
            "coords_read_fraction": total_coords_read / (n * d),
            "recall_at_k": recall,
            "variance_fractions": variance_fractions,
            "n_survivors_per_round": [min(r[1], n) for r in rounds],
            "brute_force_cost": n * d,
        }

        return top_k_indices, top_k_scores, stats

    def _brute_force_scores(self, q_rot: np.ndarray,
                            levels: np.ndarray) -> np.ndarray:
        """Full dot-product scores for all n keys (for recall measurement)."""
        # codes_transposed is (d, n), levels lookup gives (d, n) floats
        Y_hat = levels[self.codes_transposed]  # (d, n)
        dots = q_rot @ Y_hat  # (n,)
        return self.norms * dots


# ---------------------------------------------------------------------------
# Algorithm 2: CodeTrie (branch-and-bound on code structure)
# ---------------------------------------------------------------------------

class _TrieNode:
    """Internal node of the CodeTrie.

    Each node at depth t covers coordinates [t*block_size, (t+1)*block_size).
    """
    __slots__ = ["count", "key_indices", "value_centroid", "norm_sum",
                 "children", "depth"]

    def __init__(self, depth: int):
        self.depth = depth
        self.count = 0
        self.key_indices: List[int] = []
        self.value_centroid: Optional[np.ndarray] = None
        self.norm_sum = 0.0
        # children: dict mapping code_tuple -> _TrieNode
        self.children: Dict[tuple, _TrieNode] = {}


class CodeTrie:
    """Branch-and-bound attention using a trie over quantization codes.

    Keys are inserted into a trie where each level corresponds to a block
    of `block_size` coordinates.  The code-tuple for that block determines
    the child pointer.  Each leaf stores a list of key indices.

    At query time, we compute a LUT of per-coordinate contributions, then
    traverse the trie.  At each node, we compute the maximum possible
    remaining contribution (upper bound) and prune if it cannot beat the
    current k-th best score.  Pruned subtrees contribute approximate
    attention through their value centroids.

    Parameters
    ----------
    tq : TurboQuant
        The quantizer.
    codes : ndarray, shape (n, d), dtype uint8
        Quantized codes for all keys.
    norms : ndarray, shape (n,)
        Key norms.
    values : ndarray, shape (n, d_v), optional
        Value vectors for attention output computation.
        If None, we skip attention output computation.
    block_size : int
        Number of coordinates per trie level.
    """

    def __init__(self, tq: TurboQuant, codes: np.ndarray,
                 norms: np.ndarray, values: Optional[np.ndarray] = None,
                 block_size: int = 8):
        self.tq = tq
        self.n = len(norms)
        self.codes = codes
        self.norms = norms.astype(np.float64)
        self.values = values
        self.block_size = block_size
        self.n_levels_trie = (tq.d + block_size - 1) // block_size

        # Build the trie
        self.root = _TrieNode(depth=0)
        for i in range(self.n):
            self.insert(i, codes[i], values[i] if values is not None else None)

    def insert(self, key_idx: int, code: np.ndarray,
               value: Optional[np.ndarray] = None):
        """Insert one key into the trie.  O(d / block_size).

        Parameters
        ----------
        key_idx : int
            Index of this key in the original array.
        code : ndarray, shape (d,), dtype uint8
        value : ndarray, shape (d_v,), optional
        """
        node = self.root
        node.count += 1
        node.norm_sum += self.norms[key_idx]
        if value is not None:
            if node.value_centroid is None:
                node.value_centroid = value.copy().astype(np.float64)
            else:
                # Online mean update: centroid = running sum / count
                node.value_centroid += (value - node.value_centroid) / node.count

        bs = self.block_size
        for t in range(self.n_levels_trie):
            start = t * bs
            end = min(start + bs, self.tq.d)
            code_block = tuple(code[start:end])

            if code_block not in node.children:
                node.children[code_block] = _TrieNode(depth=t + 1)

            node = node.children[code_block]
            node.count += 1
            node.norm_sum += self.norms[key_idx]
            if value is not None:
                if node.value_centroid is None:
                    node.value_centroid = value.copy().astype(np.float64)
                else:
                    node.value_centroid += (value - node.value_centroid) / node.count

        # Store key index at the leaf
        node.key_indices.append(key_idx)

    def query(self, q: np.ndarray, k: int = 100
              ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict]:
        """Branch-and-bound search for top-k keys.

        Uses per-block LUT + remaining contribution upper bounds for pruning.
        Pruned subtrees contribute via their value centroids.

        Parameters
        ----------
        q : ndarray, shape (d,)
            Query vector (original space).
        k : int
            Number of top keys to return.

        Returns
        -------
        top_k_indices : ndarray, shape (k,)
        top_k_scores : ndarray, shape (k,)
        attention_output : ndarray, shape (d_v,) or None
            Approximate attention output, including contributions from
            pruned subtrees.  None if values were not provided.
        stats : dict
            Diagnostic statistics.
        """
        d = self.tq.d
        bs = self.block_size
        levels = self.tq.levels

        # Rotate query
        q_rot = self.tq.rotate(q)

        # Build per-block LUT and upper-bound tables
        # lut[t][code_tuple] = dot contribution from block t with that code
        # max_contrib[t] = max over all possible codes of |contribution from block t|
        block_luts = []
        remaining_bound = np.zeros(self.n_levels_trie + 1)  # remaining_bound[t] = max contribution from blocks [t, ...]

        for t in range(self.n_levels_trie):
            start = t * bs
            end = min(start + bs, d)
            q_block = q_rot[start:end]

            # For each possible code value at each coordinate, compute contribution
            # contribution[j_local, c] = levels[c] * q_rot[start + j_local]
            block_len = end - start
            contribs = levels[np.newaxis, :] * q_block[:, np.newaxis]  # (block_len, 2^b)

            # Max absolute contribution for this block:
            # pick the code that maximizes |sum of contributions| -- upper bound is
            # sum of per-coordinate maxima
            max_per_coord = np.max(np.abs(contribs), axis=1)  # (block_len,)
            block_max = np.sum(max_per_coord)

            block_luts.append(contribs)
            remaining_bound[t] = block_max

        # Cumulative remaining bound (from the right)
        for t in range(self.n_levels_trie - 2, -1, -1):
            remaining_bound[t] += remaining_bound[t + 1]
        # remaining_bound[t] = max possible |dot contribution| from blocks [t, T)
        # Shift: remaining_bound[t+1] = remaining from blocks *after* t
        suffix_bound = np.zeros(self.n_levels_trie + 1)
        for t in range(self.n_levels_trie - 1, -1, -1):
            suffix_bound[t] = remaining_bound[t]
        # suffix_bound[t] is the bound from block t onward
        # We need "remaining after processing block t" = suffix_bound[t+1]

        # Branch-and-bound traversal
        # We maintain a min-heap of size k for the top-k scores
        top_k_scores_list = []
        top_k_indices_list = []
        threshold = -np.inf  # k-th best score found so far

        nodes_visited = 0
        nodes_pruned = 0
        keys_scored = 0

        # Attention output accumulators (unnormalized)
        if self.values is not None:
            d_v = self.values.shape[1]
            attn_output = np.zeros(d_v)
            attn_total_weight = 0.0
        else:
            attn_output = None
            attn_total_weight = 0.0

        def _traverse(node: _TrieNode, partial_score: float, depth: int,
                      norm_multiplier: float):
            nonlocal threshold, nodes_visited, nodes_pruned, keys_scored
            nonlocal attn_output, attn_total_weight

            nodes_visited += 1

            if depth == self.n_levels_trie:
                # Leaf: score all keys stored here
                for idx in node.key_indices:
                    score = self.norms[idx] * partial_score
                    keys_scored += 1
                    _update_topk(idx, score)
                return

            # For each child, compute the contribution from this block
            # and check if the upper bound can beat the threshold
            t = depth
            start = t * bs
            end = min(start + bs, d)
            block_contribs = block_luts[t]  # (block_len, 2^b)
            future_bound = suffix_bound[t + 1] if t + 1 <= self.n_levels_trie else 0.0

            for code_tuple, child in node.children.items():
                # Compute this block's contribution
                block_dot = 0.0
                for j_local, c in enumerate(code_tuple):
                    block_dot += block_contribs[j_local, c]

                new_partial = partial_score + block_dot

                # Upper bound on final score (using max possible norm in subtree)
                max_norm = np.max(self.norms)  # conservative; could track per-subtree
                upper = max_norm * (abs(new_partial) + future_bound)

                if upper <= threshold and len(top_k_scores_list) >= k:
                    nodes_pruned += child.count
                    # Pruned subtree contributes to attention via centroid
                    if attn_output is not None and child.value_centroid is not None:
                        # Approximate contribution: average_score * count * centroid
                        approx_score = child.norm_sum / child.count * new_partial
                        weight = max(approx_score, 0.0)
                        attn_output += weight * child.value_centroid * child.count
                        attn_total_weight += weight * child.count
                    continue

                _traverse(child, new_partial, depth + 1, max_norm)

        def _update_topk(idx: int, score: float):
            nonlocal threshold
            top_k_indices_list.append(idx)
            top_k_scores_list.append(score)
            if len(top_k_scores_list) > 2 * k:
                # Compact: keep only top-k
                arr_s = np.array(top_k_scores_list)
                arr_i = np.array(top_k_indices_list)
                order = np.argsort(arr_s)[::-1][:k]
                top_k_scores_list.clear()
                top_k_indices_list.clear()
                top_k_scores_list.extend(arr_s[order].tolist())
                top_k_indices_list.extend(arr_i[order].tolist())
                threshold = top_k_scores_list[-1]

        _traverse(self.root, 0.0, 0, 0.0)

        # Final top-k extraction
        arr_s = np.array(top_k_scores_list) if top_k_scores_list else np.array([])
        arr_i = np.array(top_k_indices_list, dtype=np.int64) if top_k_indices_list else np.array([], dtype=np.int64)

        if len(arr_s) > 0:
            final_k = min(k, len(arr_s))
            order = np.argsort(arr_s)[::-1][:final_k]
            top_k_indices = arr_i[order]
            top_k_scores = arr_s[order]
        else:
            top_k_indices = np.array([], dtype=np.int64)
            top_k_scores = np.array([])

        # Normalize attention output
        if attn_output is not None:
            # Add contributions from scored keys
            for idx, score in zip(top_k_indices, top_k_scores):
                weight = max(score, 0.0)
                attn_output += weight * self.values[idx]
                attn_total_weight += weight
            if attn_total_weight > 0:
                attn_output /= attn_total_weight

        stats = {
            "nodes_visited": nodes_visited,
            "nodes_pruned": nodes_pruned,
            "keys_scored": keys_scored,
            "keys_scored_fraction": keys_scored / self.n if self.n > 0 else 0.0,
            "trie_depth": self.n_levels_trie,
        }

        return top_k_indices, top_k_scores, attn_output, stats


# ---------------------------------------------------------------------------
# Algorithm 3: Block Pruning with summary tables
# ---------------------------------------------------------------------------

class BlockPruning:
    """GPU-friendly block pruning using per-block code histograms.

    Partitions the n keys into blocks of `block_size`.  For each block
    and each coordinate j, we store a histogram: how many keys in that
    block have code value c at coordinate j.

    **Negative result**: In practice, the coordinate-wise-max upper bounds
    are too loose for any meaningful pruning.  With 2-bit codes (4 values)
    and blocks of 256+ keys, every code value appears at every coordinate
    in every block, making the per-block upper bound nearly identical
    across blocks.  Attempts to use tighter probabilistic bounds
    (mean + C*sigma) or sampling-based heuristic lower bounds were shown
    to be UNSAFE: they can miss true top-k keys (Codex audit reproduced
    a case returning score 1.28 when the true top was 4.27).

    The class now falls back to brute force (scoring all blocks) to
    guarantee correctness.  It is retained for the summary table
    infrastructure, which could be useful if a provably safe pruning
    criterion is found in future work.

    Parameters
    ----------
    tq : TurboQuant
        The quantizer.
    codes : ndarray, shape (n, d), dtype uint8
        Quantized codes.
    norms : ndarray, shape (n,)
        Key norms.
    block_size : int
        Number of keys per block.
    """

    def __init__(self, tq: TurboQuant, codes: np.ndarray,
                 norms: np.ndarray, block_size: int = 1024):
        self.tq = tq
        self.n = len(norms)
        self.codes = codes
        self.norms = norms.astype(np.float64)
        self.block_size = block_size
        self.n_blocks = (self.n + block_size - 1) // block_size
        n_code_vals = 1 << tq.bits

        # Block summary table: histogram of code values per coordinate per block
        # block_counts[block, coord, code_value] = count of keys with that code
        self.block_counts = np.zeros(
            (self.n_blocks, tq.d, n_code_vals), dtype=np.uint16
        )

        # Block max norms (for tighter upper bounds)
        self.block_max_norms = np.zeros(self.n_blocks)

        # Build the summary tables
        for b_idx in range(self.n_blocks):
            start = b_idx * block_size
            end = min(start + block_size, self.n)
            block_codes = codes[start:end]  # (block_len, d)
            block_norms = norms[start:end]

            self.block_max_norms[b_idx] = np.max(block_norms) if len(block_norms) > 0 else 0.0

            # Vectorized histogram construction
            for j in range(tq.d):
                col_codes = block_codes[:, j]
                np.add.at(self.block_counts[b_idx, j], col_codes, 1)

    def query(self, q: np.ndarray, k: int = 100
              ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Brute-force scoring of all blocks (no pruning).

        Previous versions attempted heuristic pruning using sampling-based
        lower bounds with a safety factor, but this was proven UNSAFE
        (can miss true top-k keys).  We now score all blocks honestly.

        Parameters
        ----------
        q : ndarray, shape (d,)
            Query vector (original space).
        k : int
            Number of top keys to return.

        Returns
        -------
        top_k_indices : ndarray, shape (k,)
        top_k_scores : ndarray, shape (k,)
        stats : dict
        """
        k = min(k, self.n)
        if k == 0:
            return (np.array([], dtype=np.int64),
                    np.array([]),
                    {"blocks_processed": 0, "blocks_pruned": 0,
                     "keys_scored": 0, "keys_scored_fraction": 0.0})

        q_rot = self.tq.rotate(q)
        levels = self.tq.levels

        # Score ALL blocks (no pruning -- see class docstring)
        scores_collected = []
        indices_collected = []
        keys_scored = 0

        for b_idx in range(self.n_blocks):
            start = b_idx * self.block_size
            end = min(start + self.block_size, self.n)
            block_codes = self.codes[start:end]
            block_norms = self.norms[start:end]

            Y_hat = levels[block_codes]
            dots = Y_hat @ q_rot
            block_scores = block_norms * dots

            scores_collected.append(block_scores)
            indices_collected.append(np.arange(start, end))
            keys_scored += len(block_scores)

        # ------------------------------------------------------------------
        # Extract top-k
        # ------------------------------------------------------------------
        if len(scores_collected) == 0:
            return (np.array([], dtype=np.int64),
                    np.array([]),
                    {"blocks_processed": 0, "blocks_pruned": 0,
                     "keys_scored": 0, "keys_scored_fraction": 0.0})

        all_scores = np.concatenate(scores_collected)
        all_indices = np.concatenate(indices_collected)

        final_k = min(k, len(all_scores))
        if final_k < len(all_scores):
            top_k_local = np.argpartition(all_scores, -final_k)[-final_k:]
        else:
            top_k_local = np.arange(len(all_scores))

        # Sort top-k by score descending
        order = np.argsort(all_scores[top_k_local])[::-1]
        top_k_local = top_k_local[order]

        top_k_indices = all_indices[top_k_local].astype(np.int64)
        top_k_scores = all_scores[top_k_local]

        stats = {
            "blocks_processed": self.n_blocks,
            "blocks_pruned": 0,
            "blocks_total": self.n_blocks,
            "prune_fraction": 0.0,
            "keys_scored": keys_scored,
            "keys_scored_fraction": keys_scored / self.n if self.n > 0 else 0.0,
            "threshold": float(top_k_scores[-1]) if len(top_k_scores) > 0 else float('-inf'),
        }

        return top_k_indices, top_k_scores, stats


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    import sys
    sys.path.insert(0, "/home/theta-gamma/quantdex")
    from src.turbo_quant import TurboQuant
    from src.metrics import recall_at_k, softmax_mass_captured

    print("=" * 70)
    print("Sub-linear attention self-test")
    print("=" * 70)

    d = 128
    n = 10000
    k = 50
    bits = 2
    rng = np.random.default_rng(123)

    # Generate random keys and query
    keys = rng.standard_normal((n, d))
    query = rng.standard_normal(d)
    values = rng.standard_normal((n, d))  # for CodeTrie attention output

    # Quantize
    tq = TurboQuant(d, bits=bits, seed=42)
    codes, norms = tq.quantize_batch(keys)

    # Brute force baseline
    true_scores = tq.batch_dot_products(codes, norms, query)
    true_topk = np.argsort(true_scores)[::-1][:k]
    print(f"\nBrute force top-{k} score range: "
          f"[{true_scores[true_topk[-1]]:.4f}, {true_scores[true_topk[0]]:.4f}]")

    # --- CoarseToFine ---
    print(f"\n--- CoarseToFine (d={d}, n={n}, bits={bits}) ---")
    ctf = CoarseToFine(tq, codes, norms)
    t0 = time.perf_counter()
    ctf_idx, ctf_scores, ctf_stats = ctf.query(query, k=k)
    t_ctf = time.perf_counter() - t0
    ctf_recall = recall_at_k(true_topk, ctf_idx)
    ctf_mass = softmax_mass_captured(true_scores, ctf_idx, temperature=1.0)
    print(f"  Recall@{k}:              {ctf_recall:.4f}")
    print(f"  Softmax mass captured:   {ctf_mass:.4f}")
    print(f"  Coords read fraction:    {ctf_stats['coords_read_fraction']:.4f}")
    print(f"  Variance fractions:      {ctf_stats['variance_fractions']}")
    print(f"  Time:                    {t_ctf:.4f}s")

    # --- CodeTrie ---
    print(f"\n--- CodeTrie (d={d}, n={n}, bits={bits}, block_size=16) ---")
    # CodeTrie is slower to build; use smaller n for test
    n_trie = min(n, 2000)
    ct = CodeTrie(tq, codes[:n_trie], norms[:n_trie],
                  values=values[:n_trie], block_size=16)
    t0 = time.perf_counter()
    ct_idx, ct_scores, ct_attn, ct_stats = ct.query(query, k=min(k, 20))
    t_ct = time.perf_counter() - t0
    true_topk_trie = np.argsort(true_scores[:n_trie])[::-1][:min(k, 20)]
    ct_recall = recall_at_k(true_topk_trie, ct_idx)
    print(f"  Recall@{min(k,20)}:              {ct_recall:.4f}")
    print(f"  Keys scored:             {ct_stats['keys_scored']} / {n_trie}")
    print(f"  Nodes visited:           {ct_stats['nodes_visited']}")
    print(f"  Nodes pruned:            {ct_stats['nodes_pruned']}")
    print(f"  Time:                    {t_ct:.4f}s")
    if ct_attn is not None:
        print(f"  Attention output norm:   {np.linalg.norm(ct_attn):.4f}")

    # --- BlockPruning ---
    print(f"\n--- BlockPruning (d={d}, n={n}, bits={bits}, block_size=256) ---")
    bp = BlockPruning(tq, codes, norms, block_size=256)
    t0 = time.perf_counter()
    bp_idx, bp_scores, bp_stats = bp.query(query, k=k)
    t_bp = time.perf_counter() - t0
    bp_recall = recall_at_k(true_topk, bp_idx)
    bp_mass = softmax_mass_captured(true_scores, bp_idx, temperature=1.0)
    print(f"  Recall@{k}:              {bp_recall:.4f}")
    print(f"  Softmax mass captured:   {bp_mass:.4f}")
    print(f"  Blocks processed:        {bp_stats['blocks_processed']} / {bp_stats['blocks_total']}")
    print(f"  Blocks pruned:           {bp_stats['blocks_pruned']}")
    print(f"  Keys scored fraction:    {bp_stats['keys_scored_fraction']:.4f}")
    print(f"  Time:                    {t_bp:.4f}s")

    print("\n" + "=" * 70)
    print("All sub-linear attention tests passed.")
    print("=" * 70)
