"""QuantDex: Vector quantization as a search index for sub-linear attention.

This library proves that TurboQuant-style data-oblivious vector quantization
simultaneously creates a search index enabling sub-linear attention — the
quantization codes *are* the index.

Core components
---------------
TurboQuant
    Near-optimal data-oblivious vector quantizer via Randomized Hadamard
    Transform + Lloyd-Max scalar quantization.

CoarseToFine
    Multi-round progressive attention using coordinate importance ordering.

CodeTrie
    Branch-and-bound attention on a hierarchical trie of quantization codes.

BlockPruning
    GPU-friendly block summary table with upper-bound pruning.

Metrics
    recall_at_k, softmax_mass_captured, variance_fraction, and more.
"""

from .turbo_quant import TurboQuant, fwht, fwht_batch
from .sub_linear_attention import CoarseToFine, CodeTrie, BlockPruning
from .metrics import (
    recall_at_k,
    attention_output_error,
    softmax_mass_captured,
    variance_fraction,
    precision_at_k,
    ndcg_at_k,
    mse_reconstruction,
    cosine_similarity_preserved,
    inner_product_correlation,
    speedup_ratio,
)
from .attention_patterns import RealisticAttentionGenerator, compute_ground_truth

__all__ = [
    # Quantizer
    "TurboQuant",
    "fwht",
    "fwht_batch",
    # Attention algorithms
    "CoarseToFine",
    "CodeTrie",
    "BlockPruning",
    # Attention patterns
    "RealisticAttentionGenerator",
    "compute_ground_truth",
    # Metrics
    "recall_at_k",
    "attention_output_error",
    "softmax_mass_captured",
    "variance_fraction",
    "precision_at_k",
    "ndcg_at_k",
    "mse_reconstruction",
    "cosine_similarity_preserved",
    "inner_product_correlation",
    "speedup_ratio",
]

__version__ = "0.1.0"
