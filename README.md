# QuantDex

**Sub-Linear Attention via the Quantization-Indexing Duality**

*F. Miesegaes*

---

## TL;DR

Data-oblivious vector quantization (TurboQuant-style) compresses the KV cache **and** creates a search index — for free. We exploit this to skip most of the cache during attention, achieving **5.6x wall-clock GPU speedup at 99% recall** (or **4.9x at 100% recall**) on realistic LLM attention patterns.

## The Problem

Long-context LLM inference is bottlenecked by attention: every generated token must read the entire KV cache. At 1M tokens, that's ~22 GB of memory reads per token. TurboQuant (Google, April 2025) solved the *storage* problem (4-8x compression). We solve the *bandwidth* problem.

## The Insight

After TurboQuant's random rotation + scalar quantization, each key vector becomes a sequence of small integer codes. These codes, stored in columnar layout, **are already a bitmap index** — no separate index construction needed.

A query's rotated coordinates have varying magnitudes. Reading only the top-*m* coordinates (sorted by |q'_j|) captures a disproportionate share of the ranking signal. With just 13% of coordinates, you get 50% of the discriminative power.

## Key Results

| Metric | Value |
|---|---|
| GPU speedup (n=2M, d=128, m=32) | **5.62x at 99% recall** |
| GPU speedup (n=2M, d=128, m=16) | **4.89x at 100% recall** |
| GPU speedup (n=1M) | 3.91x at 100% recall |
| Speedup at d=512 | 4.95x at >=99% recall |
| Attention output error (C2F) | < 10⁻¹⁵ (machine precision) |
| Compression (TurboQuant, b=2) | 0.1161 MSE (within 1% of theory) |
| V(m) at m=17 (13% of d=128) | 50% variance captured |

Speedup grows with both sequence length *n* and head dimension *d* — it gets better precisely where long-context LLMs operate.

## Validated on Real LLM Attention

Extracted attention from TinyLlama-1.1B (22 layers, 4 heads, d=64, 370 tokens):

- **100% recall on all 48 tested heads** (including m=8, reading only 12.5% of coordinates)
- Real attention is **sparser** than synthetic: Gini up to 0.994, 90% mass in 0.3% of keys
- Deeper layers consistently show Gini > 0.9, confirming structured sparsity is fundamental

## Algorithms

1. **Coarse-to-Fine Scoring** — Read top-*m* coordinates for all keys, prune, full-score survivors. Primary algorithm, validated on GPU with fused CUDA kernels.

2. **CodeTrie** — Branch-and-bound on a hierarchical trie of quantization codes. Reads 58x fewer key coordinates than brute force with perfect recall.

3. **Block Pruning** — GPU-native block summary tables. Honest negative result: bounds too loose for effective pruning on our test distributions.

## Repository Structure

```
src/
  turbo_quant.py          # TurboQuant implementation (RHT + Lloyd-Max)
  sub_linear_attention.py # CoarseToFine, CodeTrie, BlockPruning
  attention_patterns.py   # Realistic LLM attention generator
  metrics.py              # Evaluation metrics

experiments/
  exp01-exp06             # CPU validation (quantization, V(m), recall, quality)
  exp07                   # Realistic attention patterns
  exp08                   # GPU benchmark (Python/CuPy baseline)
  exp09                   # Fused CUDA kernels (eliminates Python overhead)
  exp10                   # Definitive: fused CUDA + realistic attention
  exp11                   # Ablations (m, d, b)
  exp12                   # Real LLM attention extraction (TinyLlama)

paper/
  main.tex                # Full paper (18 pages, 6 theorems, 10 figures)
  Makefile                # pdflatex build

figures/                  # 28 publication-quality figures
FACTSHEET.md              # Non-technical summary and FAQ
```

## Quick Start

```bash
pip install -r requirements.txt              # CPU experiments
pip install -r requirements-gpu.txt          # GPU experiments (NVIDIA)
pip install -r requirements-llm.txt          # Real attention extraction

# Run core validation
python experiments/exp01_turbo_quant_validation.py

# Run the headline experiment (requires NVIDIA GPU + CuPy)
python experiments/exp10_definitive.py

# Build the paper (requires texlive)
cd paper && make
```

Tested with: Python 3.14, NumPy 2.4.2, CuPy 14.0.1, CUDA 12.0, RTX 3070

## How It Works

```
                       QUANTIZE (per token)
                       ═══════════════════
  key vector ──→ random rotation ──→ scalar quantize ──→ store codes + norm
                    (FWHT)           (Lloyd-Max)          (columnar layout)

                       ATTENTION (per query)
                       ════════════════════
  query ──→ rotate ──→ sort coords by |q'_j| ──→ read top-m columns
                                                       │
                                          ┌─────────────┘
                                          ▼
                                    partial scores
                                          │
                                    prune (keep top candidates)
                                          │
                                    full-score survivors
                                          │
                                    softmax + weighted sum ──→ output
```

The key insight: **steps 1-3 of quantization are the same as steps 1-3 of indexing.** The random rotation that makes scalar quantization near-optimal also makes coordinates exchangeable, enabling progressive refinement. The compression IS the index.

## Paper

The paper is in `paper/main.tex` (18 pages). Build with:

```bash
cd paper && pdflatex main.tex && pdflatex main.tex && pdflatex main.tex
```

## Citation

```bibtex
@article{miesegaes2026quantdex,
  title={QuantDex: Sub-Linear Attention via the Quantization-Indexing Duality},
  author={Miesegaes, F.},
  year={2026},
  note={Preprint}
}
```

## License

MIT License
