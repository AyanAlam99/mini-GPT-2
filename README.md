<div align="center">

# ⚡ mini-GPT2
### A Decoder-Only Transformer Language Model — Built From Scratch in PyTorch

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Tokenizer](https://img.shields.io/badge/Tokenizer-tiktoken_cl100k-412991?style=for-the-badge&logo=openai&logoColor=white)](https://github.com/openai/tiktoken)
[![Vocab](https://img.shields.io/badge/Vocab_Size-100%2C256-orange?style=for-the-badge)](#️-model-hyperparameters)
[![Architecture](https://img.shields.io/badge/Architecture-Decoder--Only_Transformer-blue?style=for-the-badge)](#-model-architecture)

<br/>

> *Every matrix multiply, every attention head, every residual connection — implemented from first principles. No HuggingFace. No shortcuts.*

<br/>

---

</div>

## 🧠 What This Is

`mini-GPT2` is a ground-up PyTorch implementation of an **autoregressive, decoder-only Transformer** — the same fundamental architecture behind GPT-2, GPT-3, and the modern LLM family.

The goal isn't to compete with production models on scale. It's to demonstrate **complete, working mastery** of the core mathematics:

- **Causal self-attention** — how tokens attend to their past but not their future
- **Autoregressive next-token prediction** — how language models generate text
- **Pre-LayerNorm residual blocks** — the stability trick modern LLMs all use
- **Out-of-core data streaming** — how to train on datasets larger than RAM

---

## 🏗️ Model Architecture

```
Input Token IDs  [t₁, t₂, ..., tₙ]
        │
        ▼
┌───────────────────────────────────────┐
│         Token Embedding Table         │
│         vocab_size → n_embd (128)     │
└───────────────┬───────────────────────┘
                │
        + Positional Embeddings
          (learned, absolute)
                │
                ▼
┌───────────────────────────────────────┐   ×8
│        Transformer Block              │◄──────
│                                       │
│  ┌─────────────────────────────────┐  │
│  │  Pre-LayerNorm                  │  │
│  │  Multi-Head Causal Self-Attn    │  │  6 heads
│  │  (causal mask — no future leak) │  │
│  │  + Dropout (0.2)                │  │
│  └──────────────┬──────────────────┘  │
│      Residual   │   Connection        │
│  ┌──────────────▼──────────────────┐  │
│  │  Pre-LayerNorm                  │  │
│  │  Feed-Forward Network           │  │
│  │  (n_embd → 4×n_embd → n_embd)  │  │
│  │  + Dropout (0.2)                │  │
│  └─────────────────────────────────┘  │
└───────────────────────────────────────┘
                │
        Final LayerNorm
                │
                ▼
┌───────────────────────────────────────┐
│     Language Model Head (Linear)      │
│     n_embd (128) → vocab (100,256)    │
└───────────────────────────────────────┘
                │
                ▼
        Next-Token Logits
```

### Design Choices — and Why They Matter

| Choice | Alternative | Why This Is Better |
|--------|-------------|-------------------|
| **Pre-LayerNorm** | Post-LayerNorm (original Transformer) | Stabilises training gradient flow; used in GPT-2+ |
| **Causal Attention Mask** | Bidirectional attention | Required for autoregressive generation — no future token leakage |
| **AdamW Optimizer** | Adam | Decoupled weight decay prevents adaptive gradient scaling from undermining regularisation |
| **`tiktoken` cl100k_base** | char-level / word-level | Subword BPE handles rare words, multilingual text, and code efficiently |
| **`mmap` Data Loading** | Loading full dataset into RAM | Streams data directly from disk — training scales beyond available RAM |

---

## ⚙️ Model Hyperparameters

```python
# config.py
CONFIG = {
    "vocab_size":    100_256,   # tiktoken cl100k_base
    "n_embd":        128,       # embedding dimension
    "context_win":   256,       # tokens per training context
    "n_head":        6,         # attention heads
    "n_layer":       8,         # transformer blocks
    "dropout":       0.2,
    "learning_rate": 3e-4,      # AdamW
}
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| Vocab Size | 100,256 | `cl100k_base` — same tokenizer as GPT-4 |
| Embedding Dim | 128 | Lightweight for fast iteration |
| Context Window | 256 tokens | Expandable — see Roadmap |
| Attention Heads | 6 | Head dim = 128/6 ≈ 21 |
| Transformer Layers | 8 | Depth over width |
| Dropout | 0.2 | Applied post-attention and post-FFN |
| Learning Rate | 3e-4 | AdamW with decoupled weight decay |

---

## ⚡ Efficient Data Pipeline

Training large language models is bottlenecked as much by **data throughput** as by compute. This project addresses that with memory-mapped file I/O over raw text corpora:

```python
# dataloader.py — simplified illustration
import mmap

class MemoryMappedDataset:
    def __init__(self, filepath):
        self.file = open(filepath, 'r', encoding='utf-8')
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def get_batch(self, context_win, batch_size):
        # Read a text chunk directly from disk — no full-file RAM load
        data_chunk = self.mm.read(context_win * batch_size).decode('utf-8')
        # Tokenize on-the-fly
        tokens = enc.encode(data_chunk)
        data = torch.tensor(tokens)
        ...
```

**How it works:** Rather than loading the entire corpus into memory at startup, `mmap` lets the OS kernel page in only the bytes needed per batch. Each iteration reads a raw text chunk, decodes it, tokenizes it with `tiktoken`, and converts to a tensor — keeping the full dataset on disk throughout training. **This allows training on text corpora larger than available RAM.**

> **Note:** This pipeline operates on raw text files (`webcrawled_train.txt` / `webcrawled_val.txt`) and tokenizes dynamically at batch time. A future optimisation is to pre-tokenize to `.bin` (uint32 token IDs) to eliminate the per-batch `tiktoken` overhead and further speed up training.

---

## 📁 Project Structure

```
miniGpt/
├── data/
│   └── dataloader.py       # mmap-based out-of-core dataset streaming
│
├── utils/
│   ├── loss.py             # Cross-entropy loss estimation
│   ├── metrics.py          # Perplexity calculation & convergence monitoring
│   └── save_load.py        # Checkpoint save / resume logic
│
├── config.py               # All hyperparameters & environment paths
├── core.py                 # Shared imports across modules
├── model.py                # Full GPT architecture (Attention → Blocks → Head)
├── train.py                # Training loop, optimizer, LR scheduling
└── requirements.txt
```

---

## 🚀 Usage

### 1. Install Dependencies

```bash
pip install torch tiktoken matplotlib pandas
```

### 2. Configure Paths

Edit `config.py` to point to your dataset and checkpoint directory:

```python
CONFIG = {
    "train_file": "/path/to/webcrawled_train.txt",
    "val_file":   "/path/to/webcrawled_val.txt",
    "save_dir":   "./checkpoints/",
    ...
}
```

### 3. Train

```bash
python train.py
```

The script **automatically resumes** from the latest checkpoint in `save_dir` if one exists. Training metrics (loss, perplexity) are logged per interval and plots are saved automatically.

---

## 🗺️ Roadmap — Closing the Gap to Modern LLMs

This implementation is production-honest about what it doesn't yet have. Each item below is a concrete, well-scoped engineering improvement:

### ⚡ KV Caching
**Problem:** During autoregressive generation, the Key and Value matrices for all past tokens are recomputed at every new step — `O(N²)` redundant work per token.

**Fix:** Cache K and V tensors for all previous positions. Each new token only computes K/V for itself, then appends to the cache. Reduces decoding complexity from `O(N³)` → `O(N²)`.

### 🔦 FlashAttention
**Problem:** Standard attention materialises the full `N×N` attention score matrix in VRAM — a memory bottleneck that limits context window scaling.

**Fix:** Replace with `torch.nn.functional.scaled_dot_product_attention` (PyTorch 2.0+). FlashAttention uses tiled, fused CUDA kernels that never materialise the full matrix, cutting VRAM usage and improving throughput.

```python
# One-line drop-in replacement
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

### 🔄 Rotary Position Embeddings (RoPE)
**Problem:** Absolute positional embeddings don't generalise to sequence lengths longer than those seen during training.

**Fix:** Encode position as a rotation applied directly to the query/key vectors. RoPE is relative by construction — the model learns position-invariant attention patterns that transfer to longer contexts at inference time. Used in LLaMA, Mistral, and GPT-NeoX.

### 🔀 SwiGLU Activations
**Problem:** Standard ReLU in the feed-forward block creates dead neurons (zero gradient for negative inputs) and is outperformed by gated variants.

**Fix:** Replace FFN activation with SwiGLU — `SwiGLU(x) = (xW₁) ⊙ σ(xW₂)`. The gating mechanism selectively activates dimensions based on input content. Used in PaLM, LLaMA 2, and Mistral.

---

## 📚 References

- [**Attention Is All You Need**](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017 (Original Transformer)
- [**Language Models are Unsupervised Multitask Learners**](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al., 2019 (GPT-2)
- [**FlashAttention**](https://arxiv.org/abs/2205.14135) — Dao et al., 2022
- [**RoFormer: Enhanced Transformer with Rotary Position Embedding**](https://arxiv.org/abs/2104.09864) — Su et al., 2021
- [**GLU Variants Improve Transformer**](https://arxiv.org/abs/2002.05202) — Noam Shazeer, 2020 (SwiGLU)

---

<div align="center">

*Understanding transformers from the inside out — one matrix multiply at a time.*

</div>
