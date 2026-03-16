# mini-GPT2

> An autoregressive, decoder-only Transformer language model built entirely from scratch in PyTorch.

This project implements the core mathematics of self-attention and next-token prediction, heavily inspired by standard generative pre-trained transformer architectures. It features a custom training loop, out-of-core memory-mapped data loading for efficient streaming of large text corpora, and utilizes OpenAI's `tiktoken` for robust sub-word tokenization.

---

## Key Features

- **Custom Transformer Architecture** — Implementation of multi-head causal self-attention, pre-LayerNorm residual blocks, and feed-forward networks.
- **Efficient Data Pipeline** — Uses Python's `mmap` module for out-of-core data loading, allowing the model to train on datasets larger than available RAM by streaming data directly from disk.
- **Modern Tokenization** — Integrated with OpenAI's `tiktoken` (`cl100k_base` encoding) to handle a vocabulary size of 100,256.
- **Modular Codebase** — Clean separation of concerns with dedicated modules for data loading, model architecture, loss estimation, and configuration.
- **Checkpointing & Metrics** — Includes automated model checkpoint saving/loading and custom perplexity calculations to monitor model convergence.

---

## 📂 Project Structure

```
miniGpt/
├── data/
│   └── dataloader.py       # Memory-mapped dataset streaming
├── utlis/
│   ├── loss.py             # Cross-entropy loss estimation
│   ├── metrics.py          # Perplexity calculation
│   └── save_load.py        # Checkpoint management
├── config.py               # Hyperparameters and environment paths
├── core.py                 # Shared core library imports
├── model.py                # GPT model architecture (Attention, Blocks, Heads)
├── train.py                # Main training loop and optimizer setup
└── requirements.txt        # Project dependencies
```

---

## ⚙️ Model Hyperparameters

The current configuration (`config.py`) is set up for a lightweight, fast-training environment:

| Parameter | Value |
|---|---|
| Vocab Size | 100,256 (`cl100k_base`) |
| Embedding Dimension (`n_embd`) | 128 |
| Context Window (`context_win`) | 256 tokens |
| Attention Heads (`n_head`) | 6 |
| Transformer Layers (`n_layer`) | 8 |
| Dropout | 0.2 |
| Learning Rate | 3e-4 (AdamW) |

---

## Usage

### 1. Setup

Install the required dependencies:

```bash
pip install torch tiktoken matplotlib pandas
```

Update the `train_file`, `val_file`, and `save_dir` paths in `config.py` to point to your local dataset and desired checkpoint directory.

### 2. Training

Run the main training script. The script will automatically load the latest checkpoint if one exists in `save_dir`.

```bash
python train.py
```

---

## Roadmap & Future Optimizations

To evolve this architecture closer to modern production-grade LLMs, the following improvements are planned:

- **KV Caching** — Implement Key-Value (KV) caching in the `generate` function to reduce time complexity during autoregressive decoding from $O(N^3)$ to $O(N^2)$.
- **FlashAttention** — Integrate `torch.nn.functional.scaled_dot_product_attention` to reduce VRAM bottlenecking during forward/backward passes.
- **Rotary Position Embeddings (RoPE)** — Replace standard absolute positional embeddings for better sequence length generalization.
- **SwiGLU Activations** — Swap standard ReLU in the feed-forward network for SwiGLU, as used in modern LLM architectures.
