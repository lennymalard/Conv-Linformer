# Conv-Linformer: Boosting Linformer’s Performance with Convolution in Small-Scale Settings

This repository contains the code and experiments from my application to the **Eastern European Machine Learning (EEML) Summer School**. The project involves:

1. A **systematic study** of Linformer’s behavior under resource constraints,
2. A proposed variant — **Conv-Linformer** — aimed at improving training stability and performance in those settings.

---

## 🧠 Motivation

The [Linformer](https://arxiv.org/abs/2006.04768) reduces the quadratic complexity of self-attention using low-rank projections of keys and values. While efficient at scale, its performance in **resource-constrained environments** (limited data, long sequences, constrained compute) remains underexplored.

This project investigates:
- How Linformer behaves in such settings,
- Whether simple architectural tweaks can improve its robustness and effectiveness.

---

## 📊 Part 1: Reproducing Linformer

**Experimental setup:**
- **Dataset**: [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext), subset to 50M tokens.
- **Sequence lengths**: 128, 256, 512, 1024.
- **Model**: Encoder-only Transformer with 8 layers, hidden size 512, 8 attention heads.
- **Training**: MLM objective, Hugging Face `Trainer`, AdamW optimizer with 10% warmup and weight decay (0.001).

**Key insights:**
- **Short sequences** perform well, but **longer sequences** suffer from instability at high learning rates.
- Lower learning rates stabilize training but may also lead to slower convergence or cause the model to get stuck in a local minimum.
- The performance degradation appears tied to Linformer’s **low-rank projections**, which struggle to learn important information when training data is limited.

---

## 🚀 Part 2: Conv-Linformer

To address these issues, I introduce **Conv-Linformer**, a hybrid architecture that:

- Uses **linear projection** in the early layers to retain global context efficiently,
- Applies **1D convolution** in later layers to improve local pattern extraction.

The convolution uses:
- **Kernel size and stride** = `n/k` (sequence length divided by compression size), preserving linear complexity.

**Results:**
- **Improved training stability** across sequence lengths,
- **More consistent performance** than Linformer in constrained settings,
- **Near-Transformer-level results**, with linear complexity and minor overhead from convolution.

---

## 🔗 References

- 📄 [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768)  
- 🔗 [lucidrains/linformer](https://github.com/lucidrains/linformer)  
- 📚 [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)

---

## 🌱 Future Work

- Scale experiments to larger datasets and tasks.
- Investigate other compression strategies.
- Evaluate performance on downstream NLP benchmarks.
