# Conv-Linformer: Boosting Linformer’s Performance with Convolution in Small-Scale Settings

This repository contains the code and experiments from my application to the **Eastern European Machine Learning (EEML) Summer School**. The project is divided into two parts:

1. A **systematic study of Linformer's behavior** in resource-constrained settings (limited data, long sequences, varying learning rates).
2. A proposed variation — **Conv-Linformer** — designed to improve stability and performance using 1D convolutions in the attention mechanism.

---

## 🧠 Motivation

The [Linformer](https://arxiv.org/abs/2006.04768) reduces the quadratic complexity of standard self-attention using linear projections of keys and values. While effective at scale, **its performance in small-scale settings** (limited data, long sequences, tight compute) remains underexplored.

This project aims to:
- Understand Linformer's **strengths and failure modes** under such constraints,
- Explore how **simple architectural changes**, like adding 1D convolutions, might mitigate instability and improve performance.

---

## 📊 Phase 1: Studying Linformer Behavior

**Setup:**
- **Dataset**: [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext), subset to 50M tokens.
- **Sequence lengths**: 128, 256, 512, 1024.
- **Architecture**: Encoder-only Transformer, 8 layers, hidden size 512, 8 attention heads.
- **Training**: MLM objective, Hugging Face `Trainer`, AdamW, with warmup (10%) and weight decay (0.001).

**Findings:**
- **Longer sequences** destabilize training, especially at **higher learning rates**.
- Linformer performs **well at shorter sequence lengths**, but struggles as sequence length increases.
- Lowering the learning rate improves stability across sequence lengths.
- The instability may stem from Linformer's **low-rank approximations**, which need more training data to compensate loss of information.

---

## 🚀 Phase 2: Conv-Linformer

To address these limitations, I propose **Conv-Linformer**, a hybrid architecture that combines:
- **Linear projection** in early layers (preserving Linformer’s efficiency and global context),
- **1D convolution** in later layers (enhancing local pattern extraction in key/value compression).

The convolution uses:
- Kernel size and stride = `n/k` (sequence length / compression size), preserving linear complexity.

**Results:**
- Conv-Linformer achieves **better training stability**, especially on longer sequences.
- It approaches Transformer-level performance while maintaining linear complexity.
- It performs **more consistently** than the original Linformer under the same constraints.

---

## 🔗 References

- 📄 Linformer Paper: [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768)  
- 🔗 Linformer Code: [lucidrains/linformer](https://github.com/lucidrains/linformer)  
- 📚 Dataset: [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)

---

## 🌱 Future Work

- Scale to larger datasets to evaluate generalization.
- Explore different convolution types (e.g., depthwise separable, dilated).
- Investigate downstream NLP tasks beyond masked language modeling.

