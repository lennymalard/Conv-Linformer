# Conv-Linformer: Boosting Linformer’s Performance with Convolution

This repository contains the code and experiments from my application to the **Eastern European Machine Learning (EEML) Summer School**. The project investigates a hybrid Transformer architecture that combines **linear projection** and **1D convolution** for more effective key/value compression in self-attention.

---

## 🧠 Motivation

Transformer models suffer from quadratic time and memory complexity due to the self-attention mechanism. The [Linformer](https://arxiv.org/abs/2006.04768) addresses this by projecting the keys and values into a lower-dimensional space, achieving linear complexity. However, its performance under **data and resource constraints** is underexplored.

This study explores the idea that adding **1D convolutions** can help Linformer better extract **local patterns**, especially when trained on **limited data** and with **long sequences**.

---

## 🔍 Overview of the Method

We propose **Conv-Linformer**, a hybrid variant of Linformer, where:

- **Early layers** retain Linformer's **linear projection** to preserve global context.
- **Later layers** apply **1D convolution** (kernel size and stride = `n/k`) to compress keys and values, leveraging local dependencies.

This design aims to balance:
- **Global context capture** (linear projection),
- **Local pattern extraction** (convolution),
- While preserving **linear complexity** and improving training stability.

---

## 📊 Experimental Setup

- **Dataset**: [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext), subset to 50 million tokens.
- **Sequence lengths**: 128, 256, 512, 1024.
- **Architecture**: Encoder-only Transformer with 8 layers, hidden size 512, 8 attention heads.
- **Training**: MLM objective using Hugging Face’s `Trainer`, AdamW optimizer with warmup (10%), weight decay 0.001.

---

## 📈 Key Results

- **High learning rates** benefit **short sequences** but destabilize **longer ones**.
- **Lower learning rates** yield more **stable convergence**, albeit with some performance tradeoff.
- **Conv-Linformer** consistently performs **more stably** than Linformer and achieves **near-Transformer-level** results, especially on **longer sequences**.
- On small-scale setups, **Linformer suffers from performance drops**, likely due to its low-rank approximations requiring more training data.

---

## 🔗 References

- 📄 Linformer Paper: [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768)  
- 🔗 Linformer Code: [lucidrains/linformer](https://github.com/lucidrains/linformer)  
- 📚 Dataset: [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)

---

## 🚀 Future Directions

- Scale experiments to larger corpora to validate findings.
- Investigate downstream task performance.
- Explore other compression alternatives (e.g., dynamic convolution or depthwise separable convolutions).
