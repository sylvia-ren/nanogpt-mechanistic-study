# nanogpt-mechanistic-study

This repository contains a clean, ground-up implementation of a **decoder-only Transformer** (GPT-2 architecture) designed for causal language modeling. 

The project focuses on two key research objectives:
1.  **Mechanistic Implementation:** Building Self-Attention, LayerNorm, and Residual connections from first principles to understand the flow of gradients.
2.  **Hardware Optimization & Scaling:** Benchmarking training throughput on **Apple Silicon (M4 Pro)** using the Metal Performance Shaders (MPS) backend to scale model complexity by **50x** compared to CPU baselines.

## Technical Architecture

The model follows the standard GPT-2 specifications with the following hyperparameters tailored for the M4 Pro memory bandwidth:

* **Architecture:** Multi-Head Causal Self-Attention + Feed-Forward Networks.
* **Parameters:** ~10.81 Million.
* **Context Window:** 256 tokens.
* **Embedding Dimension:** 384.
* **Heads:** 6 attention heads.
* **Layers:** 6 transformer blocks.
* **Optimization:** AdamW Optimizer, Cross-Entropy Loss.

## Scaling Experiment: CPU vs. MPS (Metal)

A core part of this study was analyzing the "Scaling Laws" by moving from a toy CPU model to a GPU-accelerated production prototype.

| Feature | Baseline Model (CPU) | Scaled Model (MPS/M4 Pro) | Improvement |
| :--- | :--- | :--- | :--- |
| **Parameter Count** | 0.21 M | **10.81 M** | **~50x Larger** |
| **Backend** | Torch CPU | **Torch MPS (Metal)** | Hardware Acceleration |
| **Block Size** | 32 | **256** | 8x Context |
| **Embed Dimension** | 64 | **384** | Richer Representations |
| **Training Time** | ~15 min | ~15 min | Equivalent time for 50x compute |
| **Final Val Loss** | ~2.50 | **~1.20** | **Significant Convergence** |

> **Note on MPS:** By leveraging Apple's *Metal Performance Shaders* (MPS), matrix multiplications ($QK^T$ and $V$ aggregation) were offloaded to the M4 Pro GPU cores. This allowed for a massive increase in model capacity (dimensionality and depth) without increasing training latency.

## Convergence Dynamics

The model was trained for 5,000 iterations. We observe a clean descent without exploding gradients, validating the implementation of LayerNorm and Residual Connections.

*(Insert your loss_curve.png here)*

## Quick Start

### 1. Installation
Clone the repo and install dependencies:
```bash
git clone [https://github.com/sylvia-ren/nanoGPT-financial-study.git](https://github.com/sylvia-ren/nanoGPT-financial-study.git)
cd nanoGPT-financial-study
pip install -r requirements.txt