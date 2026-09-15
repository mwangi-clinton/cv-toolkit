# Attention Rollout

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-%23white.svg?logo=opencv&logoColor=white)](https://opencv.org/)

A clean, high-performance PyTorch implementation of **Attention Rollout** for quantifying and visualizing the flow of information through Vision Transformers (ViTs), based on:

> **Quantifying Attention Flow in Transformers**  
> Samira Abnar, Willem Zuidema. *ACL 2020*.  
> [arXiv:2005.00928](https://arxiv.org/abs/2005.00928)

---

## Table of Contents

- [The Problem with Raw Attention](#the-problem-with-raw-attention)
- [How Attention Rollout Works](#how-attention-rollout-works)
- [Mathematical Formulation](#mathematical-formulation)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Use Cases](#use-cases)
  - [Classification ViTs ([CLS] Token)](#classification-vits-cls-token)
  - [Dense Prediction / Pose Estimation ViTs (Patch Queries)](#dense-prediction--pose-estimation-vits-patch-queries)
- [References](#references)

---

## The Problem with Raw Attention

In multi-layer Vision Transformers, inspecting raw attention weights at layer $l$ alone gives an incomplete picture:
1. **Residual Connections**: At each layer, token representations are updated via $\mathbf{Z}_{l} = \mathbf{Z}_{l-1} + 	ext{Attn}(\mathbf{Z}_{l-1})$. Looking only at $	ext{Attn}$ ignores the identity skip connection.
2. **Indirect Dependency**: Information from an input patch may reach a final token through paths spanning multiple intermediate tokens across layers.
3. **Multi-Head Dispersion**: Individual heads attend to different semantic cues, requiring principled fusion.

**Attention Rollout** accounts for both skip connections and transitive paths by recursively multiplying attention matrices across the entire network.

---

## Mathematical Formulation

Let $\mathbf{A}_l \in \mathbb{R}^{S 	imes S}$ be the fused attention matrix at layer $l$, where $S$ is the sequence length.

### 1. Accounting for Residual Identity
Because the layer computation adds the input representation to the attended output:

$$
\hat{\mathbf{A}}_l = lpha \mathbf{A}_l + (1 - lpha) \mathbf{I}
$$

where $\mathbf{I}$ is the identity matrix and $lpha \in (0, 1]$ balances attention vs. identity (typically $lpha = 0.5$). Each row is then re-normalized to sum to 1:

$$
	ilde{\mathbf{A}}_l = 	ext{diag}(\hat{\mathbf{A}}_l \mathbf{1})^{-1} \hat{\mathbf{A}}_l
$$

### 2. Recursive Rollout Multiplication
The cumulative attention rollout $\mathbf{R}_L$ after $L$ layers is obtained by multiplying successive attention transitions:

$$
\mathbf{R}_l = 	ilde{\mathbf{A}}_l \cdot \mathbf{R}_{l-1}, \quad 	ext{with } \mathbf{R}_0 = \mathbf{I}
$$

Expanding across all $L$ layers:

$$
\mathbf{R}_L = \prod_{l=1}^L 	ilde{\mathbf{A}}_l = 	ilde{\mathbf{A}}_L \cdot 	ilde{\mathbf{A}}_{L-1} \cdots 	ilde{\mathbf{A}}_1
$$

Row $i$ of $\mathbf{R}_L$ shows the total proportion of information token $i$ at the output received from every input token at layer 0.

### 3. Noise Thresholding (`discard_ratio`)
To eliminate diffuse, low-confidence attention noise that accumulates across many layers, Abnar & Zuidema recommend zeroing out the bottom $p\%$ (e.g. 10%) of attention values before re-normalizing.

---

## Installation

```bash
pip install torch numpy opencv-python matplotlib
```

Or install via Poetry in `cv-toolkit`:

```bash
poetry install
```

---

## Quick Start

```python
import torch
from attention_rollout import compute_attention_rollout, overlay_rollout_heatmap, rollout_to_heatmap

# Assume `attentions` is a list of L tensors: [batch, heads, seq_len, seq_len]
# e.g. from HuggingFace `outputs.attentions` or timm ViT
rollout_matrix = compute_attention_rollout(
    attentions,
    discard_ratio=0.1,    # Discard lowest 10% weights
    head_fusion="mean",   # Average across heads
    residual_weight=0.5,  # Equal weight for attention and skip connection
)

# 1. Extract rollout for token 0 (e.g. [CLS] token)
cls_rollout = rollout_matrix[0]  # [seq_len]

# 2. Reshape to 2D heatmap
heatmap = rollout_to_heatmap(cls_rollout, grid_size=(14, 14), has_cls_token=True, target_size=(224, 224))

# 3. Blend with input image
overlay = overlay_rollout_heatmap(image_rgb, heatmap, alpha=0.5)
```

---

## API Reference

### `compute_attention_rollout(attentions, discard_ratio=0.0, head_fusion="mean", residual_weight=0.5)`
- `attentions`: List/tuple of attention tensors `[heads, seq_len, seq_len]` or `[batch, heads, seq_len, seq_len]`.
- `discard_ratio`: Float in $[0, 1)$. Suppresses noise by setting lowest $p\%$ attention weights to 0.
- `head_fusion`: `"mean"`, `"max"`, or `"min"`.
- `residual_weight`: Alpha weight for attention vs. residual connection (default: 0.5).

### `compute_batched_attention_rollout(attentions, ...)`
- Batched GPU-accelerated version using `torch.bmm`.

### `rollout_to_heatmap(rollout_vector, grid_size, has_cls_token=True, target_size=None)`
- Maps 1D token vector into normalized 2D spatial array.

### `overlay_rollout_heatmap(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET)`
- Resizes heatmap and blends with the RGB image.

---

## Use Cases

### Classification ViTs ([CLS] Token)
Query index `0` represents the `[CLS]` token. Rollout reveals which input patches most influenced the classification decision.

### Dense Prediction / Pose Estimation ViTs (Patch Queries)
In ViTPose or segmentation transformers without a `[CLS]` token, query index $k = y 	imes W_{	ext{patches}} + x$ reveals which surrounding context regions informed that specific keypoint or spatial location.

---

## References

1. Abnar & Zuidema, "Quantifying Attention Flow in Transformers", ACL 2020. [arXiv:2005.00928](https://arxiv.org/abs/2005.00928)
2. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021.
3. Xu et al., "ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation", NeurIPS 2022.
