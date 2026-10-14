# FlowPrune: Accelerating Attention Flow Calculation by Pruning Flow Network

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-%23000000.svg?logo=networkx&logoColor=white)](https://networkx.org/)
[![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-%23white.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://github.com/ATMxsp01/FlowPrune)

A clean, high-performance PyTorch toolkit for **FlowPrune**, an algorithm that accelerates attention-flow computation in Transformers by pruning low-capacity edges and compressing non-critical layers before running maximum flow, based on:

> **FlowPrune: Accelerating Attention Flow Calculation by Pruning Flow Network**  
> Shuo Xu, Yu Chen, Shuxia Lin, Xin Geng, Xu Yang.  
> *39th Conference on Neural Information Processing Systems (NeurIPS 2025)*.  
> Code: [https://github.com/ATMxsp01/FlowPrune](https://github.com/ATMxsp01/FlowPrune)

---

## Table of Contents

- [Motivation & Problem Statement](#motivation--problem-statement)
- [How FlowPrune Works](#how-flowprune-works)
- [Mathematical Formulation](#mathematical-formulation)
  - [1. Edge Pruning](#1-edge-pruning)
  - [2. Min-Cut Sampling Heuristics](#2-min-cut-sampling-heuristics)
  - [3. Dynamic Programming Layer Selection](#3-dynamic-programming-layer-selection)
  - [4. Layer Reconnection](#4-layer-reconnection)
  - [5. Maximum Flow Calculation](#5-maximum-flow-calculation)
- [Theoretical Complexity & Speed-Up](#theoretical-complexity--speed-up)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Functional API (`compute_flowprune`)](#functional-api-compute_flowprune)
  - [Model Hooks API (`FlowPrune`)](#model-hooks-api-flowprune)
- [API Reference](#api-reference)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualizing ViT Attention Heatmaps](#visualizing-vit-attention-heatmaps)
- [FlowPrune vs. Attention Rollout vs. Exact Flow](#flowprune-vs-attention-rollout-vs-exact-flow)
- [Citation](#citation)

---

## Motivation & Problem Statement

In multi-layer Transformer models (such as Vision Transformers, LLMs, and multimodal VLMs), single-layer attention maps only show local token-to-token interactions. **Attention Flow** (Abnar & Zuidema, 2020) provides a principled global view by tracing how information propagates from input tokens to final representations across all layers, modeling the architecture as a directed acyclic graph (DAG) and calculating maximum flow.

### Why not Exact Attention Flow?
With $L$ layers and $N$ tokens:
- Vertices: $V \approx LN$
- Edges: $E \approx N^2 L$
- Highest-Level Preflow Push (HLPP) max-flow complexity: $\mathcal{O}(V^2 \sqrt{E}) = \mathcal{O}(L^{2.5} N^3)$

This high complexity becomes intractable for modern deep Transformers and long sequence lengths.

### Why not Attention Rollout?
Attention Rollout approximates attention flow in $\mathcal{O}(LN^3)$ by recursively multiplying attention matrices. However, as demonstrated by Xu et al. (NeurIPS 2025):
1. **Ignores global capacity constraints**: Attention rollout ignores the fact that attention weights represent bottleneck capacities in a flow network.
2. **Violates flow conservation**: Information paths can distort and overestimate flow through multi-hop shortcuts, producing explanations that contradict ground-truth information paths.

**FlowPrune** resolves this trade-off: it preserves true max-flow properties (capacity constraints and conservation) while dramatically reducing computation time.

---

## How FlowPrune Works

FlowPrune is founded on the **Max-Flow Min-Cut Theorem**:
> In a flow network with non-negative edge capacities, the maximum flow between source and sink equals the capacity of the minimum cut separating them.

Consequently:
- Edges that do not participate in any minimum cut have zero influence on maximum flow.
- Layers with few min-cut bottleneck edges contribute negligibly to global information propagation.

FlowPrune exploits two key structural properties of Transformers:
1. **Softmax Sparsity**: Softmax distributions produce many near-zero attention weights that carry virtually no flow.
2. **Layered Structure**: Edges strictly flow from layer $l$ to layer $l+1$, allowing min-cut bottlenecks to be localized per layer.

```
Original Attention Graph (L layers)
         │
         ▼
[1] Edge Pruning (capacity < θ = 1e-6)
         │
         ▼
[2] Min-Cut Sampling & DP Layer Selection
         │
         ▼
[3] Layer Reconnection (Transitive Reachability A_i->i+2)
         │
         ▼
Compressed Flow Graph (L_comp << L)
         │
         ▼
[4] Maximum Flow Calculation (HLPP / Preflow Push)
         │
         ▼
High-Fidelity Attention Flow Vector / Matrix
```

---

## Mathematical Formulation

### 1. Edge Pruning
Given layer attention weights $A_l$:
$$e \text{ is removed if } c(e) < \theta$$
where default $\theta = 1 \times 10^{-6}$.

### 2. Min-Cut Sampling Heuristics
FlowPrune samples candidate source-to-sink paths and finds the bottleneck edge with minimum capacity on each path. Four tactics are available:
- **Tactic 1 (`walk` - Recommended)**: Forward random walk along non-zero attention transitions.
- **Tactic 2 (`bidirectional`)**: Bidirectional forward/backward random walks from sources and sinks.
- **Tactic 0 (`permutation`)**: Random token permutations across layers.
- **Tactic 3 (`threshold`)**: Direct histogram count of weak edges in $(0, \theta_{\text{high}}]$.

The counts of bottleneck edges in layer $i$ form the array $a_i = \text{min\_count}[i]$.

### 3. Dynamic Programming Layer Selection
To retain $K$ intermediate layers (for retain rate $1/R$) while maximizing preserved min-cut edges:
- $f(i, j, 0)$: Max min-cut edges among first $i$ layers compressed to $j$ layers with layer $i$ compressed.
- $f(i, j, 1)$: Same with layer $i$ retained.

$$
\begin{aligned}
f(i, j, 0) &= \max\big(f(i-1, j, 0),\; f(i-1, j, 1)\big) \\
f(i, j, 1) &= \max\big(f(i-1, j-1, 1),\; f(i-1, j-2, 0)\big) + a_i
\end{aligned}
$$

Backtracking identifies the intervals $(start, end)$ of layers to compress.

### 4. Layer Reconnection
For compressed layer intervals, binary connectivity is propagated:
$$A_{i \to i+2} = A_{i \to i+1} \cdot A_{i+1 \to i+2}$$
Reconnected edges are assigned capacity $c = 1.0$. Because original normalized attention weights $\le 1.0$, this capacity constraint never artificially bottlenecks the flow, guaranteeing flow is dictated solely by retained layers.

### 5. Maximum Flow Calculation
Max-flow is solved on the compressed DAG using the Highest-Level Preflow Push (HLPP) algorithm via NetworkX.

---

## Theoretical Complexity & Speed-Up

| Method | Complexity | Empirical ViT Speedup | Flow Conservation | Capacity Constraints |
|---|---|---|---|---|
| **Exact Attention Flow** | $\mathcal{O}(L^{2.5} N^3)$ | $1.0\times$ (Baseline) | Yes | Yes |
| **Attention Rollout** | $\mathcal{O}(LN^3)$ | Fast | No | No |
| **FlowPrune ($R$ retain)** | $\mathcal{O}\left(\frac{L^{2.5} N^3}{R^{2.5}}\right)$ | **Up to $18.8\times$ faster** | **Yes** | **Yes** |

---

## Installation

```bash
pip install torch networkx numpy opencv-python matplotlib
```

Or install with Poetry inside `cv-toolkit`:

```bash
poetry install
```

---

## Quick Start

### Functional API (`compute_flowprune`)

```python
import torch
from flowprune import compute_flowprune

# Assume attentions is a list of L tensors of shape [NumHeads, SeqLen, SeqLen]
# (e.g. from outputs.attentions in HuggingFace or timm)
attentions = [torch.randn(8, 197, 197).softmax(dim=-1) for _ in range(12)]

# Compute flow from all input patches to [CLS] token (token 0) with 25% retain rate
cls_flow = compute_flowprune(
    attentions,
    query_token=0,
    retain_rate=0.25,
    threshold=1e-6,
    tactic="walk",
    head_fusion="mean",
)

print("Flow vector shape:", cls_flow.shape)  # torch.Size([197])
print("Attribution sum:", cls_flow.sum().item())  # 1.0000 (normalized)
```

### Model Hooks API (`FlowPrune`)

```python
import timm
import torch
from flowprune import FlowPrune, flow_to_heatmap, overlay_flow_heatmap

# 1. Load any Vision Transformer
model = timm.create_model("deit_small_patch16_224", pretrained=True)
pruner = FlowPrune(model, attention_layer_name="attn", retain_rate=0.25)

# 2. Forward pass with hook capture
input_tensor = torch.randn(1, 3, 224, 224)
cls_attribution = pruner(input_tensor, query_token=0)  # [197]

# 3. Convert attribution vector to 2D spatial heatmap (14x14 patches)
heatmap = flow_to_heatmap(cls_attribution, grid_size=(14, 14), has_cls_token=True)

# 4. Remove hooks when done
pruner.remove_hooks()
```

---

## API Reference

### Core Functions & Classes

- `compute_flowprune(attentions, query_token=0, retain_rate=0.5, threshold=1e-6, tactic="walk", head_fusion="mean", workers=None)`:
  High-level functional interface to calculate attention flow.
- `FlowPrune(model, attention_layer_name="attn", retain_rate=0.5, threshold=1e-6, ...)`:
  Module wrapper with forward hook registration.
- `prune_edges(attentions, threshold=1e-6)`:
  Zeros out edge weights below capacity threshold and returns sparsity statistics.
- `compress_layers(attentions, compact_list, threshold=0.0, max_capacity=1.0)`:
  Compresses layer intervals and reconnects reachability.
- `prune_and_compress_attention(attentions, retain_rate=0.5, threshold=1e-6, ...)`:
  End-to-end graph reduction pipeline.

### Graph & Solver Utilities

- `build_attention_graph(attentions, threshold=0.0)`:
  Constructs NetworkX `DiGraph` representation of attention network.
- `compute_token_flow(graph, source_token, target_token)`:
  Calculates max-flow between any source and sink token.
- `compute_query_flow_vector(graph, target_token=0, num_tokens=None, workers=None)`:
  Calculates flow from all initial tokens to target query token.
- `FlowGraph`:
  High-level class wrapping graph construction and queries.

---

## Evaluation Metrics

FlowPrune includes the evaluation metrics established in Section 4.1.1 of the paper:

```python
from flowprune import (
    top_k_jaccard_similarity,
    first_rank_divergence,
    compute_approximation_errors,
)

# 1. Top-K Token Retention (Jaccard Overlap / IOU)
# IOU = |A ∩ B| / |A ∪ B|
iou_10 = top_k_jaccard_similarity(flowprune_flow, exact_flow, k=10)
print(f"Top-10 Token Retention: {iou_10 * 100:.2f}%")

# 2. First Rank Divergence (FRD)
# Normalized position where the top-ranked tokens diverge
frd = first_rank_divergence(flowprune_flow, exact_flow, normalize=True)
print(f"FRD: {frd:.4f}")

# 3. Absolute and Relative Errors
errors = compute_approximation_errors(exact_flow, flowprune_flow)
print(f"MAE: {errors['mae']:.6f}, Relative Error: {errors['relative_error']:.4f}")
```

---

## Visualizing ViT Attention Heatmaps

```python
import cv2
import numpy as np
from flowprune import flow_to_heatmap, overlay_flow_heatmap

# Convert [CLS] flow vector [197] to 224x224 spatial heatmap (excluding CLS token)
heatmap_2d = flow_to_heatmap(
    cls_flow,
    grid_size=(14, 14),
    has_cls_token=True,
    target_size=(224, 224),
)

# Overlay on input image
rgb_image = cv2.imread("example.jpg")
overlay = overlay_flow_heatmap(rgb_image, heatmap_2d, alpha=0.5, colormap=cv2.COLORMAP_JET)
cv2.imwrite("flowprune_heatmap.png", overlay)
```

---

## FlowPrune vs. Attention Rollout vs. Exact Flow

| Feature | Exact Attention Flow | FlowPrune | Attention Rollout |
|---|---|---|---|
| **Max-Flow Min-Cut Grounded** | Yes | **Yes** | No |
| **Capacity Constraints Respected** | Yes | **Yes** | No |
| **Top-K Overlap with Exact** | 100% | **~90-99%** | ~28-70% |
| **Speed on DeiT-S / LLaMA** | Baseline ($1\times$) | **Up to $18.8\times$ faster** | Fast |
| **Theoretical Formulation** | HLPP Max-Flow | Compressed HLPP | Matrix Multiplication |

---

## Citation

```bibtex
@inproceedings{xu2025flowprune,
  title={FlowPrune: Accelerating Attention Flow Calculation by Pruning Flow Network},
  author={Xu, Shuo and Chen, Yu and Lin, Shuxia and Geng, Xin and Yang, Xu},
  booktitle={Thirty-ninth Conference on Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```
