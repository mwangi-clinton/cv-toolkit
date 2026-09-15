# Centered Kernel Alignment (CKA)

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?logo=matplotlib&logoColor=black)](https://matplotlib.org/)

A high-performance PyTorch & NumPy implementation of **Centered Kernel Alignment (CKA)** for comparing representations across layers and architectures, based on:

> **Similarity of Neural Network Representations Revisited**  
> Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton. *ICML 2019*.  
> [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)

---

## Table of Contents

- [What is CKA?](#what-is-cka)
- [Mathematical Formulation](#mathematical-formulation)
- [Why CKA for Knowledge Distillation?](#why-cka-for-knowledge-distillation)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Advanced Usage: Comparing Two Models](#advanced-usage-comparing-two-models)
- [References](#references)

---

## What is CKA?

Comparing neural network hidden representations across different random seeds, widths, depths, or architectures is notoriously challenging because:
- Activations can undergo arbitrary **orthogonal transformations** (rotations/reflections) without altering the model's functionality.
- Representations can be subject to arbitrary **isotropic scaling**.
- Canonical Correlation Analysis (CCA) is sensitive to perturbations when feature dimensions exceed sample count.

**CKA** resolves these issues: it is invariant to orthogonal transformations and isotropic scaling, while robustly identifying shared representation structures across varying widths and architectures.

---

## Mathematical Formulation

Given two centered representation matrices $\mathbf{X} \in \mathbb{R}^{N 	imes D_1}$ and $\mathbf{Y} \in \mathbb{R}^{N 	imes D_2}$ from $N$ identical inputs:

### Hilbert-Schmidt Independence Criterion (HSIC)

The empirical HSIC with linear kernel is given by:

$$
	ext{HSIC}(\mathbf{K}, \mathbf{L}) = rac{1}{(N - 1)^2} 	ext{tr}(\mathbf{K} \mathbf{H} \mathbf{L} \mathbf{H}) = rac{1}{(N - 1)^2} \|\mathbf{Y}^T \mathbf{X}\|_F^2
$$

where $\mathbf{K} = \mathbf{X}\mathbf{X}^T$, $\mathbf{L} = \mathbf{Y}\mathbf{Y}^T$, and $\mathbf{H} = \mathbf{I} - rac{1}{N}\mathbf{1}\mathbf{1}^T$ is the centering projection matrix.

### Centered Kernel Alignment (CKA)

CKA normalizes HSIC to lie within $[0, 1]$:

$$
	ext{CKA}(\mathbf{X}, \mathbf{Y}) = rac{	ext{HSIC}(\mathbf{K}, \mathbf{L})}{\sqrt{	ext{HSIC}(\mathbf{K}, \mathbf{K}) \cdot 	ext{HSIC}(\mathbf{L}, \mathbf{L})}} = rac{\|\mathbf{Y}^T \mathbf{X}\|_F^2}{\|\mathbf{X}^T \mathbf{X}\|_F \cdot \|\mathbf{Y}^T \mathbf{Y}\|_F}
$$

Alternatively, in terms of centered Gram matrices $	ilde{\mathbf{K}} = \mathbf{H}\mathbf{K}\mathbf{H}$ and $	ilde{\mathbf{L}} = \mathbf{H}\mathbf{L}\mathbf{H}$:

$$
	ext{CKA}(\mathbf{K}, \mathbf{L}) = rac{	ext{tr}(	ilde{\mathbf{K}} 	ilde{\mathbf{L}})}{\|	ilde{\mathbf{K}}\|_F \|	ilde{\mathbf{L}}\|_F}
$$

This implementation automatically selects between the cross-covariance formulation ($O(D_1 D_2 N)$) and the Gram formulation ($O(N^2 \max(D_1, D_2))$) based on $N$ vs. $D$ to minimize memory and computation.

---

## Why CKA for Knowledge Distillation?

1. **Phase Transitions**: Plotting layer-to-layer CKA yields a block-diagonal matrix. The boundaries between blocks represent *phase transitions* where representations fundamentally reorganize. Distilling across these boundaries transfers the most transformative representations.
2. **Teacher-Student Alignment**: Measuring CKA between teacher layers and student layers identifies which student layer best aligns with teacher representations, optimizing feature-level KD loss placement.

---

## Installation

```bash
pip install torch numpy matplotlib
```

Or from the root `cv-toolkit` repo:

```bash
poetry install
```

---

## Quick Start

```python
import torch
from cka import linear_cka, compute_cka_matrix, plot_cka_matrix

# Two representation matrices for N=100 samples
X = torch.randn(100, 768)   # e.g., Layer 6 features
Y = torch.randn(100, 1024)  # e.g., Layer 12 features

# 1. Compute scalar CKA
score = linear_cka(X, Y)
print(f"CKA Similarity: {score:.4f}")

# 2. Pairwise matrix across multiple layers
layer_features = [torch.randn(100, 768) for _ in range(12)]
matrix = compute_cka_matrix(layer_features, kernel="linear")

# 3. Plot publication-grade heatmap
labels = [f"L{i+1}" for i in range(12)]
plot_cka_matrix(matrix, layer_labels_x=labels, layer_labels_y=labels, title="Layer CKA Similarity")
```

---

## API Reference

### `linear_cka(X, Y, debiased=False) -> float`
- `X`: `[N, D1]` tensor or ndarray.
- `Y`: `[N, D2]` tensor or ndarray.
- `debiased`: Boolean. Uses unbiased HSIC estimator (Song et al., 2012) for small $N$.

### `kernel_cka(X, Y, sigma=None, kernel="rbf") -> float`
- `sigma`: RBF kernel bandwidth. Automatically determined via median heuristic if `None`.

### `compute_cka_matrix(features_list, kernel="linear", debiased=False) -> np.ndarray`
- Computes full symmetric $L 	imes L$ matrix using precomputed Gram matrices for $O(L^2 N^2)$ speed.

### `plot_cka_matrix(cka_matrix, layer_labels_x=None, layer_labels_y=None, ...)`
- Renders and optionally saves a publication-quality heatmap figure.

---

## Advanced Usage: Comparing Two Models

```python
import torch
import torchvision.models as models
from cka import CKA

# Compare ResNet-50 and ResNet-18
model1 = models.resnet50(pretrained=True).cuda()
model2 = models.resnet18(pretrained=True).cuda()

comparator = CKA(
    model1=model1,
    model2=model2,
    model1_layers=["layer1", "layer2", "layer3", "layer4"],
    model2_layers=["layer1", "layer2", "layer3", "layer4"],
    device="cuda",
)

# Run over your PyTorch DataLoader
cka_sim_matrix = comparator.compare(val_loader, max_batches=20)
print("CKA Matrix:\n", cka_sim_matrix)
```

---

## References

1. Kornblith et al., "Similarity of Neural Network Representations Revisited", ICML 2019. [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)
2. Song et al., "Feature Selection via Dependence Maximization", JMLR 2012.
3. Nguyen et al., "Do Wide and Deep Networks Learn the Same Things? Uncovering How Neural Network Representations Vary with Width and Depth", NeurIPS 2021.
