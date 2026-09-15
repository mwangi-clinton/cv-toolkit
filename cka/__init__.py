"""Centered Kernel Alignment (CKA) toolkit module.

This package provides a clean, high-performance PyTorch and NumPy implementation
of Linear and Kernel CKA (Kornblith et al., 2019) for measuring representation
similarity across neural network layers and architectures.

Example
-------
>>> import torch
>>> from cka import linear_cka, compute_cka_matrix, plot_cka_matrix
>>> X = torch.randn(100, 768)
>>> Y = torch.randn(100, 768)
>>> score = linear_cka(X, Y)
>>> print(f"CKA similarity: {score:.4f}")
"""

from .cka import (
    CKA,
    compute_cka_matrix,
    compute_layer_cka_matrix,
    kernel_cka,
    linear_cka,
)
from .utils import (
    centering,
    gram_linear,
    gram_rbf,
    plot_cka_matrix,
)

__all__ = [
    "CKA",
    "centering",
    "compute_cka_matrix",
    "compute_layer_cka_matrix",
    "gram_linear",
    "gram_rbf",
    "kernel_cka",
    "linear_cka",
    "plot_cka_matrix",
]
