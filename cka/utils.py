"""Utility functions for CKA computations and visualizations."""

from typing import List, Optional, Sequence, Union
import matplotlib.pyplot as plt
import numpy as np
import torch


def centering(K: torch.Tensor) -> torch.Tensor:
    """Center a square Gram matrix K: H K H where H = I - 1/N 1 1^T."""
    return K - K.mean(dim=0, keepdim=True) - K.mean(dim=1, keepdim=True) + K.mean()


def gram_linear(X: torch.Tensor) -> torch.Tensor:
    """Compute linear Gram matrix K = X X^T."""
    return X @ X.T


def gram_rbf(X: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Compute RBF Gram matrix."""
    dist_sq = torch.cdist(X, X, p=2) ** 2
    return torch.exp(-dist_sq / (2.0 * sigma ** 2))


def plot_cka_matrix(
    cka_matrix: np.ndarray,
    layer_labels_x: Optional[Sequence[str]] = None,
    layer_labels_y: Optional[Sequence[str]] = None,
    title: str = "Centered Kernel Alignment (CKA) Similarity",
    cmap: str = "magma",
    vmin: float = 0.0,
    vmax: float = 1.0,
    figsize: tuple = (9, 7),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot a publication-grade CKA similarity heatmap.

    Args:
        cka_matrix: 2D numpy array of CKA values in [0, 1].
        layer_labels_x: Labels for x-axis columns.
        layer_labels_y: Labels for y-axis rows.
        title: Title of the plot.
        cmap: Matplotlib colormap (e.g. 'magma', 'viridis', 'inferno').
        vmin: Minimum value for colormap scaling.
        vmax: Maximum value for colormap scaling.
        figsize: Figure size tuple.
        save_path: If provided, saves figure to this path.
        show: If True, calls plt.show().

    Returns:
        plt.Figure: The created figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cka_matrix, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("CKA Similarity", fontsize=11, fontweight="bold")

    n_rows, n_cols = cka_matrix.shape

    if layer_labels_x is not None:
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(layer_labels_x, rotation=45, ha="right", fontsize=9)
    if layer_labels_y is not None:
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(layer_labels_y, fontsize=9)

    ax.set_xlabel("Layers (Model 2 / Post-transition)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Layers (Model 1 / Pre-transition)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"CKA plot saved to: {save_path}")

    if show:
        plt.show()

    return fig
