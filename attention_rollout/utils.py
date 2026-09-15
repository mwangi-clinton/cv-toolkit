"""Visualization utilities for Attention Rollout."""

from typing import Optional, Sequence, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def rollout_to_heatmap(
    rollout_vector: Union[torch.Tensor, np.ndarray],
    grid_size: Tuple[int, int],
    has_cls_token: bool = True,
    target_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Convert a 1D attention rollout vector to a 2D spatial heatmap.

    Args:
        rollout_vector: 1D array/tensor of shape [SeqLen].
        grid_size: (H_patches, W_patches), e.g. (14, 14) or (16, 12).
        has_cls_token: If True, index 0 is [CLS] token and is excluded from the spatial map.
        target_size: Optional (width, height) to resize output heatmap to.

    Returns:
        np.ndarray: Normalized 2D heatmap in [0, 1] of shape [H, W].
    """
    if isinstance(rollout_vector, torch.Tensor):
        vec = rollout_vector.detach().cpu().numpy()
    else:
        vec = np.array(rollout_vector)

    vec = vec.flatten()
    if has_cls_token:
        # Exclude CLS token at index 0
        patch_weights = vec[1:]
    else:
        patch_weights = vec

    gh, gw = grid_size
    if len(patch_weights) != gh * gw:
        raise ValueError(
            f"Token count ({len(patch_weights)}) does not match patch grid {gh}x{gw} = {gh*gw}"
        )

    heatmap = patch_weights.reshape(gh, gw)
    # Min-max normalize
    h_min, h_max = heatmap.min(), heatmap.max()
    heatmap = (heatmap - h_min) / (h_max - h_min + 1e-8)

    if target_size is not None:
        heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_CUBIC)
        heatmap = np.clip(heatmap, 0.0, 1.0)

    return heatmap


def overlay_rollout_heatmap(
    image: Union[np.ndarray, torch.Tensor],
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a 2D attention rollout heatmap onto an RGB image.

    Args:
        image: RGB image as uint8 [H, W, 3] or float tensor [C, H, W] in [0, 1].
        heatmap: 2D float heatmap in [0, 1] of shape [H, W].
        alpha: Blending weight for heatmap (1 - alpha for image).
        colormap: OpenCV colormap constant (e.g. cv2.COLORMAP_JET, cv2.COLORMAP_INFERNO).

    Returns:
        np.ndarray: Blended RGB image uint8 [H, W, 3].
    """
    if isinstance(image, torch.Tensor):
        img_np = image.detach().cpu().numpy()
        if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
            img_np = np.transpose(img_np, (1, 2, 0))
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
    else:
        img_np = np.array(image, copy=True)
        if img_np.dtype != np.uint8:
            img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

    h, w = img_np.shape[:2]
    if heatmap.shape != (h, w):
        hm_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        hm_resized = heatmap

    hm_uint8 = np.uint8(255 * np.clip(hm_resized, 0, 1))
    colored_hm = cv2.applyColorMap(hm_uint8, colormap)
    colored_hm = cv2.cvtColor(colored_hm, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(colored_hm, alpha, img_np, 1.0 - alpha, 0)
    return blended


def plot_rollout_evolution(
    image: np.ndarray,
    layer_rollouts: Sequence[np.ndarray],
    grid_size: Tuple[int, int],
    has_cls_token: bool = True,
    titles: Optional[Sequence[str]] = None,
    figsize: Tuple[int, int] = (18, 5),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot the progression of attention rollout across different layer depths.

    Args:
        image: Original RGB image [H, W, 3].
        layer_rollouts: Sequence of 1D rollout vectors at different depths.
        grid_size: (H_patches, W_patches).
        has_cls_token: Whether token 0 is [CLS].
        titles: Optional list of panel titles.
        figsize: Figure dimensions.
        save_path: If provided, saves figure.
        show: Whether to display figure.

    Returns:
        plt.Figure: The created figure.
    """
    n = len(layer_rollouts) + 1
    fig, axes = plt.subplots(1, n, figsize=figsize)

    axes[0].imshow(image)
    axes[0].set_title("Input Image", fontweight="bold")
    axes[0].axis("off")

    h, w = image.shape[:2]

    for idx, r_vec in enumerate(layer_rollouts):
        hm = rollout_to_heatmap(r_vec, grid_size, has_cls_token=has_cls_token, target_size=(w, h))
        overlay = overlay_rollout_heatmap(image, hm, alpha=0.5)
        ax = axes[idx + 1]
        ax.imshow(overlay)
        title = titles[idx] if titles and idx < len(titles) else f"Stage {idx+1}"
        ax.set_title(title, fontweight="bold")
        ax.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Rollout evolution saved to: {save_path}")
    if show:
        plt.show()

    return fig
