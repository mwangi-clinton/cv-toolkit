"""Visualization and evaluation metrics utilities for FlowPrune.

Includes:
- Heatmap reconstruction and overlay for Vision Transformers (ViTs).
- Evaluation metrics from the FlowPrune paper (NeurIPS 2025):
    * Top-K Token Retention (Jaccard IOU)
    * First Rank Divergence (FRD)
    * Approximation Errors (MAE, Relative Error)
- Visualization plots comparing Exact Attention Flow, FlowPrune, and Attention Rollout.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def flow_to_heatmap(
    flow_vector: Union[torch.Tensor, np.ndarray],
    grid_size: Tuple[int, int],
    has_cls_token: bool = True,
    target_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Convert a 1D attention flow vector to a normalized 2D spatial heatmap.

    Args:
        flow_vector: 1D array/tensor of shape [SeqLen].
        grid_size: (H_patches, W_patches), e.g. (14, 14) or (16, 12).
        has_cls_token: If True, index 0 is [CLS] token and is excluded from the spatial map.
        target_size: Optional (width, height) to resize output heatmap to.

    Returns:
        np.ndarray: Normalized 2D float heatmap in [0, 1] of shape [H, W].
    """
    if isinstance(flow_vector, torch.Tensor):
        vec = flow_vector.detach().cpu().numpy()
    else:
        vec = np.array(flow_vector)

    vec = vec.flatten()
    if has_cls_token:
        # Exclude [CLS] token at index 0
        patch_weights = vec[1:]
    else:
        patch_weights = vec

    gh, gw = grid_size
    if len(patch_weights) != gh * gw:
        raise ValueError(
            f"Token count ({len(patch_weights)}) does not match patch grid {gh}x{gw} = {gh * gw}"
        )

    heatmap = patch_weights.reshape(gh, gw)
    h_min, h_max = heatmap.min(), heatmap.max()
    heatmap = (heatmap - h_min) / (h_max - h_min + 1e-8)

    if target_size is not None:
        heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_CUBIC)
        heatmap = np.clip(heatmap, 0.0, 1.0)

    return heatmap


def overlay_flow_heatmap(
    image: Union[np.ndarray, torch.Tensor],
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a 2D attention flow heatmap onto an RGB image.

    Args:
        image: RGB image as uint8 [H, W, 3] or float tensor [C, H, W] in [0, 1].
        heatmap: 2D float heatmap in [0, 1] of shape [H, W].
        alpha: Blending weight for heatmap (1 - alpha for image).
        colormap: OpenCV colormap constant (e.g. cv2.COLORMAP_JET, cv2.COLORMAP_VIRIDIS).

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


def top_k_jaccard_similarity(
    flow_a: Union[torch.Tensor, np.ndarray, Sequence[float]],
    flow_b: Union[torch.Tensor, np.ndarray, Sequence[float]],
    k: int = 10,
) -> float:
    """Compute Top-K Token Retention (Jaccard Overlap / IOU) between two flow attributions.

    Defined in Section 4.1.1 of the FlowPrune paper:
        IOU = |A ∩ B| / |A ∪ B|
    where A and B are the sets of top-K highest-flow tokens.

    Args:
        flow_a: First flow attribution vector.
        flow_b: Second flow attribution vector (e.g. ground truth).
        k: Number of top-ranked tokens to consider.

    Returns:
        float: Jaccard overlap in [0.0, 1.0].
    """
    a = _to_numpy_1d(flow_a)
    b = _to_numpy_1d(flow_b)

    k_clamped = min(k, len(a), len(b))
    if k_clamped == 0:
        return 1.0

    top_a = set(np.argsort(-a)[:k_clamped])
    top_b = set(np.argsort(-b)[:k_clamped])

    intersection = len(top_a.intersection(top_b))
    union = len(top_a.union(top_b))

    return float(intersection) / float(max(1, union))


def first_rank_divergence(
    flow_a: Union[torch.Tensor, np.ndarray, Sequence[float]],
    flow_b: Union[torch.Tensor, np.ndarray, Sequence[float]],
    normalize: bool = True,
) -> float:
    """Compute First Rank Divergence (FRD).

    Defined in Section 4.1.1 of the FlowPrune paper:
    Identifies the first position where the ranking of token importance
    diverges between the approximated and ground-truth flows.

    Args:
        flow_a: First flow attribution vector.
        flow_b: Ground truth or reference flow vector.
        normalize: If True, normalizes the divergence index by sequence length.

    Returns:
        float: Divergence index (or normalized divergence ratio in [0, 1]).
    """
    a = _to_numpy_1d(flow_a)
    b = _to_numpy_1d(flow_b)

    seq_len = min(len(a), len(b))
    if seq_len == 0:
        return 0.0

    ranks_a = np.argsort(-a)
    ranks_b = np.argsort(-b)

    divergence_idx = seq_len
    for i in range(seq_len):
        if ranks_a[i] != ranks_b[i]:
            divergence_idx = i
            break

    if normalize:
        return float(divergence_idx) / float(seq_len)
    return float(divergence_idx)


def compute_approximation_errors(
    ground_truth: Union[torch.Tensor, np.ndarray],
    approx: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute absolute and relative approximation errors between two flow distributions.

    Args:
        ground_truth: True attention flow values.
        approx: Approximated attention flow values.
        eps: Small constant to avoid division by zero.

    Returns:
        Dict[str, float]: Dictionary containing:
            - 'mae': Mean Absolute Error
            - 'mse': Mean Squared Error
            - 'max_abs_err': Maximum Absolute Error
            - 'relative_error': Mean Relative Error (|gt - approx| / (gt + eps))
    """
    gt = _to_numpy_1d(ground_truth)
    ap = _to_numpy_1d(approx)

    abs_err = np.abs(gt - ap)
    rel_err = abs_err / (np.abs(gt) + eps)

    return {
        "mae": float(np.mean(abs_err)),
        "mse": float(np.mean(abs_err ** 2)),
        "max_abs_err": float(np.max(abs_err)),
        "relative_error": float(np.mean(rel_err)),
    }


def plot_flow_attribution_comparison(
    original_flow: Union[torch.Tensor, np.ndarray],
    flowprune_flow: Union[torch.Tensor, np.ndarray],
    rollout_flow: Optional[Union[torch.Tensor, np.ndarray]] = None,
    top_k: int = 15,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot bar chart comparing Top-K token attributions across methods.

    Args:
        original_flow: Exact attention flow vector.
        flowprune_flow: FlowPrune approximated flow vector.
        rollout_flow: Attention Rollout flow vector (optional).
        top_k: Number of highest-importance tokens to display.
        figsize: Figure size.
        save_path: Optional path to save the generated figure.

    Returns:
        plt.Figure: The created Matplotlib figure.
    """
    orig = _to_numpy_1d(original_flow)
    fp = _to_numpy_1d(flowprune_flow)

    top_indices = np.argsort(-orig)[:top_k]
    x = np.arange(len(top_indices))
    width = 0.25 if rollout_flow is not None else 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width, orig[top_indices], width, label="Exact Attention Flow", color="#2563EB")
    ax.bar(x, fp[top_indices], width, label="FlowPrune", color="#10B981")

    if rollout_flow is not None:
        ro = _to_numpy_1d(rollout_flow)
        ax.bar(x + width, ro[top_indices], width, label="Attention Rollout", color="#F59E0B")

    ax.set_xlabel("Token Index (Sorted by Ground Truth Importance)")
    ax.set_ylabel("Attribution Flow")
    ax.set_title(f"Top-{top_k} Attention Flow Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{idx}" for idx in top_indices], rotation=45)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def _to_numpy_1d(arr: Union[torch.Tensor, np.ndarray, Sequence[float]]) -> np.ndarray:
    """Convert input array/tensor/sequence to flat 1D numpy array."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy().flatten().astype(np.float32)
    return np.asarray(arr, dtype=np.float32).flatten()
