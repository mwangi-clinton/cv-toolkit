"""Attention Rollout toolkit module.

This package provides a clean, robust PyTorch implementation of Attention Rollout
(Abir et al., 2020) for explaining Vision Transformer predictions and tracking
information flow through self-attention layers with residual connections.

Example
-------
>>> import torch
>>> from attention_rollout import compute_attention_rollout, AttentionRollout
>>> attentions = [torch.randn(1, 12, 197, 197).softmax(dim=-1) for _ in range(12)]
>>> rollout = compute_attention_rollout(attentions, discard_ratio=0.1)
>>> print("Rollout shape:", rollout.shape)
"""

from .rollout import (
    AttentionRollout,
    compute_attention_rollout,
    compute_batched_attention_rollout,
)
from .utils import (
    overlay_rollout_heatmap,
    plot_rollout_evolution,
    rollout_to_heatmap,
)

__all__ = [
    "AttentionRollout",
    "compute_attention_rollout",
    "compute_batched_attention_rollout",
    "overlay_rollout_heatmap",
    "plot_rollout_evolution",
    "rollout_to_heatmap",
]
