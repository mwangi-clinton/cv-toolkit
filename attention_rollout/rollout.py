"""Attention Rollout core implementation.

Implements Attention Rollout based on:
    "Quantifying Attention Flow in Transformers"
    Samira Abnar, Willem Zuidema. ACL 2020.
    https://arxiv.org/abs/2005.00928
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


def compute_attention_rollout(
    attentions: Sequence[torch.Tensor],
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
    residual_weight: float = 0.5,
) -> torch.Tensor:
    """Compute attention rollout across transformer layers for a single sample or batch.

    Attention Rollout recursively multiplies attention matrices across layers
    while accounting for the identity residual connection:
        A_l = (1 - alpha) * I + alpha * fuse(A_l)
        R_l = A_l @ R_{l-1}

    Args:
        attentions: Sequence of L attention tensors, each of shape:
            [NumHeads, SeqLen, SeqLen] or [Batch, NumHeads, SeqLen, SeqLen].
        discard_ratio: Fraction (0.0 to 1.0) of lowest attention values to zero out
            at each layer to suppress noise (as proposed by Abnar & Zuidema).
        head_fusion: Strategy to fuse multi-head attention ('mean', 'max', 'min').
        residual_weight: Weight alpha for attention matrix relative to residual identity (1 - alpha).
            Default is 0.5 (equal weight between attention and identity skip connection).

    Returns:
        torch.Tensor: Computed rollout matrix of shape:
            [SeqLen, SeqLen] if input was single sample, or
            [Batch, SeqLen, SeqLen] if input was batched.

    Raises:
        ValueError: If attentions sequence is empty or tensors have unsupported shape.
    """
    if not attentions:
        raise ValueError("attentions sequence is empty.")

    first = attentions[0]
    is_batched = (first.dim() == 4)

    if not is_batched:
        # Wrap single sample into batch of size 1
        batched_attns = [a.unsqueeze(0) for a in attentions]
    else:
        batched_attns = list(attentions)

    result = compute_batched_attention_rollout(
        batched_attns,
        discard_ratio=discard_ratio,
        head_fusion=head_fusion,
        residual_weight=residual_weight,
    )

    if not is_batched:
        return result.squeeze(0)
    return result


def compute_batched_attention_rollout(
    attentions: Sequence[torch.Tensor],
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
    residual_weight: float = 0.5,
) -> torch.Tensor:
    """GPU-accelerated batched Attention Rollout using torch.bmm.

    Args:
        attentions: Sequence of L attention tensors of shape [Batch, NumHeads, SeqLen, SeqLen].
        discard_ratio: Fraction of lowest attention weights to discard (0.0 to 1.0).
        head_fusion: 'mean', 'max', or 'min'.
        residual_weight: Weight alpha for attention vs. residual connection (default 0.5).

    Returns:
        torch.Tensor: [Batch, SeqLen, SeqLen] rollout matrix.
    """
    if not attentions:
        raise ValueError("attentions sequence is empty.")

    if attentions[0].dim() != 4:
        raise ValueError(
            f"Expected 4D attention tensors [Batch, NumHeads, SeqLen, SeqLen], "
            f"got shape {attentions[0].shape}"
        )

    batch_size, num_heads, seq_len, _ = attentions[0].shape
    device = attentions[0].device

    eye_matrix = torch.eye(seq_len, device=device, dtype=attentions[0].dtype).unsqueeze(0)
    result = eye_matrix.expand(batch_size, -1, -1).clone()

    alpha = float(residual_weight)
    beta = 1.0 - alpha

    with torch.no_grad():
        for layer_idx, attention in enumerate(attentions):
            if attention.shape[0] != batch_size or attention.shape[-1] != seq_len:
                raise ValueError(
                    f"Attention tensor at layer {layer_idx} has mismatched shape {attention.shape} "
                    f"(expected [Batch={batch_size}, ..., SeqLen={seq_len}])"
                )

            # 1. Head Fusion
            if head_fusion == "mean":
                attn_weights = attention.mean(dim=1)  # [Batch, SeqLen, SeqLen]
            elif head_fusion == "max":
                attn_weights = attention.max(dim=1)[0]
            elif head_fusion == "min":
                attn_weights = attention.min(dim=1)[0]
            else:
                raise ValueError(f"Unknown head_fusion '{head_fusion}'. Use 'mean', 'max', or 'min'.")

            # 2. Discard lowest attention weights (noise suppression)
            if discard_ratio > 0.0:
                flat = attn_weights.reshape(batch_size, -1)
                k = int(flat.shape[1] * discard_ratio)
                if k > 0:
                    val, _ = torch.kthvalue(flat, k, dim=-1, keepdim=True)
                    threshold = val.unsqueeze(-1)  # [Batch, 1, 1]
                    attn_weights = torch.where(
                        attn_weights < threshold,
                        torch.zeros_like(attn_weights),
                        attn_weights,
                    )

            # 3. Add residual connection & re-normalize
            # A_l = alpha * A_l + beta * I
            attn_weights = alpha * attn_weights + beta * eye_matrix
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-10)

            # 4. Multiply recursively: R = A_l @ R
            result = torch.bmm(attn_weights, result)

    return result


class AttentionRollout:
    """High-level Attention Rollout manager for Vision Transformers.

    Attaches forward hooks to multi-head self-attention modules of a transformer
    model, collects attention weights during inference, and computes rollout maps
    for classification ([CLS] token) or dense queries (spatial patch tokens).

    Args:
        model: PyTorch transformer model.
        attention_layer_name: Substring or module type to identify attention layers.
            Common values: 'attn', 'self_attn', 'attention'.
        head_fusion: 'mean', 'max', or 'min'.
        discard_ratio: Thresholding ratio (0.0 to 1.0).
        residual_weight: Alpha weight for attention vs. residual (default 0.5).
        device: Device to run on.
    """

    def __init__(
        self,
        model: nn.Module,
        attention_layer_name: str = "attn",
        head_fusion: str = "mean",
        discard_ratio: float = 0.0,
        residual_weight: float = 0.5,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.attention_layer_name = attention_layer_name
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.residual_weight = residual_weight
        self.device = device or next(model.parameters()).device

        self.attentions: List[torch.Tensor] = []
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self):
        self._remove_hooks()
        for name, module in self.model.named_modules():
            if self.attention_layer_name in name.lower():
                # Check if this module outputs attention weights or has an internal drop/attn
                h = module.register_forward_hook(self._hook_fn)
                self._hooks.append(h)

    def _hook_fn(self, module, inp, output):
        # HuggingFace / timm often return (attn_output, attn_weights) or tensor
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            self.attentions.append(output[1].detach().cpu())
        elif isinstance(output, torch.Tensor) and output.dim() == 4:
            self.attentions.append(output.detach().cpu())

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def remove_hooks(self):
        """Remove registered hooks."""
        self._remove_hooks()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        query_index: Optional[int] = 0,
        attentions: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Run forward pass and return attention rollout vector for query_index.

        Args:
            input_tensor: Model input tensor [B, C, H, W].
            query_index: Token index to compute rollout from.
                - 0: typically [CLS] token in classification ViTs.
                - k: specific spatial patch token in dense/pose ViTs.
                If None, returns full rollout matrix [B, SeqLen, SeqLen].
            attentions: Precomputed sequence of attentions if hooks are not used.

        Returns:
            torch.Tensor: Rollout attention vector [B, SeqLen] or matrix [B, SeqLen, SeqLen].
        """
        if attentions is None:
            self.attentions.clear()
            self.model.eval()
            with torch.no_grad():
                _ = self.model(input_tensor.to(self.device))
            attns = self.attentions
        else:
            attns = list(attentions)

        if not attns:
            raise RuntimeError(
                "No attention tensors collected. Ensure model outputs attention weights "
                "or pass `output_attentions=True`."
            )

        rollout_matrix = compute_batched_attention_rollout(
            attns,
            discard_ratio=self.discard_ratio,
            head_fusion=self.head_fusion,
            residual_weight=self.residual_weight,
        )

        if query_index is not None:
            # Query row represents where query token gathers information from
            return rollout_matrix[:, query_index, :]
        return rollout_matrix

try:
    from .utils import overlay_rollout_heatmap, plot_rollout_evolution, rollout_to_heatmap
except ImportError:
    from utils import overlay_rollout_heatmap, plot_rollout_evolution, rollout_to_heatmap
