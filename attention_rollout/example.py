#!/usr/bin/env python3
"""Example demonstration of Attention Rollout.

Shows:
1. Synthetic multi-head attention rollout computation.
2. Handling residual connections and noise thresholding.
3. Converting rollout vectors into 2D heatmaps and overlays.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from attention_rollout import (
    compute_attention_rollout,
    overlay_rollout_heatmap,
    rollout_to_heatmap,
)

def main():
    print("=== Attention Rollout Demonstration ===\n")

    torch.manual_seed(42)
    np.random.seed(42)

    num_layers = 12
    num_heads = 8
    grid_h, grid_w = 14, 14
    seq_len = 1 + grid_h * grid_w  # 1 CLS token + 196 patch tokens = 197

    # 1. Create simulated multi-layer attention matrices
    attentions = []
    for l in range(num_layers):
        # Softmax normalized raw attention weights
        raw = torch.randn(1, num_heads, seq_len, seq_len)
        attn = torch.softmax(raw, dim=-1)
        attentions.append(attn)

    print(f"Layers: {num_layers}, Heads: {num_heads}, SeqLen: {seq_len} (1 CLS + 196 patches)")

    # 2. Compute Attention Rollout with 10% discard threshold
    rollout_matrix = compute_attention_rollout(
        attentions,
        discard_ratio=0.1,
        head_fusion="mean",
        residual_weight=0.5,
    )
    print(f"Rollout matrix shape: {rollout_matrix.shape} -> [SeqLen, SeqLen]")

    # 3. Extract CLS token attention (row 0)
    cls_rollout = rollout_matrix.squeeze(0)[0]  # [197]
    print(f"CLS rollout vector shape: {cls_rollout.shape}")
    print(f"CLS rollout sum: {cls_rollout.sum().item():.4f} (normalized flow)")

    # 4. Convert to 2D spatial heatmap
    img_h, img_w = 224, 224
    heatmap = rollout_to_heatmap(
        cls_rollout,
        grid_size=(grid_h, grid_w),
        has_cls_token=True,
        target_size=(img_w, img_h),
    )
    print(f"Spatial heatmap shape: {heatmap.shape}, min: {heatmap.min():.3f}, max: {heatmap.max():.3f}")

    # 5. Overlay on dummy image
    dummy_img = np.full((img_h, img_w, 3), 200, dtype=np.uint8)
    overlay = overlay_rollout_heatmap(dummy_img, heatmap, alpha=0.6)
    print(f"Blended overlay shape: {overlay.shape}")

    print("\nDemo finished successfully!")

if __name__ == "__main__":
    main()
