#!/usr/bin/env python3
"""Example demonstration of FlowPrune.

Demonstrates:
1. Simulating multi-layer, multi-head Vision Transformer attention maps.
2. Running FlowPrune with edge capacity pruning (theta = 1e-6) and layer compression.
3. Comparing FlowPrune with Attention Rollout on token attribution rankings.
4. Evaluating Top-K Token Retention (Jaccard IOU) and First Rank Divergence (FRD).
5. Converting [CLS] token flow vectors into 2D spatial heatmaps and overlaying on images.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import torch

from flowprune import (
    FlowPrune,
    compute_flowprune,
    first_rank_divergence,
    flow_to_heatmap,
    overlay_flow_heatmap,
    prune_and_compress_attention,
    top_k_jaccard_similarity,
)
from attention_rollout import compute_attention_rollout


def main():
    print("=" * 65)
    print("    FlowPrune: Attention Flow Network Pruning Demonstration")
    print("    Paper: Xu et al., NeurIPS 2025")
    print("=" * 65)

    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Setup Vision Transformer dimensions (e.g. DeiT-S / ViT-B/16)
    num_layers = 12
    num_heads = 8
    grid_h, grid_w = 7, 7
    seq_len = 1 + grid_h * grid_w  # 1 [CLS] token + 49 patch tokens = 50 tokens

    print(f"\n[1] Architecture Configuration:")
    print(f"    - Layers: {num_layers}")
    print(f"    - Attention Heads: {num_heads}")
    print(f"    - Sequence Length: {seq_len} (1 [CLS] + {grid_h}x{grid_w} patches)")

    # 2. Simulate realistic softmax attention matrices with sparse salient regions
    attentions = []
    for l in range(num_layers):
        raw_logits = torch.randn(num_heads, seq_len, seq_len) * 3.0
        # Induce salient target regions (e.g., center object patches)
        salient_indices = [0, 12, 18, 24, 25, 30, 36]
        raw_logits[:, :, salient_indices] += 4.0
        attn = torch.softmax(raw_logits, dim=-1)
        attentions.append(attn)

    # 3. Test FlowPrune Graph Reduction (Edge Pruning + DP Layer Compression)
    print(f"\n[2] Graph Compression across Retain Rates:")
    retain_rates = [0.80, 0.50, 0.25]
    pruning_threshold = 1e-4

    for rr in retain_rates:
        compacted, metadata = prune_and_compress_attention(
            attentions,
            retain_rate=rr,
            threshold=pruning_threshold,
            tactic="walk",
            head_fusion="mean",
        )
        orig_l = metadata["compression"]["original_layer_count"]
        comp_l = metadata["compression"]["compressed_layer_count"]
        cut_ratio = metadata["pruning"]["sparsity"] * 100.0
        compact_list = metadata["compression"]["compact_list"]

        print(
            f"    - Retain Rate {int(rr * 100)}%: "
            f"Layers {orig_l} -> {comp_l} | "
            f"Pruned Edges: {cut_ratio:.2f}% | "
            f"Compacted Intervals: {compact_list}"
        )

    # 4. Compute FlowPrune Attribution for [CLS] Token (query_token=0)
    print(f"\n[3] Computing Attention Flow for [CLS] token (retain_rate=0.25):")
    t0 = time.perf_counter()
    flowprune_cls = compute_flowprune(
        attentions,
        query_token=0,
        retain_rate=0.25,
        threshold=pruning_threshold,
        tactic="walk",
        head_fusion="mean",
        normalize_output=True,
        workers=4,
    )
    t_fp = time.perf_counter() - t0
    print(f"    - FlowPrune computed in {t_fp:.4f}s")
    print(f"    - Flow vector shape: {flowprune_cls.shape}")
    print(f"    - Flow sum: {flowprune_cls.sum().item():.4f} (normalized)")

    # 5. Compute Attention Rollout for Comparison
    t0 = time.perf_counter()
    rollout_matrix = compute_attention_rollout(
        attentions,
        discard_ratio=0.1,
        head_fusion="mean",
        residual_weight=0.5,
    )
    rollout_cls = rollout_matrix[0]  # [CLS] token row
    t_ro = time.perf_counter() - t0
    print(f"\n[4] Computing Attention Rollout:")
    print(f"    - Attention Rollout computed in {t_ro:.4f}s")
    print(f"    - Rollout vector shape: {rollout_cls.shape}")

    # 6. Quantitative Overlap & Rank Divergence
    print(f"\n[5] Alignment Metrics:")
    for k in [5, 10, 20]:
        iou = top_k_jaccard_similarity(flowprune_cls, rollout_cls, k=k)
        print(f"    - Top-{k} Token Overlap (Jaccard IOU): {iou * 100:.2f}%")

    frd = first_rank_divergence(flowprune_cls, rollout_cls, normalize=True)
    print(f"    - First Rank Divergence (FRD): {frd:.4f}")

    # 7. Convert to 2D Spatial Heatmap and Overlay
    print(f"\n[6] Spatial Heatmap Generation & Image Overlay:")
    img_h, img_w = 224, 224
    heatmap = flow_to_heatmap(
        flowprune_cls,
        grid_size=(grid_h, grid_w),
        has_cls_token=True,
        target_size=(img_w, img_h),
    )
    print(f"    - Heatmap shape: {heatmap.shape} (min: {heatmap.min():.3f}, max: {heatmap.max():.3f})")

    # Generate synthetic RGB test image
    dummy_image = np.full((img_h, img_w, 3), 220, dtype=np.uint8)
    # Add a visual object in center
    dummy_image[70:150, 70:150] = [80, 120, 160]

    overlay = overlay_flow_heatmap(dummy_image, heatmap, alpha=0.5)
    print(f"    - Blended overlay image shape: {overlay.shape}, dtype: {overlay.dtype}")

    print("\n" + "=" * 65)
    print("    FlowPrune demo completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
