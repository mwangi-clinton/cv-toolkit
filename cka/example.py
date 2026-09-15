#!/usr/bin/env python3
"""Example demonstration of CKA (Centered Kernel Alignment).

Shows:
1. Direct Linear CKA between two feature matrices.
2. Pairwise CKA matrix computation across simulated layers.
3. Visualization and saving of the CKA heatmap.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from cka import linear_cka, kernel_cka, compute_cka_matrix, plot_cka_matrix

def main():
    print("=== Centered Kernel Alignment (CKA) Demo ===\n")

    torch.manual_seed(42)
    np.random.seed(42)

    N = 128  # Number of samples (e.g. images)
    D = 256  # Representation dimension

    # 1. Compare identical, orthogonal, and noisy representations
    X = torch.randn(N, D)
    R, _ = torch.linalg.qr(torch.randn(D, D))  # Orthogonal rotation matrix
    Y_rotated = X @ R                          # Orthogonal transformation of X
    Z_noise = torch.randn(N, D)                # Independent noise

    cka_identical = linear_cka(X, X)
    cka_rotated = linear_cka(X, Y_rotated)
    cka_noise = linear_cka(X, Z_noise)

    print(f"CKA(X, X) [Identical]:                 {cka_identical:.4f} (expected ~1.0)")
    print(f"CKA(X, X @ R) [Orthogonal Invariance]: {cka_rotated:.4f} (expected ~1.0)")
    print(f"CKA(X, Noise) [Independent]:           {cka_noise:.4f} (expected ~0.0)\n")

    # 2. Simulate 8-layer network representations with progressive drift
    layers = [X]
    current = X.clone()
    for l in range(1, 8):
        # Gradual transformation across layers
        current = 0.8 * current + 0.2 * torch.randn(N, D)
        layers.append(current)

    layer_labels = [f"L{i+1}" for i in range(8)]

    print("Computing 8x8 pairwise CKA matrix...")
    matrix = compute_cka_matrix(layers, kernel="linear")
    print("CKA Matrix shape:", matrix.shape)

    # 3. Plot matrix (headless safe)
    fig = plot_cka_matrix(
        matrix,
        layer_labels_x=layer_labels,
        layer_labels_y=layer_labels,
        title="Simulated Layer-to-Layer CKA Matrix",
        save_path="cka_demo_matrix.png",
        show=False,
    )
    print("Demo complete! Output saved to cka_demo_matrix.png")

if __name__ == "__main__":
    main()
