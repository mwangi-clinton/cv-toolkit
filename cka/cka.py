"""Centered Kernel Alignment (CKA) core implementation.

Implements Linear CKA, Kernel CKA, and streaming minibatch CKA based on:
    "Similarity of Neural Network Representations Revisited"
    Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton. ICML 2019.
    https://arxiv.org/abs/1905.00414
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


def _to_tensor(x: Union[torch.Tensor, np.ndarray], device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert numpy array or tensor to torch.Tensor."""
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        t = x
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)}")
    if device is not None:
        t = t.to(device)
    if t.dtype in (torch.float16, torch.float64):
        t = t.float()
    return t


def linear_cka(
    X: Union[torch.Tensor, np.ndarray],
    Y: Union[torch.Tensor, np.ndarray],
    debiased: bool = False,
) -> float:
    """Compute Linear Centered Kernel Alignment (Linear CKA) between X and Y.

    Linear CKA measures the similarity of two feature matrices invariant to
    orthogonal transformations and isotropic scaling.

    Args:
        X: First representation matrix of shape [N, D1].
        Y: Second representation matrix of shape [N, D2].
        debiased: If True, uses unbiased HSIC estimator (Song et al., 2012).
            Recommended when sample size N is small (< 100).

    Returns:
        float: Linear CKA similarity score in [0, 1].

    Raises:
        ValueError: If X and Y do not have the same number of samples N.
    """
    X_t = _to_tensor(X)
    Y_t = _to_tensor(Y, device=X_t.device)

    if X_t.shape[0] != Y_t.shape[0]:
        raise ValueError(
            f"Sample size mismatch: X has {X_t.shape[0]} samples, "
            f"Y has {Y_t.shape[0]} samples."
        )

    N, D1 = X_t.shape
    _, D2 = Y_t.shape

    with torch.no_grad():
        # Center representations
        X_c = X_t - X_t.mean(dim=0, keepdim=True)
        Y_c = Y_t - Y_t.mean(dim=0, keepdim=True)

        if not debiased:
            # Memory-efficient branch selection based on dimensions
            # If N <= max(D1, D2), Gram-matrix calculation is faster & uses less memory
            if N <= max(D1, D2):
                K = X_c @ X_c.T  # [N, N]
                L = Y_c @ Y_c.T  # [N, N]
                hsic_xy = (K * L).sum()
                hsic_xx = (K * K).sum().sqrt()
                hsic_yy = (L * L).sum().sqrt()
            else:
                # When N >> D, cross-covariance calculation is O(D1*D2*N)
                cov_xy = X_c.T @ Y_c  # [D1, D2]
                cov_xx = X_c.T @ X_c  # [D1, D1]
                cov_yy = Y_c.T @ Y_c  # [D2, D2]
                hsic_xy = (cov_xy ** 2).sum()
                hsic_xx = (cov_xx ** 2).sum().sqrt()
                hsic_yy = (cov_yy ** 2).sum().sqrt()

            similarity = hsic_xy / (hsic_xx * hsic_yy + 1e-10)
            return float(torch.clamp(similarity, 0.0, 1.0).item())

        else:
            # Unbiased estimator (Kornblith et al., 2019 / Song et al., 2012)
            K = X_c @ X_c.T
            L = Y_c @ Y_c.T
            # Zero out diagonal
            K.fill_diagonal_(0.0)
            L.fill_diagonal_(0.0)

            hsic_xy = (K * L).sum() / (N * (N - 3) + 1e-10)
            hsic_xx = (K * K).sum() / (N * (N - 3) + 1e-10)
            hsic_yy = (L * L).sum() / (N * (N - 3) + 1e-10)

            similarity = hsic_xy / ((hsic_xx * hsic_yy).sqrt() + 1e-10)
            return float(torch.clamp(similarity, 0.0, 1.0).item())


def kernel_cka(
    X: Union[torch.Tensor, np.ndarray],
    Y: Union[torch.Tensor, np.ndarray],
    sigma: Optional[float] = None,
    kernel: str = "rbf",
) -> float:
    """Compute Kernel Centered Kernel Alignment (Kernel CKA).

    Args:
        X: First representation matrix of shape [N, D1].
        Y: Second representation matrix of shape [N, D2].
        sigma: Kernel bandwidth for RBF kernel. If None, uses the median heuristic.
        kernel: Kernel function ("rbf" or "linear").

    Returns:
        float: Kernel CKA similarity score in [0, 1].
    """
    if kernel == "linear":
        return linear_cka(X, Y)

    X_t = _to_tensor(X)
    Y_t = _to_tensor(Y, device=X_t.device)

    if X_t.shape[0] != Y_t.shape[0]:
        raise ValueError(f"Sample count mismatch: X({X_t.shape[0]}) vs Y({Y_t.shape[0]})")

    N = X_t.shape[0]

    with torch.no_grad():
        def _rbf_gram(M: torch.Tensor, bandwidth: Optional[float] = None) -> torch.Tensor:
            dist_sq = torch.cdist(M, M, p=2) ** 2
            if bandwidth is None:
                # Median heuristic
                median_dist = torch.median(dist_sq[dist_sq > 0])
                bandwidth = float(torch.sqrt(median_dist / 2.0).item())
                if bandwidth < 1e-6:
                    bandwidth = 1.0
            return torch.exp(-dist_sq / (2.0 * bandwidth ** 2))

        K = _rbf_gram(X_t, sigma)
        L = _rbf_gram(Y_t, sigma)

        # Center Gram matrices: H K H where H = I - 1/N 1 1^T
        # Efficient centering without explicit H matrix:
        K_c = K - K.mean(dim=0, keepdim=True) - K.mean(dim=1, keepdim=True) + K.mean()
        L_c = L - L.mean(dim=0, keepdim=True) - L.mean(dim=1, keepdim=True) + L.mean()

        hsic_xy = (K_c * L_c).sum()
        hsic_xx = (K_c * K_c).sum().sqrt()
        hsic_yy = (L_c * L_c).sum().sqrt()

        similarity = hsic_xy / (hsic_xx * hsic_yy + 1e-10)
        return float(torch.clamp(similarity, 0.0, 1.0).item())


def compute_cka_matrix(
    features_list: Sequence[Union[torch.Tensor, np.ndarray]],
    kernel: str = "linear",
    debiased: bool = False,
) -> np.ndarray:
    """Compute pairwise CKA similarity matrix for a sequence of layer features.

    Optimized using precomputed centered Gram matrices when kernel='linear'.

    Args:
        features_list: Sequence of L feature tensors/arrays, each of shape [N, D_l].
        kernel: 'linear' or 'rbf'.
        debiased: Whether to use unbiased estimator.

    Returns:
        np.ndarray: [L, L] symmetric matrix with diagonal = 1.0.
    """
    L = len(features_list)
    if L == 0:
        return np.empty((0, 0))

    matrix = np.eye(L, dtype=np.float32)
    if L == 1:
        return matrix

    # Linear fast-path via Gram matrices
    if kernel == "linear" and not debiased:
        tensors = [_to_tensor(f) for f in features_list]
        device = tensors[0].device
        tensors = [t.to(device) for t in tensors]

        with torch.no_grad():
            centered = [t - t.mean(dim=0, keepdim=True) for t in tensors]
            grams = [(F @ F.T) for F in centered]
            norms = [(G * G).sum().sqrt() for G in grams]

            for i in range(L):
                for j in range(i + 1, L):
                    hsic = (grams[i] * grams[j]).sum().item()
                    denom = (norms[i] * norms[j]).item()
                    sim = max(0.0, min(1.0, hsic / (denom + 1e-10)))
                    matrix[i, j] = sim
                    matrix[j, i] = sim
        return matrix

    # Fallback to pairwise call
    for i in range(L):
        for j in range(i + 1, L):
            if kernel == "linear":
                sim = linear_cka(features_list[i], features_list[j], debiased=debiased)
            else:
                sim = kernel_cka(features_list[i], features_list[j], kernel=kernel)
            matrix[i, j] = sim
            matrix[j, i] = sim

    return matrix


def compute_layer_cka_matrix(
    model1: nn.Module,
    model2: Optional[nn.Module],
    dataloader: torch.utils.data.DataLoader,
    model1_layers: Sequence[str],
    model2_layers: Optional[Sequence[str]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> np.ndarray:
    """Compute CKA similarity matrix between specified layers of two models across a dataloader.

    If model2 is None, computes layer-to-layer similarity within model1.
    """
    comparator = CKA(
        model1=model1,
        model2=model2,
        model1_layers=model1_layers,
        model2_layers=model2_layers,
        device=device,
    )
    return comparator.compare(dataloader)


class CKA:
    """Centered Kernel Alignment comparator for PyTorch models.

    Supports forward-hook extraction, model-to-model comparison, and streaming
    minibatch evaluation for datasets that cannot fit in memory.

    Args:
        model1: First PyTorch neural network.
        model2: Second PyTorch neural network. If None, compares model1 with itself.
        model1_layers: Names or list of submodule names in model1 to extract.
        model2_layers: Names or list of submodule names in model2 to extract.
        device: Device to run models and extraction on.
    """

    def __init__(
        self,
        model1: nn.Module,
        model2: Optional[nn.Module] = None,
        model1_layers: Optional[Sequence[str]] = None,
        model2_layers: Optional[Sequence[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model1 = model1
        self.model2 = model2 if model2 is not None else model1
        self.same_model = (model2 is None or model2 is model1)

        if device is None:
            self.device = next(model1.parameters()).device
        else:
            self.device = torch.device(device)

        self.model1_layers = list(model1_layers) if model1_layers is not None else []
        self.model2_layers = list(model2_layers) if model2_layers is not None else (
            self.model1_layers if self.same_model else []
        )

        self._hooks1: List[torch.utils.hooks.RemovableHandle] = []
        self._hooks2: List[torch.utils.hooks.RemovableHandle] = []
        self._acts1: Dict[str, List[torch.Tensor]] = {k: [] for k in self.model1_layers}
        self._acts2: Dict[str, List[torch.Tensor]] = {k: [] for k in self.model2_layers}

    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        self._remove_hooks()
        named_modules1 = dict(self.model1.named_modules())
        for name in self.model1_layers:
            if name not in named_modules1:
                raise KeyError(f"Layer '{name}' not found in model1")
            mod = named_modules1[name]
            h = mod.register_forward_hook(self._make_hook(name, self._acts1))
            self._hooks1.append(h)

        if not self.same_model:
            named_modules2 = dict(self.model2.named_modules())
            for name in self.model2_layers:
                if name not in named_modules2:
                    raise KeyError(f"Layer '{name}' not found in model2")
                mod = named_modules2[name]
                h = mod.register_forward_hook(self._make_hook(name, self._acts2))
                self._hooks2.append(h)

    def _make_hook(self, name: str, storage: Dict[str, List[torch.Tensor]]) -> Callable:
        def hook(module, inp, output):
            out = output.detach()
            # If 4D (CNN feature map B, C, H, W) -> flatten to (B, C * H * W)
            if out.dim() == 4:
                out = out.reshape(out.shape[0], -1)
            # If 3D (Transformer tokens B, S, D) -> flatten to (B, S * D) or pool
            elif out.dim() == 3:
                out = out.reshape(out.shape[0], -1)
            storage[name].append(out.cpu())
        return hook

    def _remove_hooks(self):
        for h in self._hooks1:
            h.remove()
        for h in self._hooks2:
            h.remove()
        self._hooks1.clear()
        self._hooks2.clear()

    def compare(self, dataloader: torch.utils.data.DataLoader, max_batches: Optional[int] = None) -> np.ndarray:
        """Run forward passes over dataloader and compute the CKA similarity matrix.

        Returns:
            np.ndarray: [len(model1_layers), len(model2_layers)] CKA similarity matrix.
        """
        self._acts1 = {k: [] for k in self.model1_layers}
        self._acts2 = {k: [] for k in self.model2_layers}
        self._register_hooks()

        self.model1.eval()
        if not self.same_model:
            self.model2.eval()

        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if max_batches is not None and batch_idx >= max_batches:
                        break
                    # Extract inputs
                    if isinstance(batch, (tuple, list)):
                        x = batch[0].to(self.device)
                    elif isinstance(batch, dict):
                        x = batch.get("pixel_values", batch.get("input", batch.get("x"))).to(self.device)
                    else:
                        x = batch.to(self.device)

                    self.model1(x)
                    if not self.same_model:
                        self.model2(x)
        finally:
            self._remove_hooks()

        # Concatenate activations across batches
        feats1 = [torch.cat(self._acts1[k], dim=0) for k in self.model1_layers]
        if self.same_model:
            feats2 = feats1
        else:
            feats2 = [torch.cat(self._acts2[k], dim=0) for k in self.model2_layers]

        n1, n2 = len(feats1), len(feats2)
        matrix = np.zeros((n1, n2), dtype=np.float32)

        for i in range(n1):
            for j in range(n2):
                matrix[i, j] = linear_cka(feats1[i], feats2[j])

        return matrix

try:
    from .utils import plot_cka_matrix, centering, gram_linear, gram_rbf
except ImportError:
    from utils import plot_cka_matrix, centering, gram_linear, gram_rbf
