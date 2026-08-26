"""
Grad-CAM: Gradient-weighted Class Activation Mapping

PyTorch implementation of Grad-CAM from:

    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.

This module provides a reusable `GradCAM` class that can be attached to any
PyTorch CNN and any target convolutional layer. Given an input image and a
target class (or any differentiable scalar output), it produces a coarse
localization map highlighting the regions of the image that mattered for
that decision.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _set_relu_inplace(model: nn.Module, inplace: bool) -> Dict[int, bool]:
    """Set ``inplace`` on every ReLU in ``model``.

    Returns a mapping of module id -> original ``inplace`` value only for
    modules that were actually changed. This lets multiple explainers share
    the same model without fighting over the restoration order.
    """
    original: Dict[int, bool] = {}
    for module in model.modules():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
            current = bool(module.inplace)
            if current != inplace:
                original[id(module)] = current
                module.inplace = inplace
    return original


class GradCAM:
    """Compute Grad-CAM localization maps for a target layer and class.

    The implementation follows the paper exactly:

        alpha_k^c = (1 / Z) * sum_ij d(y^c) / d(A^k_ij)
        L^c_Grad-CAM = ReLU(sum_k alpha_k^c * A^k)

    where `A^k` are the forward activation maps of the target convolutional
    layer and `y^c` is the raw score for class `c` before the softmax.

    Usage
    -----
    >>> model = torchvision.models.resnet50(pretrained=True)
    >>> gradcam = GradCAM(model, target_layer=model.layer4[-1])
    >>> heatmap = gradcam(input_tensor, target_category=243)
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        use_cuda: Optional[bool] = None,
    ) -> None:
        """Initialize GradCAM.

        Parameters
        ----------
        model : torch.nn.Module
            A CNN model that returns class logits (before softmax).
        target_layer : torch.nn.Module
            The convolutional layer whose activations and gradients are used
            to compute Grad-CAM. Typically the last convolutional layer.
        use_cuda : bool, optional
            Whether to run on GPU. If None, inferred from model parameters.
        """
        self.model = model
        self.target_layer = target_layer
        self.use_cuda = use_cuda if use_cuda is not None else next(model.parameters()).is_cuda

        # Storage for hooks
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Backward hooks are incompatible with inplace ReLU activations, so
        # disable inplace mode while this explainer is attached.
        self._relu_inplace_original = _set_relu_inplace(model, inplace=False)

        # Register forward and backward hooks
        self._forward_handle = target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

        self.model.eval()

    def _save_activation(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        """Forward hook: store the output activations of the target layer."""
        self._activations = output.detach()

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor, ...],
        grad_output: Tuple[torch.Tensor, ...],
    ) -> None:
        """Backward hook: store the gradients flowing into the target layer."""
        self._gradients = grad_output[0].detach()

    def _get_target_score(
        self,
        model_output: torch.Tensor,
        target_category: Optional[int],
    ) -> torch.Tensor:
        """Select or infer the scalar score to explain.

        If ``target_category`` is None, the predicted class is used.
        """
        if target_category is None:
            target_category = int(model_output.argmax(dim=-1).item())
            self._last_target_category = target_category
        else:
            self._last_target_category = target_category

        return model_output[:, target_category]

    def forward(
        self,
        input_tensor: torch.Tensor,
        target_category: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute the Grad-CAM heatmap for a single input.

        Parameters
        ----------
        input_tensor : torch.Tensor
            A single image tensor of shape ``(1, C, H, W)`` in the model's
            expected preprocessing (e.g. ImageNet normalization).
        target_category : int, optional
            Class index to explain. If None, the predicted class is used.

        Returns
        -------
        torch.Tensor
            Grad-CAM heatmap of shape ``(1, 1, H', W')`` where ``H'`` and
            ``W'`` match the spatial size of ``target_layer`` output.
        """
        if input_tensor.dim() != 4 or input_tensor.shape[0] != 1:
            raise ValueError("GradCAM.forward expects a batch of size 1 with shape (1, C, H, W)")

        input_tensor = input_tensor.clone().requires_grad_(True)
        if self.use_cuda:
            input_tensor = input_tensor.cuda()

        # Forward pass
        model_output = self.model(input_tensor)

        # Select scalar to explain
        target_score = self._get_target_score(model_output, target_category)

        # Zero gradients and backpropagate
        self.model.zero_grad()
        target_score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients. Is target_layer a convolutional layer?")

        activations = self._activations  # (1, K, U, V)
        gradients = self._gradients      # (1, K, U, V)

        # Global average pool gradients over spatial dimensions -> (1, K, 1, 1)
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, U, V)

        # Apply ReLU: keep only features that positively influence the class
        cam = F.relu(cam)

        # Normalize to [0, 1] for visualization
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_category: Optional[int] = None,
    ) -> torch.Tensor:
        return self.forward(input_tensor, target_category)

    def remove_hooks(self) -> None:
        """Remove the registered forward/backward hooks and restore ReLU inplace settings."""
        self._forward_handle.remove()
        self._backward_handle.remove()

        for module in self.model.modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU)) and id(module) in self._relu_inplace_original:
                module.inplace = self._relu_inplace_original[id(module)]

    @property
    def last_target_category(self) -> Optional[int]:
        """Return the class index used in the most recent forward call."""
        return getattr(self, "_last_target_category", None)


def overlay_heatmap(
    heatmap: np.ndarray,
    image: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on top of the original RGB image.

    Parameters
    ----------
    heatmap : np.ndarray
        Single-channel heatmap with values in ``[0, 1]`` and shape
        ``(H, W)`` or ``(H, W, 1)``.
    image : np.ndarray
        Original RGB image with values in ``[0, 255]`` and shape
        ``(H, W, 3)``.
    alpha : float
        Opacity of the heatmap overlay.
    colormap : int
        OpenCV colormap to use.

    Returns
    -------
    np.ndarray
        Blended RGB image of shape ``(H, W, 3)``.
    """
    heatmap = np.squeeze(heatmap)
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    heatmap_uint8 = np.uint8(255 * heatmap)
    color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

    if image.dtype != np.uint8:
        image = np.uint8(np.clip(image, 0, 255))

    overlaid = cv2.addWeighted(image, 1 - alpha, color_heatmap, alpha, 0)
    return overlaid
