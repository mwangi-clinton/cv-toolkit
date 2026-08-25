"""
Utility functions for image preprocessing, postprocessing, and visualization.

These helpers make it easy to go from a PIL image / file path to a model-ready
PyTorch tensor, and from a Grad-CAM heatmap back to a displayable RGB image.
"""

from pathlib import Path
from typing import Tuple, Union
from urllib.parse import urlparse
import urllib.request

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ImageNet mean and standard deviation used by torchvision pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_imagenet_transform(input_size: Union[int, Tuple[int, int]] = 224) -> transforms.Compose:
    """Return a torchvision transform for pretrained ImageNet models.

    Parameters
    ----------
    input_size : int or tuple
        Target spatial size. If int, the shorter side is resized to this value
        and a center crop is taken.

    Returns
    -------
    torchvision.transforms.Compose
    """
    if isinstance(input_size, int):
        size = (input_size, input_size)
    else:
        size = input_size

    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_image(path: Union[str, Path]) -> Image.Image:
    """Load an image as a PIL RGB image.

    Supports local file paths and HTTP/HTTPS URLs.
    """
    if isinstance(path, str) and urlparse(path).scheme in ("http", "https"):
        return Image.open(urllib.request.urlopen(path)).convert("RGB")
    return Image.open(path).convert("RGB")


def preprocess_image(
    image: Union[str, Path, Image.Image],
    input_size: Union[int, Tuple[int, int]] = 224,
) -> torch.Tensor:
    """Load and preprocess an image for a torchvision pretrained model.

    Parameters
    ----------
    image : str, Path, or PIL.Image
        Input image.
    input_size : int or tuple
        Target size passed to ``get_imagenet_transform``.

    Returns
    -------
    torch.Tensor
        Normalized image tensor of shape ``(1, 3, H, W)``.
    """
    if isinstance(image, (str, Path)):
        image = load_image(image)

    transform = get_imagenet_transform(input_size)
    tensor = transform(image).unsqueeze(0)
    return tensor


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization for display purposes.

    Parameters
    ----------
    tensor : torch.Tensor
        Normalized tensor of shape ``(1, 3, H, W)`` or ``(3, H, W)``.

    Returns
    -------
    torch.Tensor
        Tensor with values approximately in ``[0, 1]``.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(-1, 1, 1)

    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()

    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return tensor * std + mean


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized torch tensor to a displayable uint8 RGB image.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor of shape ``(1, 3, H, W)`` or ``(3, H, W)``.

    Returns
    -------
    np.ndarray
        RGB image array of shape ``(H, W, 3)`` with dtype ``uint8``.
    """
    tensor = denormalize_tensor(tensor)
    tensor = tensor.detach().cpu()

    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    image = tensor.permute(1, 2, 0).numpy()
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


def apply_colormap(heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Apply an OpenCV colormap to a single-channel heatmap.

    Parameters
    ----------
    heatmap : np.ndarray
        Heatmap with values in ``[0, 1]`` and shape ``(H, W)``.
    colormap : int
        OpenCV colormap constant.

    Returns
    -------
    np.ndarray
        RGB image of shape ``(H, W, 3)``.
    """
    heatmap = np.squeeze(heatmap)
    heatmap_uint8 = np.uint8(255 * heatmap)
    color = cv2.applyColorMap(heatmap_uint8, colormap)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def overlay_heatmap(
    heatmap: np.ndarray,
    image: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a heatmap on an RGB image.

    Parameters
    ----------
    heatmap : np.ndarray
        Single-channel heatmap in ``[0, 1]``.
    image : np.ndarray
        RGB image in ``[0, 255]`` with shape ``(H, W, 3)``.
    alpha : float
        Heatmap opacity.
    colormap : int
        OpenCV colormap.

    Returns
    -------
    np.ndarray
        Blended RGB image.
    """
    heatmap = np.squeeze(heatmap)
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    color_heatmap = apply_colormap(heatmap, colormap)

    if image.dtype != np.uint8:
        image = np.uint8(np.clip(image, 0, 255))

    return cv2.addWeighted(image, 1 - alpha, color_heatmap, alpha, 0)


def visualize_gradcam(
    model_input: torch.Tensor,
    heatmap: torch.Tensor,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Convenience function: normalized input tensor + Grad-CAM heatmap → overlay.

    Parameters
    ----------
    model_input : torch.Tensor
        The normalized input tensor of shape ``(1, 3, H, W)`` that was passed
        to the model.
    heatmap : torch.Tensor
        Grad-CAM heatmap tensor of shape ``(1, 1, U, V)``.
    alpha : float
        Heatmap opacity.
    colormap : int
        OpenCV colormap.

    Returns
    -------
    np.ndarray
        Overlay image of shape ``(H, W, 3)``.
    """
    image = tensor_to_image(model_input)
    heatmap_np = heatmap.detach().cpu().numpy()
    return overlay_heatmap(heatmap_np, image, alpha=alpha, colormap=colormap)
