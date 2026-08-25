"""Grad-CAM toolkit module.

This package provides a clean PyTorch implementation of Grad-CAM,
Guided Backpropagation, and Guided Grad-CAM, suitable for explaining
decisions from any convolutional neural network.

Example
-------
>>> from gradcam import GradCAM, GuidedBackprop, visualize_gradcam
>>> import torchvision
>>> model = torchvision.models.resnet50(pretrained=True)
>>> gradcam = GradCAM(model, target_layer=model.layer4[-1])
>>> heatmap = gradcam(input_tensor, target_category=243)
>>> overlay = visualize_gradcam(input_tensor, heatmap)
"""

from .gradcam import GradCAM, overlay_heatmap
from .guided_backprop import GuidedBackprop
from .utils import (
    apply_colormap,
    denormalize_tensor,
    get_imagenet_transform,
    load_image,
    overlay_heatmap as overlay_heatmap_util,
    preprocess_image,
    tensor_to_image,
    visualize_gradcam,
)

__all__ = [
    "GradCAM",
    "GuidedBackprop",
    "apply_colormap",
    "denormalize_tensor",
    "get_imagenet_transform",
    "load_image",
    "overlay_heatmap",
    "preprocess_image",
    "tensor_to_image",
    "visualize_gradcam",
]
