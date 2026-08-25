# Grad-CAM Toolkit

A clean, reusable **PyTorch** implementation of **Grad-CAM** (Gradient-weighted Class Activation Mapping) for explaining decisions from any CNN.

This module is based on the paper:

> **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization**  
> Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra  
> *ICCV 2017*  
> [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)

The original reference implementation is in Lua/Torch ([ramprs/grad-cam](https://github.com/ramprs/grad-cam/)). This Python port is designed for modern PyTorch models and for easy integration into the `computer-vision-toolkit` repo.

---

## Table of Contents

- [What is Grad-CAM?](#what-is-grad-cam)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Supported Models](#supported-models)
- [Notes and Tips](#notes-and-tips)
- [References](#references)

---

## What is Grad-CAM?

Grad-CAM produces a coarse **localization map** that highlights the regions of an input image that a CNN used to predict a particular class. It works by:

1. Passing the image through the CNN up to the target class score.
2. Backpropagating the gradient of that score to the **last convolutional layer**.
3. Globally averaging the spatial gradients to obtain a per-feature-map importance weight $\alpha_k^c$.
4. Taking a weighted sum of the forward activation maps and applying a ReLU.

The math is:

$$
\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}
$$

$$
L^c_{\text{Grad-CAM}} = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)
$$

where:

- $y^c$ is the raw score for class $c$ (before softmax),
- $A^k$ is the $k$-th feature map in the target convolutional layer,
- $Z$ is the number of spatial locations in the feature map.

**Guided Grad-CAM** fuses the coarse Grad-CAM heatmap with the high-resolution Guided Backpropagation saliency map to get visualizations that are both class-discriminative and fine-grained.

---

## Installation

From the `gradcam/` directory:

```bash
pip install -r requirements.txt
```

For CPU-only PyTorch, use the official command:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> **Note:** The `gradcam` module has no dependency on the original Lua/Torch repo. It is a standalone PyTorch implementation.

---

## Quick Start

```python
import torch
import torchvision.models as models
from gradcam import GradCAM, visualize_gradcam, preprocess_image

# Load any pretrained CNN
model = models.resnet50(pretrained=True)
model.eval()

# Choose the last convolutional block as the target layer
target_layer = model.layer4[-1]

# Create Grad-CAM object
gradcam = GradCAM(model, target_layer)

# Load and preprocess an image
input_tensor = preprocess_image("path/to/image.jpg", input_size=224)

# Compute heatmap for a specific ImageNet class (e.g. 243 = boxer dog)
heatmap = gradcam(input_tensor, target_category=243)

# Overlay heatmap on the original image
overlay = visualize_gradcam(input_tensor, heatmap, alpha=0.5)
```

To save the overlay:

```python
import cv2

cv2.imwrite("gradcam_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
```

---

## API Reference

### `GradCAM`

```python
from gradcam import GradCAM

gradcam = GradCAM(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    use_cuda: bool | None = None,
)
```

| Parameter | Description |
|---|---|
| `model` | A PyTorch CNN that outputs class logits (before softmax). |
| `target_layer` | The convolutional layer to explain. Usually the last convolutional layer. |
| `use_cuda` | Whether to run on GPU. Auto-detected if not provided. |

#### Methods

- **`forward(input_tensor, target_category=None) → torch.Tensor`**  
  Compute the Grad-CAM heatmap. `input_tensor` must have shape `(1, C, H, W)`. If `target_category` is `None`, the predicted class is used.

- **`__call__(input_tensor, target_category=None) → torch.Tensor`**  
  Convenience alias for `forward`.

- **`remove_hooks()`**  
  Remove the registered forward/backward hooks. Call this when you are done to avoid side effects.

#### Properties

- **`last_target_category`**  
  The class index used in the most recent forward call.

---

### `GuidedBackprop`

```python
from gradcam import GuidedBackprop

guided_bp = GuidedBackprop(model, use_cuda=True)
saliency = guided_bp(input_tensor, target_category=243)
```

Computes the Guided Backpropagation saliency map by modifying the ReLU backward pass so that only positive gradients flowing through positive activations are retained.

---



---

## Examples

### 1. Explain the predicted class

```python
import torchvision.models as models
from gradcam import GradCAM, visualize_gradcam, preprocess_image

model = models.resnet50(pretrained=True)
gradcam = GradCAM(model, target_layer=model.layer4[-1])

input_tensor = preprocess_image("cat_dog.jpg")
heatmap = gradcam(input_tensor)  # uses predicted class
overlay = visualize_gradcam(input_tensor, heatmap)
```

### 2. Explain a specific class

```python
# ImageNet class indices: 243 = boxer, 282 = tabby cat, 283 = tiger cat
heatmap = gradcam(input_tensor, target_category=243)
overlay = visualize_gradcam(input_tensor, heatmap)
```

### 3. Guided Grad-CAM

```python
import numpy as np
import cv2
import torch
from gradcam import GradCAM, GuidedBackprop, preprocess_image

model = models.resnet50(pretrained=True)
input_tensor = preprocess_image("cat_dog.jpg")

target_layer = model.layer4[-1]
gradcam = GradCAM(model, target_layer)
guided_bp = GuidedBackprop(model)

cam = gradcam(input_tensor, target_category=243)
gb = guided_bp(input_tensor, target_category=243)

# Resize Grad-CAM to input size and fuse
cam_resized = torch.nn.functional.interpolate(
    cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
)
guided_gradcam = gb * cam_resized

# Convert to grayscale image for display
guided_gradcam_np = guided_gradcam.squeeze().detach().cpu().numpy()
guided_gradcam_gray = np.max(np.abs(guided_gradcam_np), axis=0)
guided_gradcam_gray = guided_gradcam_gray / (guided_gradcam_gray.max() + 1e-8)
cv2.imwrite("guided_gradcam.png", np.uint8(255 * guided_gradcam_gray))
```
---

## Supported Models

The module is model-agnostic: any PyTorch CNN with convolutional layers works. Common choices for `target_layer`:

| Model | Last Conv Layer | Example |
|---|---|---|
| ResNet-18/34/50/101/152 | `model.layer4[-1]` | `GradCAM(model, model.layer4[-1])` |
| VGG-16 | `model.features[28]` | `GradCAM(model, model.features[28])` |
| VGG-19 | `model.features[34]` | `GradCAM(model, model.features[34])` |
| DenseNet-121 | `model.features.denseblock4` | `GradCAM(model, model.features.denseblock4)` |
| MobileNet-V2 | `model.features[-1]` | `GradCAM(model, model.features[-1])` |

> **Tip:** Choose the deepest convolutional layer for the most semantic, class-discriminative localization. Earlier layers have smaller receptive fields and focus on low-level textures.

---

## Notes and Tips

- **Batch size:** `GradCAM.forward` and `GuidedBackprop.forward` currently expect a batch of exactly one image: shape `(1, C, H, W)`.
- **Softmax:** Grad-CAM uses the raw class score $y^c$ *before* softmax. If your model ends with a `nn.Softmax()` or `nn.LogSoftmax()`, remove it or use the logits directly.
- **Layer choice:** The target layer must be a convolutional layer (or a block whose output is a set of feature maps). Fully-connected layers do not have spatial structure and cannot be used directly.
- **Cleanup:** Call `gradcam.remove_hooks()` and `guided_bp.remove_hooks()` when you are done to avoid keeping hooks attached to the model.
- **CUDA:** Grad-CAM works on both CPU and GPU. GPU is recommended for large models.

---

## References

1. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017. [arXiv](https://arxiv.org/abs/1610.02391)
2. Original Lua/Torch implementation: [ramprs/grad-cam](https://github.com/ramprs/grad-cam/)
3. Zhou et al., "Learning Deep Features for Discriminative Localization", CVPR 2016. (CAM)
4. Springenberg et al., "Striving for Simplicity: The All Convolutional Net", 2014. (Guided Backpropagation)
