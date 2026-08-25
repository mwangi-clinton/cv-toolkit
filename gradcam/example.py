"""
Example script: Grad-CAM and Guided Grad-CAM with a torchvision model.

This script shows how to:
  1. Load a pretrained ResNet-50 from torchvision.
  2. Preprocess an input image.
  3. Compute a Grad-CAM heatmap for a chosen ImageNet class.
  4. Compute Guided Backpropagation and fuse it with Grad-CAM.
  5. Save the resulting visualizations.

Run it with:

    python example.py --image path/to/image.jpg --output output/ --target 243

If ``--target`` is omitted, the predicted class is visualized.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.models as models

from gradcam import GradCAM, GuidedBackprop, preprocess_image, tensor_to_image, visualize_gradcam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM example")
    parser.add_argument(
        "--image",
        type=str,
        default="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        help="Path or URL to input image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Directory to save output images.",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="ImageNet class index to explain. If None, use the predicted class.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet50", "vgg16", "vgg19"],
        help="Backbone model to use.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Input image size.",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA even if a GPU is available.",
    )
    return parser.parse_args()


def get_model_and_target_layer(model_name: str):
    """Return a pretrained model and its last convolutional layer."""
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        target_layer = model.layer4[-1]
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        target_layer = model.layer4[-1]
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        target_layer = model.features[28]  # last conv layer in VGG-16 features
    elif model_name == "vgg19":
        model = models.vgg19(pretrained=True)
        target_layer = model.features[34]  # last conv layer in VGG-19 features
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.eval()
    return model, target_layer


def main() -> None:
    args = parse_args()

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load model
    model, target_layer = get_model_and_target_layer(args.model)
    model = model.to(device)

    # Prepare output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess image
    input_tensor = preprocess_image(args.image, input_size=args.input_size).to(device)

    # ------------------------------------------------------------------
    # Grad-CAM
    # ------------------------------------------------------------------
    gradcam = GradCAM(model, target_layer, use_cuda=use_cuda)
    cam = gradcam(input_tensor, target_category=args.target)
    used_class = gradcam.last_target_category
    print(f"Explaining class index: {used_class}")

    # Save Grad-CAM overlay
    overlay = visualize_gradcam(input_tensor, cam, alpha=0.5)
    cv2.imwrite(
        str(output_dir / f"gradcam_{args.model}_class{used_class}.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )

    # Save raw heatmap
    heatmap_np = cam.squeeze().detach().cpu().numpy()
    heatmap_np = cv2.resize(heatmap_np, (args.input_size, args.input_size))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_np), cv2.COLORMAP_JET)
    cv2.imwrite(str(output_dir / f"gradcam_heatmap_{args.model}_class{used_class}.png"), heatmap_colored)

    # ------------------------------------------------------------------
    # Guided Backpropagation
    # ------------------------------------------------------------------
    guided_bp = GuidedBackprop(model, use_cuda=use_cuda)
    gb = guided_bp(input_tensor, target_category=used_class)

    # Convert guided backprop to grayscale image
    gb_np = gb.squeeze().detach().cpu().numpy()
    gb_gray = np.max(np.abs(gb_np), axis=0)
    gb_gray = gb_gray / (gb_gray.max() + 1e-8)
    gb_uint8 = np.uint8(255 * gb_gray)
    cv2.imwrite(str(output_dir / f"guided_backprop_{args.model}_class{used_class}.png"), gb_uint8)

    # ------------------------------------------------------------------
    # Guided Grad-CAM = Grad-CAM * Guided Backprop
    # ------------------------------------------------------------------
    cam_resized = torch.from_numpy(heatmap_np).to(device).unsqueeze(0).unsqueeze(0)
    guided_gradcam = gb * cam_resized

    guided_gradcam_np = guided_gradcam.squeeze().detach().cpu().numpy()
    guided_gradcam_gray = np.max(np.abs(guided_gradcam_np), axis=0)
    guided_gradcam_gray = guided_gradcam_gray / (guided_gradcam_gray.max() + 1e-8)
    guided_gradcam_uint8 = np.uint8(255 * guided_gradcam_gray)
    cv2.imwrite(
        str(output_dir / f"guided_gradcam_{args.model}_class{used_class}.png"),
        guided_gradcam_uint8,
    )

    print(f"Saved visualizations to {output_dir.resolve()}")

    # Clean up hooks
    gradcam.remove_hooks()
    guided_bp.remove_hooks()


if __name__ == "__main__":
    main()
