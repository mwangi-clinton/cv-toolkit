"""Utility functions for dataset inspection, batch visualization, and interactive notebook exploration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

try:
    from .data_loader import COCO_MEAN, COCO_STD, denormalize_tensor, find_images
except ImportError:
    from data_loader import COCO_MEAN, COCO_STD, denormalize_tensor, find_images


def inspect_folder(
    roots: Union[str, Path, Sequence[Union[str, Path]]],
    sample_size: int = 5,
    recursive: bool = True,
) -> Dict[str, Any]:
    """Inspect one or multiple image directories and display statistics.

    Args:
        roots: Single folder path or list of folder paths.
        sample_size: Number of sample images to inspect for dimensions.
        recursive: Whether to scan recursively.

    Returns:
        Dictionary containing counts, formats, classes, and resolutions.
    """
    items, classes, class_to_idx, folder_counts = find_images(roots, recursive=recursive)

    ext_counts: Dict[str, int] = {}
    for item in items:
        ext = item.path.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    sample_items = items[:sample_size]
    resolutions: List[Tuple[int, int]] = []
    for item in sample_items:
        try:
            with Image.open(item.path) as img:
                resolutions.append(img.size)
        except Exception:
            pass

    info: Dict[str, Any] = {
        "roots": [str(p) for p in folder_counts.keys()],
        "total_images": len(items),
        "folder_counts": folder_counts,
        "formats": ext_counts,
        "num_classes": len(classes),
        "classes": classes[:10] + (["..."] if len(classes) > 10 else []),
        "sample_resolutions_wh": resolutions,
        "first_image": str(items[0].path) if items else None,
    }

    header = "=" * 60
    print(f"{header}\n  Image Dataset Inspection ({len(folder_counts)} folders)\n{header}")
    print(f"  Total Images    : {info['total_images']}")
    print(f"  Per Folder      : {info['folder_counts']}")
    print(f"  Formats Found   : {info['formats']}")
    print(f"  Classes Found   : {info['num_classes']} ({classes[:5]})")
    if resolutions:
        print(f"  Sample Sizes(WxH): {resolutions}")
    print(f"  First Image     : {info['first_image']}")
    print(header)

    return info


def visualize_batch(
    batch: Any,
    num_images: int = 8,
    cols: int = 4,
    denormalize: bool = True,
    mean: Tuple[float, float, float] = COCO_MEAN,
    std: Tuple[float, float, float] = COCO_STD,
    figsize: Optional[Tuple[int, int]] = None,
    title: Optional[str] = "Mini-Batch Visualization",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Visualize images from a DataLoader mini-batch in a matplotlib grid.

    Handles:
        - Tensors of shape (B, C, H, W)
        - Lists of PIL Images or numpy arrays
        - Metadata dictionaries and class labels
    """
    labels: Optional[List[Any]] = None
    if isinstance(batch, dict):
        images = batch.get("image", batch.get("images", batch.get("img")))
        metadata = batch.get("metadata", batch.get("target"))
        if isinstance(metadata, dict):
            labels = metadata.get("filename", metadata.get("label"))
        elif isinstance(metadata, (list, torch.Tensor)):
            labels = list(metadata)
    elif isinstance(batch, (tuple, list)):
        images = batch[0]
        if len(batch) > 1:
            second = batch[1]
            if isinstance(second, dict):
                labels = second.get("filename", second.get("label", second.get("class_name")))
            elif isinstance(second, (list, torch.Tensor)):
                labels = [str(x.item()) if isinstance(x, torch.Tensor) else str(x) for x in second]
    else:
        images = batch

    imgs_to_show: List[np.ndarray] = []

    if isinstance(images, torch.Tensor):
        t = images.detach().cpu()
        if denormalize and t.is_floating_point():
            t = denormalize_tensor(t, mean=mean, std=std)
        t = torch.clamp(t, 0.0, 1.0)
        np_imgs = t.permute(0, 2, 3, 1).numpy()
        for i in range(min(num_images, len(np_imgs))):
            imgs_to_show.append((np_imgs[i] * 255.0).astype(np.uint8))
    elif isinstance(images, list):
        for i in range(min(num_images, len(images))):
            item = images[i]
            if isinstance(item, Image.Image):
                imgs_to_show.append(np.array(item))
            elif isinstance(item, np.ndarray):
                imgs_to_show.append(item)
            elif isinstance(item, torch.Tensor):
                t = item.detach().cpu()
                if denormalize and t.is_floating_point():
                    t = denormalize_tensor(t, mean=mean, std=std)
                t = torch.clamp(t, 0.0, 1.0)
                if t.ndim == 3:
                    imgs_to_show.append((t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8))
    else:
        raise TypeError(f"Unsupported image batch container: {type(images)}")

    n = len(imgs_to_show)
    if n == 0:
        raise ValueError("No images found to visualize.")

    rows = (n + cols - 1) // cols
    calc_figsize = figsize or (cols * 3.5, rows * 3.5)

    fig, axes = plt.subplots(rows, cols, figsize=calc_figsize)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes_flat = np.array(axes).reshape(-1)

    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        if i < n:
            img = imgs_to_show[i]
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                ax.imshow(img.squeeze(), cmap="gray")
            else:
                ax.imshow(img)

            sub_title = str(labels[i])[:20] if labels and i < len(labels) else f"Idx {i} ({img.shape[1]}x{img.shape[0]})"
            ax.set_title(sub_title, fontsize=10)
            ax.axis("off")
        else:
            ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14, y=0.98)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def compare_pipelines(
    image_path: Union[str, Path],
    pipelines: Dict[str, Any],
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """Compare multiple transformation pipelines side-by-side on a single image."""
    img_path = Path(image_path).expanduser().resolve()
    with Image.open(img_path) as raw_img:
        base_img = raw_img.convert("RGB")

    n = len(pipelines)
    calc_figsize = figsize or (n * 3.8, 4.0)

    fig, axes = plt.subplots(1, n, figsize=calc_figsize)
    if n == 1:
        axes = [axes]

    for ax, (name, transform) in zip(axes, pipelines.items()):
        if transform is None:
            disp_img = np.array(base_img)
        else:
            out = transform(base_img)
            if isinstance(out, torch.Tensor):
                out = denormalize_tensor(out)
                disp_img = (out.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            elif isinstance(out, Image.Image):
                disp_img = np.array(out)
            elif isinstance(out, np.ndarray):
                disp_img = out
            else:
                disp_img = np.array(base_img)

        h, w = disp_img.shape[:2]
        ax.imshow(disp_img)
        ax.set_title(f"{name}\n({w}x{h})", fontsize=11)
        ax.axis("off")

    fig.suptitle(f"Pipeline Comparison: {img_path.name}", fontsize=13)
    plt.tight_layout()
    return fig


def launch_fiftyone_app(dataset_or_view: Any, port: int = 5151, **kwargs: Any) -> Any:
    """Launch the FiftyOne visualization App."""
    import fiftyone as fo

    return fo.launch_app(dataset_or_view, port=port, **kwargs)
