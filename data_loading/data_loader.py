"""Unified and minimal image data loading toolkit for PyTorch and Jupyter Notebooks.

Consolidates dataset creation, multi-folder ingestion, preprocessing, hardware
acceleration (CPU, GPU, Multi-Node DDP), batch limiting, and FiftyOne integration
into a single, flexible module.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

from PIL import Image
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
import torchvision.transforms as T

logger = logging.getLogger(__name__)

# Standard constants
SUPPORTED_EXTENSIONS: Set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
    ".ppm",
    ".pgm",
}

COCO_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
COCO_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

@dataclass
class ImageItem:
    """Represents a discovered image sample.

    Attributes:
        path: Absolute path to the image file.
        label: Integer class index, or 0 if unclassified.
        class_name: Name of the class subfolder or folder name.
        folder_source: Name of the origin folder.
        index: Index of item in the combined dataset.
    """

    path: Path
    label: int = 0
    class_name: Optional[str] = "default"
    folder_source: Optional[str] = None
    index: int = 0


def _normalize_roots(root: Union[str, Path, Sequence[Union[str, Path]]]) -> List[Path]:
    """Convert a single path or list of paths into a validated list of Paths."""
    if isinstance(root, (str, Path)):
        roots = [root]
    elif isinstance(root, (list, tuple, set)):
        roots = list(root)
    else:
        raise TypeError(f"root must be a str, Path, or sequence of paths, got {type(root)}")

    resolved: List[Path] = []
    for r in roots:
        p = Path(r).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Image directory does not exist: {p}")
        if not p.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {p}")
        resolved.append(p)

    if not resolved:
        raise ValueError("No directories provided in root list.")

    return resolved


def find_images(
    roots: Union[str, Path, Sequence[Union[str, Path]]],
    extensions: Optional[Set[str]] = None,
    recursive: bool = True,
    has_subfolders: Optional[bool] = None,
) -> Tuple[List[ImageItem], List[str], Dict[str, int], Dict[str, int]]:
    """Scan one or multiple directories and index all supported image files.

    Combines images from multiple folders into a unified dataset with shared
    class indices and origin folder tracking.

    Args:
        roots: A single directory path or a list/tuple of directory paths.
        extensions: Set of valid extensions (case-insensitive).
        recursive: Whether to search subdirectories recursively.
        has_subfolders: If True, treat subdirectories as classes.
            If False, treat folders as flat.
            If None (default), auto-detect per directory.

    Returns:
        Tuple containing:
            - items: List of ImageItem objects across all folders.
            - classes: Alphabetically sorted list of class names.
            - class_to_idx: Mapping from class name to integer label.
            - folder_counts: Number of images indexed per source folder.
    """
    root_paths = _normalize_roots(roots)
    exts = {e.lower() for e in (extensions or SUPPORTED_EXTENSIONS)}

    # Phase 1: First pass to discover classes across all folders
    discovered_classes: Set[str] = set()
    folder_subdirs: Dict[Path, List[Path]] = {}
    folder_has_sub: Dict[Path, bool] = {}

    for r in root_paths:
        subdirs = sorted([d for d in r.iterdir() if d.is_dir() and not d.name.startswith(".")])
        folder_subdirs[r] = subdirs

        if has_subfolders is not None:
            folder_has_sub[r] = has_subfolders
        else:
            if subdirs:
                has_imgs = any(
                    any(f.suffix.lower() in exts for f in d.iterdir() if f.is_file())
                    for d in subdirs
                )
                folder_has_sub[r] = has_imgs
            else:
                folder_has_sub[r] = False

        if folder_has_sub[r] and subdirs:
            for d in subdirs:
                discovered_classes.add(d.name)

    if discovered_classes:
        classes = sorted(discovered_classes)
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    else:
        # If no subfolder classes exist, use folder names if multiple folders, else 'default'
        if len(root_paths) > 1:
            classes = sorted([r.name for r in root_paths])
            class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        else:
            classes = ["default"]
            class_to_idx = {"default": 0}

    # Phase 2: Index all images
    items: List[ImageItem] = []
    folder_counts: Dict[str, int] = {r.name: 0 for r in root_paths}
    global_idx = 0

    for r in root_paths:
        r_name = r.name
        is_sub = folder_has_sub[r]
        subdirs = folder_subdirs[r]

        if is_sub and subdirs:
            for cls_dir in subdirs:
                cls_name = cls_dir.name
                pattern = "**/*" if recursive else "*"
                for file_path in sorted(cls_dir.glob(pattern)):
                    if file_path.is_file() and file_path.suffix.lower() in exts:
                        items.append(
                            ImageItem(
                                path=file_path,
                                label=class_to_idx.get(cls_name, 0),
                                class_name=cls_name,
                                folder_source=r_name,
                                index=global_idx,
                            )
                        )
                        global_idx += 1
                        folder_counts[r_name] += 1
        else:
            # Flat folder
            pattern = "**/*" if recursive else "*"
            default_cls = r_name if len(root_paths) > 1 else "default"
            default_lbl = class_to_idx.get(default_cls, 0)

            for file_path in sorted(r.glob(pattern)):
                if file_path.is_file() and file_path.suffix.lower() in exts:
                    items.append(
                        ImageItem(
                            path=file_path,
                            label=default_lbl,
                            class_name=default_cls,
                            folder_source=r_name,
                            index=global_idx,
                        )
                    )
                    global_idx += 1
                    folder_counts[r_name] += 1

    if not items:
        raise ValueError(
            f"No image files with extensions {sorted(exts)} found in {[str(p) for p in root_paths]}."
        )

    return items, classes, class_to_idx, folder_counts


def split_items(
    items: List[ImageItem],
    val_split: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[List[ImageItem], List[ImageItem]]:
    """Deterministically split image items into train and validation sets.

    Args:
        items: List of ImageItem samples to split.
        val_split: Fraction of samples for validation (0.0 < val_split < 1.0).
        seed: Random seed for reproducible splitting.
        stratify: If True, balances classes equally across splits.

    Returns:
        Tuple of (train_items, val_items).
    """
    import random
    from collections import defaultdict

    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be between 0.0 and 1.0, got {val_split}")

    rng = random.Random(seed)

    if stratify:
        by_class: Dict[int, List[ImageItem]] = defaultdict(list)
        for it in items:
            by_class[it.label].append(it)

        train_list: List[ImageItem] = []
        val_list: List[ImageItem] = []

        for _, group in sorted(by_class.items()):
            shuffled = list(group)
            rng.shuffle(shuffled)
            n_val = max(1, int(round(len(shuffled) * val_split))) if len(shuffled) > 1 else 0
            val_list.extend(shuffled[:n_val])
            train_list.extend(shuffled[n_val:])

        for idx, it in enumerate(train_list):
            it.index = idx
        for idx, it in enumerate(val_list):
            it.index = idx

        return train_list, val_list
    else:
        shuffled = list(items)
        rng.shuffle(shuffled)
        n_val = int(round(len(shuffled) * val_split))
        val_list = shuffled[:n_val]
        train_list = shuffled[n_val:]

        for idx, it in enumerate(train_list):
            it.index = idx
        for idx, it in enumerate(val_list):
            it.index = idx

        return train_list, val_list


class FolderImageDataset(Dataset):
    """PyTorch Dataset loading images from one or multiple folders.

    Supports:
        - Single folder path OR list of folder paths (combined dataset).
        - Pre-split items list (for train/val separation).
        - Flat folder or hierarchical class subfolders.
        - Metadata extraction (filepath, dimensions, label, origin folder).
        - Subsetting for rapid notebook prototyping (`max_samples`).
        - Corrupt image handling.

    Args:
        root: Single directory or list of directory paths (if items is None).
        items: Optional pre-discovered or pre-split ImageItem list.
        transform: Callable transformation applied to PIL Image.
        target_transform: Callable transformation applied to target label.
        recursive: Whether to scan folders recursively. Default True.
        has_subfolders: Subfolder mode (True/False/None auto-detect).
        max_samples: If set, limits dataset to first N samples.
        return_metadata: If True, returns `(image, metadata_dict)`.
        return_dict: If True, returns `{'image', 'target', 'metadata'}`.
        ignore_corrupt: If True, tolerates corrupted image files.
        color_mode: PIL conversion mode (default "RGB").
    """

    def __init__(
        self,
        root: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
        items: Optional[List[ImageItem]] = None,
        classes: Optional[List[str]] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        folder_counts: Optional[Dict[str, int]] = None,
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
        recursive: bool = True,
        has_subfolders: Optional[bool] = None,
        max_samples: Optional[int] = None,
        return_metadata: bool = True,
        return_dict: bool = False,
        ignore_corrupt: bool = True,
        color_mode: str = "RGB",
        extensions: Optional[Set[str]] = None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.recursive = recursive
        self.return_metadata = return_metadata
        self.return_dict = return_dict
        self.ignore_corrupt = ignore_corrupt
        self.color_mode = color_mode

        if items is not None:
            self.roots = _normalize_roots(root) if root is not None else []
            self.root = self.roots[0] if len(self.roots) == 1 else self.roots
            self._items = list(items)
            self.classes = classes or sorted(list({it.class_name for it in items if it.class_name}))
            self.class_to_idx = class_to_idx or {cls: i for i, cls in enumerate(self.classes)}
            self.folder_counts = folder_counts or {}
        elif root is not None:
            self.roots = _normalize_roots(root)
            self.root = self.roots[0] if len(self.roots) == 1 else self.roots
            self._items, self.classes, self.class_to_idx, self.folder_counts = find_images(
                roots=self.roots,
                extensions=extensions,
                recursive=recursive,
                has_subfolders=has_subfolders,
            )
        else:
            raise ValueError("Either 'root' or 'items' must be provided to FolderImageDataset.")

        if max_samples is not None and max_samples > 0:
            self._items = self._items[:max_samples]

    @property
    def samples(self) -> List[ImageItem]:
        return self._items

    @property
    def image_paths(self) -> List[Path]:
        return [item.path for item in self._items]

    def __len__(self) -> int:
        return len(self._items)

    def _load_image(self, path: Path) -> Image.Image:
        with Image.open(path) as img:
            image = img.convert(self.color_mode)
            image.load()
            return image

    def __getitem__(self, index: int) -> Any:
        item = self._items[index]

        try:
            image = self._load_image(item.path)
            orig_w, orig_h = image.size
        except Exception as err:
            if self.ignore_corrupt:
                logger.warning(
                    "Error loading image '%s': %s. Falling back to adjacent sample.",
                    item.path,
                    err,
                )
                next_idx = (index + 1) % len(self._items)
                if next_idx == index:
                    image = Image.new(self.color_mode, (224, 224), color=0)
                    orig_w, orig_h = 224, 224
                else:
                    return self.__getitem__(next_idx)
            else:
                raise IOError(f"Failed to load image from {item.path}: {err}") from err

        target = item.label
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            image = self.transform(image)

        metadata: Dict[str, Any] = {
            "path": str(item.path),
            "filename": item.path.name,
            "label": target,
            "class_name": item.class_name,
            "folder_source": item.folder_source,
            "index": index,
            "original_size": (orig_w, orig_h),
        }

        if self.return_dict:
            return {"image": image, "target": target, "metadata": metadata}
        if self.return_metadata:
            return image, metadata
        return image, target

    def summary(self) -> Dict[str, Any]:
        return {
            "roots": [str(p) for p in self.roots],
            "total_images": len(self._items),
            "folder_counts": self.folder_counts,
            "num_classes": len(self.classes),
            "classes": self.classes[:10] + (["..."] if len(self.classes) > 10 else []),
            "color_mode": self.color_mode,
            "has_transforms": self.transform is not None,
        }

    def __repr__(self) -> str:
        roots_str = str(self.root) if len(self.roots) == 1 else f"{len(self.roots)} folders"
        return (
            f"FolderImageDataset(roots={roots_str}, "
            f"samples={len(self._items)}, "
            f"classes={len(self.classes)})"
        )


def smart_collate_fn(batch: Sequence[Any]) -> Any:
    """Intelligently collate dataset samples into a mini-batch.

    Handles uniform tensors (stacked), PIL Images (kept as list),
    metadata dictionaries (aggregated), and variable shape tensors.
    """
    if not batch:
        return batch

    first = batch[0]

    if isinstance(first, dict):
        return {k: smart_collate_fn([item[k] for item in batch]) for k in first}

    if isinstance(first, (tuple, list)):
        return tuple(smart_collate_fn([item[i] for item in batch]) for i in range(len(first)))

    if isinstance(first, Image.Image):
        return list(batch)

    if isinstance(first, torch.Tensor):
        shapes = [t.shape for t in batch]
        if all(s == shapes[0] for s in shapes):
            return torch.stack(batch, dim=0)
        return list(batch)

    if isinstance(first, (int, float)):
        return torch.tensor(batch)

    if isinstance(first, str):
        return list(batch)

    try:
        from torch.utils.data.dataloader import default_collate

        return default_collate(batch)
    except Exception:
        return list(batch)

class IdentityTransform:
    """Identity transform that returns the input unchanged."""

    def __call__(self, img: Any) -> Any:
        return img

    def __repr__(self) -> str:
        return "IdentityTransform()"


class MMEnginePipelineAdapter:
    """Adapter to execute MMEngine / MMCV transform pipelines."""

    def __init__(
        self,
        pipeline: Union[List[Dict[str, Any]], Any],
        as_tensor: bool = True,
    ) -> None:
        self.as_tensor = as_tensor
        try:
            from mmcv.transforms import Compose as MMCVCompose

            self._pipeline = MMCVCompose(pipeline) if isinstance(pipeline, list) else pipeline
        except ImportError:
            try:
                from mmengine.dataset import Compose as MMEngineCompose

                self._pipeline = MMEngineCompose(pipeline) if isinstance(pipeline, list) else pipeline
            except ImportError:
                self._pipeline = self._create_fallback_pipeline(pipeline)

    def _create_fallback_pipeline(self, pipeline: Any) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        def fallback_fn(data: Dict[str, Any]) -> Dict[str, Any]:
            img = data.get("img")
            if img is not None and isinstance(pipeline, list):
                for step in pipeline:
                    stype = step.get("type", "")
                    if stype == "Resize":
                        scale = step.get("scale", (224, 224))
                        pil_img = Image.fromarray(img)
                        resized = pil_img.resize((scale[0], scale[1]), Image.Resampling.BILINEAR)
                        img = np.array(resized)
                    elif stype == "RandomFlip":
                        if np.random.rand() < step.get("prob", 0.5):
                            img = np.fliplr(img).copy()
            data["img"] = img
            return data

        return fallback_fn

    def __call__(self, img_input: Any) -> Any:
        if isinstance(img_input, dict):
            data = img_input
        elif isinstance(img_input, Image.Image):
            data = {"img": np.array(img_input)}
        elif isinstance(img_input, np.ndarray):
            data = {"img": img_input}
        else:
            raise TypeError(f"Unsupported input type for MMEngine pipeline: {type(img_input)}")

        res = self._pipeline(data)
        if not self.as_tensor:
            return res

        out_img = res.get("img") if isinstance(res, dict) else res
        if out_img is None:
            if isinstance(res, dict) and "inputs" in res:
                return res["inputs"]
            raise ValueError(f"Pipeline output does not contain 'img' key: {res}")

        if isinstance(out_img, torch.Tensor):
            return out_img

        if isinstance(out_img, np.ndarray):
            out_img = np.ascontiguousarray(out_img)
            if out_img.ndim == 2:
                out_img = np.expand_dims(out_img, axis=-1)
            transposed = np.ascontiguousarray(out_img.transpose(2, 0, 1))
            tensor = torch.from_numpy(transposed)
            if tensor.dtype == torch.uint8:
                tensor = tensor.float() / 255.0
            return tensor

        return out_img


def get_coco_transforms(
    split: Optional[str] = "default",
    img_size: Union[int, Tuple[int, int]] = (224, 224),
    mean: Tuple[float, float, float] = COCO_MEAN,
    std: Tuple[float, float, float] = COCO_STD,
    auto_augment: bool = False,
) -> T.Compose:
    """Build standard Torchvision transforms for COCO / ImageNet models."""
    target_size = (img_size, img_size) if isinstance(img_size, int) else img_size
    transforms_list: List[Any] = []

    split_str = str(split).lower() if split else "default"

    if split_str == "train":
        transforms_list.extend(
            [
                T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(p=0.5),
            ]
        )
        if auto_augment:
            transforms_list.append(T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET))
        else:
            transforms_list.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))
    else:
        # Default / Val / Test / None: deterministic clean resize without random jitter/flips
        transforms_list.append(T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR))

    transforms_list.extend([T.ToTensor(), T.Normalize(mean=list(mean), std=list(std))])
    return T.Compose(transforms_list)


def get_mmengine_transforms(
    split: Optional[str] = "default",
    img_size: Union[int, Tuple[int, int]] = (224, 224),
    custom_pipeline: Optional[List[Dict[str, Any]]] = None,
    as_tensor: bool = True,
) -> MMEnginePipelineAdapter:
    """Build an MMEngine / MMCV transform pipeline adapter."""
    if custom_pipeline is not None:
        return MMEnginePipelineAdapter(pipeline=custom_pipeline, as_tensor=as_tensor)

    scale = (img_size, img_size) if isinstance(img_size, int) else (img_size[1], img_size[0])
    pipeline: List[Dict[str, Any]] = []

    split_str = str(split).lower() if split else "default"

    if split_str == "train":
        pipeline.extend(
            [
                {"type": "Resize", "scale": scale, "keep_ratio": False},
                {"type": "RandomFlip", "prob": 0.5, "direction": "horizontal"},
            ]
        )
    else:
        pipeline.append({"type": "Resize", "scale": scale, "keep_ratio": False})

    return MMEnginePipelineAdapter(pipeline=pipeline, as_tensor=as_tensor)


def build_transforms(
    preprocessing: Union[str, Callable[[Any], Any], None] = "coco",
    split: Optional[str] = "default",
    img_size: Union[int, Tuple[int, int]] = (224, 224),
    mean: Tuple[float, float, float] = COCO_MEAN,
    std: Tuple[float, float, float] = COCO_STD,
    custom_pipeline: Optional[Any] = None,
    auto_augment: bool = False,
    as_tensor: bool = True,
) -> Optional[Callable[[Any], Any]]:
    """Factory to create image preprocessing pipelines."""
    if preprocessing is None:
        return None

    if callable(preprocessing) and not isinstance(preprocessing, str):
        return preprocessing

    mode = str(preprocessing).lower().strip()

    if mode in ("none", "raw", "identity"):
        return IdentityTransform()

    if mode in ("coco", "standard", "torchvision", "default"):
        return get_coco_transforms(
            split=split,
            img_size=img_size,
            mean=mean,
            std=std,
            auto_augment=auto_augment,
        )

    if mode in ("mmengine", "mmcv"):
        return get_mmengine_transforms(
            split=split,
            img_size=img_size,
            custom_pipeline=custom_pipeline,
            as_tensor=as_tensor,
        )

    if mode == "custom":
        if custom_pipeline is None:
            raise ValueError("preprocessing='custom' specified, but 'custom_pipeline' was not provided.")
        if callable(custom_pipeline):
            return custom_pipeline
        if isinstance(custom_pipeline, list):
            return MMEnginePipelineAdapter(pipeline=custom_pipeline, as_tensor=as_tensor)
        raise TypeError(f"Unsupported custom_pipeline type: {type(custom_pipeline)}")

    raise ValueError(f"Unknown preprocessing mode: '{preprocessing}'.")


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = COCO_MEAN,
    std: Tuple[float, float, float] = COCO_STD,
) -> torch.Tensor:
    """Reverse ImageNet/COCO normalization for visualization."""
    if not isinstance(tensor, torch.Tensor):
        return tensor

    t = tensor.clone().detach().cpu()
    mean_t = torch.tensor(mean, dtype=t.dtype, device=t.device)
    std_t = torch.tensor(std, dtype=t.dtype, device=t.device)

    if t.ndim == 3:
        mean_t = mean_t.view(-1, 1, 1)
        std_t = std_t.view(-1, 1, 1)
    elif t.ndim == 4:
        mean_t = mean_t.view(1, -1, 1, 1)
        std_t = std_t.view(1, -1, 1, 1)
    else:
        return t

    denorm = (t * std_t) + mean_t
    return torch.clamp(denorm, 0.0, 1.0)

def _move_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively transfer all tensors in a batch structure to device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, (list, tuple)):
        moved = [_move_to_device(x, device) for x in batch]
        return type(batch)(moved) if isinstance(batch, tuple) else moved
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    return batch


class BatchLimitedDataLoader:
    """A notebook-friendly wrapper around PyTorch DataLoader.

    Adds:
        - `.one_batch()`: returns exactly 1 batch immediately.
        - `.take(n)`: returns loader restricted to first n batches.
        - `.all()`: returns full unlimited loader.
        - `.summary()` / `.inspect()`: displays dataset & batch shapes.
        - `.set_epoch(epoch)`: updates DistributedSampler epoch for DDP.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
        target_device: Optional[torch.device] = None,
        sampler: Optional[Sampler] = None,
    ) -> None:
        self._dataloader = dataloader
        self._max_batches = max_batches
        self._target_device = target_device
        self._sampler = sampler or getattr(dataloader, "sampler", None)

    @property
    def raw_loader(self) -> DataLoader:
        return self._dataloader

    @property
    def dataloader(self) -> DataLoader:
        return self._dataloader

    @property
    def max_batches(self) -> Optional[int]:
        return self._max_batches

    @property
    def target_device(self) -> Optional[torch.device]:
        return self._target_device

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataloader, name)

    def __len__(self) -> int:
        total = len(self._dataloader)
        if self._max_batches is not None and self._max_batches >= 0:
            return min(self._max_batches, total)
        return total

    def __iter__(self) -> Iterator[Any]:
        count = 0
        for batch in self._dataloader:
            if self._max_batches is not None and count >= self._max_batches:
                break
            if self._target_device is not None:
                batch = _move_to_device(batch, self._target_device)
            yield batch
            count += 1

    def one_batch(self, to_device: bool = True) -> Any:
        """Fetch and return a single batch immediately."""
        for batch in self._dataloader:
            if to_device and self._target_device is not None:
                batch = _move_to_device(batch, self._target_device)
            return batch
        raise StopIteration("DataLoader is empty.")

    def take(self, n_batches: int) -> BatchLimitedDataLoader:
        """Return a loader restricted to the first `n_batches`."""
        return BatchLimitedDataLoader(
            dataloader=self._dataloader,
            max_batches=n_batches,
            target_device=self._target_device,
            sampler=self._sampler,
        )

    def all(self) -> BatchLimitedDataLoader:
        """Return an unlimited loader yielding all available batches."""
        return BatchLimitedDataLoader(
            dataloader=self._dataloader,
            max_batches=None,
            target_device=self._target_device,
            sampler=self._sampler,
        )

    def set_epoch(self, epoch: int) -> None:
        """Update epoch on DistributedSampler for DDP."""
        if self._sampler is not None and hasattr(self._sampler, "set_epoch"):
            self._sampler.set_epoch(epoch)

    def inspect(self) -> Dict[str, Any]:
        dataset = getattr(self._dataloader, "dataset", None)
        total_samples = len(dataset) if dataset is not None and hasattr(dataset, "__len__") else "Unknown"

        info: Dict[str, Any] = {
            "total_samples": total_samples,
            "batch_size": self._dataloader.batch_size,
            "total_batches_in_data": len(self._dataloader),
            "max_batches_configured": self._max_batches,
            "effective_batches": len(self),
            "num_workers": self._dataloader.num_workers,
            "pin_memory": self._dataloader.pin_memory,
            "drop_last": self._dataloader.drop_last,
            "target_device": str(self._target_device) if self._target_device else "cpu",
            "is_distributed": isinstance(self._sampler, DistributedSampler),
        }

        try:
            sample_batch = self.one_batch(to_device=False)
            if isinstance(sample_batch, (tuple, list)) and len(sample_batch) > 0:
                first = sample_batch[0]
                if isinstance(first, torch.Tensor):
                    info["sample_batch_shape"] = list(first.shape)
                    info["sample_tensor_dtype"] = str(first.dtype)
                    info["sample_tensor_min"] = float(first.min().item())
                    info["sample_tensor_max"] = float(first.max().item())
                elif isinstance(first, list) and len(first) > 0:
                    info["sample_type"] = type(first[0]).__name__
            elif isinstance(sample_batch, dict):
                info["sample_batch_keys"] = list(sample_batch.keys())
        except Exception as e:
            info["peek_error"] = str(e)

        return info

    def summary(self) -> None:
        info = self.inspect()
        header = "=" * 50
        print(f"{header}\n  DataLoader Summary\n{header}")
        for k, v in info.items():
            print(f"  {k:24s}: {v}")
        print(header)

    def __repr__(self) -> str:
        return (
            f"BatchLimitedDataLoader(batch_size={self._dataloader.batch_size}, "
            f"batches={len(self)}/{len(self._dataloader)}, "
            f"device='{self._target_device or 'cpu'}', "
            f"distributed={isinstance(self._sampler, DistributedSampler)})"
        )


def get_distributed_info() -> Dict[str, Any]:
    """Smart checking for GPU counts, multi-GPU, and multi-node cluster settings.

    Automatically inspects:
        - Initialized torch.distributed status
        - Physical CUDA GPU devices count (torch.cuda.device_count())
        - PyTorch / torchrun environment variables (WORLD_SIZE, RANK, LOCAL_RANK, LOCAL_WORLD_SIZE, MASTER_ADDR, MASTER_PORT)
        - Slurm cluster environment variables (SLURM_PROCID, SLURM_NTASKS, SLURM_LOCALID, SLURM_NNODES, SLURM_NODEID)
        - OpenMPI / PMI environment variables (OMPI_COMM_WORLD_SIZE, OMPI_COMM_WORLD_RANK)

    Returns:
        Dictionary detailing distributed and multi-node cluster configuration.
    """
    is_dist_initialized = dist.is_available() and dist.is_initialized()
    cuda_available = torch.cuda.is_available()
    cuda_devices = torch.cuda.device_count() if cuda_available else 0

    # Auto-detect world size (total processes across all nodes/GPUs)
    world_size = 1
    if is_dist_initialized:
        world_size = dist.get_world_size()
    elif "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    elif "SLURM_NTASKS" in os.environ:
        world_size = int(os.environ["SLURM_NTASKS"])
    elif "OMPI_COMM_WORLD_SIZE" in os.environ:
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    elif cuda_devices > 1:
        world_size = cuda_devices

    # Auto-detect global rank (0 to world_size - 1)
    rank = 0
    if is_dist_initialized:
        rank = dist.get_rank()
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
    elif "OMPI_COMM_WORLD_RANK" in os.environ:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])

    # Auto-detect local rank (GPU device index on current node)
    local_rank = 0
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_LOCALID" in os.environ:
        local_rank = int(os.environ["SLURM_LOCALID"])
    elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    elif cuda_devices > 0:
        local_rank = rank % cuda_devices

    # Multi-node cluster coordination settings
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")
    node_rank = int(os.environ.get("NODE_RANK", os.environ.get("GROUP_RANK", os.environ.get("SLURM_NODEID", "0"))))
    num_nodes = int(os.environ.get("NNODES", os.environ.get("SLURM_NNODES", "1")))

    is_multinode = num_nodes > 1 or (world_size > cuda_devices and cuda_devices > 0)
    is_distributed = (
        is_dist_initialized
        or "WORLD_SIZE" in os.environ
        or "SLURM_NTASKS" in os.environ
        or "OMPI_COMM_WORLD_SIZE" in os.environ
    )

    return {
        "is_distributed": is_distributed,
        "is_dist_initialized": is_dist_initialized,
        "is_multinode": is_multinode,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "num_nodes": num_nodes,
        "node_rank": node_rank,
        "master_addr": master_addr,
        "master_port": master_port,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_devices,
    }


def setup_distributed(
    backend: Optional[str] = None,
    init_method: str = "env://",
) -> Dict[str, Any]:
    """Helper to initialize PyTorch distributed process group using detected settings.

    Automatically uses 'nccl' if CUDA is available, or 'gloo' on CPU.

    Returns:
        Dictionary of distributed information from get_distributed_info().
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available on this platform.")

    dist_info = get_distributed_info()
    if dist.is_initialized():
        return dist_info

    if backend is None:
        backend = "nccl" if dist_info["cuda_available"] else "gloo"

    if dist_info["cuda_available"] and torch.cuda.is_available():
        torch.cuda.set_device(dist_info["local_rank"])

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=dist_info["world_size"],
        rank=dist_info["rank"],
    )
    return get_distributed_info()


def _resolve_device(
    device: Union[str, torch.device]
) -> Tuple[str, Optional[torch.device], bool, Dict[str, Any]]:
    dist_info = get_distributed_info()

    if isinstance(device, torch.device):
        is_dist = "dist" in str(device).lower() or "node" in str(device).lower()
        return ("gpu" if device.type == "cuda" else "cpu", device, is_dist, dist_info)

    dev_name = str(device).lower().strip()

    # Smart auto-detection mode
    if dev_name in ("auto", "default"):
        if dist_info["is_distributed"]:
            target = torch.device(f"cuda:{dist_info['local_rank']}") if dist_info["cuda_available"] else torch.device("cpu")
            return ("distributed", target, True, dist_info)
        if dist_info["cuda_available"]:
            target = torch.device(f"cuda:{dist_info['local_rank']}")
            return ("gpu", target, False, dist_info)
        return ("cpu", torch.device("cpu"), False, dist_info)

    if dev_name in ("multi-gpu", "multigpu", "distributed", "multinode", "nodes", "ddp"):
        target = torch.device(f"cuda:{dist_info['local_rank']}") if dist_info["cuda_available"] else torch.device("cpu")
        return ("distributed", target, True, dist_info)

    if dev_name in ("gpu", "cuda"):
        if dist_info["is_distributed"]:
            # Smart checking: if user passed device='gpu' but environment is running in DDP, auto-engage distributed
            target = torch.device(f"cuda:{dist_info['local_rank']}") if dist_info["cuda_available"] else torch.device("cpu")
            return ("distributed", target, True, dist_info)
        target = torch.device("cuda:0" if dist_info["cuda_available"] else "cpu")
        return ("gpu", target, False, dist_info)

    if dev_name.startswith("cuda:"):
        target = torch.device(dev_name if dist_info["cuda_available"] else "cpu")
        return ("gpu", target, False, dist_info)

    return ("cpu", torch.device("cpu"), False, dist_info)


def _check_fiftyone_available() -> Any:
    try:
        import fiftyone as fo

        return fo
    except ImportError as err:
        raise ImportError(
            "FiftyOne is required for backend='fiftyone'. Install with: `pip install fiftyone`."
        ) from err


def folder_to_fiftyone_dataset(
    folder_or_folders: Union[str, Path, Sequence[Union[str, Path]]],
    name: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    overwrite: bool = False,
    persistent: bool = False,
) -> Any:
    """Ingest one or multiple folders into a FiftyOne Dataset."""
    fo = _check_fiftyone_available()
    roots = _normalize_roots(folder_or_folders)

    ds_name = name or f"dataset_{roots[0].name.replace('-', '_').replace(' ', '_')}"

    if fo.dataset_exists(ds_name):
        if overwrite:
            fo.delete_dataset(ds_name)
        else:
            return fo.load_dataset(ds_name)

    # Ingest first folder
    dataset = fo.Dataset.from_images_dir(str(roots[0]), name=ds_name)
    if len(roots) > 1:
        dataset.tag_samples(f"folder:{roots[0].name}")
        for extra_root in roots[1:]:
            dataset.add_images_dir(str(extra_root), tags=[f"folder:{extra_root.name}"])

    if tags:
        dataset.tag_samples(list(tags))

    dataset.persistent = persistent
    return dataset


class FiftyOneImageDataset(Dataset):
    """PyTorch Dataset that reads from a FiftyOne Dataset or View."""

    def __init__(
        self,
        dataset_or_folders: Any,
        sample_ids: Optional[Sequence[str]] = None,
        transform: Optional[Callable[[Any], Any]] = None,
        label_field: Optional[str] = None,
        max_samples: Optional[int] = None,
        return_metadata: bool = True,
        return_dict: bool = False,
        color_mode: str = "RGB",
        ignore_corrupt: bool = True,
    ) -> None:
        super().__init__()
        _check_fiftyone_available()

        if isinstance(dataset_or_folders, (str, Path, list, tuple, set)):
            self.fo_dataset = folder_to_fiftyone_dataset(dataset_or_folders)
        else:
            self.fo_dataset = dataset_or_folders

        self.transform = transform
        self.label_field = label_field
        self.return_metadata = return_metadata
        self.return_dict = return_dict
        self.color_mode = color_mode
        self.ignore_corrupt = ignore_corrupt

        if sample_ids is not None:
            view = self.fo_dataset.select(list(sample_ids))
        elif max_samples and max_samples > 0:
            view = self.fo_dataset.limit(max_samples)
        else:
            view = self.fo_dataset
        self._sample_ids = view.values("id")
        self._filepaths = view.values("filepath")

        self._labels: List[Any] = []
        if label_field and label_field in view.get_field_schema():
            try:
                self._labels = view.values(f"{label_field}.label")
            except Exception:
                try:
                    self._labels = view.values(label_field)
                except Exception:
                    self._labels = [0] * len(self._filepaths)
        else:
            self._labels = [0] * len(self._filepaths)

    def __len__(self) -> int:
        return len(self._filepaths)

    def _load_image(self, path: Union[str, Path]) -> Image.Image:
        with Image.open(path) as img:
            image = img.convert(self.color_mode)
            image.load()
            return image

    def __getitem__(self, index: int) -> Any:
        filepath = self._filepaths[index]
        sample_id = self._sample_ids[index]
        label = self._labels[index] if index < len(self._labels) else 0

        try:
            image = self._load_image(filepath)
            orig_w, orig_h = image.size
        except Exception as err:
            if self.ignore_corrupt:
                next_idx = (index + 1) % len(self._filepaths)
                if next_idx == index:
                    image = Image.new(self.color_mode, (224, 224), color=0)
                    orig_w, orig_h = 224, 224
                else:
                    return self.__getitem__(next_idx)
            else:
                raise IOError(f"Could not load image from {filepath}: {err}") from err

        if self.transform is not None:
            image = self.transform(image)

        metadata: Dict[str, Any] = {
            "fiftyone_id": sample_id,
            "path": filepath,
            "filename": Path(filepath).name,
            "label": label,
            "index": index,
            "original_size": (orig_w, orig_h),
        }

        if self.return_dict:
            return {"image": image, "target": label, "metadata": metadata}
        if self.return_metadata:
            return image, metadata
        return image, label


def _build_single_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    device: Union[str, torch.device] = "cpu",
    split: Optional[str] = "default",
    max_batches: Optional[int] = None,
    num_workers: Optional[int] = None,
    shuffle: bool = False,
    pin_memory: Optional[bool] = None,
    drop_last: bool = False,
    distributed_kwargs: Optional[Dict[str, Any]] = None,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    to_device: bool = True,
) -> BatchLimitedDataLoader:
    device_category, target_torch_device, is_distributed, dist_info = _resolve_device(device)

    sampler: Optional[Sampler] = None
    if is_distributed:
        dist_kwargs = distributed_kwargs or {}
        num_replicas = dist_kwargs.get("world_size", dist_info["world_size"])
        rank = dist_kwargs.get("rank", dist_info["rank"])
        seed = dist_kwargs.get("seed", 0)
        dist_drop_last = dist_kwargs.get("drop_last", drop_last)

        sampler = DistributedSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=dist_drop_last,
        )
        shuffle = False

    if num_workers is None:
        num_workers = 0

    if pin_memory is None:
        pin_memory = (device_category in ("gpu", "distributed")) and dist_info["cuda_available"]

    effective_collate = collate_fn if collate_fn is not None else smart_collate_fn

    raw_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=effective_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return BatchLimitedDataLoader(
        dataloader=raw_loader,
        max_batches=max_batches,
        target_device=target_torch_device if to_device else None,
        sampler=sampler,
    )


# Unified Entry Point: create_dataloader
def create_dataloader(
    root: Union[str, Path, Sequence[Union[str, Path]], Dataset, Any],
    batch_size: int = 32,
    images_per_batch: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    split: Optional[str] = "default",
    preprocessing: Union[str, Callable[[Any], Any], None] = "coco",
    backend: str = "torch",
    custom_transforms: Optional[Any] = None,
    img_size: Union[int, Tuple[int, int]] = 224,
    max_batches: Optional[int] = None,
    max_samples: Optional[int] = None,
    num_workers: Optional[int] = None,
    shuffle: Optional[bool] = None,
    pin_memory: Optional[bool] = None,
    drop_last: bool = False,
    recursive: bool = True,
    has_subfolders: Optional[bool] = None,
    return_metadata: bool = True,
    return_dict: bool = False,
    ignore_corrupt: bool = True,
    color_mode: str = "RGB",
    distributed_kwargs: Optional[Dict[str, Any]] = None,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    to_device: bool = True,
    val_split: Optional[float] = None,
    val_root: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
    val_batch_size: Optional[int] = None,
    val_max_batches: Optional[int] = None,
    val_shuffle: bool = False,
    val_preprocessing: Optional[Union[str, Callable[[Any], Any]]] = None,
    split_seed: int = 42,
    stratify: bool = True,
) -> Union[BatchLimitedDataLoader, Tuple[BatchLimitedDataLoader, BatchLimitedDataLoader]]:
    """Create a PyTorch DataLoader from a single folder or list of folders.

    Accepts:
        - `root`: A folder path `'/path/dir'` OR a list of folder paths
          `['/path/dir1', '/path/dir2']` which are seamlessly combined into one dataset.
        - `backend`: `'torch'` (standard) or `'fiftyone'` (uses FiftyOne dataset engine).
        - `device`: `'cpu'`, `'gpu'`, or `'multi-gpu'` / `'distributed'` (DistributedSampler).
        - `preprocessing`: `'coco'`, `'mmengine'`, `'none'`, or `'custom'`.
        - `max_batches`: `None` (all), `1` (one batch), `20` (20 batches), or any integer.
        - `split`: Split mode:
            - `"default"` or `None`: Standard plain DataLoader (deterministic unaugmented images, shuffle=False).
            - `"train"`: Training mode (with random flips, color jitter, shuffle=True).
            - `"val"` or `"test"`: Validation / test mode.
        - `val_split`: If provided (e.g. 0.2), automatically splits dataset into
          `(train_loader, val_loader)` with appropriate train vs val transforms!
        - `val_root`: If provided, loads a separate folder as validation set and returns
          `(train_loader, val_loader)`.
        - If neither `val_split` nor `val_root` is provided, creates and returns JUST ONE
          standard DataLoader (not train or val, unless requested).

    Returns:
        `(train_loader, val_loader)` if val_split or val_root is given, else single `BatchLimitedDataLoader`.
    """
    import random

    effective_batch_size = images_per_batch if images_per_batch is not None else batch_size

    # Case 1: Separate validation directory provided
    if val_root is not None:
        train_l = create_dataloader(
            root=root,
            batch_size=effective_batch_size,
            device=device,
            split="train",
            preprocessing=preprocessing,
            backend=backend,
            custom_transforms=custom_transforms,
            img_size=img_size,
            max_batches=max_batches,
            max_samples=max_samples,
            num_workers=num_workers,
            shuffle=shuffle if shuffle is not None else True,
            pin_memory=pin_memory,
            drop_last=drop_last,
            recursive=recursive,
            has_subfolders=has_subfolders,
            return_metadata=return_metadata,
            return_dict=return_dict,
            ignore_corrupt=ignore_corrupt,
            color_mode=color_mode,
            distributed_kwargs=distributed_kwargs,
            collate_fn=collate_fn,
            to_device=to_device,
        )
        val_l = create_dataloader(
            root=val_root,
            batch_size=val_batch_size or effective_batch_size,
            device=device,
            split="val",
            preprocessing=val_preprocessing or preprocessing,
            backend=backend,
            custom_transforms=custom_transforms,
            img_size=img_size,
            max_batches=val_max_batches,
            max_samples=max_samples,
            num_workers=num_workers,
            shuffle=val_shuffle,
            pin_memory=pin_memory,
            drop_last=False,
            recursive=recursive,
            has_subfolders=has_subfolders,
            return_metadata=return_metadata,
            return_dict=return_dict,
            ignore_corrupt=ignore_corrupt,
            color_mode=color_mode,
            distributed_kwargs=distributed_kwargs,
            collate_fn=collate_fn,
            to_device=to_device,
        )
        return train_l, val_l

    # Case 2: Ratio split from dataset (val_split provided)
    if val_split is not None:
        if not 0.0 < val_split < 1.0:
            raise ValueError(f"val_split must be between 0.0 and 1.0, got {val_split}")

        train_transform = build_transforms(
            preprocessing=preprocessing,
            split="train",
            img_size=img_size,
            custom_pipeline=custom_transforms,
        )
        val_transform = build_transforms(
            preprocessing=val_preprocessing or preprocessing,
            split="val",
            img_size=img_size,
            custom_pipeline=custom_transforms,
        )

        if backend.lower() == "fiftyone":
            _check_fiftyone_available()
            if isinstance(root, (str, Path, list, tuple, set)):
                fo_ds = folder_to_fiftyone_dataset(root)
            else:
                fo_ds = root

            all_ids = list(fo_ds.values("id"))
            rng = random.Random(split_seed)
            shuffled_ids = list(all_ids)
            rng.shuffle(shuffled_ids)
            n_val = max(1, int(round(len(shuffled_ids) * val_split))) if len(shuffled_ids) > 1 else 0
            val_ids = shuffled_ids[:n_val]
            train_ids = shuffled_ids[n_val:]

            train_ds: Dataset = FiftyOneImageDataset(
                dataset_or_folders=fo_ds,
                sample_ids=train_ids,
                transform=train_transform,
                max_samples=max_samples,
                return_metadata=return_metadata,
                return_dict=return_dict,
                ignore_corrupt=ignore_corrupt,
                color_mode=color_mode,
            )
            val_ds: Dataset = FiftyOneImageDataset(
                dataset_or_folders=fo_ds,
                sample_ids=val_ids,
                transform=val_transform,
                max_samples=max_samples,
                return_metadata=return_metadata,
                return_dict=return_dict,
                ignore_corrupt=ignore_corrupt,
                color_mode=color_mode,
            )
        else:
            if isinstance(root, FolderImageDataset):
                all_items = root.samples
                classes = root.classes
                class_to_idx = root.class_to_idx
                folder_counts = root.folder_counts
            else:
                all_items, classes, class_to_idx, folder_counts = find_images(
                    roots=root,
                    recursive=recursive,
                    has_subfolders=has_subfolders,
                )

            if max_samples and max_samples > 0:
                all_items = all_items[:max_samples]

            train_items, val_items = split_items(
                items=all_items,
                val_split=val_split,
                seed=split_seed,
                stratify=stratify,
            )

            train_ds = FolderImageDataset(
                items=train_items,
                classes=classes,
                class_to_idx=class_to_idx,
                folder_counts=folder_counts,
                transform=train_transform,
                return_metadata=return_metadata,
                return_dict=return_dict,
                ignore_corrupt=ignore_corrupt,
                color_mode=color_mode,
            )
            val_ds = FolderImageDataset(
                items=val_items,
                classes=classes,
                class_to_idx=class_to_idx,
                folder_counts=folder_counts,
                transform=val_transform,
                return_metadata=return_metadata,
                return_dict=return_dict,
                ignore_corrupt=ignore_corrupt,
                color_mode=color_mode,
            )

        train_loader = _build_single_dataloader(
            dataset=train_ds,
            batch_size=effective_batch_size,
            device=device,
            split="train",
            max_batches=max_batches,
            num_workers=num_workers,
            shuffle=shuffle if shuffle is not None else True,
            pin_memory=pin_memory,
            drop_last=drop_last,
            distributed_kwargs=distributed_kwargs,
            collate_fn=collate_fn,
            to_device=to_device,
        )
        val_loader = _build_single_dataloader(
            dataset=val_ds,
            batch_size=val_batch_size or effective_batch_size,
            device=device,
            split="val",
            max_batches=val_max_batches,
            num_workers=num_workers,
            shuffle=val_shuffle,
            pin_memory=pin_memory,
            drop_last=False,
            distributed_kwargs=distributed_kwargs,
            collate_fn=collate_fn,
            to_device=to_device,
        )
        return train_loader, val_loader

    # Case 3: Standard single DataLoader creation
    if isinstance(root, Dataset):
        dataset = root
    elif backend.lower() == "fiftyone":
        transform = build_transforms(
            preprocessing=preprocessing,
            split=split,
            img_size=img_size,
            custom_pipeline=custom_transforms,
        )
        dataset = FiftyOneImageDataset(
            dataset_or_folders=root,
            transform=transform,
            max_samples=max_samples,
            return_metadata=return_metadata,
            return_dict=return_dict,
        )
    else:
        transform = build_transforms(
            preprocessing=preprocessing,
            split=split,
            img_size=img_size,
            custom_pipeline=custom_transforms,
        )
        dataset = FolderImageDataset(
            root=root,
            transform=transform,
            recursive=recursive,
            has_subfolders=has_subfolders,
            max_samples=max_samples,
            return_metadata=return_metadata,
            return_dict=return_dict,
            ignore_corrupt=ignore_corrupt,
            color_mode=color_mode,
        )

    if shuffle is None:
        shuffle = (str(split).lower() == "train") if split else False

    return _build_single_dataloader(
        dataset=dataset,
        batch_size=effective_batch_size,
        device=device,
        split=split,
        max_batches=max_batches,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
        distributed_kwargs=distributed_kwargs,
        collate_fn=collate_fn,
        to_device=to_device,
    )


def create_train_val_dataloaders(
    root: Union[str, Path, Sequence[Union[str, Path]], Dataset, Any],
    val_split: float = 0.2,
    val_root: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
    **kwargs: Any,
) -> Tuple[BatchLimitedDataLoader, BatchLimitedDataLoader]:
    """Dedicated convenience helper returning (train_loader, val_loader)."""
    result = create_dataloader(root=root, val_split=val_split, val_root=val_root, **kwargs)
    if isinstance(result, tuple):
        return result
    raise RuntimeError("Expected tuple of (train_loader, val_loader), but got a single loader.")


# Aliases
create_image_dataloader = create_dataloader


def create_fiftyone_dataloader(
    dataset_or_folder: Any,
    batch_size: int = 32,
    images_per_batch: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    split: str = "train",
    preprocessing: Union[str, Callable[[Any], Any], None] = "coco",
    max_batches: Optional[int] = None,
    **kwargs: Any,
) -> BatchLimitedDataLoader:
    """Convenience alias for create_dataloader(..., backend='fiftyone')."""
    return create_dataloader(
        root=dataset_or_folder,
        batch_size=batch_size,
        images_per_batch=images_per_batch,
        device=device,
        split=split,
        preprocessing=preprocessing,
        backend="fiftyone",
        max_batches=max_batches,
        **kwargs,
    )


def get_one_batch(
    dataloader: Union[DataLoader, BatchLimitedDataLoader],
    device: Optional[Union[str, torch.device]] = None,
) -> Any:
    """Extract a single batch from any DataLoader."""
    if isinstance(dataloader, BatchLimitedDataLoader):
        batch = dataloader.one_batch(to_device=(device is None))
    else:
        batch = next(iter(dataloader))

    if device is not None:
        target_dev = torch.device(device) if isinstance(device, str) else device
        batch = _move_to_device(batch, target_dev)

    return batch


def take_batches(
    dataloader: Union[DataLoader, BatchLimitedDataLoader],
    n_batches: int = 20,
    device: Optional[Union[str, torch.device]] = None,
) -> Iterator[Any]:
    """Yield up to `n_batches` from a DataLoader."""
    target_dev = torch.device(device) if isinstance(device, str) else device if device else None

    count = 0
    for batch in dataloader:
        if count >= n_batches:
            break
        if target_dev is not None:
            batch = _move_to_device(batch, target_dev)
        yield batch
        count += 1
