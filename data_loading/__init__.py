"""Data Loading Toolkit for Computer Vision.

A consolidated, high-performance toolkit for PyTorch and Jupyter notebooks:
- Single folder or multi-folder combined image dataset loading
- Smart multi-GPU and multi-node cluster detection (Slurm, torchrun, MPI)
- Train & validation DataLoader splitting (val_split or val_root)
- Fine-grained batch control (.one_batch(), .take(20), .all())
- Preprocessing pipelines: COCO, MMEngine dict pipelines, Raw/None, Custom
- FiftyOne dataset ingestion and PyTorch DataLoader bridging
- Notebook inspection and batch visualization utilities
"""

from __future__ import annotations

try:
    from .data_loader import (
        COCO_MEAN,
        COCO_STD,
        SUPPORTED_EXTENSIONS,
        BatchLimitedDataLoader,
        FiftyOneImageDataset,
        FolderImageDataset,
        IdentityTransform,
        ImageItem,
        MMEnginePipelineAdapter,
        build_transforms,
        create_dataloader,
        create_fiftyone_dataloader,
        create_image_dataloader,
        create_train_val_dataloaders,
        denormalize_tensor,
        find_images,
        folder_to_fiftyone_dataset,
        get_coco_transforms,
        get_distributed_info,
        get_mmengine_transforms,
        get_one_batch,
        setup_distributed,
        smart_collate_fn,
        split_items,
        take_batches,
    )
    from .utils import (
        compare_pipelines,
        inspect_folder,
        launch_fiftyone_app,
        visualize_batch,
    )
except ImportError:
    from data_loader import (
        COCO_MEAN,
        COCO_STD,
        SUPPORTED_EXTENSIONS,
        BatchLimitedDataLoader,
        FiftyOneImageDataset,
        FolderImageDataset,
        IdentityTransform,
        ImageItem,
        MMEnginePipelineAdapter,
        build_transforms,
        create_dataloader,
        create_fiftyone_dataloader,
        create_image_dataloader,
        create_train_val_dataloaders,
        denormalize_tensor,
        find_images,
        folder_to_fiftyone_dataset,
        get_coco_transforms,
        get_distributed_info,
        get_mmengine_transforms,
        get_one_batch,
        setup_distributed,
        smart_collate_fn,
        split_items,
        take_batches,
    )
    from utils import (
        compare_pipelines,
        inspect_folder,
        launch_fiftyone_app,
        visualize_batch,
    )

__all__ = [
    # Core Loader & Dataset
    "create_dataloader",
    "create_image_dataloader",
    "create_train_val_dataloaders",
    "create_fiftyone_dataloader",
    "FolderImageDataset",
    "FiftyOneImageDataset",
    "BatchLimitedDataLoader",
    "ImageItem",
    "find_images",
    "split_items",
    "folder_to_fiftyone_dataset",
    "get_one_batch",
    "take_batches",
    "smart_collate_fn",
    "SUPPORTED_EXTENSIONS",
    # Distributed & Hardware
    "get_distributed_info",
    "setup_distributed",
    # Preprocessing & Transforms
    "build_transforms",
    "get_coco_transforms",
    "get_mmengine_transforms",
    "MMEnginePipelineAdapter",
    "IdentityTransform",
    "denormalize_tensor",
    "COCO_MEAN",
    "COCO_STD",
    # Utilities
    "inspect_folder",
    "visualize_batch",
    "compare_pipelines",
    "launch_fiftyone_app",
]

__version__ = "0.1.0"
