"""Demonstration of image loading tools with multi-folder, smart multi-GPU/multi-node, and train/val splitting.

Showcases:
    1. Single folder and multi-folder combined dataset ingestion.
    2. Separating datasets into (train_loader, val_loader) via `val_split` or `val_root`.
    3. Smart multi-GPU and multi-node cluster checking (`get_distributed_info()`).
    4. Interactive notebook batch controls (.one_batch(), .take(20), .all()).
    5. Tweakable preprocessing (COCO, MMEngine dict pipelines, Raw/None, Custom).
    6. FiftyOne backend integration via `backend='fiftyone'`.
    7. Notebook inspection and mini-batch visualization.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from PIL import Image
import torch

try:
    from . import (
        create_dataloader,
        create_train_val_dataloaders,
        get_distributed_info,
        inspect_folder,
        visualize_batch,
    )
except ImportError:
    from __init__ import (
        create_dataloader,
        create_train_val_dataloaders,
        get_distributed_info,
        inspect_folder,
        visualize_batch,
    )


def create_demo_folders(tmpdir: Path) -> tuple[Path, Path]:
    """Create two sample image directories with images."""
    dir_a = tmpdir / "camera_a"
    dir_b = tmpdir / "camera_b"

    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)

    # Populate camera_a
    for i in range(4):
        ext = "jpg" if i % 2 == 0 else "png"
        img = Image.new("RGB", (128, 128), color=(200, 50 * i, 50))
        img.save(dir_a / f"camA_{i:03d}.{ext}")

    # Populate camera_b
    for i in range(4):
        ext = "jpg" if i % 2 == 0 else "png"
        img = Image.new("RGB", (128, 128), color=(50, 50 * i, 200))
        img.save(dir_b / f"camB_{i:03d}.{ext}")

    return dir_a, dir_b


def main() -> None:
    print("=" * 70)
    print("  Computer Vision Toolkit: Image Data Loading Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        dir_a, dir_b = create_demo_folders(tmp_path)

        print("\n--- 1. Multi-Folder Combined Dataset ---")
        inspect_folder([dir_a, dir_b])

        combined_loader = create_dataloader(
            root=[dir_a, dir_b],          # List of folders combined!
            batch_size=2,
            preprocessing="coco",
            img_size=(224, 224),
            device="cpu",
        )
        print(f"Total batches in combined loader: {len(combined_loader)}")
        print(f"Total samples: {len(combined_loader.dataset)}")

        # Inspect origin of samples
        batch_imgs, batch_meta = combined_loader.one_batch()
        print(f"One batch shape: {batch_imgs.shape}")
        print(f"Sample filenames: {batch_meta['filename']}")
        print(f"Sample folder sources: {batch_meta['folder_source']}")

        print("\n--- 2. Train / Validation DataLoader Separation ---")
        train_loader, val_loader = create_dataloader(
            root=[dir_a, dir_b],
            val_split=0.25,               # 25% for validation, 75% for train
            batch_size=2,
            preprocessing="coco",
            img_size=(224, 224),
        )
        print(f"Train samples: {len(train_loader.dataset)}, batches: {len(train_loader)}")
        print(f"Validation samples: {len(val_loader.dataset)}, batches: {len(val_loader)}")

        # Validation with separate folder
        train_l2, val_l2 = create_dataloader(
            root=dir_a,
            val_root=dir_b,               # Separate validation folder
            batch_size=2,
        )
        print(f"Explicit val_root: Train={len(train_l2.dataset)}, Val={len(val_l2.dataset)}")

        print("\n--- 3. Smart Multi-GPU & Multi-Node Checking ---")
        dist_info = get_distributed_info()
        print("Distributed Settings Detected:")
        print(f"  - GPUs Available       : {dist_info['cuda_device_count']}")
        print(f"  - CUDA Active          : {dist_info['cuda_available']}")
        print(f"  - World Size           : {dist_info['world_size']}")
        print(f"  - Global Rank          : {dist_info['rank']}")
        print(f"  - Local Rank (GPU idx) : {dist_info['local_rank']}")
        print(f"  - Multi-Node Cluster   : {dist_info['is_multinode']}")
        print(f"  - Master Address:Port  : {dist_info['master_addr']}:{dist_info['master_port']}")

        print("\n--- 4. Notebook Batch Controls ---")
        print("Iterating over exactly 2 batches using .take(2):")
        for idx, (imgs, meta) in enumerate(combined_loader.take(2)):
            print(f"  Batch {idx}: shape={imgs.shape}, files={meta['filename']}")

        print("\n--- 5. Preprocessing Modes ---")

        # COCO
        coco_loader = create_dataloader([dir_a, dir_b], batch_size=2, preprocessing="coco", max_batches=1)
        print(f"  [COCO] Output tensor shape: {coco_loader.one_batch()[0].shape}")

        # MMEngine
        mm_loader = create_dataloader([dir_a, dir_b], batch_size=2, preprocessing="mmengine", max_batches=1)
        print(f"  [MMEngine] Output tensor shape: {mm_loader.one_batch()[0].shape}")

        # None / Raw (Preserves raw PIL images without errors)
        raw_loader = create_dataloader([dir_a, dir_b], batch_size=2, preprocessing="none", max_batches=1)
        raw_imgs, _ = raw_loader.one_batch()
        print(f"  [None/Raw] First image type: {type(raw_imgs[0])}, size: {raw_imgs[0].size}")

        # Custom callable
        custom_fn = lambda pil: torch.zeros(3, 100, 100)
        cust_loader = create_dataloader([dir_a, dir_b], batch_size=2, preprocessing="custom", custom_transforms=custom_fn, max_batches=1)
        print(f"  [Custom] Output tensor shape: {cust_loader.one_batch()[0].shape}")

        print("\n--- 6. FiftyOne Backend Integration ---")
        try:
            fo_loader = create_dataloader(
                root=[dir_a, dir_b],
                batch_size=2,
                backend="fiftyone",        # Simply tweak the backend!
                preprocessing="coco",
                max_batches=1,
            )
            fo_imgs, fo_meta = fo_loader.one_batch()
            print(f"  FiftyOne Loader batch shape: {fo_imgs.shape}")
            print(f"  FiftyOne sample IDs: {fo_meta.get('fiftyone_id')}")
        except Exception as e:
            print(f"  FiftyOne note: {e}")

        print("\n--- 7. Loader Summary Table ---")
        train_loader.summary()

    print("\n[SUCCESS] Unified image loading toolkit executed successfully!")


if __name__ == "__main__":
    main()
