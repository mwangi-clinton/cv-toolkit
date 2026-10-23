# Data Loading Toolkit for PyTorch

A flexible image loading toolkit designed for building training pipelines with PyTorch.

---

## Features

- **Single Folder or Multi-Folder Ingestion**:
  Pass a single directory string `"/data/folder"` or a list of directories `["/data/cam1", "/data/cam2"]` to automatically build a combined dataset. Source folder origins and unified class mappings are tracked automatically.
- **Train & Validation Splitting**:
  Split your dataset directly into `(train_loader, val_loader)` by passing `val_split=0.2` (ratio) or `val_root="/data/val"`. Automatically configures training augmentations for train, and deterministic validation transforms for val.
- **Multi-Node Slurm Job & Multi-GPU Detection**:
  - Automatically queries available GPUs (`torch.cuda.device_count()`).
  - Detects Slurm multi-node job allocations (`SLURM_PROCID`, `SLURM_NTASKS`, `SLURM_NNODES`).
  - Automatically configures `DistributedSampler` partitioned across all Slurm nodes.
- **Minimal, Cohesive Architecture**:
  The entire engine is streamlined into `data_loader.py` and `utils.py` (matching the rest of `cv-toolkit` like `gradcam`, `cka`, and `flowprune`). Everything is tweakable through parameters on a single unified entry point `create_dataloader()`.
- **Interactive Notebook First**:
  - `loader.one_batch()`: Retrieve a single batch instantly without writing `next(iter(loader))`.
  - `loader.take(20)`: Restrict training or benchmarking to exactly 20 batches (or any $N$).
  - `loader.all()`: Access the full dataset loader.
  - `loader.summary()`: Print rich diagnostic tables (sample count, batch shapes, devices, tensor min/max values).
- **Tweakable Preprocessing**:
  - `preprocessing="coco"`: Standard COCO/ImageNet augmentations (Resize, Flips, ColorJitter, Normalization).
  - `preprocessing="mmengine"`: OpenMMLab / MMEngine dict pipelines (e.g. `[dict(type='Resize', scale=(224, 224)), ...]`).
  - `preprocessing="none"`: Zero processing. Preserves untouched PIL Images or numpy arrays without collation crashes.
  - `preprocessing="custom"`: Any user-defined callable or torchvision pipeline.
- **FiftyOne Backend**:
  Tweak with `backend="fiftyone"` to leverage FiftyOne dataset management, sample tagging, and viewer integration without changing your code structure.

---

## Creating a Plain DataLoader (No Train / Val Split)

If you simply want a single DataLoader without train or validation splitting (e.g. for evaluation, feature extraction, offline processing, or dataset inspection), omit `val_split` and `val_root`. `create_dataloader()` returns a single `BatchLimitedDataLoader` (deterministic unaugmented resize, `shuffle=False` by default):

```python
from data_loading import create_dataloader
loader = create_dataloader(
    root=["/data/camera_a", "/data/camera_b"],
    batch_size=32,
    preprocessing="coco",        # Clean deterministic resize & normalization (no random augmentations)
    device="cpu",                # or 'gpu', 'multi-gpu'
)

# Returns a single DataLoader object (not a tuple)
print(f"Total samples: {len(loader.dataset)}")
batch_imgs, batch_meta = loader.one_batch()
```

---

## Train / Validation Separation

When you want train and validation loaders, simply provide either `val_split` or `val_root`:
Pass `val_split=0.2` (or any ratio $0.0 < \text{val\_split} < 1.0$) to automatically return two DataLoaders:

```python
from data_loading import create_dataloader

train_loader, val_loader = create_dataloader(
    root=["/data/camera_a", "/data/camera_b"],
    val_split=0.2,               # 20% for validation, 80% for train
    batch_size=32,
    val_batch_size=64,           # Optional: larger batch size for validation
    preprocessing="coco",        # Applies train augmentations to train_loader, deterministic resize to val_loader
    split_seed=42,               # Reproducible split
    stratify=True,               # Balances class distribution across splits
)

print(f"Train samples: {len(train_loader.dataset)}")
print(f"Val samples:   {len(val_loader.dataset)}")
```

### 2. Separate Validation Folder (`val_root`)
If you have a dedicated validation folder on disk:

```python
train_loader, val_loader = create_dataloader(
    root="/data/train",
    val_root="/data/val",
    batch_size=32,
)
```

---

## Multi-Node Slurm Job Data Loading

The toolkit automatically detects Slurm multi-node job environments and configures distributed data loading without requiring manual rank configuration:

### 1. How Slurm Environment Detection Works
When launched within a Slurm multi-node allocation (`srun`), the data loader inspects:
- `SLURM_PROCID`: Assigned global process rank across all cluster nodes.
- `SLURM_NTASKS`: Total process count across all nodes (world size).
- `SLURM_LOCALID`: Local worker / GPU index on the current node.
- `SLURM_NNODES`: Total number of physical compute nodes in the allocation.
- `torch.cuda.device_count()`: Number of physical GPUs on the current node.

### 2. Slurm Data Loading in Python
When passing `device="multi-gpu"` (or `device="distributed"`), `create_dataloader()` automatically attaches a `DistributedSampler` initialized with Slurm's detected rank and world size:

```python
from data_loading import create_dataloader, get_distributed_info

# Inspect cluster environment detected by the toolkit
info = get_distributed_info()
print(f"Slurm World Size (tasks): {info['world_size']}")
print(f"Slurm Global Rank:        {info['rank']}")
print(f"Local GPU Index:          {info['local_rank']}")
print(f"Multi-Node Cluster:       {info['is_multinode']}")

# Creates a DistributedSampler partitioned across all Slurm nodes & tasks
loader = create_dataloader(
    root=["/shared/data/cam1", "/shared/data/cam2"],
    batch_size=32,
    device="multi-gpu",          # Automatically engages DistributedSampler for Slurm
)
```

Each Slurm node and worker process only loads its designated, non-overlapping partition of the dataset, preventing duplicate image processing across cluster nodes.

---

## Notebook Usage

```python
from data_loading import create_dataloader, inspect_folder, visualize_batch

folders = ["/data/camera_front", "/data/camera_rear"]

inspect_folder(folders)

loader = create_dataloader(
    root=folders,
    batch_size=8,
    preprocessing="coco",
    device="cpu",
)

images, metadata = loader.one_batch()
print(f"Batch shape: {images.shape}")
print(f"Origin folders: {metadata['folder_source']}")

visualize_batch((images, metadata), num_images=8, cols=4)

for batch_idx, (imgs, meta) in enumerate(loader.take(20)):
    pass
```

---

## Preprocessing Options

```python
# Standard COCO / Vision pipeline
loader = create_dataloader(root=folders, preprocessing="coco", split="train")

# MMEngine / MMCV dict pipeline
mm_pipeline = [
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
]
loader = create_dataloader(root=folders, preprocessing="mmengine", custom_transforms=mm_pipeline)

# No processing (Raw PIL Images, no collation errors)
loader = create_dataloader(root=folders, preprocessing="none")
raw_images, metadata = loader.one_batch()

# Custom transform callable
loader = create_dataloader(root=folders, preprocessing="custom", custom_transforms=my_callable)
```

---

## FiftyOne Backend Integration

```python
# Tweak backend with a single parameter
fo_loader = create_dataloader(
    root=folders,
    backend="fiftyone",
    batch_size=16,
    max_batches=20,
)

images, metadata = fo_loader.one_batch()
print("FiftyOne sample IDs:", metadata["fiftyone_id"])
```

---

