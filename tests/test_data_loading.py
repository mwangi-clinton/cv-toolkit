"""Unit tests for the unified data-loading toolkit."""

from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
import pytest
import torch

from data_loading import (
    BatchLimitedDataLoader,
    FiftyOneImageDataset,
    FolderImageDataset,
    build_transforms,
    create_dataloader,
    create_fiftyone_dataloader,
    create_train_val_dataloaders,
    denormalize_tensor,
    find_images,
    get_coco_transforms,
    get_distributed_info,
    get_mmengine_transforms,
    get_one_batch,
    inspect_folder,
    smart_collate_fn,
    split_items,
    take_batches,
    visualize_batch,
)


@pytest.fixture
def multi_folder_setup():
    """Create two sample image directories with images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        dir_a = root / "folder_a"
        dir_b = root / "folder_b"
        dir_a.mkdir(parents=True)
        dir_b.mkdir(parents=True)

        for i in range(4):
            ext = "jpg" if i % 2 == 0 else "png"
            img_a = Image.new("RGB", (64, 64), color=(200, 10 * i, 10))
            img_a.save(dir_a / f"img_a_{i}.{ext}")

        for i in range(4):
            ext = "jpg" if i % 2 == 0 else "png"
            img_b = Image.new("RGB", (64, 64), color=(10, 10 * i, 200))
            img_b.save(dir_b / f"img_b_{i}.{ext}")

        yield dir_a, dir_b


def test_find_images_single_folder(multi_folder_setup):
    dir_a, _ = multi_folder_setup
    items, classes, class_to_idx, counts = find_images(dir_a)
    assert len(items) == 4
    assert counts[dir_a.name] == 4


def test_find_images_multi_folder(multi_folder_setup):
    dir_a, dir_b = multi_folder_setup
    items, classes, class_to_idx, counts = find_images([dir_a, dir_b])
    # 4 from dir_a + 4 from dir_b = 8 total
    assert len(items) == 8
    assert counts[dir_a.name] == 4
    assert counts[dir_b.name] == 4
    assert {item.folder_source for item in items} == {dir_a.name, dir_b.name}


def test_folder_dataset_multi_folder(multi_folder_setup):
    dir_a, dir_b = multi_folder_setup
    dataset = FolderImageDataset(root=[dir_a, dir_b], return_metadata=True)
    assert len(dataset) == 8

    img, meta = dataset[0]
    assert isinstance(img, Image.Image)
    assert "folder_source" in meta
    assert meta["folder_source"] in (dir_a.name, dir_b.name)


def test_split_items(multi_folder_setup):
    dir_a, dir_b = multi_folder_setup
    items, _, _, _ = find_images([dir_a, dir_b])
    train_items, val_items = split_items(items, val_split=0.25, seed=42, stratify=True)
    assert len(train_items) + len(val_items) == len(items)
    assert len(val_items) == 2
    assert len(train_items) == 6


def test_create_train_val_split_dataloaders(multi_folder_setup):
    dir_a, dir_b = multi_folder_setup
    train_loader, val_loader = create_dataloader(
        root=[dir_a, dir_b],
        val_split=0.25,
        batch_size=2,
        preprocessing="coco",
        img_size=64,
        split_seed=42,
    )
    assert isinstance(train_loader, BatchLimitedDataLoader)
    assert isinstance(val_loader, BatchLimitedDataLoader)

    # Total 8 samples: 6 train, 2 val
    assert len(train_loader.dataset) == 6
    assert len(val_loader.dataset) == 2

    train_imgs, train_meta = train_loader.one_batch()
    val_imgs, val_meta = val_loader.one_batch()

    assert train_imgs.shape == (2, 3, 64, 64)
    assert val_imgs.shape == (2, 3, 64, 64)


def test_create_train_val_dataloaders_helper(multi_folder_setup):
    dir_a, dir_b = multi_folder_setup
    train_loader, val_loader = create_train_val_dataloaders(
        root=[dir_a, dir_b],
        val_split=0.25,
        batch_size=2,
    )
    assert len(train_loader.dataset) == 6
    assert len(val_loader.dataset) == 2


def test_create_dataloader_val_root(multi_folder_setup):
    dir_a, dir_b = multi_folder_setup
    train_loader, val_loader = create_dataloader(
        root=dir_a,
        val_root=dir_b,
        batch_size=2,
    )
    assert len(train_loader.dataset) == 4
    assert len(val_loader.dataset) == 4


def test_fiftyone_backend_val_split(multi_folder_setup):
    dir_a, dir_b = multi_folder_setup
    train_loader, val_loader = create_dataloader(
        root=[dir_a, dir_b],
        backend="fiftyone",
        val_split=0.25,
        batch_size=2,
        img_size=64,
    )
    assert len(train_loader.dataset) == 6
    assert len(val_loader.dataset) == 2
    imgs, meta = train_loader.one_batch()
    assert imgs.shape == (2, 3, 64, 64)
    assert "fiftyone_id" in meta


def test_get_distributed_info():
    info = get_distributed_info()
    assert "is_distributed" in info
    assert "world_size" in info
    assert "rank" in info
    assert "local_rank" in info
    assert "master_addr" in info
    assert "master_port" in info
    assert isinstance(info["cuda_device_count"], int)


def test_transforms_coco():
    tf = build_transforms(preprocessing="coco", split="train", img_size=(128, 128))
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    tensor = tf(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 128, 128)
    assert tensor.dtype == torch.float32


def test_transforms_mmengine():
    tf = build_transforms(preprocessing="mmengine", split="val", img_size=(100, 100))
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    tensor = tf(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 100, 100)


def test_transforms_none():
    tf = build_transforms(preprocessing="none")
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    out = tf(img)
    assert isinstance(out, Image.Image)


def test_transforms_custom():
    custom_fn = lambda img: torch.ones((3, 50, 50))
    tf = build_transforms(preprocessing="custom", custom_pipeline=custom_fn)
    img = Image.new("RGB", (64, 64))
    res = tf(img)
    assert isinstance(res, torch.Tensor)
    assert res.shape == (3, 50, 50)


def test_denormalize_tensor():
    t = torch.randn(2, 3, 32, 32)
    denorm = denormalize_tensor(t)
    assert denorm.min() >= 0.0
    assert denorm.max() <= 1.0


def test_smart_collate_fn():
    tensors = [torch.zeros(3, 10, 10), torch.ones(3, 10, 10)]
    batch_t = smart_collate_fn(tensors)
    assert isinstance(batch_t, torch.Tensor)
    assert batch_t.shape == (2, 3, 10, 10)

    pils = [Image.new("RGB", (10, 10)), Image.new("RGB", (20, 20))]
    batch_p = smart_collate_fn(pils)
    assert isinstance(batch_p, list)
    assert len(batch_p) == 2


def test_inspect_folder_and_visualize(multi_folder_setup):
    dir_a, dir_b = multi_folder_setup
    info = inspect_folder([dir_a, dir_b])
    assert info["total_images"] == 8
    assert len(info["folder_counts"]) == 2

    loader = create_dataloader(root=[dir_a, dir_b], batch_size=4, preprocessing="coco", img_size=32)
    batch = loader.one_batch()
    fig = visualize_batch(batch, num_images=4, cols=2)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_single_dataloader_without_val_split_or_root(multi_folder_setup):
    """When val_split and val_root are omitted, returns a single DataLoader (not a tuple)."""
    dir_a, dir_b = multi_folder_setup

    # 1. Single folder
    loader_single = create_dataloader(root=dir_a, batch_size=2)
    assert isinstance(loader_single, BatchLimitedDataLoader)
    assert not isinstance(loader_single, tuple)
    assert len(loader_single.dataset) == 4
    # Default is not train, so sampler is SequentialSampler (not RandomSampler)
    assert isinstance(loader_single.sampler, torch.utils.data.SequentialSampler)

    # 2. Multi folder combined
    loader_multi = create_dataloader(root=[dir_a, dir_b], batch_size=3)
    assert isinstance(loader_multi, BatchLimitedDataLoader)
    assert not isinstance(loader_multi, tuple)
    assert len(loader_multi.dataset) == 8


def test_single_dataloader_split_none_deterministic(multi_folder_setup):
    """split=None or 'default' creates a deterministic plain loader without augmentations."""
    dir_a, _ = multi_folder_setup
    loader = create_dataloader(root=dir_a, split=None, batch_size=2)
    assert isinstance(loader, BatchLimitedDataLoader)
    batch_1, _ = loader.one_batch()
    batch_2, _ = loader.one_batch()
    assert torch.allclose(batch_1, batch_2)


def test_single_dataloader_explicit_train_mode(multi_folder_setup):
    """split='train' without val_split still yields a single loader with training shuffle."""
    dir_a, _ = multi_folder_setup
    loader = create_dataloader(root=dir_a, split="train", batch_size=2)
    assert isinstance(loader, BatchLimitedDataLoader)
    assert not isinstance(loader, tuple)
    assert len(loader.dataset) == 4
