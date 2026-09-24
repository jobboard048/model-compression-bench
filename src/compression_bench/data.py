from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

CLASS_NAMES = {
    "fashion_mnist": [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ],
    "cifar10": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
}


def dataset_class_names(dataset: str) -> list[str]:
    if dataset not in CLASS_NAMES:
        raise ValueError(f"Unknown dataset: {dataset}")
    return list(CLASS_NAMES[dataset])


def _load_datasets(
    dataset: str,
    data_dir: str | Path,
) -> tuple[Dataset, Dataset, int, int]:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "fashion_mnist":
        mean, std = (0.2860,), (0.3530,)
        in_channels = 1
        num_classes = 10
        tfm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        train_ds = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tfm)
        test_ds = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tfm)
    elif dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        in_channels = 3
        num_classes = 10
        tfm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=tfm)
        test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=tfm)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return train_ds, test_ds, in_channels, num_classes


def build_dataloaders(
    dataset: str,
    data_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, int, int]:
    train_ds, test_ds, in_channels, num_classes = _load_datasets(dataset, data_dir)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader, in_channels, num_classes


def build_train_val_test_loaders(
    dataset: str,
    data_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, int, int]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")

    train_ds, test_ds, in_channels, num_classes = _load_datasets(dataset, data_dir)
    n_val = int(len(train_ds) * val_fraction)
    n_train = len(train_ds) - n_val
    if n_val < 1 or n_train < 1:
        raise ValueError(
            f"val_fraction={val_fraction} yields empty split on {len(train_ds)} samples"
        )

    generator = torch.Generator().manual_seed(seed)
    train_split, val_split = random_split(train_ds, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_split, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_split, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader, in_channels, num_classes
