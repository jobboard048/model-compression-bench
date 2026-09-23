from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_dataloaders(
    dataset: str,
    data_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, int, int]:
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

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader, in_channels, num_classes
