from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    desc: str = "train",
) -> float:
    model.train()
    running = 0.0
    n = 0
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        running += loss.item() * x.size(0)
        n += x.size(0)
    return running / max(n, 1)


def train_supervised(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    desc: str = "train",
) -> list[dict]:
    model.to(device)
    opt = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()
    losses: list[dict] = []
    for epoch in range(epochs):
        loss = train_one_epoch(
            model,
            loader,
            opt,
            loss_fn,
            device,
            desc=f"{desc} epoch {epoch + 1}/{epochs}",
        )
        print(f"  {desc} epoch {epoch + 1}: loss={loss:.4f}")
        losses.append({"epoch": epoch + 1, "train_loss": loss})
    return losses


def suggest_plateau_epoch(
    accuracies: list[float],
    min_delta: float = 0.002,
    patience: int = 4,
) -> int | None:
    """Return the 1-based best epoch once val accuracy has plateaued, else None.

    Plateau means the best accuracy so far has not improved by at least
    ``min_delta`` for ``patience`` following epochs.
    """
    if not accuracies:
        return None
    best = accuracies[0]
    best_epoch = 1
    no_improve = 0
    for epoch, acc in enumerate(accuracies, start=1):
        if acc >= best + min_delta:
            best = acc
            best_epoch = epoch
            no_improve = 0
        elif epoch > best_epoch:
            no_improve += 1
            if no_improve >= patience:
                return best_epoch
    return None
