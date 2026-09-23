from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_supervised(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    desc: str = "train",
) -> None:
    model.to(device)
    model.train()
    opt = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running = 0.0
        n = 0
        for x, y in tqdm(loader, desc=f"{desc} epoch {epoch + 1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        print(f"  {desc} epoch {epoch + 1}: loss={running / max(n, 1):.4f}")
