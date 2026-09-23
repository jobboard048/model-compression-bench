from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    soft = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature**2)
    hard = F.cross_entropy(student_logits, labels)
    return alpha * soft + (1.0 - alpha) * hard


def train_distill(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    epochs: int,
    device: torch.device,
    temperature: float,
    alpha: float,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
) -> None:
    student.to(device)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student.train()
    opt = torch.optim.SGD(
        student.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    for epoch in range(epochs):
        running = 0.0
        n = 0
        for x, y in tqdm(loader, desc=f"distill epoch {epoch + 1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            loss = distillation_loss(s_logits, t_logits, y, temperature, alpha)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        print(f"  distill epoch {epoch + 1}: loss={running / max(n, 1):.4f}")
