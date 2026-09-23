from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import prune


def apply_global_unstructured_pruning(model: nn.Module, amount: float) -> nn.Module:
    """Magnitude prune Conv2d/Linear weights globally, then make sparsity permanent."""
    parameters_to_prune: list[tuple[nn.Module, str]] = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight is not None:
            parameters_to_prune.append((module, "weight"))
    if not parameters_to_prune:
        raise ValueError("No prunable Conv2d/Linear weights found")
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for module, name in parameters_to_prune:
        prune.remove(module, name)
    return model


def sparsity(model: nn.Module) -> float:
    zeros = 0
    total = 0
    for p in model.parameters():
        if p.ndim >= 1:
            total += p.numel()
            zeros += int((p == 0).sum().item())
    return zeros / total if total else 0.0
