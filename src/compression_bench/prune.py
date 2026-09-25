from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import prune

_PRUNABLE = (nn.Conv2d, nn.Linear)


def _prunable_modules(model: nn.Module) -> list[tuple[nn.Module, str]]:
    pairs: list[tuple[nn.Module, str]] = []
    for module in model.modules():
        if isinstance(module, _PRUNABLE) and module.weight is not None:
            pairs.append((module, "weight"))
    return pairs


def apply_global_unstructured_pruning(model: nn.Module, amount: float) -> nn.Module:
    """Magnitude-prune Conv2d/Linear weights. Leave the mask on until finalize."""
    if amount <= 0:
        return model
    parameters_to_prune = _prunable_modules(model)
    if not parameters_to_prune:
        raise ValueError("No prunable Conv2d/Linear weights found")
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return model


def finalize_pruning(
    model: nn.Module,
    expected_amount: float | None = None,
    tolerance: float = 0.02,
) -> nn.Module:
    """Bake masks into real zero weights, then check sparsity is near the target."""
    for module, name in _prunable_modules(model):
        if prune.is_pruned(module):
            prune.remove(module, name)
    if expected_amount is not None and expected_amount > 0:
        measured = prunable_weight_sparsity(model)
        if abs(measured - expected_amount) > tolerance:
            raise RuntimeError(
                f"Prune sparsity {measured:.4f} is not within {tolerance:.2f} "
                f"of requested amount {expected_amount:.4f}. "
                "The mask was likely removed before fine-tuning."
            )
    return model


def prunable_weight_sparsity(model: nn.Module) -> float:
    """Zero fraction of Conv2d/Linear weights (masked values while a hook is on)."""
    zeros = 0
    total = 0
    for module, _name in _prunable_modules(model):
        weight = module.weight.detach()
        total += weight.numel()
        zeros += int((weight == 0).sum().item())
    return zeros / total if total else 0.0


def sparsity(model: nn.Module) -> float:
    zeros = 0
    total = 0
    for p in model.parameters():
        if p.ndim >= 1:
            total += p.numel()
            zeros += int((p == 0).sum().item())
    return zeros / total if total else 0.0
