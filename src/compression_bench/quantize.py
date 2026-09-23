from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader


def prepare_for_static_quant(model: nn.Module, backend: str = "x86") -> nn.Module:
    model = copy.deepcopy(model)
    model.eval()
    model.cpu()
    if hasattr(model, "fuse_model"):
        try:
            model.fuse_model()
        except Exception:
            pass
    if backend == "qnnpack":
        torch.backends.quantized.engine = "qnnpack"
        qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
    else:
        torch.backends.quantized.engine = "x86"
        qconfig = torch.ao.quantization.get_default_qconfig("x86")
    model.qconfig = qconfig
    torch.ao.quantization.prepare(model, inplace=True)
    return model


def calibrate(model: nn.Module, loader: DataLoader, max_batches: int) -> None:
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            model(x)
            if i + 1 >= max_batches:
                break


def convert_static_quant(prepared: nn.Module) -> nn.Module:
    return torch.ao.quantization.convert(prepared.eval(), inplace=False)


def dynamic_quant(model: nn.Module) -> nn.Module:
    """INT8 dynamic quant on Linear layers (Conv stays fp32)."""
    model = copy.deepcopy(model)
    model.eval()
    model.cpu()
    return torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
