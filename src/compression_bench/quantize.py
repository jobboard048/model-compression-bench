from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader

PREFERRED_QUANT_BACKENDS = ("x86", "fbgemm", "onednn", "qnnpack")


def resolve_quant_backend(
    requested: str = "auto",
    supported: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Pick a quantized engine this PyTorch build can actually run."""
    engines = list(
        supported
        if supported is not None
        else getattr(torch.backends.quantized, "supported_engines", [])
    )
    name = (requested or "auto").strip().lower()
    if name not in {"auto", "", "none"} and name in engines:
        return name
    for candidate in PREFERRED_QUANT_BACKENDS:
        if candidate in engines:
            return candidate
    if engines:
        return engines[0]
    raise RuntimeError(
        "No quantized engine in this PyTorch build "
        f"(requested={requested!r}, supported={engines})"
    )


def prepare_for_static_quant(model: nn.Module, backend: str = "auto") -> nn.Module:
    model = copy.deepcopy(model)
    model.eval()
    model.cpu()
    if hasattr(model, "fuse_model"):
        try:
            model.fuse_model()
        except Exception:
            pass
    resolved = resolve_quant_backend(backend)
    torch.backends.quantized.engine = resolved
    qconfig = torch.ao.quantization.get_default_qconfig(resolved)
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
