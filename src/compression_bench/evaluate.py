from __future__ import annotations

import copy
import io
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from compression_bench.prune import sparsity


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += y.size(0)
    return correct / total if total else 0.0


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def nonzero_parameter_count(model: nn.Module) -> int:
    n = 0
    for p in model.parameters():
        n += int((p != 0).sum().item())
    return n


def serialized_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 * 1024)


def latency_ms(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup: int = 10,
    measure: int = 50,
) -> dict[str, float]:
    model.eval()
    it = iter(loader)
    x, _ = next(it)
    x = x.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times: list[float] = []
        for _ in range(measure):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    return {"latency_p50_ms": p50, "latency_p95_ms": p95, "batch": float(x.size(0))}


def collect_metrics(
    name: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    # Quantized models typically run on CPU.
    run_device = device
    try:
        model.to(device)
    except Exception:
        run_device = torch.device("cpu")
        model.cpu()
    acc = accuracy(model, loader, run_device)
    lat = latency_ms(model, loader, run_device)
    return {
        "name": name,
        "accuracy": acc,
        "params": parameter_count(model),
        "nonzero_params": nonzero_parameter_count(model),
        "sparsity": sparsity(model),
        "size_mb": serialized_size_mb(model),
        "device": str(run_device),
        **lat,
    }


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(copy.deepcopy(model).cpu().state_dict(), path)
