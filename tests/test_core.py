from compression_bench.distill import distillation_loss
from compression_bench.models import student_model, teacher_model
from compression_bench.prune import apply_global_unstructured_pruning, sparsity
from compression_bench.quantize import resolve_quant_backend
from compression_bench.train import suggest_plateau_epoch

import torch


def test_teacher_student_shapes() -> None:
    t = teacher_model()
    s = student_model()
    x = torch.randn(2, 1, 28, 28)
    assert t(x).shape == (2, 10)
    assert s(x).shape == (2, 10)


def test_teacher_student_cifar_shapes() -> None:
    t = teacher_model(in_channels=3)
    s = student_model(in_channels=3)
    x = torch.randn(2, 3, 32, 32)
    assert t(x).shape == (2, 10)
    assert s(x).shape == (2, 10)


def test_pruning_increases_sparsity() -> None:
    m = student_model()
    before = sparsity(m)
    apply_global_unstructured_pruning(m, amount=0.5)
    after = sparsity(m)
    assert after > before
    assert after > 0.3


def test_distillation_loss_finite() -> None:
    s = torch.randn(4, 10)
    t = torch.randn(4, 10)
    y = torch.randint(0, 10, (4,))
    loss = distillation_loss(s, t, y, temperature=4.0, alpha=0.7)
    assert torch.isfinite(loss)


def test_suggest_plateau_epoch_rises_then_flattens() -> None:
    accuracies = [0.50, 0.70, 0.85, 0.90, 0.91, 0.9105, 0.9103, 0.9104, 0.9102]
    assert suggest_plateau_epoch(accuracies, min_delta=0.002, patience=4) == 5


def test_suggest_plateau_epoch_none_if_still_climbing() -> None:
    accuracies = [0.50, 0.60, 0.70, 0.80, 0.90]
    assert suggest_plateau_epoch(accuracies, min_delta=0.002, patience=4) is None


def test_suggest_plateau_epoch_empty() -> None:
    assert suggest_plateau_epoch([]) is None


def test_resolve_quant_backend_honours_requested() -> None:
    assert resolve_quant_backend("onednn", supported=["x86", "onednn"]) == "onednn"


def test_resolve_quant_backend_auto_prefers_x86() -> None:
    assert resolve_quant_backend("auto", supported=["onednn", "x86"]) == "x86"


def test_resolve_quant_backend_falls_back_when_missing() -> None:
    assert resolve_quant_backend("x86", supported=["onednn"]) == "onednn"


def test_resolve_quant_backend_raises_without_engines() -> None:
    try:
        resolve_quant_backend("auto", supported=[])
    except RuntimeError as exc:
        assert "No quantized engine" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_dataset_class_names() -> None:
    from compression_bench.data import dataset_class_names

    assert len(dataset_class_names("fashion_mnist")) == 10
    assert len(dataset_class_names("cifar10")) == 10
