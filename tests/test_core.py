from compression_bench.distill import distillation_loss
from compression_bench.models import student_model, teacher_model
from compression_bench.prune import apply_global_unstructured_pruning, sparsity

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
