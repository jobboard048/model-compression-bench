from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch
import yaml

from compression_bench.data import build_dataloaders, dataset_class_names
from compression_bench.distill import train_distill
from compression_bench.evaluate import collect_metrics, per_class_accuracy, save_checkpoint
from compression_bench.models import student_model, teacher_model
from compression_bench.prune import apply_global_unstructured_pruning
from compression_bench.quantize import (
    calibrate,
    convert_static_quant,
    dynamic_quant,
    prepare_for_static_quant,
    resolve_quant_backend,
)
from compression_bench.train import train_supervised


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run(
    config_path: Path,
    out_dir: Path,
    skip_static_quant: bool = False,
    epochs_teacher: int | None = None,
    epochs_student: int | None = None,
    epochs_finetune: int | None = None,
) -> pd.DataFrame:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(str(cfg.get("device", "cpu")))
    out_dir.mkdir(parents=True, exist_ok=True)
    n_teacher = int(cfg["epochs_teacher"] if epochs_teacher is None else epochs_teacher)
    n_student = int(cfg["epochs_student"] if epochs_student is None else epochs_student)
    n_finetune = int(cfg["epochs_finetune"] if epochs_finetune is None else epochs_finetune)

    train_loader, test_loader, in_ch, n_cls = build_dataloaders(
        dataset=cfg["dataset"],
        data_dir=cfg["data_dir"],
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg.get("num_workers", 0)),
    )

    teacher = teacher_model(in_channels=in_ch, num_classes=n_cls)
    print("Training teacher (fp32)...")
    train_supervised(
        teacher,
        train_loader,
        epochs=n_teacher,
        device=device,
        lr=float(cfg["lr"]),
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight_decay"]),
        desc="teacher",
    )
    return finish_comparison(
        teacher,
        train_loader,
        test_loader,
        in_ch,
        n_cls,
        cfg,
        device,
        out_dir,
        n_student,
        n_finetune,
        skip_static_quant=skip_static_quant,
    )


def finish_comparison(
    teacher: torch.nn.Module,
    train_loader,
    test_loader,
    in_ch: int,
    n_cls: int,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
    n_student: int,
    n_finetune: int,
    skip_static_quant: bool = False,
) -> pd.DataFrame:
    def write_curve(name: str, rows: list[dict]) -> None:
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / name, index=False)

    save_checkpoint(teacher, out_dir / "teacher_fp32.pt")

    student = student_model(in_channels=in_ch, num_classes=n_cls)
    print("Training student without distillation (control)...")
    student_rows = train_supervised(
        student,
        train_loader,
        epochs=n_student,
        device=device,
        lr=float(cfg["lr"]),
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight_decay"]),
        desc="student-ce",
    )
    write_curve("student_ce_curve.csv", student_rows)
    save_checkpoint(student, out_dir / "student_ce.pt")

    student_kd = student_model(in_channels=in_ch, num_classes=n_cls)
    print("Distilling student from teacher...")
    dcfg = cfg["distill"]
    kd_rows = train_distill(
        student_kd,
        teacher,
        train_loader,
        epochs=n_student,
        device=device,
        temperature=float(dcfg["temperature"]),
        alpha=float(dcfg["alpha"]),
        lr=float(cfg["lr"]),
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    write_curve("student_kd_curve.csv", kd_rows)
    save_checkpoint(student_kd, out_dir / "student_kd.pt")

    pruned = teacher_model(in_channels=in_ch, num_classes=n_cls)
    pruned.load_state_dict(teacher.state_dict())
    print("Pruning teacher (global unstructured L1)...")
    apply_global_unstructured_pruning(pruned, amount=float(cfg["prune"]["amount"]))
    if cfg["prune"].get("finetune", True):
        prune_rows = train_supervised(
            pruned,
            train_loader,
            epochs=n_finetune,
            device=device,
            lr=float(cfg["lr"]) * 0.1,
            momentum=float(cfg["momentum"]),
            weight_decay=float(cfg["weight_decay"]),
            desc="prune-ft",
        )
        write_curve("prune_finetune_curve.csv", prune_rows)
    save_checkpoint(pruned, out_dir / "teacher_pruned.pt")

    print("Dynamic INT8 quantization (Linear layers)...")
    teacher_dyn = dynamic_quant(teacher.cpu())
    try:
        torch.save(teacher_dyn, out_dir / "teacher_dynamic_int8.pt")
    except Exception as exc:
        print(f"  could not save dynamic INT8 checkpoint: {exc}")

    rows = [
        collect_metrics("teacher_fp32", teacher, test_loader, device),
        collect_metrics("student_ce", student, test_loader, device),
        collect_metrics("student_kd", student_kd, test_loader, device),
        collect_metrics("teacher_pruned", pruned, test_loader, device),
        collect_metrics("teacher_dynamic_int8", teacher_dyn, test_loader, torch.device("cpu")),
    ]
    scored: list[tuple[str, torch.nn.Module, torch.device]] = [
        ("teacher_fp32", teacher, device),
        ("student_ce", student, device),
        ("student_kd", student_kd, device),
        ("teacher_pruned", pruned, device),
        ("teacher_dynamic_int8", teacher_dyn, torch.device("cpu")),
    ]

    if not skip_static_quant:
        requested = str(cfg.get("quant", {}).get("backend", "auto"))
        print("Static PTQ (calibrate + convert)...")
        try:
            resolved = resolve_quant_backend(requested)
            print(f"  using quantized engine {resolved!r} (config {requested!r})")
            prepared = prepare_for_static_quant(teacher, backend=resolved)
            calibrate(prepared, train_loader, max_batches=int(cfg["quant"]["calibration_batches"]))
            teacher_ptq = convert_static_quant(prepared)
            rows.append(
                collect_metrics("teacher_static_int8", teacher_ptq, test_loader, torch.device("cpu"))
            )
            scored.append(("teacher_static_int8", teacher_ptq, torch.device("cpu")))
            try:
                torch.save(teacher_ptq, out_dir / "teacher_static_int8.pt")
            except Exception as save_exc:
                print(f"  could not save static INT8 checkpoint: {save_exc}")
        except Exception as exc:
            print(f"  Static PTQ skipped: {exc}")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "comparison.csv"
    json_path = out_dir / "comparison.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")

    names = dataset_class_names(str(cfg["dataset"]))
    per_class_rows = []
    for name, model, run_dev in scored:
        try:
            model.to(run_dev)
            accs = per_class_accuracy(model, test_loader, run_dev, names)
        except Exception:
            model.cpu()
            accs = per_class_accuracy(model, test_loader, torch.device("cpu"), names)
        per_class_rows.append({"name": name, **accs})
    pd.DataFrame(per_class_rows).to_csv(out_dir / "per_class_accuracy.csv", index=False)

    print("\n=== Comparison ===")
    print(df.to_markdown(index=False))
    print(f"\nWrote {csv_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compression technique comparison")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--skip-static-quant", action="store_true")
    args = parser.parse_args()
    run(args.config, args.out, skip_static_quant=args.skip_static_quant)


if __name__ == "__main__":
    main()
