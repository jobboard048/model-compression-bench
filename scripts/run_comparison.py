from __future__ import annotations

import argparse
import copy
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
from compression_bench.prune import (
    apply_global_unstructured_pruning,
    finalize_pruning,
    sparsity,
)
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


def curve_amounts(cfg: dict) -> list[float]:
    raw = cfg.get("prune", {}).get("curve_amounts", [0.0, 0.3, 0.5, 0.8])
    return [float(amount) for amount in raw]


def finetune_at_sparsity(
    teacher: torch.nn.Module,
    amount: float,
    train_loader,
    cfg: dict,
    device: torch.device,
    n_finetune: int,
    desc: str,
) -> tuple[torch.nn.Module, list[dict]]:
    """Copy the teacher, optionally prune with the mask held, then fine-tune."""
    model = copy.deepcopy(teacher)
    if amount > 0:
        print(
            f"Pruning teacher (global unstructured L1, amount={amount:.2f}); "
            "mask stays on during fine-tune..."
        )
        apply_global_unstructured_pruning(model, amount=amount)
    else:
        print("Fine-tune control (amount=0, no prune)...")
    rows: list[dict] = []
    if cfg.get("prune", {}).get("finetune", True) and n_finetune > 0:
        rows = train_supervised(
            model,
            train_loader,
            epochs=n_finetune,
            device=device,
            lr=float(cfg["lr"]) * 0.1,
            momentum=float(cfg["momentum"]),
            weight_decay=float(cfg["weight_decay"]),
            desc=desc,
        )
    if amount > 0:
        finalize_pruning(model, expected_amount=amount)
        print(f"  baked sparsity={sparsity(model):.4f} (target {amount:.2f})")
    return model, rows


def run_prune_curve(
    teacher: torch.nn.Module,
    train_loader,
    test_loader,
    cfg: dict,
    device: torch.device,
    n_finetune: int,
    out_dir: Path,
) -> tuple[torch.nn.Module, pd.DataFrame]:
    """Fine-tune one copy per sparsity. Reuse the configured amount for teacher_pruned."""
    target = float(cfg["prune"]["amount"])
    amounts = curve_amounts(cfg)
    if not any(abs(amount - target) < 1e-6 for amount in amounts):
        amounts = [*amounts, target]
    pruned: torch.nn.Module | None = None
    target_rows: list[dict] = []
    curve_rows: list[dict] = []
    for amount in amounts:
        desc = "prune-ft" if abs(amount - target) < 1e-6 else f"prune-ft-{amount:.1f}"
        model, rows = finetune_at_sparsity(
            teacher, amount, train_loader, cfg, device, n_finetune, desc
        )
        metrics = collect_metrics(f"amount_{amount:.1f}", model, test_loader, device)
        curve_rows.append(
            {
                "amount": amount,
                "sparsity": metrics["sparsity"],
                "accuracy": metrics["accuracy"],
                "nonzero_params": metrics["nonzero_params"],
            }
        )
        if abs(amount - target) < 1e-6:
            pruned = model
            target_rows = rows
    if pruned is None:
        raise RuntimeError(f"Prune curve did not include amount {target}")
    if target_rows:
        pd.DataFrame(target_rows).to_csv(out_dir / "prune_finetune_curve.csv", index=False)
    curve = pd.DataFrame(curve_rows)
    curve_path = out_dir / "prune_sparsity_curve.csv"
    curve.to_csv(curve_path, index=False)
    print(f"Wrote {curve_path}")
    save_checkpoint(pruned, out_dir / "teacher_pruned.pt")
    return pruned, curve


def upsert_named_row(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    name = row["name"]
    if "name" in df.columns and (df["name"] == name).any():
        idx = df.index[df["name"] == name][0]
        for key, value in row.items():
            df.loc[idx, key] = value
        return df
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def refresh_pruned_rows(
    pruned: torch.nn.Module,
    test_loader,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
) -> pd.DataFrame:
    """Replace only the teacher_pruned rows. Leave distillation and INT8 rows."""
    metrics = collect_metrics("teacher_pruned", pruned, test_loader, device)
    csv_path = out_dir / "comparison.csv"
    json_path = out_dir / "comparison.json"
    if csv_path.is_file():
        df = upsert_named_row(pd.read_csv(csv_path), metrics)
    else:
        df = pd.DataFrame([metrics])
    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")

    names = dataset_class_names(str(cfg["dataset"]))
    try:
        pruned.to(device)
        accs = per_class_accuracy(pruned, test_loader, device, names)
    except Exception:
        pruned.cpu()
        accs = per_class_accuracy(pruned, test_loader, torch.device("cpu"), names)
    per_path = out_dir / "per_class_accuracy.csv"
    per_row = {"name": "teacher_pruned", **accs}
    if per_path.is_file():
        per_df = upsert_named_row(pd.read_csv(per_path), per_row)
    else:
        per_df = pd.DataFrame([per_row])
    per_df.to_csv(per_path, index=False)
    print("\n=== Prune refresh ===")
    print(df.to_markdown(index=False))
    print(f"\nWrote {csv_path}")
    return df


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

    pruned, _curve = run_prune_curve(
        teacher, train_loader, test_loader, cfg, device, n_finetune, out_dir
    )

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
