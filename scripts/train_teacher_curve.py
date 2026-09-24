from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import nn

from compression_bench.data import build_train_val_test_loaders
from compression_bench.evaluate import accuracy, save_checkpoint
from compression_bench.models import teacher_model
from compression_bench.train import suggest_plateau_epoch, train_one_epoch


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
    epochs: int,
    val_fraction: float,
) -> tuple[pd.DataFrame, int | None]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = resolve_device(str(cfg.get("device", "cpu")))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _, in_ch, n_cls = build_train_val_test_loaders(
        dataset=cfg["dataset"],
        data_dir=cfg["data_dir"],
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg.get("num_workers", 0)),
        val_fraction=val_fraction,
        seed=seed,
    )

    teacher = teacher_model(in_channels=in_ch, num_classes=n_cls)
    teacher.to(device)
    opt = torch.optim.SGD(
        teacher.parameters(),
        lr=float(cfg["lr"]),
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    loss_fn = nn.CrossEntropyLoss()

    csv_path = out_dir / "teacher_curve.csv"
    best_path = out_dir / "teacher_curve_best.pt"
    rows: list[dict] = []
    val_accs: list[float] = []
    best_val = -1.0

    print(f"Training teacher curve for {epochs} epochs on {cfg['dataset']} ({device})...")
    print(f"Val fraction={val_fraction}; official test set is not used.")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            teacher,
            train_loader,
            opt,
            loss_fn,
            device,
            desc=f"teacher epoch {epoch}/{epochs}",
        )
        val_acc = accuracy(teacher, val_loader, device)
        val_accs.append(val_acc)
        row = {"epoch": epoch, "train_loss": train_loss, "val_accuracy": val_acc}
        rows.append(row)
        print(f"  epoch {epoch}: train_loss={train_loss:.4f} val_accuracy={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            save_checkpoint(teacher, best_path)
            print(f"  saved best checkpoint ({best_val:.4f}) -> {best_path}")

        pd.DataFrame(rows).to_csv(csv_path, index=False)

    suggested = suggest_plateau_epoch(val_accs)
    df = pd.DataFrame(rows)
    print("\n=== Teacher curve ===")
    print(df.to_markdown(index=False))
    if suggested is None:
        print(
            "\nSuggested epochs_teacher: none (val accuracy did not plateau; "
            "raise --epochs and rerun, or inspect the CSV)."
        )
    else:
        print(f"\nSuggested epochs_teacher: {suggested}")
    print(f"\nWrote {csv_path}")
    return df, suggested


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the teacher and log held-out val accuracy each epoch"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/fashion_mnist.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results/fashion_mnist"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()
    run(args.config, args.out, args.epochs, args.val_fraction)


if __name__ == "__main__":
    main()
