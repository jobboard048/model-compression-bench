"""One experiment command: python scripts/run.py

Train the teacher once (stop when val accuracy plateaus, or at --epochs).
Then compare distillation, pruning, and quantization on that same teacher.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_VENV_PY = _ROOT / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
_VENV_FLAG = "MCB_IN_PROJECT_VENV"


def _deps_ok() -> bool:
    try:
        import pandas  # noqa: F401
        import torch  # noqa: F401
        import yaml  # noqa: F401
    except ImportError:
        return False
    return True


if not _deps_ok():
    if os.environ.get(_VENV_FLAG) == "1":
        raise SystemExit(
            "This interpreter is missing torch/pandas. Run: python scripts/setup.py"
        )
    if not _VENV_PY.is_file():
        raise SystemExit("No .venv found. Run: python scripts/setup.py")
    env = os.environ.copy()
    env[_VENV_FLAG] = "1"
    raise SystemExit(
        subprocess.call(
            [str(_VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=_ROOT,
            env=env,
        )
    )

import argparse
import json
import random

import pandas as pd
import torch
import yaml
from torch import nn

from compression_bench.data import build_train_val_test_loaders
from compression_bench.evaluate import accuracy, save_checkpoint
from compression_bench.models import teacher_model
from compression_bench.train import suggest_plateau_epoch, train_one_epoch
from run_comparison import finish_comparison, resolve_device
from save_sample_grid import save_sample_grid

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "fashion_mnist": ROOT / "configs" / "fashion_mnist.yaml",
    "cifar10": ROOT / "configs" / "default.yaml",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_teacher(
    cfg: dict,
    train_loader,
    val_loader,
    device: torch.device,
    out_dir: Path,
    max_epochs: int,
    in_ch: int,
    n_cls: int,
) -> tuple[nn.Module, int]:
    teacher = teacher_model(in_channels=in_ch, num_classes=n_cls)
    teacher.to(device)
    opt = torch.optim.SGD(
        teacher.parameters(),
        lr=float(cfg["lr"]),
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    loss_fn = nn.CrossEntropyLoss()
    best_path = out_dir / "teacher_fp32.pt"
    rows: list[dict] = []
    val_accs: list[float] = []
    best_val = -1.0
    chosen = max_epochs

    print(f"Training teacher (max {max_epochs} epochs, stop when val accuracy plateaus)...")
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            teacher,
            train_loader,
            opt,
            loss_fn,
            device,
            desc=f"teacher {epoch}/{max_epochs}",
        )
        val_acc = accuracy(teacher, val_loader, device)
        val_accs.append(val_acc)
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_accuracy": val_acc})
        print(f"  epoch {epoch}: train_loss={train_loss:.4f} val_accuracy={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            save_checkpoint(teacher, best_path)
        pd.DataFrame(rows).to_csv(out_dir / "teacher_curve.csv", index=False)
        plateau = suggest_plateau_epoch(val_accs)
        if plateau is not None:
            chosen = plateau
            print(f"Val accuracy plateaued. Using epoch {chosen} (best checkpoint).")
            break
    else:
        print(f"No plateau within {max_epochs} epochs; using {chosen}.")

    teacher.load_state_dict(torch.load(best_path, map_location="cpu"))
    teacher.to(device)
    return teacher, chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the compression experiment")
    parser.add_argument("--dataset", choices=sorted(CONFIGS), default="fashion_mnist")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum teacher epochs (training stops earlier if val accuracy plateaus)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--skip-static-quant", action="store_true")
    args = parser.parse_args()

    config = args.config if args.config is not None else CONFIGS[args.dataset]
    if not config.is_absolute():
        config = (Path.cwd() / config).resolve()
    dataset_name = args.dataset if args.config is None else config.stem
    out = args.out if args.out is not None else ROOT / "results" / dataset_name
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    print(f"Saving class sample grid to {out / 'samples.png'} ...")
    save_sample_grid(str(cfg["dataset"]), Path(cfg["data_dir"]), out / "samples.png")
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(str(cfg.get("device", "cpu")))

    train_loader, val_loader, test_loader, in_ch, n_cls = build_train_val_test_loaders(
        dataset=cfg["dataset"],
        data_dir=cfg["data_dir"],
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg.get("num_workers", 0)),
        val_fraction=args.val_fraction,
        seed=int(cfg.get("seed", 42)),
    )

    teacher, n_epochs = train_teacher(
        cfg, train_loader, val_loader, device, out, args.epochs, in_ch, n_cls
    )
    print(f"Students will train for {n_epochs} epochs.")
    n_finetune = int(cfg.get("epochs_finetune", 1))
    summary = {
        "dataset": cfg["dataset"],
        "device": str(device),
        "max_epochs": args.epochs,
        "selected_epochs": n_epochs,
        "val_fraction": args.val_fraction,
        "epochs_finetune": n_finetune,
        "out_dir": str(out),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    finish_comparison(
        teacher,
        train_loader,
        test_loader,
        in_ch,
        n_cls,
        cfg,
        device,
        out,
        n_student=n_epochs,
        n_finetune=n_finetune,
        skip_static_quant=args.skip_static_quant,
    )


if __name__ == "__main__":
    main()
