"""Create .venv and install this project. OS-agnostic: python scripts/setup.py"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=cwd)


def has_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None


def torch_installed(py: Path) -> bool:
    r = subprocess.run(
        [str(py), "-c", "import torch, torchvision"],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def main() -> None:
    print(f"Repo: {ROOT}")
    if not VENV.exists():
        print(f"Creating {VENV} ...")
        venv.create(VENV, with_pip=True)
    py = venv_python()
    if not py.exists():
        raise SystemExit(f"venv python missing: {py}")

    run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    if not torch_installed(py):
        torch_cmd = [str(py), "-m", "pip", "install", "torch", "torchvision"]
        if has_nvidia():
            print("NVIDIA GPU detected; installing default PyTorch wheels.")
        else:
            print("No nvidia-smi; installing CPU PyTorch.")
            torch_cmd.extend(["--index-url", CPU_INDEX])
        run(torch_cmd)
    else:
        print("torch already installed in .venv; skipping.")

    run([str(py), "-m", "pip", "install", "-e", ".[test]"], cwd=ROOT)
    print("\nSetup done. Next:")
    print("  python scripts/test.py")
    print("  python scripts/run.py")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
