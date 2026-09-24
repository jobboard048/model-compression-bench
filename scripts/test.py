"""Run unit tests (no dataset download). OS-agnostic: python scripts/test.py"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def main() -> None:
    py = venv_python()
    if not py.exists():
        raise SystemExit("No .venv found. Run: python scripts/setup.py")
    cmd = [str(py), "-m", "pytest", str(ROOT / "tests"), "-q"]
    print("+", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
