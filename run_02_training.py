"""Project root launcher: MI training."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


def main() -> int:
    target = Path(__file__).resolve().parent / "code" / "training" / "run_training_pycharm.py"
    target_dir = str(target.parent)
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
