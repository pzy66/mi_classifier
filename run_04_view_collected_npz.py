"""Project root launcher: view one collected MI run bundle."""

from __future__ import annotations

from pathlib import Path
import sys


def show_startup_error(title: str, message: str) -> None:
    """Show startup error in popup and stderr."""
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, title, 0x10)
    except Exception:
        pass
    print(f"{title}\n{message}", file=sys.stderr)


def main() -> int:
    target = Path(__file__).resolve().parent / "code" / "viewer" / "run_npz_viewer_pycharm.py"
    if not target.exists():
        show_startup_error("Viewer Startup Failed", f"Viewer entry file not found:\n{target}")
        return 1

    target_dir = str(target.parent)
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)

    try:
        from run_npz_viewer_pycharm import main as viewer_main

        return int(viewer_main())
    except Exception as error:
        show_startup_error("Viewer Startup Failed", str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
