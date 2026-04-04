"""PyCharm launcher for training from datasets/custom_mi."""

from __future__ import annotations

import sys
import traceback


def show_startup_error(title: str, message: str) -> None:
    """Show a visible startup error."""
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, title, 0x10)
    except Exception:
        pass
    print(f"{title}\n{message}", file=sys.stderr)


def main() -> int:
    """Run training with default settings."""
    try:
        from train_custom_dataset import main as train_main
    except ModuleNotFoundError as error:
        missing_name = getattr(error, "name", None) or str(error)
        show_startup_error(
            "Startup Failed",
            f"Missing dependency: {missing_name}\n\n"
            "Please switch PyCharm interpreter to MI env, then install:\n"
            "pip install -r requirements.txt\n"
            "pip install -r requirements-realtime.txt",
        )
        return 1
    except Exception:
        show_startup_error("Startup Failed", traceback.format_exc())
        return 1

    try:
        return train_main(["--enforce-readiness"])
    except Exception as error:
        show_startup_error(
            "Training Failed",
            f"{error}\n\n"
            "Please ensure datasets/custom_mi contains collected task files "
            "(*_mi_epochs.npz/*_gate_epochs.npz/*_artifact_epochs.npz/*_continuous.npz).",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
