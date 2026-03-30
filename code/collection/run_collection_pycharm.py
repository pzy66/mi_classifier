"""PyCharm launcher for MI collection UI (data collection only)."""

from __future__ import annotations

import sys
import traceback


def show_startup_error(title: str, message: str) -> None:
    """Show a visible startup error even when Qt is unavailable."""
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, title, 0x10)
    except Exception:
        pass
    print(f"{title}\n{message}", file=sys.stderr)


def main() -> int:
    """Launch collection UI."""
    try:
        from PyQt5.QtWidgets import QApplication
        from mi_data_collector import MIDataCollectorWindow
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
        app = QApplication(sys.argv)
        app.setApplicationName("MI Collection Only")
        window = MIDataCollectorWindow()
        window.show()
        return app.exec_()
    except Exception:
        show_startup_error("Startup Failed", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

