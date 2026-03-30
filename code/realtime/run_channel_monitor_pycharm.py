"""PyCharm launcher for realtime 8-channel EEG waveform monitor."""

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
    """Launch realtime channel monitor UI."""
    try:
        from mi_realtime_channel_monitor import main as monitor_main
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
        return int(monitor_main())
    except Exception:
        show_startup_error("Monitor Startup Failed", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
