"""Realtime display test variant using FBCCA-style preprocessing for visualization only.

This script keeps raw board data untouched and only changes waveform display filtering.
"""

from __future__ import annotations

import numpy as np
import time

from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes

from test import CytonRealtimeDualMode


class CytonRealtimeDualModeFBCCAPreproc(CytonRealtimeDualMode):
    """Display-only preprocessing variant based on FBCCA_new.py."""

    DISPLAY_BASELINE_HZ = 3.0
    DISPLAY_BASELINE_ORDER = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track which channel is currently enabled for impedance lead-off test on hardware.
        self._hw_impedance_channel = None

    def _normalize_target_channel(self, target_channel):
        if not self.eeg_channels:
            return int(target_channel or 1)
        return int(np.clip(int(target_channel or self.impedance_channel), 1, len(self.eeg_channels)))

    def _fast_switch_impedance_channel(self, target_channel: int) -> bool:
        """Switch impedance channel with minimum board commands while already in IMP mode."""
        if self.board is None:
            return False

        prev = self._hw_impedance_channel
        if prev is None:
            return False
        if int(prev) == int(target_channel):
            return True

        ok_disable = self.send_board_command(self.build_impedance_command(int(prev), False, False))
        ok_enable = self.send_board_command(self.build_impedance_command(int(target_channel), True, False))
        if ok_disable and ok_enable:
            self._hw_impedance_channel = int(target_channel)
            return True

        # Any partial failure falls back to full reset path in caller.
        print(
            f"warn: impedance fast-switch failed (disable CH{prev}: {ok_disable}, "
            f"enable CH{target_channel}: {ok_enable}), fallback to full switch"
        )
        return False

    def apply_mode_change_safely(self, target_mode=None, target_channel=None, reset_default=False):
        """Faster impedance switching to reduce UI stutter in this FBCCA display variant."""
        t0 = time.perf_counter()
        stopped_for_switch = False
        target = self._normalize_target_channel(target_channel)

        try:
            # Fast path: already in impedance mode and only changing channel.
            if target_mode == self.MODE_IMP and self.mode == self.MODE_IMP:
                if self._fast_switch_impedance_channel(target):
                    self.impedance_channel = target
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    print(f"impedance channel fast-switch -> CH{target} ({dt_ms:.0f} ms)")
                    return

            # Full path for mode transitions / fallback.
            self.stop_stream()
            stopped_for_switch = True

            if reset_default:
                self.send_board_command("d")
                self._hw_impedance_channel = None

            if target_mode == self.MODE_IMP:
                # Enter impedance mode with minimum commands:
                # reset to board defaults, then enable only target channel.
                if not reset_default:
                    self.send_board_command("d")
                self.send_board_command(self.build_impedance_command(target, True, False))
                self.mode = self.MODE_IMP
                self.impedance_channel = target
                self._hw_impedance_channel = target

            elif target_mode == self.MODE_EEG:
                if not reset_default:
                    self.send_board_command("d")
                self.mode = self.MODE_EEG
                self._hw_impedance_channel = None

            self.clear_buffers()
            self.last_impedance_ohms = [None] * len(self.eeg_channels)

        finally:
            if stopped_for_switch:
                self.start_stream()
            self.update_title_and_status()
            self.update_impedance_texts()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if target_mode == self.MODE_IMP:
                print(f"switch to impedance complete (CH{target}, {dt_ms:.0f} ms)")
            elif target_mode == self.MODE_EEG:
                print(f"switch to EEG complete ({dt_ms:.0f} ms)")

    def update_title_and_status(self):
        """Keep parent behavior, then override EEG status text for clarity."""
        super().update_title_and_status()
        if self.fig is None or self.status_text is None:
            return
        if self.mode != self.MODE_EEG:
            return

        self.fig.suptitle("OpenBCI Cyton 8-channel realtime EEG (FBCCA-style display preproc)", fontsize=14)
        self.status_text.set_text(
            f"mode: EEG | port: {self.com_port} | "
            "display preproc: (raw - 3Hz LP baseline) + 50Hz notch + mean removal | "
            "raw board stream remains unchanged"
        )
        self.fig.canvas.draw_idle()

    def get_plot_data(self):
        """Display path only: baseline removal by subtraction, notch, then per-channel de-mean."""
        out = []

        for ch in range(len(self.buffers)):
            y = np.array(self.buffers[ch], dtype=np.float64)

            if y.size <= 10:
                out.append(y)
                continue

            if self.mode == self.MODE_EEG:
                y_plot = y.copy()

                baseline = y_plot.copy()
                try:
                    DataFilter.perform_lowpass(
                        baseline,
                        self.sampling_rate,
                        float(self.DISPLAY_BASELINE_HZ),
                        int(self.DISPLAY_BASELINE_ORDER),
                        FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
                        0,
                    )
                    y_plot = y_plot - baseline
                except Exception:
                    pass

                try:
                    DataFilter.remove_environmental_noise(
                        y_plot,
                        self.sampling_rate,
                        NoiseTypes.FIFTY.value,
                    )
                except Exception:
                    pass

                y_plot = y_plot - np.mean(y_plot)
                out.append(y_plot)
            else:
                out.append(y)

        return out


if __name__ == "__main__":
    plotter = CytonRealtimeDualModeFBCCAPreproc(
        board_id=0,
        com_port="COM3",
        window_seconds=5,
        impedance_channel=1,
        lead_off_current_amps=6e-9,
        series_resistor_ohms=2200.0,
    )
    plotter.run()
