"""Realtime display test variant using FBCCA-style preprocessing for visualization only.

This script keeps raw board data untouched and only changes waveform display filtering.
"""

from __future__ import annotations

import numpy as np

from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes

from test import CytonRealtimeDualMode


class CytonRealtimeDualModeFBCCAPreproc(CytonRealtimeDualMode):
    """Display-only preprocessing variant based on FBCCA_new.py."""

    DISPLAY_BASELINE_HZ = 3.0
    DISPLAY_BASELINE_ORDER = 1

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

