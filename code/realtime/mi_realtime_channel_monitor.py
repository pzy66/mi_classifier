"""Realtime 8-channel EEG monitor (optimized for Cyton board id 0)."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import sys
import threading
from typing import Any

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes, NoiseTypes
from PyQt5.QtCore import QObject, QPointF, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


DEFAULT_BOARD_ID = 0
DEFAULT_SERIAL_PORT = "COM4"
DEFAULT_WINDOW_SEC = 5.0
DEFAULT_POLL_SEC = 0.05
DEFAULT_SCALE_UV = 150.0
DEFAULT_CHANNEL_NAMES = [f"CH{i + 1}" for i in range(8)]


def detect_serial_ports() -> list[str]:
    """Detect serial ports, fallback to COM1..COM20."""
    try:
        from serial.tools import list_ports

        devices = sorted(
            {
                str(port.device).strip()
                for port in list_ports.comports()
                if str(port.device).strip()
            }
        )
        if devices:
            return devices
    except Exception:
        pass
    return [f"COM{i}" for i in range(1, 21)]


def board_options() -> list[tuple[str, int]]:
    """Return common board choices with Cyton first."""
    options: list[tuple[str, int]] = [("Cyton (Board ID 0)", 0)]
    synthetic = getattr(BoardIds, "SYNTHETIC_BOARD", None)
    if synthetic is not None and int(synthetic.value) != 0:
        options.append(("Synthetic (Demo)", int(synthetic.value)))

    for name in ("CYTON_DAISY_BOARD", "GANGLION_BOARD"):
        board = getattr(BoardIds, name, None)
        if board is None:
            continue
        options.append((name.replace("_BOARD", "").replace("_", " ").title(), int(board.value)))
    return options


class StackedWaveformWidget(QWidget):
    """Simple stacked waveform renderer with optional auto scaling."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(460)

        self._channel_names = list(DEFAULT_CHANNEL_NAMES)
        self._data = np.empty((len(self._channel_names), 0), dtype=np.float32)
        self._sampling_rate = 250.0
        self._window_sec = DEFAULT_WINDOW_SEC
        self._fixed_scale_uv = DEFAULT_SCALE_UV
        self._auto_scale = True

        self._colors = [
            QColor("#FF6B6B"),
            QColor("#4ECDC4"),
            QColor("#45B7D1"),
            QColor("#96CEB4"),
            QColor("#FFEAA7"),
            QColor("#FF9FF3"),
            QColor("#98D8C8"),
            QColor("#F7DC6F"),
        ]

    def set_channel_names(self, names: list[str]) -> None:
        self._channel_names = [str(item) for item in names]
        self.update()

    def set_sampling_rate(self, sampling_rate: float) -> None:
        self._sampling_rate = max(float(sampling_rate), 1.0)
        self.update()

    def set_view_config(self, window_sec: float, fixed_scale_uv: float, auto_scale: bool) -> None:
        self._window_sec = max(float(window_sec), 1.0)
        self._fixed_scale_uv = max(float(fixed_scale_uv), 10.0)
        self._auto_scale = bool(auto_scale)
        self.update()

    def set_data(self, eeg_data: np.ndarray) -> None:
        arr = np.asarray(eeg_data, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] <= 0:
            return
        self._data = np.ascontiguousarray(arr, dtype=np.float32)
        self.update()

    @staticmethod
    def _auto_center_and_span(signal: np.ndarray) -> tuple[float, float]:
        if signal.size < 20:
            return 0.0, 2.0 * DEFAULT_SCALE_UV
        low = float(np.percentile(signal, 5.0))
        high = float(np.percentile(signal, 95.0))
        span = high - low
        if span < 20.0:
            center = 0.5 * (high + low)
            return center, 20.0
        center = 0.5 * (high + low)
        return center, span * 1.2

    def paintEvent(self, event) -> None:  # noqa: N802
        del event

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#101728"))

        if self._data.size == 0:
            painter.setPen(QColor("#E6EDF7"))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(self.rect(), Qt.AlignCenter, "Waiting for realtime EEG data ...")
            return

        margin_left, margin_right = 76, 16
        margin_top, margin_bottom = 20, 32
        plot_rect = self.rect().adjusted(
            margin_left,
            margin_top,
            -margin_right,
            -margin_bottom,
        )
        if plot_rect.width() < 30 or plot_rect.height() < 30:
            return

        channel_count = min(int(self._data.shape[0]), len(self._channel_names))
        if channel_count <= 0:
            return

        samples_to_show = max(8, int(round(self._window_sec * self._sampling_rate)))
        y_all = self._data[:, -samples_to_show:] if self._data.shape[1] > samples_to_show else self._data
        if y_all.shape[1] < 2:
            return

        max_points = max(24, int(plot_rect.width()))
        if y_all.shape[1] > max_points:
            sample_idx = np.linspace(0, y_all.shape[1] - 1, max_points, dtype=np.int64)
            y_all = y_all[:, sample_idx]

        x_vals = np.linspace(
            float(plot_rect.left()),
            float(plot_rect.right()),
            y_all.shape[1],
            dtype=np.float64,
        )
        channel_height = float(plot_rect.height()) / float(channel_count)

        grid_pen = QPen(QColor("#1F2B44"), 1)
        painter.setPen(grid_pen)
        for row in range(channel_count + 1):
            y_grid = plot_rect.top() + row * channel_height
            painter.drawLine(
                plot_rect.left(),
                int(round(y_grid)),
                plot_rect.right(),
                int(round(y_grid)),
            )

        for channel_index in range(channel_count):
            signal = y_all[channel_index, :]
            y_center = plot_rect.top() + (channel_index + 0.5) * channel_height
            amp_px = channel_height * 0.42

            if self._auto_scale:
                center_uv, span_uv = self._auto_center_and_span(signal)
            else:
                center_uv, span_uv = 0.0, 2.0 * self._fixed_scale_uv
            half_span = max(1.0, 0.5 * span_uv)
            normalized = np.clip((signal - center_uv) / half_span, -1.8, 1.8)
            y_vals = y_center - normalized * amp_px

            line_pen = QPen(self._colors[channel_index % len(self._colors)], 1.4)
            painter.setPen(line_pen)
            path = QPainterPath(QPointF(float(x_vals[0]), float(y_vals[0])))
            for x, y in zip(x_vals[1:], y_vals[1:]):
                path.lineTo(float(x), float(y))
            painter.drawPath(path)

            painter.setPen(QColor("#D7E2F0"))
            painter.setFont(QFont("Segoe UI", 9))
            name = self._channel_names[channel_index]
            painter.drawText(8, int(round(y_center + 4)), name)

        painter.setPen(QColor("#8FA3BF"))
        painter.setFont(QFont("Segoe UI", 8))
        if self._auto_scale:
            scale_text = "Scale: Auto"
        else:
            scale_text = f"Scale: +/-{self._fixed_scale_uv:.0f}uV"
        footer = f"Window: {self._window_sec:.1f}s | {scale_text} | Fs: {self._sampling_rate:.0f}Hz"
        painter.drawText(plot_rect.left(), self.height() - 6, footer)


class EEGWorker(QObject):
    """Worker thread: board connect + incremental data fetch + filtering."""

    data_ready = pyqtSignal(object)
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self._stop_event = threading.Event()
        self._board: BoardShim | None = None

    @staticmethod
    def _select_eeg_rows(board_id: int) -> list[int]:
        eeg_rows = list(BoardShim.get_eeg_channels(board_id))
        if not eeg_rows:
            eeg_rows = list(BoardShim.get_exg_channels(board_id))
        if not eeg_rows:
            raise RuntimeError("Board does not expose EEG/EXG channels.")
        if len(eeg_rows) < 8:
            raise RuntimeError(f"EEG/EXG channels < 8 (got {len(eeg_rows)}).")
        return eeg_rows[:8]

    @staticmethod
    def _filter_window(channel_signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        y = np.asarray(channel_signal, dtype=np.float64).copy()
        if y.size <= 10:
            return y

        DataFilter.detrend(y, DetrendOperations.CONSTANT.value)
        DataFilter.remove_environmental_noise(y, int(sampling_rate), NoiseTypes.FIFTY.value)
        DataFilter.perform_bandpass(
            y,
            int(sampling_rate),
            1.0,
            40.0,
            4,
            FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
            0,
        )
        return y

    @pyqtSlot()
    def run(self) -> None:
        try:
            BoardShim.enable_dev_board_logger()
            try:
                BoardShim.release_all_sessions()
            except Exception:
                pass

            board_id = int(self.config["board_id"])
            serial_port = str(self.config.get("serial_port", "")).strip()
            timeout = int(self.config.get("timeout", 0))
            streamer_params = str(self.config.get("streamer_params", "")).strip()
            window_sec = float(self.config.get("window_sec", DEFAULT_WINDOW_SEC))
            poll_sec = float(self.config.get("poll_sec", DEFAULT_POLL_SEC))
            names = [str(item) for item in self.config.get("channel_names", DEFAULT_CHANNEL_NAMES)]

            params = BrainFlowInputParams()
            params.serial_port = serial_port
            params.timeout = timeout

            self._board = BoardShim(board_id, params)
            self._board.prepare_session()
            self._board.start_stream(450000, streamer_params)

            runtime_board_id = int(self._board.get_board_id())
            sampling_rate = int(BoardShim.get_sampling_rate(runtime_board_id))
            eeg_rows = self._select_eeg_rows(runtime_board_id)
            max_points = max(16, int(round(window_sec * sampling_rate)))
            buffers = [deque(maxlen=max_points) for _ in eeg_rows]

            self.status_update.emit(
                "Connected | "
                f"board_id={runtime_board_id} | serial={serial_port or '-'} | "
                f"fs={sampling_rate}Hz | eeg_rows={eeg_rows}"
            )

            while not self._stop_event.wait(max(0.02, poll_sec)):
                if self._board is None:
                    break

                data = self._board.get_board_data()  # get all new data and clear board buffer
                if data is None or data.size == 0 or int(data.shape[1]) <= 0:
                    continue

                eeg_chunk = np.asarray(data[eeg_rows, :], dtype=np.float64)
                if eeg_chunk.ndim != 2 or eeg_chunk.shape[0] != len(eeg_rows):
                    continue

                sample_count = int(eeg_chunk.shape[1])
                for ch in range(len(eeg_rows)):
                    buffers[ch].extend(float(v) for v in eeg_chunk[ch, :sample_count])

                available_points = len(buffers[0])
                if available_points < 2:
                    continue

                filtered_rows: list[np.ndarray] = []
                for ch in range(len(eeg_rows)):
                    signal = np.fromiter(buffers[ch], dtype=np.float64, count=available_points)
                    try:
                        filtered = self._filter_window(signal, sampling_rate)
                    except Exception:
                        filtered = signal
                    filtered_rows.append(filtered.astype(np.float32, copy=False))

                eeg_window = np.ascontiguousarray(np.vstack(filtered_rows), dtype=np.float32)
                self.data_ready.emit(
                    {
                        "eeg_data": eeg_window,
                        "sampling_rate": float(sampling_rate),
                        "channel_names": names[: len(eeg_rows)],
                        "points": int(eeg_window.shape[1]),
                    }
                )
        except Exception as error:
            self.error_occurred.emit(f"EEG Error: {error}")
        finally:
            if self._board is not None:
                try:
                    self._board.stop_stream()
                except Exception:
                    pass
                try:
                    self._board.release_session()
                except Exception:
                    pass
            self._board = None
            self.finished.emit()

    @pyqtSlot()
    def request_stop(self) -> None:
        self._stop_event.set()


class MonitorWindow(QMainWindow):
    """Main monitor UI."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BrainFlow Cyton Realtime EEG Monitor (05)")
        self.resize(1280, 860)

        self.worker: EEGWorker | None = None
        self.worker_thread: QThread | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        control_group = QGroupBox("Acquisition Control")
        control_form = QFormLayout(control_group)

        self.board_combo = QComboBox()
        for label, board_id in board_options():
            self.board_combo.addItem(label, int(board_id))
        board_index = self.board_combo.findData(DEFAULT_BOARD_ID)
        if board_index >= 0:
            self.board_combo.setCurrentIndex(board_index)

        self.serial_combo = QComboBox()
        self.serial_combo.setEditable(True)
        self.refresh_port_btn = QPushButton("Refresh Ports")
        self.refresh_port_btn.clicked.connect(self.refresh_ports)
        port_row = QHBoxLayout()
        port_row.addWidget(self.serial_combo, stretch=1)
        port_row.addWidget(self.refresh_port_btn, stretch=0)

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(0, 120)
        self.timeout_spin.setValue(0)

        self.streamer_edit = QComboBox()
        self.streamer_edit.setEditable(True)
        self.streamer_edit.addItem("")

        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(2.0, 12.0)
        self.window_spin.setSingleStep(0.5)
        self.window_spin.setValue(DEFAULT_WINDOW_SEC)

        self.poll_spin = QDoubleSpinBox()
        self.poll_spin.setRange(0.02, 1.0)
        self.poll_spin.setSingleStep(0.01)
        self.poll_spin.setValue(DEFAULT_POLL_SEC)

        self.auto_scale_check = QCheckBox("Auto Vertical Scale")
        self.auto_scale_check.setChecked(True)
        self.fixed_scale_spin = QDoubleSpinBox()
        self.fixed_scale_spin.setRange(20.0, 500.0)
        self.fixed_scale_spin.setSingleStep(10.0)
        self.fixed_scale_spin.setValue(DEFAULT_SCALE_UV)
        self.fixed_scale_spin.setEnabled(False)

        scale_row = QHBoxLayout()
        scale_row.addWidget(self.auto_scale_check, stretch=0)
        scale_row.addWidget(QLabel("Fixed Range +/-uV"), stretch=0)
        scale_row.addWidget(self.fixed_scale_spin, stretch=0)
        scale_row.addStretch(1)

        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)

        self.status_label = QLabel("Status: Ready")

        control_form.addRow("Board", self.board_combo)
        control_form.addRow("Serial Port", port_row)
        control_form.addRow("Connect Timeout (s)", self.timeout_spin)
        control_form.addRow("Streamer Params", self.streamer_edit)
        control_form.addRow("Display Window (s)", self.window_spin)
        control_form.addRow("Poll Interval (s)", self.poll_spin)
        control_form.addRow("Display Scale", scale_row)
        control_form.addRow("", btn_row)
        control_form.addRow("Runtime Status", self.status_label)

        self.waveform = StackedWaveformWidget()

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(300)
        self.log_box.setFixedHeight(150)

        root_layout.addWidget(control_group, stretch=0)
        root_layout.addWidget(self.waveform, stretch=1)
        root_layout.addWidget(self.log_box, stretch=0)

        self.start_btn.clicked.connect(self.start_monitoring)
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.auto_scale_check.toggled.connect(self.on_scale_mode_changed)
        self.window_spin.valueChanged.connect(self.apply_view_config)
        self.fixed_scale_spin.valueChanged.connect(self.apply_view_config)
        self.board_combo.currentIndexChanged.connect(self.on_board_changed)

        self.refresh_ports()
        self.on_board_changed()
        self.apply_view_config()

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{timestamp}] {message}")

    def on_scale_mode_changed(self, checked: bool) -> None:
        self.fixed_scale_spin.setEnabled(not checked)
        self.apply_view_config()

    def apply_view_config(self) -> None:
        self.waveform.set_view_config(
            self.window_spin.value(),
            self.fixed_scale_spin.value(),
            self.auto_scale_check.isChecked(),
        )

    def refresh_ports(self) -> None:
        current = self.serial_combo.currentText().strip()
        ports = detect_serial_ports()

        self.serial_combo.clear()
        self.serial_combo.addItems(ports)

        if current and current not in ports:
            self.serial_combo.addItem(current)

        if current:
            idx = self.serial_combo.findText(current)
            if idx >= 0:
                self.serial_combo.setCurrentIndex(idx)
                return

        preferred = self.serial_combo.findText(DEFAULT_SERIAL_PORT)
        if preferred >= 0:
            self.serial_combo.setCurrentIndex(preferred)
        elif self.serial_combo.count() > 0:
            self.serial_combo.setCurrentIndex(0)

    def on_board_changed(self) -> None:
        board_id = int(self.board_combo.currentData())
        synthetic = getattr(BoardIds, "SYNTHETIC_BOARD", None)
        is_synthetic = synthetic is not None and board_id == int(synthetic.value)
        self.serial_combo.setEnabled(not is_synthetic)
        self.refresh_port_btn.setEnabled(not is_synthetic)

        if is_synthetic:
            self.status_label.setText("Status: Synthetic mode selected (no serial needed)")
        elif self.worker_thread is None:
            self.status_label.setText("Status: Ready")

    def build_config(self) -> dict[str, Any]:
        board_id = int(self.board_combo.currentData())
        synthetic = getattr(BoardIds, "SYNTHETIC_BOARD", None)
        is_synthetic = synthetic is not None and board_id == int(synthetic.value)

        serial_port = "" if is_synthetic else self.serial_combo.currentText().strip()
        if not is_synthetic and not serial_port:
            raise ValueError("Physical board requires a serial port, for example COM4.")

        return {
            "board_id": board_id,
            "serial_port": serial_port,
            "timeout": int(self.timeout_spin.value()),
            "streamer_params": self.streamer_edit.currentText().strip(),
            "window_sec": float(self.window_spin.value()),
            "poll_sec": float(self.poll_spin.value()),
            "channel_names": list(DEFAULT_CHANNEL_NAMES),
        }

    def start_monitoring(self) -> None:
        if self.worker_thread is not None:
            self.log("Acquisition is already running.")
            return

        try:
            config = self.build_config()
        except Exception as error:
            self.show_error(str(error))
            return

        self.waveform.set_channel_names(list(DEFAULT_CHANNEL_NAMES))
        self.waveform.set_data(np.zeros((8, 16), dtype=np.float32))
        self.apply_view_config()

        self.worker_thread = QThread(self)
        self.worker = EEGWorker(config)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.data_ready.connect(self.on_data_ready, type=Qt.QueuedConnection)
        self.worker.status_update.connect(self.on_worker_status)
        self.worker.error_occurred.connect(self.on_worker_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.on_worker_finished)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Status: Connecting ...")
        self.log(
            f"Starting acquisition: board_id={config['board_id']}, serial={config['serial_port'] or '-'}"
        )

    def stop_monitoring(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(3000)

    def on_worker_status(self, message: str) -> None:
        self.log(message)
        self.status_label.setText("Status: Streaming")

    def on_worker_error(self, message: str) -> None:
        self.log(message)
        self.show_error(message)
        self.stop_monitoring()

    def on_worker_finished(self) -> None:
        self.worker = None
        self.worker_thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Status: Stopped")
        self.log("Acquisition stopped and resources released.")

    def on_data_ready(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        data = np.asarray(payload.get("eeg_data"), dtype=np.float32)
        if data.ndim != 2 or data.shape[0] <= 0:
            return

        sampling_rate = float(payload.get("sampling_rate", 0.0))
        names = payload.get("channel_names")
        if isinstance(names, list) and names:
            self.waveform.set_channel_names([str(item) for item in names])

        self.waveform.set_sampling_rate(sampling_rate)
        self.waveform.set_data(data)
        self.status_label.setText(
            f"Status: Streaming | Fs={sampling_rate:.0f}Hz | Points={int(data.shape[1])}"
        )

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "EEG Error", message)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_monitoring()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Realtime Channel Monitor 05")
    window = MonitorWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
