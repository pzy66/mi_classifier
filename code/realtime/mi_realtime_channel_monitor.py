"""Realtime 8-channel EEG waveform monitor for channel validation."""

from __future__ import annotations

from datetime import datetime
import sys
import threading

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from PyQt5.QtCore import QObject, QPointF, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


DEFAULT_CHANNEL_NAMES = ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"]
DEFAULT_CHANNEL_POSITIONS = list(range(8))


def parse_channel_names(raw_value: str, expected_count: int = 8) -> list[str]:
    """Parse comma-separated channel names and enforce channel count."""
    parsed = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if len(parsed) != expected_count:
        raise ValueError(
            f"Channel-name count must be exactly {expected_count}, got {len(parsed)}."
        )
    if len(set(parsed)) != len(parsed):
        raise ValueError("Channel names contain duplicates.")
    return parsed


def parse_channel_positions(raw_value: str, expected_count: int = 8) -> list[int]:
    """Parse comma-separated channel positions and enforce channel count."""
    parsed = [int(item.strip()) for item in str(raw_value).split(",") if item.strip()]
    if len(parsed) != expected_count:
        raise ValueError(
            f"Channel-position count must be exactly {expected_count}, got {len(parsed)}."
        )
    if min(parsed) < 0:
        raise ValueError("Channel positions cannot be negative.")
    if len(set(parsed)) != len(parsed):
        raise ValueError("Channel positions contain duplicates.")
    return parsed


def available_board_options() -> list[tuple[str, int]]:
    """Return common BrainFlow board presets."""
    options: list[tuple[str, int]] = []
    candidate_names = [
        "CYTON_BOARD",
        "CYTON_DAISY_BOARD",
        "GANGLION_BOARD",
        "SYNTHETIC_BOARD",
    ]
    for name in candidate_names:
        board = getattr(BoardIds, name, None)
        if board is None:
            continue
        label = name.replace("_BOARD", "").replace("_", " ").title()
        if name == "SYNTHETIC_BOARD":
            label = "Synthetic (Demo)"
        options.append((label, int(board.value)))
    return options


def detect_serial_ports() -> list[str]:
    """Detect available serial ports; fallback to COM1..COM20."""
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


class EightChannelWaveformWidget(QWidget):
    """Stacked waveform renderer for 8-channel realtime EEG."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._channel_names = list(DEFAULT_CHANNEL_NAMES)
        self._data = np.empty((len(self._channel_names), 0), dtype=np.float32)
        self._sampling_rate = 250.0
        self._window_sec = 6.0
        self._scale_uv = 150.0
        self.setMinimumHeight(420)

        self._palette = [
            QColor("#60A5FA"),
            QColor("#34D399"),
            QColor("#F59E0B"),
            QColor("#F472B6"),
            QColor("#22D3EE"),
            QColor("#A78BFA"),
            QColor("#F87171"),
            QColor("#4ADE80"),
        ]

    def set_channel_names(self, channel_names: list[str]) -> None:
        self._channel_names = list(channel_names)
        self.update()

    def set_sampling_rate(self, sampling_rate: float) -> None:
        self._sampling_rate = max(float(sampling_rate), 1e-6)
        self.update()

    def set_view_config(self, *, window_sec: float, scale_uv: float) -> None:
        self._window_sec = max(float(window_sec), 0.5)
        self._scale_uv = max(float(scale_uv), 1.0)
        self.update()

    def set_data(self, eeg_data: np.ndarray) -> None:
        array = np.asarray(eeg_data, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] <= 0:
            return
        self._data = np.ascontiguousarray(array, dtype=np.float32)
        self.update()

    def _make_display_slice(self) -> np.ndarray:
        if self._data.size == 0:
            return self._data
        sample_count = int(round(self._window_sec * self._sampling_rate))
        sample_count = max(8, sample_count)
        return self._data[:, -sample_count:]

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#0B1220"))

        if self._data.size == 0:
            painter.setPen(QColor("#CBD5E1"))
            painter.setFont(QFont("Consolas", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, "Waiting for realtime EEG data ...")
            return

        drawing_rect = self.rect().adjusted(72, 18, -20, -34)
        if drawing_rect.width() <= 10 or drawing_rect.height() <= 10:
            return

        channel_count = min(len(self._channel_names), self._data.shape[0])
        if channel_count <= 0:
            return

        display = self._make_display_slice()[:channel_count, :]
        if display.shape[1] < 2:
            return

        max_points = max(16, int(drawing_rect.width()))
        if display.shape[1] > max_points:
            indices = np.linspace(0, display.shape[1] - 1, num=max_points, dtype=np.int64)
            display = display[:, indices]

        x_values = np.linspace(
            float(drawing_rect.left()),
            float(drawing_rect.right()),
            num=display.shape[1],
            dtype=np.float64,
        )

        lane_height = float(drawing_rect.height()) / float(channel_count)
        amplitude_pixels = lane_height * 0.36

        grid_pen = QPen(QColor("#1E293B"), 1)
        text_pen = QPen(QColor("#E2E8F0"), 1)
        zero_pen = QPen(QColor("#334155"), 1, Qt.DashLine)

        whole_seconds = int(np.floor(self._window_sec))
        painter.setPen(grid_pen)
        for second in range(whole_seconds + 1):
            x = drawing_rect.left() + (float(second) / float(max(self._window_sec, 1e-6))) * drawing_rect.width()
            painter.drawLine(int(round(x)), drawing_rect.top(), int(round(x)), drawing_rect.bottom())

        for channel_index in range(channel_count):
            lane_top = drawing_rect.top() + lane_height * channel_index
            lane_bottom = lane_top + lane_height
            lane_mid = 0.5 * (lane_top + lane_bottom)

            painter.setPen(QPen(QColor("#0F172A"), 1))
            painter.drawLine(
                int(round(drawing_rect.left())),
                int(round(lane_bottom)),
                int(round(drawing_rect.right())),
                int(round(lane_bottom)),
            )

            painter.setPen(zero_pen)
            painter.drawLine(
                int(round(drawing_rect.left())),
                int(round(lane_mid)),
                int(round(drawing_rect.right())),
                int(round(lane_mid)),
            )

            painter.setPen(text_pen)
            painter.setFont(QFont("Consolas", 10))
            label_text = self._channel_names[channel_index]
            painter.drawText(6, int(round(lane_mid + 4)), f"{label_text}")

            normalized = np.clip(
                display[channel_index, :] / float(self._scale_uv),
                -1.15,
                1.15,
            )
            y_values = lane_mid - normalized * amplitude_pixels

            waveform_pen = QPen(self._palette[channel_index % len(self._palette)], 1.4)
            painter.setPen(waveform_pen)
            painter.save()
            painter.setClipRect(
                drawing_rect.left(),
                int(round(lane_top + 1)),
                drawing_rect.width(),
                int(round(max(2.0, lane_height - 2.0))),
            )
            path = QPainterPath(QPointF(float(x_values[0]), float(y_values[0])))
            for x, y in zip(x_values[1:], y_values[1:]):
                path.lineTo(float(x), float(y))
            painter.drawPath(path)
            painter.restore()

        painter.setPen(QColor("#94A3B8"))
        painter.setFont(QFont("Consolas", 9))
        bottom_text = (
            f"Window: {self._window_sec:.1f}s | Scale: +/-{self._scale_uv:.0f} uV "
            f"| Fs: {self._sampling_rate:.1f} Hz"
        )
        painter.drawText(
            drawing_rect.left(),
            self.height() - 10,
            bottom_text,
        )


class ChannelMonitorWorker(QObject):
    """Background BrainFlow worker that streams EEG for plotting."""

    data_ready = pyqtSignal(object)
    connected = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = dict(config)
        self.board: BoardShim | None = None
        self.stop_event = threading.Event()

    @pyqtSlot()
    def run(self) -> None:
        try:
            self.stop_event.clear()
            BoardShim.enable_dev_board_logger()
            try:
                BoardShim.release_all_sessions()
            except Exception:
                pass

            board_id = int(self.config["board_id"])
            serial_port = str(self.config.get("serial_port", "")).strip()
            selected_positions = [int(item) for item in self.config["channel_positions"]]
            channel_names = [str(item) for item in self.config["channel_names"]]
            poll_sec = float(self.config["poll_sec"])
            buffer_sec = float(self.config["buffer_sec"])

            params = BrainFlowInputParams()
            if serial_port:
                params.serial_port = serial_port

            self.board = BoardShim(board_id, params)
            self.board.prepare_session()
            self.board.start_stream(450000)

            sampling_rate = float(BoardShim.get_sampling_rate(board_id))
            eeg_rows = BoardShim.get_eeg_channels(board_id)
            if max(selected_positions) >= len(eeg_rows):
                raise ValueError(
                    f"Board exposes {len(eeg_rows)} EEG rows, but requested positions are {selected_positions}."
                )
            selected_rows = [int(eeg_rows[index]) for index in selected_positions]

            max_buffer_samples = max(16, int(round(buffer_sec * sampling_rate)))
            live_buffer = np.empty((len(selected_rows), 0), dtype=np.float32)
            stream_sample_count = 0

            self.connected.emit(
                {
                    "sampling_rate": sampling_rate,
                    "selected_rows": selected_rows,
                    "channel_names": channel_names,
                }
            )
            self.status_changed.emit(
                "EEG stream started | "
                f"fs={sampling_rate:g} Hz | rows={selected_rows} | channels={channel_names}"
            )

            while not self.stop_event.is_set():
                data = self.board.get_board_data()
                if data.size > 0 and data.shape[1] > 0:
                    eeg_chunk = np.ascontiguousarray(data[selected_rows, :], dtype=np.float32)
                    new_samples = int(eeg_chunk.shape[1])
                    if new_samples > 0:
                        stream_sample_count += new_samples
                        if live_buffer.size == 0:
                            live_buffer = eeg_chunk
                        else:
                            live_buffer = np.concatenate((live_buffer, eeg_chunk), axis=1)
                        if live_buffer.shape[1] > max_buffer_samples:
                            live_buffer = live_buffer[:, -max_buffer_samples:]

                        self.data_ready.emit(
                            {
                                "eeg_data": np.ascontiguousarray(live_buffer, dtype=np.float32),
                                "sampling_rate": sampling_rate,
                                "stream_sample_count": int(stream_sample_count),
                                "new_samples": new_samples,
                            }
                        )

                if self.stop_event.wait(poll_sec):
                    break
        except Exception as error:
            self.error_occurred.emit(f"Acquisition error: {error}")
        finally:
            if self.board is not None:
                try:
                    self.board.stop_stream()
                except Exception:
                    pass
                try:
                    self.board.release_session()
                except Exception:
                    pass
                self.board = None
            self.status_changed.emit("EEG stream stopped.")
            self.finished.emit()

    @pyqtSlot()
    def request_stop(self) -> None:
        self.stop_event.set()


class ChannelMonitorWindow(QMainWindow):
    """Main window for realtime 8-channel waveform validation."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Realtime 8-Channel EEG Monitor")
        self.resize(1320, 860)

        self.worker_thread: QThread | None = None
        self.worker: ChannelMonitorWorker | None = None

        self.waveform = EightChannelWaveformWidget()
        self.status_label = QLabel("Status: Idle")
        self.stream_label = QLabel("Samples: 0")
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(1200)

        self.board_combo = QComboBox()
        for label, board_id in available_board_options():
            self.board_combo.addItem(label, int(board_id))

        self.serial_combo = QComboBox()
        self.serial_combo.setEditable(True)
        self.btn_refresh_ports = QPushButton("Refresh Ports")
        self.btn_refresh_ports.clicked.connect(self.refresh_serial_ports)

        self.channel_names_edit = QLineEdit(",".join(DEFAULT_CHANNEL_NAMES))
        self.channel_positions_edit = QLineEdit(",".join(str(item) for item in DEFAULT_CHANNEL_POSITIONS))

        self.window_sec_spin = QDoubleSpinBox()
        self.window_sec_spin.setRange(1.0, 20.0)
        self.window_sec_spin.setSingleStep(0.5)
        self.window_sec_spin.setValue(6.0)

        self.scale_uv_spin = QDoubleSpinBox()
        self.scale_uv_spin.setRange(20.0, 1000.0)
        self.scale_uv_spin.setSingleStep(10.0)
        self.scale_uv_spin.setValue(150.0)

        self.poll_sec_spin = QDoubleSpinBox()
        self.poll_sec_spin.setRange(0.02, 1.0)
        self.poll_sec_spin.setSingleStep(0.01)
        self.poll_sec_spin.setValue(0.10)

        self.buffer_sec_spin = QDoubleSpinBox()
        self.buffer_sec_spin.setRange(2.0, 60.0)
        self.buffer_sec_spin.setSingleStep(1.0)
        self.buffer_sec_spin.setValue(10.0)

        self.btn_start = QPushButton("Start Monitor")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        self.btn_start.clicked.connect(self.start_monitor)
        self.btn_stop.clicked.connect(self.stop_monitor)
        self.window_sec_spin.valueChanged.connect(self.on_view_config_changed)
        self.scale_uv_spin.valueChanged.connect(self.on_view_config_changed)
        self.board_combo.currentIndexChanged.connect(self.on_board_changed)

        self._init_ui()
        self.refresh_serial_ports()
        self.on_view_config_changed()
        self.on_board_changed()

    def _init_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(10)

        cfg_group = QGroupBox("Acquisition Config")
        cfg_layout = QGridLayout(cfg_group)
        cfg_form_left = QFormLayout()
        cfg_form_right = QFormLayout()

        serial_row = QHBoxLayout()
        serial_row.addWidget(self.serial_combo, stretch=1)
        serial_row.addWidget(self.btn_refresh_ports, stretch=0)

        cfg_form_left.addRow("Board", self.board_combo)
        cfg_form_left.addRow("Serial Port", serial_row)
        cfg_form_left.addRow("Channel Names", self.channel_names_edit)
        cfg_form_left.addRow("Channel Positions", self.channel_positions_edit)

        cfg_form_right.addRow("Display Window (s)", self.window_sec_spin)
        cfg_form_right.addRow("Y Scale (+/-uV)", self.scale_uv_spin)
        cfg_form_right.addRow("Poll Interval (s)", self.poll_sec_spin)
        cfg_form_right.addRow("Worker Buffer (s)", self.buffer_sec_spin)

        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.btn_start)
        buttons_row.addWidget(self.btn_stop)

        status_row = QHBoxLayout()
        status_row.addWidget(self.status_label, stretch=2)
        status_row.addWidget(self.stream_label, stretch=1)

        cfg_layout.addLayout(cfg_form_left, 0, 0)
        cfg_layout.addLayout(cfg_form_right, 0, 1)
        cfg_layout.addLayout(buttons_row, 1, 0, 1, 2)
        cfg_layout.addLayout(status_row, 2, 0, 1, 2)

        main_layout.addWidget(cfg_group)
        main_layout.addWidget(self.waveform, stretch=1)
        main_layout.addWidget(self.log_text, stretch=0)

    def is_running(self) -> bool:
        return self.worker_thread is not None

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{timestamp}] {message}")

    def refresh_serial_ports(self) -> None:
        current_text = self.serial_combo.currentText().strip()
        ports = detect_serial_ports()
        self.serial_combo.clear()
        self.serial_combo.addItems(ports)
        if current_text and current_text not in ports:
            self.serial_combo.addItem(current_text)
        if current_text:
            index = self.serial_combo.findText(current_text)
            if index >= 0:
                self.serial_combo.setCurrentIndex(index)

    def on_board_changed(self) -> None:
        board_id = int(self.board_combo.currentData())
        synthetic_board = getattr(BoardIds, "SYNTHETIC_BOARD", None)
        is_synthetic = synthetic_board is not None and board_id == int(synthetic_board.value)
        self.serial_combo.setEnabled(not is_synthetic)
        self.btn_refresh_ports.setEnabled(not is_synthetic)
        if is_synthetic:
            self.status_label.setText("Status: Synthetic board selected (serial port not required)")
        elif not self.is_running():
            self.status_label.setText("Status: Idle")

    def on_view_config_changed(self) -> None:
        self.waveform.set_view_config(
            window_sec=float(self.window_sec_spin.value()),
            scale_uv=float(self.scale_uv_spin.value()),
        )

    def build_runtime_config(self) -> dict:
        board_id = int(self.board_combo.currentData())
        channel_names = parse_channel_names(self.channel_names_edit.text(), expected_count=8)
        channel_positions = parse_channel_positions(self.channel_positions_edit.text(), expected_count=8)

        synthetic_board = getattr(BoardIds, "SYNTHETIC_BOARD", None)
        is_synthetic = synthetic_board is not None and board_id == int(synthetic_board.value)
        serial_port = self.serial_combo.currentText().strip()
        if not is_synthetic and not serial_port:
            raise ValueError("Serial port cannot be empty for non-synthetic boards.")

        return {
            "board_id": board_id,
            "serial_port": "" if is_synthetic else serial_port,
            "channel_names": channel_names,
            "channel_positions": channel_positions,
            "poll_sec": float(self.poll_sec_spin.value()),
            "buffer_sec": float(self.buffer_sec_spin.value()),
        }

    def start_monitor(self) -> None:
        if self.is_running():
            self.log("Monitor is already running.")
            return

        try:
            config = self.build_runtime_config()
        except Exception as error:
            self.show_error(f"Invalid configuration: {error}")
            return

        self.waveform.set_channel_names(list(config["channel_names"]))
        self.waveform.set_data(np.zeros((8, 8), dtype=np.float32))

        self.worker_thread = QThread(self)
        self.worker = ChannelMonitorWorker(config)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.data_ready.connect(self.on_data_ready, type=Qt.QueuedConnection)
        self.worker.connected.connect(self.on_connected, type=Qt.QueuedConnection)
        self.worker.status_changed.connect(self.log)
        self.worker.error_occurred.connect(self.on_worker_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.on_worker_thread_finished)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_label.setText("Status: Starting stream ...")
        self.log("Starting realtime channel monitor ...")

    def stop_monitor(self) -> None:
        if not self.is_running():
            return
        if self.worker is not None:
            self.worker.request_stop()
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(4000)

    def on_connected(self, info: object) -> None:
        if not isinstance(info, dict):
            return
        sampling_rate = float(info.get("sampling_rate", 0.0))
        selected_rows = info.get("selected_rows", [])
        self.waveform.set_sampling_rate(sampling_rate)
        self.status_label.setText(
            f"Status: Streaming | fs={sampling_rate:g} Hz | rows={selected_rows}"
        )
        self.log(
            f"Connected. Sampling rate={sampling_rate:g} Hz | board rows={selected_rows}"
        )

    def on_data_ready(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        eeg_data = np.asarray(payload.get("eeg_data"), dtype=np.float32)
        sampling_rate = float(payload.get("sampling_rate", 0.0))
        stream_sample_count = int(payload.get("stream_sample_count", eeg_data.shape[1]))
        self.waveform.set_sampling_rate(sampling_rate)
        self.waveform.set_data(eeg_data)
        self.stream_label.setText(
            f"Samples: {stream_sample_count} | Buffer: {int(eeg_data.shape[1])}"
        )

    def on_worker_error(self, message: str) -> None:
        self.log(message)
        self.show_error(message)
        self.stop_monitor()

    def on_worker_thread_finished(self) -> None:
        self.worker = None
        self.worker_thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText("Status: Idle")
        self.stream_label.setText("Samples: 0")
        self.log("Realtime channel monitor stopped.")

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", str(message))

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_monitor()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Realtime 8-Channel EEG Monitor")
    window = ChannelMonitorWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())


