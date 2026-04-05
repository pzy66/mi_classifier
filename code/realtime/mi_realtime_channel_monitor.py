"""Realtime 8-channel EEG monitor with impedance mode for Cyton."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import sys
import threading
from typing import Any
from pathlib import Path

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes, NoiseTypes
from PyQt5.QtCore import QObject, QPointF, QThread, Qt, pyqtSignal
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_ROOT = PROJECT_ROOT / "code" / "shared"
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

from src.serial_ports import detect_serial_ports  # noqa: E402


DEFAULT_BOARD_ID = 0
DEFAULT_SERIAL_PORT = "COM4"
DEFAULT_WINDOW_SEC = 3.0
DEFAULT_POLL_SEC = 0.08
DEFAULT_SCALE_UV = 150.0
DEFAULT_CHANNEL_NAMES = [f"CH{i + 1}" for i in range(8)]
IMPEDANCE_SUPPORTED_BOARD_IDS = {0, 2}


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
    """Stacked waveform renderer with mode-aware styling."""

    MODE_EEG = "EEG"
    MODE_IMPEDANCE = "IMPEDANCE"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(480)

        self._channel_names = list(DEFAULT_CHANNEL_NAMES)
        self._data = np.empty((len(self._channel_names), 0), dtype=np.float32)
        self._sampling_rate = 250.0
        self._window_sec = DEFAULT_WINDOW_SEC
        self._fixed_scale_uv = DEFAULT_SCALE_UV
        self._auto_scale = True
        self._mode = self.MODE_EEG
        self._impedance_channel = 1

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

    def set_runtime_state(self, mode: str, impedance_channel: int) -> None:
        self._mode = str(mode).upper()
        self._impedance_channel = max(1, int(impedance_channel))
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

        impedance_mode = self._mode == self.MODE_IMPEDANCE
        for channel_index in range(channel_count):
            ch_id = channel_index + 1
            signal = y_all[channel_index, :]
            lane_top = plot_rect.top() + channel_index * channel_height
            y_center = lane_top + 0.5 * channel_height
            amp_px = channel_height * 0.42

            highlighted = impedance_mode and ch_id == self._impedance_channel
            if highlighted:
                lane_color = QColor(255, 203, 107, 26)
                painter.fillRect(
                    plot_rect.left(),
                    int(round(lane_top + 1)),
                    plot_rect.width(),
                    max(1, int(round(channel_height - 2))),
                    lane_color,
                )

            if self._auto_scale:
                center_uv, span_uv = self._auto_center_and_span(signal)
            else:
                center_uv, span_uv = 0.0, 2.0 * self._fixed_scale_uv
            half_span = max(1.0, 0.5 * span_uv)
            normalized = np.clip((signal - center_uv) / half_span, -1.8, 1.8)
            y_vals = y_center - normalized * amp_px

            color = QColor(self._colors[channel_index % len(self._colors)])
            if impedance_mode and not highlighted:
                color.setAlpha(96)
            line_width = 1.8 if highlighted else 1.2
            painter.setPen(QPen(color, line_width))
            path = QPainterPath(QPointF(float(x_vals[0]), float(y_vals[0])))
            for x, y in zip(x_vals[1:], y_vals[1:]):
                path.lineTo(float(x), float(y))
            painter.drawPath(path)

            label_color = QColor("#FFF3DD") if highlighted else QColor("#D7E2F0")
            painter.setPen(label_color)
            painter.setFont(QFont("Segoe UI", 9))
            name = self._channel_names[channel_index]
            painter.drawText(8, int(round(y_center + 4)), name)

        painter.setPen(QColor("#8FA3BF"))
        painter.setFont(QFont("Segoe UI", 8))
        if self._auto_scale:
            scale_text = "Scale: Auto"
        else:
            scale_text = f"Scale: +/-{self._fixed_scale_uv:.0f}uV"
        mode_text = (
            f"Mode: {self.MODE_IMPEDANCE} (CH{self._impedance_channel})"
            if impedance_mode
            else f"Mode: {self.MODE_EEG}"
        )
        footer = (
            f"Window: {self._window_sec:.1f}s | {scale_text} | "
            f"Fs: {self._sampling_rate:.0f}Hz | {mode_text}"
        )
        painter.drawText(plot_rect.left(), self.height() - 6, footer)


class EEGWorker(QObject):
    """Worker thread: board connect + mode switching + filtering."""

    MODE_EEG = "EEG"
    MODE_IMPEDANCE = "IMPEDANCE"

    data_ready = pyqtSignal(object)
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self._stop_event = threading.Event()
        self._board: BoardShim | None = None

        self._runtime_board_id = int(self.config.get("board_id", DEFAULT_BOARD_ID))
        self._sampling_rate = 250
        self._eeg_rows: list[int] = []
        self._buffers: list[deque[float]] = []
        self._streaming = False
        self._streamer_params = ""
        self._channel_names = list(DEFAULT_CHANNEL_NAMES)
        self._mode = self.MODE_EEG
        self._impedance_channel = 1
        self._impedance_supported = False

        self._control_lock = threading.Lock()
        self._control_queue: deque[tuple[str, Any]] = deque()

    def enqueue_command(self, command: str, payload: Any = None) -> None:
        """Thread-safe command queue for UI -> worker control."""
        with self._control_lock:
            self._control_queue.append((str(command), payload))

    def _pop_control_commands(self) -> list[tuple[str, Any]]:
        with self._control_lock:
            if not self._control_queue:
                return []
            commands = list(self._control_queue)
            self._control_queue.clear()
            return commands

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
    def _filter_window(channel_signal: np.ndarray, sampling_rate: int, mode: str) -> np.ndarray:
        y = np.asarray(channel_signal, dtype=np.float64).copy()
        if y.size <= 10:
            return y

        if mode == EEGWorker.MODE_EEG:
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
        # In impedance mode we keep lead-off test waveform as-is,
        # so we intentionally skip the EEG filter chain.
        return y

    def _start_stream(self) -> None:
        if self._board is None or self._streaming:
            return
        self._board.start_stream(450000, self._streamer_params)
        self._streaming = True

    def _stop_stream(self) -> None:
        if self._board is None or not self._streaming:
            return
        self._board.stop_stream()
        self._streaming = False

    def _clear_buffers(self) -> None:
        for buf in self._buffers:
            buf.clear()

    def _build_impedance_command(self, channel: int, test_p: bool, test_n: bool) -> str:
        p = 1 if test_p else 0
        n = 1 if test_n else 0
        return f"z{int(channel)}{p}{n}Z"

    def _safe_reconfigure(
        self,
        commands: list[str] | str,
        clear_buffers: bool = True,
        restart_stream: bool = True,
    ) -> None:
        if self._board is None:
            return
        command_list = [commands] if isinstance(commands, str) else list(commands)
        was_streaming = self._streaming
        try:
            if was_streaming:
                self._stop_stream()
            for cmd in command_list:
                response = self._board.config_board(str(cmd))
                self.status_update.emit(f"Board cmd {cmd!r} -> {response}")
            if clear_buffers:
                self._clear_buffers()
        finally:
            if restart_stream and was_streaming and not self._stop_event.is_set():
                self._start_stream()

    def _disable_all_impedance_tests(self) -> None:
        if not self._impedance_supported:
            return
        cmds = [
            self._build_impedance_command(ch, False, False)
            for ch in range(1, len(self._eeg_rows) + 1)
        ]
        self._safe_reconfigure(cmds, clear_buffers=True, restart_stream=True)

    def _enable_impedance_for_channel(self, channel: int) -> None:
        if not self._impedance_supported:
            return
        clamped = int(np.clip(channel, 1, len(self._eeg_rows)))
        cmds = [
            self._build_impedance_command(ch, False, False)
            for ch in range(1, len(self._eeg_rows) + 1)
        ]
        cmds.append(self._build_impedance_command(clamped, True, False))
        self._safe_reconfigure(cmds, clear_buffers=True, restart_stream=True)

    def _reset_default_channel_settings(self) -> None:
        if not self._impedance_supported:
            return
        cmds = [
            self._build_impedance_command(ch, False, False)
            for ch in range(1, len(self._eeg_rows) + 1)
        ]
        cmds.append("d")
        self._safe_reconfigure(cmds, clear_buffers=True, restart_stream=True)

    def _emit_mode_status(self) -> None:
        if self._mode == self.MODE_EEG:
            self.status_update.emit("Mode=EEG | filter=detrend + 50Hz notch + 1-40Hz bandpass")
        else:
            self.status_update.emit(
                f"Mode=IMPEDANCE | testing CH{self._impedance_channel} | waveform only (not final kOhm)"
            )

    def _switch_to_eeg_mode(self) -> None:
        if self._mode == self.MODE_EEG:
            return
        self._mode = self.MODE_EEG
        if self._impedance_supported:
            self._reset_default_channel_settings()
        self._emit_mode_status()

    def _switch_to_impedance_mode(self) -> None:
        if not self._impedance_supported:
            self.status_update.emit(
                f"Impedance mode is not supported on board_id={self._runtime_board_id}."
            )
            return
        self._mode = self.MODE_IMPEDANCE
        self._enable_impedance_for_channel(self._impedance_channel)
        self._emit_mode_status()

    def _set_impedance_channel(self, channel: int, apply_when_impedance: bool = True) -> None:
        if not self._eeg_rows:
            return
        self._impedance_channel = int(np.clip(channel, 1, len(self._eeg_rows)))
        self.status_update.emit(f"Impedance channel -> CH{self._impedance_channel}")
        if self._mode == self.MODE_IMPEDANCE and apply_when_impedance and self._impedance_supported:
            self._enable_impedance_for_channel(self._impedance_channel)
            self._emit_mode_status()

    def _handle_control_commands(self) -> None:
        commands = self._pop_control_commands()
        if not commands:
            return
        for command, payload in commands:
            try:
                if command == "mode":
                    target_mode = str(payload).upper()
                    if target_mode == self.MODE_EEG:
                        self._switch_to_eeg_mode()
                    elif target_mode == self.MODE_IMPEDANCE:
                        self._switch_to_impedance_mode()
                elif command == "set_impedance_channel":
                    self._set_impedance_channel(int(payload), apply_when_impedance=True)
                elif command == "prev_impedance_channel":
                    self._set_impedance_channel(self._impedance_channel - 1, apply_when_impedance=True)
                elif command == "next_impedance_channel":
                    self._set_impedance_channel(self._impedance_channel + 1, apply_when_impedance=True)
                elif command == "reset_board":
                    self._mode = self.MODE_EEG
                    if self._impedance_supported:
                        self._reset_default_channel_settings()
                    self._emit_mode_status()
            except Exception as error:
                self.status_update.emit(f"Control command '{command}' failed: {error}")

    def _emit_payload(self, eeg_window: np.ndarray) -> None:
        self.data_ready.emit(
            {
                "eeg_data": eeg_window,
                "sampling_rate": float(self._sampling_rate),
                "channel_names": list(self._channel_names[: eeg_window.shape[0]]),
                "points": int(eeg_window.shape[1]),
                "mode": self._mode,
                "impedance_channel": int(self._impedance_channel),
                "impedance_supported": bool(self._impedance_supported),
            }
        )

    def request_stop(self) -> None:
        self._stop_event.set()

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
            self._streamer_params = str(self.config.get("streamer_params", "")).strip()
            window_sec = float(self.config.get("window_sec", DEFAULT_WINDOW_SEC))
            poll_sec = float(self.config.get("poll_sec", DEFAULT_POLL_SEC))
            self._channel_names = [str(item) for item in self.config.get("channel_names", DEFAULT_CHANNEL_NAMES)]
            self._impedance_channel = int(self.config.get("impedance_channel", 1))
            self._mode = self.MODE_EEG

            params = BrainFlowInputParams()
            params.serial_port = serial_port
            params.timeout = timeout

            self._board = BoardShim(board_id, params)
            self._board.prepare_session()
            self._start_stream()

            self._runtime_board_id = int(self._board.get_board_id())
            self._sampling_rate = int(BoardShim.get_sampling_rate(self._runtime_board_id))
            self._eeg_rows = self._select_eeg_rows(self._runtime_board_id)
            self._impedance_supported = self._runtime_board_id in IMPEDANCE_SUPPORTED_BOARD_IDS
            self._impedance_channel = int(np.clip(self._impedance_channel, 1, len(self._eeg_rows)))

            if len(self._channel_names) < len(self._eeg_rows):
                self._channel_names.extend(
                    [f"CH{i + 1}" for i in range(len(self._channel_names), len(self._eeg_rows))]
                )
            else:
                self._channel_names = self._channel_names[: len(self._eeg_rows)]

            max_points = max(16, int(round(window_sec * self._sampling_rate)))
            self._buffers = [deque(maxlen=max_points) for _ in self._eeg_rows]

            self.status_update.emit(
                "Connected | "
                f"board_id={self._runtime_board_id} | serial={serial_port or '-'} | "
                f"fs={self._sampling_rate}Hz | eeg_rows={self._eeg_rows} | "
                f"impedance_supported={self._impedance_supported}"
            )
            self._emit_mode_status()

            while not self._stop_event.wait(max(0.02, poll_sec)):
                if self._board is None:
                    break

                self._handle_control_commands()

                data = self._board.get_board_data()
                if data is None or data.size == 0 or int(data.shape[1]) <= 0:
                    continue

                eeg_chunk = np.asarray(data[self._eeg_rows, :], dtype=np.float64)
                if eeg_chunk.ndim != 2 or eeg_chunk.shape[0] != len(self._eeg_rows):
                    continue

                sample_count = int(eeg_chunk.shape[1])
                for ch in range(len(self._eeg_rows)):
                    # Batch extend is faster than per-sample append loops.
                    self._buffers[ch].extend(eeg_chunk[ch, :sample_count].tolist())

                available_points = len(self._buffers[0])
                if available_points < 2:
                    continue

                filtered_rows: list[np.ndarray] = []
                for ch in range(len(self._eeg_rows)):
                    signal = np.fromiter(self._buffers[ch], dtype=np.float64, count=available_points)
                    try:
                        filtered = self._filter_window(signal, self._sampling_rate, self._mode)
                    except Exception:
                        filtered = signal
                    filtered_rows.append(filtered.astype(np.float32, copy=False))

                eeg_window = np.ascontiguousarray(np.vstack(filtered_rows), dtype=np.float32)
                self._emit_payload(eeg_window)
        except Exception as error:
            self.error_occurred.emit(f"EEG Error: {error}")
        finally:
            if self._board is not None:
                try:
                    if self._impedance_supported:
                        self._disable_all_impedance_tests()
                except Exception:
                    pass
                try:
                    self._stop_stream()
                except Exception:
                    pass
                try:
                    self._board.release_session()
                except Exception:
                    pass
            self._board = None
            self._streaming = False
            self.finished.emit()

class MonitorWindow(QMainWindow):
    """Main monitor UI."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BrainFlow Cyton Realtime EEG Monitor (05)")
        self.resize(1320, 900)

        self.worker: EEGWorker | None = None
        self.worker_thread: QThread | None = None
        self.current_mode = EEGWorker.MODE_EEG
        self.current_impedance_channel = 1
        self.current_impedance_supported = False

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
        transport_row = QHBoxLayout()
        transport_row.addWidget(self.start_btn)
        transport_row.addWidget(self.stop_btn)

        self.mode_eeg_btn = QPushButton("EEG Mode (E)")
        self.mode_imp_btn = QPushButton("Impedance Mode (I)")
        self.prev_ch_btn = QPushButton("Prev CH (<-)")
        self.next_ch_btn = QPushButton("Next CH (->)")
        self.reset_board_btn = QPushButton("Reset Board (R)")
        self.impedance_ch_spin = QSpinBox()
        self.impedance_ch_spin.setRange(1, 8)
        self.impedance_ch_spin.setValue(1)
        self.impedance_ch_spin.setSingleStep(1)
        self.impedance_ch_spin.setMinimumWidth(70)

        mode_row = QHBoxLayout()
        mode_row.addWidget(self.mode_eeg_btn)
        mode_row.addWidget(self.mode_imp_btn)
        mode_row.addWidget(self.prev_ch_btn)
        mode_row.addWidget(self.next_ch_btn)
        mode_row.addWidget(QLabel("Impedance CH"))
        mode_row.addWidget(self.impedance_ch_spin)
        mode_row.addWidget(self.reset_board_btn)
        mode_row.addStretch(1)

        self.status_label = QLabel("Status: Ready")
        self.mode_label = QLabel("Mode: EEG")

        control_form.addRow("Board", self.board_combo)
        control_form.addRow("Serial Port", port_row)
        control_form.addRow("Connect Timeout (s)", self.timeout_spin)
        control_form.addRow("Streamer Params", self.streamer_edit)
        control_form.addRow("Display Window (s)", self.window_spin)
        control_form.addRow("Poll Interval (s)", self.poll_spin)
        control_form.addRow("Display Scale", scale_row)
        control_form.addRow("Transport", transport_row)
        control_form.addRow("Mode Control", mode_row)
        control_form.addRow("Runtime Status", self.status_label)
        control_form.addRow("Mode Status", self.mode_label)

        self.waveform = StackedWaveformWidget()

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(500)
        self.log_box.setFixedHeight(180)

        root_layout.addWidget(control_group, stretch=0)
        root_layout.addWidget(self.waveform, stretch=1)
        root_layout.addWidget(self.log_box, stretch=0)

        self.start_btn.clicked.connect(self.start_monitoring)
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.auto_scale_check.toggled.connect(self.on_scale_mode_changed)
        self.window_spin.valueChanged.connect(self.apply_view_config)
        self.fixed_scale_spin.valueChanged.connect(self.apply_view_config)
        self.board_combo.currentIndexChanged.connect(self.on_board_changed)

        self.mode_eeg_btn.clicked.connect(self.switch_to_eeg_mode)
        self.mode_imp_btn.clicked.connect(self.switch_to_impedance_mode)
        self.prev_ch_btn.clicked.connect(self.prev_impedance_channel)
        self.next_ch_btn.clicked.connect(self.next_impedance_channel)
        self.reset_board_btn.clicked.connect(self.reset_board_settings)
        self.impedance_ch_spin.valueChanged.connect(self.on_impedance_channel_spin_changed)

        self.refresh_ports()
        self.on_board_changed()
        self.apply_view_config()
        self._update_mode_ui()

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{timestamp}] {message}")

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key_E:
            self.switch_to_eeg_mode()
            event.accept()
            return
        if key == Qt.Key_I:
            self.switch_to_impedance_mode()
            event.accept()
            return
        if key == Qt.Key_Left:
            self.prev_impedance_channel()
            event.accept()
            return
        if key == Qt.Key_Right:
            self.next_impedance_channel()
            event.accept()
            return
        if key == Qt.Key_R:
            self.reset_board_settings()
            event.accept()
            return
        super().keyPressEvent(event)

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

        cyton_like = board_id in IMPEDANCE_SUPPORTED_BOARD_IDS
        self.mode_imp_btn.setEnabled(cyton_like)
        self.prev_ch_btn.setEnabled(cyton_like)
        self.next_ch_btn.setEnabled(cyton_like)
        self.reset_board_btn.setEnabled(cyton_like)
        self.impedance_ch_spin.setEnabled(cyton_like)

        if is_synthetic:
            self.status_label.setText("Status: Synthetic mode selected (no serial needed)")
        elif self.worker_thread is None:
            self.status_label.setText("Status: Ready")

    def _set_impedance_channel_spin(self, channel: int) -> None:
        self.impedance_ch_spin.blockSignals(True)
        self.impedance_ch_spin.setValue(int(channel))
        self.impedance_ch_spin.blockSignals(False)

    def _update_mode_ui(self) -> None:
        mode_text = f"Mode: {self.current_mode}"
        if self.current_mode == EEGWorker.MODE_IMPEDANCE:
            mode_text += f" | CH{self.current_impedance_channel}"
            if not self.current_impedance_supported:
                mode_text += " | unsupported on this board"
        self.mode_label.setText(mode_text)
        self.waveform.set_runtime_state(self.current_mode, self.current_impedance_channel)
        self._set_impedance_channel_spin(self.current_impedance_channel)

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
            "impedance_channel": int(self.impedance_ch_spin.value()),
        }

    def _send_worker_command(self, command: str, payload: Any = None) -> None:
        if self.worker is None:
            self.log("Worker is not running.")
            return
        self.worker.enqueue_command(command, payload)

    def start_monitoring(self) -> None:
        if self.worker_thread is not None:
            self.log("Acquisition is already running.")
            return

        try:
            config = self.build_config()
        except Exception as error:
            self.show_error(str(error))
            return

        self.current_mode = EEGWorker.MODE_EEG
        self.current_impedance_channel = int(config["impedance_channel"])
        self.current_impedance_supported = int(config["board_id"]) in IMPEDANCE_SUPPORTED_BOARD_IDS
        self._update_mode_ui()

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

    def switch_to_eeg_mode(self) -> None:
        self.current_mode = EEGWorker.MODE_EEG
        self._update_mode_ui()
        self._send_worker_command("mode", EEGWorker.MODE_EEG)

    def switch_to_impedance_mode(self) -> None:
        self.current_mode = EEGWorker.MODE_IMPEDANCE
        self._update_mode_ui()
        self._send_worker_command("mode", EEGWorker.MODE_IMPEDANCE)

    def prev_impedance_channel(self) -> None:
        value = self.impedance_ch_spin.value() - 1
        if value < self.impedance_ch_spin.minimum():
            value = self.impedance_ch_spin.maximum()
        self.impedance_ch_spin.setValue(value)

    def next_impedance_channel(self) -> None:
        value = self.impedance_ch_spin.value() + 1
        if value > self.impedance_ch_spin.maximum():
            value = self.impedance_ch_spin.minimum()
        self.impedance_ch_spin.setValue(value)

    def reset_board_settings(self) -> None:
        self.current_mode = EEGWorker.MODE_EEG
        self._update_mode_ui()
        self._send_worker_command("reset_board")

    def on_impedance_channel_spin_changed(self, value: int) -> None:
        self.current_impedance_channel = int(value)
        self._update_mode_ui()
        self._send_worker_command("set_impedance_channel", int(value))

    def on_worker_status(self, message: str) -> None:
        self.log(message)

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
        mode = str(payload.get("mode", EEGWorker.MODE_EEG)).upper()
        impedance_channel = int(payload.get("impedance_channel", 1))
        impedance_supported = bool(payload.get("impedance_supported", False))

        if isinstance(names, list) and names:
            self.waveform.set_channel_names([str(item) for item in names])
            self.impedance_ch_spin.setRange(1, len(names))

        self.current_mode = mode
        self.current_impedance_channel = impedance_channel
        self.current_impedance_supported = impedance_supported
        self._update_mode_ui()

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
