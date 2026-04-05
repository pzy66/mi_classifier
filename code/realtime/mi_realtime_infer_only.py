"""Realtime MI inference launcher.

This entrypoint loads an already-trained MI artifact and supports two
runtime modes:

- ``continuous``: classify every rolling update and show ``UNCERTAIN`` when
  the evidence does not pass the decision thresholds.
- ``guided``: run a cue-aligned baseline -> cue -> imagery -> recovery loop
  so online timing matches collection and training more closely (defaults:
  baseline/cue/imagery/iti = 2.0/2.0/4.0/2.0s).
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from PyQt5.QtCore import QObject, QThread, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_ROOT = PROJECT_ROOT / "code" / "shared"
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

from src.realtime_mi import RealtimeMIPredictor, load_realtime_model  # noqa: E402
from src.serial_ports import detect_serial_ports  # noqa: E402


USER_CONFIG = {
    "realtime_mode": "continuous",
    "serial_port": "",
    "board_id": BoardIds.CYTON_BOARD.value,
    "model_path": PROJECT_ROOT / "code" / "realtime" / "models" / "custom_mi_realtime.joblib",
    "window_model_paths": [],
    "fusion_method": "log_weighted_mean",
    "fusion_weights": None,
    "use_artifact_recommended_thresholds": True,
    "use_artifact_recommended_gate_thresholds": True,
    "step_sec": 0.25,
    "history_len": 3,
    "confidence_threshold": 0.45,
    "gate_confidence_threshold": 0.60,
    "probability_smoothing": 0.35,
    "margin_threshold": 0.12,
    "gate_margin_threshold": 0.00,
    "switch_delta": 0.08,
    "hold_confidence_drop": 0.10,
    "hold_margin_drop": 0.04,
    "release_windows": 2,
    "gate_release_windows": 2,
    "min_stable_windows": 2,
    "flatline_std_threshold": 1e-7,
    "dominant_channel_ratio_threshold": 8.0,
    "max_bad_channels": 1,
    "artifact_freeze_windows": 2,
    "board_channel_positions": None,
    "protocol_baseline_sec": 2.0,
    "protocol_cue_sec": 2.0,
    "protocol_imagery_sec": 4.0,
    "protocol_iti_sec": 2.0,
    "protocol_trials_per_class": 4,
    "protocol_random_seed": 42,
}

VALID_REALTIME_MODES = {"continuous", "guided"}


CLASS_UI_NAMES = {
    "left_hand": "Left Hand",
    "right_hand": "Right Hand",
    "feet": "Feet",
    "tongue": "Tongue",
}

PHASE_UI_NAMES = {
    "idle": "Idle",
    "baseline": "Baseline",
    "cue": "Cue",
    "imagery": "Imagery",
    "iti": "Recovery",
    "finished": "Finished",
}

PHASE_BANNER_STYLES = {
    "idle": "background: #64748B; color: white;",
    "baseline": "background: #2563EB; color: white;",
    "cue": "background: #7C3AED; color: white;",
    "imagery": "background: #059669; color: white;",
    "iti": "background: #D97706; color: white;",
    "finished": "background: #1F2937; color: white;",
}


def build_balanced_protocol_sequence(trials_per_class: int, random_seed: int, class_names: list[str]) -> list[str]:
    """Build a block-balanced MI prompt order for the guided realtime protocol."""
    if trials_per_class <= 0:
        raise ValueError("protocol_trials_per_class must be positive.")
    if not class_names:
        raise ValueError("class_names cannot be empty.")

    rng = np.random.default_rng(int(random_seed))
    sequence: list[str] = []
    for _ in range(int(trials_per_class)):
        block = list(class_names)
        rng.shuffle(block)
        sequence.extend(block)
    return sequence


def normalize_realtime_mode(value: object) -> str:
    """Normalize the realtime mode string and validate the supported options."""
    mode = str(value or "continuous").strip().lower()
    if mode not in VALID_REALTIME_MODES:
        raise ValueError(
            f"Unsupported realtime_mode={mode!r}. Expected one of {sorted(VALID_REALTIME_MODES)}."
        )
    return mode


def validate_runtime_config(config: dict, artifact: dict) -> None:
    """Validate runtime settings before starting realtime acquisition."""
    normalize_realtime_mode(config.get("realtime_mode", "continuous"))
    board_id = int(config["board_id"])
    synthetic_board = getattr(BoardIds, "SYNTHETIC_BOARD", None)
    is_synthetic = synthetic_board is not None and board_id == int(synthetic_board.value)
    if not is_synthetic and not str(config.get("serial_port", "")).strip():
        raise ValueError("serial_port cannot be empty for non-synthetic boards.")
    resolve_board_channel_positions(
        board_id=board_id,
        expected_channels=len(artifact["channel_names"]),
        positions=config.get("board_channel_positions"),
        model_channel_names=artifact["channel_names"],
    )


def resolve_board_channel_positions(
    *,
    board_id: int,
    expected_channels: int,
    positions: object,
    model_channel_names: list[str] | tuple[str, ...],
) -> list[int]:
    """Normalize or auto-resolve realtime board channel positions."""
    available_eeg_rows = BoardShim.get_eeg_channels(int(board_id))
    available_count = int(len(available_eeg_rows))
    if expected_channels <= 0:
        raise ValueError("Expected at least one realtime channel.")

    if positions is None:
        if available_count == int(expected_channels):
            return list(range(int(expected_channels)))
        raise ValueError(
            "board_channel_positions must be set explicitly when the board EEG row count "
            f"({available_count}) differs from the model channel count ({expected_channels}). "
            f"Model channels: {list(model_channel_names)}."
        )

    normalized = [int(index) for index in positions]
    if len(normalized) != expected_channels:
        raise ValueError(
            f"board_channel_positions length mismatch: expected {expected_channels}, got {len(normalized)}."
        )
    if min(normalized) < 0:
        raise ValueError("board_channel_positions cannot contain negative indices.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("board_channel_positions contains duplicated indices.")
    if max(normalized) >= available_count:
        raise ValueError(
            f"board_channel_positions is invalid: board has {available_count} EEG rows "
            f"but the largest requested index is {max(normalized)}."
        )
    return normalized


def resolve_default_model_path() -> Path:
    """Prefer the custom trained model, then fall back to the bundled reference model."""
    candidates = [
        PROJECT_ROOT / "code" / "realtime" / "models" / "custom_mi_realtime.joblib",
        PROJECT_ROOT / "code" / "realtime" / "models" / "subject_1_mi.joblib",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def available_board_options() -> list[tuple[str, int]]:
    """Return a short list of common BrainFlow board presets."""
    options = []
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


def _scaled_px(base_px: float, scale: float, min_px: int, max_px: int | None = None) -> int:
    value = int(round(float(base_px) * float(scale)))
    if max_px is not None:
        value = min(value, int(max_px))
    return max(int(min_px), value)


class EEGWorker(QObject):
    data_ready = pyqtSignal(object)
    stream_progress = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()
    sampling_rate_ready = pyqtSignal(float)

    def __init__(self, config: dict, predictor: RealtimeMIPredictor) -> None:
        super().__init__()
        self.config = config
        self.predictor = predictor
        self.board = None
        self.is_running = False
        self.stop_event = threading.Event()
        self.selected_rows = None
        self.stream_sample_count = 0
        self.max_window_samples = 0
        self.live_buffer = None

    def start_collection(self) -> None:
        try:
            self.stop_event.clear()
            BoardShim.enable_dev_board_logger()
            try:
                BoardShim.release_all_sessions()
            except Exception:
                pass

            params = BrainFlowInputParams()
            params.serial_port = str(self.config["serial_port"])

            self.board = BoardShim(int(self.config["board_id"]), params)
            self.board.prepare_session()
            self.board.start_stream(450000)

            sampling_rate = float(BoardShim.get_sampling_rate(int(self.config["board_id"])))
            eeg_rows = BoardShim.get_eeg_channels(int(self.config["board_id"]))
            expected_channels = self.predictor.expected_channel_count
            selected_positions = resolve_board_channel_positions(
                board_id=int(self.config["board_id"]),
                expected_channels=expected_channels,
                positions=self.config.get("board_channel_positions"),
                model_channel_names=self.predictor.artifact["channel_names"],
            )

            self.selected_rows = [eeg_rows[index] for index in selected_positions]
            startup_samples = 8
            protocol_imagery_samples = int(
                round(float(self.config.get("protocol_imagery_sec", 0.0)) * sampling_rate)
            )
            max_window_samples = max(
                startup_samples,
                int(getattr(self.predictor, "stream_buffer_samples", self.predictor.maximum_required_samples)),
                protocol_imagery_samples,
            )
            self.max_window_samples = int(max_window_samples)
            self.stream_sample_count = 0
            self.live_buffer = np.empty((expected_channels, 0), dtype=np.float32)

            self.is_running = True
            self.sampling_rate_ready.emit(sampling_rate)
            self.status_changed.emit(
                "EEG stream started | "
                f"fs={sampling_rate:g} Hz | "
                f"board_rows={self.selected_rows} | "
                f"model_channels={self.predictor.artifact['channel_names']}"
            )

            while self.is_running and self.board.get_board_data_count() < startup_samples:
                available = self.board.get_board_data_count()
                self.status_changed.emit(
                    f"Waiting for startup buffer: {available}/{startup_samples} | max={max_window_samples}"
                )
                if self.stop_event.wait(0.2):
                    break

            while self.is_running and not self.stop_event.is_set():
                data = self.board.get_board_data()
                if data.size > 0 and data.shape[1] > 0:
                    eeg_chunk = np.ascontiguousarray(
                        data[self.selected_rows, :],
                        dtype=np.float32,
                    )
                    new_samples = int(eeg_chunk.shape[1])
                    if new_samples > 0:
                        self.stream_sample_count += new_samples
                        if self.live_buffer is None or self.live_buffer.size == 0:
                            self.live_buffer = eeg_chunk
                        else:
                            self.live_buffer = np.concatenate((self.live_buffer, eeg_chunk), axis=1)
                        if self.live_buffer.shape[1] > self.max_window_samples:
                            self.live_buffer = self.live_buffer[:, -self.max_window_samples :]
                        self.data_ready.emit(
                            {
                                "eeg_data": np.ascontiguousarray(self.live_buffer, dtype=np.float32),
                                "stream_sample_count": int(self.stream_sample_count),
                                "new_samples": int(new_samples),
                                "buffer_start_sample": int(
                                    max(0, self.stream_sample_count - int(self.live_buffer.shape[1]))
                                ),
                            }
                        )
                        self.stream_progress.emit(
                            {
                                "stream_sample_count": int(self.stream_sample_count),
                                "new_samples": int(new_samples),
                                "buffer_samples": int(self.live_buffer.shape[1]),
                            }
                        )
                if self.stop_event.wait(float(self.config["step_sec"])):
                    break

        except Exception as error:
            self.error_occurred.emit(f"Acquisition error: {error}")
        finally:
            self.cleanup()
            self.finished.emit()

    def request_stop(self) -> None:
        self.is_running = False
        self.stop_event.set()

    def cleanup(self) -> None:
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


class AnalysisWorker(QObject):
    result_ready = pyqtSignal(object)
    status_ready = pyqtSignal(str)

    def __init__(self, predictor: RealtimeMIPredictor, realtime_mode: str) -> None:
        super().__init__()
        self.predictor = predictor
        self.realtime_mode = normalize_realtime_mode(realtime_mode)
        self.live_sampling_rate = None
        self.protocol_state = {
            "phase": "idle",
            "phase_display_name": PHASE_UI_NAMES["idle"],
            "trial_number": 0,
            "trial_total": 0,
            "target_class": None,
            "target_display_name": "-",
            "imagery_elapsed_sec": 0.0,
            "countdown_sec": 0.0,
            "phase_start_stream_sample": 0,
        }

    def set_live_sampling_rate(self, sampling_rate: float) -> None:
        self.live_sampling_rate = float(sampling_rate)

    def update_protocol_state(self, protocol_state: dict) -> None:
        state = dict(protocol_state or {})
        if bool(state.get("reset_predictor")):
            self.predictor.reset_state()
        state["reset_predictor"] = False
        self.protocol_state = state

    def process_data(self, payload: object) -> None:
        try:
            if self.live_sampling_rate is None:
                self.status_ready.emit("Sampling rate is not ready yet.")
                return

            if isinstance(payload, dict):
                eeg_data = np.asarray(payload.get("eeg_data"), dtype=np.float32)
                stream_sample_count = int(payload.get("stream_sample_count", eeg_data.shape[1]))
            else:
                eeg_data = np.asarray(payload, dtype=np.float32)
                stream_sample_count = int(eeg_data.shape[1])

            minimum_live_samples = max(8, int(round(float(self.predictor.minimum_window_sec) * self.live_sampling_rate)))
            if int(eeg_data.shape[1]) < minimum_live_samples:
                return

            if self.realtime_mode == "continuous":
                buffer_start_sample = max(0, stream_sample_count - int(eeg_data.shape[1]))
                result = self.predictor.analyze_window(eeg_data, self.live_sampling_rate)
                result["realtime_mode"] = self.realtime_mode
                result["protocol_phase"] = "continuous"
                result["protocol_phase_display_name"] = "Continuous"
                result["protocol_trial_number"] = 0
                result["protocol_trial_total"] = 0
                result["protocol_target_class"] = None
                result["protocol_target_display_name"] = "-"
                result["protocol_imagery_elapsed_sec"] = float(result.get("input_duration_sec", 0.0))
                result["protocol_countdown_sec"] = 0.0
                result["protocol_stream_sample_count"] = int(stream_sample_count)
                result["protocol_phase_start_stream_sample"] = int(buffer_start_sample)
                self.result_ready.emit(result)
                return

            phase = str(self.protocol_state.get("phase", "idle"))
            if phase != "imagery":
                return

            phase_start_stream_sample = int(self.protocol_state.get("phase_start_stream_sample", stream_sample_count))
            imagery_elapsed_samples = max(0, int(stream_sample_count) - phase_start_stream_sample)
            imagery_elapsed_samples = min(int(eeg_data.shape[1]), imagery_elapsed_samples)
            if imagery_elapsed_samples < 8:
                return

            imagery_elapsed_sec = float(imagery_elapsed_samples) / float(self.live_sampling_rate)

            imagery_window = np.ascontiguousarray(eeg_data[:, -imagery_elapsed_samples:], dtype=np.float32)
            result = self.predictor.analyze_guided_imagery(
                imagery_window,
                self.live_sampling_rate,
                imagery_elapsed_sec=imagery_elapsed_sec,
            )
            result["realtime_mode"] = self.realtime_mode
            result["protocol_phase"] = phase
            result["protocol_phase_display_name"] = str(
                self.protocol_state.get("phase_display_name", PHASE_UI_NAMES["imagery"])
            )
            result["protocol_trial_number"] = int(self.protocol_state.get("trial_number", 0))
            result["protocol_trial_total"] = int(self.protocol_state.get("trial_total", 0))
            result["protocol_target_class"] = self.protocol_state.get("target_class")
            result["protocol_target_display_name"] = str(self.protocol_state.get("target_display_name", "-"))
            result["protocol_imagery_elapsed_sec"] = imagery_elapsed_sec
            result["protocol_countdown_sec"] = float(self.protocol_state.get("countdown_sec", 0.0))
            result["protocol_stream_sample_count"] = int(stream_sample_count)
            result["protocol_phase_start_stream_sample"] = int(phase_start_stream_sample)
            self.result_ready.emit(result)
        except Exception as error:
            self.status_ready.emit(f"Inference error: {error}")


class MIRealtimeWindow(QMainWindow):
    protocol_state_changed = pyqtSignal(object)

    def __init__(self, config: dict, predictor: RealtimeMIPredictor) -> None:
        super().__init__()
        self.config = dict(config)
        self.realtime_mode = normalize_realtime_mode(self.config.get("realtime_mode", "continuous"))
        self.predictor = predictor
        self.device_connected = False
        self.eeg_thread = None
        self.analysis_thread = None
        self.worker = None
        self.analysis_worker = None
        self.class_labels = {}
        self._stopping = False
        self._ui_scale = 1.0

        self.class_name_to_display = {
            name: display
            for name, display in zip(
                self.predictor.artifact["class_names"],
                self.predictor.artifact["display_class_names"],
            )
        }

        self.protocol_timer = QTimer(self)
        self.protocol_timer.setInterval(100)
        self.protocol_timer.timeout.connect(self.on_protocol_tick)
        self.protocol_running = False
        self.protocol_autostop_pending = False
        self.protocol_sequence: list[str] = []
        self.protocol_trial_index = -1
        self.protocol_phase = "idle"
        self.protocol_phase_started_at = 0.0
        self.protocol_phase_deadline = 0.0
        self.protocol_phase_duration_sec = 0.0
        self.protocol_target_class = None
        self.protocol_remaining_sec = 0.0
        self.protocol_sampling_rate = None
        self.current_stream_sample_count = 0
        self.last_stream_new_samples = 0
        self.protocol_phase_start_sample = 0
        self.protocol_phase_duration_samples = 0
        self.protocol_phase_deadline_sample = 0
        self.protocol_pending_start = False

        self.init_ui()
        self.set_protocol_idle_state(update_widgets=True)

    def init_ui(self) -> None:
        self.setWindowTitle(self.window_title_text())
        self.resize(1200, 860)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        connect_group = QGroupBox("Device Connection")
        connect_form = QFormLayout(connect_group)
        connect_form.setLabelAlignment(Qt.AlignRight)

        self.board_combo = QComboBox()
        for label, board_id in available_board_options():
            self.board_combo.addItem(f"{label} ({board_id})", board_id)
        for index in range(self.board_combo.count()):
            if int(self.board_combo.itemData(index)) == int(self.config["board_id"]):
                self.board_combo.setCurrentIndex(index)
                break
        self.board_combo.currentIndexChanged.connect(self.refresh_board_input_state)

        self.serial_combo = QComboBox()
        self.serial_combo.setEditable(True)
        self.refresh_serial_ports()

        port_row = QHBoxLayout()
        self.btn_refresh_ports = QPushButton("Refresh Ports")
        self.btn_refresh_ports.clicked.connect(self.refresh_serial_ports)
        port_row.addWidget(self.serial_combo, 1)
        port_row.addWidget(self.btn_refresh_ports)
        port_widget = QWidget()
        port_widget.setLayout(port_row)

        control_row = QHBoxLayout()
        self.btn_connect = QPushButton("Connect Device")
        self.btn_disconnect = QPushButton("Disconnect")
        self.btn_disconnect.setEnabled(False)
        self.btn_connect.clicked.connect(self.connect_device)
        self.btn_disconnect.clicked.connect(self.disconnect_device)
        control_row.addWidget(self.btn_connect)
        control_row.addWidget(self.btn_disconnect)
        control_widget = QWidget()
        control_widget.setLayout(control_row)

        connect_form.addRow("Board", self.board_combo)
        connect_form.addRow("Serial Port", port_widget)
        connect_form.addRow("", control_widget)
        layout.addWidget(connect_group)

        self.mode_banner_label = QLabel(self.mode_banner_text())
        self.mode_banner_label.setWordWrap(True)
        self.mode_banner_label.setAlignment(Qt.AlignCenter)
        self.mode_banner_label.setStyleSheet(
            "background: #FEF3C7; color: #92400E; border: 1px solid #F59E0B; border-radius: 8px; padding: 8px;"
        )
        layout.addWidget(self.mode_banner_label)

        self.header_label = QLabel()
        self.header_label.setAlignment(Qt.AlignCenter)
        self.header_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 12px;")
        layout.addWidget(self.header_label)

        self.channel_label = QLabel("Model channels: " + ", ".join(self.predictor.artifact["channel_names"]))
        self.channel_label.setAlignment(Qt.AlignCenter)
        self.channel_label.setStyleSheet("font-size: 14px; padding-bottom: 8px;")
        layout.addWidget(self.channel_label)

        protocol_group = QGroupBox(self.status_group_title())
        protocol_layout = QVBoxLayout(protocol_group)

        self.phase_banner_label = QLabel("Idle")
        self.phase_banner_label.setAlignment(Qt.AlignCenter)
        self.phase_banner_label.setProperty("phase_key", "idle")
        self.phase_banner_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; border-radius: 10px; padding: 14px;"
        )
        protocol_layout.addWidget(self.phase_banner_label)

        protocol_meta = QGridLayout()
        self.target_label = QLabel("Target: -")
        self.trial_label = QLabel("Trial: 0/0")
        self.countdown_label = QLabel("Remaining: --")
        self.protocol_hint_label = QLabel(self.default_protocol_hint())
        for widget in (
            self.target_label,
            self.trial_label,
            self.countdown_label,
            self.protocol_hint_label,
        ):
            widget.setStyleSheet("font-size: 15px; padding: 4px;")
        protocol_meta.addWidget(self.target_label, 0, 0)
        protocol_meta.addWidget(self.trial_label, 0, 1)
        protocol_meta.addWidget(self.countdown_label, 1, 0)
        protocol_meta.addWidget(self.protocol_hint_label, 1, 1)
        protocol_layout.addLayout(protocol_meta)
        layout.addWidget(protocol_group)

        button_row = QHBoxLayout()
        self.btn_start = QPushButton(self.start_button_text())
        self.btn_stop = QPushButton(self.stop_button_text())
        self.btn_exit = QPushButton("Exit")
        self.btn_stop.setEnabled(False)
        for button in (self.btn_start, self.btn_stop, self.btn_exit):
            button.setMinimumHeight(42)
            button_row.addWidget(button)
        layout.addLayout(button_row)

        self.result_label = QLabel("WAITING")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet(
            "font-size: 40px; font-weight: bold; color: white; background: #2c3e50; padding: 28px;"
        )
        layout.addWidget(self.result_label)

        self.confidence_label = QLabel("Prediction: --")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("font-size: 18px; padding: 8px;")
        layout.addWidget(self.confidence_label)

        score_grid = QGridLayout()
        for index, display_name in enumerate(self.predictor.artifact["display_class_names"]):
            card = QLabel(f"{display_name}\n--")
            card.setAlignment(Qt.AlignCenter)
            card.setMinimumHeight(110)
            card.setProperty("card_state", "default")
            score_grid.addWidget(card, index // 2, index % 2)
            self.class_labels[index] = card
        layout.addLayout(score_grid)

        bottom_row = QHBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("System log...")
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setPlaceholderText("Realtime probabilities...")
        bottom_row.addWidget(self.log_text, 1)
        bottom_row.addWidget(self.detail_text, 1)
        layout.addLayout(bottom_row)

        self.btn_start.clicked.connect(self.start_realtime)
        self.btn_stop.clicked.connect(self.stop_realtime)
        self.btn_exit.clicked.connect(self.close)

        self.set_running_state(False)
        self.refresh_board_input_state()
        self.refresh_header()
        self._apply_responsive_styles()

    def _compute_ui_scale(self) -> float:
        width = max(960, self.width())
        height = max(640, self.height())
        return max(0.78, min(2.0, min(width / 1200.0, height / 860.0)))

    def _scaled_ui_px(self, base_px: float, min_px: int, max_px: int | None = None) -> int:
        return _scaled_px(base_px, self._ui_scale, min_px=min_px, max_px=max_px)

    def _phase_banner_style(self, phase_style: str) -> str:
        font_size = self._scaled_ui_px(24, min_px=18, max_px=44)
        radius = self._scaled_ui_px(10, min_px=6, max_px=20)
        padding = self._scaled_ui_px(14, min_px=8, max_px=28)
        return f"font-size: {font_size}px; font-weight: bold; border-radius: {radius}px; padding: {padding}px; {phase_style}"

    def _class_card_style(self, state: str) -> str:
        state_key = str(state).strip().lower()
        border_color = "#bdc3c7"
        bg_color = "#ecf0f1"
        if state_key == "stable":
            border_color = "#1abc9c"
            bg_color = "#d1f2eb"
        elif state_key == "prediction":
            border_color = "#3498db"
            bg_color = "#d6eaf8"
        font_size = self._scaled_ui_px(18, min_px=13, max_px=32)
        border_width = self._scaled_ui_px(2, min_px=1, max_px=4)
        radius = self._scaled_ui_px(10, min_px=6, max_px=18)
        padding = self._scaled_ui_px(12, min_px=7, max_px=22)
        return (
            f"font-size: {font_size}px; font-weight: bold; border: {border_width}px solid {border_color}; "
            f"background: {bg_color}; border-radius: {radius}px; padding: {padding}px;"
        )

    def _set_class_card_state(self, index: int, state: str) -> None:
        card = self.class_labels[int(index)]
        normalized_state = str(state).strip().lower()
        if normalized_state not in {"default", "prediction", "stable"}:
            normalized_state = "default"
        card.setProperty("card_state", normalized_state)
        card.setStyleSheet(self._class_card_style(normalized_state))

    def _apply_responsive_styles(self) -> None:
        self._ui_scale = self._compute_ui_scale()
        self.setFont(QFont("Microsoft YaHei", self._scaled_ui_px(10, min_px=9, max_px=16)))

        if hasattr(self, "mode_banner_label"):
            banner_radius = self._scaled_ui_px(8, min_px=6, max_px=16)
            banner_pad = self._scaled_ui_px(8, min_px=6, max_px=16)
            banner_font = self._scaled_ui_px(13, min_px=11, max_px=22)
            self.mode_banner_label.setStyleSheet(
                f"background: #FEF3C7; color: #92400E; border: 1px solid #F59E0B; "
                f"border-radius: {banner_radius}px; padding: {banner_pad}px; font-size: {banner_font}px;"
            )

        if hasattr(self, "header_label"):
            header_font = self._scaled_ui_px(18, min_px=14, max_px=32)
            header_pad = self._scaled_ui_px(12, min_px=8, max_px=22)
            self.header_label.setStyleSheet(
                f"font-size: {header_font}px; font-weight: bold; padding: {header_pad}px;"
            )

        if hasattr(self, "channel_label"):
            channel_font = self._scaled_ui_px(14, min_px=11, max_px=24)
            channel_pad = self._scaled_ui_px(8, min_px=5, max_px=14)
            self.channel_label.setStyleSheet(f"font-size: {channel_font}px; padding-bottom: {channel_pad}px;")

        if hasattr(self, "phase_banner_label"):
            phase_key = str(self.phase_banner_label.property("phase_key") or "idle")
            phase_style = PHASE_BANNER_STYLES.get(phase_key, PHASE_BANNER_STYLES["idle"])
            self.phase_banner_label.setStyleSheet(self._phase_banner_style(phase_style))

        meta_font = self._scaled_ui_px(15, min_px=12, max_px=26)
        meta_pad = self._scaled_ui_px(4, min_px=2, max_px=10)
        for widget in (self.target_label, self.trial_label, self.countdown_label, self.protocol_hint_label):
            widget.setStyleSheet(f"font-size: {meta_font}px; padding: {meta_pad}px;")

        button_height = self._scaled_ui_px(42, min_px=34, max_px=64)
        for button in (self.btn_start, self.btn_stop, self.btn_exit):
            button.setMinimumHeight(button_height)

        result_font = self._scaled_ui_px(40, min_px=24, max_px=72)
        result_pad = self._scaled_ui_px(28, min_px=14, max_px=48)
        self.result_label.setStyleSheet(
            f"font-size: {result_font}px; font-weight: bold; color: white; background: #2c3e50; padding: {result_pad}px;"
        )

        confidence_font = self._scaled_ui_px(18, min_px=13, max_px=32)
        confidence_pad = self._scaled_ui_px(8, min_px=4, max_px=16)
        self.confidence_label.setStyleSheet(f"font-size: {confidence_font}px; padding: {confidence_pad}px;")

        card_height = self._scaled_ui_px(110, min_px=78, max_px=180)
        for index, card in self.class_labels.items():
            card.setMinimumHeight(card_height)
            state = str(card.property("card_state") or "default")
            self._set_class_card_state(int(index), state)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_responsive_styles()

    def phase_display_name(self, phase: str) -> str:
        return PHASE_UI_NAMES.get(str(phase), str(phase).title())

    def is_guided_mode(self) -> bool:
        return self.realtime_mode == "guided"

    def mode_display_name(self) -> str:
        return "Guided" if self.is_guided_mode() else "Continuous"

    def runtime_session_name(self) -> str:
        return "guided session" if self.is_guided_mode() else "continuous recognition"

    def window_title_text(self) -> str:
        return "MI Guided Realtime Inference" if self.is_guided_mode() else "MI Continuous Realtime Inference"

    def status_group_title(self) -> str:
        return "Guided Protocol" if self.is_guided_mode() else "Continuous Status"

    def start_button_text(self) -> str:
        return "Start Guided Session" if self.is_guided_mode() else "Start Continuous Recognition"

    def stop_button_text(self) -> str:
        return "Stop Session" if self.is_guided_mode() else "Stop Recognition"

    def mode_banner_text(self) -> str:
        if self.is_guided_mode():
            baseline_sec = float(self.config.get("protocol_baseline_sec", 0.0))
            cue_sec = float(self.config.get("protocol_cue_sec", 0.0))
            imagery_sec = float(self.config.get("protocol_imagery_sec", 0.0))
            iti_sec = float(self.config.get("protocol_iti_sec", 0.0))
            return (
                "Guided MI session (test protocol). Classification only runs during imagery. "
                f"Current timing baseline/cue/imagery/iti={baseline_sec:.1f}/{cue_sec:.1f}/{imagery_sec:.1f}/{iti_sec:.1f}s "
                "(collection-aligned default is 2.0/2.0/4.0/2.0s)."
            )
        return (
            "Continuous MI recognition. The classifier scores every rolling window and stays UNCERTAIN "
            "when confidence or margin does not pass the thresholds."
        )

    def default_protocol_hint(self) -> str:
        if self.is_guided_mode():
            return "Guided protocol is configurable via USER_CONFIG protocol_*; scoring starts at imagery."
        return "The classifier runs continuously and rejects low-evidence windows."

    def class_display_name(self, class_name: str | None) -> str:
        if class_name is None:
            return "-"
        if class_name in CLASS_UI_NAMES:
            return CLASS_UI_NAMES[class_name]
        return self.class_name_to_display.get(class_name, str(class_name))

    def current_board_id(self) -> int:
        return int(self.board_combo.currentData())

    def current_trial_number(self) -> int:
        if self.protocol_trial_index < 0:
            return 0
        return int(self.protocol_trial_index + 1)

    def protocol_trial_total(self) -> int:
        return int(len(self.protocol_sequence))

    def show_error(self, message: str) -> None:
        self.log(message)
        QMessageBox.critical(self, "Error", message)

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def refresh_serial_ports(self) -> None:
        configured = str(self.config.get("serial_port", "")).strip()
        selected = self.serial_combo.currentText().strip() or configured
        ports = detect_serial_ports()

        self.serial_combo.blockSignals(True)
        self.serial_combo.clear()
        for port in ports:
            self.serial_combo.addItem(port)
        if selected and self.serial_combo.findText(selected) < 0:
            self.serial_combo.addItem(selected)
        if selected:
            self.serial_combo.setCurrentText(selected)
        elif ports:
            self.serial_combo.setCurrentIndex(0)
        else:
            self.serial_combo.setCurrentText("")
        self.serial_combo.blockSignals(False)

    def refresh_board_input_state(self) -> None:
        synthetic_board = getattr(BoardIds, "SYNTHETIC_BOARD", None)
        is_synthetic = synthetic_board is not None and self.current_board_id() == int(synthetic_board.value)
        self.serial_combo.setEnabled(not is_synthetic and not self.device_connected)
        self.btn_refresh_ports.setEnabled(not is_synthetic and not self.device_connected)
        self.board_combo.setEnabled(not self.device_connected)
        line_edit = self.serial_combo.lineEdit()
        if line_edit is not None:
            line_edit.setPlaceholderText("No serial port needed for synthetic board" if is_synthetic else "Example: COM4")

    def refresh_header(self) -> None:
        board_text = self.board_combo.currentText() if hasattr(self, "board_combo") else str(self.config.get("board_id"))
        serial_text = str(self.config.get("serial_port", "")).strip() or "-"
        status_text = "connected" if self.device_connected else "disconnected"
        self.header_label.setText(
            f"{self.mode_display_name()} MI realtime | "
            f"device={status_text} | board={board_text} | serial={serial_text} | "
            f"pipeline={self.predictor.artifact['selected_pipeline']}"
        )

    def connect_device(self) -> None:
        if self.device_connected:
            self.log("Device is already connected.")
            return
        if self.eeg_thread is not None:
            self.show_error(f"Stop the current {self.runtime_session_name()} before reconnecting the device.")
            return

        selected_config = dict(self.config)
        selected_config["board_id"] = self.current_board_id()
        selected_config["serial_port"] = self.serial_combo.currentText().strip()

        try:
            normalized_positions = resolve_board_channel_positions(
                board_id=int(selected_config["board_id"]),
                expected_channels=self.predictor.expected_channel_count,
                positions=selected_config.get("board_channel_positions"),
                model_channel_names=self.predictor.artifact["channel_names"],
            )
            validate_runtime_config(selected_config, self.predictor.artifact)
        except Exception as error:
            self.show_error(f"Invalid connection settings: {error}")
            return

        if selected_config.get("board_channel_positions") is None:
            selected_config["board_channel_positions"] = list(normalized_positions)
            self.log(
                "Auto-resolved board_channel_positions to "
                f"{selected_config['board_channel_positions']} because the board EEG row count "
                "matches the model channel count."
            )

        params = BrainFlowInputParams()
        params.serial_port = str(selected_config["serial_port"])
        board = None
        try:
            BoardShim.release_all_sessions()
        except Exception:
            pass

        try:
            board = BoardShim(int(selected_config["board_id"]), params)
            board.prepare_session()
        except Exception as error:
            self.show_error(f"Connection failed: {error}")
            return
        finally:
            if board is not None:
                try:
                    board.release_session()
                except Exception:
                    pass

        self.config.update(selected_config)
        self.device_connected = True
        self.set_running_state(False)
        self.refresh_header()
        self.log(
            "Device connected | "
            f"board_id={self.config['board_id']} | serial={self.config['serial_port'] or '-'}"
        )

    def disconnect_device(self) -> None:
        if self.eeg_thread is not None:
            self.show_error(f"Stop the {self.runtime_session_name()} before disconnecting the device.")
            return
        if not self.device_connected:
            self.log("Device is already disconnected.")
            return

        self.device_connected = False
        self.set_running_state(False)
        self.refresh_header()
        self.log("Device disconnected.")

    def set_protocol_idle_state(self, *, update_widgets: bool) -> None:
        self.protocol_running = False
        self.protocol_autostop_pending = False
        self.protocol_sequence = []
        self.protocol_trial_index = -1
        self.protocol_phase = "idle"
        self.protocol_phase_started_at = 0.0
        self.protocol_phase_deadline = 0.0
        self.protocol_phase_duration_sec = 0.0
        self.protocol_target_class = None
        self.protocol_remaining_sec = 0.0
        self.protocol_sampling_rate = None
        self.current_stream_sample_count = 0
        self.last_stream_new_samples = 0
        self.protocol_phase_start_sample = 0
        self.protocol_phase_duration_samples = 0
        self.protocol_phase_deadline_sample = 0
        self.protocol_pending_start = False
        if update_widgets:
            self.update_protocol_widgets()
            self.clear_prediction_display("WAITING")

    def build_protocol_detail_lines(self) -> list[str]:
        if not self.is_guided_mode():
            return [
                "mode: Continuous",
                f"stream sample: {self.current_stream_sample_count}",
                f"step: {float(self.config['step_sec']):.3f}s",
                f"minimum window: {float(self.predictor.minimum_window_sec):.3f}s",
                "classification runs on every rolling update",
            ]
        return [
            f"protocol phase: {self.phase_display_name(self.protocol_phase)}",
            f"trial: {self.current_trial_number()}/{self.protocol_trial_total()}",
            f"target: {self.class_display_name(self.protocol_target_class)}",
            f"remaining: {self.protocol_remaining_sec:.2f}s",
            "classification is paused outside the imagery phase",
        ]

    def clear_prediction_display(self, title: str = "WAITING") -> None:
        self.result_label.setText(title)
        self.confidence_label.setText("Prediction: --")
        for index, display_name in enumerate(self.predictor.artifact["display_class_names"]):
            self._set_class_card_state(index, "default")
            self.class_labels[index].setText(f"{display_name}\n--")
        self.detail_text.setText("\n".join(self.build_protocol_detail_lines()))

    def emit_protocol_state(self, *, reset_predictor: bool = False) -> None:
        if not self.is_guided_mode():
            self.protocol_state_changed.emit(
                {
                    "phase": "continuous",
                    "phase_display_name": "Continuous",
                    "trial_number": 0,
                    "trial_total": 0,
                    "target_class": None,
                    "target_display_name": "-",
                    "imagery_elapsed_sec": 0.0,
                    "countdown_sec": 0.0,
                    "phase_start_stream_sample": int(max(0, self.current_stream_sample_count)),
                    "reset_predictor": bool(reset_predictor),
                }
            )
            return

        if self.protocol_sampling_rate:
            remaining_samples = max(0, self.protocol_phase_deadline_sample - self.current_stream_sample_count)
            self.protocol_remaining_sec = float(remaining_samples) / float(self.protocol_sampling_rate)
            if self.protocol_phase == "imagery":
                imagery_elapsed_samples = max(0, self.current_stream_sample_count - self.protocol_phase_start_sample)
                imagery_elapsed_samples = min(imagery_elapsed_samples, self.protocol_phase_duration_samples)
                imagery_elapsed_sec = float(imagery_elapsed_samples) / float(self.protocol_sampling_rate)
            else:
                imagery_elapsed_sec = 0.0
        else:
            now = time.perf_counter()
            remaining = 0.0
            if self.protocol_phase_deadline > 0.0:
                remaining = max(0.0, self.protocol_phase_deadline - now)
            self.protocol_remaining_sec = remaining
            if self.protocol_phase == "imagery":
                phase_elapsed = max(0.0, now - self.protocol_phase_started_at)
                imagery_elapsed_sec = min(phase_elapsed, self.protocol_phase_duration_sec)
            else:
                imagery_elapsed_sec = 0.0

        self.protocol_state_changed.emit(
            {
                "phase": self.protocol_phase,
                "phase_display_name": self.phase_display_name(self.protocol_phase),
                "trial_number": self.current_trial_number(),
                "trial_total": self.protocol_trial_total(),
                "target_class": self.protocol_target_class,
                "target_display_name": self.class_display_name(self.protocol_target_class),
                "imagery_elapsed_sec": float(imagery_elapsed_sec),
                "countdown_sec": float(self.protocol_remaining_sec),
                "phase_start_stream_sample": int(self.protocol_phase_start_sample),
                "reset_predictor": bool(reset_predictor),
            }
        )

    def update_protocol_widgets(self) -> None:
        if not self.is_guided_mode():
            is_running = self.eeg_thread is not None
            phase_display = "Continuous" if is_running else "Idle"
            phase_key = "imagery" if is_running else "idle"
            phase_style = PHASE_BANNER_STYLES[phase_key]
            self.phase_banner_label.setText(phase_display)
            self.phase_banner_label.setProperty("phase_key", phase_key)
            self.phase_banner_label.setStyleSheet(self._phase_banner_style(phase_style))
            self.target_label.setText("Target: -")
            self.trial_label.setText(f"Stream sample: {self.current_stream_sample_count}")
            self.countdown_label.setText(f"Step: {float(self.config['step_sec']):.2f}s")
            if is_running:
                self.protocol_hint_label.setText(
                    "Classifier is active on every update. Low-confidence windows stay UNCERTAIN."
                )
            else:
                self.protocol_hint_label.setText("Start continuous recognition to begin scoring.")
            return

        phase_display = self.phase_display_name(self.protocol_phase)
        self.phase_banner_label.setText(phase_display)
        banner_style = PHASE_BANNER_STYLES.get(self.protocol_phase, PHASE_BANNER_STYLES["idle"])
        self.phase_banner_label.setProperty("phase_key", str(self.protocol_phase))
        self.phase_banner_label.setStyleSheet(self._phase_banner_style(banner_style))
        self.target_label.setText(f"Target: {self.class_display_name(self.protocol_target_class)}")
        self.trial_label.setText(f"Trial: {self.current_trial_number()}/{self.protocol_trial_total()}")
        self.countdown_label.setText(f"Remaining: {self.protocol_remaining_sec:.1f}s")
        if self.protocol_phase == "imagery":
            self.protocol_hint_label.setText("Classifier is active and accumulates evidence inside imagery.")
        elif self.protocol_phase == "cue":
            self.protocol_hint_label.setText("Prepare the requested class. Scoring starts when imagery begins.")
        elif self.protocol_phase == "baseline":
            self.protocol_hint_label.setText("Relax and keep still. The classifier is idle.")
        elif self.protocol_phase == "iti":
            self.protocol_hint_label.setText("Short recovery before the next trial.")
        elif self.protocol_phase == "finished":
            self.protocol_hint_label.setText("Guided session finished.")
        else:
            self.protocol_hint_label.setText("Start a guided MI session to begin scoring.")

    def enter_protocol_phase(
        self,
        phase: str,
        duration_sec: float,
        class_name: str | None = None,
        *,
        start_sample: int | None = None,
        reset_predictor: bool = False,
    ) -> None:
        if not self.is_guided_mode():
            return

        now = time.perf_counter()
        self.protocol_phase = str(phase)
        self.protocol_phase_started_at = now
        self.protocol_phase_duration_sec = max(0.0, float(duration_sec))
        self.protocol_phase_deadline = now + self.protocol_phase_duration_sec
        self.protocol_remaining_sec = self.protocol_phase_duration_sec
        self.protocol_target_class = class_name
        if start_sample is None:
            start_sample = int(self.current_stream_sample_count)
        self.protocol_phase_start_sample = int(start_sample)
        if self.protocol_sampling_rate is not None:
            raw_duration_samples = int(round(self.protocol_phase_duration_sec * self.protocol_sampling_rate))
            if self.protocol_phase_duration_sec > 0.0:
                raw_duration_samples = max(1, raw_duration_samples)
            self.protocol_phase_duration_samples = raw_duration_samples
        else:
            self.protocol_phase_duration_samples = 0
        self.protocol_phase_deadline_sample = self.protocol_phase_start_sample + self.protocol_phase_duration_samples

        if phase == "imagery":
            self.clear_prediction_display("WARMING UP")
        else:
            self.clear_prediction_display(self.phase_display_name(phase).upper())

        self.update_protocol_widgets()
        self.emit_protocol_state(reset_predictor=reset_predictor)
        self.log(
            f"Protocol phase={self.phase_display_name(phase)} | "
            f"trial={self.current_trial_number()}/{self.protocol_trial_total()} | "
            f"target={self.class_display_name(class_name)}"
        )

    def start_protocol_session(self, *, anchor_sample: int) -> None:
        if not self.is_guided_mode():
            return

        class_names = list(self.predictor.artifact["class_names"])
        self.protocol_sequence = build_balanced_protocol_sequence(
            trials_per_class=int(self.config["protocol_trials_per_class"]),
            random_seed=int(self.config["protocol_random_seed"]),
            class_names=class_names,
        )
        self.protocol_running = True
        self.protocol_autostop_pending = False
        self.protocol_trial_index = -1
        self.protocol_timer.start()
        self.log(
            "Guided session started | "
            f"baseline={self.config['protocol_baseline_sec']:.2f}s | "
            f"cue={self.config['protocol_cue_sec']:.2f}s | "
            f"imagery={self.config['protocol_imagery_sec']:.2f}s | "
            f"iti={self.config['protocol_iti_sec']:.2f}s | "
            f"trials={self.protocol_trial_total()}"
        )
        self.start_next_protocol_trial(start_sample=int(anchor_sample))

    def start_next_protocol_trial(self, *, start_sample: int | None = None) -> None:
        if not self.is_guided_mode():
            return

        next_index = self.protocol_trial_index + 1
        if next_index >= self.protocol_trial_total():
            self.finish_protocol_session()
            return

        self.protocol_trial_index = next_index
        target_class = self.protocol_sequence[self.protocol_trial_index]
        self.enter_protocol_phase(
            phase="baseline",
            duration_sec=float(self.config["protocol_baseline_sec"]),
            class_name=target_class,
            start_sample=start_sample,
        )

    def finish_protocol_session(self) -> None:
        if not self.is_guided_mode():
            return

        if self.protocol_autostop_pending:
            return

        self.protocol_running = False
        self.protocol_timer.stop()
        self.protocol_phase = "finished"
        self.protocol_phase_started_at = time.perf_counter()
        self.protocol_phase_duration_sec = 0.0
        self.protocol_phase_deadline = 0.0
        self.protocol_remaining_sec = 0.0
        self.protocol_pending_start = False
        self.update_protocol_widgets()
        self.emit_protocol_state(reset_predictor=True)
        self.clear_prediction_display("FINISHED")
        self.log("Guided session finished. Stopping realtime stream.")
        self.protocol_autostop_pending = True
        QTimer.singleShot(250, self.stop_realtime)

    def on_protocol_tick(self) -> None:
        if not self.is_guided_mode():
            return
        if not self.protocol_running:
            return
        self.update_protocol_widgets()
        self.emit_protocol_state()

    def update_result(self, result: dict) -> None:
        stable_name = result["stable_prediction_display_name"]
        confidence = float(result["confidence"])
        decision_state = str(result["decision_state"])
        realtime_mode = str(result.get("realtime_mode", self.realtime_mode))
        gate_enabled = bool(result.get("gate_enabled", False))
        gate_passed = bool(result.get("gate_passed", True))
        artifact_rejector_enabled = bool(result.get("artifact_rejector_enabled", False))
        artifact_rejected = bool(result.get("artifact_rejected", False))

        title = stable_name
        if stable_name == "UNCERTAIN" and decision_state == "warming_up":
            title = "WARMING UP"
        self.result_label.setText(title)
        if artifact_rejector_enabled and artifact_rejected:
            self.confidence_label.setText(
                f"Artifact: {result.get('artifact_prediction_display_name', 'ARTIFACT')} | "
                f"conf={float(result.get('artifact_confidence', 0.0)):.3f} | "
                f"margin={float(result.get('artifact_margin', 0.0)):.3f} | state={decision_state}"
            )
        elif gate_enabled and not gate_passed:
            self.confidence_label.setText(
                f"Gate: {result['gate_prediction_display_name']} | conf={float(result['gate_confidence']):.3f} | "
                f"margin={float(result['gate_margin']):.3f} | state={decision_state}"
            )
        else:
            self.confidence_label.setText(
                f"Prediction: {result['prediction_display_name']} | conf={confidence:.3f} | "
                f"margin={result['margin']:.3f} | state={decision_state}"
            )

        for index, display_name in enumerate(result["display_class_names"]):
            score = result["probabilities"][index]
            state = "default"
            if result["stable_prediction_index"] == index:
                state = "stable"
            elif result["prediction_index"] == index:
                state = "prediction"
            self._set_class_card_state(index, state)
            self.class_labels[index].setText(f"{display_name}\n{score:.3f}")

        if realtime_mode == "guided":
            detail_lines = [
                f"protocol: phase={result.get('protocol_phase_display_name', '-')} | "
                f"trial={result.get('protocol_trial_number', 0)}/{result.get('protocol_trial_total', 0)} | "
                f"target={result.get('protocol_target_display_name', '-')}",
                f"imagery elapsed: {float(result.get('protocol_imagery_elapsed_sec', 0.0)):.3f}s | "
                f"countdown={float(result.get('protocol_countdown_sec', 0.0)):.3f}s",
                f"stream sample: {int(result.get('protocol_stream_sample_count', 0))} | "
                f"phase start sample: {int(result.get('protocol_phase_start_stream_sample', 0))}",
                "",
            ]
        else:
            detail_lines = [
                "mode: Continuous",
                f"stream sample: {int(result.get('protocol_stream_sample_count', 0))} | "
                f"buffer start sample: {int(result.get('protocol_phase_start_stream_sample', 0))}",
                f"input duration: {float(result.get('input_duration_sec', 0.0)):.3f}s | "
                f"step={float(self.config['step_sec']):.3f}s",
                "",
            ]
        if gate_enabled:
            gate_thresholds = dict(result.get("gate_thresholds", {}))
            detail_lines.append(
                f"gate: {result.get('gate_prediction_display_name', '-')} | "
                f"conf={float(result.get('gate_confidence', 0.0)):.3f} | "
                f"margin={float(result.get('gate_margin', 0.0)):.3f} | "
                f"passed={gate_passed} | state={result.get('gate_decision_state', '-')}"
            )
            detail_lines.append(
                f"gate thresholds: conf>={float(gate_thresholds.get('confidence_threshold', 0.0)):.3f} | "
                f"margin>={float(gate_thresholds.get('margin_threshold', 0.0)):.3f} | "
                f"release={int(result.get('gate_reject_count', 0))}/{int(result.get('gate_release_windows', 0))}"
            )
            for item in result.get("gate_window_evidence", []):
                detail_lines.append(
                    f"gate window={float(item['window_sec']):.2f}s | "
                    f"w={float(item.get('normalized_weight', item.get('weight', 0.0))):.2f} | "
                    f"{item.get('prediction_display_name', '-')} | conf={float(item.get('confidence', 0.0)):.3f} | "
                    f"margin={float(item.get('margin', 0.0)):.3f} | {item.get('selected_pipeline', '-')}"
                )
            detail_lines.append("")
        if artifact_rejector_enabled:
            artifact_thresholds = dict(result.get("artifact_thresholds", {}))
            detail_lines.append(
                f"artifact rejector: {result.get('artifact_prediction_display_name', '-')} | "
                f"conf={float(result.get('artifact_confidence', 0.0)):.3f} | "
                f"margin={float(result.get('artifact_margin', 0.0)):.3f} | "
                f"rejected={artifact_rejected} | state={result.get('artifact_decision_state', '-')}"
            )
            detail_lines.append(
                f"artifact thresholds: conf>={float(artifact_thresholds.get('confidence_threshold', 0.0)):.3f} | "
                f"margin>={float(artifact_thresholds.get('margin_threshold', 0.0)):.3f}"
            )
            for item in result.get("artifact_window_evidence", []):
                detail_lines.append(
                    f"artifact window={float(item['window_sec']):.2f}s | "
                    f"w={float(item.get('normalized_weight', item.get('weight', 0.0))):.2f} | "
                    f"{item.get('prediction_display_name', '-')} | conf={float(item.get('confidence', 0.0)):.3f} | "
                    f"margin={float(item.get('margin', 0.0)):.3f} | {item.get('selected_pipeline', '-')}"
                )
            detail_lines.append("")
        for display, raw_score, score in zip(
            result["display_class_names"],
            result["raw_probabilities"],
            result["probabilities"],
        ):
            detail_lines.append(f"{display}: raw={raw_score:.4f} | smooth={score:.4f}")
        detail_lines.append("")
        detail_lines.append(
            f"raw top1: {result['raw_prediction_display_name']} | "
            f"conf={result['raw_confidence']:.3f} | margin={result['raw_margin']:.3f}"
        )
        detail_lines.append(
            f"smooth top1: {result['prediction_display_name']} | "
            f"conf={result['confidence']:.3f} | margin={result['margin']:.3f}"
        )
        detail_lines.append(
            f"fusion: {result['fusion_method']} | input={result['input_duration_sec']:.3f}s | "
            f"active_windows={result['active_window_secs']} | offsets={result['configured_window_offset_secs']}"
        )
        for item in result["window_evidence"]:
            offset_text = ""
            if "window_offset_sec" in item:
                offset_text = (
                    f"offset={item['window_offset_sec']:.2f}s | "
                    f"decision_t={item['guided_end_sec']:.2f}s | "
                )
            detail_lines.append(
                f"{offset_text}window={item['window_sec']:.2f}s | w={item['normalized_weight']:.2f} | "
                f"{item['prediction_display_name']} | conf={item['confidence']:.3f} | "
                f"margin={item['margin']:.3f} | {item['selected_pipeline']}"
            )
        if result["pending_prediction_display_name"] is not None:
            detail_lines.append(
                f"pending: {result['pending_prediction_display_name']} | "
                f"{result['pending_count']}/{result['confirmation_windows']}"
            )
        detail_lines.append(
            f"stable age: {result['stable_age']} | min hold: {result['min_stable_windows']} | "
            f"release: {result['release_count']}/{result['release_windows']}"
        )
        quality_label = "ok" if result["quality_ok"] else "bad"
        detail_lines.append(
            f"quality: {quality_label} | reason={result['quality_reason']} | "
            f"bad_channels={result['quality_bad_channel_names']} | "
            f"artifact={result['artifact_count']}/{result['artifact_freeze_windows']}"
        )
        detail_lines.append(
            f"stable: {stable_name} | conf={result['stable_confidence']:.3f} | "
            f"margin={result['stable_margin']:.3f}"
        )
        self.detail_text.setText("\n".join(detail_lines))

    def on_sampling_rate_ready(self, sampling_rate: float) -> None:
        self.protocol_sampling_rate = float(sampling_rate)
        self.log(f"Sampling rate ready: {self.protocol_sampling_rate:g} Hz")
        if not self.is_guided_mode():
            self.update_protocol_widgets()

    def advance_protocol_with_stream(self) -> None:
        if not self.is_guided_mode():
            return
        while self.protocol_running and self.current_stream_sample_count >= self.protocol_phase_deadline_sample:
            phase_boundary_sample = int(self.protocol_phase_deadline_sample)

            if self.protocol_phase == "baseline":
                self.enter_protocol_phase(
                    phase="cue",
                    duration_sec=float(self.config["protocol_cue_sec"]),
                    class_name=self.protocol_target_class,
                    start_sample=phase_boundary_sample,
                )
                continue

            if self.protocol_phase == "cue":
                self.enter_protocol_phase(
                    phase="imagery",
                    duration_sec=float(self.config["protocol_imagery_sec"]),
                    class_name=self.protocol_target_class,
                    start_sample=phase_boundary_sample,
                    reset_predictor=True,
                )
                continue

            if self.protocol_phase == "imagery":
                self.enter_protocol_phase(
                    phase="iti",
                    duration_sec=float(self.config["protocol_iti_sec"]),
                    class_name=self.protocol_target_class,
                    start_sample=phase_boundary_sample,
                )
                continue

            if self.protocol_phase == "iti":
                self.start_next_protocol_trial(start_sample=phase_boundary_sample)
                continue

            break

    def on_stream_progress(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return

        self.current_stream_sample_count = int(payload.get("stream_sample_count", self.current_stream_sample_count))
        self.last_stream_new_samples = int(payload.get("new_samples", 0))

        if not self.is_guided_mode():
            self.update_protocol_widgets()
            return

        if self.protocol_pending_start and self.protocol_sampling_rate is not None and not self.protocol_running:
            self.protocol_pending_start = False
            anchor_sample = max(0, self.current_stream_sample_count - self.last_stream_new_samples)
            self.start_protocol_session(anchor_sample=int(anchor_sample))
            return

        if self.protocol_running:
            self.advance_protocol_with_stream()

    def set_running_state(self, running: bool) -> None:
        self.btn_start.setEnabled(not running and self.device_connected)
        self.btn_stop.setEnabled(running)
        self.btn_connect.setEnabled((not running) and (not self.device_connected))
        self.btn_disconnect.setEnabled((not running) and self.device_connected)
        self.refresh_board_input_state()

    def start_realtime(self) -> None:
        if self.eeg_thread is not None:
            self.log(f"{self.mode_display_name()} realtime is already running.")
            return
        if not self.device_connected:
            self.show_error(f"Connect the device before starting {self.runtime_session_name()}.")
            return
        try:
            normalized_positions = resolve_board_channel_positions(
                board_id=int(self.config["board_id"]),
                expected_channels=self.predictor.expected_channel_count,
                positions=self.config.get("board_channel_positions"),
                model_channel_names=self.predictor.artifact["channel_names"],
            )
            validate_runtime_config(self.config, self.predictor.artifact)
        except Exception as error:
            self.show_error(f"Invalid runtime configuration: {error}")
            return
        if self.config.get("board_channel_positions") is None:
            self.config["board_channel_positions"] = list(normalized_positions)
            self.log(
                "Auto-resolved board_channel_positions to "
                f"{self.config['board_channel_positions']} before starting realtime decoding."
            )

        self._stopping = False
        self.eeg_thread = QThread(self)
        self.analysis_thread = QThread(self)

        self.worker = EEGWorker(self.config, self.predictor)
        self.analysis_worker = AnalysisWorker(self.predictor, self.realtime_mode)
        self.predictor.reset_state()

        self.worker.moveToThread(self.eeg_thread)
        self.analysis_worker.moveToThread(self.analysis_thread)

        self.eeg_thread.started.connect(self.worker.start_collection)
        self.worker.finished.connect(self.eeg_thread.quit)
        self.worker.finished.connect(self.analysis_thread.quit)
        self.worker.status_changed.connect(self.log)
        self.worker.error_occurred.connect(self.on_worker_error)
        self.worker.sampling_rate_ready.connect(self.analysis_worker.set_live_sampling_rate)
        self.worker.sampling_rate_ready.connect(self.on_sampling_rate_ready)
        self.worker.data_ready.connect(self.analysis_worker.process_data, type=Qt.QueuedConnection)
        self.worker.stream_progress.connect(self.on_stream_progress, type=Qt.QueuedConnection)
        self.protocol_state_changed.connect(self.analysis_worker.update_protocol_state, type=Qt.QueuedConnection)
        self.analysis_worker.result_ready.connect(self.update_result)
        self.analysis_worker.status_ready.connect(self.log)
        self.eeg_thread.finished.connect(self.worker.deleteLater)
        self.analysis_thread.finished.connect(self.analysis_worker.deleteLater)

        self.analysis_thread.start()
        self.eeg_thread.start()
        self.set_running_state(True)
        self.refresh_header()
        self.current_stream_sample_count = 0
        self.last_stream_new_samples = 0
        self.clear_prediction_display("ARMING")
        if self.is_guided_mode():
            self.protocol_pending_start = True
            self.log("Realtime stream started. Waiting for the first buffer to anchor the guided protocol.")
        else:
            self.protocol_pending_start = False
            self.update_protocol_widgets()
            self.emit_protocol_state(reset_predictor=True)
            self.log(
                "Continuous recognition started. "
                "The output stays UNCERTAIN until the rolling window builds enough evidence."
            )

    def on_worker_error(self, message: str) -> None:
        self.log(message)
        if not self._stopping:
            self.stop_realtime()

    def stop_realtime(self) -> None:
        if self._stopping:
            return
        self._stopping = True

        self.protocol_timer.stop()
        if self.analysis_worker is not None:
            self.protocol_state_changed.emit(
                {
                    "phase": "idle",
                    "phase_display_name": PHASE_UI_NAMES["idle"],
                    "trial_number": 0,
                    "trial_total": 0,
                    "target_class": None,
                    "target_display_name": "-",
                    "imagery_elapsed_sec": 0.0,
                    "countdown_sec": 0.0,
                    "phase_start_stream_sample": int(self.protocol_phase_start_sample),
                    "reset_predictor": True,
                }
            )

        if self.worker is not None:
            self.worker.request_stop()

        if self.eeg_thread is not None:
            self.eeg_thread.quit()
            self.eeg_thread.wait()

        if self.analysis_thread is not None:
            self.analysis_thread.quit()
            self.analysis_thread.wait()

        self.eeg_thread = None
        self.analysis_thread = None
        self.worker = None
        self.analysis_worker = None
        self.set_protocol_idle_state(update_widgets=True)
        self.set_running_state(False)
        self.refresh_header()
        self.log(f"{self.mode_display_name()} realtime stopped.")
        self._stopping = False

    def closeEvent(self, event) -> None:
        self.stop_realtime()
        self.device_connected = False
        self.refresh_header()
        event.accept()


def main() -> None:
    raw_model_paths = USER_CONFIG.get("window_model_paths") or []
    model_paths = [Path(path) for path in raw_model_paths]
    using_fallback_reference_model = False
    runtime_model_reference: object

    if model_paths:
        missing_paths = [str(path) for path in model_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(
                "Some multi-window model files were not found: " + ", ".join(missing_paths)
            )
        runtime_model_reference = model_paths
        artifact = load_realtime_model(
            model_paths,
            fusion_weights=USER_CONFIG.get("fusion_weights"),
            fusion_method=str(USER_CONFIG.get("fusion_method", "log_weighted_mean")),
        )
    else:
        model_path = Path(USER_CONFIG["model_path"])
        requested_model_path = model_path
        if not model_path.exists():
            model_path = resolve_default_model_path()
            using_fallback_reference_model = model_path != requested_model_path
        if not model_path.exists():
            raise FileNotFoundError(
                "Realtime model file not found. "
                "Run the training pipeline first, or place a model under code/realtime/models/."
            )
        runtime_model_reference = model_path
        artifact = load_realtime_model(model_path)

    recommended_runtime = {}
    if bool(USER_CONFIG.get("use_artifact_recommended_thresholds", True)):
        recommended_runtime = dict(artifact.get("recommended_runtime") or {})
    recommended_gate_runtime = {}
    control_gate_artifact = artifact.get("control_gate") if isinstance(artifact, dict) else None
    if bool(USER_CONFIG.get("use_artifact_recommended_gate_thresholds", True)) and isinstance(control_gate_artifact, dict):
        recommended_gate_runtime = dict(control_gate_artifact.get("recommended_runtime") or {})
    confidence_threshold = float(recommended_runtime.get("confidence_threshold", USER_CONFIG["confidence_threshold"]))
    margin_threshold = float(recommended_runtime.get("margin_threshold", USER_CONFIG["margin_threshold"]))
    gate_confidence_threshold = float(
        recommended_gate_runtime.get("confidence_threshold", USER_CONFIG["gate_confidence_threshold"])
    )
    gate_margin_threshold = float(
        recommended_gate_runtime.get("margin_threshold", USER_CONFIG["gate_margin_threshold"])
    )

    predictor = RealtimeMIPredictor(
        artifact=artifact,
        history_len=int(USER_CONFIG["history_len"]),
        confidence_threshold=confidence_threshold,
        gate_confidence_threshold=gate_confidence_threshold,
        probability_smoothing=float(USER_CONFIG["probability_smoothing"]),
        margin_threshold=margin_threshold,
        gate_margin_threshold=gate_margin_threshold,
        switch_delta=float(USER_CONFIG["switch_delta"]),
        hold_confidence_drop=float(USER_CONFIG["hold_confidence_drop"]),
        hold_margin_drop=float(USER_CONFIG["hold_margin_drop"]),
        release_windows=int(USER_CONFIG["release_windows"]),
        gate_release_windows=int(USER_CONFIG["gate_release_windows"]),
        min_stable_windows=int(USER_CONFIG["min_stable_windows"]),
        flatline_std_threshold=float(USER_CONFIG["flatline_std_threshold"]),
        dominant_channel_ratio_threshold=float(USER_CONFIG["dominant_channel_ratio_threshold"]),
        max_bad_channels=int(USER_CONFIG["max_bad_channels"]),
        artifact_freeze_windows=int(USER_CONFIG["artifact_freeze_windows"]),
    )

    runtime_config = dict(USER_CONFIG)
    runtime_config["realtime_mode"] = normalize_realtime_mode(runtime_config.get("realtime_mode", "continuous"))
    runtime_config["model_path"] = runtime_model_reference
    runtime_config["confidence_threshold"] = confidence_threshold
    runtime_config["margin_threshold"] = margin_threshold
    runtime_config["recommended_runtime"] = recommended_runtime
    runtime_config["gate_confidence_threshold"] = gate_confidence_threshold
    runtime_config["gate_margin_threshold"] = gate_margin_threshold
    runtime_config["recommended_gate_runtime"] = recommended_gate_runtime
    app = QApplication(sys.argv)
    app.setApplicationName(
        "MI Guided Realtime Inference"
        if runtime_config["realtime_mode"] == "guided"
        else "MI Continuous Realtime Inference"
    )
    window = MIRealtimeWindow(runtime_config, predictor)
    window.show()

    if recommended_runtime:
        window.log(
            "Applied artifact-recommended thresholds | "
            f"confidence={confidence_threshold:.3f} | margin={margin_threshold:.3f}"
        )
    if recommended_gate_runtime:
        window.log(
            "Applied control-gate thresholds | "
            f"confidence={gate_confidence_threshold:.3f} | margin={gate_margin_threshold:.3f}"
        )

    if using_fallback_reference_model:
        warning_message = (
            "custom_mi_realtime.joblib was not found. "
            "The UI fell back to the bundled reference model subject_1_mi.joblib.\n"
            "Train a custom model before relying on this session."
        )
        window.log(warning_message.replace("\n", " "))
        QMessageBox.warning(window, "Model fallback", warning_message)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
