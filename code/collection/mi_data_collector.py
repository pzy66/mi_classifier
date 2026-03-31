"""全中文运动想象采集界面。"""

from __future__ import annotations

import argparse
from collections import deque
import secrets
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes, NoiseTypes
from PyQt5.QtCore import QObject, QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QBrush, QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    import winsound
except ImportError:  # pragma: no cover - non-Windows fallback
    winsound = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_ROOT = PROJECT_ROOT / "code" / "shared"
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

from src.mi_collection import (
    CLASS_LOOKUP,
    CONTINUOUS_PROMPT_EVENT_NAMES,
    CONTINUOUS_PROMPT_LABELS,
    DEFAULT_ARTIFACT_TYPES,
    SessionSettings,
    TrialRecord,
    build_balanced_trial_sequence,
    make_event,
    normalize_artifact_types,
    parse_channel_names,
    parse_channel_positions,
    save_mi_session,
)


DEFAULT_CHANNEL_NAMES = ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"]
IMAGERY_START_TONE_HZ = 1200
IMAGERY_START_TONE_MS = 140
IMAGERY_END_TONE_HZ = 800
IMAGERY_END_TONE_MS = 180
PHASE_LABELS = {
    "idle": "等待开始",
    "baseline": "准备阶段",
    "cue": "任务提示",
    "imagery": "想象执行",
    "iti": "休息恢复",
    "paused": "已暂停",
    "quality_check": "质量检查",
    "calibration_open": "睁眼静息",
    "calibration_closed": "闭眼静息",
    "calibration_eye_move": "眼动校准",
    "calibration_blink": "眨眼伪迹",
    "calibration_swallow": "吞咽伪迹",
    "calibration_jaw": "咬牙伪迹",
    "calibration_head": "头动伪迹",
    "practice": "想象训练",
    "run_rest": "轮次间休息",
    "idle_block": "无控制",
    "idle_prepare": "准备但不执行",
    "continuous": "连续仿真",
}
PHASE_ACCENT_COLORS = {
    "idle": "#64748B",
    "baseline": "#0F766E",
    "cue": "#2563EB",
    "imagery": "#8B5CF6",
    "iti": "#D97706",
    "paused": "#475569",
    "quality_check": "#334155",
    "calibration_open": "#0F766E",
    "calibration_closed": "#0369A1",
    "calibration_eye_move": "#7C3AED",
    "calibration_blink": "#B45309",
    "calibration_swallow": "#B91C1C",
    "calibration_jaw": "#BE123C",
    "calibration_head": "#C2410C",
    "practice": "#4F46E5",
    "run_rest": "#B45309",
    "idle_block": "#0D9488",
    "idle_prepare": "#0F766E",
    "continuous": "#0284C7",
}
PHASE_BACKGROUND_COLORS = {
    "idle": ("#E2E8F0", "#CBD5E1"),
    "baseline": ("#DCFCE7", "#BBF7D0"),
    "cue": ("#DBEAFE", "#BFDBFE"),
    "imagery": ("#F3E8FF", "#E9D5FF"),
    "iti": ("#FEF3C7", "#FDE68A"),
    "paused": ("#E5E7EB", "#CBD5E1"),
    "quality_check": ("#E2E8F0", "#CBD5E1"),
    "calibration_open": ("#DCFCE7", "#BBF7D0"),
    "calibration_closed": ("#DBEAFE", "#BFDBFE"),
    "calibration_eye_move": ("#F3E8FF", "#E9D5FF"),
    "calibration_blink": ("#FEF3C7", "#FDE68A"),
    "calibration_swallow": ("#FEE2E2", "#FECACA"),
    "calibration_jaw": ("#FCE7F3", "#FBCFE8"),
    "calibration_head": ("#FFEDD5", "#FED7AA"),
    "practice": ("#EDE9FE", "#DDD6FE"),
    "run_rest": ("#FEF3C7", "#FDE68A"),
    "idle_block": ("#CCFBF1", "#99F6E4"),
    "idle_prepare": ("#CCFBF1", "#99F6E4"),
    "continuous": ("#E0F2FE", "#BAE6FD"),
}
CUE_ASSET_DIR = PROJECT_ROOT / "code" / "collection" / "assets" / "cues"
CUE_IMAGE_FILES = {
    "left_hand": "left_hand.png",
    "right_hand": "right_hand.jpg",
    "feet": "feet.png",
    "tongue": "tongue.png",
}
CLASS_UI_NAMES = {
    "left_hand": "左手",
    "right_hand": "右手",
    "feet": "双脚",
    "tongue": "舌头",
}
CLASS_UI_IMAGERY_HINTS = {
    "left_hand": "持续想象左手握拳与放松的动作，不要真实移动手臂。",
    "right_hand": "持续想象右手握拳与放松的动作，不要真实移动手臂。",
    "feet": "持续想象双脚交替踩踏或抬脚动作，不要真实移动腿部。",
    "tongue": "持续想象舌头前伸/上抬动作，不要真实张口或吐舌。",
}

DEFAULT_CONFIG = {
    "serial_port": "COM3",
    "board_id": BoardIds.CYTON_BOARD.value,
    "subject_id": "001",
    "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "output_root": str(PROJECT_ROOT / "datasets" / "custom_mi"),
    "channel_names": ",".join(DEFAULT_CHANNEL_NAMES),
    "channel_positions": "0,1,2,3,4,5,6,7",
    "trials_per_class": 10,
    "baseline_sec": 2.0,
    "cue_sec": 1.0,
    "imagery_sec": 4.0,
    "iti_sec": 2.5,
    "run_count": 4,
    "max_consecutive_same_class": 2,
    "run_rest_sec": 90.0,
    "long_run_rest_every": 2,
    "long_run_rest_sec": 180.0,
    "quality_check_sec": 45.0,
    "practice_sec": 180.0,
    "calibration_open_sec": 120.0,
    "calibration_closed_sec": 60.0,
    "calibration_eye_sec": 60.0,
    "calibration_blink_sec": 30.0,
    "calibration_swallow_sec": 30.0,
    "calibration_jaw_sec": 30.0,
    "calibration_head_sec": 30.0,
    "idle_block_count": 2,
    "idle_block_sec": 90.0,
    "idle_prepare_block_count": 1,
    "idle_prepare_sec": 90.0,
    "continuous_block_count": 2,
    "continuous_block_sec": 150.0,
    "continuous_command_min_sec": 3.0,
    "continuous_command_max_sec": 6.0,
    "continuous_gap_min_sec": 1.0,
    "continuous_gap_max_sec": 3.0,
    "include_eyes_closed_rest_in_gate_neg": False,
    "artifact_types": ",".join(DEFAULT_ARTIFACT_TYPES),
    "reference_mode": "",
    "participant_state": "normal",
    "caffeine_intake": "unknown",
    "recent_exercise": "unknown",
    "sleep_note": "",
    "random_seed": 0,
    "save_epochs_npz": True,
    "use_separate_participant_screen": True,
    "notes": "",
}

PARTICIPANT_STATE_OPTIONS = [
    ("normal", "正常"),
    ("tired", "疲劳"),
    ("excited", "兴奋"),
]
CAFFEINE_OPTIONS = [
    ("unknown", "未知"),
    ("none", "无"),
    ("tea", "茶"),
    ("coffee", "咖啡"),
]
RECENT_EXERCISE_OPTIONS = [
    ("unknown", "未知"),
    ("no", "无"),
    ("yes", "有"),
]


def available_board_options() -> list[tuple[str, int]]:
    """Return a short list of common BrainFlow board presets."""
    options = []
    label_map = {
        "CYTON_BOARD": "Cyton（8 通道）",
        "CYTON_DAISY_BOARD": "Cyton Daisy（16 通道）",
        "GANGLION_BOARD": "Ganglion（4 通道）",
        "SYNTHETIC_BOARD": "模拟板卡（演示模式）",
    }
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
        label = label_map.get(name, name.replace("_BOARD", "").replace("_", " ").title())
        options.append((label, int(board.value)))
    return options


def detect_serial_ports() -> list[str]:
    """Detect available serial ports, falling back to common COM names."""
    try:
        from serial.tools import list_ports

        devices = sorted({str(port.device).strip() for port in list_ports.comports() if str(port.device).strip()})
        if devices:
            return devices
    except Exception:
        pass

    return [f"COM{i}" for i in range(1, 21)]


class CueIllustrationWidget(QWidget):
    """Large central cue card used to guide the participant."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.phase = "idle"
        self.class_name: str | None = None
        self.title = "等待开始"
        self.subtitle = "连接设备后开始采集"
        self.cue_images = self._load_cue_images()
        self.setMinimumHeight(430)

    def set_state(self, *, phase: str, class_name: str | None, title: str, subtitle: str) -> None:
        self.phase = phase
        self.class_name = class_name
        self.title = title
        self.subtitle = subtitle
        self.update()

    def _resolve_accent(self) -> QColor:
        if self.class_name in CLASS_LOOKUP:
            return QColor(str(CLASS_LOOKUP[self.class_name]["color"]))
        return QColor(PHASE_ACCENT_COLORS.get(self.phase, "#64748B"))

    @staticmethod
    def _fit_rect(container: QRectF, image_width: float, image_height: float) -> QRectF:
        if image_width <= 0 or image_height <= 0:
            return QRectF(container)
        container_ratio = container.width() / max(container.height(), 1.0)
        image_ratio = image_width / max(image_height, 1.0)
        if container_ratio > image_ratio:
            height = container.height()
            width = height * image_ratio
            x = container.center().x() - width / 2.0
            return QRectF(x, container.top(), width, height)
        width = container.width()
        height = width / image_ratio
        y = container.center().y() - height / 2.0
        return QRectF(container.left(), y, width, height)

    def _load_cue_images(self) -> dict[str, QImage]:
        loaded: dict[str, QImage] = {}
        for class_name, filename in CUE_IMAGE_FILES.items():
            path = CUE_ASSET_DIR / filename
            if not path.exists():
                continue
            image = QImage(str(path))
            if image.isNull():
                continue
            loaded[class_name] = image
        return loaded

    def _draw_cue_image(self, painter: QPainter, rect: QRectF, class_name: str, accent: QColor) -> bool:
        image = self.cue_images.get(class_name)
        if image is None:
            return False

        frame = rect.adjusted(8, 6, -8, -6)
        painter.setPen(QPen(QColor("#D6DEE8"), 2))
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawRoundedRect(frame, 24, 24)

        inner = frame.adjusted(18, 18, -18, -18)
        target = self._fit_rect(inner, float(image.width()), float(image.height()))
        painter.drawImage(target, image)

        if self.phase in {"cue", "imagery"}:
            painter.setPen(QPen(accent, 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(frame.adjusted(2, 2, -2, -2), 22, 22)
        return True

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        bg_top, bg_bottom = PHASE_BACKGROUND_COLORS.get(self.phase, ("#E2E8F0", "#CBD5E1"))
        canvas = QLinearGradient(0.0, 0.0, 0.0, float(self.height()))
        canvas.setColorAt(0.0, QColor(bg_top))
        canvas.setColorAt(1.0, QColor(bg_bottom))
        painter.fillRect(self.rect(), QBrush(canvas))

        card = self.rect().adjusted(20, 20, -20, -20)
        accent = self._resolve_accent()
        background = QColor("#FFFFFF")
        if self.phase in {"idle", "paused"}:
            background = QColor("#F8FAFC")

        painter.setPen(QPen(QColor("#C5CFDB"), 2))
        painter.setBrush(background)
        painter.drawRoundedRect(card, 28, 28)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(accent))
        painter.drawRoundedRect(QRectF(card.left() + 30, card.top() + 24, card.width() - 60, 12), 6, 6)

        text_rect = QRectF(card.left() + 36, card.top() + 52, card.width() - 72, 110)
        painter.setPen(QColor("#0F172A"))
        painter.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignTop, self.title)

        painter.setPen(QColor("#475569"))
        painter.setFont(QFont("Microsoft YaHei", 13))
        painter.drawText(
            QRectF(text_rect.left(), text_rect.bottom() - 4, text_rect.width(), 66),
            Qt.AlignLeft | Qt.AlignTop,
            self.subtitle,
        )

        drawing_rect = QRectF(card.left() + 60, card.top() + 180, card.width() - 120, card.height() - 230)
        if self.phase in {"idle", "baseline", "iti", "paused"} or self.class_name is None:
            self._draw_fixation(painter, drawing_rect, accent)
            return

        if self._draw_cue_image(painter, drawing_rect, self.class_name, accent):
            return

        if self.class_name == "left_hand":
            self._draw_hand(painter, drawing_rect, accent, mirrored=False)
        elif self.class_name == "right_hand":
            self._draw_hand(painter, drawing_rect, accent, mirrored=True)
        elif self.class_name == "feet":
            self._draw_feet(painter, drawing_rect, accent)
        elif self.class_name == "tongue":
            self._draw_tongue(painter, drawing_rect, accent)

    @staticmethod
    def _draw_fixation(painter: QPainter, rect: QRectF, color: QColor) -> None:
        center = rect.center()
        painter.setPen(QPen(color, 8, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(QPointF(center.x() - 44, center.y()), QPointF(center.x() + 44, center.y()))
        painter.drawLine(QPointF(center.x(), center.y() - 44), QPointF(center.x(), center.y() + 44))

    @staticmethod
    def _draw_hand(painter: QPainter, rect: QRectF, color: QColor, *, mirrored: bool) -> None:
        width = rect.width()
        height = rect.height()
        palm = QRectF(rect.left() + width * 0.33, rect.top() + height * 0.34, width * 0.22, height * 0.28)
        if mirrored:
            palm.moveLeft(rect.right() - palm.width() - (palm.left() - rect.left()))

        painter.setPen(QPen(color, 6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(palm, 24, 24)

        finger_spacing = palm.width() / 5.0
        for finger in range(5):
            x = palm.left() + finger_spacing * (finger + 0.5)
            painter.drawLine(QPointF(x, palm.top() + 2), QPointF(x, palm.top() - height * 0.16))

        if mirrored:
            thumb = QPainterPath(QPointF(palm.right() - 2, palm.top() + palm.height() * 0.34))
            thumb.lineTo(palm.right() + width * 0.10, palm.top() + palm.height() * 0.48)
            thumb.lineTo(palm.right() + width * 0.04, palm.top() + palm.height() * 0.70)
        else:
            thumb = QPainterPath(QPointF(palm.left() + 2, palm.top() + palm.height() * 0.34))
            thumb.lineTo(palm.left() - width * 0.10, palm.top() + palm.height() * 0.48)
            thumb.lineTo(palm.left() - width * 0.04, palm.top() + palm.height() * 0.70)
        painter.drawPath(thumb)

    @staticmethod
    def _draw_feet(painter: QPainter, rect: QRectF, color: QColor) -> None:
        painter.setPen(QPen(color, 6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)
        left = QRectF(rect.left() + rect.width() * 0.23, rect.top() + rect.height() * 0.24, rect.width() * 0.20, rect.height() * 0.44)
        right = QRectF(rect.left() + rect.width() * 0.57, rect.top() + rect.height() * 0.24, rect.width() * 0.20, rect.height() * 0.44)
        for foot in (left, right):
            painter.drawRoundedRect(foot, 26, 26)
            toe_radius = foot.width() * 0.08
            toe_y = foot.top() - toe_radius * 0.4
            for toe_index in range(5):
                toe_x = foot.left() + foot.width() * (0.16 + toe_index * 0.17)
                painter.drawEllipse(QPointF(toe_x, toe_y), toe_radius, toe_radius)

    @staticmethod
    def _draw_tongue(painter: QPainter, rect: QRectF, color: QColor) -> None:
        painter.setPen(QPen(color, 6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)
        face = QPainterPath()
        face.moveTo(rect.left() + rect.width() * 0.28, rect.top() + rect.height() * 0.24)
        face.cubicTo(
            rect.left() + rect.width() * 0.58,
            rect.top() + rect.height() * 0.08,
            rect.left() + rect.width() * 0.78,
            rect.top() + rect.height() * 0.30,
            rect.left() + rect.width() * 0.70,
            rect.top() + rect.height() * 0.54,
        )
        face.cubicTo(
            rect.left() + rect.width() * 0.64,
            rect.top() + rect.height() * 0.80,
            rect.left() + rect.width() * 0.38,
            rect.top() + rect.height() * 0.84,
            rect.left() + rect.width() * 0.28,
            rect.top() + rect.height() * 0.58,
        )
        painter.drawPath(face)
        painter.drawLine(
            QPointF(rect.left() + rect.width() * 0.50, rect.top() + rect.height() * 0.44),
            QPointF(rect.left() + rect.width() * 0.66, rect.top() + rect.height() * 0.47),
        )
        tongue = QPainterPath(QPointF(rect.left() + rect.width() * 0.66, rect.top() + rect.height() * 0.47))
        tongue.cubicTo(
            rect.left() + rect.width() * 0.80,
            rect.top() + rect.height() * 0.50,
            rect.left() + rect.width() * 0.80,
            rect.top() + rect.height() * 0.64,
            rect.left() + rect.width() * 0.68,
            rect.top() + rect.height() * 0.64,
        )
        painter.drawPath(tongue)


class ParticipantDisplayWindow(QWidget):
    """Full-screen participant-facing prompt window."""

    pause_requested = pyqtSignal()
    mark_bad_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__(None, Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("受试者提示屏")
        self.setStyleSheet("background: #04111C;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 36, 40, 36)
        layout.setSpacing(18)

        self.stage_label = QLabel("等待开始")
        self.stage_label.setAlignment(Qt.AlignCenter)
        self.stage_label.setStyleSheet(
            "color: white; font-size: 28px; font-weight: bold; "
            "background: #475569; border-radius: 18px; padding: 12px 20px;"
        )
        layout.addWidget(self.stage_label)

        self.cue_widget = CueIllustrationWidget()
        self.cue_widget.setMinimumHeight(360)
        layout.addWidget(self.cue_widget, stretch=1)

        self.countdown_label = QLabel("--")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet(
            "color: #F8FAFC; font-size: 56px; font-weight: bold; "
            "background: rgba(15, 23, 42, 170); border: 2px solid #334155; border-radius: 22px; padding: 10px 20px;"
        )
        layout.addWidget(self.countdown_label)

        self.hint_label = QLabel("空格 暂停/继续    B 标记坏试次/命令失败    Esc 停止并保存")
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.hint_label.setStyleSheet("color: #94A3B8; font-size: 15px;")
        layout.addWidget(self.hint_label)

    @staticmethod
    def _resolve_phase_accent(phase: str, class_name: str | None) -> str:
        if class_name in CLASS_LOOKUP and phase in {"cue", "imagery"}:
            return str(CLASS_LOOKUP[class_name]["color"])
        return PHASE_ACCENT_COLORS.get(phase, "#64748B")

    def set_prompt(
        self,
        *,
        phase: str,
        class_name: str | None,
        stage_text: str,
        title: str,
        subtitle: str,
        countdown_text: str,
    ) -> None:
        accent = self._resolve_phase_accent(phase, class_name)
        self.stage_label.setText(stage_text)
        self.stage_label.setStyleSheet(
            f"color: white; font-size: 28px; font-weight: bold; "
            f"background: {accent}; border-radius: 18px; padding: 12px 20px;"
        )
        self.cue_widget.set_state(
            phase=phase,
            class_name=class_name,
            title=title,
            subtitle=subtitle,
        )
        self.countdown_label.setText(countdown_text)
        self.countdown_label.setStyleSheet(
            f"color: #F8FAFC; font-size: 56px; font-weight: bold; "
            f"background: rgba(15, 23, 42, 170); border: 2px solid {accent}; border-radius: 22px; padding: 10px 20px;"
        )

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Space:
            self.pause_requested.emit()
            event.accept()
            return
        if event.key() == Qt.Key_B:
            self.mark_bad_requested.emit()
            event.accept()
            return
        if event.key() == Qt.Key_Escape:
            self.stop_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_requested.emit()
        event.ignore()


class RealtimeEEGPreviewWidget(QWidget):
    """Operator-facing EEG/impedance preview for connection quality checks."""

    MODE_EEG = "EEG"
    MODE_IMPEDANCE = "IMP"
    LEAD_OFF_CURRENT_AMPS = 6e-9
    SERIES_RESISTOR_OHMS = 2200.0

    CHANNEL_COLORS = [
        "#38BDF8",
        "#22C55E",
        "#F59E0B",
        "#A855F7",
        "#EF4444",
        "#14B8A6",
        "#EAB308",
        "#F97316",
    ]

    def __init__(self, parent: QWidget | None = None, *, window_seconds: float = 5.0) -> None:
        super().__init__(parent)
        self.window_seconds = max(1.0, float(window_seconds))
        self.sampling_rate = 0.0
        self.channel_names: list[str] = []
        self.max_points = 0
        self.buffers: list[deque[float]] = []
        self.last_chunk_perf = 0.0
        self.last_repaint_perf = 0.0
        self.mode = self.MODE_EEG
        self.impedance_channel = 1
        self.last_impedance_ohms: list[float | None] = []
        self.placeholder_text = f"连接设备后显示最近 {self.window_seconds:.1f} 秒波形。"
        self._dirty = False

        self.setMinimumHeight(420)
        self.setStyleSheet("background: #08111F; border: 1px solid #1E293B; border-radius: 14px;")

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(50)
        self.refresh_timer.timeout.connect(self._flush_repaint)
        self.refresh_timer.start()

    def configure_stream(self, *, sampling_rate: float, channel_names: list[str]) -> None:
        self.sampling_rate = max(1.0, float(sampling_rate))
        self.channel_names = [str(name) for name in channel_names]
        self.max_points = max(32, int(round(self.window_seconds * self.sampling_rate)))
        self.buffers = [deque(maxlen=self.max_points) for _ in self.channel_names]
        self.last_impedance_ohms = [None] * len(self.channel_names)
        self.mode = self.MODE_EEG
        self.impedance_channel = 1
        self.last_chunk_perf = 0.0
        self.last_repaint_perf = 0.0
        self.placeholder_text = "正在等待实时数据..."
        self._dirty = True
        self.update()

    def clear_stream(self, message: str | None = None) -> None:
        self.buffers = []
        self.channel_names = []
        self.max_points = 0
        self.sampling_rate = 0.0
        self.last_chunk_perf = 0.0
        self.last_repaint_perf = 0.0
        self.mode = self.MODE_EEG
        self.impedance_channel = 1
        self.last_impedance_ohms = []
        if message:
            self.placeholder_text = str(message)
        else:
            self.placeholder_text = f"连接设备后显示最近 {self.window_seconds:.1f} 秒波形。"
        self._dirty = True
        self.update()

    def set_quality_mode(self, mode: str, *, impedance_channel: int | None = None) -> None:
        normalized = self.MODE_IMPEDANCE if str(mode).strip().upper().startswith("IMP") else self.MODE_EEG
        self.mode = normalized
        if impedance_channel is not None:
            self.set_impedance_channel(int(impedance_channel))
        else:
            self.set_impedance_channel(int(self.impedance_channel))
        if self.mode == self.MODE_IMPEDANCE:
            self._update_impedance_value()
        self._dirty = True
        self.update()

    def set_impedance_channel(self, channel: int) -> int:
        channel_count = len(self.channel_names)
        if channel_count <= 0:
            self.impedance_channel = 1
            return self.impedance_channel
        self.impedance_channel = int(np.clip(int(channel), 1, channel_count))
        self._dirty = True
        return self.impedance_channel

    def append_chunk(self, eeg_chunk: np.ndarray) -> None:
        if not self.buffers:
            return

        chunk = np.asarray(eeg_chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[0] != len(self.buffers) or chunk.shape[1] == 0:
            return

        for channel_index, buffer in enumerate(self.buffers):
            buffer.extend(np.asarray(chunk[channel_index], dtype=np.float32).tolist())

        if self.mode == self.MODE_IMPEDANCE:
            self._update_impedance_value()

        self.last_chunk_perf = time.perf_counter()
        self._dirty = True

    def _flush_repaint(self) -> None:
        now = time.perf_counter()
        if not self._dirty and not (self.channel_names and now - self.last_repaint_perf >= 0.25):
            return
        self._dirty = False
        self.last_repaint_perf = now
        self.update()

    def _estimate_impedance_ohms_from_raw_window(self, y_uvolts: np.ndarray) -> float | None:
        if y_uvolts is None or y_uvolts.size < 20:
            return None
        std_uvolts = float(np.std(y_uvolts, ddof=0))
        z_ohm = (np.sqrt(2.0) * std_uvolts * 1e-6) / float(self.LEAD_OFF_CURRENT_AMPS)
        z_ohm -= float(self.SERIES_RESISTOR_OHMS)
        if not np.isfinite(z_ohm):
            return None
        return max(z_ohm, 0.0)

    def _update_impedance_value(self) -> None:
        active_index = int(self.impedance_channel) - 1
        if active_index < 0 or active_index >= len(self.buffers):
            return
        if active_index >= len(self.last_impedance_ohms):
            return
        y_raw = np.asarray(self.buffers[active_index], dtype=np.float64)
        self.last_impedance_ohms[active_index] = self._estimate_impedance_ohms_from_raw_window(y_raw)

    def _build_plot_signal(self, y_raw: np.ndarray) -> np.ndarray:
        y = np.asarray(y_raw, dtype=np.float64)
        if y.size <= 10 or self.mode != self.MODE_EEG or self.sampling_rate <= 1.0:
            return y

        y_plot = y.copy()
        try:
            DataFilter.detrend(y_plot, DetrendOperations.CONSTANT.value)
        except Exception:
            pass
        try:
            DataFilter.remove_environmental_noise(y_plot, self.sampling_rate, NoiseTypes.FIFTY.value)
        except Exception:
            pass
        try:
            DataFilter.perform_bandpass(
                y_plot,
                self.sampling_rate,
                1.0,
                40.0,
                4,
                FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
                0,
            )
        except Exception:
            pass
        return y_plot

    def _draw_placeholder(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QColor("#CBD5E1"))
        painter.setFont(QFont("Microsoft YaHei", 12))
        painter.drawText(rect, Qt.AlignCenter | Qt.TextWordWrap, self.placeholder_text)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        if self.mode == self.MODE_IMPEDANCE:
            self._update_impedance_value()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#08111F"))

        outer_rect = QRectF(self.rect()).adjusted(12, 12, -12, -12)
        painter.setPen(QPen(QColor("#22314A"), 1.0))
        painter.setBrush(QColor("#0B1727"))
        painter.drawRoundedRect(outer_rect, 14, 14)

        header_rect = QRectF(outer_rect.left() + 14, outer_rect.top() + 10, outer_rect.width() - 28, 28)
        painter.setPen(QColor("#E2E8F0"))
        painter.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))

        if self.channel_names and self.sampling_rate > 0:
            freshness_sec = 0.0 if self.last_chunk_perf <= 0 else max(0.0, time.perf_counter() - self.last_chunk_perf)
            if self.mode == self.MODE_EEG:
                mode_text = "EEG mode (display: detrend + 50Hz + 1-40Hz)"
            else:
                mode_text = f"Impedance mode (CH{self.impedance_channel}, raw only)"
            header_text = (
                f"{mode_text} | {len(self.channel_names)} ch | {self.sampling_rate:g} Hz | "
                f"window {self.window_seconds:.1f}s"
            )
            painter.drawText(header_rect, Qt.AlignLeft | Qt.AlignVCenter, header_text)
            painter.setPen(QColor("#F59E0B" if freshness_sec > 1.0 else "#94A3B8"))
            painter.setFont(QFont("Consolas", 10))
            painter.drawText(
                header_rect,
                Qt.AlignRight | Qt.AlignVCenter,
                f"last update {freshness_sec:.2f}s",
            )
        else:
            painter.drawText(header_rect, Qt.AlignLeft | Qt.AlignVCenter, "EEG / Impedance Preview")

        content_rect = QRectF(outer_rect.left() + 14, outer_rect.top() + 48, outer_rect.width() - 28, outer_rect.height() - 62)
        if not self.channel_names or not self.buffers:
            self._draw_placeholder(painter, content_rect)
            return

        channel_count = len(self.channel_names)
        row_gap = 8.0
        row_height = max(28.0, (content_rect.height() - row_gap * (channel_count - 1)) / max(1, channel_count))
        label_width = 58.0
        right_info_width = 130.0

        for channel_index, channel_name in enumerate(self.channel_names):
            row_top = content_rect.top() + channel_index * (row_height + row_gap)
            row_rect = QRectF(content_rect.left(), row_top, content_rect.width(), row_height)
            plot_rect = QRectF(
                row_rect.left() + label_width,
                row_rect.top() + 4,
                max(10.0, row_rect.width() - label_width - right_info_width),
                max(16.0, row_rect.height() - 8),
            )

            active_impedance_channel = self.mode == self.MODE_IMPEDANCE and (channel_index + 1 == self.impedance_channel)
            row_fill = QColor("#0E1B2D" if not active_impedance_channel else "#1F2937")
            painter.setPen(Qt.NoPen)
            painter.setBrush(row_fill)
            painter.drawRoundedRect(row_rect, 10, 10)

            painter.setPen(QColor("#E2E8F0" if active_impedance_channel else "#94A3B8"))
            painter.setFont(QFont("Consolas", 10, QFont.Bold))
            painter.drawText(
                QRectF(row_rect.left() + 8, row_rect.top(), label_width - 10, row_rect.height()),
                Qt.AlignLeft | Qt.AlignVCenter,
                channel_name,
            )

            grid_pen = QPen(QColor("#1E293B"), 1.0)
            painter.setPen(grid_pen)
            for grid_index in range(5):
                x = plot_rect.left() + plot_rect.width() * grid_index / 4.0
                painter.drawLine(QPointF(x, plot_rect.top()), QPointF(x, plot_rect.bottom()))
            painter.drawLine(
                QPointF(plot_rect.left(), plot_rect.center().y()),
                QPointF(plot_rect.right(), plot_rect.center().y()),
            )

            y_raw = np.asarray(self.buffers[channel_index], dtype=np.float64)
            if y_raw.size < 2:
                painter.setPen(QColor("#64748B"))
                painter.setFont(QFont("Microsoft YaHei", 9))
                painter.drawText(plot_rect, Qt.AlignCenter, "waiting...")
                continue

            y = self._build_plot_signal(y_raw)

            if y.size >= 20:
                low = float(np.percentile(y, 5))
                high = float(np.percentile(y, 95))
            else:
                low = float(np.min(y))
                high = float(np.max(y))

            if high - low < 20.0:
                center = (high + low) / 2.0
                low = center - 10.0
                high = center + 10.0

            pad = max(5.0, 0.2 * (high - low))
            y_min = low - pad
            y_max = high + pad
            span = max(1.0, y_max - y_min)

            if y_min <= 0.0 <= y_max:
                zero_y = plot_rect.bottom() - ((0.0 - y_min) / span) * plot_rect.height()
                painter.setPen(QPen(QColor("#334155"), 1.0, Qt.DashLine))
                painter.drawLine(QPointF(plot_rect.left(), zero_y), QPointF(plot_rect.right(), zero_y))

            waveform = QPainterPath()
            x_scale = plot_rect.width() / max(1, y.size - 1)
            for point_index, sample in enumerate(y):
                x = plot_rect.left() + point_index * x_scale
                y_pos = plot_rect.bottom() - ((float(sample) - y_min) / span) * plot_rect.height()
                if point_index == 0:
                    waveform.moveTo(x, y_pos)
                else:
                    waveform.lineTo(x, y_pos)

            color = QColor(self.CHANNEL_COLORS[channel_index % len(self.CHANNEL_COLORS)])
            if self.mode == self.MODE_IMPEDANCE and not active_impedance_channel:
                color.setAlpha(95)
            painter.setPen(
                QPen(
                    color,
                    1.6 if active_impedance_channel else 1.2,
                    Qt.SolidLine,
                    Qt.RoundCap,
                    Qt.RoundJoin,
                )
            )
            painter.drawPath(waveform)

            info_text = ""
            info_color = QColor("#64748B")
            if self.mode == self.MODE_IMPEDANCE:
                if active_impedance_channel:
                    z_ohm = None
                    if channel_index < len(self.last_impedance_ohms):
                        z_ohm = self.last_impedance_ohms[channel_index]
                    if z_ohm is None:
                        info_text = "Imp: --"
                    else:
                        info_text = f"Imp ~ {z_ohm / 1000.0:.1f} kOhm"
                    info_color = QColor("#FDE68A")
                else:
                    info_text = "Imp: --"
            else:
                info_text = f"{np.ptp(y_raw):.0f} uV"

            painter.setPen(info_color)
            painter.setFont(QFont("Consolas", 9))
            painter.drawText(
                QRectF(plot_rect.right() + 6, row_rect.top(), right_info_width - 6, row_rect.height()),
                Qt.AlignLeft | Qt.AlignVCenter,
                info_text,
            )


class BoardCaptureWorker(QObject):
    """Background BrainFlow worker used only for recording, not analysis."""

    MODE_EEG = "EEG"
    MODE_IMPEDANCE = "IMP"

    connection_ready = pyqtSignal(object)
    preview_data_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    session_data_ready = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(
        self,
        *,
        board_id: int,
        serial_port: str,
        channel_positions: list[int],
        channel_names: list[str],
        poll_interval_sec: float = 0.05,
    ) -> None:
        super().__init__()
        self.board_id = int(board_id)
        self.serial_port = serial_port
        self.channel_positions = list(channel_positions)
        self.channel_names = list(channel_names)
        self.poll_interval_sec = float(poll_interval_sec)
        self.stop_event = threading.Event()
        self.board_lock = threading.Lock()
        self.board: BoardShim | None = None
        self.selected_rows: list[int] = []
        self.marker_row: int | None = None
        self.timestamp_row: int | None = None
        self.sampling_rate: float | None = None

    @staticmethod
    def build_impedance_command(channel: int, test_p: bool = True, test_n: bool = False) -> str:
        p = 1 if test_p else 0
        n = 1 if test_n else 0
        return f"z{int(channel)}{p}{n}Z"

    def supports_impedance_mode(self) -> bool:
        cyton_ids: set[int] = set()
        for board_name in ("CYTON_BOARD", "CYTON_DAISY_BOARD"):
            board_enum = getattr(BoardIds, board_name, None)
            if board_enum is not None:
                cyton_ids.add(int(board_enum.value))
        return int(self.board_id) in cyton_ids

    def _selected_channel_count(self) -> int:
        if self.selected_rows:
            return max(1, int(len(self.selected_rows)))
        if self.channel_names:
            return max(1, int(len(self.channel_names)))
        try:
            return max(1, int(len(BoardShim.get_eeg_channels(self.board_id))))
        except Exception:
            return 8

    def switch_quality_mode_sync(
        self,
        *,
        target_mode: str,
        target_channel: int = 1,
        reset_default: bool = False,
    ) -> tuple[bool, str]:
        mode = str(target_mode).strip().upper()
        if mode not in {self.MODE_EEG, self.MODE_IMPEDANCE}:
            return False, f"Unsupported preview mode: {target_mode!r}"
        if mode == self.MODE_IMPEDANCE and not self.supports_impedance_mode():
            return False, "Impedance mode is available only for Cyton/Cyton Daisy boards."

        with self.board_lock:
            if self.board is None:
                return False, "Device is not connected."

            board = self.board
            channel_count = self._selected_channel_count()
            channel = int(np.clip(int(target_channel), 1, channel_count))

            try:
                board.stop_stream()
            except Exception:
                pass

            try:
                if self.supports_impedance_mode():
                    for ch in range(1, channel_count + 1):
                        board.config_board(self.build_impedance_command(ch, False, False))
                    if bool(reset_default) or mode == self.MODE_EEG:
                        board.config_board("d")
                    if mode == self.MODE_IMPEDANCE:
                        board.config_board(self.build_impedance_command(channel, True, False))

                board.start_stream(450000)
            except Exception as error:
                try:
                    board.start_stream(450000)
                except Exception:
                    pass
                return False, f"Failed to switch quality-check mode: {error}"

        if mode == self.MODE_IMPEDANCE:
            self.status_changed.emit(f"Quality-check mode -> IMPEDANCE (CH{channel})")
        elif bool(reset_default):
            self.status_changed.emit("Quality-check mode -> EEG (reset defaults)")
        else:
            self.status_changed.emit("Quality-check mode -> EEG")
        return True, ""

    @pyqtSlot()
    def run(self) -> None:
        final_payload = None
        data_chunks: list[np.ndarray] = []
        try:
            self.stop_event.clear()
            try:
                BoardShim.release_all_sessions()
            except Exception:
                pass

            params = BrainFlowInputParams()
            if self.serial_port:
                params.serial_port = self.serial_port

            with self.board_lock:
                self.board = BoardShim(self.board_id, params)
                self.board.prepare_session()
                self.board.start_stream(450000)

            eeg_rows = BoardShim.get_eeg_channels(self.board_id)
            if len(self.channel_positions) > len(eeg_rows):
                raise ValueError(
                    f"当前板卡只提供 {len(eeg_rows)} 个 EEG 通道，但你选择了 {len(self.channel_positions)} 个位置。"
                )

            self.selected_rows = [int(eeg_rows[index]) for index in self.channel_positions]
            self.marker_row = int(BoardShim.get_marker_channel(self.board_id))
            self.timestamp_row = int(BoardShim.get_timestamp_channel(self.board_id))
            self.sampling_rate = float(BoardShim.get_sampling_rate(self.board_id))

            self.connection_ready.emit(
                {
                    "sampling_rate": self.sampling_rate,
                    "selected_rows": self.selected_rows,
                    "marker_row": self.marker_row,
                    "timestamp_row": self.timestamp_row,
                    "channel_names": self.channel_names,
                }
            )
            self.status_changed.emit(
                f"设备已连接 | 采样率 {self.sampling_rate:g} Hz | 通道 {', '.join(self.channel_names)}"
            )

            while not self.stop_event.wait(self.poll_interval_sec):
                if self.board is None:
                    break
                try:
                    with self.board_lock:
                        chunk = None if self.board is None else self.board.get_board_data()
                except Exception:
                    chunk = None
                if chunk is not None and np.size(chunk):
                    chunk_array = np.asarray(chunk, dtype=np.float32)
                    data_chunks.append(chunk_array)
                    if self.selected_rows:
                        try:
                            preview_chunk = np.asarray(chunk_array[self.selected_rows, :], dtype=np.float32)
                        except Exception:
                            preview_chunk = None
                        if preview_chunk is not None and preview_chunk.size:
                            self.preview_data_ready.emit(preview_chunk)
        except Exception as error:
            self.error_occurred.emit(f"设备采集线程出错：{error}")
        finally:
            if self.board is not None:
                try:
                    with self.board_lock:
                        tail_chunk = None if self.board is None else self.board.get_board_data()
                except Exception:
                    tail_chunk = None
                if tail_chunk is not None and np.size(tail_chunk):
                    data_chunks.append(np.asarray(tail_chunk, dtype=np.float32))

                if data_chunks:
                    try:
                        full_data = np.concatenate(data_chunks, axis=1)
                    except Exception:
                        full_data = data_chunks[-1]
                else:
                    full_data = None

                if full_data is not None and np.size(full_data):
                    final_payload = {
                        "brainflow_data": np.asarray(full_data, dtype=np.float32),
                        "sampling_rate": None if self.sampling_rate is None else float(self.sampling_rate),
                        "selected_rows": list(self.selected_rows),
                        "marker_row": self.marker_row,
                        "timestamp_row": self.timestamp_row,
                    }

                with self.board_lock:
                    try:
                        if self.board is not None:
                            self.board.stop_stream()
                    except Exception:
                        pass
                    try:
                        if self.board is not None:
                            self.board.release_session()
                    except Exception:
                        pass
                    self.board = None

            if final_payload is not None:
                self.session_data_ready.emit(final_payload)
            self.status_changed.emit("设备已断开。")
            self.finished.emit()

    @pyqtSlot(float)
    def insert_marker(self, marker_code: float) -> None:
        if self.board is None:
            return
        try:
            self.board.insert_marker(float(marker_code))
        except Exception as error:
            self.error_occurred.emit(f"写入标记失败：{error}")

    def insert_marker_sync(self, marker_code: float) -> tuple[bool, str]:
        with self.board_lock:
            if self.board is None:
                return False, "设备未连接，无法写入标记。"
            try:
                self.board.insert_marker(float(marker_code))
            except Exception as error:
                return False, f"写入标记失败：{error}"
        return True, ""

    @pyqtSlot()
    def request_stop(self) -> None:
        self.stop_event.set()


class MIDataCollectorWindow(QMainWindow):
    """主采集窗口，负责流程控制与状态展示。"""

    marker_requested = pyqtSignal(float)
    worker_stop_requested = pyqtSignal()

    def __init__(self, initial_config: dict | None = None) -> None:
        super().__init__()
        self.config = dict(DEFAULT_CONFIG)
        if initial_config:
            self.config.update(initial_config)

        self.worker_thread: QThread | None = None
        self.worker: BoardCaptureWorker | None = None
        self.device_info: dict | None = None
        self.capture_on_stop = False
        self.session_running = False
        self.session_paused = False
        self.waiting_for_save = False
        self.current_phase = "idle"
        self.phase_deadline = 0.0
        self.phase_started_perf = 0.0
        self.pause_started_perf = 0.0
        self.remaining_phase_sec = 0.0
        self.session_start_perf = 0.0
        self.sequence: list[str] = []
        self.sequence_by_run: list[list[str]] = []
        self.trials_per_run = 0
        self.current_run_index = 0
        self.current_run_trial_index = 0
        self.pending_run_rest_before_index: int | None = None
        self.event_log: list[dict[str, object]] = []
        self.trial_records: list[TrialRecord] = []
        self.current_trial_index = -1
        self.current_trial: TrialRecord | None = None
        self.current_settings: SessionSettings | None = None
        self.completed_trials = 0
        self.operator_hidden_for_session = False
        self.use_separate_participant_screen = bool(self.config.get("use_separate_participant_screen", True))
        self.calibration_plan: list[dict[str, object]] = []
        self.calibration_step_index = -1
        self.idle_block_index = 0
        self.idle_prepare_block_index = 0
        self.continuous_block_index = 0
        self.continuous_prompt_plan: list[dict[str, object]] = []
        self.continuous_prompt_index = -1
        self.current_continuous_prompt: dict[str, object] | None = None
        self.continuous_schedule_after_runs: list[int] = []
        self.next_continuous_schedule_index = 0
        self.marker_failure_active = False
        self.marker_failure_message = ""
        self.config_panel_widget: QWidget | None = None
        self.config_groups_grid: QGridLayout | None = None
        self.config_groups: list[QWidget] = []
        self.config_grid_columns = 0
        self.main_splitter: QSplitter | None = None
        self.operator_preview_panel: QWidget | None = None
        self.hero_title_label: QLabel | None = None
        self.hero_subtitle_label: QLabel | None = None
        self.preview_widget: RealtimeEEGPreviewWidget | None = None
        self.preview_status_label: QLabel | None = None
        self.preview_mode_label: QLabel | None = None
        self.preview_to_eeg_button: QPushButton | None = None
        self.preview_to_imp_button: QPushButton | None = None
        self.preview_prev_ch_button: QPushButton | None = None
        self.preview_next_ch_button: QPushButton | None = None
        self.preview_reset_button: QPushButton | None = None
        self.preview_mode = RealtimeEEGPreviewWidget.MODE_EEG
        self.preview_impedance_channel = 1
        self.log_group: QGroupBox | None = None
        self.log_text: QTextEdit | None = None

        self.phase_timer = QTimer(self)
        self.phase_timer.setInterval(100)
        self.phase_timer.timeout.connect(self.on_phase_tick)

        self._init_ui()
        self.participant_window = ParticipantDisplayWindow()
        self.participant_window.pause_requested.connect(self.toggle_pause)
        self.participant_window.mark_bad_requested.connect(self.mark_bad_trial)
        self.participant_window.stop_requested.connect(self.stop_and_save)
        self.apply_default_values()
        self.refresh_board_input_state()
        self._apply_phase_theme("idle", None)

    def _desired_config_columns(self) -> int:
        if self.config_panel_widget is None:
            return 3
        width = self.config_panel_widget.width()
        height = self.config_panel_widget.height()
        if width >= 1500:
            columns = 4
        elif width >= 980:
            columns = 3
        else:
            columns = 2
        # When vertical space is tight, prefer more columns to avoid clipping without scrollbars.
        if height < 760:
            columns = max(columns, 3 if width >= 980 else 2)
        return columns

    def _refresh_config_group_layout(self, force: bool = False) -> None:
        if self.config_groups_grid is None or not self.config_groups:
            return
        target_columns = self._desired_config_columns()
        if not force and target_columns == self.config_grid_columns:
            return
        while self.config_groups_grid.count():
            item = self.config_groups_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                self.config_groups_grid.removeWidget(widget)
        for index, group in enumerate(self.config_groups):
            row = index // target_columns
            column = index % target_columns
            self.config_groups_grid.addWidget(group, row, column)
        for column in range(4):
            self.config_groups_grid.setColumnStretch(column, 1 if column < target_columns else 0)
        self.config_grid_columns = target_columns

    @staticmethod
    def _configure_form_layout(form: QFormLayout) -> None:
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignTop)
        form.setSpacing(6)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

    @staticmethod
    def _generate_session_seed() -> int:
        """Generate a per-session random seed and keep it in a visible 6-digit range."""
        return int(secrets.randbelow(900000) + 100000)

    def _init_ui(self) -> None:
        self.setWindowTitle("运动想象采集台（仅采集，不判别）")
        self.resize(1700, 960)
        self.setMinimumSize(1280, 760)
        self.setStyleSheet(
            """
            QMainWindow { background: #E8EEF5; }
            QStatusBar {
                background: #0F172A;
                color: #E2E8F0;
                border-top: 1px solid #334155;
            }
            QLabel { color: #102A43; font-size: 12px; }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #D5DFEB;
                border-radius: 18px;
                margin-top: 12px;
                padding-top: 8px;
                background: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: #1E293B;
                background: #FFFFFF;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
                border: 1px solid #CBD5E1;
                border-radius: 11px;
                padding: 7px 10px;
                background: #FFFFFF;
                selection-background-color: #2563EB;
                selection-color: #FFFFFF;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {
                border: 2px solid #2563EB;
            }
            QPushButton {
                border: 1px solid transparent;
                border-radius: 12px;
                padding: 10px 16px;
                font-weight: bold;
                background: #1D4ED8;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background: #1E40AF;
            }
            QPushButton:pressed {
                background: #1E3A8A;
            }
            QPushButton:disabled {
                background: #D1D5DB;
                color: #64748B;
                border-color: #D1D5DB;
            }
            QPushButton#btnConnect { background: #0F766E; }
            QPushButton#btnConnect:hover { background: #0B5E59; }
            QPushButton#btnStart { background: #2563EB; }
            QPushButton#btnStart:hover { background: #1D4ED8; }
            QPushButton#btnPause { background: #7C3AED; }
            QPushButton#btnPause:hover { background: #6D28D9; }
            QPushButton#btnWarn { background: #EA580C; }
            QPushButton#btnWarn:hover { background: #C2410C; }
            QPushButton#btnStop { background: #DC2626; }
            QPushButton#btnStop:hover { background: #B91C1C; }
            QPushButton#btnDisconnect { background: #334155; }
            QPushButton#btnDisconnect:hover { background: #1E293B; }
            QTextEdit#logPanel {
                background: #0B1727;
                color: #D1E3FF;
                border: 1px solid #1E3A5F;
                border-radius: 12px;
                font-family: Consolas, "Microsoft YaHei";
            }
            QProgressBar {
                border: 1px solid #CBD5E1;
                border-radius: 10px;
                text-align: center;
                background: #F8FAFC;
                color: #1E293B;
            }
            QProgressBar::chunk {
                background: #0F766E;
                border-radius: 9px;
            }
            """
        )

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)

        self.hero_title_label = QLabel("运动想象数据采集（纯采集，不判别）")
        self.hero_title_label.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        self.hero_title_label.setStyleSheet("color: #0F172A; letter-spacing: 1px;")
        root_layout.addWidget(self.hero_title_label)

        self.hero_subtitle_label = QLabel("当前版本按标准试次流程工作：每个试次都包含准备、提示、想象和休息，只负责事件标记与数据保存。")
        self.hero_subtitle_label.setWordWrap(True)
        self.hero_subtitle_label.setStyleSheet(
            "color: #334155; font-size: 13px; background: #FFFFFF; border: 1px solid #D5DFEB; border-radius: 10px; padding: 8px 12px;"
        )
        root_layout.addWidget(self.hero_subtitle_label)

        body = QHBoxLayout()
        body.setSpacing(14)
        root_layout.addLayout(body, stretch=1)
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.config_panel_widget = self._build_config_panel()
        session_panel = self._build_session_panel()
        self.main_splitter.addWidget(self.config_panel_widget)
        self.main_splitter.addWidget(session_panel)
        self.main_splitter.setStretchFactor(0, 7)
        self.main_splitter.setStretchFactor(1, 3)
        self.main_splitter.setSizes([1200, 500])
        self.main_splitter.splitterMoved.connect(lambda _pos, _idx: self._refresh_config_group_layout())
        body.addWidget(self.main_splitter, stretch=1)

        self.operator_preview_panel = self._build_focus_panel()
        self.operator_preview_panel.setMinimumWidth(460)
        body.addWidget(self.operator_preview_panel, stretch=0)
        self._refresh_config_group_layout(force=True)

        self.log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(self.log_group)
        self.log_text = QTextEdit()
        self.log_text.setObjectName("logPanel")
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(92)
        log_layout.addWidget(self.log_text)
        root_layout.addWidget(self.log_group, stretch=0)

        self.statusBar().showMessage("准备就绪")
        self._update_responsive_chrome()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_config_group_layout()
        self._update_responsive_chrome()

    def _update_responsive_chrome(self) -> None:
        compact = self.height() < 880 or self.width() < 1500
        if self.hero_subtitle_label is not None:
            self.hero_subtitle_label.setVisible(not compact)
        if self.log_text is not None:
            self.log_text.setMinimumHeight(76 if compact else 92)
        if self.main_splitter is not None:
            if compact:
                self.main_splitter.setStretchFactor(0, 8)
                self.main_splitter.setStretchFactor(1, 2)
            else:
                self.main_splitter.setStretchFactor(0, 7)
                self.main_splitter.setStretchFactor(1, 3)

    def _build_config_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.subject_edit = QLineEdit()
        self.subject_edit.setPlaceholderText("例如 S01 或 张三")
        self.session_edit = QLineEdit()
        self.output_edit = QLineEdit()
        browse_button = QPushButton("选择目录")
        browse_button.clicked.connect(self.browse_output_directory)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, stretch=1)
        output_row.addWidget(browse_button)
        output_widget = QWidget()
        output_widget.setLayout(output_row)

        self.board_combo = QComboBox()
        for label, board_id in available_board_options():
            self.board_combo.addItem(f"{label} ({board_id})", board_id)
        self.board_combo.currentIndexChanged.connect(self.refresh_board_input_state)

        self.serial_combo = QComboBox()
        self.serial_combo.setEditable(True)
        self.refresh_ports_button = QPushButton("刷新串口")
        self.refresh_ports_button.clicked.connect(self.refresh_serial_ports)
        serial_row = QHBoxLayout()
        serial_row.setContentsMargins(0, 0, 0, 0)
        serial_row.setSpacing(8)
        serial_row.addWidget(self.serial_combo, stretch=1)
        serial_row.addWidget(self.refresh_ports_button, stretch=0)
        serial_widget = QWidget()
        serial_widget.setLayout(serial_row)
        self.channel_names_edit = QLineEdit()
        self.channel_positions_edit = QLineEdit()
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(1, 999)
        self.baseline_spin = QDoubleSpinBox()
        self.baseline_spin.setRange(0.5, 60.0)
        self.baseline_spin.setDecimals(1)
        self.baseline_spin.setSingleStep(0.5)
        self.cue_spin = QDoubleSpinBox()
        self.cue_spin.setRange(0.5, 20.0)
        self.cue_spin.setDecimals(1)
        self.cue_spin.setSingleStep(0.5)
        self.imagery_spin = QDoubleSpinBox()
        self.imagery_spin.setRange(0.5, 20.0)
        self.imagery_spin.setDecimals(1)
        self.imagery_spin.setSingleStep(0.5)
        self.iti_spin = QDoubleSpinBox()
        self.iti_spin.setRange(0.5, 30.0)
        self.iti_spin.setDecimals(1)
        self.iti_spin.setSingleStep(0.5)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(100000, 999999)
        self.seed_spin.setReadOnly(True)
        self.seed_spin.setToolTip("每次开始采集时自动生成并保存到会话元数据")
        self.save_epochs_check = QCheckBox("同时导出 *_epochs.npz（自动编号）")
        self.separate_screen_check = QCheckBox("开始采集后弹出受试者全屏提示窗")
        self.notes_edit = QTextEdit()
        self.notes_edit.setMinimumHeight(88)
        self.run_count_spin = QSpinBox()
        self.run_count_spin.setRange(1, 12)
        self.max_consecutive_spin = QSpinBox()
        self.max_consecutive_spin.setRange(1, 6)
        self.run_rest_spin = QDoubleSpinBox()
        self.run_rest_spin.setRange(0.0, 600.0)
        self.run_rest_spin.setDecimals(1)
        self.long_run_every_spin = QSpinBox()
        self.long_run_every_spin.setRange(0, 10)
        self.long_run_rest_spin = QDoubleSpinBox()
        self.long_run_rest_spin.setRange(0.0, 900.0)
        self.long_run_rest_spin.setDecimals(1)

        self.quality_check_spin = QDoubleSpinBox()
        self.quality_check_spin.setRange(0.0, 600.0)
        self.quality_check_spin.setDecimals(1)
        self.practice_spin = QDoubleSpinBox()
        self.practice_spin.setRange(0.0, 1200.0)
        self.practice_spin.setDecimals(1)
        self.calib_open_spin = QDoubleSpinBox()
        self.calib_open_spin.setRange(0.0, 600.0)
        self.calib_open_spin.setDecimals(1)
        self.calib_closed_spin = QDoubleSpinBox()
        self.calib_closed_spin.setRange(0.0, 600.0)
        self.calib_closed_spin.setDecimals(1)
        self.calib_eye_spin = QDoubleSpinBox()
        self.calib_eye_spin.setRange(0.0, 600.0)
        self.calib_eye_spin.setDecimals(1)
        self.calib_blink_spin = QDoubleSpinBox()
        self.calib_blink_spin.setRange(0.0, 300.0)
        self.calib_blink_spin.setDecimals(1)
        self.calib_swallow_spin = QDoubleSpinBox()
        self.calib_swallow_spin.setRange(0.0, 300.0)
        self.calib_swallow_spin.setDecimals(1)
        self.calib_jaw_spin = QDoubleSpinBox()
        self.calib_jaw_spin.setRange(0.0, 300.0)
        self.calib_jaw_spin.setDecimals(1)
        self.calib_head_spin = QDoubleSpinBox()
        self.calib_head_spin.setRange(0.0, 300.0)
        self.calib_head_spin.setDecimals(1)

        self.idle_count_spin = QSpinBox()
        self.idle_count_spin.setRange(0, 10)
        self.idle_sec_spin = QDoubleSpinBox()
        self.idle_sec_spin.setRange(0.0, 600.0)
        self.idle_sec_spin.setDecimals(1)
        self.idle_prepare_count_spin = QSpinBox()
        self.idle_prepare_count_spin.setRange(0, 6)
        self.idle_prepare_spin = QDoubleSpinBox()
        self.idle_prepare_spin.setRange(0.0, 600.0)
        self.idle_prepare_spin.setDecimals(1)

        self.continuous_count_spin = QSpinBox()
        self.continuous_count_spin.setRange(0, 10)
        self.continuous_sec_spin = QDoubleSpinBox()
        self.continuous_sec_spin.setRange(0.0, 1200.0)
        self.continuous_sec_spin.setDecimals(1)
        self.cont_cmd_min_spin = QDoubleSpinBox()
        self.cont_cmd_min_spin.setRange(0.5, 30.0)
        self.cont_cmd_min_spin.setDecimals(1)
        self.cont_cmd_max_spin = QDoubleSpinBox()
        self.cont_cmd_max_spin.setRange(0.5, 30.0)
        self.cont_cmd_max_spin.setDecimals(1)
        self.cont_gap_min_spin = QDoubleSpinBox()
        self.cont_gap_min_spin.setRange(0.0, 30.0)
        self.cont_gap_min_spin.setDecimals(1)
        self.cont_gap_max_spin = QDoubleSpinBox()
        self.cont_gap_max_spin.setRange(0.0, 30.0)
        self.cont_gap_max_spin.setDecimals(1)

        self.artifact_types_edit = QLineEdit()
        self.reference_edit = QLineEdit()
        self.sleep_edit = QLineEdit()
        self.eyes_closed_for_gate_check = QCheckBox("闭眼静息加入门控负类")
        self.participant_state_combo = QComboBox()
        for key, label in PARTICIPANT_STATE_OPTIONS:
            self.participant_state_combo.addItem(label, key)
        self.caffeine_combo = QComboBox()
        for key, label in CAFFEINE_OPTIONS:
            self.caffeine_combo.addItem(label, key)
        self.exercise_combo = QComboBox()
        for key, label in RECENT_EXERCISE_OPTIONS:
            self.exercise_combo.addItem(label, key)

        session_group = QGroupBox("会话信息")
        session_form = QFormLayout(session_group)
        self._configure_form_layout(session_form)
        session_form.addRow("被试编号/名称", self.subject_edit)
        session_form.addRow("会话编号", self.session_edit)
        session_form.addRow("输出目录", output_widget)
        session_form.addRow("状态（主观）", self.participant_state_combo)
        session_form.addRow("咖啡/茶", self.caffeine_combo)
        session_form.addRow("刚运动", self.exercise_combo)
        session_form.addRow("睡眠备注", self.sleep_edit)
        session_form.addRow("参考电极设置", self.reference_edit)
        session_form.addRow("备注", self.notes_edit)

        board_group = QGroupBox("设备参数")
        board_form = QFormLayout(board_group)
        self._configure_form_layout(board_form)
        board_form.addRow("板卡类型", self.board_combo)
        board_form.addRow("串口", serial_widget)
        board_form.addRow("通道名称", self.channel_names_edit)
        board_form.addRow("通道位置", self.channel_positions_edit)

        timing_group = QGroupBox("试次流程")
        timing_form = QFormLayout(timing_group)
        self._configure_form_layout(timing_form)
        timing_form.addRow("每类试次数", self.trials_spin)
        timing_form.addRow("准备阶段（秒）", self.baseline_spin)
        timing_form.addRow("提示阶段（秒）", self.cue_spin)
        timing_form.addRow("想象阶段（秒）", self.imagery_spin)
        timing_form.addRow("休息阶段（秒）", self.iti_spin)
        timing_form.addRow("轮次数量", self.run_count_spin)
        timing_form.addRow("同类最多连续", self.max_consecutive_spin)
        timing_form.addRow("轮次休息（秒）", self.run_rest_spin)
        timing_form.addRow("每几轮加长休息", self.long_run_every_spin)
        timing_form.addRow("加长休息（秒）", self.long_run_rest_spin)
        timing_form.addRow("随机种子（自动）", self.seed_spin)
        timing_form.addRow("", self.save_epochs_check)
        timing_form.addRow("", self.separate_screen_check)

        calibration_group = QGroupBox("静息/伪迹/训练")
        calibration_form = QFormLayout(calibration_group)
        self._configure_form_layout(calibration_form)
        calibration_form.addRow("质量检查参考（秒）", self.quality_check_spin)
        calibration_form.addRow("睁眼静息（秒）", self.calib_open_spin)
        calibration_form.addRow("闭眼静息（秒）", self.calib_closed_spin)
        calibration_form.addRow("", self.eyes_closed_for_gate_check)
        calibration_form.addRow("眼动（秒）", self.calib_eye_spin)
        calibration_form.addRow("眨眼（秒）", self.calib_blink_spin)
        calibration_form.addRow("吞咽（秒）", self.calib_swallow_spin)
        calibration_form.addRow("咬牙（秒）", self.calib_jaw_spin)
        calibration_form.addRow("头动（秒）", self.calib_head_spin)
        calibration_form.addRow("想象训练（秒）", self.practice_spin)
        calibration_form.addRow("伪迹类型", self.artifact_types_edit)

        post_group = QGroupBox("无控制与连续模式")
        post_form = QFormLayout(post_group)
        self._configure_form_layout(post_form)
        post_form.addRow("无控制段数", self.idle_count_spin)
        post_form.addRow("无控制时长（秒）", self.idle_sec_spin)
        post_form.addRow("仅准备不执行段数", self.idle_prepare_count_spin)
        post_form.addRow("仅准备不执行时长（秒）", self.idle_prepare_spin)
        post_form.addRow("连续模式段数", self.continuous_count_spin)
        post_form.addRow("连续模式时长（秒）", self.continuous_sec_spin)
        post_form.addRow("命令最短时长（秒）", self.cont_cmd_min_spin)
        post_form.addRow("命令最长时长（秒）", self.cont_cmd_max_spin)
        post_form.addRow("命令间隔最短（秒）", self.cont_gap_min_spin)
        post_form.addRow("命令间隔最长（秒）", self.cont_gap_max_spin)

        control_group = QGroupBox("操作")
        control_layout = QGridLayout(control_group)
        control_layout.setHorizontalSpacing(10)
        control_layout.setVerticalSpacing(10)
        self.connect_button = QPushButton("连接设备")
        self.connect_button.setObjectName("btnConnect")
        self.start_button = QPushButton("开始采集")
        self.start_button.setObjectName("btnStart")
        self.pause_button = QPushButton("暂停")
        self.pause_button.setObjectName("btnPause")
        self.bad_trial_button = QPushButton("标记坏试次")
        self.bad_trial_button.setObjectName("btnWarn")
        self.stop_button = QPushButton("停止并保存")
        self.stop_button.setObjectName("btnStop")
        self.disconnect_button = QPushButton("断开设备")
        self.disconnect_button.setObjectName("btnDisconnect")
        self.connect_button.clicked.connect(self.connect_device)
        self.start_button.clicked.connect(self.start_session)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.bad_trial_button.clicked.connect(self.mark_bad_trial)
        self.stop_button.clicked.connect(self.stop_and_save)
        self.disconnect_button.clicked.connect(self.disconnect_device)
        control_layout.addWidget(self.connect_button, 0, 0)
        control_layout.addWidget(self.start_button, 0, 1)
        control_layout.addWidget(self.pause_button, 1, 0)
        control_layout.addWidget(self.bad_trial_button, 1, 1)
        control_layout.addWidget(self.stop_button, 2, 0)
        control_layout.addWidget(self.disconnect_button, 2, 1)
        control_layout.setColumnStretch(0, 1)
        control_layout.setColumnStretch(1, 1)
        self.config_groups = [
            session_group,
            board_group,
            timing_group,
            calibration_group,
            post_group,
            control_group,
        ]
        self.config_groups_grid = QGridLayout()
        self.config_groups_grid.setContentsMargins(0, 0, 0, 0)
        self.config_groups_grid.setHorizontalSpacing(8)
        self.config_groups_grid.setVerticalSpacing(8)
        layout.addLayout(self.config_groups_grid, stretch=1)
        self._refresh_config_group_layout(force=True)

        tip = QLabel(
            "连接设备后先在右侧原始波形面板完成实验员测试；若勾选“弹出受试者全屏提示窗”，点击“开始采集”后会直接切到受试者全屏。"
            "运行中快捷键：空格 暂停/继续，B 标记坏试次（连续模式下标记命令失败），Esc 停止并保存。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #475569; font-size: 12px; padding: 4px;")
        layout.addWidget(tip)
        layout.addStretch(1)
        return panel

    def _build_focus_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        cue_group = QGroupBox("任务提示预览")
        cue_layout = QVBoxLayout(cue_group)

        header_row = QHBoxLayout()
        self.phase_label = QLabel("等待开始")
        self.phase_label.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
        self.phase_label.setAlignment(Qt.AlignCenter)
        self.phase_label.setMinimumHeight(56)
        self.phase_label.setStyleSheet(
            "color: #FFFFFF; background: #64748B; border-radius: 14px; padding: 8px 14px;"
        )
        header_row.addWidget(self.phase_label, stretch=1)

        self.countdown_label = QLabel("剩余时间：--")
        self.countdown_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.countdown_label.setStyleSheet(
            "color: #334155; font-size: 18px; font-weight: bold; "
            "background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 12px; padding: 10px 14px;"
        )
        header_row.addWidget(self.countdown_label, stretch=0)
        cue_layout.addLayout(header_row)

        self.trial_banner_label = QLabel("当前试次：未开始")
        self.trial_banner_label.setStyleSheet(
            "color: #1E293B; font-size: 14px; background: #EEF2FF; border-radius: 10px; padding: 8px 10px;"
        )
        cue_layout.addWidget(self.trial_banner_label)

        self.cue_widget = CueIllustrationWidget()
        cue_layout.addWidget(self.cue_widget, stretch=1)

        self.instruction_label = QLabel("连接设备后点击“开始采集”。")
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setStyleSheet(
            "font-size: 15px; color: #0F172A; background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 12px; padding: 12px;"
        )
        cue_layout.addWidget(self.instruction_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        cue_layout.addWidget(self.progress_bar)

        footer_row = QHBoxLayout()
        self.progress_text = QLabel("总进度：0 / 0")
        self.progress_text.setStyleSheet("color: #475569; font-size: 12px;")
        footer_row.addWidget(self.progress_text, stretch=1)

        self.next_task_label = QLabel("下一任务：--")
        self.next_task_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.next_task_label.setStyleSheet("color: #475569; font-size: 12px;")
        footer_row.addWidget(self.next_task_label, stretch=0)
        cue_layout.addLayout(footer_row)

        layout.addWidget(cue_group, stretch=1)
        return panel

    def _build_session_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        protocol_group = QGroupBox("实验逻辑")
        protocol_layout = QVBoxLayout(protocol_group)
        protocol_text = QLabel(
            "操作员流程：连接设备后先在当前界面观察原始波形并调整参数，确认稳定后再开始正式采集。\n"
            "正式流程：静息/伪迹校准 → 想象训练 → 多轮次主任务 → "
            "无控制 → 连续仿真。\n"
            "连续仿真默认会按 MI run 边界拆开插入，不再全部堆到最后。\n"
            "主任务单试次固定顺序：注视(2秒) → 提示(1秒) → 想象(4秒) → 放松(2.5秒)。\n"
            "点击开始后，如勾选受试者全屏，当前屏幕会直接切到受试者提示界面。"
        )
        protocol_text.setWordWrap(True)
        protocol_text.setStyleSheet("font-size: 13px; color: #334155;")
        protocol_layout.addWidget(protocol_text)
        layout.addWidget(protocol_group)

        status_group = QGroupBox("当前状态")
        status_layout = QVBoxLayout(status_group)
        self.device_label = QLabel("设备：未连接")
        self.summary_label = QLabel("状态：等待连接设备")
        self.current_label = QLabel("当前任务：无")
        self.sequence_summary_label = QLabel("计划：尚未生成试次序列")
        self.accepted_label = QLabel("有效试次：0")
        self.rejected_label = QLabel("坏试次：0")
        for label in (
            self.device_label,
            self.summary_label,
            self.current_label,
            self.sequence_summary_label,
            self.accepted_label,
            self.rejected_label,
        ):
            label.setWordWrap(True)
            label.setStyleSheet("font-size: 13px; color: #1E293B;")
            status_layout.addWidget(label)
        layout.addWidget(status_group)

        preview_group = QGroupBox("连接后质量检查（EEG / 阻抗）")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_status_label = QLabel(
            "连接设备后先做质量检查：可在 EEG/阻抗模式切换，确认无掉线、坏道、饱和、异常漂移和接触不良。"
        )
        self.preview_status_label.setWordWrap(True)
        self.preview_status_label.setStyleSheet(
            "font-size: 12px; color: #475569; background: #F8FAFC; border-radius: 8px; padding: 6px 8px;"
        )
        preview_layout.addWidget(self.preview_status_label)

        preview_control_layout = QHBoxLayout()
        self.preview_mode_label = QLabel("Quality mode: waiting for device")
        self.preview_mode_label.setStyleSheet("font-size: 12px; color: #1E293B;")
        preview_control_layout.addWidget(self.preview_mode_label, 1)

        self.preview_to_eeg_button = QPushButton("EEG模式")
        self.preview_to_imp_button = QPushButton("阻抗模式")
        self.preview_prev_ch_button = QPushButton("上一通道")
        self.preview_next_ch_button = QPushButton("下一通道")
        self.preview_reset_button = QPushButton("恢复默认")
        for button in (
            self.preview_to_eeg_button,
            self.preview_to_imp_button,
            self.preview_prev_ch_button,
            self.preview_next_ch_button,
            self.preview_reset_button,
        ):
            button.setMinimumHeight(30)
            button.setEnabled(False)

        self.preview_to_eeg_button.clicked.connect(self.on_preview_to_eeg_clicked)
        self.preview_to_imp_button.clicked.connect(self.on_preview_to_imp_clicked)
        self.preview_prev_ch_button.clicked.connect(self.on_preview_prev_channel_clicked)
        self.preview_next_ch_button.clicked.connect(self.on_preview_next_channel_clicked)
        self.preview_reset_button.clicked.connect(self.on_preview_reset_clicked)

        preview_control_layout.addWidget(self.preview_to_eeg_button)
        preview_control_layout.addWidget(self.preview_to_imp_button)
        preview_control_layout.addWidget(self.preview_prev_ch_button)
        preview_control_layout.addWidget(self.preview_next_ch_button)
        preview_control_layout.addWidget(self.preview_reset_button)
        preview_layout.addLayout(preview_control_layout)

        self.preview_widget = RealtimeEEGPreviewWidget(window_seconds=5.0)
        preview_layout.addWidget(self.preview_widget, stretch=1)
        layout.addWidget(preview_group, stretch=1)

        order_group = QGroupBox("试次安排")
        order_layout = QVBoxLayout(order_group)
        self.sequence_hint_label = QLabel("灰色：未开始  蓝边：当前试次  实色：已完成  删除线：坏试次")
        self.sequence_hint_label.setWordWrap(True)
        self.sequence_hint_label.setStyleSheet("font-size: 12px; color: #64748B; background: #F8FAFC; border-radius: 8px; padding: 6px 8px;")
        order_layout.addWidget(self.sequence_hint_label)
        self.sequence_label = QLabel("当前还未生成试次顺序。")
        self.sequence_label.setWordWrap(True)
        self.sequence_label.setTextFormat(Qt.RichText)
        self.sequence_label.setStyleSheet(
            "font-size: 13px; color: #334155; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 10px;"
        )
        order_layout.addWidget(self.sequence_label)
        layout.addWidget(order_group, stretch=1)
        return panel

    def apply_default_values(self) -> None:
        self.subject_edit.setText(self.config["subject_id"])
        self.session_edit.setText(self.config["session_id"])
        self.output_edit.setText(self.config["output_root"])
        self.refresh_serial_ports()
        self.serial_combo.setCurrentText(self.config["serial_port"])
        self.channel_names_edit.setText(self.config["channel_names"])
        self.channel_positions_edit.setText(self.config["channel_positions"])
        self.trials_spin.setValue(int(self.config["trials_per_class"]))
        self.baseline_spin.setValue(float(self.config["baseline_sec"]))
        self.cue_spin.setValue(float(self.config["cue_sec"]))
        self.imagery_spin.setValue(float(self.config["imagery_sec"]))
        self.iti_spin.setValue(float(self.config["iti_sec"]))
        self.run_count_spin.setValue(int(self.config["run_count"]))
        self.max_consecutive_spin.setValue(int(self.config["max_consecutive_same_class"]))
        self.run_rest_spin.setValue(float(self.config["run_rest_sec"]))
        self.long_run_every_spin.setValue(int(self.config["long_run_rest_every"]))
        self.long_run_rest_spin.setValue(float(self.config["long_run_rest_sec"]))
        self.quality_check_spin.setValue(float(self.config["quality_check_sec"]))
        self.practice_spin.setValue(float(self.config["practice_sec"]))
        self.calib_open_spin.setValue(float(self.config["calibration_open_sec"]))
        self.calib_closed_spin.setValue(float(self.config["calibration_closed_sec"]))
        self.calib_eye_spin.setValue(float(self.config["calibration_eye_sec"]))
        self.calib_blink_spin.setValue(float(self.config["calibration_blink_sec"]))
        self.calib_swallow_spin.setValue(float(self.config["calibration_swallow_sec"]))
        self.calib_jaw_spin.setValue(float(self.config["calibration_jaw_sec"]))
        self.calib_head_spin.setValue(float(self.config["calibration_head_sec"]))
        self.idle_count_spin.setValue(int(self.config["idle_block_count"]))
        self.idle_sec_spin.setValue(float(self.config["idle_block_sec"]))
        self.idle_prepare_count_spin.setValue(int(self.config["idle_prepare_block_count"]))
        self.idle_prepare_spin.setValue(float(self.config["idle_prepare_sec"]))
        self.continuous_count_spin.setValue(int(self.config["continuous_block_count"]))
        self.continuous_sec_spin.setValue(float(self.config["continuous_block_sec"]))
        self.cont_cmd_min_spin.setValue(float(self.config["continuous_command_min_sec"]))
        self.cont_cmd_max_spin.setValue(float(self.config["continuous_command_max_sec"]))
        self.cont_gap_min_spin.setValue(float(self.config["continuous_gap_min_sec"]))
        self.cont_gap_max_spin.setValue(float(self.config["continuous_gap_max_sec"]))
        artifact_types_value = self.config["artifact_types"]
        if isinstance(artifact_types_value, (list, tuple)):
            artifact_types_text = ",".join(str(item).strip() for item in artifact_types_value if str(item).strip())
        else:
            artifact_types_text = str(artifact_types_value)
        self.artifact_types_edit.setText(artifact_types_text)
        self.eyes_closed_for_gate_check.setChecked(bool(self.config.get("include_eyes_closed_rest_in_gate_neg", False)))
        self.reference_edit.setText(str(self.config["reference_mode"]))
        self.sleep_edit.setText(str(self.config["sleep_note"]))
        for idx in range(self.participant_state_combo.count()):
            if str(self.participant_state_combo.itemData(idx)) == str(self.config["participant_state"]):
                self.participant_state_combo.setCurrentIndex(idx)
                break
        for idx in range(self.caffeine_combo.count()):
            if str(self.caffeine_combo.itemData(idx)) == str(self.config["caffeine_intake"]):
                self.caffeine_combo.setCurrentIndex(idx)
                break
        for idx in range(self.exercise_combo.count()):
            if str(self.exercise_combo.itemData(idx)) == str(self.config["recent_exercise"]):
                self.exercise_combo.setCurrentIndex(idx)
                break
        configured_seed = int(self.config.get("random_seed", 0))
        if configured_seed < self.seed_spin.minimum() or configured_seed > self.seed_spin.maximum():
            configured_seed = self._generate_session_seed()
        self.seed_spin.setValue(configured_seed)
        self.save_epochs_check.setChecked(bool(self.config["save_epochs_npz"]))
        self.separate_screen_check.setChecked(bool(self.config.get("use_separate_participant_screen", True)))
        self.use_separate_participant_screen = bool(self.separate_screen_check.isChecked())
        self.notes_edit.setPlainText(self.config["notes"])
        for index in range(self.board_combo.count()):
            if int(self.board_combo.itemData(index)) == int(self.config["board_id"]):
                self.board_combo.setCurrentIndex(index)
                break
        self.update_button_states()

    def refresh_serial_ports(self) -> None:
        selected = self.serial_combo.currentText().strip()
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
            self.serial_combo.setCurrentText("COM3")
        self.serial_combo.blockSignals(False)

    def refresh_board_input_state(self) -> None:
        synthetic = getattr(BoardIds, "SYNTHETIC_BOARD", None)
        is_synthetic = synthetic is not None and int(self.current_board_id()) == int(synthetic.value)
        can_edit = self.board_combo.isEnabled()
        self.serial_combo.setEnabled(can_edit and not is_synthetic)
        self.refresh_ports_button.setEnabled(can_edit and not is_synthetic)
        line_edit = self.serial_combo.lineEdit()
        if line_edit is not None:
            line_edit.setPlaceholderText("演示模式不需要串口" if is_synthetic else "例如 COM3")

    def current_board_id(self) -> int:
        return int(self.board_combo.currentData())

    def board_display_name(self) -> str:
        return str(self.board_combo.currentText())

    def browse_output_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_edit.text().strip() or str(PROJECT_ROOT))
        if selected:
            self.output_edit.setText(selected)

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.statusBar().showMessage(message, 4000)

    def show_error(self, message: str) -> None:
        if self.operator_hidden_for_session:
            self.restore_operator_window()
        self.log(message)
        QMessageBox.critical(self, "错误", message)

    def collect_settings(self) -> SessionSettings:
        try:
            channel_names = parse_channel_names(self.channel_names_edit.text(), expected_count=None)
        except Exception as error:
            raise ValueError(f"通道名称配置有误：{error}") from error
        if not channel_names:
            raise ValueError("通道名称不能为空。")

        try:
            channel_positions = parse_channel_positions(self.channel_positions_edit.text(), expected_count=len(channel_names))
        except Exception as error:
            raise ValueError(f"通道位置配置有误：{error}") from error
        if list(channel_names) != list(DEFAULT_CHANNEL_NAMES):
            raise ValueError(f"通道名称必须固定为：{', '.join(DEFAULT_CHANNEL_NAMES)}")
        if list(channel_positions) != list(range(len(DEFAULT_CHANNEL_NAMES))):
            raise ValueError("通道位置必须固定为 0,1,2,3,4,5,6,7。")

        serial_port = self.serial_combo.currentText().strip()
        board_id = self.current_board_id()
        synthetic = getattr(BoardIds, "SYNTHETIC_BOARD", None)
        is_synthetic = synthetic is not None and int(board_id) == int(synthetic.value)
        if not is_synthetic and not serial_port:
            raise ValueError("当前板卡需要串口，请先选择有效串口。")

        output_root = self.output_edit.text().strip() or str(PROJECT_ROOT / "datasets" / "custom_mi")
        try:
            Path(output_root).mkdir(parents=True, exist_ok=True)
        except Exception as error:
            raise ValueError(f"输出目录不可用：{error}") from error

        continuous_command_min_sec = float(self.cont_cmd_min_spin.value())
        continuous_command_max_sec = float(self.cont_cmd_max_spin.value())
        if continuous_command_max_sec < continuous_command_min_sec:
            raise ValueError("连续模式命令最长时长不能小于最短时长。")
        continuous_gap_min_sec = float(self.cont_gap_min_spin.value())
        continuous_gap_max_sec = float(self.cont_gap_max_spin.value())
        if continuous_gap_max_sec < continuous_gap_min_sec:
            raise ValueError("连续模式间隔最长时长不能小于最短时长。")
        run_count = int(self.run_count_spin.value())
        continuous_block_count = int(self.continuous_count_spin.value())
        continuous_block_sec = float(self.continuous_sec_spin.value())
        if continuous_block_count > 0 and continuous_block_sec > 0 and continuous_block_count > run_count:
            raise ValueError("连续模式段数不能大于 MI run 数。当前流程每个 run 边界最多插入一段连续模式。")
        if continuous_block_count > 0 and continuous_block_sec > 0 and continuous_block_sec < continuous_command_min_sec:
            raise ValueError("连续模式时长不能小于命令最短时长，否则无法生成任何连续命令。")

        raw_artifact_types = self.artifact_types_edit.text().strip()
        artifact_types = normalize_artifact_types(raw_artifact_types)
        if not artifact_types:
            artifact_types = list(DEFAULT_ARTIFACT_TYPES)

        subject_token = self.subject_edit.text().strip()
        if not subject_token:
            raise ValueError("被试编号/名称不能为空。")

        return SessionSettings(
            subject_id=subject_token,
            session_id=self.session_edit.text().strip() or datetime.now().strftime("%Y%m%d_%H%M%S"),
            output_root=output_root,
            board_id=board_id,
            serial_port=serial_port,
            channel_names=channel_names,
            channel_positions=channel_positions,
            trials_per_class=int(self.trials_spin.value()),
            baseline_sec=float(self.baseline_spin.value()),
            cue_sec=float(self.cue_spin.value()),
            imagery_sec=float(self.imagery_spin.value()),
            iti_sec=float(self.iti_spin.value()),
            run_count=run_count,
            max_consecutive_same_class=int(self.max_consecutive_spin.value()),
            run_rest_sec=float(self.run_rest_spin.value()),
            long_run_rest_every=int(self.long_run_every_spin.value()),
            long_run_rest_sec=float(self.long_run_rest_spin.value()),
            quality_check_sec=float(self.quality_check_spin.value()),
            practice_sec=float(self.practice_spin.value()),
            calibration_open_sec=float(self.calib_open_spin.value()),
            calibration_closed_sec=float(self.calib_closed_spin.value()),
            calibration_eye_sec=float(self.calib_eye_spin.value()),
            calibration_blink_sec=float(self.calib_blink_spin.value()),
            calibration_swallow_sec=float(self.calib_swallow_spin.value()),
            calibration_jaw_sec=float(self.calib_jaw_spin.value()),
            calibration_head_sec=float(self.calib_head_spin.value()),
            idle_block_count=int(self.idle_count_spin.value()),
            idle_block_sec=float(self.idle_sec_spin.value()),
            idle_prepare_block_count=int(self.idle_prepare_count_spin.value()),
            idle_prepare_sec=float(self.idle_prepare_spin.value()),
            continuous_block_count=continuous_block_count,
            continuous_block_sec=continuous_block_sec,
            continuous_command_min_sec=continuous_command_min_sec,
            continuous_command_max_sec=continuous_command_max_sec,
            continuous_gap_min_sec=continuous_gap_min_sec,
            continuous_gap_max_sec=continuous_gap_max_sec,
            include_eyes_closed_rest_in_gate_neg=bool(self.eyes_closed_for_gate_check.isChecked()),
            artifact_types=list(artifact_types),
            reference_mode=self.reference_edit.text().strip(),
            participant_state=str(self.participant_state_combo.currentData()),
            caffeine_intake=str(self.caffeine_combo.currentData()),
            recent_exercise=str(self.exercise_combo.currentData()),
            sleep_note=self.sleep_edit.text().strip(),
            random_seed=int(self.seed_spin.value()),
            save_epochs_npz=bool(self.save_epochs_check.isChecked()),
            operator="",
            notes=self.notes_edit.toPlainText().strip(),
            board_name=self.board_display_name(),
        )

    def on_worker_status(self, message: str) -> None:
        self.log(message)

    def on_worker_error(self, message: str) -> None:
        self.show_error(message)

    def set_config_enabled(self, enabled: bool) -> None:
        widgets = [
            self.subject_edit,
            self.session_edit,
            self.output_edit,
            self.board_combo,
            self.serial_combo,
            self.refresh_ports_button,
            self.channel_names_edit,
            self.channel_positions_edit,
            self.trials_spin,
            self.baseline_spin,
            self.cue_spin,
            self.imagery_spin,
            self.iti_spin,
            self.run_count_spin,
            self.max_consecutive_spin,
            self.run_rest_spin,
            self.long_run_every_spin,
            self.long_run_rest_spin,
            self.quality_check_spin,
            self.practice_spin,
            self.calib_open_spin,
            self.calib_closed_spin,
            self.calib_eye_spin,
            self.calib_blink_spin,
            self.calib_swallow_spin,
            self.calib_jaw_spin,
            self.calib_head_spin,
            self.idle_count_spin,
            self.idle_sec_spin,
            self.idle_prepare_count_spin,
            self.idle_prepare_spin,
            self.continuous_count_spin,
            self.continuous_sec_spin,
            self.cont_cmd_min_spin,
            self.cont_cmd_max_spin,
            self.cont_gap_min_spin,
            self.cont_gap_max_spin,
            self.artifact_types_edit,
            self.eyes_closed_for_gate_check,
            self.reference_edit,
            self.participant_state_combo,
            self.caffeine_combo,
            self.exercise_combo,
            self.sleep_edit,
            self.seed_spin,
            self.save_epochs_check,
            self.separate_screen_check,
            self.notes_edit,
        ]
        for widget in widgets:
            widget.setEnabled(enabled)
        self.refresh_board_input_state()

    def _class_ui_name(self, class_name: str | None) -> str:
        if not class_name:
            return ""
        if class_name in CLASS_UI_NAMES:
            return CLASS_UI_NAMES[class_name]
        info = CLASS_LOOKUP.get(class_name, {})
        return str(info.get("short_name") or info.get("display_name") or class_name)

    def _class_prep_hint(self, class_name: str | None) -> str:
        name = self._class_ui_name(class_name)
        return f"请看图标，准备进行{name}运动想象。保持身体不动。"

    def _class_imagery_hint(self, class_name: str | None) -> str:
        if not class_name:
            return "请保持想象状态，不要做真实动作。"
        return CLASS_UI_IMAGERY_HINTS.get(class_name, f"持续想象{self._class_ui_name(class_name)}运动，不要做真实动作。")

    def _phase_texts(self, phase: str, class_name: str | None) -> tuple[str, str]:
        if phase == "idle":
            return "等待开始", "连接设备后先观察原始波形并确认稳定，再点击“开始采集”。"
        if phase == "quality_check":
            return "质量检查", "观察 30-60 秒原始波形，确认无掉线、坏道、饱和和异常漂移。"
        if phase == "calibration_open":
            return "睁眼静息", "注视中央十字，不做任何运动想象，保持自然眨眼。"
        if phase == "calibration_closed":
            return "闭眼静息", "闭眼放松，不做运动想象。"
        if phase == "calibration_eye_move":
            return "眼动校准", "按提示做左右/上下眼动，不转头。"
        if phase == "calibration_blink":
            return "眨眼伪迹", "按节奏眨眼，包含慢眨与快眨。"
        if phase == "calibration_swallow":
            return "吞咽伪迹", "自然吞咽，不做运动想象。"
        if phase == "calibration_jaw":
            return "咬牙伪迹", "轻微咬牙，避免大幅头动。"
        if phase == "calibration_head":
            return "头动伪迹", "轻微左右/上下摆头，动作保持可控。"
        if phase == "practice":
            return "想象训练", "强调本体感觉想象：想象肌肉发力和关节感觉，而不是视觉动画。"
        if phase == "baseline":
            if self.current_trial is not None:
                return "准备阶段", f"请注视中央十字并保持稳定，下一步将提示：{self.current_trial.display_name}。"
            return "准备阶段", "请注视中央十字，保持放松稳定，尽量减少眨眼和吞咽。"
        if phase == "iti":
            return "休息恢复", "放空当前想象内容，等待下一轮任务提示。"
        if phase == "run_rest":
            return "轮次间休息", "请放松 1-3 分钟，可口头反馈哪个类别更难想象。"
        if phase == "idle_block":
            return "无控制", "保持注意但不要做运动想象，自然眨眼即可。"
        if phase == "idle_prepare":
            return "准备但不执行", "屏幕会提示准备控制，但你需要保持无控制，不执行运动想象。"
        if phase == "continuous":
            if self.current_continuous_prompt is not None:
                label = str(self.current_continuous_prompt.get("class_label", ""))
                if label == "no_control":
                    return "连续仿真", "当前命令：无控制（保持注意，不执行运动想象）。"
                return "连续仿真", f"当前命令：{self._class_ui_name(label)}（持续 3-6 秒）。"
            return "连续仿真", "等待下一条随机命令。"
        if phase == "paused":
            return "已暂停", "恢复后将从当前阶段继续。"
        if class_name in CLASS_LOOKUP:
            name = self._class_ui_name(class_name)
            if phase == "cue":
                return f"任务提示：{name}", self._class_prep_hint(class_name)
            if phase == "imagery":
                return f"请想象：{name}", self._class_imagery_hint(class_name)
        return PHASE_LABELS.get(phase, phase), ""

    def _participant_prompt(self) -> tuple[str, str, str, str | None]:
        if self.session_paused:
            return ("已暂停", "已暂停", "请保持稳定，等待继续。", None)
        if self.current_phase == "quality_check":
            return ("质量检查", "质量检查", "请保持静止，等待操作者确认信号质量。", None)
        if self.current_phase == "calibration_open":
            return ("静息", "睁眼静息", "注视中央十字，不做运动想象。", None)
        if self.current_phase == "calibration_closed":
            return ("静息", "闭眼静息", "闭眼并保持放松，不做运动想象。", None)
        if self.current_phase == "calibration_eye_move":
            return ("伪迹校准", "眼动", "按提示只动眼睛，不要转头。", None)
        if self.current_phase == "calibration_blink":
            return ("伪迹校准", "眨眼", "按节奏眨眼。", None)
        if self.current_phase == "calibration_swallow":
            return ("伪迹校准", "吞咽", "自然吞咽，不做想象。", None)
        if self.current_phase == "calibration_jaw":
            return ("伪迹校准", "咬牙", "轻微咬牙。", None)
        if self.current_phase == "calibration_head":
            return ("伪迹校准", "头动", "轻微摆头。", None)
        if self.current_phase == "practice":
            return ("训练", "想象训练", "想象真实发力和关节感觉，不要做真实动作。", None)
        if self.current_phase == "baseline":
            return ("静息", "静息", "请注视中央标记，保持放松与稳定。", None)
        if self.current_phase == "iti":
            return ("静息", "静息", "请放空当前想象，等待下一个试次。", None)
        if self.current_phase == "run_rest":
            return ("休息", "轮次间休息", "请放松并保持注意，稍后继续。", None)
        if self.current_phase == "idle_block":
            return ("无控制", "无控制", "保持注意，不要执行任何运动想象。", None)
        if self.current_phase == "idle_prepare":
            return ("无控制", "准备但不执行", "看到准备提示时也不要执行运动想象。", None)
        if self.current_phase == "continuous":
            if self.current_continuous_prompt is not None:
                label = str(self.current_continuous_prompt.get("class_label", ""))
                if label == "no_control":
                    return ("连续模式", "无控制", "保持注意，不执行运动想象。", None)
                return ("连续模式", f"命令：{self._class_ui_name(label)}", "持续执行该想象直到下一条命令。", label)
            return ("连续模式", "等待命令", "请等待随机命令。", None)
        if self.current_phase == "paused":
            return ("已暂停", "已暂停", "请保持稳定，等待继续。", None)
        if self.current_phase == "idle":
            return ("等待开始", "等待开始", "请等待实验开始。", None)
        if self.current_trial is not None:
            class_name = self.current_trial.class_name
            name = self._class_ui_name(class_name)
            if self.current_phase == "cue":
                return ("任务提示", f"准备：{name}", self._class_prep_hint(class_name), class_name)
            if self.current_phase == "imagery":
                return ("开始想象", f"想象：{name}", self._class_imagery_hint(class_name), class_name)
        return ("等待开始", "等待开始", "请等待实验开始。", None)

    def _participant_countdown_text(self) -> str:
        if self.current_phase == "idle":
            return "--"
        if self.session_paused:
            return f"{self.remaining_phase_sec:.1f}s"
        return f"{max(0.0, self.remaining_phase_sec):.1f}s"

    def _sync_participant_display(self) -> None:
        if not self.use_separate_participant_screen or self.participant_window is None:
            return
        stage_text, title, subtitle, class_name = self._participant_prompt()
        self.participant_window.set_prompt(
            phase=self.current_phase if not self.session_paused else "paused",
            class_name=class_name,
            stage_text=stage_text,
            title=title,
            subtitle=subtitle,
            countdown_text=self._participant_countdown_text(),
        )

    def show_participant_display(self) -> None:
        if not self.use_separate_participant_screen:
            return
        self._sync_participant_display()
        self.participant_window.showFullScreen()
        self.participant_window.raise_()
        self.participant_window.activateWindow()
        self.operator_hidden_for_session = True
        self.hide()

    def restore_operator_window(self) -> None:
        if not self.use_separate_participant_screen:
            return
        self.operator_hidden_for_session = False
        self.participant_window.hide()
        if not self.isVisible():
            self.showNormal()
        self.raise_()
        self.activateWindow()

    def _resolve_phase_accent(self, phase: str, class_name: str | None) -> str:
        if class_name in CLASS_LOOKUP and phase in {"cue", "imagery"}:
            return str(CLASS_LOOKUP[class_name]["color"])
        return PHASE_ACCENT_COLORS.get(phase, "#64748B")

    def _apply_phase_theme(self, phase: str, class_name: str | None) -> None:
        accent = self._resolve_phase_accent(phase, class_name)
        self.phase_label.setStyleSheet(
            f"color: #FFFFFF; background: {accent}; border-radius: 14px; padding: 8px 14px;"
        )
        self.countdown_label.setStyleSheet(
            f"color: #1E293B; font-size: 18px; font-weight: bold; "
            f"background: #F8FAFC; border: 1px solid {accent}; border-radius: 12px; padding: 10px 14px;"
        )
        self.instruction_label.setStyleSheet(
            f"font-size: 15px; color: #0F172A; background: #FFFFFF; "
            f"border: 1px solid {accent}; border-radius: 12px; padding: 12px;"
        )
        self.progress_bar.setStyleSheet(
            f"QProgressBar {{border: 1px solid #CBD5E1; border-radius: 10px; text-align: center; "
            f"background: #F8FAFC; color: #1E293B;}} "
            f"QProgressBar::chunk {{background: {accent}; border-radius: 9px;}}"
        )

    def _refresh_phase_display(self) -> None:
        class_name = None if self.current_trial is None else self.current_trial.class_name
        title, subtitle = self._phase_texts(self.current_phase, class_name)
        self.phase_label.setText(PHASE_LABELS.get(self.current_phase, self.current_phase))
        self.instruction_label.setText(subtitle or "请保持稳定并等待下一步提示。")
        self._apply_phase_theme(self.current_phase if not self.session_paused else "paused", class_name)
        self.cue_widget.set_state(
            phase=self.current_phase if not self.session_paused else "paused",
            class_name=class_name,
            title=title,
            subtitle=subtitle,
        )
        self._update_preview_status()
        self._update_trial_meta_labels()
        self._sync_participant_display()

    def _preview_channel_count(self) -> int:
        if self.device_info is not None:
            rows = self.device_info.get("selected_rows", [])
            if isinstance(rows, list) and rows:
                return max(1, int(len(rows)))
        if self.preview_widget is not None and self.preview_widget.channel_names:
            return max(1, int(len(self.preview_widget.channel_names)))
        return max(1, len(DEFAULT_CHANNEL_NAMES))

    def _set_preview_mode_label(self) -> None:
        if self.preview_mode_label is None:
            return
        if self.device_info is None:
            self.preview_mode_label.setText("Quality mode: waiting for device")
            return
        if self.preview_mode == RealtimeEEGPreviewWidget.MODE_IMPEDANCE:
            self.preview_mode_label.setText(f"Quality mode: IMPEDANCE (CH{self.preview_impedance_channel})")
        else:
            self.preview_mode_label.setText("Quality mode: EEG (filtered display only; saving stays raw)")

    def _apply_preview_mode_locally(self, mode: str, *, channel: int | None = None) -> None:
        normalized = (
            RealtimeEEGPreviewWidget.MODE_IMPEDANCE
            if str(mode).strip().upper().startswith("IMP")
            else RealtimeEEGPreviewWidget.MODE_EEG
        )
        channel_count = self._preview_channel_count()
        target_channel = self.preview_impedance_channel if channel is None else int(channel)
        target_channel = int(np.clip(target_channel, 1, channel_count))
        self.preview_mode = normalized
        self.preview_impedance_channel = target_channel
        if self.preview_widget is not None:
            self.preview_widget.set_quality_mode(normalized, impedance_channel=target_channel)
        self._set_preview_mode_label()
        self._update_preview_status()

    def _switch_preview_mode(
        self,
        mode: str,
        *,
        channel: int | None = None,
        reset_default: bool = False,
        show_error_dialog: bool = True,
        write_log: bool = True,
    ) -> bool:
        normalized = (
            RealtimeEEGPreviewWidget.MODE_IMPEDANCE
            if str(mode).strip().upper().startswith("IMP")
            else RealtimeEEGPreviewWidget.MODE_EEG
        )
        channel_count = self._preview_channel_count()
        target_channel = self.preview_impedance_channel if channel is None else int(channel)
        target_channel = int(np.clip(target_channel, 1, channel_count))

        if self.worker is None or self.device_info is None:
            self._apply_preview_mode_locally(normalized, channel=target_channel)
            return False

        ok, message = self.worker.switch_quality_mode_sync(
            target_mode=normalized,
            target_channel=target_channel,
            reset_default=bool(reset_default),
        )
        if not ok:
            if show_error_dialog:
                self.show_error(message)
            else:
                self.log(message)
            return False

        self._apply_preview_mode_locally(normalized, channel=target_channel)
        if write_log:
            if normalized == RealtimeEEGPreviewWidget.MODE_IMPEDANCE:
                self.log(f"质量检查模式 -> 阻抗模式（CH{target_channel}）")
            elif bool(reset_default):
                self.log("质量检查模式 -> EEG（已恢复默认设置）")
            else:
                self.log("质量检查模式 -> EEG")
        return True

    def on_preview_to_eeg_clicked(self) -> None:
        self._switch_preview_mode(RealtimeEEGPreviewWidget.MODE_EEG, channel=self.preview_impedance_channel)
        self.update_button_states()

    def on_preview_to_imp_clicked(self) -> None:
        self._switch_preview_mode(RealtimeEEGPreviewWidget.MODE_IMPEDANCE, channel=self.preview_impedance_channel)
        self.update_button_states()

    def on_preview_prev_channel_clicked(self) -> None:
        channel_count = self._preview_channel_count()
        self.preview_impedance_channel -= 1
        if self.preview_impedance_channel < 1:
            self.preview_impedance_channel = channel_count
        if self.preview_mode == RealtimeEEGPreviewWidget.MODE_IMPEDANCE:
            switched = self._switch_preview_mode(
                RealtimeEEGPreviewWidget.MODE_IMPEDANCE,
                channel=self.preview_impedance_channel,
                write_log=False,
            )
            if switched:
                self.log(f"阻抗检测通道 -> CH{self.preview_impedance_channel}")
        else:
            self._apply_preview_mode_locally(self.preview_mode, channel=self.preview_impedance_channel)
        self.update_button_states()

    def on_preview_next_channel_clicked(self) -> None:
        channel_count = self._preview_channel_count()
        self.preview_impedance_channel += 1
        if self.preview_impedance_channel > channel_count:
            self.preview_impedance_channel = 1
        if self.preview_mode == RealtimeEEGPreviewWidget.MODE_IMPEDANCE:
            switched = self._switch_preview_mode(
                RealtimeEEGPreviewWidget.MODE_IMPEDANCE,
                channel=self.preview_impedance_channel,
                write_log=False,
            )
            if switched:
                self.log(f"阻抗检测通道 -> CH{self.preview_impedance_channel}")
        else:
            self._apply_preview_mode_locally(self.preview_mode, channel=self.preview_impedance_channel)
        self.update_button_states()

    def on_preview_reset_clicked(self) -> None:
        self._switch_preview_mode(
            RealtimeEEGPreviewWidget.MODE_EEG,
            channel=self.preview_impedance_channel,
            reset_default=True,
        )
        self.update_button_states()

    def _update_preview_status(self) -> None:
        if self.preview_status_label is None:
            return

        suggested_seconds = 0.0 if not hasattr(self, "quality_check_spin") else float(self.quality_check_spin.value())
        if self.preview_mode == RealtimeEEGPreviewWidget.MODE_IMPEDANCE:
            mode_hint = f"当前模式：阻抗模式（CH{self.preview_impedance_channel}，原始波形，不滤波）。"
        else:
            mode_hint = "当前模式：EEG模式（仅用于可视化滤波；保存始终为原始数据）。"

        if self.device_info is None:
            self.preview_status_label.setText(
                f"连接设备后开始质量检查。{mode_hint} 建议先观察约 {suggested_seconds:.0f} 秒，确认无掉线、坏道、饱和和异常漂移。"
            )
            self._set_preview_mode_label()
            return

        if self.current_phase == "quality_check":
            self.preview_status_label.setText(
                f"质量检查中：请观察波形稳定性并检查接触质量。{mode_hint}"
            )
            self._set_preview_mode_label()
            return

        if self.session_running:
            self.preview_status_label.setText(
                f"正式采集中：当前面板仅用于辅助监看。{mode_hint}"
            )
            self._set_preview_mode_label()
            return

        self.preview_status_label.setText(
            f"设备已连接：先完成连接后质量检查（约 {suggested_seconds:.0f} 秒）再开始采集。{mode_hint}"
        )
        self._set_preview_mode_label()

    def _start_formal_protocol(self) -> None:
        self.record_event("calibration_start")
        if self.marker_failure_active:
            return
        self._start_next_calibration_step()
        if self.marker_failure_active:
            return
        if self.use_separate_participant_screen:
            self.show_participant_display()
        else:
            self.participant_window.hide()

    def _update_trial_meta_labels(self) -> None:
        total = len(self.sequence)
        if total <= 0:
            self.trial_banner_label.setText("当前试次：未开始")
            self.next_task_label.setText("下一任务：--")
            self.sequence_summary_label.setText("计划：尚未生成试次序列")
            return

        if self.current_trial is not None:
            current_text = f"当前试次：第 {self.current_trial.trial_id} / {total} 个 - {self.current_trial.display_name}"
        else:
            current_text = f"当前试次：未开始，共 {total} 个"
        self.trial_banner_label.setText(current_text)

        if self.current_phase == "baseline" and self.current_trial is not None:
            next_text = f"下一任务：即将提示 {self.current_trial.display_name}"
        elif self.current_trial_index + 1 < total:
            upcoming = self.sequence[self.current_trial_index + 1]
            next_text = f"下一任务：{CLASS_LOOKUP[upcoming]['display_name']}"
        else:
            next_text = "下一任务：当前已是最后一个试次"
        self.next_task_label.setText(next_text)

        rejected = sum(1 for trial in self.trial_records if not trial.accepted)
        self.sequence_summary_label.setText(
            f"计划：总试次 {total} | 已完成 {self.completed_trials} | 坏试次 {rejected}"
        )

    def _mark_incomplete_trial_if_needed(self) -> None:
        if self.current_trial is None or self.current_phase not in {"baseline", "cue", "imagery"}:
            return
        self.current_trial.accepted = False
        if not self.current_trial.note:
            self.current_trial.note = "当前会话手动提前结束"

    def _event_already_recorded(self, event_name: str, *, trial_id: int | None, class_name: str | None) -> bool:
        for event in reversed(self.event_log):
            if str(event.get("event_name", "")) != str(event_name):
                continue
            if trial_id is not None and int(event.get("trial_id", -1)) != int(trial_id):
                continue
            if class_name is not None and str(event.get("class_name", "")) != str(class_name):
                continue
            return True
        return False

    def connect_device(self) -> None:
        if self.worker_thread is not None:
            self.show_error("设备线程已经在运行，请不要重复连接。")
            return

        try:
            settings = self.collect_settings()
        except Exception as error:
            self.show_error(str(error))
            return

        self.device_label.setText("设备：正在连接…")
        self.summary_label.setText("状态：正在准备 BrainFlow 采集线程")
        self.current_label.setText("当前任务：无")
        self.log(f"开始连接设备：{settings.board_name}")

        self.worker_thread = QThread(self)
        self.worker = BoardCaptureWorker(
            board_id=settings.board_id,
            serial_port=settings.serial_port,
            channel_positions=settings.channel_positions,
            channel_names=settings.channel_names,
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.connection_ready.connect(self.on_connection_ready)
        self.worker.preview_data_ready.connect(self.on_preview_data_ready)
        self.worker.status_changed.connect(self.on_worker_status)
        self.worker.error_occurred.connect(self.on_worker_error)
        self.worker.session_data_ready.connect(self.on_session_data_ready)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.on_worker_thread_finished)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_stop_requested.connect(self.worker.request_stop, Qt.DirectConnection)

        self.update_button_states()
        self.set_config_enabled(False)
        self.worker_thread.start()

    def on_connection_ready(self, info: dict) -> None:
        self.device_info = dict(info)
        sampling_rate = float(info.get("sampling_rate", 0.0))
        channel_names = [str(item) for item in info.get("channel_names", [])]
        self.device_label.setText(
            f"设备：已连接 | 采样率 {sampling_rate:g} Hz | 通道 {', '.join(channel_names)}"
        )
        self.summary_label.setText("状态：设备已准备好，请先检查原始波形稳定性")
        self.current_label.setText("当前任务：实验员测试 / 等待开始")
        self.preview_mode = RealtimeEEGPreviewWidget.MODE_EEG
        self.preview_impedance_channel = 1
        if self.preview_widget is not None:
            self.preview_widget.configure_stream(sampling_rate=sampling_rate, channel_names=channel_names)
            self.preview_widget.set_quality_mode(self.preview_mode, impedance_channel=self.preview_impedance_channel)
        if self.worker is not None and self.worker.supports_impedance_mode():
            ok, message = self.worker.switch_quality_mode_sync(
                target_mode=RealtimeEEGPreviewWidget.MODE_EEG,
                target_channel=self.preview_impedance_channel,
                reset_default=True,
            )
            if not ok:
                self.log(message)
        self._set_preview_mode_label()
        self._update_preview_status()
        self._update_trial_meta_labels()
        self.log("设备连接完成。")
        self.update_button_states()

    def on_preview_data_ready(self, preview_chunk: object) -> None:
        if self.preview_widget is None:
            return
        try:
            preview_array = np.asarray(preview_chunk, dtype=np.float32)
        except Exception:
            return
        if preview_array.ndim != 2 or preview_array.size == 0:
            return
        self.preview_widget.append_chunk(preview_array)

    def update_button_states(self) -> None:
        connected = self.device_info is not None
        busy_connecting = self.worker_thread is not None and self.device_info is None and not self.waiting_for_save

        self.connect_button.setEnabled(not connected and not busy_connecting and not self.waiting_for_save)
        self.start_button.setEnabled(connected and not self.session_running and not self.waiting_for_save)
        self.pause_button.setEnabled(self.session_running and not self.waiting_for_save)
        self.pause_button.setText("继续" if self.session_paused else "暂停")
        if self.current_phase == "continuous":
            self.bad_trial_button.setText("标记命令失败")
        else:
            self.bad_trial_button.setText("标记坏试次")
        self.bad_trial_button.setEnabled(
            self.session_running
            and not self.session_paused
            and (
                (self.current_trial is not None and self.current_phase in {"cue", "imagery", "iti"})
                or (self.current_phase == "continuous" and self.current_continuous_prompt is not None)
            )
        )
        self.stop_button.setEnabled(self.session_running and self.worker_thread is not None and not self.waiting_for_save)
        self.disconnect_button.setEnabled(connected and not self.session_running and not self.waiting_for_save)

        preview_controls_enabled = connected and not self.session_running and not self.waiting_for_save
        supports_impedance = self.worker is not None and self.worker.supports_impedance_mode()
        channel_step_enabled = preview_controls_enabled and supports_impedance and self._preview_channel_count() > 1
        if self.preview_to_eeg_button is not None:
            self.preview_to_eeg_button.setEnabled(preview_controls_enabled)
        if self.preview_to_imp_button is not None:
            self.preview_to_imp_button.setEnabled(preview_controls_enabled and supports_impedance)
        if self.preview_prev_ch_button is not None:
            self.preview_prev_ch_button.setEnabled(channel_step_enabled)
        if self.preview_next_ch_button is not None:
            self.preview_next_ch_button.setEnabled(channel_step_enabled)
        if self.preview_reset_button is not None:
            self.preview_reset_button.setEnabled(preview_controls_enabled and supports_impedance)
        self._set_preview_mode_label()

        editable = self.worker_thread is None and not self.waiting_for_save
        self.set_config_enabled(editable)

    def _build_calibration_plan(self, settings: SessionSettings) -> list[dict[str, object]]:
        plan: list[dict[str, object]] = []

        def add_step(phase: str, start_event: str, end_event: str, duration: float) -> None:
            if duration <= 0:
                return
            plan.append(
                {
                    "phase": phase,
                    "start_event": start_event,
                    "end_event": end_event,
                    "duration": float(duration),
                }
            )

        add_step("calibration_open", "eyes_open_rest_start", "eyes_open_rest_end", settings.calibration_open_sec)
        add_step("calibration_closed", "eyes_closed_rest_start", "eyes_closed_rest_end", settings.calibration_closed_sec)
        if "eye_movement" in settings.artifact_types:
            add_step("calibration_eye_move", "eye_movement_block_start", "eye_movement_block_end", settings.calibration_eye_sec)
        if "blink" in settings.artifact_types:
            add_step("calibration_blink", "blink_block_start", "blink_block_end", settings.calibration_blink_sec)
        if "swallow" in settings.artifact_types:
            add_step("calibration_swallow", "swallow_block_start", "swallow_block_end", settings.calibration_swallow_sec)
        if "jaw" in settings.artifact_types:
            add_step("calibration_jaw", "jaw_block_start", "jaw_block_end", settings.calibration_jaw_sec)
        if "head_motion" in settings.artifact_types:
            add_step("calibration_head", "head_motion_block_start", "head_motion_block_end", settings.calibration_head_sec)
        return plan

    def _build_continuous_schedule(self, settings: SessionSettings) -> list[int]:
        block_count = max(0, int(settings.continuous_block_count))
        run_count = max(0, int(settings.run_count))
        if block_count <= 0 or run_count <= 0 or float(settings.continuous_block_sec) <= 0:
            return []
        if block_count == 1:
            return [run_count]

        schedule: list[int] = []
        for block_index in range(block_count):
            boundary = int(np.ceil(float((block_index + 1) * run_count) / float(block_count)))
            boundary = min(max(1, boundary), run_count)
            if not schedule or schedule[-1] != boundary:
                schedule.append(boundary)
        schedule.sort()
        return schedule

    def _play_imagery_tone(self, kind: str) -> None:
        if kind == "start":
            frequency = IMAGERY_START_TONE_HZ
            duration_ms = IMAGERY_START_TONE_MS
        else:
            frequency = IMAGERY_END_TONE_HZ
            duration_ms = IMAGERY_END_TONE_MS

        def _play() -> None:
            if winsound is not None:
                try:
                    winsound.Beep(frequency, duration_ms)
                    return
                except Exception:
                    pass
            app = QApplication.instance()
            if app is not None:
                QTimer.singleShot(0, app.beep)

        threading.Thread(target=_play, daemon=True).start()

    def _maybe_start_scheduled_continuous(self) -> bool:
        if self.current_settings is None:
            return False
        if self.next_continuous_schedule_index >= len(self.continuous_schedule_after_runs):
            return False
        target_run = int(self.continuous_schedule_after_runs[self.next_continuous_schedule_index])
        if int(self.current_run_index) != target_run:
            return False
        self.next_continuous_schedule_index += 1
        self.start_continuous_block()
        return True

    def _start_next_calibration_step(self) -> None:
        if self.current_settings is None:
            return
        self.calibration_step_index += 1
        if self.calibration_step_index >= len(self.calibration_plan):
            self.record_event("calibration_end")
            if self.marker_failure_active:
                return
            if self.current_settings.practice_sec > 0:
                self.record_event("practice_start")
                if self.marker_failure_active:
                    return
                self.enter_phase(
                    phase="practice",
                    duration_sec=self.current_settings.practice_sec,
                    class_name=None,
                    title="",
                    subtitle="",
                )
                self.current_label.setText("当前任务：想象训练")
            else:
                self._start_next_mi_run()
            return

        step = self.calibration_plan[self.calibration_step_index]
        self.record_event(str(step["start_event"]))
        if self.marker_failure_active:
            return
        self.enter_phase(
            phase=str(step["phase"]),
            duration_sec=float(step["duration"]),
            class_name=None,
            title="",
            subtitle="",
        )
        self.current_label.setText(f"当前任务：{PHASE_LABELS.get(str(step['phase']), str(step['phase']))}")

    def _run_rest_duration_for_completed_run(self, completed_run_index: int) -> float:
        if self.current_settings is None:
            return 0.0
        every = int(self.current_settings.long_run_rest_every)
        if every > 0 and completed_run_index % every == 0:
            return max(0.0, float(self.current_settings.long_run_rest_sec))
        return max(0.0, float(self.current_settings.run_rest_sec))

    def _start_next_mi_run(self) -> None:
        if self.current_settings is None:
            return
        if self.current_run_index >= int(self.current_settings.run_count):
            self.start_post_collection_blocks_or_finish()
            return

        self.current_run_index += 1
        self.current_run_trial_index = 0
        self.record_event("mi_run_start", run_index=self.current_run_index)
        if self.marker_failure_active:
            return
        self.summary_label.setText(
            f"状态：主任务采集中，轮次 {self.current_run_index}/{int(self.current_settings.run_count)}"
        )
        self.current_label.setText(f"当前任务：轮次 {self.current_run_index} 开始")
        self.start_next_trial()

    def start_post_collection_blocks_or_finish(self) -> None:
        if self.current_settings is None:
            self.finish_session_and_request_save(manual_stop=False)
            return
        if int(self.current_settings.idle_block_count) > 0 and float(self.current_settings.idle_block_sec) > 0:
            self.idle_block_index = 0
            self.start_idle_block()
            return
        if int(self.current_settings.idle_prepare_block_count) > 0 and float(self.current_settings.idle_prepare_sec) > 0:
            self.start_idle_prepare_block()
            return
        self.finish_session_and_request_save(manual_stop=False)

    def start_idle_block(self) -> None:
        if self.current_settings is None:
            return
        self.idle_block_index += 1
        self.record_event("idle_block_start", block_index=self.idle_block_index)
        if self.marker_failure_active:
            return
        self.enter_phase(
            phase="idle_block",
            duration_sec=float(self.current_settings.idle_block_sec),
            class_name=None,
            title="",
            subtitle="",
        )
        self.current_label.setText(f"当前任务：无控制第 {self.idle_block_index} 段")

    def _finish_idle_block(self) -> None:
        if self.current_settings is None:
            return
        self.record_event("idle_block_end", block_index=self.idle_block_index)
        if self.marker_failure_active:
            return
        if self.idle_block_index < int(self.current_settings.idle_block_count):
            self.start_idle_block()
            return
        if int(self.current_settings.idle_prepare_block_count) > 0 and float(self.current_settings.idle_prepare_sec) > 0:
            self.start_idle_prepare_block()
            return
        self.finish_session_and_request_save(manual_stop=False)

    def start_idle_prepare_block(self) -> None:
        if self.current_settings is None:
            return
        self.idle_prepare_block_index += 1
        self.record_event("idle_prepare_start", block_index=self.idle_prepare_block_index)
        if self.marker_failure_active:
            return
        self.enter_phase(
            phase="idle_prepare",
            duration_sec=float(self.current_settings.idle_prepare_sec),
            class_name=None,
            title="",
            subtitle="",
        )
        self.current_label.setText(f"当前任务：仅准备不执行第 {self.idle_prepare_block_index} 段")

    def _finish_idle_prepare_block(self) -> None:
        self.record_event("idle_prepare_end", block_index=self.idle_prepare_block_index)
        if self.marker_failure_active:
            return
        if (
            self.current_settings is not None
            and self.idle_prepare_block_index < int(self.current_settings.idle_prepare_block_count)
            and float(self.current_settings.idle_prepare_sec) > 0
        ):
            self.start_idle_prepare_block()
            return
        self.finish_session_and_request_save(manual_stop=False)

    def _build_continuous_prompt_plan(self, *, block_duration_sec: float) -> list[dict[str, object]]:
        if self.current_settings is None or block_duration_sec <= 0:
            return []
        rng = np.random.default_rng(int(self.current_settings.random_seed) + int(self.continuous_block_index))
        labels = list(CONTINUOUS_PROMPT_LABELS)
        command_min = float(self.current_settings.continuous_command_min_sec)
        command_max = float(self.current_settings.continuous_command_max_sec)
        gap_min = float(self.current_settings.continuous_gap_min_sec)
        gap_max = float(self.current_settings.continuous_gap_max_sec)
        current = 0.0
        prompts: list[dict[str, object]] = []
        prompt_index = 0
        while current < block_duration_sec:
            remaining_time = float(block_duration_sec - current)
            if remaining_time < command_min:
                break
            duration_high = min(command_max, remaining_time)
            duration = float(rng.uniform(command_min, duration_high))
            label_token = rng.choice(np.asarray(labels, dtype=object))
            label = str(label_token.item() if hasattr(label_token, "item") else label_token)
            prompt_index += 1
            prompts.append(
                {
                    "prompt_index": prompt_index,
                    "start_offset_sec": current,
                    "duration_sec": duration,
                    "end_offset_sec": current + duration,
                    "class_label": label,
                }
            )
            gap = float(rng.uniform(gap_min, gap_max))
            current = current + duration + gap
        return prompts

    def _close_current_continuous_prompt(self) -> None:
        if self.current_continuous_prompt is None:
            return
        prompt = dict(self.current_continuous_prompt)
        execution_success = int(bool(prompt.get("execution_success", 1)))
        self.record_event(
            "continuous_command_end",
            class_name=str(prompt["class_label"]),
            block_index=self.continuous_block_index,
            prompt_index=int(prompt["prompt_index"]),
            command_duration_sec=float(prompt["duration_sec"]),
            execution_success=execution_success,
        )
        if self.marker_failure_active:
            return
        self.current_continuous_prompt = None
        self._refresh_phase_display()

    def _update_continuous_prompt(self) -> None:
        if self.current_phase != "continuous":
            return
        elapsed = max(0.0, time.perf_counter() - self.phase_started_perf)
        if self.current_continuous_prompt is not None:
            if elapsed >= float(self.current_continuous_prompt["end_offset_sec"]):
                self._close_current_continuous_prompt()

        next_index = self.continuous_prompt_index + 1
        if next_index >= len(self.continuous_prompt_plan):
            return
        next_prompt = self.continuous_prompt_plan[next_index]
        if elapsed < float(next_prompt["start_offset_sec"]):
            return
        self.continuous_prompt_index = next_index
        self.current_continuous_prompt = dict(next_prompt)
        self.current_continuous_prompt["execution_success"] = 1
        label = str(next_prompt["class_label"])
        event_name = CONTINUOUS_PROMPT_EVENT_NAMES[label]
        self.record_event(
            event_name,
            class_name=label,
            block_index=self.continuous_block_index,
            prompt_index=int(next_prompt["prompt_index"]),
            command_duration_sec=float(next_prompt["duration_sec"]),
            execution_success=None,
        )
        if self.marker_failure_active:
            return
        self._refresh_phase_display()

    def start_continuous_block(self) -> None:
        if self.current_settings is None:
            return
        self.continuous_block_index += 1
        duration_sec = float(self.current_settings.continuous_block_sec)
        self.continuous_prompt_plan = self._build_continuous_prompt_plan(block_duration_sec=duration_sec)
        if not self.continuous_prompt_plan:
            self.continuous_block_index = max(0, self.continuous_block_index - 1)
            self.show_error("连续模式配置无法生成任何命令计划，请调整连续模式时长或命令时长后重新开始采集。")
            self.current_phase = "idle"
            self.current_continuous_prompt = None
            self.finish_session_and_request_save(manual_stop=True)
            return
        self.continuous_prompt_index = -1
        self.current_continuous_prompt = None
        self.record_event("continuous_block_start", block_index=self.continuous_block_index)
        if self.marker_failure_active:
            return
        self.enter_phase(
            phase="continuous",
            duration_sec=duration_sec,
            class_name=None,
            title="",
            subtitle="",
        )
        self.current_label.setText(f"当前任务：连续模式第 {self.continuous_block_index} 段")
        self._update_continuous_prompt()

    def _finish_continuous_block(self) -> None:
        self._close_current_continuous_prompt()
        if self.marker_failure_active:
            return
        self.record_event("continuous_block_end", block_index=self.continuous_block_index)
        if self.marker_failure_active:
            return
        if self._maybe_start_scheduled_continuous():
            return
        if self.current_settings is not None:
            if self.current_run_index < int(self.current_settings.run_count):
                self._start_next_mi_run()
                return
            self.start_post_collection_blocks_or_finish()
            return
        self.finish_session_and_request_save(manual_stop=False)

    def start_session(self) -> None:
        if self.device_info is None or self.worker is None:
            self.show_error("请先连接设备。")
            return
        if self.session_running or self.waiting_for_save:
            self.show_error("当前已有采集任务在运行或正在保存。")
            return
        if self.worker.supports_impedance_mode():
            switched = self._switch_preview_mode(
                RealtimeEEGPreviewWidget.MODE_EEG,
                channel=self.preview_impedance_channel,
                reset_default=True,
                show_error_dialog=True,
                write_log=False,
            )
            if not switched:
                return

        # Always use a fresh seed per session and persist it in settings/metadata.
        self.seed_spin.setValue(self._generate_session_seed())

        try:
            settings = self.collect_settings()
            sequence_by_run = [
                build_balanced_trial_sequence(
                    settings.trials_per_class,
                    settings.random_seed + run_offset,
                    settings.max_consecutive_same_class,
                )
                for run_offset in range(int(settings.run_count))
            ]
        except Exception as error:
            self.show_error(str(error))
            return

        if len(settings.channel_names) != len(self.device_info.get("selected_rows", [])):
            self.show_error("通道名称数量与连接时的通道数量不一致。请断开设备后重新连接。")
            return

        self.current_settings = settings
        self.use_separate_participant_screen = bool(self.separate_screen_check.isChecked())
        self.sequence_by_run = sequence_by_run
        self.sequence = [item for run_items in sequence_by_run for item in run_items]
        self.trials_per_run = int(settings.trials_per_class * len(CLASS_LOOKUP))
        self.current_run_index = 0
        self.current_run_trial_index = 0
        self.event_log = []
        self.trial_records = []
        self.current_trial_index = -1
        self.current_trial = None
        self.completed_trials = 0
        self.calibration_plan = self._build_calibration_plan(settings)
        self.calibration_step_index = -1
        self.idle_block_index = 0
        self.idle_prepare_block_index = 0
        self.continuous_block_index = 0
        self.continuous_schedule_after_runs = self._build_continuous_schedule(settings)
        self.next_continuous_schedule_index = 0
        self.continuous_prompt_plan = []
        self.continuous_prompt_index = -1
        self.current_continuous_prompt = None
        self.marker_failure_active = False
        self.marker_failure_message = ""
        self.session_running = True
        self.session_paused = False
        self.waiting_for_save = False
        self.capture_on_stop = True
        self.session_start_perf = time.perf_counter()

        self._update_sequence_label()
        self.refresh_trial_counters()
        if self.marker_failure_active:
            return
        self._update_progress_label()
        self._update_trial_meta_labels()

        self.summary_label.setText(f"状态：采集中，共 {len(self.sequence)} 个主任务试次")
        self.current_label.setText("当前任务：正在进入正式流程")
        self.log(
            f"开始采集：被试 {settings.subject_id} | 会话 {settings.session_id} | "
            f"轮次 {settings.run_count} | 每类 {settings.trials_per_class} 试次"
        )
        self.log(f"本次会话随机种子（自动生成）：{settings.random_seed}")
        if self.continuous_schedule_after_runs:
            schedule_text = "、".join(str(item) for item in self.continuous_schedule_after_runs)
            self.log(f"连续模式插入点：在完成 MI run {schedule_text} 后插入。")

        self.record_event("session_start")
        if self.marker_failure_active:
            return
        self._start_formal_protocol()
        self.update_button_states()

    def enter_phase(self, *, phase: str, duration_sec: float, class_name: str | None, title: str, subtitle: str) -> None:
        del title, subtitle
        self.current_phase = phase
        self.session_paused = False
        self.pause_started_perf = 0.0
        self.remaining_phase_sec = max(0.0, float(duration_sec))
        self.phase_started_perf = time.perf_counter()
        self.phase_deadline = time.perf_counter() + self.remaining_phase_sec
        self._refresh_phase_display()
        self.update_countdown_text()
        if self.remaining_phase_sec > 0:
            self.phase_timer.start()
        else:
            self.phase_timer.stop()
            QTimer.singleShot(0, self.on_phase_tick)
        self.update_button_states()

    def update_countdown_text(self) -> None:
        if self.current_phase == "idle":
            self.countdown_label.setText("剩余时间：--")
            self._sync_participant_display()
            return

        if self.session_paused:
            self.countdown_label.setText(f"已暂停，当前阶段剩余 {self.remaining_phase_sec:.1f} 秒")
            self._sync_participant_display()
            return

        remaining = max(0.0, self.phase_deadline - time.perf_counter())
        self.remaining_phase_sec = remaining
        self.countdown_label.setText(f"当前阶段剩余 {remaining:.1f} 秒")
        self._sync_participant_display()

    def on_phase_tick(self) -> None:
        if not self.session_running or self.session_paused:
            return

        if self.current_phase == "continuous":
            self._update_continuous_prompt()
            if self.marker_failure_active:
                return

        remaining = self.phase_deadline - time.perf_counter()
        if remaining > 0:
            self.remaining_phase_sec = remaining
            self.update_countdown_text()
            return

        self.phase_timer.stop()

        if self.current_phase == "quality_check":
            self.record_event("quality_check_end")
            if self.marker_failure_active:
                return
            self._start_formal_protocol()
            return

        if self.current_phase.startswith("calibration_"):
            if 0 <= self.calibration_step_index < len(self.calibration_plan):
                step = self.calibration_plan[self.calibration_step_index]
                self.record_event(str(step["end_event"]))
                if self.marker_failure_active:
                    return
            self._start_next_calibration_step()
            return

        if self.current_phase == "practice":
            self.record_event("practice_end")
            if self.marker_failure_active:
                return
            self._start_next_mi_run()
            return

        if self.current_phase == "run_rest":
            self.record_event("run_rest_end", run_index=self.current_run_index)
            if self.marker_failure_active:
                return
            self._start_next_mi_run()
            return

        if self.current_phase == "idle_block":
            self._finish_idle_block()
            return

        if self.current_phase == "idle_prepare":
            self._finish_idle_prepare_block()
            return

        if self.current_phase == "continuous":
            self._finish_continuous_block()
            return

        if self.current_phase == "baseline":
            if self.current_trial is not None:
                self.record_event(
                    "baseline_end",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
                self.record_event(
                    "cue_start",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
                self.record_event(
                    f"cue_{self.current_trial.class_name}",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
                if self.marker_failure_active:
                    return
                self.current_label.setText(
                    f"当前任务：第 {self.current_trial.trial_id} 个试次 - {self.current_trial.display_name}（提示）"
                )
                self.enter_phase(
                    phase="cue",
                    duration_sec=self.current_settings.cue_sec if self.current_settings is not None else 0.0,
                    class_name=self.current_trial.class_name,
                    title="",
                    subtitle="",
                )
            return

        if self.current_phase == "cue":
            self.start_imagery_phase()
            return

        if self.current_phase == "imagery":
            if self.current_trial is not None:
                self.record_event(
                    "imagery_end",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
                self.record_event(
                    "iti_start",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
            if self.marker_failure_active:
                return
            self.enter_phase(
                phase="iti",
                duration_sec=self.current_settings.iti_sec if self.current_settings is not None else 0.0,
                class_name=None,
                title="",
                subtitle="",
            )
            self.current_label.setText("当前任务：休息恢复")
            self._play_imagery_tone("end")
            return

        if self.current_phase == "iti":
            if self.current_trial is not None:
                self.record_event(
                    "trial_end",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
                if self.marker_failure_active:
                    return
                self.completed_trials = max(self.completed_trials, self.current_trial_index + 1)
                self._update_progress_label()
                self._update_sequence_label()
                self._update_trial_meta_labels()
            if self.current_settings is not None and self.current_run_trial_index < self.trials_per_run:
                self.start_next_trial()
            else:
                if self.current_settings is not None:
                    self.record_event("mi_run_end", run_index=self.current_run_index)
                    if self.marker_failure_active:
                        return
                    if self._maybe_start_scheduled_continuous():
                        return
                    if self.current_run_index < int(self.current_settings.run_count):
                        rest_duration = self._run_rest_duration_for_completed_run(self.current_run_index)
                        if rest_duration > 0:
                            self.record_event("run_rest_start", run_index=self.current_run_index)
                            if self.marker_failure_active:
                                return
                            self.enter_phase(
                                phase="run_rest",
                                duration_sec=rest_duration,
                                class_name=None,
                                title="",
                                subtitle="",
                            )
                            self.current_label.setText(f"当前任务：轮次 {self.current_run_index} 休息")
                            return
                        self._start_next_mi_run()
                        return
                self.start_post_collection_blocks_or_finish()

    def start_next_trial(self) -> None:
        if self.current_settings is None:
            return
        if self.current_trial_index + 1 >= len(self.sequence):
            self.start_post_collection_blocks_or_finish()
            return

        self.current_trial_index += 1
        self.current_run_trial_index += 1
        class_name = self.sequence[self.current_trial_index]
        info = CLASS_LOOKUP[class_name]
        trial = TrialRecord(
            trial_id=self.current_trial_index + 1,
            class_name=class_name,
            display_name=str(info["display_name"]),
            run_index=self.current_run_index,
            run_trial_index=self.current_run_trial_index,
        )
        self.current_trial = trial
        self.trial_records.append(trial)

        self.record_event(
            "trial_start",
            trial_id=trial.trial_id,
            class_name=class_name,
            run_index=trial.run_index,
            run_trial_index=trial.run_trial_index,
        )
        self.record_event(
            "fixation_start",
            trial_id=trial.trial_id,
            class_name=class_name,
            run_index=trial.run_index,
            run_trial_index=trial.run_trial_index,
        )
        self.record_event(
            "baseline_start",
            trial_id=trial.trial_id,
            class_name=class_name,
            run_index=trial.run_index,
            run_trial_index=trial.run_trial_index,
        )
        self.current_label.setText(f"当前任务：第 {trial.trial_id} 个试次 - 准备阶段")
        self.summary_label.setText(f"状态：正在进行第 {trial.trial_id} / {len(self.sequence)} 个试次")
        self._update_progress_label()
        self._update_sequence_label()
        self._update_trial_meta_labels()
        if self.marker_failure_active:
            return
        self.enter_phase(
            phase="baseline",
            duration_sec=self.current_settings.baseline_sec,
            class_name=None,
            title="",
            subtitle="",
        )

    def start_imagery_phase(self) -> None:
        if self.current_settings is None or self.current_trial is None:
            return

        class_name = self.current_trial.class_name
        self.record_event(
            "imagery_start",
            trial_id=self.current_trial.trial_id,
            class_name=class_name,
            run_index=self.current_trial.run_index,
            run_trial_index=self.current_trial.run_trial_index,
        )
        self.record_event(
            f"imagery_{class_name}",
            trial_id=self.current_trial.trial_id,
            class_name=class_name,
            run_index=self.current_trial.run_index,
            run_trial_index=self.current_trial.run_trial_index,
        )
        self.current_label.setText(
            f"当前任务：第 {self.current_trial.trial_id} 个试次 - {CLASS_LOOKUP[class_name]['display_name']}（想象中）"
        )
        self.summary_label.setText(f"状态：正在执行第 {self.current_trial.trial_id} / {len(self.sequence)} 个试次")
        if self.marker_failure_active:
            return
        self.enter_phase(
            phase="imagery",
            duration_sec=self.current_settings.imagery_sec,
            class_name=class_name,
            title="",
            subtitle="",
        )
        self._play_imagery_tone("start")

    def toggle_pause(self) -> None:
        if not self.session_running or self.waiting_for_save:
            return

        if not self.session_paused:
            pause_perf = time.perf_counter()
            self.remaining_phase_sec = max(0.0, self.phase_deadline - pause_perf)
            self.pause_started_perf = pause_perf
            self.session_paused = True
            self.phase_timer.stop()
            if self.current_trial is not None:
                self.record_event(
                    "pause",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
            else:
                self.record_event("pause")
            if self.marker_failure_active:
                return
            self.phase_label.setText(PHASE_LABELS["paused"])
            self.cue_widget.set_state(
                phase="paused",
                class_name=None,
                title="已暂停",
                subtitle="恢复后将从当前阶段继续。",
            )
            self.countdown_label.setText(f"已暂停，当前阶段剩余 {self.remaining_phase_sec:.1f} 秒")
            self.summary_label.setText("状态：采集已暂停")
            self._apply_phase_theme("paused", None)
            self._sync_participant_display()
            self.log("采集已暂停。")
        else:
            resume_perf = time.perf_counter()
            self.session_paused = False
            paused_duration = max(0.0, resume_perf - self.pause_started_perf) if self.pause_started_perf > 0 else 0.0
            if self.phase_started_perf > 0:
                self.phase_started_perf += paused_duration
            self.pause_started_perf = 0.0
            self.phase_deadline = resume_perf + self.remaining_phase_sec
            if self.current_trial is not None:
                self.record_event(
                    "resume",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
            else:
                self.record_event("resume")
            if self.marker_failure_active:
                return
            self._refresh_phase_display()
            self.update_countdown_text()
            self.phase_timer.start()
            self.summary_label.setText("状态：采集继续进行")
            self.log("采集已继续。")

        self.update_button_states()

    def mark_bad_trial(self) -> None:
        if self.current_phase == "continuous":
            self.mark_continuous_prompt_failed()
            return
        if self.current_trial is None:
            self.show_error("当前没有可标记的试次。")
            return
        if not self.current_trial.accepted:
            self.log(f"第 {self.current_trial.trial_id} 个试次已经被标记为坏试次。")
            return

        self.current_trial.accepted = False
        self.current_trial.note = "操作员标记为坏试次"
        self.record_event(
            "bad_trial_marked",
            trial_id=self.current_trial.trial_id,
            class_name=self.current_trial.class_name,
            run_index=self.current_trial.run_index,
            run_trial_index=self.current_trial.run_trial_index,
        )
        if self.marker_failure_active:
            return
        self.refresh_trial_counters()
        self._update_sequence_label()
        self.log(f"已将第 {self.current_trial.trial_id} 个试次标记为坏试次。")
        self._update_trial_meta_labels()

    def mark_continuous_prompt_failed(self) -> None:
        if self.current_phase != "continuous":
            self.show_error("当前不在连续模式，无法标记命令失败。")
            return
        if self.current_continuous_prompt is None:
            self.show_error("当前没有进行中的连续命令可标记。")
            return
        if int(self.current_continuous_prompt.get("execution_success", 1)) == 0:
            self.log("当前连续命令已经标记为失败。")
            return
        self.current_continuous_prompt["execution_success"] = 0
        label = str(self.current_continuous_prompt.get("class_label", ""))
        prompt_index = int(self.current_continuous_prompt.get("prompt_index", 0))
        self.log(f"已标记连续命令失败：第 {prompt_index} 条（{self._class_ui_name(label) or label}）。")
        self._refresh_phase_display()

    def refresh_trial_counters(self) -> None:
        accepted = sum(1 for trial in self.trial_records if trial.accepted)
        rejected = sum(1 for trial in self.trial_records if not trial.accepted)
        self.accepted_label.setText(f"有效试次：{accepted}")
        self.rejected_label.setText(f"坏试次：{rejected}")
        self._update_trial_meta_labels()

    def _update_progress_label(self) -> None:
        total = len(self.sequence)
        if total <= 0:
            self.progress_bar.setValue(0)
            self.progress_text.setText("总进度：0 / 0")
            return

        started = 0
        if self.session_running and self.current_trial_index >= 0:
            started = self.current_trial_index + 1
        elif self.completed_trials:
            started = self.completed_trials

        percent = int(round(100.0 * self.completed_trials / total))
        self.progress_bar.setValue(max(0, min(100, percent)))
        self.progress_text.setText(
            f"总进度：已完成 {self.completed_trials} / {total}，当前已开始 {started} / {total}"
        )

    def _update_sequence_label(self) -> None:
        if not self.sequence:
            self.sequence_label.setText("当前还未生成试次顺序。")
            return

        parts = []
        for index, class_name in enumerate(self.sequence):
            info = CLASS_LOOKUP[class_name]
            color = str(info["color"])
            border = f"1px solid {color}"
            background = "#FFFFFF"
            text_color = color
            extra_style = ""

            if index < self.completed_trials:
                background = color
                text_color = "#FFFFFF"
            elif index == self.current_trial_index and self.session_running:
                background = "#EFF6FF"
                border = f"2px solid {color}"

            if index < len(self.trial_records) and not self.trial_records[index].accepted:
                extra_style = "text-decoration: line-through;"

            parts.append(
                f"<span style='display:inline-block; margin:2px 4px 2px 0; padding:6px 10px; "
                f"border-radius:999px; border:{border}; background:{background}; color:{text_color}; {extra_style}'>"
                f"{index + 1}. {info['short_name']}</span>"
            )

        self.sequence_label.setText("".join(parts))

    def finish_session_and_request_save(self, *, manual_stop: bool) -> None:
        if self.waiting_for_save or self.worker_thread is None:
            return

        self.phase_timer.stop()
        self.session_paused = False
        self.pause_started_perf = 0.0

        if self.session_running:
            if self.current_phase == "quality_check":
                self.record_event("quality_check_end")
            elif self.current_phase.startswith("calibration_"):
                if 0 <= self.calibration_step_index < len(self.calibration_plan):
                    step = self.calibration_plan[self.calibration_step_index]
                    self.record_event(str(step["end_event"]))
                self.record_event("calibration_end")
            elif self.current_phase == "practice":
                self.record_event("practice_end")
            elif self.current_phase == "run_rest":
                self.record_event("run_rest_end", run_index=self.current_run_index)
            elif self.current_phase == "idle_block":
                self.record_event("idle_block_end", block_index=self.idle_block_index)
            elif self.current_phase == "idle_prepare":
                self.record_event(
                    "idle_prepare_end",
                    block_index=max(1, int(self.idle_prepare_block_index)),
                )
            elif self.current_phase == "continuous":
                self._close_current_continuous_prompt()
                self.record_event("continuous_block_end", block_index=self.continuous_block_index)

            if self.current_phase == "baseline" and self.current_trial is not None:
                self._mark_incomplete_trial_if_needed()
                self.record_event(
                    "baseline_end",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
                self.record_event(
                    "trial_end",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
            elif self.current_phase == "cue" and self.current_trial is not None:
                self._mark_incomplete_trial_if_needed()
                self.record_event(
                    "trial_end",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
            elif self.current_phase == "imagery" and self.current_trial is not None:
                self._mark_incomplete_trial_if_needed()
                self.record_event(
                    "imagery_end",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
                self.record_event(
                    "trial_end",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
            elif self.current_phase == "iti" and self.current_trial is not None:
                if not self._event_already_recorded(
                    "trial_end",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                ):
                    self.record_event(
                        "trial_end",
                        trial_id=self.current_trial.trial_id,
                        class_name=self.current_trial.class_name,
                        run_index=self.current_trial.run_index,
                        run_trial_index=self.current_trial.run_trial_index,
                    )
                self.completed_trials = max(self.completed_trials, self.current_trial_index + 1)

            run_end_recorded = any(
                str(event.get("event_name", "")) == "mi_run_end"
                and int(event.get("run_index") or -1) == int(self.current_run_index)
                for event in self.event_log
            )
            if self.current_run_index > 0 and not run_end_recorded:
                self.record_event("mi_run_end", run_index=self.current_run_index)
            if not self._event_already_recorded("session_end", trial_id=None, class_name=None):
                self.record_event("session_end")
            if self.marker_failure_active:
                return

        self.refresh_trial_counters()
        self._update_progress_label()
        self._update_sequence_label()
        self._update_trial_meta_labels()
        self.session_running = False
        self.waiting_for_save = True
        self.current_phase = "idle"
        self.capture_on_stop = True
        self.phase_label.setText("保存中")
        self.countdown_label.setText("正在等待设备线程返回完整数据…")
        self._apply_phase_theme("idle", None)
        self.cue_widget.set_state(
            phase="idle",
            class_name=None,
            title="正在保存",
            subtitle="请等待当前会话数据写入磁盘。",
        )
        self.participant_window.set_prompt(
            phase="idle",
            class_name=None,
            stage_text="保存中",
            title="实验结束",
            subtitle="正在保存本次采集数据，请保持静止。",
            countdown_text="...",
        )
        self.summary_label.setText("状态：采集结束，正在保存数据")
        self.current_label.setText("当前任务：保存中")
        if manual_stop:
            self.log("已手动停止采集，开始保存当前会话。")
        else:
            self.log("所有试次完成，开始保存当前会话。")

        # Give BrainFlow a tiny buffer window so final markers (trial_end/session_end) are flushed.
        time.sleep(0.08)
        self.update_button_states()
        self.worker_stop_requested.emit()

    def stop_and_save(self) -> None:
        if self.waiting_for_save:
            self.log("当前正在保存，请稍候。")
            return
        if not self.session_running:
            self.show_error("当前没有正在进行的会话。")
            return
        self.finish_session_and_request_save(manual_stop=True)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if not self.session_running:
            super().keyPressEvent(event)
            return
        if event.key() == Qt.Key_Space:
            self.toggle_pause()
            event.accept()
            return
        if event.key() == Qt.Key_B:
            self.mark_bad_trial()
            event.accept()
            return
        if event.key() == Qt.Key_Escape:
            self.stop_and_save()
            event.accept()
            return
        super().keyPressEvent(event)

    def disconnect_device(self) -> None:
        if self.waiting_for_save:
            self.show_error("当前正在保存数据，请等待保存完成。")
            return
        if self.session_running:
            self.show_error("当前正在采集，请先点击“停止并保存”。")
            return
        if self.worker_thread is None:
            self.log("当前没有已连接的设备。")
            return

        self.capture_on_stop = False
        self.summary_label.setText("状态：正在断开设备")
        self.device_label.setText("设备：正在断开…")
        self.current_label.setText("当前任务：无")
        self.log("正在断开设备。")
        self.update_button_states()
        self.worker_stop_requested.emit()

    def record_event(
        self,
        event_name: str,
        *,
        trial_id: int | None = None,
        class_name: str | None = None,
        run_index: int | None = None,
        run_trial_index: int | None = None,
        block_index: int | None = None,
        prompt_index: int | None = None,
        command_duration_sec: float | None = None,
        execution_success: int | bool | None = None,
    ) -> None:
        if self.marker_failure_active:
            return
        elapsed_sec = 0.0
        if self.session_start_perf > 0:
            elapsed_sec = max(0.0, time.perf_counter() - self.session_start_perf)
        event = make_event(
            event_name,
            trial_id=trial_id,
            class_name=class_name,
            run_index=run_index,
            run_trial_index=run_trial_index,
            block_index=block_index,
            prompt_index=prompt_index,
            command_duration_sec=command_duration_sec,
            execution_success=execution_success,
            elapsed_sec=elapsed_sec,
        )
        if self.worker is None:
            self._abort_session_due_to_marker_failure("采集线程未就绪，无法写入标记。")
            return
        ok, error_message = self.worker.insert_marker_sync(float(event["marker_code"]))
        if not ok:
            self._abort_session_due_to_marker_failure(error_message or "写入标记失败。")
            return
        self.event_log.append(event)

    def _abort_session_due_to_marker_failure(self, message: str) -> None:
        if self.marker_failure_active:
            return
        self.marker_failure_active = True
        self.marker_failure_message = str(message)
        self.capture_on_stop = False
        self.waiting_for_save = False
        self.phase_timer.stop()
        self.session_running = False
        self.session_paused = False
        self.pause_started_perf = 0.0
        self.current_phase = "idle"
        self.phase_label.setText("标记失败")
        self.countdown_label.setText("本次会话未保存，请检查设备连接和标记写入后重试。")
        self.summary_label.setText("状态：标记写入失败，本次会话已中止")
        self.current_label.setText("当前任务：请检查设备后重新采集")
        self._apply_phase_theme("idle", None)
        self.cue_widget.set_state(
            phase="idle",
            class_name=None,
            title="标记失败",
            subtitle="本次会话未保存，请检查设备连接和标记写入后重新开始。",
        )
        self.participant_window.set_prompt(
            phase="idle",
            class_name=None,
            stage_text="标记失败",
            title="实验已中止",
            subtitle="检测到标记写入失败，本次会话不会保存，请联系操作员重新开始。",
            countdown_text="-",
        )
        self.update_button_states()
        if self.worker_thread is not None:
            self.worker_stop_requested.emit()
        self.show_error(f"{message}\n\n本次会话不会保存，请检查设备连接后重新采集。")

    def _set_save_failed_state(self, message: str) -> None:
        self.waiting_for_save = False
        self.capture_on_stop = False
        self.phase_label.setText("保存失败")
        self.countdown_label.setText("本次会话未保存，请检查原因后重新采集。")
        self.summary_label.setText("状态：保存失败，本次会话未保存")
        self.current_label.setText("当前任务：保存失败，请重新采集")
        self._apply_phase_theme("idle", None)
        self.cue_widget.set_state(
            phase="idle",
            class_name=None,
            title="保存失败",
            subtitle="当前会话未成功写入磁盘，请排查后重新开始。",
        )
        self.trial_banner_label.setText("当前试次：本次会话未保存")
        self.next_task_label.setText("下一任务：检查原因后重新开始")
        if self.use_separate_participant_screen:
            self.restore_operator_window()
        else:
            self.participant_window.hide()
        self.show_error(message)
        self.update_button_states()

    def _handle_missing_save_result(self) -> None:
        if not self.waiting_for_save:
            return
        self._set_save_failed_state("保存失败：设备线程已结束，但没有返回完整会话数据。本次会话未保存，请重新采集。")

    def on_session_data_ready(self, payload: dict) -> None:
        if not self.capture_on_stop or self.current_settings is None or not self.event_log:
            self.log("设备停止，但没有需要保存的有效会话。")
            return

        try:
            result = save_mi_session(
                brainflow_data=np.asarray(payload["brainflow_data"], dtype=np.float32),
                sampling_rate=float(payload["sampling_rate"]),
                eeg_rows=list(payload["selected_rows"]),
                marker_row=int(payload["marker_row"]),
                timestamp_row=None if payload["timestamp_row"] is None else int(payload["timestamp_row"]),
                settings=self.current_settings,
                event_log=list(self.event_log),
                trial_records=list(self.trial_records),
            )
        except Exception as error:
            self._set_save_failed_state(f"保存会话失败：{error}")
            return

        self.waiting_for_save = False
        self.capture_on_stop = False
        self.summary_label.setText(f"状态：保存完成，共保存 {result['trial_count']} 个试次")
        self.current_label.setText(f"当前任务：数据已保存到 {result['session_dir']}")
        self.phase_label.setText("保存完成")
        self.countdown_label.setText("当前会话已完成，可重新连接并开始新的采集。")
        self._apply_phase_theme("idle", None)
        self.cue_widget.set_state(
            phase="idle",
            class_name=None,
            title="保存完成",
            subtitle="可以开始下一次采集，或关闭程序。",
        )
        self.refresh_trial_counters()
        self.trial_banner_label.setText(f"当前试次：本次会话已完成，共 {result['trial_count']} 个试次")
        self.next_task_label.setText("下一任务：可以开始新的会话")
        self.log(f"会话已保存：{result['session_dir']}")
        if "run_index" in result and "run_stem" in result:
            self.log(f"采集编号：run-{int(result['run_index']):03d} | 标识：{result['run_stem']}")
        self.log(f"原始数据：{result['fif_path']}")
        if result.get("epochs_path"):
            self.log(f"训练缓存：{result['epochs_path']}")
        if result.get("mi_epochs_path"):
            self.log(f"主分类训练集：{result['mi_epochs_path']}")
        if result.get("gate_epochs_path"):
            self.log(f"门控训练集：{result['gate_epochs_path']}")
        if result.get("artifact_epochs_path"):
            self.log(f"坏窗训练集：{result['artifact_epochs_path']}")
        if result.get("continuous_path"):
            self.log(f"连续验证集：{result['continuous_path']}")
        if result.get("manifest_csv_path"):
            self.log(f"数据索引：{result['manifest_csv_path']}")
        self.session_edit.setText(datetime.now().strftime("%Y%m%d_%H%M%S"))
        if self.use_separate_participant_screen:
            self.restore_operator_window()
        else:
            self.participant_window.hide()
        self.update_button_states()

    def on_worker_thread_finished(self) -> None:
        marker_failure_active = self.marker_failure_active
        saved_completed = self.current_label.text().startswith("当前任务：数据已保存到")
        save_failed = self.current_label.text().startswith("当前任务：保存失败")
        pending_save_result = self.waiting_for_save
        self.worker = None
        self.worker_thread = None
        self.device_info = None
        self.session_running = False
        self.session_paused = False
        self.current_phase = "idle"
        self.phase_started_perf = 0.0
        self.pause_started_perf = 0.0
        self.phase_timer.stop()
        self.sequence = []
        self.sequence_by_run = []
        self.trials_per_run = 0
        self.current_run_index = 0
        self.current_run_trial_index = 0
        self.calibration_plan = []
        self.calibration_step_index = -1
        self.idle_block_index = 0
        self.idle_prepare_block_index = 0
        self.continuous_block_index = 0
        self.continuous_schedule_after_runs = []
        self.next_continuous_schedule_index = 0
        self.continuous_prompt_plan = []
        self.continuous_prompt_index = -1
        self.current_continuous_prompt = None
        self.preview_mode = RealtimeEEGPreviewWidget.MODE_EEG
        self.preview_impedance_channel = 1
        self.device_label.setText("设备：未连接")
        if self.preview_widget is not None:
            self.preview_widget.clear_stream("设备断开后停止实时波形显示。")
            self.preview_widget.set_quality_mode(self.preview_mode, impedance_channel=self.preview_impedance_channel)
        self._apply_phase_theme("idle", None)
        if marker_failure_active:
            self.summary_label.setText("状态：标记写入失败，本次会话已丢弃")
            self.current_label.setText("当前任务：检查设备后重新开始")
            self.trial_banner_label.setText("当前试次：本次会话因标记写入失败未保存")
            self.next_task_label.setText("下一任务：排查设备连接或 marker 写入")
        elif not saved_completed and not save_failed:
            self.current_label.setText("当前任务：无")
            self.trial_banner_label.setText("当前试次：未开始")
            self.next_task_label.setText("下一任务：--")
        if not self.waiting_for_save and not saved_completed and not save_failed and not marker_failure_active:
            self.summary_label.setText("状态：设备已断开")
        self.marker_failure_active = False
        self.marker_failure_message = ""
        self._set_preview_mode_label()
        self._update_preview_status()
        self.update_button_states()
        if pending_save_result:
            QTimer.singleShot(0, self._handle_missing_save_result)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.waiting_for_save:
            QMessageBox.information(self, "请稍候", "当前正在保存数据，请等待保存完成后再关闭窗口。")
            event.ignore()
            return

        if self.session_running:
            QMessageBox.information(self, "请先停止并保存", "当前会话正在运行，请先点击“停止并保存”。")
            event.ignore()
            return

        if self.worker_thread is None:
            self.participant_window.hide()
            event.accept()
            return

        reply = QMessageBox.question(
            self,
            "退出确认",
            "设备仍处于连接状态。关闭窗口前将先断开设备，是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            event.ignore()
            return

        self.capture_on_stop = False
        self.participant_window.hide()
        self.worker_stop_requested.emit()
        if self.worker_thread is not None:
            self.worker_thread.wait(5000)
        event.accept()


def build_initial_config_from_args(args: argparse.Namespace) -> dict:
    """Translate CLI options into initial window values."""
    config = {}
    if args.synthetic:
        synthetic = getattr(BoardIds, "SYNTHETIC_BOARD", None)
        if synthetic is not None:
            config["board_id"] = int(synthetic.value)
            config["serial_port"] = ""
    if args.serial_port:
        config["serial_port"] = args.serial_port
    if args.output_root:
        config["output_root"] = args.output_root
    if args.subject_id:
        config["subject_id"] = args.subject_id
    if args.session_id:
        config["session_id"] = args.session_id
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="运动想象脑电数据采集器（仅采集）")
    parser.add_argument("--serial-port", type=str, default="", help="覆盖默认串口，例如 COM3")
    parser.add_argument("--output-root", type=str, default="", help="覆盖默认输出目录")
    parser.add_argument("--subject-id", type=str, default="", help="预填被试编号")
    parser.add_argument("--session-id", type=str, default="", help="预填会话编号")
    parser.add_argument("--synthetic", action="store_true", help="使用 BrainFlow 模拟板卡演示界面")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("运动想象数据采集器")
    window = MIDataCollectorWindow(initial_config=build_initial_config_from_args(args))
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
