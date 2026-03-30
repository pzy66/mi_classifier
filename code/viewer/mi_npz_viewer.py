# -*- coding: utf-8 -*-
"""MI epochs.npz 可视化工具：数字统计 + 波形 + 频谱。"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib import rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "custom_mi"
NPZ_PATTERNS = ("*_epochs.npz", "epochs.npz")
NPZ_EXCLUDED_SUFFIXES = ("_gate_epochs.npz", "_artifact_epochs.npz")
CLASS_DISPLAY = {"left_hand": "左手", "right_hand": "右手", "feet": "双脚", "tongue": "舌头"}
RUN_PATTERN = re.compile(
    r"sub-(?P<subject>.+?)_ses-(?P<session>.+?)_run-(?P<run>\d{3})_tpc-(?P<tpc>\d+)_n-(?P<n>\d+)_ok-(?P<ok>\d+)_epochs\.npz$"
)

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


@dataclass
class EpochData:
    path: Path
    X_uV: np.ndarray
    y: np.ndarray
    accepted: np.ndarray
    trial_ids: np.ndarray
    class_names: list[str]
    channel_names: list[str]
    sampling_rate: float
    source_signal_unit: str

    @property
    def n_trials(self) -> int:
        return int(self.X_uV.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.X_uV.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.X_uV.shape[2])

    @property
    def window_sec(self) -> float:
        return float(self.n_samples / self.sampling_rate) if self.sampling_rate > 0 else 0.0


def _s(v: object) -> str:
    return v.decode("utf-8", errors="ignore").strip() if isinstance(v, bytes) else str(v).strip()


def _normalize_unit(unit: str) -> str:
    token = unit.strip().lower()
    if token in {"v", "volt", "volts"}:
        return "volt"
    if token in {"uv", "µv", "μv", "microvolt", "microvolts"}:
        return "microvolt"
    return "unknown"


def _display_class(name: str) -> str:
    return CLASS_DISPLAY.get(name, name)


def _scaled_px(base_px: float, scale: float, min_px: int, max_px: int | None = None) -> int:
    value = int(round(float(base_px) * float(scale)))
    if max_px is not None:
        value = min(value, int(max_px))
    return max(int(min_px), value)


def discover_epoch_files(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        return []
    out: list[Path] = []
    seen: set[Path] = set()
    for pattern in NPZ_PATTERNS:
        for p in dataset_root.rglob(pattern):
            if any(str(p.name).endswith(suffix) for suffix in NPZ_EXCLUDED_SUFFIXES):
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            out.append(rp)
    out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out


def parse_run_text(path: Path) -> str:
    m = RUN_PATTERN.search(path.name)
    if not m:
        return "未解析到 run 信息（旧命名或手工命名）"
    d = m.groupdict()
    return f"sub-{d['subject']} | ses-{d['session']} | run-{d['run']} | tpc={d['tpc']} | n={d['n']} | ok={d['ok']}"


def load_epochs_npz(path: Path) -> EpochData:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    with np.load(path, allow_pickle=True) as data:
        if "X" not in data.files or "y" not in data.files:
            raise KeyError("npz 缺少字段：X 或 y")
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64).reshape(-1)
        if X.ndim != 3 or y.shape[0] != X.shape[0]:
            raise ValueError(f"X/y 形状不合法：X={X.shape}, y={y.shape}")
        accepted = np.asarray(data["accepted"], dtype=np.int8).reshape(-1) if "accepted" in data.files else np.ones(X.shape[0], dtype=np.int8)
        if accepted.shape[0] != X.shape[0]:
            accepted = np.ones(X.shape[0], dtype=np.int8)
        trial_ids = np.asarray(data["trial_ids"], dtype=np.int64).reshape(-1) if "trial_ids" in data.files else np.arange(1, X.shape[0] + 1, dtype=np.int64)
        if trial_ids.shape[0] != X.shape[0]:
            trial_ids = np.arange(1, X.shape[0] + 1, dtype=np.int64)
        channel_names = [_s(v) for v in np.asarray(data["channel_names"]).reshape(-1).tolist()] if "channel_names" in data.files else [f"EEG{i+1}" for i in range(X.shape[1])]
        class_names = [_s(v) for v in np.asarray(data["class_names"]).reshape(-1).tolist()] if "class_names" in data.files else []
        if len(channel_names) != X.shape[1]:
            channel_names = [f"EEG{i+1}" for i in range(X.shape[1])]
        max_label = int(np.max(y)) if y.size else -1
        while len(class_names) <= max_label:
            class_names.append(f"class_{len(class_names)}")
        fs = float(np.asarray(data["sampling_rate"]).reshape(-1)[0]) if "sampling_rate" in data.files else 250.0
        unit = _s(np.asarray(data["signal_unit"]).reshape(-1)[0]) if "signal_unit" in data.files else "volt"
    X_uV = X * 1e6 if _normalize_unit(unit) == "volt" else X
    return EpochData(
        path=path.resolve(),
        X_uV=X_uV.astype(np.float32),
        y=y.astype(np.int64),
        accepted=accepted.astype(bool),
        trial_ids=trial_ids.astype(np.int64),
        class_names=class_names,
        channel_names=channel_names,
        sampling_rate=fs,
        source_signal_unit=unit,
    )


def class_rows(data: EpochData) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for i, name in enumerate(data.class_names):
        total = int(np.sum(data.y == i))
        ok = int(np.sum((data.y == i) & data.accepted))
        out.append({"label": i, "class_key": name, "class_display": _display_class(name), "total": total, "accepted": ok, "rejected": total - ok})
    return out


def channel_rows(data: EpochData) -> list[dict[str, object]]:
    mask = data.accepted if np.any(data.accepted) else np.ones(data.n_trials, dtype=bool)
    subset = data.X_uV[mask] if np.any(mask) else data.X_uV
    out: list[dict[str, object]] = []
    for i, name in enumerate(data.channel_names):
        v = subset[:, i, :].reshape(-1)
        out.append(
            {
                "channel": name,
                "mean_uV": float(np.mean(v)) if v.size else 0.0,
                "std_uV": float(np.std(v)) if v.size else 0.0,
                "rms_uV": float(np.sqrt(np.mean(v**2))) if v.size else 0.0,
                "ptp_uV": float(np.ptp(v)) if v.size else 0.0,
                "abs_mean_uV": float(np.mean(np.abs(v))) if v.size else 0.0,
            }
        )
    return out


class NPZViewer(QMainWindow):
    def __init__(self, initial: Path | None = None) -> None:
        super().__init__()
        self.data: EpochData | None = None
        self.stats: dict[str, object] | None = None
        self._ui_scale = 1.0
        self.setWindowTitle("MI 数据可视化（NPZ）")
        self.resize(1520, 920)
        self._build_ui()
        self._style()
        self.scan_files()
        if initial is not None:
            self.path_edit.setText(str(initial))
        elif self.file_combo.count() > 0:
            self.path_edit.setText(str(self.file_combo.currentData()))
        if self.path_edit.text().strip():
            self.load_file()

    def _build_ui(self) -> None:
        c = QWidget()
        self.setCentralWidget(c)
        root = QVBoxLayout(c)

        file_group = QGroupBox("1) 文件选择")
        fg = QGridLayout(file_group)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("选择 *_mi_epochs.npz、*_epochs.npz 或 epochs.npz")
        self.browse_btn, self.load_btn = QPushButton("浏览"), QPushButton("加载")
        self.root_edit = QLineEdit(str(DEFAULT_DATASET_ROOT))
        self.scan_btn = QPushButton("扫描")
        self.file_combo = QComboBox()
        self.load_selected_btn = QPushButton("加载选中")
        fg.addWidget(QLabel("文件"), 0, 0)
        fg.addWidget(self.path_edit, 0, 1)
        fg.addWidget(self.browse_btn, 0, 2)
        fg.addWidget(self.load_btn, 0, 3)
        fg.addWidget(QLabel("数据集目录"), 1, 0)
        fg.addWidget(self.root_edit, 1, 1)
        fg.addWidget(self.scan_btn, 1, 2)
        fg.addWidget(QLabel("扫描结果"), 2, 0)
        fg.addWidget(self.file_combo, 2, 1)
        fg.addWidget(self.load_selected_btn, 2, 2)
        root.addWidget(file_group)

        sg = QGridLayout()
        summary_group = QGroupBox("2) 总览")
        sf = QFormLayout(summary_group)
        self.lb_file, self.lb_run, self.lb_trials, self.lb_channels = QLabel("--"), QLabel("--"), QLabel("--"), QLabel("--")
        self.lb_fs, self.lb_win, self.lb_acc, self.lb_unit, self.lb_rms = QLabel("--"), QLabel("--"), QLabel("--"), QLabel("--"), QLabel("--")
        self.lb_file.setTextInteractionFlags(Qt.TextSelectableByMouse)
        sf.addRow("文件", self.lb_file)
        sf.addRow("Run 信息", self.lb_run)
        sf.addRow("试次数", self.lb_trials)
        sf.addRow("通道数", self.lb_channels)
        sf.addRow("采样率", self.lb_fs)
        sf.addRow("每 trial 时长", self.lb_win)
        sf.addRow("有效率", self.lb_acc)
        sf.addRow("原始单位", self.lb_unit)
        sf.addRow("全局 RMS(uV)", self.lb_rms)

        cls_group = QGroupBox("3) 类别统计")
        cls_v = QVBoxLayout(cls_group)
        self.cls_table = QTableWidget(0, 5)
        self.cls_table.setHorizontalHeaderLabels(["Label", "类别", "总试次", "有效", "无效"])
        self.cls_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cls_table.verticalHeader().setVisible(False)
        self.cls_table.setEditTriggers(QTableWidget.NoEditTriggers)
        cls_v.addWidget(self.cls_table)

        ch_group = QGroupBox("4) 通道统计")
        ch_v = QVBoxLayout(ch_group)
        self.ch_table = QTableWidget(0, 6)
        self.ch_table.setHorizontalHeaderLabels(["Channel", "Mean", "Std", "RMS", "PtP", "AbsMean"])
        self.ch_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ch_table.verticalHeader().setVisible(False)
        self.ch_table.setEditTriggers(QTableWidget.NoEditTriggers)
        ch_v.addWidget(self.ch_table)

        sg.addWidget(summary_group, 0, 0)
        sg.addWidget(cls_group, 0, 1)
        sg.addWidget(ch_group, 0, 2)
        root.addLayout(sg)

        viz_group = QGroupBox("5) 波形/频谱")
        vv = QVBoxLayout(viz_group)
        ctrl = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("单试次波形", "trial")
        self.mode_combo.addItem("类别平均波形", "class_mean")
        self.mode_combo.addItem("类别平均频谱(PSD)", "class_psd")
        self.class_combo = QComboBox()
        self.class_combo.addItem("全部类别", -1)
        self.trial_spin = QSpinBox()
        self.trial_spin.setRange(1, 1)
        self.trial_id = QLabel("TrialID: --")
        self.ch_combo = QComboBox()
        self.ch_combo.addItem("全部通道（堆叠）", -1)
        self.ck_ok = QCheckBox("仅有效")
        self.ck_ok.setChecked(True)
        self.ck_demean = QCheckBox("去直流")
        self.ck_demean.setChecked(True)
        self.replot_btn, self.exp_json_btn, self.exp_csv_btn = QPushButton("刷新"), QPushButton("导出JSON"), QPushButton("导出CSV")
        for w in [QLabel("模式"), self.mode_combo, QLabel("类别"), self.class_combo, QLabel("Trial"), self.trial_spin, self.trial_id, QLabel("通道"), self.ch_combo, self.ck_ok, self.ck_demean, self.replot_btn, self.exp_json_btn, self.exp_csv_btn]:
            ctrl.addWidget(w)
        ctrl.addStretch(1)
        vv.addLayout(ctrl)

        self.fig = Figure(figsize=(12, 6), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        vv.addWidget(self.canvas, stretch=1)
        root.addWidget(viz_group, stretch=1)

        self.status = QLabel("就绪")
        root.addWidget(self.status)

        self.browse_btn.clicked.connect(self.browse_file)
        self.load_btn.clicked.connect(self.load_file)
        self.scan_btn.clicked.connect(self.scan_files)
        self.load_selected_btn.clicked.connect(self.load_selected)
        self.mode_combo.currentIndexChanged.connect(self._on_ctrl_changed)
        self.class_combo.currentIndexChanged.connect(self._on_ctrl_changed)
        self.trial_spin.valueChanged.connect(self._on_ctrl_changed)
        self.ch_combo.currentIndexChanged.connect(self._on_ctrl_changed)
        self.ck_ok.stateChanged.connect(self._on_ctrl_changed)
        self.ck_demean.stateChanged.connect(self._on_ctrl_changed)
        self.replot_btn.clicked.connect(self.refresh_plot)
        self.exp_json_btn.clicked.connect(self.export_json)
        self.exp_csv_btn.clicked.connect(self.export_csv)

    def _style(self) -> None:
        self._ui_scale = self._compute_ui_scale()
        group_font = self._scaled_ui_px(13, min_px=11, max_px=22)
        table_font = self._scaled_ui_px(12, min_px=10, max_px=20)
        radius = self._scaled_ui_px(8, min_px=6, max_px=16)
        group_margin = self._scaled_ui_px(10, min_px=6, max_px=18)
        group_pad_top = self._scaled_ui_px(8, min_px=5, max_px=14)
        title_left = self._scaled_ui_px(10, min_px=6, max_px=18)
        title_pad = self._scaled_ui_px(4, min_px=2, max_px=10)
        btn_radius = self._scaled_ui_px(6, min_px=4, max_px=12)
        btn_pad_v = self._scaled_ui_px(6, min_px=4, max_px=12)
        btn_pad_h = self._scaled_ui_px(12, min_px=8, max_px=22)
        self.setStyleSheet(
            f"QMainWindow{{background:#f7f9fc;}}"
            f"QGroupBox{{font-size:{group_font}px;font-weight:600;border:1px solid #d4dae4;"
            f"border-radius:{radius}px;margin-top:{group_margin}px;padding-top:{group_pad_top}px;background:#fff;}}"
            f"QGroupBox::title{{subcontrol-origin:margin;left:{title_left}px;padding:0 {title_pad}px;}}"
            f"QPushButton{{background:#eaf1ff;border:1px solid #c7d7ff;border-radius:{btn_radius}px;"
            f"padding:{btn_pad_v}px {btn_pad_h}px;}}"
            f"QPushButton:hover{{background:#dce9ff;}}"
            f"QTableWidget{{background:#fbfdff;gridline-color:#d8dee8;font-size:{table_font}px;}}"
        )
        if hasattr(self, "status"):
            status_font = self._scaled_ui_px(12, min_px=10, max_px=20)
            self.status.setStyleSheet(f"color:#334155; font-size:{status_font}px;")

    def _compute_ui_scale(self) -> float:
        width = max(980, self.width())
        height = max(680, self.height())
        return max(0.82, min(1.8, min(width / 1520.0, height / 920.0)))

    def _scaled_ui_px(self, base_px: float, min_px: int, max_px: int | None = None) -> int:
        return _scaled_px(base_px, self._ui_scale, min_px=min_px, max_px=max_px)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._style()

    def _set_status(self, text: str) -> None:
        self.status.setText(text)

    def browse_file(self) -> None:
        start = Path(self.root_edit.text().strip() or str(DEFAULT_DATASET_ROOT))
        if not start.exists():
            start = PROJECT_ROOT
        p, _ = QFileDialog.getOpenFileName(self, "选择 npz 文件", str(start), "NPZ files (*.npz)")
        if p:
            self.path_edit.setText(p)

    def scan_files(self) -> None:
        files = discover_epoch_files(Path(self.root_edit.text().strip() or str(DEFAULT_DATASET_ROOT)))
        self.file_combo.clear()
        for p in files:
            ts = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            self.file_combo.addItem(f"{p.name} | {p.parent.name} | {ts}", str(p))
        self._set_status(f"扫描完成：{len(files)} 个文件")

    def load_selected(self) -> None:
        v = self.file_combo.currentData()
        if v:
            self.path_edit.setText(str(v))
            self.load_file()

    def load_file(self) -> None:
        raw = self.path_edit.text().strip()
        if not raw:
            QMessageBox.warning(self, "提示", "请先选择文件")
            return
        try:
            self.data = load_epochs_npz(Path(raw))
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))
            self._set_status(f"加载失败：{e}")
            return
        self._refresh_stats()
        self._refresh_controls()
        self.refresh_plot()
        self._set_status(f"已加载：{self.data.path}")

    def _refresh_stats(self) -> None:
        assert self.data is not None
        ok = int(np.sum(self.data.accepted))
        total = self.data.n_trials
        rej = total - ok
        ratio = (ok / total * 100.0) if total > 0 else 0.0
        rms = float(np.sqrt(np.mean(self.data.X_uV.reshape(-1) ** 2))) if self.data.X_uV.size else 0.0
        self.lb_file.setText(str(self.data.path))
        self.lb_run.setText(parse_run_text(self.data.path))
        self.lb_trials.setText(f"{total}（有效 {ok} / 无效 {rej}）")
        self.lb_channels.setText(str(self.data.n_channels))
        self.lb_fs.setText(f"{self.data.sampling_rate:.3f} Hz")
        self.lb_win.setText(f"{self.data.window_sec:.3f} s ({self.data.n_samples} 点)")
        self.lb_acc.setText(f"{ratio:.1f}%")
        self.lb_unit.setText(self.data.source_signal_unit)
        self.lb_rms.setText(f"{rms:.3f}")

        crows = class_rows(self.data)
        self.cls_table.setRowCount(len(crows))
        for r, row in enumerate(crows):
            vals = [str(row["label"]), str(row["class_display"]), str(row["total"]), str(row["accepted"]), str(row["rejected"])]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignCenter)
                self.cls_table.setItem(r, c, it)

        rows = channel_rows(self.data)
        self.ch_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            vals = [row["channel"], f"{row['mean_uV']:.3f}", f"{row['std_uV']:.3f}", f"{row['rms_uV']:.3f}", f"{row['ptp_uV']:.3f}", f"{row['abs_mean_uV']:.3f}"]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c > 0:
                    it.setTextAlignment(Qt.AlignCenter)
                self.ch_table.setItem(r, c, it)

        self.stats = {"generated_at": datetime.now().isoformat(timespec="seconds"), "source_npz": str(self.data.path), "summary": {"run_info": parse_run_text(self.data.path), "trials_total": total, "trials_accepted": ok, "trials_rejected": rej, "accept_ratio_percent": ratio, "channels": self.data.n_channels, "sampling_rate_hz": self.data.sampling_rate, "window_sec": self.data.window_sec, "samples_per_trial": self.data.n_samples, "source_signal_unit": self.data.source_signal_unit, "global_rms_uV": rms}, "class_stats": crows, "channel_stats": rows}

    def _refresh_controls(self) -> None:
        assert self.data is not None
        for widget in (self.class_combo, self.ch_combo, self.trial_spin):
            widget.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItem("全部类别", -1)
        for i, n in enumerate(self.data.class_names):
            self.class_combo.addItem(f"{i}: {_display_class(n)}", i)
        self.ch_combo.clear()
        self.ch_combo.addItem("全部通道（堆叠）", -1)
        for i, n in enumerate(self.data.channel_names):
            self.ch_combo.addItem(n, i)
        self.trial_spin.setRange(1, max(1, self.data.n_trials))
        self.trial_spin.setValue(1)
        for widget in (self.class_combo, self.ch_combo, self.trial_spin):
            widget.blockSignals(False)
        self._sync_mode()
        self._update_trial_id()

    def _sync_mode(self) -> None:
        is_trial = str(self.mode_combo.currentData()) == "trial"
        self.trial_spin.setEnabled(is_trial)
        self.class_combo.setEnabled(not is_trial)

    @staticmethod
    def _combo_int(combo: QComboBox, default: int) -> int:
        value = combo.currentData()
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _update_trial_id(self) -> None:
        if self.data is None:
            self.trial_id.setText("TrialID: --")
            return
        i = max(0, min(self.trial_spin.value() - 1, self.data.n_trials - 1))
        tid = int(self.data.trial_ids[i]) if self.data.trial_ids.size else i + 1
        self.trial_id.setText(f"TrialID: {tid}")

    def _on_ctrl_changed(self) -> None:
        if self.data is None:
            return
        self._sync_mode()
        self._update_trial_id()
        try:
            self.refresh_plot()
        except Exception as error:
            self._set_status(f"绘图失败: {error}")

    def _ch_indices(self) -> list[int]:
        if self.data is None:
            return []
        idx = self._combo_int(self.ch_combo, -1)
        return list(range(self.data.n_channels)) if idx < 0 else [idx]

    def _trial_index(self) -> int | None:
        assert self.data is not None
        i = max(0, min(self.trial_spin.value() - 1, self.data.n_trials - 1))
        if not self.ck_ok.isChecked() or self.data.accepted[i]:
            return i
        valid = np.flatnonzero(self.data.accepted)
        if valid.size == 0:
            return None
        j = int(valid[np.argmin(np.abs(valid - i))])
        if j != i:
            self.trial_spin.blockSignals(True)
            self.trial_spin.setValue(j + 1)
            self.trial_spin.blockSignals(False)
        return j

    def _agg_mask(self) -> np.ndarray:
        assert self.data is not None
        m = np.ones(self.data.n_trials, dtype=bool)
        ci = self._combo_int(self.class_combo, -1)
        if ci >= 0:
            m &= self.data.y == ci
        if self.ck_ok.isChecked():
            m &= self.data.accepted
        return m

    def refresh_plot(self) -> None:
        if self.data is None:
            self.fig.clear()
            self.canvas.draw_idle()
            return
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3.0, 1.2])
        ax, axb = self.fig.add_subplot(gs[0, 0]), self.fig.add_subplot(gs[1, 0])
        mode = str(self.mode_combo.currentData())
        ch_idx = self._ch_indices()
        t = np.arange(self.data.n_samples, dtype=np.float32) / max(self.data.sampling_rate, 1e-6)

        if mode == "trial":
            ti = self._trial_index()
            if ti is None:
                ax.text(0.5, 0.5, "没有可显示的有效 trial", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            else:
                x = self.data.X_uV[ti, ch_idx, :].copy()
                if self.ck_demean.isChecked():
                    x -= np.mean(x, axis=1, keepdims=True)
                cname = _display_class(self.data.class_names[int(self.data.y[ti])]) if self.data.class_names else str(int(self.data.y[ti]))
                tid = int(self.data.trial_ids[ti]) if self.data.trial_ids.size else ti + 1
                if len(ch_idx) == 1:
                    ax.plot(t, x[0], lw=1.2, color="#2463eb")
                    ax.set_ylabel("振幅 (uV)")
                    ax.set_title(f"单试次波形 | TrialID={tid} | 类别={cname} | 通道={self.data.channel_names[ch_idx[0]]}")
                else:
                    step = max(8.0, float(np.percentile(np.ptp(x, axis=1), 85)) * 1.25)
                    off = np.arange(len(ch_idx), dtype=np.float32) * step
                    for i, ci in enumerate(ch_idx):
                        ax.plot(t, x[i] + off[i], lw=1.0)
                    ax.set_yticks(off)
                    ax.set_yticklabels([self.data.channel_names[i] for i in ch_idx])
                    ax.set_ylabel("通道（堆叠）")
                    ax.set_title(f"单试次波形（多通道） | TrialID={tid} | 类别={cname}")
                ax.set_xlabel("时间 (s)")
                ax.grid(alpha=0.25)
        else:
            m = self._agg_mask()
            x = self.data.X_uV[m][:, ch_idx, :]
            if x.size == 0:
                ax.text(0.5, 0.5, "当前筛选条件下没有 trial", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            else:
                if self.ck_demean.isChecked():
                    x -= np.mean(x, axis=2, keepdims=True)
                ci = self._combo_int(self.class_combo, -1)
                cdisp = "全部类别" if ci < 0 or ci >= len(self.data.class_names) else _display_class(self.data.class_names[ci])
                if mode == "class_mean":
                    if len(ch_idx) == 1:
                        y = x[:, 0, :]
                        mu, sd = np.mean(y, axis=0), np.std(y, axis=0)
                        ax.plot(t, mu, lw=1.6, color="#ea580c", label="均值")
                        ax.fill_between(t, mu - sd, mu + sd, alpha=0.25, color="#fb923c", label="±1 标准差")
                        ax.legend(loc="upper right")
                        ax.set_ylabel("振幅 (uV)")
                        ax.set_title(f"类别平均波形 | 类别={cdisp} | 通道={self.data.channel_names[ch_idx[0]]} | 试次={x.shape[0]}")
                    else:
                        mu = np.mean(x, axis=0)
                        step = max(8.0, float(np.percentile(np.ptp(mu, axis=1), 85)) * 1.25)
                        off = np.arange(len(ch_idx), dtype=np.float32) * step
                        for i, ci2 in enumerate(ch_idx):
                            ax.plot(t, mu[i] + off[i], lw=1.1)
                        ax.set_yticks(off)
                        ax.set_yticklabels([self.data.channel_names[i] for i in ch_idx])
                        ax.set_ylabel("通道（堆叠）")
                        ax.set_title(f"类别平均波形（多通道） | 类别={cdisp} | 试次={x.shape[0]}")
                    ax.set_xlabel("时间 (s)")
                    ax.grid(alpha=0.25)
                else:
                    n, fs = self.data.n_samples, max(self.data.sampling_rate, 1e-6)
                    f = np.fft.rfftfreq(n, d=1.0 / fs)
                    sp = np.fft.rfft(x, axis=-1)
                    psd = (np.abs(sp) ** 2) / (fs * n)
                    if n > 1:
                        psd[..., 1:-1] *= 2.0
                    fm = f <= min(80.0, fs / 2.0)
                    if len(ch_idx) == 1:
                        p = np.mean(psd[:, 0, :], axis=0)[fm]
                        ax.semilogy(f[fm], np.maximum(p, 1e-12), lw=1.4, color="#0f766e")
                        ax.set_title(f"类别平均频谱(PSD) | 类别={cdisp} | 通道={self.data.channel_names[ch_idx[0]]} | 试次={x.shape[0]}")
                    else:
                        for i, ci2 in enumerate(ch_idx):
                            p = np.mean(psd[:, i, :], axis=0)[fm]
                            ax.semilogy(f[fm], np.maximum(p, 1e-12), lw=1.0, label=self.data.channel_names[ci2])
                        if len(ch_idx) <= 10:
                            ax.legend(loc="upper right", fontsize=8)
                        ax.set_title(f"类别平均频谱(PSD)（多通道） | 类别={cdisp} | 试次={x.shape[0]}")
                    ax.set_xlabel("频率 (Hz)")
                    ax.set_ylabel("PSD (uV²/Hz)")
                    ax.grid(alpha=0.25)

        idx = np.arange(len(self.data.class_names), dtype=np.int64)
        total = np.array([np.sum(self.data.y == i) for i in idx], dtype=np.int64)
        ok = np.array([np.sum((self.data.y == i) & self.data.accepted) for i in idx], dtype=np.int64)
        x = np.arange(len(idx), dtype=np.float32)
        axb.bar(x - 0.18, total, width=0.36, color="#60a5fa", label="总试次")
        axb.bar(x + 0.18, ok, width=0.36, color="#22c55e", label="有效试次")
        axb.set_xticks(x)
        axb.set_xticklabels([_display_class(n) for n in self.data.class_names])
        axb.set_title("类别分布")
        axb.set_ylabel("数量")
        axb.grid(axis="y", alpha=0.2)
        axb.legend(loc="upper right")
        self.canvas.draw_idle()

    def _unique(self, p: Path) -> Path:
        return p if not p.exists() else p.with_name(f"{p.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{p.suffix}")

    def export_json(self) -> None:
        if self.data is None or self.stats is None:
            QMessageBox.information(self, "提示", "请先加载文件")
            return
        out = self._unique(self.data.path.with_name(f"{self.data.path.stem}_viewer_stats.json"))
        with out.open("w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        self._set_status(f"JSON 导出完成：{out}")

    def export_csv(self) -> None:
        if self.data is None:
            QMessageBox.information(self, "提示", "请先加载文件")
            return
        cls = self._unique(self.data.path.with_name(f"{self.data.path.stem}_class_stats.csv"))
        ch = self._unique(self.data.path.with_name(f"{self.data.path.stem}_channel_stats.csv"))
        with cls.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["label", "class_key", "class_display", "total", "accepted", "rejected"])
            w.writeheader()
            w.writerows(class_rows(self.data))
        with ch.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["channel", "mean_uV", "std_uV", "rms_uV", "ptp_uV", "abs_mean_uV"])
            w.writeheader()
            w.writerows(channel_rows(self.data))
        self._set_status(f"CSV 导出完成：{cls.name} / {ch.name}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="可视化查看 MI 采集 npz 文件")
    p.add_argument("--npz", type=str, default="", help="指定一个 npz 文件路径")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    initial = Path(args.npz).resolve() if args.npz else None
    app = QApplication(sys.argv)
    app.setApplicationName("MI NPZ Viewer")
    app.setFont(QFont("Microsoft YaHei", 10))
    win = NPZViewer(initial=initial)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
