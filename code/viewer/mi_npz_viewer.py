# -*- coding: utf-8 -*-
"""Run-bundle viewer for collected MI data."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "custom_mi"
COLLECTION_MANIFEST_NAME = "collection_manifest.csv"
RUN_FILENAME_PATTERN = re.compile(
    r"sub-(?P<subject>.+?)_ses-(?P<session>.+?)_run-(?P<run>\d{3})_tpc-(?P<tpc>\d+)_n-(?P<n>\d+)_ok-(?P<ok>\d+)"
)
NPZ_PATTERNS = ("*_mi_epochs.npz",)
NPZ_EXCLUDED_SUFFIXES = ("_gate_epochs.npz", "_artifact_epochs.npz")
RUN_META_PATTERN = "*_session_meta.json"
CLASS_DISPLAY = {
    "left_hand": "Left Hand",
    "right_hand": "Right Hand",
    "feet": "Feet",
    "tongue": "Tongue",
    "no_control": "No Control",
}
SEGMENT_COLORS = {
    "baseline": "#cbd5e1",
    "cue": "#fbbf24",
    "imagery": "#60a5fa",
    "iti": "#bbf7d0",
    "continuous_prompt": "#fca5a5",
    "artifact_block": "#fda4af",
}
RUN_BUNDLE_SUFFIXES = (
    "_session_meta.json",
    "_quality_report.json",
    "_board_map.json",
    "_board_data.npy",
    "_events.csv",
    "_trials.csv",
    "_segments.csv",
    "_mi_epochs.meta.json",
    "_mi_epochs.npz",
    "_gate_epochs.meta.json",
    "_gate_epochs.npz",
    "_artifact_epochs.meta.json",
    "_artifact_epochs.npz",
    "_continuous.meta.json",
    "_continuous.npz",
    "_raw.fif",
)
RUN_FILE_FILTER = (
    "Run bundle files (*.json *.npz *.csv *.npy *.fif);;"
    "JSON files (*.json);;"
    "NPZ files (*.npz);;"
    "All files (*.*)"
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


@dataclass
class RunBundle:
    source_path: Path
    meta_path: Path
    session_dir: Path
    dataset_root: Path
    run_stem: str
    subject_id: str
    session_id: str
    save_index: int
    sampling_rate: float
    sample_count: int
    duration_sec: float
    channel_names: list[str]
    class_names: list[str]
    eeg_rows: list[int]
    marker_row: int | None
    timestamp_row: int | None
    board_data: np.ndarray | None
    meta: dict[str, Any]
    board_map: dict[str, Any]
    quality_summary: dict[str, Any]
    files_rel: dict[str, str]
    files: dict[str, Path]
    trials: list[dict[str, Any]]
    events: list[dict[str, Any]]
    segments: list[dict[str, Any]]
    mi_epochs: EpochData | None
    gate_summary: dict[str, Any]
    artifact_summary: dict[str, Any]
    continuous_summary: dict[str, Any]

    @property
    def trial_count(self) -> int:
        return int(len(self.trials))

    @property
    def accepted_trial_count(self) -> int:
        return int(sum(1 for row in self.trials if bool(row.get("accepted"))))

    @property
    def rejected_trial_count(self) -> int:
        return int(self.trial_count - self.accepted_trial_count)

    @property
    def event_count(self) -> int:
        return int(len(self.events))

    @property
    def segment_count(self) -> int:
        return int(len(self.segments))

    @property
    def board_shape(self) -> tuple[int, ...]:
        return tuple(int(item) for item in self.board_data.shape) if self.board_data is not None else (0, 0)


def _s(value: object) -> str:
    return value.decode("utf-8", errors="ignore").strip() if isinstance(value, bytes) else str(value).strip()


def _parse_int(value: object, default: int | None = None) -> int | None:
    text = _s(value)
    if not text:
        return default
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def _parse_float(value: object, default: float | None = None) -> float | None:
    text = _s(value)
    if not text:
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _parse_bool(value: object, default: bool | None = None) -> bool | None:
    parsed = _parse_int(value, None)
    if parsed is None:
        return default
    return bool(parsed)


def _normalize_unit(unit: str) -> str:
    token = unit.strip().lower()
    if token in {"v", "volt", "volts"}:
        return "volt"
    if token in {"uv", "microvolt", "microvolts"}:
        return "microvolt"
    return "unknown"


def _display_class(name: str) -> str:
    return CLASS_DISPLAY.get(name, name.replace("_", " ").title())


def _scaled_px(base_px: float, scale: float, min_px: int, max_px: int | None = None) -> int:
    value = int(round(float(base_px) * float(scale)))
    if max_px is not None:
        value = min(value, int(max_px))
    return max(int(min_px), value)


def _shape_of(data: np.lib.npyio.NpzFile, key: str) -> tuple[int, ...]:
    return tuple(int(item) for item in np.asarray(data[key]).shape) if key in data.files else ()


def _npz_text_array(data: np.lib.npyio.NpzFile, key: str) -> list[str]:
    if key not in data.files:
        return []
    return [_s(item) for item in np.asarray(data[key]).reshape(-1).tolist()]


def _count_strings(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in values:
        key = _s(item)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _safe_json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _load_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_trial_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "trial_id": _parse_int(row.get("trial_id"), 0) or 0,
                "save_index": _parse_int(row.get("save_index"), 0),
                "mi_run_index": _parse_int(row.get("mi_run_index"), 0),
                "run_trial_index": _parse_int(row.get("run_trial_index"), 0),
                "class_name": _s(row.get("class_name", "")),
                "display_name": _s(row.get("display_name", "")),
                "accepted": bool(_parse_int(row.get("accepted"), 0) or 0),
                "cue_onset_sample": _parse_int(row.get("cue_onset_sample"), None),
                "imagery_onset_sample": _parse_int(row.get("imagery_onset_sample"), None),
                "imagery_offset_sample": _parse_int(row.get("imagery_offset_sample"), None),
                "trial_end_sample": _parse_int(row.get("trial_end_sample"), None),
                "note": _s(row.get("note", "")),
            }
        )
    return out


def _normalize_event_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "event_index": _parse_int(row.get("event_index"), 0) or 0,
                "save_index": _parse_int(row.get("save_index"), 0),
                "event_name": _s(row.get("event_name", "")),
                "marker_code": _parse_int(row.get("marker_code"), None),
                "trial_id": _parse_int(row.get("trial_id"), None),
                "mi_run_index": _parse_int(row.get("mi_run_index"), None),
                "run_trial_index": _parse_int(row.get("run_trial_index"), None),
                "block_index": _parse_int(row.get("block_index"), None),
                "prompt_index": _parse_int(row.get("prompt_index"), None),
                "class_name": _s(row.get("class_name", "")),
                "command_duration_sec": _parse_float(row.get("command_duration_sec"), None),
                "execution_success": _parse_bool(row.get("execution_success"), None),
                "sample_index": _parse_int(row.get("sample_index"), None),
                "absolute_sample_index": _parse_int(row.get("absolute_sample_index"), None),
                "elapsed_sec": _parse_float(row.get("elapsed_sec"), None),
                "iso_time": _s(row.get("iso_time", "")),
            }
        )
    return out


def _normalize_segment_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "segment_id": _parse_int(row.get("segment_id"), 0) or 0,
                "save_index": _parse_int(row.get("save_index"), 0),
                "segment_type": _s(row.get("segment_type", "")),
                "label": _s(row.get("label", "")),
                "start_sample": _parse_int(row.get("start_sample"), None),
                "end_sample": _parse_int(row.get("end_sample"), None),
                "duration_sec": _parse_float(row.get("duration_sec"), 0.0) or 0.0,
                "trial_id": _parse_int(row.get("trial_id"), None),
                "mi_run_index": _parse_int(row.get("mi_run_index"), None),
                "run_trial_index": _parse_int(row.get("run_trial_index"), None),
                "block_index": _parse_int(row.get("block_index"), None),
                "prompt_index": _parse_int(row.get("prompt_index"), None),
                "accepted": _parse_bool(row.get("accepted"), None),
                "execution_success": _parse_bool(row.get("execution_success"), None),
                "source_start_event": _s(row.get("source_start_event", "")),
                "source_end_event": _s(row.get("source_end_event", "")),
            }
        )
    return out


def _derive_stem_from_path(path: Path) -> str:
    if path.name == "session_meta_latest.json":
        payload = _safe_json_load(path)
        return _s(payload.get("run_stem", ""))
    for suffix in sorted(RUN_BUNDLE_SUFFIXES, key=len, reverse=True):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return ""


def parse_run_text(path_or_text: Path | str) -> str:
    text = path_or_text.name if isinstance(path_or_text, Path) else str(path_or_text)
    stem = text
    for suffix in RUN_BUNDLE_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    matched = RUN_FILENAME_PATTERN.search(stem)
    if not matched:
        return stem
    parts = matched.groupdict()
    return (
        f"sub-{parts['subject']} | ses-{parts['session']} | run-{parts['run']} | "
        f"tpc={parts['tpc']} | n={parts['n']} | ok={parts['ok']}"
    )


def discover_epoch_files(dataset_root: Path) -> list[Path]:
    """Backward-compatible discovery for MI epoch files."""
    if not dataset_root.exists():
        return []
    out: list[Path] = []
    seen: set[Path] = set()
    for pattern in NPZ_PATTERNS:
        for path in dataset_root.rglob(pattern):
            if any(str(path.name).endswith(suffix) for suffix in NPZ_EXCLUDED_SUFFIXES):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(resolved)
    out.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return out


def discover_run_bundles(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        return []
    out: list[Path] = []
    seen: set[Path] = set()
    for path in dataset_root.rglob(RUN_META_PATTERN):
        if path.name == "session_meta_latest.json":
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    out.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return out


def _resolve_bundle_meta_path(source: Path) -> Path:
    path = source.resolve()
    if path.is_dir():
        meta_files = discover_run_bundles(path)
        if meta_files:
            return meta_files[0]
        latest_meta = path / "session_meta_latest.json"
        if latest_meta.exists():
            return _resolve_bundle_meta_path(latest_meta)
        raise FileNotFoundError(f"No run bundle metadata found under directory: {path}")

    if path.name == "session_meta_latest.json":
        run_stem = _derive_stem_from_path(path)
        candidate = path.parent / f"{run_stem}_session_meta.json"
        return candidate if run_stem and candidate.exists() else path

    if path.name.endswith("_session_meta.json"):
        return path

    stem = _derive_stem_from_path(path)
    if stem:
        candidate = path.parent / f"{stem}_session_meta.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Unable to resolve run bundle metadata from path. Select a session directory, "
        "`*_session_meta.json`, or any artifact from the same run bundle."
    )


def load_epochs_npz(path: Path) -> EpochData:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "X_mi" not in data.files or "y_mi" not in data.files:
            raise KeyError("MI epoch file must contain X_mi and y_mi.")
        X = np.asarray(data["X_mi"], dtype=np.float32)
        y = np.asarray(data["y_mi"], dtype=np.int64).reshape(-1)
        if X.ndim != 3 or y.shape[0] != X.shape[0]:
            raise ValueError(f"Invalid X/y shape: X={X.shape}, y={y.shape}")
        accepted = np.ones(X.shape[0], dtype=np.int8)
        trial_ids = (
            np.asarray(data["mi_trial_ids"], dtype=np.int64).reshape(-1)
            if "mi_trial_ids" in data.files
            else np.arange(1, X.shape[0] + 1, dtype=np.int64)
        )
        if trial_ids.shape[0] != X.shape[0]:
            trial_ids = np.arange(1, X.shape[0] + 1, dtype=np.int64)
        channel_names = (
            [_s(item) for item in np.asarray(data["channel_names"]).reshape(-1).tolist()]
            if "channel_names" in data.files
            else [f"EEG{i + 1}" for i in range(X.shape[1])]
        )
        class_names = _npz_text_array(data, "class_names")
        if len(channel_names) != X.shape[1]:
            channel_names = [f"EEG{i + 1}" for i in range(X.shape[1])]
        max_label = int(np.max(y)) if y.size else -1
        while len(class_names) <= max_label:
            class_names.append(f"class_{len(class_names)}")
        sampling_rate = float(np.asarray(data["sampling_rate"]).reshape(-1)[0]) if "sampling_rate" in data.files else 250.0
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
        sampling_rate=sampling_rate,
        source_signal_unit=unit,
    )


def _load_gate_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "path": "",
            "available": False,
            "pos_shape": (),
            "neg_shape": (),
            "hard_neg_shape": (),
            "neg_source_counts": {},
            "hard_neg_source_counts": {},
        }
    with np.load(path, allow_pickle=False) as data:
        neg_sources = _npz_text_array(data, "gate_neg_sources")
        hard_sources = _npz_text_array(data, "gate_hard_neg_sources")
        return {
            "path": str(path),
            "available": True,
            "pos_shape": _shape_of(data, "X_gate_pos"),
            "neg_shape": _shape_of(data, "X_gate_neg"),
            "hard_neg_shape": _shape_of(data, "X_gate_hard_neg"),
            "neg_source_counts": _count_strings(neg_sources),
            "hard_neg_source_counts": _count_strings(hard_sources),
        }


def _load_artifact_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "path": "",
            "available": False,
            "shape": (),
            "label_counts": {},
        }
    with np.load(path, allow_pickle=False) as data:
        labels = _npz_text_array(data, "artifact_labels")
        return {
            "path": str(path),
            "available": True,
            "shape": _shape_of(data, "X_artifact"),
            "label_counts": _count_strings(labels),
        }


def _load_continuous_summary(path: Path | None, sampling_rate: float) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "path": "",
            "available": False,
            "shape": (),
            "block_rows": [],
            "prompt_rows": [],
            "label_counts": {},
        }
    with np.load(path, allow_pickle=False) as data:
        shape = _shape_of(data, "X_continuous")
        labels = _npz_text_array(data, "continuous_event_labels")
        start_samples = np.asarray(data["continuous_event_samples"], dtype=np.int64).reshape(-1) if "continuous_event_samples" in data.files else np.asarray([], dtype=np.int64)
        end_samples = np.asarray(data["continuous_event_end_samples"], dtype=np.int64).reshape(-1) if "continuous_event_end_samples" in data.files else np.asarray([], dtype=np.int64)
        block_indices = np.asarray(data["continuous_block_indices"], dtype=np.int64).reshape(-1) if "continuous_block_indices" in data.files else np.asarray([], dtype=np.int64)
        prompt_indices = np.asarray(data["continuous_prompt_indices"], dtype=np.int64).reshape(-1) if "continuous_prompt_indices" in data.files else np.asarray([], dtype=np.int64)
        execution_success = np.asarray(data["continuous_execution_success"], dtype=np.int64).reshape(-1) if "continuous_execution_success" in data.files else np.asarray([], dtype=np.int64)
        durations = np.asarray(data["continuous_command_duration_sec"], dtype=np.float32).reshape(-1) if "continuous_command_duration_sec" in data.files else np.asarray([], dtype=np.float32)
        block_starts = np.asarray(data["continuous_block_start_samples"], dtype=np.int64).reshape(-1) if "continuous_block_start_samples" in data.files else np.asarray([], dtype=np.int64)
        block_ends = np.asarray(data["continuous_block_end_samples"], dtype=np.int64).reshape(-1) if "continuous_block_end_samples" in data.files else np.asarray([], dtype=np.int64)

    prompt_rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        start_sample = int(start_samples[index]) if index < start_samples.shape[0] else None
        end_sample = int(end_samples[index]) if index < end_samples.shape[0] else None
        duration_sec = float(durations[index]) if index < durations.shape[0] and np.isfinite(float(durations[index])) else None
        if duration_sec is None and start_sample is not None and end_sample is not None and sampling_rate > 0:
            duration_sec = float((end_sample - start_sample) / sampling_rate)
        prompt_rows.append(
            {
                "row_index": index,
                "block_index": int(block_indices[index]) if index < block_indices.shape[0] else None,
                "prompt_index": int(prompt_indices[index]) if index < prompt_indices.shape[0] else None,
                "label": label,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "duration_sec": duration_sec,
                "execution_success": (
                    None
                    if index >= execution_success.shape[0] or int(execution_success[index]) < 0
                    else bool(int(execution_success[index]))
                ),
            }
        )

    block_rows: list[dict[str, Any]] = []
    for index in range(min(block_starts.shape[0], block_ends.shape[0])):
        start_sample = int(block_starts[index])
        end_sample = int(block_ends[index])
        block_rows.append(
            {
                "block_index": index + 1,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "duration_sec": float((end_sample - start_sample) / sampling_rate) if sampling_rate > 0 else 0.0,
            }
        )
    return {
        "path": str(path),
        "available": True,
        "shape": shape,
        "block_rows": block_rows,
        "prompt_rows": prompt_rows,
        "label_counts": _count_strings(labels),
    }


def _bundle_file_paths(meta_path: Path, meta: dict[str, Any], run_stem: str) -> tuple[Path, dict[str, str], dict[str, Path]]:
    session_dir = meta_path.parent
    dataset_root = meta_path.parents[2] if len(meta_path.parents) >= 3 else session_dir
    files_rel = {str(key): _s(value) for key, value in dict(meta.get("files", {})).items() if _s(value)}
    files: dict[str, Path] = {}
    for key, relpath in files_rel.items():
        candidate = (dataset_root / relpath).resolve()
        if candidate.exists():
            files[key] = candidate
            continue
        fallback = session_dir / Path(relpath).name
        if fallback.exists():
            files[key] = fallback.resolve()
    derived_names = {
        "board_data_npy": f"{run_stem}_board_data.npy",
        "board_map_json": f"{run_stem}_board_map.json",
        "session_raw_fif": f"{run_stem}_raw.fif",
        "events_csv": f"{run_stem}_events.csv",
        "trials_csv": f"{run_stem}_trials.csv",
        "segments_csv": f"{run_stem}_segments.csv",
        "session_meta_json": f"{run_stem}_session_meta.json",
        "quality_report_json": f"{run_stem}_quality_report.json",
        "mi_epochs_npz": f"{run_stem}_mi_epochs.npz",
        "mi_epochs_meta_json": f"{run_stem}_mi_epochs.meta.json",
        "gate_epochs_npz": f"{run_stem}_gate_epochs.npz",
        "gate_epochs_meta_json": f"{run_stem}_gate_epochs.meta.json",
        "artifact_epochs_npz": f"{run_stem}_artifact_epochs.npz",
        "artifact_epochs_meta_json": f"{run_stem}_artifact_epochs.meta.json",
        "continuous_npz": f"{run_stem}_continuous.npz",
        "continuous_meta_json": f"{run_stem}_continuous.meta.json",
    }
    for key, filename in derived_names.items():
        candidate = session_dir / filename
        if candidate.exists() and key not in files:
            files[key] = candidate.resolve()
    return dataset_root, files_rel, files


def load_run_bundle(source: Path) -> RunBundle:
    meta_path = _resolve_bundle_meta_path(source)
    meta = _safe_json_load(meta_path)
    run_stem = _s(meta.get("run_stem", "")) or _derive_stem_from_path(meta_path)
    dataset_root, files_rel, files = _bundle_file_paths(meta_path, meta, run_stem)
    board_map = _safe_json_load(files["board_map_json"]) if "board_map_json" in files else {}
    quality_summary = _safe_json_load(files["quality_report_json"]) if "quality_report_json" in files else dict(meta.get("quality_report", {}))
    board_data = np.load(files["board_data_npy"], mmap_mode="r", allow_pickle=False) if "board_data_npy" in files else None
    trials = _normalize_trial_rows(_load_csv_dicts(files["trials_csv"])) if "trials_csv" in files else []
    events = _normalize_event_rows(_load_csv_dicts(files["events_csv"])) if "events_csv" in files else []
    segments = _normalize_segment_rows(_load_csv_dicts(files["segments_csv"])) if "segments_csv" in files else []
    mi_epochs = load_epochs_npz(files["mi_epochs_npz"]) if "mi_epochs_npz" in files else None

    channel_names = list(meta.get("session", {}).get("channel_names", []))
    if not channel_names and mi_epochs is not None:
        channel_names = list(mi_epochs.channel_names)
    if not channel_names and isinstance(board_map.get("channel_rows"), list):
        channel_names = [_s(item.get("channel_name", "")) for item in board_map.get("channel_rows", []) if _s(item.get("channel_name", ""))]

    class_names = list(mi_epochs.class_names) if mi_epochs is not None else []
    if not class_names:
        seen_classes: list[str] = []
        for row in trials:
            class_name = _s(row.get("class_name", ""))
            if class_name and class_name not in seen_classes:
                seen_classes.append(class_name)
        class_names = seen_classes

    eeg_rows = [int(item) for item in list(meta.get("selected_eeg_rows", [])) or list(board_map.get("selected_eeg_rows", []))]
    if not eeg_rows and board_data is not None:
        eeg_rows = list(range(len(channel_names) or max(0, int(board_data.shape[0]) - 1)))
    if len(channel_names) != len(eeg_rows):
        channel_names = [f"EEG{i + 1}" for i in range(len(eeg_rows))]

    sampling_rate = float(meta.get("sampling_rate_hz") or (mi_epochs.sampling_rate if mi_epochs is not None else 250.0))
    sample_count = int(meta.get("sample_count") or (board_data.shape[1] if board_data is not None else 0))
    duration_sec = float(meta.get("duration_sec") or (sample_count / sampling_rate if sampling_rate > 0 else 0.0))

    return RunBundle(
        source_path=source.resolve(),
        meta_path=meta_path.resolve(),
        session_dir=meta_path.parent.resolve(),
        dataset_root=dataset_root.resolve(),
        run_stem=run_stem,
        subject_id=_s(meta.get("subject_id", "")),
        session_id=_s(meta.get("session_id", "")),
        save_index=int(meta.get("save_index") or 0),
        sampling_rate=sampling_rate,
        sample_count=sample_count,
        duration_sec=duration_sec,
        channel_names=channel_names,
        class_names=class_names,
        eeg_rows=eeg_rows,
        marker_row=_parse_int(meta.get("marker_row"), None),
        timestamp_row=_parse_int(meta.get("timestamp_row"), None),
        board_data=board_data,
        meta=meta,
        board_map=board_map,
        quality_summary=quality_summary,
        files_rel=files_rel,
        files=files,
        trials=trials,
        events=events,
        segments=segments,
        mi_epochs=mi_epochs,
        gate_summary=_load_gate_summary(files.get("gate_epochs_npz")),
        artifact_summary=_load_artifact_summary(files.get("artifact_epochs_npz")),
        continuous_summary=_load_continuous_summary(files.get("continuous_npz"), sampling_rate=sampling_rate),
    )


def class_rows(bundle: RunBundle) -> list[dict[str, Any]]:
    class_keys = list(bundle.class_names)
    seen = set(class_keys)
    for row in bundle.trials:
        class_name = _s(row.get("class_name", ""))
        if class_name and class_name not in seen:
            class_keys.append(class_name)
            seen.add(class_name)
    out: list[dict[str, Any]] = []
    for label, class_name in enumerate(class_keys):
        total = int(sum(1 for row in bundle.trials if _s(row.get("class_name", "")) == class_name))
        accepted = int(sum(1 for row in bundle.trials if _s(row.get("class_name", "")) == class_name and bool(row.get("accepted"))))
        out.append(
            {
                "label": label,
                "class_key": class_name,
                "class_display": _display_class(class_name),
                "total": total,
                "accepted": accepted,
                "rejected": total - accepted,
            }
        )
    return out


def channel_rows(bundle: RunBundle) -> list[dict[str, Any]]:
    board_rows: dict[str, dict[str, Any]] = {}
    if bundle.board_data is not None and bundle.eeg_rows:
        eeg = np.asarray(bundle.board_data[bundle.eeg_rows, :], dtype=np.float32)
        for index, channel_name in enumerate(bundle.channel_names):
            values = eeg[index].reshape(-1)
            board_rows[channel_name] = {
                "channel": channel_name,
                "mean_uV": float(np.mean(values)) if values.size else 0.0,
                "std_uV": float(np.std(values)) if values.size else 0.0,
                "rms_uV": float(np.sqrt(np.mean(values**2))) if values.size else 0.0,
                "ptp_uV": float(np.ptp(values)) if values.size else 0.0,
                "abs_mean_uV": float(np.mean(np.abs(values))) if values.size else 0.0,
            }

    channels = bundle.quality_summary.get("channels", []) if isinstance(bundle.quality_summary, dict) else []
    if isinstance(channels, list) and channels:
        rows: list[dict[str, Any]] = []
        seen_channels: set[str] = set()
        for item in channels:
            if not isinstance(item, dict):
                continue
            channel_name = _s(item.get("channel_name", ""))
            if not channel_name:
                continue
            board_row = board_rows.get(channel_name, {})
            rows.append(
                {
                    "channel": channel_name,
                    "mean_uV": float(item.get("mean_uV", board_row.get("mean_uV", 0.0)) or 0.0),
                    "std_uV": float(item.get("std_uV", board_row.get("std_uV", 0.0)) or 0.0),
                    "rms_uV": float(item.get("rms_uV", board_row.get("rms_uV", 0.0)) or 0.0),
                    "ptp_uV": float(item.get("peak_to_peak_uV", item.get("ptp_uV", board_row.get("ptp_uV", 0.0))) or 0.0),
                    "abs_mean_uV": float(item.get("abs_mean_uV", board_row.get("abs_mean_uV", 0.0)) or 0.0),
                }
            )
            seen_channels.add(channel_name)
        for channel_name, board_row in board_rows.items():
            if channel_name in seen_channels:
                continue
            rows.append(dict(board_row))
        if rows:
            return rows

    return list(board_rows.values())


def segment_summary_rows(bundle: RunBundle) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for row in bundle.segments:
        segment_type = _s(row.get("segment_type", ""))
        if not segment_type:
            continue
        bucket = buckets.setdefault(
            segment_type,
            {
                "segment_type": segment_type,
                "count": 0,
                "total_duration_sec": 0.0,
                "accepted_count": 0,
                "rejected_count": 0,
            },
        )
        bucket["count"] += 1
        bucket["total_duration_sec"] += float(row.get("duration_sec") or 0.0)
        accepted = row.get("accepted")
        if accepted is True:
            bucket["accepted_count"] += 1
        elif accepted is False:
            bucket["rejected_count"] += 1
    rows = list(buckets.values())
    rows.sort(key=lambda item: str(item["segment_type"]))
    return rows


def prompt_rows(bundle: RunBundle) -> list[dict[str, Any]]:
    rows = list(bundle.continuous_summary.get("prompt_rows", []))
    if rows:
        return rows
    out: list[dict[str, Any]] = []
    for row in bundle.segments:
        if _s(row.get("segment_type", "")) != "continuous_prompt":
            continue
        out.append(
            {
                "row_index": len(out),
                "block_index": row.get("block_index"),
                "prompt_index": row.get("prompt_index"),
                "label": _s(row.get("label", "")),
                "start_sample": row.get("start_sample"),
                "end_sample": row.get("end_sample"),
                "duration_sec": row.get("duration_sec"),
                "execution_success": row.get("execution_success"),
            }
        )
    return out


def file_rows(bundle: RunBundle) -> list[dict[str, Any]]:
    order = [
        "session_meta_json",
        "quality_report_json",
        "board_map_json",
        "board_data_npy",
        "events_csv",
        "trials_csv",
        "segments_csv",
        "mi_epochs_npz",
        "gate_epochs_npz",
        "artifact_epochs_npz",
        "continuous_npz",
        "session_raw_fif",
    ]
    rows: list[dict[str, Any]] = []
    for key in order:
        relpath = bundle.files_rel.get(key, "")
        abspath = bundle.files.get(key)
        rows.append(
            {
                "file_key": key,
                "status": "OK" if abspath is not None and abspath.exists() else "Missing",
                "relative_path": relpath,
                "absolute_path": str(abspath) if abspath is not None else "",
            }
        )
    return rows


def _bundle_overview(bundle: RunBundle) -> dict[str, Any]:
    return {
        "run_text": parse_run_text(bundle.run_stem),
        "trial_count": bundle.trial_count,
        "accepted_trial_count": bundle.accepted_trial_count,
        "rejected_trial_count": bundle.rejected_trial_count,
        "event_count": bundle.event_count,
        "segment_count": bundle.segment_count,
        "sampling_rate_hz": bundle.sampling_rate,
        "duration_sec": bundle.duration_sec,
        "sample_count": bundle.sample_count,
        "board_shape": bundle.board_shape,
        "mi_epochs_shape": () if bundle.mi_epochs is None else tuple(int(item) for item in bundle.mi_epochs.X_uV.shape),
        "gate_pos_shape": bundle.gate_summary.get("pos_shape", ()),
        "gate_neg_shape": bundle.gate_summary.get("neg_shape", ()),
        "gate_hard_neg_shape": bundle.gate_summary.get("hard_neg_shape", ()),
        "artifact_shape": bundle.artifact_summary.get("shape", ()),
        "continuous_shape": bundle.continuous_summary.get("shape", ()),
        "continuous_prompt_count": len(prompt_rows(bundle)),
    }


def build_stats_payload(bundle: RunBundle) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "bundle": {
            "source_path": str(bundle.source_path),
            "meta_path": str(bundle.meta_path),
            "session_dir": str(bundle.session_dir),
            "dataset_root": str(bundle.dataset_root),
            "run_stem": bundle.run_stem,
            "subject_id": bundle.subject_id,
            "session_id": bundle.session_id,
            "save_index": bundle.save_index,
        },
        "overview": _bundle_overview(bundle),
        "class_stats": class_rows(bundle),
        "segment_summary": segment_summary_rows(bundle),
        "channel_stats": channel_rows(bundle),
        "trials": bundle.trials,
        "segments": bundle.segments,
        "events": bundle.events,
        "prompt_rows": prompt_rows(bundle),
        "gate_summary": bundle.gate_summary,
        "artifact_summary": bundle.artifact_summary,
        "continuous_summary": {
            key: value
            for key, value in bundle.continuous_summary.items()
            if key != "path"
        },
        "files": file_rows(bundle),
    }


def _write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _extract_bundle_signal(bundle: RunBundle, start_sample: int, end_sample: int, channels: list[int]) -> np.ndarray:
    if bundle.board_data is None or not bundle.eeg_rows:
        return np.empty((0, 0), dtype=np.float32)
    total_samples = int(bundle.board_data.shape[1])
    start_index = max(0, int(start_sample))
    stop_index = min(total_samples, max(int(end_sample), start_index + 1))
    eeg_rows = [bundle.eeg_rows[index] for index in channels if 0 <= index < len(bundle.eeg_rows)]
    if not eeg_rows:
        return np.empty((0, 0), dtype=np.float32)
    return np.asarray(bundle.board_data[eeg_rows, start_index:stop_index], dtype=np.float32)


class RunBundleViewer(QMainWindow):
    def __init__(self, initial: Path | None = None) -> None:
        super().__init__()
        self.bundle: RunBundle | None = None
        self.stats_payload: dict[str, Any] | None = None
        self._ui_scale = 1.0
        self._class_rows: list[dict[str, Any]] = []
        self._segment_summary_rows: list[dict[str, Any]] = []
        self._channel_rows: list[dict[str, Any]] = []
        self._trial_rows: list[dict[str, Any]] = []
        self._segment_rows: list[dict[str, Any]] = []
        self._prompt_rows: list[dict[str, Any]] = []
        self._event_rows: list[dict[str, Any]] = []
        self._file_rows: list[dict[str, Any]] = []
        self.setWindowTitle("MI Collection Viewer")
        self.resize(1640, 980)
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
        container = QWidget()
        self.setCentralWidget(container)
        root = QVBoxLayout(container)

        file_group = QGroupBox("1) Run Bundle")
        file_grid = QGridLayout(file_group)
        self.root_edit = QLineEdit(str(DEFAULT_DATASET_ROOT))
        self.scan_btn = QPushButton("Scan")
        self.file_combo = QComboBox()
        self.load_selected_btn = QPushButton("Load Selected")
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select session_meta.json, run directory, or any run artifact")
        self.browse_btn = QPushButton("Browse")
        self.load_btn = QPushButton("Load")
        file_grid.addWidget(QLabel("Dataset Root"), 0, 0)
        file_grid.addWidget(self.root_edit, 0, 1)
        file_grid.addWidget(self.scan_btn, 0, 2)
        file_grid.addWidget(QLabel("Discovered Runs"), 1, 0)
        file_grid.addWidget(self.file_combo, 1, 1)
        file_grid.addWidget(self.load_selected_btn, 1, 2)
        file_grid.addWidget(QLabel("Direct Path"), 2, 0)
        file_grid.addWidget(self.path_edit, 2, 1)
        file_grid.addWidget(self.browse_btn, 2, 2)
        file_grid.addWidget(self.load_btn, 2, 3)
        root.addWidget(file_group)

        summary_layout = QGridLayout()
        overview_group = QGroupBox("2) Overview")
        overview_form = QFormLayout(overview_group)
        self.lb_file = QLabel("--")
        self.lb_file.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lb_run = QLabel("--")
        self.lb_subject = QLabel("--")
        self.lb_save = QLabel("--")
        self.lb_duration = QLabel("--")
        self.lb_board = QLabel("--")
        self.lb_trials = QLabel("--")
        self.lb_events = QLabel("--")
        self.lb_derivatives = QLabel("--")
        overview_form.addRow("Source", self.lb_file)
        overview_form.addRow("Run", self.lb_run)
        overview_form.addRow("Subject / Session", self.lb_subject)
        overview_form.addRow("Save Index", self.lb_save)
        overview_form.addRow("Duration", self.lb_duration)
        overview_form.addRow("Board", self.lb_board)
        overview_form.addRow("Trials", self.lb_trials)
        overview_form.addRow("Events / Segments", self.lb_events)
        overview_form.addRow("Derivatives", self.lb_derivatives)

        class_group = QGroupBox("3) Class Stats")
        class_layout = QVBoxLayout(class_group)
        self.class_table = self._new_table(["Label", "Class", "Total", "Accepted", "Rejected"])
        class_layout.addWidget(self.class_table)

        segment_group = QGroupBox("4) Segment Summary")
        segment_layout = QVBoxLayout(segment_group)
        self.segment_summary_table = self._new_table(["Type", "Count", "Duration(s)", "Accepted", "Rejected"])
        segment_layout.addWidget(self.segment_summary_table)

        summary_layout.addWidget(overview_group, 0, 0)
        summary_layout.addWidget(class_group, 0, 1)
        summary_layout.addWidget(segment_group, 0, 2)
        root.addLayout(summary_layout)

        splitter = QSplitter(Qt.Vertical)
        root.addWidget(splitter, stretch=1)

        tabs = QTabWidget()
        splitter.addWidget(tabs)

        self.trial_table = self._new_table(["Trial", "Run", "RunTrial", "Class", "Accepted", "Cue", "Imagery", "TrialEnd", "Note"])
        tabs.addTab(self.trial_table, "Trials")

        self.segment_table = self._new_table(
            ["ID", "Type", "Label", "Start", "End", "Dur(s)", "Trial", "Run", "Block", "Prompt", "Accepted", "Success", "SourceStart", "SourceEnd"]
        )
        tabs.addTab(self.segment_table, "Segments")

        self.prompt_table = self._new_table(["Row", "Block", "Prompt", "Label", "Start", "End", "Dur(s)", "Success"])
        tabs.addTab(self.prompt_table, "Continuous")

        self.event_table = self._new_table(["Idx", "Event", "Code", "Sample", "Trial", "Run", "Block", "Prompt", "Class", "Success", "ISO Time"])
        tabs.addTab(self.event_table, "Events")

        self.channel_table = self._new_table(["Channel", "Mean", "Std", "RMS", "PtP", "AbsMean"])
        tabs.addTab(self.channel_table, "Channels")

        self.file_table = self._new_table(["Key", "Status", "Relative Path", "Absolute Path"])
        tabs.addTab(self.file_table, "Files")

        plot_group = QGroupBox("5) Plot")
        plot_layout = QVBoxLayout(plot_group)
        plot_ctrl = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Selected Trial (continuous)", "selected_trial")
        self.mode_combo.addItem("Selected Segment (continuous)", "selected_segment")
        self.mode_combo.addItem("Selected Prompt (continuous)", "selected_prompt")
        self.mode_combo.addItem("Selected Trial (MI epoch)", "mi_epoch")
        self.mode_combo.addItem("Class Mean (MI)", "class_mean")
        self.mode_combo.addItem("Class PSD (MI)", "class_psd")
        self.class_combo = QComboBox()
        self.class_combo.addItem("All Classes", -1)
        self.ch_combo = QComboBox()
        self.ch_combo.addItem("All Channels (stacked)", -1)
        self.context_spin = QDoubleSpinBox()
        self.context_spin.setRange(0.0, 10.0)
        self.context_spin.setDecimals(2)
        self.context_spin.setSingleStep(0.25)
        self.context_spin.setValue(0.5)
        self.ck_ok = QCheckBox("Accepted Only (MI aggregate)")
        self.ck_ok.setChecked(True)
        self.ck_demean = QCheckBox("Demean")
        self.ck_demean.setChecked(True)
        self.ck_annotations = QCheckBox("Show Events / Segments")
        self.ck_annotations.setChecked(True)
        self.replot_btn = QPushButton("Refresh")
        self.exp_json_btn = QPushButton("Export JSON")
        self.exp_csv_btn = QPushButton("Export CSV")
        for widget in [
            QLabel("Mode"),
            self.mode_combo,
            QLabel("Class"),
            self.class_combo,
            QLabel("Channel"),
            self.ch_combo,
            QLabel("Context(s)"),
            self.context_spin,
            self.ck_ok,
            self.ck_demean,
            self.ck_annotations,
            self.replot_btn,
            self.exp_json_btn,
            self.exp_csv_btn,
        ]:
            plot_ctrl.addWidget(widget)
        plot_ctrl.addStretch(1)
        plot_layout.addLayout(plot_ctrl)

        self.fig = Figure(figsize=(12, 6), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas, stretch=1)
        splitter.addWidget(plot_group)
        splitter.setSizes([430, 430])

        self.status = QLabel("Ready")
        root.addWidget(self.status)

        self.scan_btn.clicked.connect(self.scan_files)
        self.load_selected_btn.clicked.connect(self.load_selected)
        self.browse_btn.clicked.connect(self.browse_file)
        self.load_btn.clicked.connect(self.load_file)
        self.mode_combo.currentIndexChanged.connect(self._on_ctrl_changed)
        self.class_combo.currentIndexChanged.connect(self._on_ctrl_changed)
        self.ch_combo.currentIndexChanged.connect(self._on_ctrl_changed)
        self.context_spin.valueChanged.connect(self._on_ctrl_changed)
        self.ck_ok.stateChanged.connect(self._on_ctrl_changed)
        self.ck_demean.stateChanged.connect(self._on_ctrl_changed)
        self.ck_annotations.stateChanged.connect(self._on_ctrl_changed)
        self.replot_btn.clicked.connect(self.refresh_plot)
        self.exp_json_btn.clicked.connect(self.export_json)
        self.exp_csv_btn.clicked.connect(self.export_csv)
        self.trial_table.itemSelectionChanged.connect(self._on_ctrl_changed)
        self.segment_table.itemSelectionChanged.connect(self._on_ctrl_changed)
        self.prompt_table.itemSelectionChanged.connect(self._on_ctrl_changed)
        self.trial_table.itemDoubleClicked.connect(lambda _item: self._set_mode("selected_trial"))
        self.segment_table.itemDoubleClicked.connect(lambda _item: self._set_mode("selected_segment"))
        self.prompt_table.itemDoubleClicked.connect(lambda _item: self._set_mode("selected_prompt"))

    def _new_table(self, headers: list[str]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        return table

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
        return max(0.82, min(1.8, min(width / 1640.0, height / 980.0)))

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
        path, _ = QFileDialog.getOpenFileName(self, "Select run bundle file", str(start), RUN_FILE_FILTER)
        if path:
            self.path_edit.setText(path)

    def scan_files(self) -> None:
        dataset_root = Path(self.root_edit.text().strip() or str(DEFAULT_DATASET_ROOT))
        runs = discover_run_bundles(dataset_root)
        self.file_combo.clear()
        for path in runs:
            timestamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            self.file_combo.addItem(f"{path.stem} | {timestamp}", str(path))
        self._set_status(f"Scan complete: {len(runs)} run bundle(s)")

    def load_selected(self) -> None:
        value = self.file_combo.currentData()
        if value:
            self.path_edit.setText(str(value))
            self.load_file()

    def load_file(self) -> None:
        raw_value = self.path_edit.text().strip()
        if not raw_value:
            QMessageBox.warning(self, "Select Input", "Choose a run bundle path first.")
            return
        try:
            self.bundle = load_run_bundle(Path(raw_value))
        except Exception as error:  # pragma: no cover - UI path
            QMessageBox.critical(self, "Load Failed", str(error))
            self._set_status(f"Load failed: {error}")
            return
        self._refresh_stats()
        self._refresh_controls()
        self.refresh_plot()
        self._set_status(f"Loaded: {self.bundle.run_stem}")

    def _refresh_stats(self) -> None:
        assert self.bundle is not None
        bundle = self.bundle
        overview = _bundle_overview(bundle)
        self._class_rows = class_rows(bundle)
        self._segment_summary_rows = segment_summary_rows(bundle)
        self._channel_rows = channel_rows(bundle)
        self._trial_rows = list(bundle.trials)
        self._segment_rows = list(bundle.segments)
        self._prompt_rows = prompt_rows(bundle)
        self._event_rows = list(bundle.events)
        self._file_rows = file_rows(bundle)
        self.stats_payload = build_stats_payload(bundle)

        self.lb_file.setText(str(bundle.source_path))
        self.lb_run.setText(str(overview["run_text"]))
        self.lb_subject.setText(f"{bundle.subject_id} / {bundle.session_id}")
        self.lb_save.setText(str(bundle.save_index))
        self.lb_duration.setText(f"{bundle.duration_sec:.2f} s | {bundle.sample_count} samples | {bundle.sampling_rate:.3f} Hz")
        self.lb_board.setText(f"{bundle.board_shape} | EEG rows={bundle.eeg_rows} | marker={bundle.marker_row}")
        self.lb_trials.setText(f"{bundle.trial_count} total | {bundle.accepted_trial_count} accepted | {bundle.rejected_trial_count} rejected")
        self.lb_events.setText(f"{bundle.event_count} events | {bundle.segment_count} segments")
        self.lb_derivatives.setText(
            f"MI={overview['mi_epochs_shape']} | Gate={overview['gate_neg_shape']} | "
            f"Artifact={overview['artifact_shape']} | Continuous={overview['continuous_shape']}"
        )

        self._fill_class_table()
        self._fill_segment_summary_table()
        self._fill_trial_table()
        self._fill_segment_table()
        self._fill_prompt_table()
        self._fill_event_table()
        self._fill_channel_table()
        self._fill_file_table()

    def _fill_class_table(self) -> None:
        self.class_table.setRowCount(len(self._class_rows))
        for row_index, row in enumerate(self._class_rows):
            self._set_row(
                self.class_table,
                row_index,
                [str(row["label"]), str(row["class_display"]), str(row["total"]), str(row["accepted"]), str(row["rejected"])],
                center_columns={0, 2, 3, 4},
            )

    def _fill_segment_summary_table(self) -> None:
        self.segment_summary_table.setRowCount(len(self._segment_summary_rows))
        for row_index, row in enumerate(self._segment_summary_rows):
            self._set_row(
                self.segment_summary_table,
                row_index,
                [str(row["segment_type"]), str(row["count"]), f"{float(row['total_duration_sec']):.3f}", str(row["accepted_count"]), str(row["rejected_count"])],
                center_columns={1, 2, 3, 4},
            )

    def _fill_trial_table(self) -> None:
        self.trial_table.setRowCount(len(self._trial_rows))
        for row_index, row in enumerate(self._trial_rows):
            self._set_row(
                self.trial_table,
                row_index,
                [
                    str(row["trial_id"]),
                    str(row.get("mi_run_index") or ""),
                    str(row.get("run_trial_index") or ""),
                    _display_class(_s(row.get("class_name", ""))),
                    "Yes" if bool(row.get("accepted")) else "No",
                    str(row.get("cue_onset_sample") or ""),
                    str(row.get("imagery_onset_sample") or ""),
                    str(row.get("trial_end_sample") or ""),
                    _s(row.get("note", "")),
                ],
                center_columns={0, 1, 2, 4, 5, 6, 7},
            )
        if self._trial_rows and self.trial_table.currentRow() < 0:
            self.trial_table.selectRow(0)

    def _fill_segment_table(self) -> None:
        self.segment_table.setRowCount(len(self._segment_rows))
        for row_index, row in enumerate(self._segment_rows):
            self._set_row(
                self.segment_table,
                row_index,
                [
                    str(row["segment_id"]),
                    str(row["segment_type"]),
                    str(row["label"]),
                    str(row.get("start_sample") or ""),
                    str(row.get("end_sample") or ""),
                    f"{float(row['duration_sec']):.3f}",
                    str(row.get("trial_id") or ""),
                    str(row.get("mi_run_index") or ""),
                    str(row.get("block_index") or ""),
                    str(row.get("prompt_index") or ""),
                    "" if row.get("accepted") is None else ("Yes" if bool(row.get("accepted")) else "No"),
                    "" if row.get("execution_success") is None else ("Yes" if bool(row.get("execution_success")) else "No"),
                    str(row.get("source_start_event") or ""),
                    str(row.get("source_end_event") or ""),
                ],
                center_columns={0, 3, 4, 5, 6, 7, 8, 9, 10, 11},
            )
        if self._segment_rows and self.segment_table.currentRow() < 0:
            self.segment_table.selectRow(0)

    def _fill_prompt_table(self) -> None:
        self.prompt_table.setRowCount(len(self._prompt_rows))
        for row_index, row in enumerate(self._prompt_rows):
            self._set_row(
                self.prompt_table,
                row_index,
                [
                    str(row.get("row_index", row_index)),
                    str(row.get("block_index") or ""),
                    str(row.get("prompt_index") or ""),
                    _display_class(_s(row.get("label", ""))),
                    str(row.get("start_sample") or ""),
                    str(row.get("end_sample") or ""),
                    "" if row.get("duration_sec") is None else f"{float(row['duration_sec']):.3f}",
                    "" if row.get("execution_success") is None else ("Yes" if bool(row.get("execution_success")) else "No"),
                ],
                center_columns={0, 1, 2, 4, 5, 6, 7},
            )
        if self._prompt_rows and self.prompt_table.currentRow() < 0:
            self.prompt_table.selectRow(0)

    def _fill_event_table(self) -> None:
        self.event_table.setRowCount(len(self._event_rows))
        for row_index, row in enumerate(self._event_rows):
            self._set_row(
                self.event_table,
                row_index,
                [
                    str(row["event_index"]),
                    str(row["event_name"]),
                    str(row.get("marker_code") or ""),
                    str(row.get("sample_index") or ""),
                    str(row.get("trial_id") or ""),
                    str(row.get("mi_run_index") or ""),
                    str(row.get("block_index") or ""),
                    str(row.get("prompt_index") or ""),
                    _display_class(_s(row.get("class_name", ""))) if _s(row.get("class_name", "")) else "",
                    "" if row.get("execution_success") is None else ("Yes" if bool(row.get("execution_success")) else "No"),
                    str(row.get("iso_time") or ""),
                ],
                center_columns={0, 2, 3, 4, 5, 6, 7, 9},
            )

    def _fill_channel_table(self) -> None:
        self.channel_table.setRowCount(len(self._channel_rows))
        for row_index, row in enumerate(self._channel_rows):
            self._set_row(
                self.channel_table,
                row_index,
                [
                    str(row["channel"]),
                    f"{float(row['mean_uV']):.3f}",
                    f"{float(row['std_uV']):.3f}",
                    f"{float(row['rms_uV']):.3f}",
                    f"{float(row['ptp_uV']):.3f}",
                    f"{float(row['abs_mean_uV']):.3f}",
                ],
                center_columns={1, 2, 3, 4, 5},
            )

    def _fill_file_table(self) -> None:
        self.file_table.setRowCount(len(self._file_rows))
        for row_index, row in enumerate(self._file_rows):
            self._set_row(
                self.file_table,
                row_index,
                [str(row["file_key"]), str(row["status"]), str(row["relative_path"]), str(row["absolute_path"])],
                center_columns={1},
            )

    @staticmethod
    def _set_row(table: QTableWidget, row_index: int, values: list[str], center_columns: set[int] | None = None) -> None:
        centered = center_columns or set()
        for column_index, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column_index in centered:
                item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_index, column_index, item)

    def _refresh_controls(self) -> None:
        bundle = self.bundle
        if bundle is None:
            return
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItem("All Classes", -1)
        for index, class_name in enumerate(bundle.class_names):
            self.class_combo.addItem(f"{index}: {_display_class(class_name)}", index)
        self.class_combo.blockSignals(False)

        self.ch_combo.blockSignals(True)
        self.ch_combo.clear()
        self.ch_combo.addItem("All Channels (stacked)", -1)
        for index, channel_name in enumerate(bundle.channel_names):
            self.ch_combo.addItem(channel_name, index)
        self.ch_combo.blockSignals(False)
        self._sync_mode()

    def _sync_mode(self) -> None:
        mode = str(self.mode_combo.currentData())
        self.class_combo.setEnabled(mode in {"class_mean", "class_psd"})

    def _set_mode(self, mode: str) -> None:
        for index in range(self.mode_combo.count()):
            if str(self.mode_combo.itemData(index)) == mode:
                self.mode_combo.setCurrentIndex(index)
                break

    def _combo_int(self, combo: QComboBox, default: int) -> int:
        value = combo.currentData()
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _on_ctrl_changed(self) -> None:
        self._sync_mode()
        if self.bundle is not None:
            self.refresh_plot()

    def _channel_indices(self) -> list[int]:
        bundle = self.bundle
        if bundle is None:
            return []
        index = self._combo_int(self.ch_combo, -1)
        return list(range(len(bundle.channel_names))) if index < 0 else [index]

    def _selected_trial(self) -> dict[str, Any] | None:
        if not self._trial_rows:
            return None
        row = self.trial_table.currentRow()
        if row < 0 or row >= len(self._trial_rows):
            row = 0
        return self._trial_rows[row]

    def _selected_segment(self) -> dict[str, Any] | None:
        if not self._segment_rows:
            return None
        row = self.segment_table.currentRow()
        if row < 0 or row >= len(self._segment_rows):
            row = 0
        return self._segment_rows[row]

    def _selected_prompt(self) -> dict[str, Any] | None:
        if not self._prompt_rows:
            return None
        row = self.prompt_table.currentRow()
        if row < 0 or row >= len(self._prompt_rows):
            row = 0
        return self._prompt_rows[row]

    def _selected_mi_epoch_index(self) -> int | None:
        bundle = self.bundle
        if bundle is None or bundle.mi_epochs is None or bundle.mi_epochs.n_trials == 0:
            return None
        trial = self._selected_trial()
        if trial is not None:
            trial_id = int(trial["trial_id"])
            matches = np.flatnonzero(bundle.mi_epochs.trial_ids == trial_id)
            if matches.size > 0:
                return int(matches[0])
        if self.ck_ok.isChecked():
            valid = np.flatnonzero(bundle.mi_epochs.accepted)
            if valid.size > 0:
                return int(valid[0])
        return 0

    def _trial_interval(self, trial: dict[str, Any]) -> tuple[int, int] | None:
        trial_id = int(trial["trial_id"])
        for row in self._segment_rows:
            if row.get("segment_type") == "trial" and int(row.get("trial_id") or 0) == trial_id:
                start_sample = row.get("start_sample")
                end_sample = row.get("end_sample")
                if start_sample is not None and end_sample is not None:
                    return int(start_sample), int(end_sample)
        cue = trial.get("cue_onset_sample")
        end = trial.get("trial_end_sample")
        if cue is not None and end is not None and self.bundle is not None:
            return max(0, int(cue) - int(round(self.bundle.sampling_rate))), int(end)
        return None

    def _events_in_interval(self, start_sample: int, end_sample: int) -> list[dict[str, Any]]:
        return [row for row in self._event_rows if row.get("sample_index") is not None and start_sample <= int(row["sample_index"]) <= end_sample]

    def _segments_in_interval(self, start_sample: int, end_sample: int) -> list[dict[str, Any]]:
        return [
            row
            for row in self._segment_rows
            if row.get("start_sample") is not None
            and row.get("end_sample") is not None
            and int(row["end_sample"]) >= start_sample
            and int(row["start_sample"]) <= end_sample
        ]

    def _plot_stacked_signal(self, ax, signal_uV: np.ndarray, start_sample: int, channel_indices: list[int], title: str) -> None:
        bundle = self.bundle
        if bundle is None or signal_uV.size == 0:
            ax.text(0.5, 0.5, "No signal available", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return
        sampling_rate = max(bundle.sampling_rate, 1e-6)
        times = np.arange(signal_uV.shape[1], dtype=np.float32) / sampling_rate
        signal = signal_uV.copy()
        if self.ck_demean.isChecked():
            signal -= np.mean(signal, axis=1, keepdims=True)
        if signal.shape[0] == 1:
            ax.plot(times, signal[0], lw=1.2, color="#2563eb")
            ax.set_ylabel("Amplitude (uV)")
        else:
            step = max(8.0, float(np.percentile(np.ptp(signal, axis=1), 85)) * 1.25)
            offsets = np.arange(signal.shape[0], dtype=np.float32) * step
            for index in range(signal.shape[0]):
                ax.plot(times, signal[index] + offsets[index], lw=1.0)
            ax.set_yticks(offsets)
            ax.set_yticklabels([bundle.channel_names[index] for index in channel_indices])
            ax.set_ylabel("Channels")
        ax.set_xlabel("Time (s)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        if self.ck_annotations.isChecked():
            self._overlay_segments(ax, start_sample, start_sample + signal_uV.shape[1])
            self._overlay_events(ax, start_sample, start_sample + signal_uV.shape[1])

    def _overlay_segments(self, ax, start_sample: int, end_sample: int) -> None:
        bundle = self.bundle
        if bundle is None:
            return
        sampling_rate = max(bundle.sampling_rate, 1e-6)
        for row in self._segments_in_interval(start_sample, end_sample):
            color = SEGMENT_COLORS.get(_s(row.get("segment_type", "")))
            if color is None:
                continue
            seg_start = int(row.get("start_sample") or start_sample)
            seg_end = int(row.get("end_sample") or seg_start + 1)
            left = max(seg_start, start_sample)
            right = min(seg_end, end_sample)
            if right <= left:
                continue
            ax.axvspan((left - start_sample) / sampling_rate, (right - start_sample) / sampling_rate, color=color, alpha=0.16)

    def _overlay_events(self, ax, start_sample: int, end_sample: int) -> None:
        bundle = self.bundle
        if bundle is None:
            return
        sampling_rate = max(bundle.sampling_rate, 1e-6)
        label_budget = 14
        labeled = 0
        for row in self._events_in_interval(start_sample, end_sample):
            sample = int(row.get("sample_index") or start_sample)
            xpos = (sample - start_sample) / sampling_rate
            ax.axvline(xpos, color="#94a3b8", alpha=0.35, linestyle="--", lw=0.8)
            if labeled < label_budget:
                ax.text(
                    xpos,
                    0.98,
                    str(row.get("event_name", "")),
                    transform=ax.get_xaxis_transform(),
                    rotation=90,
                    ha="right",
                    va="top",
                    fontsize=7,
                    color="#475569",
                )
                labeled += 1

    def _plot_class_distribution(self, ax) -> None:
        if not self._class_rows:
            ax.text(0.5, 0.5, "No class stats", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return
        indices = np.arange(len(self._class_rows), dtype=np.float32)
        totals = np.asarray([int(row["total"]) for row in self._class_rows], dtype=np.int64)
        accepted = np.asarray([int(row["accepted"]) for row in self._class_rows], dtype=np.int64)
        ax.bar(indices - 0.18, totals, width=0.36, color="#60a5fa", label="Total")
        ax.bar(indices + 0.18, accepted, width=0.36, color="#22c55e", label="Accepted")
        ax.set_xticks(indices)
        ax.set_xticklabels([str(row["class_display"]) for row in self._class_rows])
        ax.set_ylabel("Count")
        ax.set_title("Class Distribution")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="upper right")

    def refresh_plot(self) -> None:
        self.fig.clear()
        bundle = self.bundle
        if bundle is None:
            self.canvas.draw_idle()
            return

        grid = self.fig.add_gridspec(2, 1, height_ratios=[3.0, 1.1])
        ax_main = self.fig.add_subplot(grid[0, 0])
        ax_bottom = self.fig.add_subplot(grid[1, 0])
        mode = str(self.mode_combo.currentData())
        channel_indices = self._channel_indices()

        if mode == "selected_trial":
            trial = self._selected_trial()
            interval = None if trial is None else self._trial_interval(trial)
            if trial is None or interval is None:
                ax_main.text(0.5, 0.5, "No trial selected", ha="center", va="center", transform=ax_main.transAxes)
                ax_main.set_axis_off()
            else:
                context = int(round(self.context_spin.value() * bundle.sampling_rate))
                start_sample = max(0, int(interval[0]) - context)
                end_sample = min(bundle.sample_count, int(interval[1]) + context)
                signal = _extract_bundle_signal(bundle, start_sample, end_sample, channel_indices)
                title = f"Trial {trial['trial_id']} | Class={_display_class(_s(trial.get('class_name', '')))} | Accepted={'Yes' if bool(trial.get('accepted')) else 'No'}"
                self._plot_stacked_signal(ax_main, signal, start_sample, channel_indices, title)
        elif mode == "selected_segment":
            row = self._selected_segment()
            if row is None or row.get("start_sample") is None or row.get("end_sample") is None:
                ax_main.text(0.5, 0.5, "No segment selected", ha="center", va="center", transform=ax_main.transAxes)
                ax_main.set_axis_off()
            else:
                context = int(round(self.context_spin.value() * bundle.sampling_rate))
                start_sample = max(0, int(row["start_sample"]) - context)
                end_sample = min(bundle.sample_count, int(row["end_sample"]) + context)
                signal = _extract_bundle_signal(bundle, start_sample, end_sample, channel_indices)
                title = f"Segment {row['segment_id']} | Type={row['segment_type']} | Label={row['label'] or '-'} | Trial={row.get('trial_id') or '-'}"
                self._plot_stacked_signal(ax_main, signal, start_sample, channel_indices, title)
        elif mode == "selected_prompt":
            row = self._selected_prompt()
            if row is None or row.get("start_sample") is None or row.get("end_sample") is None:
                ax_main.text(0.5, 0.5, "No prompt selected", ha="center", va="center", transform=ax_main.transAxes)
                ax_main.set_axis_off()
            else:
                context = int(round(self.context_spin.value() * bundle.sampling_rate))
                start_sample = max(0, int(row["start_sample"]) - context)
                end_sample = min(bundle.sample_count, int(row["end_sample"]) + context)
                signal = _extract_bundle_signal(bundle, start_sample, end_sample, channel_indices)
                success_text = "" if row.get("execution_success") is None else ("Yes" if bool(row.get("execution_success")) else "No")
                title = f"Prompt block={row.get('block_index') or '-'} idx={row.get('prompt_index') or '-'} | Label={_display_class(_s(row.get('label', '')))} | Success={success_text}"
                self._plot_stacked_signal(ax_main, signal, start_sample, channel_indices, title)
        elif mode == "mi_epoch":
            if bundle.mi_epochs is None or bundle.mi_epochs.n_trials == 0:
                ax_main.text(0.5, 0.5, "No MI epoch file available", ha="center", va="center", transform=ax_main.transAxes)
                ax_main.set_axis_off()
            else:
                epoch_index = self._selected_mi_epoch_index()
                if epoch_index is None:
                    ax_main.text(0.5, 0.5, "No accepted MI epoch available", ha="center", va="center", transform=ax_main.transAxes)
                    ax_main.set_axis_off()
                else:
                    epoch = bundle.mi_epochs
                    signal = epoch.X_uV[epoch_index, channel_indices, :].copy()
                    if self.ck_demean.isChecked():
                        signal -= np.mean(signal, axis=1, keepdims=True)
                    times = np.arange(epoch.n_samples, dtype=np.float32) / max(epoch.sampling_rate, 1e-6)
                    class_label = _display_class(epoch.class_names[int(epoch.y[epoch_index])]) if epoch.class_names else str(int(epoch.y[epoch_index]))
                    trial_id = int(epoch.trial_ids[epoch_index]) if epoch.trial_ids.size else epoch_index + 1
                    if signal.shape[0] == 1:
                        ax_main.plot(times, signal[0], lw=1.2, color="#2563eb")
                        ax_main.set_ylabel("Amplitude (uV)")
                    else:
                        step = max(8.0, float(np.percentile(np.ptp(signal, axis=1), 85)) * 1.25)
                        offsets = np.arange(signal.shape[0], dtype=np.float32) * step
                        for index in range(signal.shape[0]):
                            ax_main.plot(times, signal[index] + offsets[index], lw=1.0)
                        ax_main.set_yticks(offsets)
                        ax_main.set_yticklabels([epoch.channel_names[index] for index in channel_indices])
                        ax_main.set_ylabel("Channels")
                    ax_main.set_xlabel("Time (s)")
                    ax_main.set_title(f"MI Epoch | Trial {trial_id} | Class={class_label}")
                    ax_main.grid(alpha=0.25)
        else:
            epoch = bundle.mi_epochs
            if epoch is None or epoch.n_trials == 0:
                ax_main.text(0.5, 0.5, "No MI epoch file available", ha="center", va="center", transform=ax_main.transAxes)
                ax_main.set_axis_off()
            else:
                mask = np.ones(epoch.n_trials, dtype=bool)
                class_index = self._combo_int(self.class_combo, -1)
                if class_index >= 0:
                    mask &= epoch.y == class_index
                if self.ck_ok.isChecked():
                    mask &= epoch.accepted
                signal = epoch.X_uV[mask][:, channel_indices, :]
                if signal.size == 0:
                    ax_main.text(0.5, 0.5, "No MI epochs match current filters", ha="center", va="center", transform=ax_main.transAxes)
                    ax_main.set_axis_off()
                else:
                    if self.ck_demean.isChecked():
                        signal -= np.mean(signal, axis=2, keepdims=True)
                    class_label = "All Classes" if class_index < 0 or class_index >= len(epoch.class_names) else _display_class(epoch.class_names[class_index])
                    if mode == "class_mean":
                        times = np.arange(epoch.n_samples, dtype=np.float32) / max(epoch.sampling_rate, 1e-6)
                        if signal.shape[1] == 1:
                            values = signal[:, 0, :]
                            mean = np.mean(values, axis=0)
                            std = np.std(values, axis=0)
                            ax_main.plot(times, mean, lw=1.6, color="#ea580c", label="Mean")
                            ax_main.fill_between(times, mean - std, mean + std, alpha=0.25, color="#fb923c", label="Std")
                            ax_main.legend(loc="upper right")
                            ax_main.set_ylabel("Amplitude (uV)")
                        else:
                            mean = np.mean(signal, axis=0)
                            step = max(8.0, float(np.percentile(np.ptp(mean, axis=1), 85)) * 1.25)
                            offsets = np.arange(mean.shape[0], dtype=np.float32) * step
                            for index in range(mean.shape[0]):
                                ax_main.plot(times, mean[index] + offsets[index], lw=1.0)
                            ax_main.set_yticks(offsets)
                            ax_main.set_yticklabels([epoch.channel_names[index] for index in channel_indices])
                            ax_main.set_ylabel("Channels")
                        ax_main.set_xlabel("Time (s)")
                        ax_main.set_title(f"Class Mean | Class={class_label} | Trials={signal.shape[0]}")
                        ax_main.grid(alpha=0.25)
                    else:
                        sample_count = epoch.n_samples
                        sampling_rate = max(epoch.sampling_rate, 1e-6)
                        frequencies = np.fft.rfftfreq(sample_count, d=1.0 / sampling_rate)
                        spectrum = np.fft.rfft(signal, axis=-1)
                        psd = (np.abs(spectrum) ** 2) / (sampling_rate * sample_count)
                        if sample_count > 1:
                            psd[..., 1:-1] *= 2.0
                        mask_freq = frequencies <= min(80.0, sampling_rate / 2.0)
                        if signal.shape[1] == 1:
                            values = np.mean(psd[:, 0, :], axis=0)[mask_freq]
                            ax_main.semilogy(frequencies[mask_freq], np.maximum(values, 1e-12), lw=1.4, color="#0f766e")
                        else:
                            for index, channel_index in enumerate(channel_indices):
                                values = np.mean(psd[:, index, :], axis=0)[mask_freq]
                                ax_main.semilogy(frequencies[mask_freq], np.maximum(values, 1e-12), lw=1.0, label=epoch.channel_names[channel_index])
                            if len(channel_indices) <= 10:
                                ax_main.legend(loc="upper right", fontsize=8)
                        ax_main.set_xlabel("Frequency (Hz)")
                        ax_main.set_ylabel("PSD (uV^2/Hz)")
                        ax_main.set_title(f"Class PSD | Class={class_label} | Trials={signal.shape[0]}")
                        ax_main.grid(alpha=0.25)

        self._plot_class_distribution(ax_bottom)
        self.canvas.draw_idle()

    def _unique(self, path: Path) -> Path:
        if not path.exists():
            return path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")

    def export_json(self) -> None:
        if self.bundle is None or self.stats_payload is None:
            QMessageBox.information(self, "No Data", "Load a run bundle first.")
            return
        output = self._unique(self.bundle.meta_path.with_name(f"{self.bundle.run_stem}_viewer_summary.json"))
        with output.open("w", encoding="utf-8") as handle:
            json.dump(self.stats_payload, handle, indent=2, ensure_ascii=False)
        self._set_status(f"Exported JSON: {output}")

    def export_csv(self) -> None:
        if self.bundle is None:
            QMessageBox.information(self, "No Data", "Load a run bundle first.")
            return
        bundle = self.bundle
        outputs = {
            "class": self._unique(bundle.meta_path.with_name(f"{bundle.run_stem}_viewer_class_stats.csv")),
            "segment_summary": self._unique(bundle.meta_path.with_name(f"{bundle.run_stem}_viewer_segment_summary.csv")),
            "trials": self._unique(bundle.meta_path.with_name(f"{bundle.run_stem}_viewer_trials.csv")),
            "segments": self._unique(bundle.meta_path.with_name(f"{bundle.run_stem}_viewer_segments.csv")),
            "prompts": self._unique(bundle.meta_path.with_name(f"{bundle.run_stem}_viewer_prompts.csv")),
            "channels": self._unique(bundle.meta_path.with_name(f"{bundle.run_stem}_viewer_channels.csv")),
            "files": self._unique(bundle.meta_path.with_name(f"{bundle.run_stem}_viewer_files.csv")),
        }
        _write_rows_csv(outputs["class"], self._class_rows, ["label", "class_key", "class_display", "total", "accepted", "rejected"])
        _write_rows_csv(outputs["segment_summary"], self._segment_summary_rows, ["segment_type", "count", "total_duration_sec", "accepted_count", "rejected_count"])
        _write_rows_csv(outputs["trials"], self._trial_rows, ["trial_id", "save_index", "mi_run_index", "run_trial_index", "class_name", "display_name", "accepted", "cue_onset_sample", "imagery_onset_sample", "imagery_offset_sample", "trial_end_sample", "note"])
        _write_rows_csv(outputs["segments"], self._segment_rows, ["segment_id", "save_index", "segment_type", "label", "start_sample", "end_sample", "duration_sec", "trial_id", "mi_run_index", "run_trial_index", "block_index", "prompt_index", "accepted", "execution_success", "source_start_event", "source_end_event"])
        _write_rows_csv(outputs["prompts"], self._prompt_rows, ["row_index", "block_index", "prompt_index", "label", "start_sample", "end_sample", "duration_sec", "execution_success"])
        _write_rows_csv(outputs["channels"], self._channel_rows, ["channel", "mean_uV", "std_uV", "rms_uV", "ptp_uV", "abs_mean_uV"])
        _write_rows_csv(outputs["files"], self._file_rows, ["file_key", "status", "relative_path", "absolute_path"])
        self._set_status("Exported CSV bundle summaries")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View one collected MI run bundle")
    parser.add_argument("--source", type=str, default="", help="Session directory, session_meta.json, or any run artifact")
    parser.add_argument("--npz", type=str, default="", help="Backward-compatible alias for --source")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    initial_raw = args.source or args.npz
    initial = Path(initial_raw).resolve() if initial_raw else None
    app = QApplication(sys.argv)
    app.setApplicationName("MI Collection Viewer")
    app.setFont(QFont("Microsoft YaHei", 10))
    window = RunBundleViewer(initial=initial)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
