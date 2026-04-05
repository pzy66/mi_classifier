"""Train a realtime-compatible MI classifier from custom collected epochs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_ROOT = PROJECT_ROOT / "code" / "shared"
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

from src.models import (
    TORCH_AVAILABLE,
    apply_probability_calibration,
    build_optimized_candidates,
    fit_probability_calibration,
    predict_probability_matrix,
)
from src.preprocessing import (
    DEFAULT_PREPROCESS_APPLY_CAR,
    DEFAULT_PREPROCESS_BANDPASS,
    DEFAULT_PREPROCESS_NOTCH,
    DEFAULT_PREPROCESS_STANDARDIZE,
    preprocess,
)
from src.realtime_mi import build_realtime_artifact_bank


MI_CLASSES = [
    {"key": "left_hand", "display_name": "\u5de6\u624b\u60f3\u8c61"},
    {"key": "right_hand", "display_name": "\u53f3\u624b\u60f3\u8c61"},
    {"key": "feet", "display_name": "\u53cc\u811a\u60f3\u8c61"},
    {"key": "tongue", "display_name": "\u820c\u5934\u60f3\u8c61"},
]
CLASS_NAME_TO_DISPLAY = {item["key"]: item["display_name"] for item in MI_CLASSES}
GATE_CLASS_NAMES = ["rest", "control"]
GATE_DISPLAY_CLASS_NAMES = ["NO CONTROL", "CONTROL"]
ARTIFACT_REJECTOR_DISPLAY_CLASS_NAMES = ["CLEAN", "ARTIFACT"]
CONTINUOUS_NO_CONTROL_LABEL = "no_control"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "custom_mi"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "code" / "realtime" / "models" / "custom_mi_realtime.joblib"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "code" / "training" / "reports" / "custom_mi_training_summary.json"
DEFAULT_WINDOW_SECS = [2.0, 2.5, 3.0]
DEFAULT_WINDOW_OFFSET_SEC = 0.5
DEFAULT_WINDOW_OFFSET_SECS = [0.5, 0.75]
DEFAULT_FUSION_METHOD = "log_weighted_mean"
DEFAULT_FUSION_WEIGHT_GRID_STEP = 0.05
DEFAULT_EPOCH_LENGTH_TOLERANCE_SAMPLES = 8
DEFAULT_REST_WINDOW_STEP_SEC = 0.5
DEFAULT_REST_FALSE_ACTIVATION_TARGET = 0.10
DEFAULT_MIN_GATE_CONTROL_DETECTION_RATE = 0.05
DEFAULT_MIN_ARTIFACT_DETECTION_RATE = 0.05
DEFAULT_MIN_BINARY_STAGE_CLASS_ACCURACY = 0.05
DEFAULT_RECOMMENDED_TOTAL_CLASS_TRIALS = 30
DEFAULT_RECOMMENDED_RUN_CLASS_TRIALS = 8
DEFAULT_MAIN_CANDIDATE_NAMES = [
    "central_fbcsp_lda",
    "central_prior_dual_branch_fblight_tcn",
    "riemann+lda",
]
DEFAULT_GATE_CANDIDATE_NAMES = ["central_gate_fblight", "central_prior_gate_fblight"]
DEFAULT_ARTIFACT_CANDIDATE_NAMES = ["full8_fblight"]
DEFAULT_TORCH_EPOCHS = 80
DEFAULT_TORCH_BATCH_SIZE = 32
DEFAULT_TORCH_LEARNING_RATE = 1e-3
DEFAULT_TORCH_WEIGHT_DECAY = 1e-4
DEFAULT_TORCH_PATIENCE = 12
DEFAULT_TORCH_VALIDATION_SPLIT = 0.15
DEFAULT_FBCSP_BANDS = [
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 28.0),
    (28.0, 32.0),
    (32.0, 36.0),
    (36.0, 40.0),
]
DEFAULT_DEEP_STAGE_PRETRAIN_WINDOW_SECS = [2.5, 2.0]
DEFAULT_DEEP_STAGE_FINETUNE_WINDOW_SECS = [2.0, 1.5]
DEFAULT_CENTRAL_PRIOR_ALPHA = 0.75
DEFAULT_CENTRAL_AUX_LOSS_WEIGHT = 0.3
STRICT_MAIN_SPLIT_STRATEGIES = {"group_shuffle", "session_holdout"}
STRICT_AUX_SPLIT_STRATEGIES = {"group_shuffle", "session_holdout", "aligned_to_main_split"}
DEFAULT_SELECTION_OBJECTIVE_SPEC = {
    "name": "offline_plus_continuous_weighted",
    "components": {
        "selection_val_kappa": {"weight": 0.35, "direction": "maximize"},
        "selection_val_macro_f1": {"weight": 0.25, "direction": "maximize"},
        "continuous_mi_prompt_accuracy": {"weight": 0.25, "direction": "maximize"},
        "continuous_no_control_specificity": {"weight": 0.15, "direction": "maximize"},
    },
    "continuous_no_control_specificity_formula": "1 - no_control_false_activation_rate",
    "fallback_when_no_continuous": {
        "selection_val_kappa": {"weight": 0.60, "direction": "maximize"},
        "selection_val_macro_f1": {"weight": 0.40, "direction": "maximize"},
    },
}
RUN_STEM_PATTERN = re.compile(
    r"sub-(?P<subject>.+?)_ses-(?P<session>.+?)_run-(?P<run>\d{3})_tpc-(?P<tpc>\d+)_n-(?P<n>\d+)_ok-(?P<ok>\d+)$"
)
RUN_FILE_SUFFIXES = {
    "_mi_epochs.npz": "mi",
    "_gate_epochs.npz": "gate",
    "_artifact_epochs.npz": "artifact",
    "_continuous.npz": "continuous",
}
ARTIFACT_REJECTOR_CLASS_NAMES = ["clean", "artifact"]


def _decode_npz_text(value: object) -> str:
    """Decode potential bytes values in npz metadata."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


def _parse_run_tokens_from_filename(path: Path) -> dict[str, object]:
    """Parse run metadata from new collection filename schema."""
    name = path.name
    stem_token = ""
    for suffix in RUN_FILE_SUFFIXES:
        if name.endswith(suffix):
            stem_token = name[: -len(suffix)]
            break
    if not stem_token:
        return {}

    matched = RUN_STEM_PATTERN.search(stem_token)
    if not matched:
        return {}
    token = matched.groupdict()
    return {
        "subject_id": token["subject"],
        "session_id": token["session"],
        "save_index": int(token["run"]),
        "trials_per_class": int(token["tpc"]),
        "total_trials_tag": int(token["n"]),
        "accepted_trials_tag": int(token["ok"]),
    }


def normalize_subject_filter(subject: str | None) -> str | None:
    """Normalize subject filter to the on-disk folder token."""
    if not subject:
        return None
    token = str(subject).strip()
    if not token:
        return None
    return token if token.startswith("sub-") else f"sub-{token}"


def parse_float_list(raw_value: str | list[float] | tuple[float, ...] | None) -> list[float]:
    """Parse a comma-separated float list."""
    if raw_value is None:
        return []
    if isinstance(raw_value, (list, tuple)):
        return [float(item) for item in raw_value]
    tokens = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    return [float(item) for item in tokens]


def parse_string_list(raw_value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Parse a comma-separated string list."""
    if raw_value is None:
        return []
    if isinstance(raw_value, (list, tuple)):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]


def window_token(window_sec: float) -> str:
    """Convert a window length into a filesystem-friendly token."""
    return f"w{float(window_sec):.2f}s".replace(".", "p")


def default_window_weights(window_secs: list[float]) -> list[float]:
    """Prefer shorter windows by default when building a multi-window bank."""
    rounded = [round(float(item), 3) for item in window_secs]
    if rounded == [2.0, 2.5, 3.0]:
        return [0.45, 0.35, 0.20]

    descending = np.arange(len(window_secs), 0, -1, dtype=np.float64)
    descending = descending / np.sum(descending)
    return descending.astype(np.float64).tolist()


def default_window_offsets() -> list[float]:
    """Return the default onset-offset candidates used during selection."""
    return [float(item) for item in DEFAULT_WINDOW_OFFSET_SECS]


def default_candidate_names() -> list[str]:
    """Return the default main-model anchor set used during selection."""
    if TORCH_AVAILABLE:
        return list(DEFAULT_MAIN_CANDIDATE_NAMES)
    return [name for name in DEFAULT_MAIN_CANDIDATE_NAMES if name != "central_prior_dual_branch_fblight_tcn"]


def default_gate_candidate_names(candidate_names: list[str] | None = None) -> list[str]:
    """Return gate-specific candidate names (binary control-vs-rest)."""
    requested = [str(item).strip() for item in (candidate_names or []) if str(item).strip()]
    if requested:
        return requested
    if TORCH_AVAILABLE:
        return list(DEFAULT_GATE_CANDIDATE_NAMES)
    return ["central_fbcsp_lda", "hybrid+lda"]


def default_artifact_candidate_names(candidate_names: list[str] | None = None) -> list[str]:
    """Return bad-window rejector candidates (clean-vs-artifact)."""
    requested = [str(item).strip() for item in (candidate_names or []) if str(item).strip()]
    if requested:
        return requested
    if TORCH_AVAILABLE:
        return list(DEFAULT_ARTIFACT_CANDIDATE_NAMES)
    return ["hybrid+lda"]


def find_epoch_files(dataset_root: Path, subject_filter: str | None) -> list[Path]:
    """Find all saved MI epoch npz files."""
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    candidates: list[Path] = []
    seen: set[Path] = set()
    for pattern in ("*_mi_epochs.npz",):
        for path in dataset_root.rglob(pattern):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(resolved)

    epoch_files = sorted(candidates)
    if subject_filter is not None:
        bare_subject = str(subject_filter).replace("sub-", "")
        epoch_files = [
            path
            for path in epoch_files
            if (
                subject_filter in path.parts
                or str(path).find(subject_filter) >= 0
                or str(path).find(f"sub-{bare_subject}") >= 0
            )
        ]
    if not epoch_files:
        raise FileNotFoundError(f"No trainable *_mi_epochs.npz files found under: {dataset_root}")
    return epoch_files


def load_custom_epochs(dataset_root: Path, subject_filter: str | None = None) -> dict[str, object]:
    """Load and merge custom MI epochs from all matching sessions."""
    epoch_files = find_epoch_files(dataset_root, subject_filter)

    file_payloads = []
    source_records = []
    channel_names = None
    class_names = None
    sampling_rate = None

    for path in epoch_files:
        file_token = _parse_run_tokens_from_filename(path)
        with np.load(path, allow_pickle=False) as data:
            X = np.asarray(data["X_mi"], dtype=np.float32)
            y = np.asarray(data["y_mi"], dtype=np.int64)
            current_channel_names = [_decode_npz_text(item) for item in data["channel_names"].tolist()]
            current_class_names = [_decode_npz_text(item) for item in data["class_names"].tolist()]
            current_sampling_rate = float(np.asarray(data["sampling_rate"]).reshape(-1)[0])
            signal_unit = "volt"
            if "signal_unit" in data.files:
                signal_unit = _decode_npz_text(np.asarray(data["signal_unit"]).reshape(-1)[0]).lower() or "volt"

            subject_id = file_token.get("subject_id", "")
            if "subject_id" in data.files:
                subject_id = _decode_npz_text(np.asarray(data["subject_id"]).reshape(-1)[0]) or subject_id

            session_id = file_token.get("session_id", "")
            if "session_id" in data.files:
                session_id = _decode_npz_text(np.asarray(data["session_id"]).reshape(-1)[0]) or session_id

            save_index = file_token.get("save_index")
            if "save_index" in data.files:
                save_index = int(np.asarray(data["save_index"]).reshape(-1)[0])

            run_stem = ""
            if "run_stem" in data.files:
                run_stem = _decode_npz_text(np.asarray(data["run_stem"]).reshape(-1)[0])
            if not run_stem:
                run_stem = path.stem.replace("_mi_epochs", "")

            trials_per_class = file_token.get("trials_per_class")
            if "trials_per_class" in data.files:
                trials_per_class = int(np.asarray(data["trials_per_class"]).reshape(-1)[0])

            total_trials_tag = file_token.get("total_trials_tag")
            if "total_trials" in data.files:
                total_trials_tag = int(np.asarray(data["total_trials"]).reshape(-1)[0])

            accepted_trials_tag = file_token.get("accepted_trials_tag")
            if "accepted_trials" in data.files:
                accepted_trials_tag = int(np.asarray(data["accepted_trials"]).reshape(-1)[0])

            trial_ids = (
                np.asarray(data["mi_trial_ids"], dtype=np.int64)
                if "mi_trial_ids" in data.files
                else np.arange(X.shape[0], dtype=np.int64)
            )

        if signal_unit in {"uv", "microvolt", "microvolts", "\u00b5v", "\u03bcv"}:
            X = (X * 1e-6).astype(np.float32)

        raw_trial_count = int(X.shape[0])
        accepted_trial_count = int(raw_trial_count)
        rejected_trial_count = 0
        file_record = {
            "file": str(path.relative_to(dataset_root)),
            "subject_id": subject_id,
            "session_id": session_id,
            "save_index": save_index,
            "run_stem": run_stem,
            "trials_per_class": trials_per_class,
            "total_trials_in_file": raw_trial_count,
            "accepted_trials_in_file": accepted_trial_count,
            "rejected_trials_in_file": rejected_trial_count,
            "total_trials_tag": total_trials_tag,
            "accepted_trials_tag": accepted_trials_tag,
            "sampling_rate_hz": current_sampling_rate,
            "baseline_rest_segments_in_file": 0,
            "iti_rest_segments_in_file": 0,
            "dropped_reason": "",
        }

        if channel_names is None:
            channel_names = current_channel_names
        elif current_channel_names != channel_names:
            raise ValueError(f"Channel names are inconsistent: {path}")

        if class_names is None:
            class_names = current_class_names
        elif current_class_names != class_names:
            raise ValueError(f"Class names are inconsistent: {path}")

        if sampling_rate is None:
            sampling_rate = current_sampling_rate
        elif abs(current_sampling_rate - sampling_rate) > 1e-6:
            raise ValueError(f"Sampling rate is inconsistent: {path}")

        source_records.append(file_record)
        group_token = run_stem
        if not file_token and group_token == "epochs":
            group_token = str(path.relative_to(dataset_root))
        if not group_token:
            group_token = str(path.relative_to(dataset_root))
        file_payloads.append(
            {
                "relative_path": str(path.relative_to(dataset_root)),
                "sample_count": int(X.shape[-1]),
                "X": np.asarray(X, dtype=np.float32),
                "y": np.asarray(y, dtype=np.int64),
                "groups": np.asarray([group_token] * X.shape[0], dtype=object),
                "trial_keys": np.asarray(
                    [f"{group_token}::trial-{int(trial_id):04d}" for trial_id in trial_ids],
                    dtype=object,
                ),
                "rest_payloads": [
                    {
                        "phase": "baseline",
                        "X": np.empty((0, X.shape[1], X.shape[2]), dtype=np.float32),
                        "trial_ids": np.empty((0,), dtype=np.int64),
                    },
                    {
                        "phase": "iti",
                        "X": np.empty((0, X.shape[1], X.shape[2]), dtype=np.float32),
                        "trial_ids": np.empty((0,), dtype=np.int64),
                    },
                ],
                "file_record": file_record,
            }
        )

    all_X = []
    all_y = []
    all_groups = []
    all_trial_keys = []
    all_rest_segments = []
    all_rest_groups = []
    all_rest_trial_keys = []
    all_rest_parent_trial_keys = []
    all_rest_source_phases = []
    session_paths = []
    min_samples = None

    if file_payloads:
        sample_counts = [int(item["sample_count"]) for item in file_payloads]
        length_histogram = Counter(sample_counts)
        reference_samples = max(length_histogram.items(), key=lambda item: (int(item[1]), int(item[0])))[0]
        retained_payloads = []
        dropped_short_runs = []

        for payload in file_payloads:
            sample_count = int(payload["sample_count"])
            if sample_count < (int(reference_samples) - int(DEFAULT_EPOCH_LENGTH_TOLERANCE_SAMPLES)):
                payload["file_record"]["dropped_reason"] = (
                    f"too_short_for_common_epoch_length:{sample_count}<{reference_samples}"
                )
                dropped_short_runs.append(str(payload["relative_path"]))
                continue
            retained_payloads.append(payload)

        if retained_payloads:
            min_samples = min(int(item["sample_count"]) for item in retained_payloads)
            for payload in retained_payloads:
                session_paths.append(str(payload["relative_path"]))
                all_X.append(np.asarray(payload["X"][:, :, :min_samples], dtype=np.float32))
                all_y.append(np.asarray(payload["y"], dtype=np.int64))
                all_groups.append(np.asarray(payload["groups"], dtype=object))
                all_trial_keys.append(np.asarray(payload["trial_keys"], dtype=object))
                payload_group = str(payload["groups"][0]) if payload["groups"].size else str(payload["relative_path"])
                for rest_payload in payload.get("rest_payloads", []):
                    phase_name = str(rest_payload["phase"])
                    rest_X = np.asarray(rest_payload["X"], dtype=np.float32)
                    rest_trial_ids = np.asarray(rest_payload["trial_ids"], dtype=np.int64)
                    if rest_X.ndim != 3 or rest_X.shape[0] == 0:
                        continue
                    for segment_index, parent_trial_id in enumerate(rest_trial_ids.tolist()):
                        all_rest_segments.append(np.asarray(rest_X[segment_index], dtype=np.float32))
                        all_rest_groups.append(payload_group)
                        all_rest_source_phases.append(phase_name)
                        all_rest_trial_keys.append(f"{payload_group}::{phase_name}-trial-{int(parent_trial_id):04d}")
                        all_rest_parent_trial_keys.append(f"{payload_group}::trial-{int(parent_trial_id):04d}")
        else:
            reference_samples = None
            dropped_short_runs = []
    else:
        reference_samples = None
        dropped_short_runs = []

    if not all_X:
        raise RuntimeError(
            "No usable valid trials were found. Please collect data first and avoid marking all trials as bad."
        )

    X_merged = np.concatenate([array[:, :, :min_samples] for array in all_X], axis=0)
    y_merged = np.concatenate(all_y, axis=0)
    return {
        "X": X_merged,
        "y": y_merged,
        "groups": np.concatenate(all_groups, axis=0),
        "trial_keys": np.concatenate(all_trial_keys, axis=0),
        "channel_names": channel_names,
        "class_names": class_names,
        "sampling_rate": sampling_rate,
        "full_window_sec": float(min_samples / float(sampling_rate)),
        "reference_epoch_samples": None if reference_samples is None else int(reference_samples),
        "target_epoch_samples": int(min_samples),
        "dropped_short_runs": dropped_short_runs,
        "rest_segments": all_rest_segments,
        "rest_groups": np.asarray(all_rest_groups, dtype=object),
        "rest_trial_keys": np.asarray(all_rest_trial_keys, dtype=object),
        "rest_parent_trial_keys": np.asarray(all_rest_parent_trial_keys, dtype=object),
        "rest_source_phases": np.asarray(all_rest_source_phases, dtype=object),
        "session_paths": session_paths,
        "source_records": source_records,
    }


def _infer_npz_task_and_run_stem(path: Path) -> tuple[str | None, str]:
    """Infer task type and run stem from one npz file path."""
    name = path.name
    for suffix, task in RUN_FILE_SUFFIXES.items():
        if name.endswith(suffix):
            return task, name[: -len(suffix)]
    return None, ""


def find_run_npz_files(dataset_root: Path, subject_filter: str | None) -> list[dict[str, object]]:
    """Discover per-run npz files and group them by run stem."""
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    grouped: dict[str, dict[str, object]] = {}
    for path in dataset_root.rglob("*.npz"):
        resolved = path.resolve()
        if subject_filter is not None:
            bare_subject = str(subject_filter).replace("sub-", "")
            full_path_token = str(resolved)
            if (
                subject_filter not in resolved.parts
                and full_path_token.find(subject_filter) < 0
                and full_path_token.find(f"sub-{bare_subject}") < 0
            ):
                continue
        task, run_stem = _infer_npz_task_and_run_stem(resolved)
        if task is None:
            continue
        if not run_stem:
            run_stem = str(resolved.relative_to(dataset_root)).replace("\\", "/")
        entry = grouped.setdefault(run_stem, {"run_stem": run_stem, "paths": {}})
        entry["paths"][task] = resolved

    runs = [grouped[key] for key in sorted(grouped.keys())]
    if not runs:
        raise FileNotFoundError(
            "No trainable data files were found. Expected at least one of: "
            "*_mi_epochs.npz / *_gate_epochs.npz / *_artifact_epochs.npz / *_continuous.npz."
        )
    return runs


def _load_npz_text(data: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    """Safely decode one scalar text field from npz."""
    if key not in data.files:
        return default
    return _decode_npz_text(np.asarray(data[key]).reshape(-1)[0]) or default


def _load_npz_int(data: np.lib.npyio.NpzFile, key: str, default: int = 0) -> int:
    """Safely decode one scalar integer field from npz."""
    if key not in data.files:
        return int(default)
    try:
        return int(np.asarray(data[key]).reshape(-1)[0])
    except Exception:
        return int(default)


def _load_derivative_sidecar(npz_path: Path) -> dict[str, object]:
    """Load one optional derivative sidecar json file."""
    candidate = npz_path.with_suffix(".meta.json")
    if not candidate.exists() or not candidate.is_file():
        return {}
    try:
        with candidate.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _ensure_common_metadata(
    *,
    path: Path,
    current_channel_names: list[str],
    current_class_names: list[str],
    current_sampling_rate: float,
    channel_names: list[str] | None,
    class_names: list[str] | None,
    sampling_rate: float | None,
) -> tuple[list[str], list[str], float]:
    """Validate and carry forward shared metadata across runs."""
    if channel_names is None:
        channel_names = list(current_channel_names)
    elif list(current_channel_names) != list(channel_names):
        raise ValueError(f"Channel names are inconsistent: {path}")

    if class_names is None:
        class_names = list(current_class_names)
    elif list(current_class_names) != list(class_names):
        raise ValueError(f"Class names are inconsistent: {path}")

    if sampling_rate is None:
        sampling_rate = float(current_sampling_rate)
    elif abs(float(current_sampling_rate) - float(sampling_rate)) > 1e-6:
        raise ValueError(f"Sampling rate is inconsistent: {path}")

    return channel_names, class_names, float(sampling_rate)


def _convert_signal_unit_to_volt(X: np.ndarray, signal_unit: str) -> np.ndarray:
    """Convert supported EEG units into volts."""
    unit = str(signal_unit).strip().lower()
    if unit in {"uv", "microvolt", "microvolts", "\u00b5v", "\u03bcv"}:
        return np.asarray(X, dtype=np.float32) * 1e-6
    return np.asarray(X, dtype=np.float32)


def _stack_task_segments(
    arrays: list[np.ndarray],
    *,
    channel_count: int,
    target_samples: int,
) -> np.ndarray:
    """Stack 3D arrays while trimming to a fixed sample length."""
    if target_samples <= 0:
        return np.empty((0, channel_count, 0), dtype=np.float32)
    normalized = [np.asarray(item, dtype=np.float32) for item in arrays if np.asarray(item).ndim == 3 and np.asarray(item).shape[0] > 0]
    if not normalized:
        return np.empty((0, channel_count, target_samples), dtype=np.float32)
    return np.concatenate([item[:, :, :target_samples] for item in normalized], axis=0).astype(np.float32)


def _exclude_continuous_sourced_segments(
    X: np.ndarray,
    sources: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop gate negatives that were extracted from continuous blocks to avoid evaluation leakage."""
    X = np.asarray(X, dtype=np.float32)
    sources = np.asarray(sources, dtype=object)
    if X.ndim != 3 or X.shape[0] == 0:
        return X, sources, 0
    if sources.shape[0] != X.shape[0]:
        return X, sources, 0
    keep_mask = np.asarray(
        [not _decode_npz_text(item).strip().lower().startswith("continuous") for item in sources.tolist()],
        dtype=bool,
    )
    dropped = int(np.sum(~keep_mask))
    return (
        np.asarray(X[keep_mask], dtype=np.float32),
        np.asarray(sources[keep_mask], dtype=object),
        dropped,
    )


def _keep_readable_npz(path: Path | None, *, label: str, run_issues: list[str]) -> Path | None:
    """Keep only readable/non-empty npz paths and record issues for dropped files."""
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        run_issues.append(f"{label}:missing")
        return None
    if not candidate.is_file():
        run_issues.append(f"{label}:not_file")
        return None
    try:
        if int(candidate.stat().st_size) <= 0:
            run_issues.append(f"{label}:empty")
            return None
    except OSError as error:
        run_issues.append(f"{label}:stat_error:{type(error).__name__}")
        return None
    try:
        with np.load(candidate, allow_pickle=False) as data:
            _ = tuple(data.files)
    except Exception as error:
        run_issues.append(f"{label}:load_error:{type(error).__name__}")
        return None
    return candidate


def load_custom_task_datasets(dataset_root: Path, subject_filter: str | None = None) -> dict[str, object]:
    """Load MI / gate / artifact / continuous datasets by task-specific npz files."""
    run_entries = find_run_npz_files(dataset_root, subject_filter)

    channel_names: list[str] | None = None
    class_names: list[str] | None = None
    sampling_rate: float | None = None

    mi_payloads: list[dict[str, object]] = []
    gate_payloads: list[dict[str, object]] = []
    artifact_payloads: list[dict[str, object]] = []
    continuous_records: list[dict[str, object]] = []
    source_records: list[dict[str, object]] = []

    for entry in run_entries:
        run_stem = str(entry["run_stem"])
        paths = dict(entry["paths"])
        mi_path = Path(paths["mi"]) if "mi" in paths else None
        gate_path = Path(paths["gate"]) if "gate" in paths else None
        artifact_path = Path(paths["artifact"]) if "artifact" in paths else None
        continuous_path = Path(paths["continuous"]) if "continuous" in paths else None
        run_issues: list[str] = []
        mi_path = _keep_readable_npz(mi_path, label="mi", run_issues=run_issues)
        gate_path = _keep_readable_npz(gate_path, label="gate", run_issues=run_issues)
        artifact_path = _keep_readable_npz(artifact_path, label="artifact", run_issues=run_issues)
        continuous_path = _keep_readable_npz(continuous_path, label="continuous", run_issues=run_issues)
        probe_path = mi_path or gate_path or artifact_path or continuous_path
        if probe_path is None:
            fallback_probe = next(
                (
                    Path(item)
                    for item in (
                        paths.get("mi"),
                        paths.get("gate"),
                        paths.get("artifact"),
                        paths.get("continuous"),
                    )
                    if item
                ),
                None,
            )
            probe_path = fallback_probe
        if probe_path is None:
            continue

        token = _parse_run_tokens_from_filename(probe_path)
        subject_id = str(token.get("subject_id", ""))
        session_id = str(token.get("session_id", ""))
        save_index = token.get("save_index")
        schema_version = 2
        sidecar_probe_path = next((path for path in (mi_path, gate_path, artifact_path, continuous_path) if path is not None), None)
        sidecar_probe = _load_derivative_sidecar(sidecar_probe_path) if sidecar_probe_path is not None else {}
        if sidecar_probe.get("schema_version") is not None:
            try:
                schema_version = int(sidecar_probe.get("schema_version") or schema_version)
            except Exception:
                schema_version = 2

        file_record = {
            "run_stem": run_stem,
            "subject_id": subject_id,
            "session_id": session_id,
            "save_index": save_index,
            "schema_version": int(schema_version),
            "mi_file": "" if mi_path is None else str(mi_path.relative_to(dataset_root)),
            "gate_file": "" if gate_path is None else str(gate_path.relative_to(dataset_root)),
            "artifact_file": "" if artifact_path is None else str(artifact_path.relative_to(dataset_root)),
            "continuous_file": "" if continuous_path is None else str(continuous_path.relative_to(dataset_root)),
            "mi_trials": 0,
            "mi_class_counts": {},
            "gate_pos_segments": 0,
            "gate_neg_segments": 0,
            "gate_hard_neg_segments": 0,
            "gate_neg_dropped_continuous": 0,
            "artifact_segments": 0,
            "continuous_blocks": 0,
            "continuous_prompts": 0,
            "dropped_reason": "",
        }
        if run_issues:
            file_record["dropped_reason"] = "|".join(run_issues)

        mi_X = np.empty((0, 0, 0), dtype=np.float32)
        mi_y = np.empty((0,), dtype=np.int64)
        mi_trial_ids = np.empty((0,), dtype=np.int64)
        gate_pos = np.empty((0, 0, 0), dtype=np.float32)
        gate_neg = np.empty((0, 0, 0), dtype=np.float32)
        gate_hard_neg = np.empty((0, 0, 0), dtype=np.float32)
        gate_neg_sources = np.asarray([], dtype=object)
        gate_hard_neg_sources = np.asarray([], dtype=object)
        artifact_X = np.empty((0, 0, 0), dtype=np.float32)
        artifact_labels = np.asarray([], dtype=object)

        if mi_path is not None:
            with np.load(mi_path, allow_pickle=False) as data:
                signal_unit = _load_npz_text(data, "signal_unit", "volt")
                current_channel_names = [_decode_npz_text(item) for item in np.asarray(data["channel_names"]).tolist()]
                current_class_names = [_decode_npz_text(item) for item in np.asarray(data["class_names"]).tolist()]
                current_sampling_rate = float(np.asarray(data["sampling_rate"]).reshape(-1)[0])
                channel_names, class_names, sampling_rate = _ensure_common_metadata(
                    path=mi_path,
                    current_channel_names=current_channel_names,
                    current_class_names=current_class_names,
                    current_sampling_rate=current_sampling_rate,
                    channel_names=channel_names,
                    class_names=class_names,
                    sampling_rate=sampling_rate,
                )
                subject_id = _load_npz_text(data, "subject_id", subject_id)
                session_id = _load_npz_text(data, "session_id", session_id)
                schema_version = _load_npz_int(data, "schema_version", schema_version)
                save_index = _load_npz_int(data, "save_index", save_index or 0)
                run_stem = _load_npz_text(data, "run_stem", run_stem) or run_stem

                mi_X = np.asarray(data["X_mi"], dtype=np.float32)
                mi_y = np.asarray(data["y_mi"], dtype=np.int64)
                mi_trial_ids = (
                    np.asarray(data["mi_trial_ids"], dtype=np.int64)
                    if "mi_trial_ids" in data.files
                    else np.arange(mi_X.shape[0], dtype=np.int64)
                )

                mi_X = _convert_signal_unit_to_volt(mi_X, signal_unit)

        if gate_path is not None:
            with np.load(gate_path, allow_pickle=False) as data:
                signal_unit = _load_npz_text(data, "signal_unit", "volt")
                current_channel_names = [_decode_npz_text(item) for item in np.asarray(data["channel_names"]).tolist()]
                current_class_names = [_decode_npz_text(item) for item in np.asarray(data["class_names"]).tolist()]
                current_sampling_rate = float(np.asarray(data["sampling_rate"]).reshape(-1)[0])
                channel_names, class_names, sampling_rate = _ensure_common_metadata(
                    path=gate_path,
                    current_channel_names=current_channel_names,
                    current_class_names=current_class_names,
                    current_sampling_rate=current_sampling_rate,
                    channel_names=channel_names,
                    class_names=class_names,
                    sampling_rate=sampling_rate,
                )
                schema_version = _load_npz_int(data, "schema_version", schema_version)
                save_index = _load_npz_int(data, "save_index", save_index or 0)
                run_stem = _load_npz_text(data, "run_stem", run_stem) or run_stem
                gate_pos = (
                    _convert_signal_unit_to_volt(np.asarray(data["X_gate_pos"], dtype=np.float32), signal_unit)
                    if "X_gate_pos" in data.files
                    else gate_pos
                )
                gate_neg = (
                    _convert_signal_unit_to_volt(np.asarray(data["X_gate_neg"], dtype=np.float32), signal_unit)
                    if "X_gate_neg" in data.files
                    else gate_neg
                )
                gate_neg_sources = (
                    np.asarray([_decode_npz_text(item) for item in np.asarray(data["gate_neg_sources"]).tolist()], dtype=object)
                    if "gate_neg_sources" in data.files
                    else gate_neg_sources
                )
                gate_hard_neg = (
                    _convert_signal_unit_to_volt(np.asarray(data["X_gate_hard_neg"], dtype=np.float32), signal_unit)
                    if "X_gate_hard_neg" in data.files
                    else gate_hard_neg
                )
                gate_hard_neg_sources = (
                    np.asarray(
                        [_decode_npz_text(item) for item in np.asarray(data["gate_hard_neg_sources"]).tolist()],
                        dtype=object,
                    )
                    if "gate_hard_neg_sources" in data.files
                    else gate_hard_neg_sources
                )

        if artifact_path is not None:
            with np.load(artifact_path, allow_pickle=False) as data:
                signal_unit = _load_npz_text(data, "signal_unit", "volt")
                current_channel_names = [_decode_npz_text(item) for item in np.asarray(data["channel_names"]).tolist()]
                current_class_names = [_decode_npz_text(item) for item in np.asarray(data["class_names"]).tolist()]
                current_sampling_rate = float(np.asarray(data["sampling_rate"]).reshape(-1)[0])
                channel_names, class_names, sampling_rate = _ensure_common_metadata(
                    path=artifact_path,
                    current_channel_names=current_channel_names,
                    current_class_names=current_class_names,
                    current_sampling_rate=current_sampling_rate,
                    channel_names=channel_names,
                    class_names=class_names,
                    sampling_rate=sampling_rate,
                )
                schema_version = _load_npz_int(data, "schema_version", schema_version)
                save_index = _load_npz_int(data, "save_index", save_index or 0)
                run_stem = _load_npz_text(data, "run_stem", run_stem) or run_stem
                if "X_artifact" in data.files:
                    artifact_X = _convert_signal_unit_to_volt(np.asarray(data["X_artifact"], dtype=np.float32), signal_unit)
                if "artifact_labels" in data.files:
                    artifact_labels = np.asarray(data["artifact_labels"], dtype=object)

        if continuous_path is not None:
            with np.load(continuous_path, allow_pickle=False) as data:
                signal_unit = _load_npz_text(data, "signal_unit", "volt")
                current_channel_names = [_decode_npz_text(item) for item in np.asarray(data["channel_names"]).tolist()]
                current_class_names = [_decode_npz_text(item) for item in np.asarray(data["class_names"]).tolist()]
                current_sampling_rate = float(np.asarray(data["sampling_rate"]).reshape(-1)[0])
                channel_names, class_names, sampling_rate = _ensure_common_metadata(
                    path=continuous_path,
                    current_channel_names=current_channel_names,
                    current_class_names=current_class_names,
                    current_sampling_rate=current_sampling_rate,
                    channel_names=channel_names,
                    class_names=class_names,
                    sampling_rate=sampling_rate,
                )
                schema_version = _load_npz_int(data, "schema_version", schema_version)
                save_index = _load_npz_int(data, "save_index", save_index or 0)
                run_stem = _load_npz_text(data, "run_stem", run_stem) or run_stem

                continuous_X = (
                    _convert_signal_unit_to_volt(np.asarray(data["X_continuous"], dtype=np.float32), signal_unit)
                    if "X_continuous" in data.files
                    else np.empty((0, len(channel_names or []), 0), dtype=np.float32)
                )
                event_labels = (
                    np.asarray([_decode_npz_text(item) for item in np.asarray(data["continuous_event_labels"]).tolist()], dtype=object)
                    if "continuous_event_labels" in data.files
                    else np.asarray([], dtype=object)
                )
                event_samples = (
                    np.asarray(data["continuous_event_samples"], dtype=np.int64)
                    if "continuous_event_samples" in data.files
                    else np.asarray([], dtype=np.int64)
                )
                block_starts = (
                    np.asarray(data["continuous_block_start_samples"], dtype=np.int64)
                    if "continuous_block_start_samples" in data.files
                    else np.asarray([], dtype=np.int64)
                )
                block_ends = (
                    np.asarray(data["continuous_block_end_samples"], dtype=np.int64)
                    if "continuous_block_end_samples" in data.files
                    else np.asarray([], dtype=np.int64)
                )
                event_block_indices = np.full(event_labels.shape[0], -1, dtype=np.int64)
                if "continuous_block_indices" in data.files:
                    raw_block_indices = np.asarray(data["continuous_block_indices"], dtype=np.int64).reshape(-1)
                    if raw_block_indices.shape[0] == event_block_indices.shape[0]:
                        event_block_indices = np.asarray(
                            [int(item - 1) if int(item) > 0 else -1 for item in raw_block_indices.tolist()],
                            dtype=np.int64,
                        )

                if block_starts.size and event_samples.size and block_ends.size:
                    for index, sample_index in enumerate(event_samples.tolist()):
                        if event_block_indices[index] >= 0:
                            continue
                        candidate = np.flatnonzero(
                            np.logical_and(block_starts <= int(sample_index), int(sample_index) < block_ends)
                        )
                        if candidate.size:
                            event_block_indices[index] = int(candidate[0])

                continuous_records.append(
                    {
                        "run_stem": run_stem,
                        "session_id": session_id or run_stem,
                        "relative_path": str(continuous_path.relative_to(dataset_root)),
                        "X": np.asarray(continuous_X, dtype=np.float32),
                        "event_labels": np.asarray(event_labels, dtype=object),
                        "event_samples": np.asarray(event_samples, dtype=np.int64),
                        "event_block_indices": np.asarray(event_block_indices, dtype=np.int64),
                        "block_start_samples": np.asarray(block_starts, dtype=np.int64),
                        "block_end_samples": np.asarray(block_ends, dtype=np.int64),
                    }
                )
                file_record["continuous_blocks"] = int(continuous_X.shape[0])
                file_record["continuous_prompts"] = int(event_labels.shape[0])

        group_token = run_stem or str(probe_path.relative_to(dataset_root)).replace("\\", "/")
        session_token = session_id or group_token
        file_record["run_stem"] = group_token
        file_record["subject_id"] = subject_id
        file_record["session_id"] = session_token
        file_record["save_index"] = save_index
        file_record["schema_version"] = int(schema_version)
        if run_issues:
            file_record["dropped_reason"] = "|".join(run_issues)

        if mi_X.ndim == 3 and mi_X.shape[0] > 0:
            class_count_lookup = Counter(int(label) for label in np.asarray(mi_y, dtype=np.int64).tolist())
            file_record["mi_class_counts"] = {
                str(class_name): int(class_count_lookup.get(index, 0))
                for index, class_name in enumerate(class_names or [])
            }
            mi_payloads.append(
                {
                    "relative_path": "" if mi_path is None else str(mi_path.relative_to(dataset_root)),
                    "group": group_token,
                    "session": session_token,
                    "sample_count": int(mi_X.shape[-1]),
                    "X": np.asarray(mi_X, dtype=np.float32),
                    "y": np.asarray(mi_y, dtype=np.int64),
                    "trial_keys": np.asarray(
                        [f"{group_token}::trial-{int(trial_id):04d}" for trial_id in np.asarray(mi_trial_ids, dtype=np.int64).tolist()],
                        dtype=object,
                    ),
                    "file_record": file_record,
                }
            )
            file_record["mi_trials"] = int(mi_X.shape[0])
        elif not str(file_record.get("dropped_reason", "")).strip():
            file_record["dropped_reason"] = "missing_or_invalid_mi_epochs"

        if gate_pos.ndim == 3 and gate_pos.shape[0] > 0:
            gate_payloads.append(
                {
                    "group": group_token,
                    "session": session_token,
                    "X_pos": np.asarray(gate_pos, dtype=np.float32),
                    "X_neg": np.asarray(gate_neg, dtype=np.float32),
                    "X_hard_neg": np.asarray(gate_hard_neg, dtype=np.float32),
                    "neg_sources": np.asarray(gate_neg_sources, dtype=object),
                    "hard_neg_sources": np.asarray(gate_hard_neg_sources, dtype=object),
                }
            )
        file_record["gate_pos_segments"] = int(gate_pos.shape[0]) if gate_pos.ndim == 3 else 0
        file_record["gate_neg_segments"] = int(gate_neg.shape[0]) if gate_neg.ndim == 3 else 0
        file_record["gate_hard_neg_segments"] = int(gate_hard_neg.shape[0]) if gate_hard_neg.ndim == 3 else 0

        if artifact_X.ndim == 3 and artifact_X.shape[0] > 0:
            artifact_payloads.append(
                {
                    "group": group_token,
                    "session": session_token,
                    "X_artifact": np.asarray(artifact_X, dtype=np.float32),
                    "artifact_labels": np.asarray(artifact_labels, dtype=object),
                }
            )
        file_record["artifact_segments"] = int(artifact_X.shape[0]) if artifact_X.ndim == 3 else 0
        source_records.append(file_record)

    if not mi_payloads:
        dropped_details = [
            f"{str(item.get('run_stem', 'unknown'))}:{str(item.get('dropped_reason', 'unknown'))}"
            for item in source_records
            if str(item.get("dropped_reason", "")).strip()
        ]
        detail_message = f" Dropped runs: {dropped_details[:8]}" if dropped_details else ""
        raise RuntimeError(
            "No usable MI training samples were found. "
            "Expected at least one valid *_mi_epochs.npz."
            + detail_message
        )
    if channel_names is None or class_names is None or sampling_rate is None:
        raise RuntimeError("Training metadata is incomplete: missing channel_names/class_names/sampling_rate.")

    sample_counts = [int(item["sample_count"]) for item in mi_payloads]
    length_histogram = Counter(sample_counts)
    reference_samples = max(length_histogram.items(), key=lambda item: (int(item[1]), int(item[0])))[0]
    retained_payloads: list[dict[str, object]] = []
    dropped_short_runs: list[str] = []
    for payload in mi_payloads:
        sample_count = int(payload["sample_count"])
        if sample_count < (int(reference_samples) - int(DEFAULT_EPOCH_LENGTH_TOLERANCE_SAMPLES)):
            payload["file_record"]["dropped_reason"] = f"too_short_for_common_epoch_length:{sample_count}<{reference_samples}"
            dropped_short_runs.append(str(payload["relative_path"]))
            continue
        retained_payloads.append(payload)
    if not retained_payloads:
        raise RuntimeError(
            "All MI runs are too short for the common epoch-length constraint, so training cannot continue."
        )

    target_mi_samples = int(min(int(item["sample_count"]) for item in retained_payloads))
    X_mi = np.concatenate(
        [np.asarray(item["X"], dtype=np.float32)[:, :, :target_mi_samples] for item in retained_payloads],
        axis=0,
    )
    y_mi = np.concatenate([np.asarray(item["y"], dtype=np.int64) for item in retained_payloads], axis=0)
    mi_groups = np.concatenate(
        [np.asarray([str(item["group"])] * np.asarray(item["X"]).shape[0], dtype=object) for item in retained_payloads],
        axis=0,
    )
    mi_sessions = np.concatenate(
        [np.asarray([str(item["session"])] * np.asarray(item["X"]).shape[0], dtype=object) for item in retained_payloads],
        axis=0,
    )
    mi_trial_keys = np.concatenate([np.asarray(item["trial_keys"], dtype=object) for item in retained_payloads], axis=0)
    session_paths = [str(item["relative_path"]) for item in retained_payloads if str(item["relative_path"])]

    gate_pos_arrays = [np.asarray(item["X_pos"], dtype=np.float32) for item in gate_payloads if np.asarray(item["X_pos"]).ndim == 3 and np.asarray(item["X_pos"]).shape[0] > 0]
    gate_neg_arrays = [np.asarray(item["X_neg"], dtype=np.float32) for item in gate_payloads if np.asarray(item["X_neg"]).ndim == 3 and np.asarray(item["X_neg"]).shape[0] > 0]
    gate_hard_neg_arrays = [np.asarray(item["X_hard_neg"], dtype=np.float32) for item in gate_payloads if np.asarray(item["X_hard_neg"]).ndim == 3 and np.asarray(item["X_hard_neg"]).shape[0] > 0]

    gate_target_samples = 0
    if gate_pos_arrays and gate_neg_arrays:
        gate_target_samples = int(
            min(
                min(array.shape[-1] for array in gate_pos_arrays),
                min(array.shape[-1] for array in gate_neg_arrays),
            )
        )
    elif gate_pos_arrays:
        gate_target_samples = int(min(array.shape[-1] for array in gate_pos_arrays))
    elif gate_neg_arrays:
        gate_target_samples = int(min(array.shape[-1] for array in gate_neg_arrays))

    X_gate_pos = np.empty((0, len(channel_names), 0), dtype=np.float32)
    X_gate_neg = np.empty((0, len(channel_names), 0), dtype=np.float32)
    X_gate_hard_neg = np.empty((0, len(channel_names), 0), dtype=np.float32)
    gate_pos_groups = np.asarray([], dtype=object)
    gate_neg_groups = np.asarray([], dtype=object)
    gate_hard_neg_groups = np.asarray([], dtype=object)
    gate_pos_sessions = np.asarray([], dtype=object)
    gate_neg_sessions = np.asarray([], dtype=object)
    gate_hard_neg_sessions = np.asarray([], dtype=object)
    gate_neg_sources_stacked = np.asarray([], dtype=object)
    gate_hard_neg_sources_stacked = np.asarray([], dtype=object)
    if gate_target_samples > 0:
        gate_pos_stack = []
        gate_neg_stack = []
        gate_hard_neg_stack = []
        gate_pos_group_list = []
        gate_neg_group_list = []
        gate_hard_neg_group_list = []
        gate_pos_session_list = []
        gate_neg_session_list = []
        gate_hard_neg_session_list = []
        gate_neg_source_list: list[str] = []
        gate_hard_neg_source_list: list[str] = []
        for payload in gate_payloads:
            pos_array = np.asarray(payload["X_pos"], dtype=np.float32)
            neg_array = np.asarray(payload["X_neg"], dtype=np.float32)
            hard_neg_array = np.asarray(payload["X_hard_neg"], dtype=np.float32)
            neg_sources = np.asarray(payload.get("neg_sources", []), dtype=object)
            hard_neg_sources = np.asarray(payload.get("hard_neg_sources", []), dtype=object)
            if pos_array.ndim == 3 and pos_array.shape[0] > 0:
                gate_pos_stack.append(pos_array[:, :, :gate_target_samples])
                gate_pos_group_list.extend([str(payload["group"])] * pos_array.shape[0])
                gate_pos_session_list.extend([str(payload["session"])] * pos_array.shape[0])
            if neg_array.ndim == 3 and neg_array.shape[0] > 0:
                gate_neg_stack.append(neg_array[:, :, :gate_target_samples])
                gate_neg_group_list.extend([str(payload["group"])] * neg_array.shape[0])
                gate_neg_session_list.extend([str(payload["session"])] * neg_array.shape[0])
                if neg_sources.shape[0] == neg_array.shape[0]:
                    gate_neg_source_list.extend([_decode_npz_text(item) for item in neg_sources.tolist()])
                else:
                    gate_neg_source_list.extend(["rest"] * neg_array.shape[0])
            if hard_neg_array.ndim == 3 and hard_neg_array.shape[0] > 0:
                gate_hard_neg_stack.append(hard_neg_array[:, :, :gate_target_samples])
                gate_hard_neg_group_list.extend([str(payload["group"])] * hard_neg_array.shape[0])
                gate_hard_neg_session_list.extend([str(payload["session"])] * hard_neg_array.shape[0])
                if hard_neg_sources.shape[0] == hard_neg_array.shape[0]:
                    gate_hard_neg_source_list.extend([_decode_npz_text(item) for item in hard_neg_sources.tolist()])
                else:
                    gate_hard_neg_source_list.extend(["hard_negative"] * hard_neg_array.shape[0])
        if gate_pos_stack:
            X_gate_pos = np.concatenate(gate_pos_stack, axis=0).astype(np.float32)
            gate_pos_groups = np.asarray(gate_pos_group_list, dtype=object)
            gate_pos_sessions = np.asarray(gate_pos_session_list, dtype=object)
        if gate_neg_stack:
            X_gate_neg = np.concatenate(gate_neg_stack, axis=0).astype(np.float32)
            gate_neg_groups = np.asarray(gate_neg_group_list, dtype=object)
            gate_neg_sessions = np.asarray(gate_neg_session_list, dtype=object)
            gate_neg_sources_stacked = np.asarray(gate_neg_source_list, dtype=object)
        if gate_hard_neg_stack:
            X_gate_hard_neg = np.concatenate(gate_hard_neg_stack, axis=0).astype(np.float32)
            gate_hard_neg_groups = np.asarray(gate_hard_neg_group_list, dtype=object)
            gate_hard_neg_sessions = np.asarray(gate_hard_neg_session_list, dtype=object)
            gate_hard_neg_sources_stacked = np.asarray(gate_hard_neg_source_list, dtype=object)

    artifact_arrays = [np.asarray(item["X_artifact"], dtype=np.float32) for item in artifact_payloads if np.asarray(item["X_artifact"]).ndim == 3 and np.asarray(item["X_artifact"]).shape[0] > 0]
    artifact_target_samples = 0
    if artifact_arrays:
        artifact_target_samples = int(min(array.shape[-1] for array in artifact_arrays))
        if X_gate_neg.shape[0] > 0 and X_gate_neg.shape[-1] > 0:
            artifact_target_samples = int(min(artifact_target_samples, X_gate_neg.shape[-1]))
        if X_gate_pos.shape[0] > 0 and X_gate_pos.shape[-1] > 0:
            artifact_target_samples = int(min(artifact_target_samples, X_gate_pos.shape[-1]))
    elif (X_gate_neg.shape[0] > 0 and X_gate_neg.shape[-1] > 0) or (X_gate_pos.shape[0] > 0 and X_gate_pos.shape[-1] > 0):
        candidate_lengths: list[int] = []
        if X_gate_neg.shape[0] > 0 and X_gate_neg.shape[-1] > 0:
            candidate_lengths.append(int(X_gate_neg.shape[-1]))
        if X_gate_pos.shape[0] > 0 and X_gate_pos.shape[-1] > 0:
            candidate_lengths.append(int(X_gate_pos.shape[-1]))
        artifact_target_samples = int(min(candidate_lengths))

    X_artifact = np.empty((0, len(channel_names), 0), dtype=np.float32)
    artifact_groups = np.asarray([], dtype=object)
    artifact_sessions = np.asarray([], dtype=object)
    artifact_segment_labels = np.asarray([], dtype=object)
    X_clean_negative = np.empty((0, len(channel_names), 0), dtype=np.float32)
    clean_negative_groups = np.asarray([], dtype=object)
    clean_negative_sessions = np.asarray([], dtype=object)
    if artifact_target_samples > 0:
        artifact_stack = []
        artifact_group_list = []
        artifact_session_list = []
        artifact_label_list = []
        for payload in artifact_payloads:
            artifact_array = np.asarray(payload["X_artifact"], dtype=np.float32)
            if artifact_array.ndim != 3 or artifact_array.shape[0] == 0:
                continue
            artifact_stack.append(artifact_array[:, :, :artifact_target_samples])
            artifact_group_list.extend([str(payload["group"])] * artifact_array.shape[0])
            artifact_session_list.extend([str(payload["session"])] * artifact_array.shape[0])
            labels = np.asarray(payload.get("artifact_labels", []), dtype=object)
            if labels.shape[0] == artifact_array.shape[0]:
                artifact_label_list.extend([_decode_npz_text(item) for item in labels.tolist()])
            else:
                artifact_label_list.extend(["artifact"] * artifact_array.shape[0])
        if artifact_stack:
            X_artifact = np.concatenate(artifact_stack, axis=0).astype(np.float32)
            artifact_groups = np.asarray(artifact_group_list, dtype=object)
            artifact_sessions = np.asarray(artifact_session_list, dtype=object)
            artifact_segment_labels = np.asarray(artifact_label_list, dtype=object)
        elif X_gate_hard_neg.shape[0] > 0:
            X_artifact = np.asarray(X_gate_hard_neg[:, :, :artifact_target_samples], dtype=np.float32)
            artifact_groups = np.asarray(gate_hard_neg_groups, dtype=object)
            artifact_sessions = np.asarray(gate_hard_neg_sessions, dtype=object)
            if gate_hard_neg_sources_stacked.shape[0] == X_artifact.shape[0]:
                artifact_segment_labels = np.asarray(
                    [_decode_npz_text(item) for item in gate_hard_neg_sources_stacked.tolist()],
                    dtype=object,
                )
            else:
                artifact_segment_labels = np.asarray(["hard_negative"] * X_artifact.shape[0], dtype=object)
        clean_negative_arrays: list[np.ndarray] = []
        clean_negative_group_parts: list[np.ndarray] = []
        clean_negative_session_parts: list[np.ndarray] = []
        if X_gate_pos.shape[0] > 0:
            clean_negative_arrays.append(np.asarray(X_gate_pos[:, :, :artifact_target_samples], dtype=np.float32))
            clean_negative_group_parts.append(np.asarray(gate_pos_groups, dtype=object))
            clean_negative_session_parts.append(np.asarray(gate_pos_sessions, dtype=object))
        if X_gate_neg.shape[0] > 0:
            clean_negative_arrays.append(np.asarray(X_gate_neg[:, :, :artifact_target_samples], dtype=np.float32))
            clean_negative_group_parts.append(np.asarray(gate_neg_groups, dtype=object))
            clean_negative_session_parts.append(np.asarray(gate_neg_sessions, dtype=object))
        if clean_negative_arrays:
            X_clean_negative = np.concatenate(clean_negative_arrays, axis=0).astype(np.float32)
            clean_negative_groups = np.concatenate(clean_negative_group_parts, axis=0)
            clean_negative_sessions = np.concatenate(clean_negative_session_parts, axis=0)

    return {
        "channel_names": list(channel_names),
        "class_names": list(class_names),
        "sampling_rate": float(sampling_rate),
        "session_paths": session_paths,
        "source_records": source_records,
        "mi": {
            "X": np.asarray(X_mi, dtype=np.float32),
            "y": np.asarray(y_mi, dtype=np.int64),
            "groups": np.asarray(mi_groups, dtype=object),
            "session_labels": np.asarray(mi_sessions, dtype=object),
            "trial_keys": np.asarray(mi_trial_keys, dtype=object),
            "full_window_sec": float(target_mi_samples / float(sampling_rate)),
            "reference_epoch_samples": int(reference_samples),
            "target_epoch_samples": int(target_mi_samples),
            "dropped_short_runs": dropped_short_runs,
        },
        "gate": {
            "X_pos": np.asarray(X_gate_pos, dtype=np.float32),
            "X_neg": np.asarray(X_gate_neg, dtype=np.float32),
            "X_hard_neg": np.asarray(X_gate_hard_neg, dtype=np.float32),
            "records": gate_payloads,
            "pos_groups": np.asarray(gate_pos_groups, dtype=object),
            "neg_groups": np.asarray(gate_neg_groups, dtype=object),
            "hard_neg_groups": np.asarray(gate_hard_neg_groups, dtype=object),
            "pos_session_labels": np.asarray(gate_pos_sessions, dtype=object),
            "neg_session_labels": np.asarray(gate_neg_sessions, dtype=object),
            "hard_neg_session_labels": np.asarray(gate_hard_neg_sessions, dtype=object),
            "neg_sources": np.asarray(gate_neg_sources_stacked, dtype=object),
            "hard_neg_sources": np.asarray(gate_hard_neg_sources_stacked, dtype=object),
            "target_epoch_samples": int(gate_target_samples),
        },
        "artifact": {
            "X_artifact": np.asarray(X_artifact, dtype=np.float32),
            "records": artifact_payloads,
            "artifact_labels": np.asarray(artifact_segment_labels, dtype=object),
            "artifact_groups": np.asarray(artifact_groups, dtype=object),
            "artifact_session_labels": np.asarray(artifact_sessions, dtype=object),
            "X_clean_negative": np.asarray(X_clean_negative, dtype=np.float32),
            "clean_negative_groups": np.asarray(clean_negative_groups, dtype=object),
            "clean_negative_session_labels": np.asarray(clean_negative_sessions, dtype=object),
            "target_epoch_samples": int(artifact_target_samples),
        },
        "continuous": {
            "records": continuous_records,
            "record_count": int(len(continuous_records)),
            "prompt_count": int(sum(int(record["event_labels"].shape[0]) for record in continuous_records)),
        },
    }

def preprocess_trials(X: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Apply the optimized classical preprocessing used by the realtime pipeline."""
    return preprocess(
        X,
        fs=float(sampling_rate),
        bandpass=DEFAULT_PREPROCESS_BANDPASS,
        notch=DEFAULT_PREPROCESS_NOTCH,
        apply_car=DEFAULT_PREPROCESS_APPLY_CAR,
        standardize_data=DEFAULT_PREPROCESS_STANDARDIZE,
    )


def build_preprocessing_config(
    *,
    window_sec: float,
    window_offset_sec: float,
    window_offset_secs_used: list[float] | None = None,
) -> dict[str, object]:
    """Build the preprocessing payload saved into each realtime artifact."""
    offset_candidates = [float(item) for item in (window_offset_secs_used or [window_offset_sec])]
    return {
        "bandpass": [float(DEFAULT_PREPROCESS_BANDPASS[0]), float(DEFAULT_PREPROCESS_BANDPASS[1])],
        "optimized_input_bandpass": [float(DEFAULT_PREPROCESS_BANDPASS[0]), float(DEFAULT_PREPROCESS_BANDPASS[1])],
        "notch": float(DEFAULT_PREPROCESS_NOTCH),
        "apply_car": bool(DEFAULT_PREPROCESS_APPLY_CAR),
        "standardize": bool(DEFAULT_PREPROCESS_STANDARDIZE),
        "epoch_window": [0.0, float(window_sec)],
        "window_offset_sec": float(window_offset_sec),
        "window_offset_secs_used": offset_candidates,
    }


def preprocessing_fingerprint(preprocessing_config: dict[str, object]) -> str:
    """Return a deterministic fingerprint so train/realtime preprocessing can be audited."""
    serialized = json.dumps(
        dict(preprocessing_config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_dataset_distribution(
    y: np.ndarray,
    class_names: list[str],
    *,
    min_class_trials: int,
) -> dict[str, int]:
    """Validate class counts before stratified splitting."""
    counts = Counter(int(label) for label in y.tolist())
    missing = [index for index in range(len(class_names)) if counts.get(index, 0) == 0]
    if missing:
        missing_names = [class_names[index] for index in missing]
        raise RuntimeError(f"Incomplete classes: missing {missing_names}")

    too_few = {index: count for index, count in counts.items() if count < int(min_class_trials)}
    if too_few:
        details = {class_names[index]: int(count) for index, count in sorted(too_few.items())}
        raise RuntimeError(
            f"Each class needs at least {int(min_class_trials)} valid trials for stable stratified training; current counts: {details}"
        )

    return {class_names[index]: int(count) for index, count in sorted(counts.items())}


def evaluate_dataset_readiness(
    *,
    label_distribution: dict[str, int],
    source_records: list[dict[str, object]],
    class_names: list[str],
    recommended_total_class_trials: int,
    recommended_run_class_trials: int,
) -> dict[str, object]:
    """Build readiness diagnostics so dataset quality checks are explicit and reproducible."""
    total_min = max(0, int(recommended_total_class_trials))
    run_min = max(0, int(recommended_run_class_trials))

    total_counts = {str(name): int(label_distribution.get(str(name), 0)) for name in class_names}
    classes_below_total = (
        {str(name): int(count) for name, count in total_counts.items() if int(count) < total_min}
        if total_min > 0
        else {}
    )

    run_checks: list[dict[str, object]] = []
    runs_below_min: list[dict[str, object]] = []
    for raw_record in source_records:
        if not isinstance(raw_record, dict):
            continue
        class_counts_raw = raw_record.get("mi_class_counts")
        if not isinstance(class_counts_raw, dict) or not class_counts_raw:
            continue

        class_counts = {
            str(name): int(class_counts_raw.get(str(name), 0))
            for name in class_names
        }
        min_count = int(min(class_counts.values())) if class_counts else 0
        run_summary = {
            "run_stem": str(raw_record.get("run_stem", "")),
            "session_id": str(raw_record.get("session_id", "")),
            "mi_class_counts": class_counts,
            "min_class_trials": int(min_count),
            "meets_recommended_min": bool(min_count >= run_min) if run_min > 0 else True,
        }
        run_checks.append(run_summary)
        if run_min > 0 and min_count < run_min:
            runs_below_min.append(run_summary)

    warnings: list[str] = []
    if classes_below_total:
        warnings.append(
            "per-class totals below recommended minimum "
            f"({total_min}): {classes_below_total}"
        )
    if runs_below_min:
        warnings.append(
            "runs with low per-class accepted trials "
            f"(recommended >= {run_min}): {[item['run_stem'] for item in runs_below_min]}"
        )

    dropped_continuous_gate_neg_total = int(
        sum(
            int(record.get("gate_neg_dropped_continuous", 0))
            for record in source_records
            if isinstance(record, dict)
        )
    )

    return {
        "ready_for_stable_comparison": bool(not warnings),
        "recommended_total_class_trials": int(total_min),
        "recommended_run_class_trials": int(run_min),
        "total_class_counts": total_counts,
        "classes_below_total_min": classes_below_total,
        "run_checks": run_checks,
        "runs_below_run_min": runs_below_min,
        "gate_neg_dropped_continuous_total": dropped_continuous_gate_neg_total,
        "warnings": warnings,
    }


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, float]:
    """Compute per-class test accuracy for easier diagnosis."""
    metrics: dict[str, float] = {}
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    for index, class_name in enumerate(class_names):
        mask = y_true == index
        if not np.any(mask):
            metrics[class_name] = 0.0
            continue
        metrics[class_name] = float(np.mean(y_pred[mask] == y_true[mask]))
    return metrics


def _pair_confusion_summary(
    matrix: np.ndarray,
    class_names: list[str],
    *,
    first_name: str,
    second_name: str,
) -> dict[str, float]:
    """Return directional and mutual confusion rates for one class pair."""
    normalized_lookup = {str(name).strip().lower(): index for index, name in enumerate(class_names)}
    first_index = normalized_lookup.get(str(first_name).strip().lower())
    second_index = normalized_lookup.get(str(second_name).strip().lower())
    if first_index is None or second_index is None:
        return {
            "first_to_second": 0.0,
            "second_to_first": 0.0,
            "mutual_confusion_rate": 0.0,
        }

    matrix = np.asarray(matrix, dtype=np.float64)
    first_row_total = float(np.sum(matrix[first_index, :]))
    second_row_total = float(np.sum(matrix[second_index, :]))
    first_to_second = float(matrix[first_index, second_index] / first_row_total) if first_row_total > 0.0 else 0.0
    second_to_first = float(matrix[second_index, first_index] / second_row_total) if second_row_total > 0.0 else 0.0
    pair_total = first_row_total + second_row_total
    mutual = (
        float((matrix[first_index, second_index] + matrix[second_index, first_index]) / pair_total)
        if pair_total > 0.0
        else 0.0
    )
    return {
        "first_to_second": first_to_second,
        "second_to_first": second_to_first,
        "mutual_confusion_rate": mutual,
    }


def macro_accuracy(per_class_metrics: dict[str, float]) -> float:
    """Return the unweighted mean over per-class accuracies."""
    if not per_class_metrics:
        return 0.0
    return float(np.mean(list(per_class_metrics.values())))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, object]:
    """Compute selection/evaluation metrics with macro-aware ranking support."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    labels = np.arange(len(class_names), dtype=np.int64)
    matrix = confusion_matrix(y_true, y_pred, labels=labels).astype(np.int64)
    per_class = per_class_accuracy(y_true, y_pred, class_names)
    left_right_confusion = _pair_confusion_summary(
        matrix,
        class_names,
        first_name="left_hand",
        second_name="right_hand",
    )
    feet_tongue_confusion = _pair_confusion_summary(
        matrix,
        class_names,
        first_name="feet",
        second_name="tongue",
    )
    acc = float(accuracy_score(y_true, y_pred))
    return {
        "acc": acc,
        "overall_acc": acc,
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_acc": macro_accuracy(per_class),
        "confusion_matrix": matrix.tolist(),
        "per_class_acc": per_class,
        "per_class_recall": dict(per_class),
        "left_right_confusion": left_right_confusion,
        "left_right_confusion_rate": float(left_right_confusion["mutual_confusion_rate"]),
        "feet_tongue_confusion": feet_tongue_confusion,
        "feet_tongue_confusion_rate": float(feet_tongue_confusion["mutual_confusion_rate"]),
    }


def metric_sort_key(metrics: dict[str, object]) -> tuple[float, float, float]:
    """Sort metrics with kappa first, then macro accuracy, then overall accuracy."""
    return (
        float(metrics.get("kappa", 0.0)),
        float(metrics.get("macro_acc", 0.0)),
        float(metrics.get("acc", 0.0)),
    )


def _clip_unit_interval(value: float) -> float:
    """Clamp a float into [0, 1]."""
    return float(np.clip(float(value), 0.0, 1.0))


def compute_selection_objective_score(
    *,
    metrics: dict[str, object],
    continuous_summary: dict[str, object],
) -> dict[str, object]:
    """Compute a deployment-oriented objective score from offline + continuous metrics."""
    selection_val_kappa = _clip_unit_interval(
        float(metrics.get("selection_val_kappa", metrics.get("bank_kappa", 0.0)))
    )
    selection_val_macro_f1 = _clip_unit_interval(
        float(metrics.get("selection_val_macro_f1", metrics.get("bank_macro_f1", 0.0)))
    )
    components = {
        "selection_val_kappa": selection_val_kappa,
        "selection_val_macro_f1": selection_val_macro_f1,
    }

    continuous_available = bool(continuous_summary.get("available", False))
    if continuous_available:
        mi_prompt_accuracy = _clip_unit_interval(float(continuous_summary.get("mi_prompt_accuracy", 0.0)))
        no_control_specificity = _clip_unit_interval(
            1.0 - float(continuous_summary.get("no_control_false_activation_rate", 1.0))
        )
        components["continuous_mi_prompt_accuracy"] = mi_prompt_accuracy
        components["continuous_no_control_specificity"] = no_control_specificity
        weights = {
            "selection_val_kappa": 0.35,
            "selection_val_macro_f1": 0.25,
            "continuous_mi_prompt_accuracy": 0.25,
            "continuous_no_control_specificity": 0.15,
        }
    else:
        weights = {
            "selection_val_kappa": 0.60,
            "selection_val_macro_f1": 0.40,
        }

    weight_sum = float(sum(float(weight) for weight in weights.values()))
    if weight_sum <= 0.0:
        score = 0.0
    else:
        score = float(
            sum(float(weights[name]) * float(components.get(name, 0.0)) for name in weights.keys()) / weight_sum
        )
    return {
        "score": _clip_unit_interval(score),
        "continuous_available": bool(continuous_available),
        "components": {str(name): float(value) for name, value in components.items()},
        "weights": {str(name): float(value) for name, value in weights.items()},
        "spec": dict(DEFAULT_SELECTION_OBJECTIVE_SPEC),
    }


def _predict_probability_matrix(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Return an (n_samples, n_classes) probability-like matrix for fusion."""
    return predict_probability_matrix(model, X, n_classes)


def calibrate_probability_matrix(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    role: str,
    selection_source: str,
) -> tuple[dict[str, object], np.ndarray]:
    """Fit and apply a temperature-scaling calibration object on validation probabilities."""
    calibration = fit_probability_calibration(
        probabilities,
        labels,
        role=role,
        selection_source=selection_source,
    )
    calibrated = apply_probability_calibration(probabilities, calibration)
    return calibration, np.asarray(calibrated, dtype=np.float64)


def fuse_probability_stack(
    probability_vectors: list[np.ndarray],
    fusion_weights: list[float] | tuple[float, ...] | np.ndarray,
    *,
    fusion_method: str,
) -> np.ndarray:
    """Fuse multi-window probability matrices sample-by-sample."""
    if not probability_vectors:
        raise ValueError("probability_vectors cannot be empty.")

    stacked = np.stack(probability_vectors, axis=0)
    normalized_weights = np.asarray(fusion_weights, dtype=np.float64)
    normalized_weights = np.nan_to_num(normalized_weights, nan=0.0, posinf=0.0, neginf=0.0)
    normalized_weights = np.clip(normalized_weights, a_min=0.0, a_max=None)
    total_weight = float(np.sum(normalized_weights))
    if total_weight <= 0.0:
        normalized_weights = np.full(stacked.shape[0], 1.0 / stacked.shape[0], dtype=np.float64)
    else:
        normalized_weights = normalized_weights / total_weight

    if str(fusion_method).strip().lower() == "log_weighted_mean":
        clipped = np.clip(stacked, 1e-8, 1.0)
        fused = np.exp(np.sum(normalized_weights[:, np.newaxis, np.newaxis] * np.log(clipped), axis=0))
    else:
        fused = np.sum(stacked * normalized_weights[:, np.newaxis, np.newaxis], axis=0)

    row_sums = np.sum(fused, axis=1, keepdims=True)
    row_sums[row_sums <= 0.0] = 1.0
    return fused / row_sums


def _simplex_weight_grid(weight_count: int, *, step: float) -> list[list[float]]:
    """Generate a simplex grid of non-negative fusion weights summing to 1."""
    if weight_count <= 0:
        raise ValueError("weight_count must be positive.")
    if weight_count == 1:
        return [[1.0]]

    normalized_step = float(step)
    if normalized_step <= 0.0 or normalized_step > 1.0:
        raise ValueError(f"Invalid fusion-weight grid step: {step}.")
    total_units = int(round(1.0 / normalized_step))
    if not np.isclose(total_units * normalized_step, 1.0, atol=1e-8):
        raise ValueError(f"fusion weight step must divide 1.0 exactly, got {step}.")

    unit_vectors: list[list[int]] = []

    def _recurse(remaining: int, remaining_slots: int, prefix: list[int]) -> None:
        if remaining_slots == 1:
            unit_vectors.append(prefix + [remaining])
            return
        for value in range(remaining + 1):
            _recurse(remaining - value, remaining_slots - 1, prefix + [value])

    _recurse(total_units, int(weight_count), [])
    return [[float(unit) / float(total_units) for unit in units] for units in unit_vectors]


def select_fusion_weights(
    probability_vectors: list[np.ndarray],
    y_true: np.ndarray,
    class_names: list[str],
    *,
    fusion_method: str,
    fallback_weights: list[float],
    search_step: float = DEFAULT_FUSION_WEIGHT_GRID_STEP,
) -> tuple[list[float], dict[str, object]]:
    """Tune fusion weights on validation outputs using kappa-first ranking."""
    if len(probability_vectors) == 1:
        only_metrics = classification_metrics(
            y_true,
            np.argmax(probability_vectors[0], axis=1),
            class_names,
        )
        return [1.0], {
            "strategy": "single_window",
            "candidate_count": 1,
            "metrics": only_metrics,
        }

    candidate_weights = _simplex_weight_grid(len(probability_vectors), step=float(search_step))
    best_weights = list(fallback_weights)
    best_metrics: dict[str, object] | None = None
    best_predictions: np.ndarray | None = None

    for weights in candidate_weights:
        fused = fuse_probability_stack(
            probability_vectors,
            weights,
            fusion_method=fusion_method,
        )
        predictions = np.argmax(fused, axis=1)
        metrics = classification_metrics(y_true, predictions, class_names)
        if best_metrics is None or metric_sort_key(metrics) > metric_sort_key(best_metrics):
            best_weights = [float(item) for item in weights]
            best_metrics = metrics
            best_predictions = predictions

    if best_metrics is None or best_predictions is None:
        raise RuntimeError("Failed to tune fusion weights on validation outputs.")

    return best_weights, {
        "strategy": "validation_grid_search",
        "candidate_count": len(candidate_weights),
        "grid_step": float(search_step),
        "metrics": best_metrics,
    }


def crop_trials_window(
    X: np.ndarray,
    sampling_rate: float,
    *,
    window_sec: float,
    offset_sec: float,
) -> np.ndarray:
    """Crop a fixed-length window from the imagery epoch."""
    X = np.asarray(X, dtype=np.float32)
    window_samples = int(round(float(window_sec) * float(sampling_rate)))
    start_sample = int(round(float(offset_sec) * float(sampling_rate)))
    stop_sample = start_sample + window_samples

    if window_samples <= 0:
        raise ValueError(f"window_sec must be positive, got {window_sec}.")
    if start_sample < 0:
        raise ValueError(f"offset_sec cannot be negative, got {offset_sec}.")
    if stop_sample > X.shape[-1]:
        full_window_sec = float(X.shape[-1] / float(sampling_rate))
        raise ValueError(
            f"Requested crop ({offset_sec:.3f}s + {window_sec:.3f}s) exceeds available epoch "
            f"length {full_window_sec:.3f}s."
        )

    return np.ascontiguousarray(X[:, :, start_sample:stop_sample], dtype=np.float32)


def crop_trials_multi_offset(
    X: np.ndarray,
    sampling_rate: float,
    *,
    window_sec: float,
    offset_secs: list[float],
) -> np.ndarray:
    """Crop and stack one control window per requested offset from each imagery trial."""
    windows = [
        crop_trials_window(
            X,
            sampling_rate,
            window_sec=float(window_sec),
            offset_sec=float(offset_sec),
        )
        for offset_sec in offset_secs
    ]
    if not windows:
        return np.empty((0, X.shape[1], 0), dtype=np.float32)
    return np.concatenate(windows, axis=0).astype(np.float32)


def required_window_samples(window_sec: float, sampling_rate: float, *, offset_sec: float = 0.0) -> int:
    """Return the sample count required for one cropped window."""
    total_sec = float(window_sec) + float(offset_sec)
    return int(round(total_sec * float(sampling_rate)))


def _crop_aux_segments(
    array: np.ndarray,
    *,
    required_samples: int,
    channel_count: int,
) -> np.ndarray:
    """Keep auxiliary segments only when they satisfy the required runtime window length."""
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] == 0:
        return np.empty((0, channel_count, required_samples), dtype=np.float32)
    if array.shape[1] != int(channel_count):
        raise ValueError(
            "Auxiliary dataset channel mismatch: "
            f"expected {channel_count}, got {array.shape[1]}."
        )
    if required_samples <= 0 or array.shape[2] < int(required_samples):
        return np.empty((0, channel_count, int(required_samples)), dtype=np.float32)
    return np.asarray(array[:, :, : int(required_samples)], dtype=np.float32)


def build_gate_branch_dataset(
    gate_records: list[dict[str, object]],
    *,
    required_samples: int,
    channel_count: int,
) -> dict[str, object]:
    """Materialize gate datasets from raw per-run records without global shortest-length truncation."""
    pos_parts: list[np.ndarray] = []
    neg_parts: list[np.ndarray] = []
    hard_neg_parts: list[np.ndarray] = []
    pos_groups: list[str] = []
    neg_groups: list[str] = []
    hard_neg_groups: list[str] = []
    pos_sessions: list[str] = []
    neg_sessions: list[str] = []
    hard_neg_sessions: list[str] = []
    neg_sources_out: list[str] = []
    hard_neg_sources_out: list[str] = []
    pos_max_samples = 0
    neg_max_samples = 0
    hard_neg_max_samples = 0
    raw_present = False

    for record in gate_records:
        group = str(record.get("group", ""))
        session = str(record.get("session", group))

        pos_array = np.asarray(record.get("X_pos", np.empty((0, channel_count, 0), dtype=np.float32)), dtype=np.float32)
        neg_array = np.asarray(record.get("X_neg", np.empty((0, channel_count, 0), dtype=np.float32)), dtype=np.float32)
        hard_neg_array = np.asarray(
            record.get("X_hard_neg", np.empty((0, channel_count, 0), dtype=np.float32)),
            dtype=np.float32,
        )
        neg_sources = np.asarray(record.get("neg_sources", []), dtype=object)
        hard_neg_sources = np.asarray(record.get("hard_neg_sources", []), dtype=object)

        if pos_array.ndim == 3 and pos_array.shape[0] > 0:
            raw_present = True
            pos_max_samples = max(pos_max_samples, int(pos_array.shape[2]))
            cropped = _crop_aux_segments(pos_array, required_samples=required_samples, channel_count=channel_count)
            if cropped.shape[0] > 0:
                pos_parts.append(cropped)
                pos_groups.extend([group] * cropped.shape[0])
                pos_sessions.extend([session] * cropped.shape[0])

        if neg_array.ndim == 3 and neg_array.shape[0] > 0:
            raw_present = True
            neg_max_samples = max(neg_max_samples, int(neg_array.shape[2]))
            cropped = _crop_aux_segments(neg_array, required_samples=required_samples, channel_count=channel_count)
            if cropped.shape[0] > 0:
                neg_parts.append(cropped)
                neg_groups.extend([group] * cropped.shape[0])
                neg_sessions.extend([session] * cropped.shape[0])
                if neg_sources.shape[0] == neg_array.shape[0]:
                    neg_sources_out.extend([_decode_npz_text(item) for item in neg_sources[: cropped.shape[0]].tolist()])
                else:
                    neg_sources_out.extend(["rest"] * cropped.shape[0])

        if hard_neg_array.ndim == 3 and hard_neg_array.shape[0] > 0:
            raw_present = True
            hard_neg_max_samples = max(hard_neg_max_samples, int(hard_neg_array.shape[2]))
            cropped = _crop_aux_segments(hard_neg_array, required_samples=required_samples, channel_count=channel_count)
            if cropped.shape[0] > 0:
                hard_neg_parts.append(cropped)
                hard_neg_groups.extend([group] * cropped.shape[0])
                hard_neg_sessions.extend([session] * cropped.shape[0])
                if hard_neg_sources.shape[0] == hard_neg_array.shape[0]:
                    hard_neg_sources_out.extend(
                        [_decode_npz_text(item) for item in hard_neg_sources[: cropped.shape[0]].tolist()]
                    )
                else:
                    hard_neg_sources_out.extend(["hard_negative"] * cropped.shape[0])

    X_pos = np.concatenate(pos_parts, axis=0).astype(np.float32) if pos_parts else np.empty((0, channel_count, required_samples), dtype=np.float32)
    X_neg = np.concatenate(neg_parts, axis=0).astype(np.float32) if neg_parts else np.empty((0, channel_count, required_samples), dtype=np.float32)
    X_hard_neg = (
        np.concatenate(hard_neg_parts, axis=0).astype(np.float32)
        if hard_neg_parts
        else np.empty((0, channel_count, required_samples), dtype=np.float32)
    )
    return {
        "X_pos": X_pos,
        "X_neg": X_neg,
        "X_hard_neg": X_hard_neg,
        "pos_groups": np.asarray(pos_groups, dtype=object),
        "neg_groups": np.asarray(neg_groups, dtype=object),
        "hard_neg_groups": np.asarray(hard_neg_groups, dtype=object),
        "pos_session_labels": np.asarray(pos_sessions, dtype=object),
        "neg_session_labels": np.asarray(neg_sessions, dtype=object),
        "hard_neg_session_labels": np.asarray(hard_neg_sessions, dtype=object),
        "neg_sources": np.asarray(neg_sources_out, dtype=object),
        "hard_neg_sources": np.asarray(hard_neg_sources_out, dtype=object),
        "raw_present": bool(raw_present),
        "pos_max_samples": int(pos_max_samples),
        "neg_max_samples": int(neg_max_samples),
        "hard_neg_max_samples": int(hard_neg_max_samples),
        "required_samples": int(required_samples),
    }


def build_artifact_branch_dataset(
    artifact_records: list[dict[str, object]],
    gate_records: list[dict[str, object]],
    *,
    required_samples: int,
    channel_count: int,
) -> dict[str, object]:
    """Materialize artifact datasets from raw per-run records without borrowing the shortest gate run."""
    artifact_parts: list[np.ndarray] = []
    artifact_groups: list[str] = []
    artifact_sessions: list[str] = []
    artifact_labels: list[str] = []
    clean_negative_parts: list[np.ndarray] = []
    clean_negative_groups: list[str] = []
    clean_negative_sessions: list[str] = []
    artifact_max_samples = 0
    clean_negative_max_samples = 0
    raw_present = False

    for record in artifact_records:
        group = str(record.get("group", ""))
        session = str(record.get("session", group))
        artifact_array = np.asarray(
            record.get("X_artifact", np.empty((0, channel_count, 0), dtype=np.float32)),
            dtype=np.float32,
        )
        labels = np.asarray(record.get("artifact_labels", []), dtype=object)
        if artifact_array.ndim != 3 or artifact_array.shape[0] == 0:
            continue
        raw_present = True
        artifact_max_samples = max(artifact_max_samples, int(artifact_array.shape[2]))
        cropped = _crop_aux_segments(artifact_array, required_samples=required_samples, channel_count=channel_count)
        if cropped.shape[0] == 0:
            continue
        artifact_parts.append(cropped)
        artifact_groups.extend([group] * cropped.shape[0])
        artifact_sessions.extend([session] * cropped.shape[0])
        if labels.shape[0] == artifact_array.shape[0]:
            artifact_labels.extend([_decode_npz_text(item) for item in labels[: cropped.shape[0]].tolist()])
        else:
            artifact_labels.extend(["artifact"] * cropped.shape[0])

    for record in gate_records:
        group = str(record.get("group", ""))
        session = str(record.get("session", group))

        pos_array = np.asarray(record.get("X_pos", np.empty((0, channel_count, 0), dtype=np.float32)), dtype=np.float32)
        neg_array = np.asarray(record.get("X_neg", np.empty((0, channel_count, 0), dtype=np.float32)), dtype=np.float32)
        hard_neg_array = np.asarray(
            record.get("X_hard_neg", np.empty((0, channel_count, 0), dtype=np.float32)),
            dtype=np.float32,
        )
        hard_neg_sources = np.asarray(record.get("hard_neg_sources", []), dtype=object)

        for candidate in (pos_array, neg_array):
            if candidate.ndim != 3 or candidate.shape[0] == 0:
                continue
            raw_present = True
            clean_negative_max_samples = max(clean_negative_max_samples, int(candidate.shape[2]))
            cropped = _crop_aux_segments(candidate, required_samples=required_samples, channel_count=channel_count)
            if cropped.shape[0] == 0:
                continue
            clean_negative_parts.append(cropped)
            clean_negative_groups.extend([group] * cropped.shape[0])
            clean_negative_sessions.extend([session] * cropped.shape[0])

        if not artifact_parts and hard_neg_array.ndim == 3 and hard_neg_array.shape[0] > 0:
            raw_present = True
            artifact_max_samples = max(artifact_max_samples, int(hard_neg_array.shape[2]))
            cropped_hard = _crop_aux_segments(
                hard_neg_array,
                required_samples=required_samples,
                channel_count=channel_count,
            )
            if cropped_hard.shape[0] > 0:
                artifact_parts.append(cropped_hard)
                artifact_groups.extend([group] * cropped_hard.shape[0])
                artifact_sessions.extend([session] * cropped_hard.shape[0])
                if hard_neg_sources.shape[0] == hard_neg_array.shape[0]:
                    artifact_labels.extend([_decode_npz_text(item) for item in hard_neg_sources[: cropped_hard.shape[0]].tolist()])
                else:
                    artifact_labels.extend(["hard_negative"] * cropped_hard.shape[0])

    X_artifact = (
        np.concatenate(artifact_parts, axis=0).astype(np.float32)
        if artifact_parts
        else np.empty((0, channel_count, required_samples), dtype=np.float32)
    )
    X_clean_negative = (
        np.concatenate(clean_negative_parts, axis=0).astype(np.float32)
        if clean_negative_parts
        else np.empty((0, channel_count, required_samples), dtype=np.float32)
    )
    return {
        "X_artifact": X_artifact,
        "artifact_labels": np.asarray(artifact_labels, dtype=object),
        "artifact_groups": np.asarray(artifact_groups, dtype=object),
        "artifact_session_labels": np.asarray(artifact_sessions, dtype=object),
        "X_clean_negative": X_clean_negative,
        "clean_negative_groups": np.asarray(clean_negative_groups, dtype=object),
        "clean_negative_session_labels": np.asarray(clean_negative_sessions, dtype=object),
        "raw_present": bool(raw_present),
        "artifact_max_samples": int(artifact_max_samples),
        "clean_negative_max_samples": int(clean_negative_max_samples),
        "required_samples": int(required_samples),
    }


def split_continuous_records_by_group_sets(
    records: list[dict[str, object]],
    *,
    train_groups: set[str],
    val_groups: set[str],
    test_groups: set[str],
    train_sessions: set[str] | None = None,
    val_sessions: set[str] | None = None,
    test_sessions: set[str] | None = None,
) -> dict[str, object]:
    """Assign continuous records to train/val/test buckets using run-group, then session fallback."""
    buckets = {
        "train": [],
        "val": [],
        "test": [],
        "unassigned": [],
    }
    assignments: list[dict[str, str]] = []
    train_sessions = set(str(item) for item in (train_sessions or set()))
    val_sessions = set(str(item) for item in (val_sessions or set()))
    test_sessions = set(str(item) for item in (test_sessions or set()))

    for raw_record in records:
        record = dict(raw_record)
        group = str(record.get("run_stem", "")).strip()
        session = str(record.get("session_id", "")).strip()
        bucket = "unassigned"
        matched_by = "none"
        if group and group in train_groups:
            bucket = "train"
            matched_by = "group"
        elif group and group in val_groups:
            bucket = "val"
            matched_by = "group"
        elif group and group in test_groups:
            bucket = "test"
            matched_by = "group"
        elif session and session in train_sessions:
            bucket = "train"
            matched_by = "session"
        elif session and session in val_sessions:
            bucket = "val"
            matched_by = "session"
        elif session and session in test_sessions:
            bucket = "test"
            matched_by = "session"
        buckets[bucket].append(record)
        assignments.append(
            {
                "run_stem": group,
                "session_id": session,
                "split": bucket,
                "matched_by": matched_by,
            }
        )

    buckets["all"] = [dict(item) for item in records]
    buckets["trainval"] = [dict(item) for item in buckets["train"]] + [dict(item) for item in buckets["val"]]
    buckets["assignments"] = assignments
    return buckets


def stack_multi_offset_split(
    processed_by_offset: dict[float, dict[str, np.ndarray]],
    split_name: str,
    offset_secs: list[float],
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack one split across all offsets and repeat labels to match the augmentation."""
    offset_windows = [
        np.asarray(processed_by_offset[float(offset_sec)][split_name], dtype=np.float32)
        for offset_sec in offset_secs
    ]
    if not offset_windows:
        raise ValueError(f"No offset windows were provided for split={split_name!r}.")
    stacked_X = np.concatenate(offset_windows, axis=0).astype(np.float32)
    repeated_y = np.concatenate(
        [np.asarray(labels, dtype=np.int64) for _ in offset_secs],
        axis=0,
    )
    return stacked_X, repeated_y


def predict_multi_offset_probability_matrix(
    pipeline,
    processed_by_offset: dict[float, dict[str, np.ndarray]],
    split_name: str,
    offset_secs: list[float],
    class_count: int,
    probability_calibration: dict[str, object] | None = None,
) -> np.ndarray:
    """Predict one probability row per original trial by averaging over all configured offsets."""
    probability_matrices = []

    for offset_sec in offset_secs:
        split_X = np.asarray(processed_by_offset[float(offset_sec)][split_name], dtype=np.float32)
        probabilities = _predict_probability_matrix(pipeline, split_X, class_count)
        probability_matrices.append(np.asarray(probabilities, dtype=np.float64))

    if not probability_matrices:
        raise ValueError(f"No probability matrices were produced for split={split_name!r}.")

    stacked = np.stack(probability_matrices, axis=0)
    averaged = np.mean(stacked, axis=0, dtype=np.float64)
    if probability_calibration:
        averaged = apply_probability_calibration(averaged, probability_calibration)
    return np.asarray(averaged, dtype=np.float64)


def build_binary_dataset(
    control_X: np.ndarray,
    rest_X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine control and rest windows into a binary dataset."""
    control_X = np.asarray(control_X, dtype=np.float32)
    rest_X = np.asarray(rest_X, dtype=np.float32)
    if control_X.ndim != 3 or rest_X.ndim != 3:
        raise ValueError("Binary gate datasets must have shape (n_trials, channels, samples).")
    if control_X.shape[1:] != rest_X.shape[1:]:
        raise ValueError(
            f"Control/rest shape mismatch: {control_X.shape[1:]} vs {rest_X.shape[1:]}."
        )
    X = np.concatenate([control_X, rest_X], axis=0).astype(np.float32)
    y = np.concatenate(
        [
            np.ones(control_X.shape[0], dtype=np.int64),
            np.zeros(rest_X.shape[0], dtype=np.int64),
        ],
        axis=0,
    )
    return X, y


def balance_binary_dataset(
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample the majority class so binary gate training stays balanced."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    class_ids, class_counts = np.unique(y, return_counts=True)
    if class_ids.shape[0] != 2:
        return X, y
    target_count = int(np.min(class_counts))
    if target_count <= 0:
        return X, y

    rng = np.random.default_rng(int(random_state))
    kept_indices = []
    for class_id in class_ids.tolist():
        class_indices = np.flatnonzero(y == int(class_id))
        if class_indices.shape[0] > target_count:
            class_indices = np.sort(rng.choice(class_indices, size=target_count, replace=False))
        kept_indices.append(class_indices)

    selected = np.sort(np.concatenate(kept_indices, axis=0))
    return X[selected].astype(np.float32), y[selected].astype(np.int64)


def sample_rest_windows(
    rest_segments: list[np.ndarray],
    rest_source_phases: np.ndarray,
    sampling_rate: float,
    *,
    window_sec: float,
    step_sec: float = DEFAULT_REST_WINDOW_STEP_SEC,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract fixed-length sliding windows from baseline/ITI rest segments."""
    window_samples = int(round(float(window_sec) * float(sampling_rate)))
    step_samples = max(1, int(round(float(step_sec) * float(sampling_rate))))
    windows: list[np.ndarray] = []
    window_phases: list[str] = []

    for segment, phase_name in zip(rest_segments, np.asarray(rest_source_phases, dtype=object).tolist()):
        segment = np.asarray(segment, dtype=np.float32)
        if segment.ndim != 2 or segment.shape[1] < window_samples:
            continue
        stop_positions = list(range(window_samples, int(segment.shape[1]) + 1, step_samples))
        if not stop_positions or stop_positions[-1] != int(segment.shape[1]):
            stop_positions.append(int(segment.shape[1]))
        for stop_sample in stop_positions:
            start_sample = int(stop_sample - window_samples)
            windows.append(np.ascontiguousarray(segment[:, start_sample:stop_sample], dtype=np.float32))
            window_phases.append(str(phase_name))

    if not windows:
        channel_count = 0 if not rest_segments else int(np.asarray(rest_segments[0]).shape[0])
        return (
            np.empty((0, channel_count, window_samples), dtype=np.float32),
            np.asarray([], dtype=object),
        )

    return np.stack(windows, axis=0), np.asarray(window_phases, dtype=object)


def sample_rest_windows_aligned(
    rest_segments: list[np.ndarray],
    rest_source_phases: np.ndarray,
    sampling_rate: float,
    *,
    window_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one deterministic trailing rest window per rest segment."""
    window_samples = int(round(float(window_sec) * float(sampling_rate)))
    windows: list[np.ndarray] = []
    window_phases: list[str] = []

    for segment, phase_name in zip(rest_segments, np.asarray(rest_source_phases, dtype=object).tolist()):
        segment = np.asarray(segment, dtype=np.float32)
        if segment.ndim != 2 or segment.shape[1] < window_samples:
            continue
        windows.append(np.ascontiguousarray(segment[:, -window_samples:], dtype=np.float32))
        window_phases.append(str(phase_name))

    if not windows:
        channel_count = 0 if not rest_segments else int(np.asarray(rest_segments[0]).shape[0])
        return (
            np.empty((0, channel_count, window_samples), dtype=np.float32),
            np.asarray([], dtype=object),
        )

    return np.stack(windows, axis=0), np.asarray(window_phases, dtype=object)


def evaluate_bank_on_raw_windows(
    raw_windows: np.ndarray,
    member_artifacts: list[dict[str, object]],
    *,
    sampling_rate: float,
    fusion_weights: list[float] | tuple[float, ...] | np.ndarray,
    fusion_method: str,
    class_count: int,
    probability_calibration: dict[str, object] | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run each selected member on raw windows and fuse the resulting probabilities."""
    raw_windows = np.asarray(raw_windows, dtype=np.float32)
    if raw_windows.ndim != 3:
        raise ValueError("raw_windows must have shape (n_windows, channels, samples).")
    if raw_windows.shape[0] == 0:
        raise ValueError("raw_windows cannot be empty.")

    probability_vectors: list[np.ndarray] = []
    for member_artifact in member_artifacts:
        window_sec = float(member_artifact["window_sec"])
        window_samples = int(round(window_sec * float(sampling_rate)))
        if raw_windows.shape[-1] < window_samples:
            raise ValueError(
                f"Raw windows are shorter than required member window {window_sec:.3f}s."
            )
        member_input = np.ascontiguousarray(raw_windows[:, :, -window_samples:], dtype=np.float32)
        processed = preprocess_trials(member_input, sampling_rate)
        probabilities = predict_probability_matrix(
            member_artifact["pipeline"],
            processed,
            class_count,
            probability_calibration=member_artifact.get("probability_calibration"),
        )
        probability_vectors.append(np.asarray(probabilities, dtype=np.float64))

    fused = fuse_probability_stack(
        probability_vectors,
        fusion_weights,
        fusion_method=fusion_method,
    )
    if probability_calibration:
        fused = apply_probability_calibration(fused, probability_calibration)
    return fused, probability_vectors


def confidence_margin_from_probabilities(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return top-1 confidence and top1-top2 margin from class probability rows."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D matrix.")
    if probabilities.shape[1] == 0:
        raise ValueError("probabilities must contain at least one class.")
    sorted_probabilities = np.sort(probabilities, axis=1)
    top1 = sorted_probabilities[:, -1]
    top2 = np.zeros_like(top1) if probabilities.shape[1] == 1 else sorted_probabilities[:, -2]
    return top1, top1 - top2


def select_reject_thresholds(
    control_probabilities: np.ndarray,
    rest_probabilities: np.ndarray,
    *,
    target_rest_activation_rate: float = DEFAULT_REST_FALSE_ACTIVATION_TARGET,
) -> dict[str, object]:
    """Pick confidence and margin thresholds that suppress rest false activations."""
    control_confidence, control_margin = confidence_margin_from_probabilities(control_probabilities)
    rest_confidence, rest_margin = confidence_margin_from_probabilities(rest_probabilities)

    confidence_grid = np.round(np.arange(0.30, 0.86, 0.025), 3)
    margin_grid = np.round(np.arange(0.02, 0.31, 0.02), 3)
    best_result: dict[str, object] | None = None
    best_key: tuple[float, ...] | None = None

    for confidence_threshold in confidence_grid.tolist():
        for margin_threshold in margin_grid.tolist():
            control_mask = np.logical_and(
                control_confidence >= float(confidence_threshold),
                control_margin >= float(margin_threshold),
            )
            rest_mask = np.logical_and(
                rest_confidence >= float(confidence_threshold),
                rest_margin >= float(margin_threshold),
            )
            control_detection_rate = float(np.mean(control_mask)) if control_mask.size else 0.0
            rest_false_activation_rate = float(np.mean(rest_mask)) if rest_mask.size else 0.0
            rest_reject_rate = 1.0 - rest_false_activation_rate
            feasible = 1.0 if rest_false_activation_rate <= float(target_rest_activation_rate) else 0.0
            ranking_key = (
                feasible,
                control_detection_rate if feasible else rest_reject_rate,
                rest_reject_rate if feasible else control_detection_rate,
                -float(confidence_threshold),
                -float(margin_threshold),
            )
            if best_key is None or ranking_key > best_key:
                best_key = ranking_key
                best_result = {
                    "confidence_threshold": float(confidence_threshold),
                    "margin_threshold": float(margin_threshold),
                    "control_detection_rate": control_detection_rate,
                    "rest_false_activation_rate": rest_false_activation_rate,
                    "rest_reject_rate": rest_reject_rate,
                    "target_rest_activation_rate": float(target_rest_activation_rate),
                    "selection_feasible": bool(feasible),
                    "control_window_count": int(control_mask.size),
                    "rest_window_count": int(rest_mask.size),
                }

    if best_result is None:
        raise RuntimeError("Failed to select reject thresholds from rest/control probabilities.")
    return best_result


def select_gate_thresholds(
    control_probabilities: np.ndarray,
    rest_probabilities: np.ndarray,
    *,
    target_rest_activation_rate: float = DEFAULT_REST_FALSE_ACTIVATION_TARGET,
    control_class_index: int = 1,
) -> dict[str, object]:
    """Pick gate thresholds for the explicit control-vs-rest stage."""
    control_probabilities = np.asarray(control_probabilities, dtype=np.float64)
    rest_probabilities = np.asarray(rest_probabilities, dtype=np.float64)
    if control_probabilities.ndim != 2 or rest_probabilities.ndim != 2:
        raise ValueError("Gate probabilities must be 2D matrices.")
    if control_probabilities.shape[1] <= int(control_class_index):
        raise ValueError("control_class_index is out of range for gate probabilities.")

    control_confidence = np.asarray(control_probabilities[:, int(control_class_index)], dtype=np.float64)
    rest_confidence = np.asarray(rest_probabilities[:, int(control_class_index)], dtype=np.float64)
    _, control_margin = confidence_margin_from_probabilities(control_probabilities)
    _, rest_margin = confidence_margin_from_probabilities(rest_probabilities)
    control_predicted_class = np.argmax(control_probabilities, axis=1)
    rest_predicted_class = np.argmax(rest_probabilities, axis=1)

    confidence_grid = np.round(np.arange(0.40, 0.96, 0.025), 3)
    margin_grid = np.round(np.arange(0.00, 0.41, 0.02), 3)
    best_result: dict[str, object] | None = None
    best_key: tuple[float, ...] | None = None

    for confidence_threshold in confidence_grid.tolist():
        for margin_threshold in margin_grid.tolist():
            control_mask = np.logical_and.reduce(
                [
                    control_predicted_class == int(control_class_index),
                    control_confidence >= float(confidence_threshold),
                    control_margin >= float(margin_threshold),
                ]
            )
            rest_mask = np.logical_and.reduce(
                [
                    rest_predicted_class == int(control_class_index),
                    rest_confidence >= float(confidence_threshold),
                    rest_margin >= float(margin_threshold),
                ]
            )
            control_detection_rate = float(np.mean(control_mask)) if control_mask.size else 0.0
            rest_false_activation_rate = float(np.mean(rest_mask)) if rest_mask.size else 0.0
            feasible = 1.0 if rest_false_activation_rate <= float(target_rest_activation_rate) else 0.0
            ranking_key = (
                feasible,
                control_detection_rate if feasible else -rest_false_activation_rate,
                -rest_false_activation_rate,
                -float(confidence_threshold),
                -float(margin_threshold),
            )
            if best_key is None or ranking_key > best_key:
                best_key = ranking_key
                best_result = {
                    "confidence_threshold": float(confidence_threshold),
                    "margin_threshold": float(margin_threshold),
                    "control_detection_rate": control_detection_rate,
                    "rest_false_activation_rate": rest_false_activation_rate,
                    "target_rest_activation_rate": float(target_rest_activation_rate),
                    "selection_feasible": bool(feasible),
                    "control_window_count": int(control_mask.size),
                    "rest_window_count": int(rest_mask.size),
                    "control_class_index": int(control_class_index),
                }

    if best_result is None:
        raise RuntimeError("Failed to select control-gate thresholds.")
    return best_result


def evaluate_gate_thresholds(
    control_probabilities: np.ndarray,
    rest_probabilities: np.ndarray,
    *,
    confidence_threshold: float,
    margin_threshold: float,
    control_class_index: int = 1,
) -> dict[str, float]:
    """Evaluate a fixed gate-threshold pair on held-out control/rest probabilities."""
    control_probabilities = np.asarray(control_probabilities, dtype=np.float64)
    rest_probabilities = np.asarray(rest_probabilities, dtype=np.float64)
    control_confidence = np.asarray(control_probabilities[:, int(control_class_index)], dtype=np.float64)
    rest_confidence = np.asarray(rest_probabilities[:, int(control_class_index)], dtype=np.float64)
    _, control_margin = confidence_margin_from_probabilities(control_probabilities)
    _, rest_margin = confidence_margin_from_probabilities(rest_probabilities)
    control_predicted_class = np.argmax(control_probabilities, axis=1)
    rest_predicted_class = np.argmax(rest_probabilities, axis=1)
    control_mask = np.logical_and.reduce(
        [
            control_predicted_class == int(control_class_index),
            control_confidence >= float(confidence_threshold),
            control_margin >= float(margin_threshold),
        ]
    )
    rest_mask = np.logical_and.reduce(
        [
            rest_predicted_class == int(control_class_index),
            rest_confidence >= float(confidence_threshold),
            rest_margin >= float(margin_threshold),
        ]
    )
    return {
        "control_detection_rate": float(np.mean(control_mask)) if control_mask.size else 0.0,
        "rest_false_activation_rate": float(np.mean(rest_mask)) if rest_mask.size else 0.0,
    }


def _threshold_accept(
    probabilities: np.ndarray,
    *,
    target_class_index: int,
    confidence_threshold: float | None,
    margin_threshold: float | None,
) -> bool:
    """Evaluate whether one probability row passes class + confidence + margin checks."""
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probabilities.size == 0:
        return False
    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[int(target_class_index)])
    sorted_probabilities = np.sort(probabilities)
    runner_up = float(sorted_probabilities[-2]) if sorted_probabilities.size > 1 else 0.0
    margin = confidence - runner_up
    if prediction != int(target_class_index):
        return False
    if confidence_threshold is not None and confidence < float(confidence_threshold):
        return False
    if margin_threshold is not None and margin < float(margin_threshold):
        return False
    return True


def evaluate_continuous_online_like(
    *,
    continuous_records: list[dict[str, object]],
    main_member_artifacts: list[dict[str, object]],
    main_fusion_weights: list[float],
    main_fusion_method: str,
    main_probability_calibration: dict[str, object] | None,
    class_names: list[str],
    sampling_rate: float,
    main_runtime: dict[str, object] | None,
    gate_member_artifacts: list[dict[str, object]] | None = None,
    gate_fusion_weights: list[float] | None = None,
    gate_fusion_method: str = DEFAULT_FUSION_METHOD,
    gate_probability_calibration: dict[str, object] | None = None,
    gate_runtime: dict[str, object] | None = None,
    artifact_member_artifacts: list[dict[str, object]] | None = None,
    artifact_fusion_weights: list[float] | None = None,
    artifact_fusion_method: str = DEFAULT_FUSION_METHOD,
    artifact_probability_calibration: dict[str, object] | None = None,
    artifact_runtime: dict[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate final trained models on continuous prompts without using those blocks for fitting."""
    records = [dict(item) for item in (continuous_records or [])]
    if not records:
        return {
            "available": False,
            "reason": "no_continuous_blocks",
            "record_count": 0,
            "prompt_count": 0,
            "evaluated_prompt_count": 0,
        }
    if not main_member_artifacts:
        return {
            "available": False,
            "reason": "missing_main_members",
            "record_count": int(len(records)),
            "prompt_count": int(sum(int(np.asarray(item.get("event_labels", [])).shape[0]) for item in records)),
            "evaluated_prompt_count": 0,
        }

    label_to_index = {str(name): int(index) for index, name in enumerate(class_names)}
    main_class_count = int(len(class_names))
    main_window_sec = float(max(float(item["window_sec"]) for item in main_member_artifacts))
    main_window_samples = int(round(main_window_sec * float(sampling_rate)))
    if main_window_samples <= 0:
        return {
            "available": False,
            "reason": "invalid_main_window",
            "record_count": int(len(records)),
            "prompt_count": int(sum(int(np.asarray(item.get("event_labels", [])).shape[0]) for item in records)),
            "evaluated_prompt_count": 0,
        }

    gate_enabled = bool(gate_member_artifacts and gate_fusion_weights)
    artifact_enabled = bool(artifact_member_artifacts and artifact_fusion_weights)
    gate_control_index = int((gate_runtime or {}).get("calibration", {}).get("control_class_index", 1))
    artifact_class_index = int((artifact_runtime or {}).get("calibration", {}).get("control_class_index", 1))

    main_confidence_threshold = (
        float(main_runtime["confidence_threshold"])
        if isinstance(main_runtime, dict) and "confidence_threshold" in main_runtime
        else None
    )
    main_margin_threshold = (
        float(main_runtime["margin_threshold"])
        if isinstance(main_runtime, dict) and "margin_threshold" in main_runtime
        else None
    )
    gate_confidence_threshold = (
        float(gate_runtime["confidence_threshold"])
        if isinstance(gate_runtime, dict) and "confidence_threshold" in gate_runtime
        else None
    )
    gate_margin_threshold = (
        float(gate_runtime["margin_threshold"])
        if isinstance(gate_runtime, dict) and "margin_threshold" in gate_runtime
        else None
    )
    artifact_confidence_threshold = (
        float(artifact_runtime["confidence_threshold"])
        if isinstance(artifact_runtime, dict) and "confidence_threshold" in artifact_runtime
        else None
    )
    artifact_margin_threshold = (
        float(artifact_runtime["margin_threshold"])
        if isinstance(artifact_runtime, dict) and "margin_threshold" in artifact_runtime
        else None
    )

    prompt_total = 0
    prompt_evaluated = 0
    prompt_skipped = 0
    unknown_label_prompt_count = 0
    mi_prompt_total = 0
    mi_prompt_detected = 0
    mi_prompt_correct = 0
    mi_prompt_rejected_by_main = 0
    mi_prompt_rejected_by_gate = 0
    mi_prompt_rejected_by_artifact = 0
    no_control_total = 0
    no_control_false_activation = 0
    per_label_counts: dict[str, dict[str, int]] = {
        str(name): {"total": 0, "detected": 0, "correct": 0}
        for name in class_names
    }

    for record in records:
        X_blocks = np.asarray(record.get("X", np.empty((0, 0, 0), dtype=np.float32)), dtype=np.float32)
        if X_blocks.ndim != 3 or X_blocks.shape[0] == 0:
            prompt_skipped += int(np.asarray(record.get("event_labels", [])).shape[0])
            continue

        event_labels = np.asarray(record.get("event_labels", []), dtype=object)
        event_samples = np.asarray(record.get("event_samples", []), dtype=np.int64)
        event_block_indices = np.asarray(record.get("event_block_indices", []), dtype=np.int64)
        block_starts = np.asarray(record.get("block_start_samples", []), dtype=np.int64)
        block_ends = np.asarray(record.get("block_end_samples", []), dtype=np.int64)
        block_count = int(X_blocks.shape[0])
        event_count = int(min(event_labels.shape[0], event_samples.shape[0]))

        for event_index in range(event_count):
            prompt_total += 1
            raw_label = _decode_npz_text(event_labels[event_index]).strip().lower()
            sample_index = int(event_samples[event_index])
            block_index = -1
            if event_block_indices.shape[0] > event_index:
                candidate_index = int(event_block_indices[event_index])
                if 0 <= candidate_index < block_count:
                    block_index = candidate_index
            if block_index < 0 and block_starts.shape[0] == block_count and block_ends.shape[0] == block_count:
                matches = np.flatnonzero(np.logical_and(block_starts <= sample_index, sample_index < block_ends))
                if matches.size:
                    block_index = int(matches[0])
            if block_index < 0 and block_count == 1:
                block_index = 0
            if block_index < 0:
                prompt_skipped += 1
                continue

            block_signal = np.asarray(X_blocks[block_index], dtype=np.float32)
            if block_signal.ndim != 2:
                prompt_skipped += 1
                continue

            block_start = int(block_starts[block_index]) if block_starts.shape[0] == block_count else 0
            rel_start = int(sample_index - block_start)
            rel_end = int(rel_start + main_window_samples)
            if rel_start < 0 or rel_end > int(block_signal.shape[1]):
                prompt_skipped += 1
                continue
            raw_window = np.ascontiguousarray(block_signal[:, rel_start:rel_end], dtype=np.float32)[np.newaxis, ...]

            main_probs, _ = evaluate_bank_on_raw_windows(
                raw_window,
                main_member_artifacts,
                sampling_rate=sampling_rate,
                fusion_weights=main_fusion_weights,
                fusion_method=main_fusion_method,
                class_count=main_class_count,
                probability_calibration=main_probability_calibration,
            )
            main_row = np.asarray(main_probs[0], dtype=np.float64)
            main_prediction = int(np.argmax(main_row))
            main_accepted = _threshold_accept(
                main_row,
                target_class_index=main_prediction,
                confidence_threshold=main_confidence_threshold,
                margin_threshold=main_margin_threshold,
            )

            artifact_rejected = False
            if artifact_enabled:
                artifact_probs, _ = evaluate_bank_on_raw_windows(
                    raw_window,
                    list(artifact_member_artifacts or []),
                    sampling_rate=sampling_rate,
                    fusion_weights=list(artifact_fusion_weights or []),
                    fusion_method=artifact_fusion_method,
                    class_count=len(ARTIFACT_REJECTOR_CLASS_NAMES),
                    probability_calibration=artifact_probability_calibration,
                )
                artifact_row = np.asarray(artifact_probs[0], dtype=np.float64)
                artifact_rejected = _threshold_accept(
                    artifact_row,
                    target_class_index=artifact_class_index,
                    confidence_threshold=artifact_confidence_threshold,
                    margin_threshold=artifact_margin_threshold,
                )

            gate_accept = True
            if gate_enabled and not artifact_rejected:
                gate_probs, _ = evaluate_bank_on_raw_windows(
                    raw_window,
                    list(gate_member_artifacts or []),
                    sampling_rate=sampling_rate,
                    fusion_weights=list(gate_fusion_weights or []),
                    fusion_method=gate_fusion_method,
                    class_count=len(GATE_CLASS_NAMES),
                    probability_calibration=gate_probability_calibration,
                )
                gate_row = np.asarray(gate_probs[0], dtype=np.float64)
                gate_accept = _threshold_accept(
                    gate_row,
                    target_class_index=gate_control_index,
                    confidence_threshold=gate_confidence_threshold,
                    margin_threshold=gate_margin_threshold,
                )

            final_active = bool(main_accepted and gate_accept and (not artifact_rejected))
            prompt_evaluated += 1

            if raw_label == CONTINUOUS_NO_CONTROL_LABEL:
                no_control_total += 1
                if final_active:
                    no_control_false_activation += 1
                continue

            expected_index = label_to_index.get(raw_label)
            if expected_index is None:
                unknown_label_prompt_count += 1
                continue

            mi_prompt_total += 1
            per_label_counts[raw_label]["total"] += 1
            if not main_accepted:
                mi_prompt_rejected_by_main += 1
            elif artifact_rejected:
                mi_prompt_rejected_by_artifact += 1
            elif not gate_accept:
                mi_prompt_rejected_by_gate += 1

            if final_active:
                mi_prompt_detected += 1
                per_label_counts[raw_label]["detected"] += 1
                if int(main_prediction) == int(expected_index):
                    mi_prompt_correct += 1
                    per_label_counts[raw_label]["correct"] += 1

    return {
        "available": bool(prompt_evaluated > 0),
        "record_count": int(len(records)),
        "prompt_count": int(prompt_total),
        "evaluated_prompt_count": int(prompt_evaluated),
        "skipped_prompt_count": int(prompt_skipped),
        "unknown_label_prompt_count": int(unknown_label_prompt_count),
        "mi_prompt_total": int(mi_prompt_total),
        "mi_prompt_detected": int(mi_prompt_detected),
        "mi_prompt_correct": int(mi_prompt_correct),
        "mi_prompt_accuracy": float(mi_prompt_correct / mi_prompt_total) if mi_prompt_total else 0.0,
        "mi_detected_accuracy": float(mi_prompt_correct / mi_prompt_detected) if mi_prompt_detected else 0.0,
        "mi_activation_rate": float(mi_prompt_detected / mi_prompt_total) if mi_prompt_total else 0.0,
        "mi_rejected_by_main_threshold": int(mi_prompt_rejected_by_main),
        "mi_rejected_by_gate": int(mi_prompt_rejected_by_gate),
        "mi_rejected_by_artifact_rejector": int(mi_prompt_rejected_by_artifact),
        "no_control_prompt_total": int(no_control_total),
        "no_control_false_activation_count": int(no_control_false_activation),
        "no_control_false_activation_rate": (
            float(no_control_false_activation / no_control_total) if no_control_total else 0.0
        ),
        "per_label": {
            label: {
                "total": int(stats["total"]),
                "detected": int(stats["detected"]),
                "correct": int(stats["correct"]),
                "accuracy": float(stats["correct"] / stats["total"]) if stats["total"] else 0.0,
                "detected_accuracy": (
                    float(stats["correct"] / stats["detected"]) if stats["detected"] else 0.0
                ),
                "activation_rate": float(stats["detected"] / stats["total"]) if stats["total"] else 0.0,
            }
            for label, stats in per_label_counts.items()
        },
        "main_thresholds_enabled": bool(main_confidence_threshold is not None and main_margin_threshold is not None),
        "control_gate_enabled": bool(gate_enabled),
        "artifact_rejector_enabled": bool(artifact_enabled),
        "main_window_sec": float(main_window_sec),
    }


def split_contains_all_classes(y: np.ndarray, class_count: int) -> bool:
    """Return True when a split contains every class at least once."""
    present = np.unique(np.asarray(y, dtype=np.int64))
    return present.shape[0] == int(class_count)


def _split_train_val_with_groups(
    y: np.ndarray,
    groups: np.ndarray,
    trainval_idx: np.ndarray,
    *,
    random_state: int,
    class_count: int,
    max_group_attempts: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Split train/val inside an already chosen trainval partition."""
    inner_groups = np.asarray(groups[trainval_idx], dtype=object)
    if np.unique(inner_groups).shape[0] >= 2:
        for attempt in range(int(max_group_attempts)):
            inner = GroupShuffleSplit(
                n_splits=1,
                test_size=0.25,
                random_state=int(random_state) + 1000 + attempt,
            )
            inner_train_rel, val_rel = next(inner.split(np.zeros(trainval_idx.shape[0]), y[trainval_idx], inner_groups))
            train_idx = trainval_idx[inner_train_rel]
            val_idx = trainval_idx[val_rel]
            if split_contains_all_classes(y[train_idx], class_count) and split_contains_all_classes(y[val_idx], class_count):
                return train_idx, val_idx

    if trainval_idx.shape[0] < int(class_count) * 2:
        return None
    try:
        fallback_train, fallback_val = train_test_split(
            trainval_idx,
            test_size=0.25,
            random_state=int(random_state),
            stratify=y[trainval_idx],
        )
    except ValueError:
        return None
    if split_contains_all_classes(y[fallback_train], class_count) and split_contains_all_classes(y[fallback_val], class_count):
        return fallback_train, fallback_val
    return None


def split_trials(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    session_labels: np.ndarray | None = None,
    random_state: int,
    class_count: int,
    max_group_attempts: int = 64,
    allow_trial_stratified_fallback: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Split trials into train/val/test with session->run->trial fallback priority."""
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups, dtype=object)
    unique_groups = np.unique(groups)
    session_labels = (
        np.asarray(session_labels, dtype=object)
        if session_labels is not None and np.asarray(session_labels).shape[0] == y.shape[0]
        else np.asarray([], dtype=object)
    )

    if session_labels.size:
        unique_sessions = np.unique(session_labels)
        if unique_sessions.shape[0] >= 2:
            session_sizes = {
                str(session): int(np.sum(session_labels == session))
                for session in unique_sessions.tolist()
            }
            ordered_sessions = sorted(
                unique_sessions.tolist(),
                key=lambda session: (-session_sizes[str(session)], str(session)),
            )
            for session_token in ordered_sessions:
                test_idx = np.flatnonzero(session_labels == session_token)
                if test_idx.size == 0:
                    continue
                if not split_contains_all_classes(y[test_idx], class_count):
                    continue
                trainval_idx = np.flatnonzero(session_labels != session_token)
                if trainval_idx.size == 0:
                    continue
                maybe_split = _split_train_val_with_groups(
                    y,
                    groups,
                    trainval_idx,
                    random_state=int(random_state),
                    class_count=int(class_count),
                    max_group_attempts=int(max_group_attempts),
                )
                if maybe_split is None:
                    continue
                train_idx, val_idx = maybe_split
                if split_contains_all_classes(y[train_idx], class_count) and split_contains_all_classes(y[val_idx], class_count):
                    return train_idx, val_idx, test_idx, "session_holdout"

    if unique_groups.shape[0] >= 3:
        for attempt in range(int(max_group_attempts)):
            outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=int(random_state) + attempt)
            trainval_idx, test_idx = next(outer.split(np.zeros_like(y), y, groups))
            if not split_contains_all_classes(y[test_idx], class_count):
                continue

            maybe_split = _split_train_val_with_groups(
                y,
                groups,
                trainval_idx,
                random_state=int(random_state) + attempt,
                class_count=int(class_count),
                max_group_attempts=int(max_group_attempts),
            )
            if maybe_split is None:
                continue
            train_idx, val_idx = maybe_split
            if split_contains_all_classes(y[train_idx], class_count) and split_contains_all_classes(y[val_idx], class_count):
                return train_idx, val_idx, test_idx, "group_shuffle"

    if not allow_trial_stratified_fallback:
        raise RuntimeError(
            "Unable to build a group/session split with full class coverage. "
            "Collect more runs/sessions or pass --allow-trial-level-fallback."
        )

    trainval_idx, test_idx = train_test_split(
        np.arange(y.shape[0]),
        test_size=0.2,
        random_state=int(random_state),
        stratify=y,
    )
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=0.25,
        random_state=int(random_state),
        stratify=y[trainval_idx],
    )
    return train_idx, val_idx, test_idx, "trial_stratified_fallback"


def split_by_group_sets(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    train_groups: set[str],
    val_groups: set[str],
    test_groups: set[str],
    class_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Build split indices by explicit group sets, validating class coverage."""
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups, dtype=object)
    train_idx = np.flatnonzero([str(group) in train_groups for group in groups.tolist()])
    val_idx = np.flatnonzero([str(group) in val_groups for group in groups.tolist()])
    test_idx = np.flatnonzero([str(group) in test_groups for group in groups.tolist()])
    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        return None
    if not split_contains_all_classes(y[train_idx], class_count):
        return None
    if not split_contains_all_classes(y[val_idx], class_count):
        return None
    if not split_contains_all_classes(y[test_idx], class_count):
        return None
    return train_idx, val_idx, test_idx


def _sorted_tokens(values: np.ndarray) -> list[str]:
    """Convert token-like arrays to sorted unique string tokens."""
    if np.asarray(values).size == 0:
        return []
    return sorted(set(str(item) for item in np.asarray(values, dtype=object).tolist()))


def summarize_split_assignments(
    *,
    groups: np.ndarray,
    sessions: np.ndarray | None,
    split_indices: dict[str, np.ndarray],
) -> dict[str, dict[str, object]]:
    """Summarize sample/group/session membership for each split partition."""
    result: dict[str, dict[str, object]] = {}
    groups = np.asarray(groups, dtype=object)
    sessions_array = (
        np.asarray(sessions, dtype=object)
        if sessions is not None and np.asarray(sessions).shape[0] == groups.shape[0]
        else np.asarray([], dtype=object)
    )
    for split_name, indices in split_indices.items():
        split_idx = np.asarray(indices, dtype=np.int64)
        split_groups = np.asarray(groups[split_idx], dtype=object) if split_idx.size else np.asarray([], dtype=object)
        split_sessions = (
            np.asarray(sessions_array[split_idx], dtype=object)
            if split_idx.size and sessions_array.shape[0] == groups.shape[0]
            else np.asarray([], dtype=object)
        )
        group_tokens = _sorted_tokens(split_groups)
        session_tokens = _sorted_tokens(split_sessions)
        result[str(split_name)] = {
            "sample_count": int(split_idx.size),
            "group_count": int(len(group_tokens)),
            "session_count": int(len(session_tokens)),
            "groups": group_tokens,
            "sessions": session_tokens,
        }
    return result


def make_member_output_path(output_model_path: Path, window_sec: float) -> Path:
    """Build the per-window model path next to the bank artifact."""
    return output_model_path.with_name(f"{output_model_path.stem}_{window_token(window_sec)}{output_model_path.suffix}")


def make_member_report_path(report_path: Path, window_sec: float) -> Path:
    """Build the per-window report path next to the aggregate report."""
    return report_path.with_name(f"{report_path.stem}_{window_token(window_sec)}{report_path.suffix}")


def build_candidates(
    sampling_rate: float,
    *,
    channel_names: list[str] | None = None,
    candidate_names: list[str] | None = None,
    torch_epochs: int = DEFAULT_TORCH_EPOCHS,
    torch_batch_size: int = DEFAULT_TORCH_BATCH_SIZE,
    torch_learning_rate: float = DEFAULT_TORCH_LEARNING_RATE,
    torch_weight_decay: float = DEFAULT_TORCH_WEIGHT_DECAY,
    torch_patience: int = DEFAULT_TORCH_PATIENCE,
    torch_validation_split: float = DEFAULT_TORCH_VALIDATION_SPLIT,
    torch_device: str | None = None,
    deep_stage_pretrain_window_secs: list[float] | None = None,
    deep_stage_finetune_window_secs: list[float] | None = None,
    central_prior_alpha: float = DEFAULT_CENTRAL_PRIOR_ALPHA,
    central_aux_loss_weight: float = DEFAULT_CENTRAL_AUX_LOSS_WEIGHT,
) -> dict[str, object]:
    """Build the optimized candidate pipelines."""
    resolved_candidate_names = [str(item).strip() for item in (candidate_names or default_candidate_names()) if str(item).strip()]
    deep_aliases = {
        "eegnet",
        "shallow",
        "deep",
        "fblight",
        "fblight_tcn",
        "central_only_fblight",
        "central_gate_fblight",
        "full8_fblight",
        "central_prior_dual_branch_fblight_tcn",
        "central_prior_gate_fblight",
    }

    def _contains_deep_model(candidate_name: str) -> bool:
        normalized = str(candidate_name).strip().lower()
        if normalized in deep_aliases:
            return True
        for prefix in ("central-", "latefusion-"):
            if normalized.startswith(prefix):
                return normalized[len(prefix) :] in deep_aliases
        return False

    requested_deep_models = [name for name in resolved_candidate_names if _contains_deep_model(name)]
    if requested_deep_models and not TORCH_AVAILABLE:
        raise ModuleNotFoundError(
            "PyTorch is not installed, so deep candidates are unavailable. "
            f"Install torch to use: {sorted(set(requested_deep_models))}"
        )
    return build_optimized_candidates(
        candidate_names=resolved_candidate_names,
        fs=float(sampling_rate),
        bands=DEFAULT_FBCSP_BANDS,
        n_components=4,
        riemann_band=DEFAULT_PREPROCESS_BANDPASS,
        estimator="lwf",
        metric="riemann",
        kernel="rbf",
        C=1.0,
        torch_epochs=int(torch_epochs),
        torch_batch_size=int(torch_batch_size),
        torch_learning_rate=float(torch_learning_rate),
        torch_weight_decay=float(torch_weight_decay),
        torch_patience=int(torch_patience),
        torch_validation_split=float(torch_validation_split),
        torch_device=torch_device,
        channel_names=channel_names,
        deep_stage_pretrain_window_secs=(
            [float(item) for item in (deep_stage_pretrain_window_secs or DEFAULT_DEEP_STAGE_PRETRAIN_WINDOW_SECS)]
        ),
        deep_stage_finetune_window_secs=(
            [float(item) for item in (deep_stage_finetune_window_secs or DEFAULT_DEEP_STAGE_FINETUNE_WINDOW_SECS)]
        ),
        central_prior_alpha=float(central_prior_alpha),
        central_aux_loss_weight=float(central_aux_loss_weight),
    )


def train_custom_model(
    *,
    dataset_root: Path,
    subject_filter: str | None,
    output_model_path: Path,
    report_path: Path,
    random_state: int = 42,
    min_class_trials: int = 5,
    recommended_total_class_trials: int = DEFAULT_RECOMMENDED_TOTAL_CLASS_TRIALS,
    recommended_run_class_trials: int = DEFAULT_RECOMMENDED_RUN_CLASS_TRIALS,
    enforce_readiness: bool = False,
    allow_trial_level_fallback: bool = False,
    window_secs: list[float] | None = None,
    window_offset_secs: list[float] | None = None,
    fusion_method: str = DEFAULT_FUSION_METHOD,
    fusion_weights: list[float] | None = None,
    candidate_names: list[str] | None = None,
    gate_candidate_names: list[str] | None = None,
    artifact_candidate_names: list[str] | None = None,
    torch_epochs: int = DEFAULT_TORCH_EPOCHS,
    torch_batch_size: int = DEFAULT_TORCH_BATCH_SIZE,
    torch_learning_rate: float = DEFAULT_TORCH_LEARNING_RATE,
    torch_weight_decay: float = DEFAULT_TORCH_WEIGHT_DECAY,
    torch_patience: int = DEFAULT_TORCH_PATIENCE,
    torch_validation_split: float = DEFAULT_TORCH_VALIDATION_SPLIT,
    torch_device: str | None = None,
    deep_stage_pretrain_window_secs: list[float] | None = None,
    deep_stage_finetune_window_secs: list[float] | None = None,
    central_prior_alpha: float = DEFAULT_CENTRAL_PRIOR_ALPHA,
    central_aux_loss_weight: float = DEFAULT_CENTRAL_AUX_LOSS_WEIGHT,
) -> dict[str, object]:
    """Train multi-window optimized classifiers and export a realtime bank artifact."""
    subject_filter = normalize_subject_filter(subject_filter)
    loaded = load_custom_task_datasets(dataset_root, subject_filter)
    mi_loaded = dict(loaded["mi"])
    gate_loaded = dict(loaded["gate"])
    artifact_loaded = dict(loaded["artifact"])
    continuous_loaded = dict(loaded["continuous"])

    X_raw = np.asarray(mi_loaded["X"], dtype=np.float32)
    y = np.asarray(mi_loaded["y"], dtype=np.int64)
    groups = np.asarray(mi_loaded["groups"], dtype=object)
    session_labels = np.asarray(mi_loaded.get("session_labels", []), dtype=object)
    trial_keys = np.asarray(mi_loaded.get("trial_keys", []), dtype=object)
    class_names = list(loaded["class_names"])
    sampling_rate = float(loaded["sampling_rate"])
    full_window_sec = float(mi_loaded["full_window_sec"])
    requested_window_secs = [float(item) for item in (window_secs or DEFAULT_WINDOW_SECS)]
    requested_offset_secs = [float(item) for item in (window_offset_secs or default_window_offsets())]
    aligned_fusion_weights = None if fusion_weights is None else [float(item) for item in fusion_weights]
    if aligned_fusion_weights is not None:
        if len(aligned_fusion_weights) != len(requested_window_secs):
            raise ValueError(
                f"fusion_weights length mismatch: expected {len(requested_window_secs)}, got {len(aligned_fusion_weights)}."
            )
        paired = sorted(zip(requested_window_secs, aligned_fusion_weights), key=lambda item: item[0])
        selected_window_secs = [float(window_sec) for window_sec, _ in paired]
        aligned_fusion_weights = [float(weight) for _, weight in paired]
    else:
        selected_window_secs = sorted(requested_window_secs)
    selected_offset_secs = sorted(set(requested_offset_secs))
    selected_candidate_names = [str(item).strip() for item in (candidate_names or default_candidate_names()) if str(item).strip()]
    selected_gate_candidate_names = default_gate_candidate_names(gate_candidate_names)
    selected_artifact_candidate_names = default_artifact_candidate_names(artifact_candidate_names)

    if not selected_window_secs:
        raise ValueError("window_secs cannot be empty.")
    if not selected_offset_secs:
        raise ValueError("window_offset_secs cannot be empty.")
    if not selected_candidate_names:
        raise ValueError("candidate_names cannot be empty.")
    if not selected_gate_candidate_names:
        raise ValueError("gate_candidate_names cannot be empty.")
    if not selected_artifact_candidate_names:
        raise ValueError("artifact_candidate_names cannot be empty.")
    if any(window_sec <= 0.0 for window_sec in selected_window_secs):
        raise ValueError(f"window_secs must all be positive, got {selected_window_secs}.")
    if any(offset_sec < 0.0 for offset_sec in selected_offset_secs):
        raise ValueError(f"window_offset_secs cannot contain negative values, got {selected_offset_secs}.")
    if any(
        (window_sec + offset_sec) > (full_window_sec + 1e-6)
        for window_sec in selected_window_secs
        for offset_sec in selected_offset_secs
    ):
        raise ValueError(
            f"Requested windows {selected_window_secs} with offsets {selected_offset_secs} exceed the "
            f"available {full_window_sec:.3f}s imagery epoch."
        )

    label_distribution = validate_dataset_distribution(
        y,
        class_names,
        min_class_trials=int(min_class_trials),
    )
    dataset_readiness = evaluate_dataset_readiness(
        label_distribution=label_distribution,
        source_records=list(loaded.get("source_records", [])),
        class_names=class_names,
        recommended_total_class_trials=int(recommended_total_class_trials),
        recommended_run_class_trials=int(recommended_run_class_trials),
    )
    if enforce_readiness and not bool(dataset_readiness.get("ready_for_stable_comparison", False)):
        warnings = list(dataset_readiness.get("warnings", []))
        raise RuntimeError(
            "Dataset readiness check failed (--enforce-readiness). "
            f"Warnings: {warnings}"
        )

    train_idx, val_idx, test_idx, split_strategy = split_trials(
        y,
        groups,
        session_labels=session_labels,
        random_state=int(random_state),
        class_count=len(class_names),
        allow_trial_stratified_fallback=bool(allow_trial_level_fallback),
    )
    print(f"split_strategy={split_strategy}")
    if split_strategy not in STRICT_MAIN_SPLIT_STRATEGIES:
        print(
            "warning=insufficient distinct runs for group-based split; "
            "evaluation may be optimistic because fallback trial-level stratification was used"
        )

    split_indices = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "trainval": np.concatenate([train_idx, val_idx], axis=0),
    }
    split_labels = {
        "train": y[train_idx],
        "val": y[val_idx],
        "test": y[test_idx],
        "trainval": y[split_indices["trainval"]],
    }

    train_group_set = set(str(item) for item in groups[train_idx].tolist())
    val_group_set = set(str(item) for item in groups[val_idx].tolist())
    test_group_set = set(str(item) for item in groups[test_idx].tolist())
    trainval_group_set = set(str(item) for item in groups[split_indices["trainval"]].tolist())

    gate_required_samples = required_window_samples(
        max(selected_window_secs),
        sampling_rate,
        offset_sec=max(selected_offset_secs),
    )
    gate_branch_loaded = build_gate_branch_dataset(
        list(gate_loaded.get("records", [])),
        required_samples=gate_required_samples,
        channel_count=X_raw.shape[1],
    )
    X_gate_pos_all = np.asarray(
        gate_branch_loaded.get("X_pos", np.empty((0, X_raw.shape[1], gate_required_samples), dtype=np.float32)),
        dtype=np.float32,
    )
    X_gate_neg_all = np.asarray(
        gate_branch_loaded.get("X_neg", np.empty((0, X_raw.shape[1], gate_required_samples), dtype=np.float32)),
        dtype=np.float32,
    )
    X_gate_hard_neg_all = np.asarray(
        gate_branch_loaded.get("X_hard_neg", np.empty((0, X_raw.shape[1], gate_required_samples), dtype=np.float32)),
        dtype=np.float32,
    )
    gate_pos_groups_all = np.asarray(gate_branch_loaded.get("pos_groups", []), dtype=object)
    gate_neg_groups_all = np.asarray(gate_branch_loaded.get("neg_groups", []), dtype=object)
    gate_hard_neg_groups_all = np.asarray(gate_branch_loaded.get("hard_neg_groups", []), dtype=object)
    gate_pos_sessions_all = np.asarray(gate_branch_loaded.get("pos_session_labels", []), dtype=object)
    gate_neg_sessions_all = np.asarray(gate_branch_loaded.get("neg_session_labels", []), dtype=object)
    gate_hard_neg_sessions_all = np.asarray(gate_branch_loaded.get("hard_neg_session_labels", []), dtype=object)
    gate_neg_sources_all = np.asarray(gate_branch_loaded.get("neg_sources", []), dtype=object)
    gate_hard_neg_sources_all = np.asarray(gate_branch_loaded.get("hard_neg_sources", []), dtype=object)
    gate_target_samples = int(gate_branch_loaded.get("required_samples", gate_required_samples))

    gate_available = bool(
        gate_target_samples > 0
        and X_gate_pos_all.shape[0] > 0
        and (X_gate_neg_all.shape[0] + X_gate_hard_neg_all.shape[0]) > 0
    )
    gate_split_strategy = "unavailable"
    gate_split_block_reason: str | None = None
    gate_X_all = np.empty((0, X_raw.shape[1], 0), dtype=np.float32)
    gate_y_all = np.empty((0,), dtype=np.int64)
    gate_groups_all = np.asarray([], dtype=object)
    gate_sessions_all = np.asarray([], dtype=object)
    gate_sources_all = np.asarray([], dtype=object)
    gate_split_indices = {
        "train": np.asarray([], dtype=np.int64),
        "val": np.asarray([], dtype=np.int64),
        "test": np.asarray([], dtype=np.int64),
        "trainval": np.asarray([], dtype=np.int64),
    }
    if not gate_available and bool(gate_branch_loaded.get("raw_present", False)):
        gate_split_strategy = "insufficient_epoch_length"
        gate_split_block_reason = (
            "Gate records were found, but too few segments survived the runtime-window requirement after per-run "
            "filtering. "
            f"required={(gate_required_samples / float(sampling_rate)):.3f}s, "
            f"max_control={(int(gate_branch_loaded.get('pos_max_samples', 0)) / float(sampling_rate)):.3f}s, "
            f"max_clean={(int(gate_branch_loaded.get('neg_max_samples', 0)) / float(sampling_rate)):.3f}s, "
            f"max_hard={(int(gate_branch_loaded.get('hard_neg_max_samples', 0)) / float(sampling_rate)):.3f}s."
        )
    if gate_available:
        gate_X_all = np.concatenate([X_gate_pos_all, X_gate_neg_all, X_gate_hard_neg_all], axis=0).astype(np.float32)
        gate_y_all = np.concatenate(
            [
                np.ones(X_gate_pos_all.shape[0], dtype=np.int64),
                np.zeros(X_gate_neg_all.shape[0], dtype=np.int64),
                np.zeros(X_gate_hard_neg_all.shape[0], dtype=np.int64),
            ],
            axis=0,
        )
        gate_groups_all = np.concatenate([gate_pos_groups_all, gate_neg_groups_all, gate_hard_neg_groups_all], axis=0)
        gate_sessions_all = np.concatenate(
            [gate_pos_sessions_all, gate_neg_sessions_all, gate_hard_neg_sessions_all],
            axis=0,
        )
        gate_sources_all = np.concatenate(
            [
                np.asarray(["imagery"] * X_gate_pos_all.shape[0], dtype=object),
                np.asarray(
                    gate_neg_sources_all.tolist()
                    if gate_neg_sources_all.shape[0] == X_gate_neg_all.shape[0]
                    else ["rest"] * X_gate_neg_all.shape[0],
                    dtype=object,
                ),
                np.asarray(
                    gate_hard_neg_sources_all.tolist()
                    if gate_hard_neg_sources_all.shape[0] == X_gate_hard_neg_all.shape[0]
                    else ["hard_negative"] * X_gate_hard_neg_all.shape[0],
                    dtype=object,
                ),
            ],
            axis=0,
        )

        aligned_gate_split = split_by_group_sets(
            gate_y_all,
            gate_groups_all,
            train_groups=train_group_set,
            val_groups=val_group_set,
            test_groups=test_group_set,
            class_count=2,
        )
        if aligned_gate_split is not None:
            gate_train_idx, gate_val_idx, gate_test_idx = aligned_gate_split
            gate_split_strategy = "aligned_to_main_split"
        else:
            gate_train_idx, gate_val_idx, gate_test_idx, gate_split_strategy = split_trials(
                gate_y_all,
                gate_groups_all,
                session_labels=gate_sessions_all,
                random_state=int(random_state) + 77,
                class_count=2,
                allow_trial_stratified_fallback=bool(allow_trial_level_fallback),
            )
        gate_split_indices = {
            "train": gate_train_idx,
            "val": gate_val_idx,
            "test": gate_test_idx,
            "trainval": np.concatenate([gate_train_idx, gate_val_idx], axis=0),
        }
        if gate_split_strategy not in STRICT_AUX_SPLIT_STRATEGIES:
            message = (
                "Gate split resolved to trial-level stratification fallback; "
                "collect more rest/control runs for leakage-safe gate evaluation."
            )
            if allow_trial_level_fallback:
                print(f"warning={message}")
            else:
                gate_available = False
                gate_split_strategy = "strict_split_unavailable"
                gate_split_block_reason = message
                gate_split_indices = {
                    "train": np.asarray([], dtype=np.int64),
                    "val": np.asarray([], dtype=np.int64),
                    "test": np.asarray([], dtype=np.int64),
                    "trainval": np.asarray([], dtype=np.int64),
                }

    artifact_required_samples = required_window_samples(max(selected_window_secs), sampling_rate, offset_sec=0.0)
    artifact_branch_loaded = build_artifact_branch_dataset(
        list(artifact_loaded.get("records", [])),
        list(gate_loaded.get("records", [])),
        required_samples=artifact_required_samples,
        channel_count=X_raw.shape[1],
    )
    X_artifact_all = np.asarray(
        artifact_branch_loaded.get(
            "X_artifact",
            np.empty((0, X_raw.shape[1], artifact_required_samples), dtype=np.float32),
        ),
        dtype=np.float32,
    )
    X_clean_negative_all = np.asarray(
        artifact_branch_loaded.get(
            "X_clean_negative",
            np.empty((0, X_raw.shape[1], artifact_required_samples), dtype=np.float32),
        ),
        dtype=np.float32,
    )
    artifact_groups_pos = np.asarray(artifact_branch_loaded.get("artifact_groups", []), dtype=object)
    artifact_groups_neg = np.asarray(artifact_branch_loaded.get("clean_negative_groups", []), dtype=object)
    artifact_sessions_pos = np.asarray(artifact_branch_loaded.get("artifact_session_labels", []), dtype=object)
    artifact_sessions_neg = np.asarray(artifact_branch_loaded.get("clean_negative_session_labels", []), dtype=object)
    artifact_target_samples = int(artifact_branch_loaded.get("required_samples", artifact_required_samples))
    artifact_rejector_available = bool(
        X_artifact_all.shape[0] > 0
        and X_clean_negative_all.shape[0] > 0
    )
    artifact_split_strategy = "unavailable"
    artifact_split_block_reason: str | None = None
    artifact_binary_X_all = np.empty((0, X_raw.shape[1], 0), dtype=np.float32)
    artifact_binary_y_all = np.empty((0,), dtype=np.int64)
    artifact_binary_groups_all = np.asarray([], dtype=object)
    artifact_binary_sessions_all = np.asarray([], dtype=object)
    artifact_split_indices = {
        "train": np.asarray([], dtype=np.int64),
        "val": np.asarray([], dtype=np.int64),
        "test": np.asarray([], dtype=np.int64),
        "trainval": np.asarray([], dtype=np.int64),
    }
    if not artifact_rejector_available and bool(artifact_branch_loaded.get("raw_present", False)):
        artifact_split_strategy = "insufficient_epoch_length"
        artifact_split_block_reason = (
            "Artifact records were found, but too few clean/artifact segments survived the runtime-window requirement "
            "after per-run filtering. "
            f"required={(artifact_required_samples / float(sampling_rate)):.3f}s, "
            f"max_artifact={(int(artifact_branch_loaded.get('artifact_max_samples', 0)) / float(sampling_rate)):.3f}s, "
            f"max_clean={(int(artifact_branch_loaded.get('clean_negative_max_samples', 0)) / float(sampling_rate)):.3f}s."
        )
    if artifact_rejector_available:
        artifact_binary_X_all = np.concatenate([X_artifact_all, X_clean_negative_all], axis=0).astype(np.float32)
        artifact_binary_y_all = np.concatenate(
            [
                np.ones(X_artifact_all.shape[0], dtype=np.int64),
                np.zeros(X_clean_negative_all.shape[0], dtype=np.int64),
            ],
            axis=0,
        )
        artifact_binary_groups_all = np.concatenate([artifact_groups_pos, artifact_groups_neg], axis=0)
        artifact_binary_sessions_all = np.concatenate([artifact_sessions_pos, artifact_sessions_neg], axis=0)

        aligned_artifact_split = split_by_group_sets(
            artifact_binary_y_all,
            artifact_binary_groups_all,
            train_groups=train_group_set,
            val_groups=val_group_set,
            test_groups=test_group_set,
            class_count=2,
        )
        if aligned_artifact_split is not None:
            artifact_train_idx, artifact_val_idx, artifact_test_idx = aligned_artifact_split
            artifact_split_strategy = "aligned_to_main_split"
        else:
            artifact_train_idx, artifact_val_idx, artifact_test_idx, artifact_split_strategy = split_trials(
                artifact_binary_y_all,
                artifact_binary_groups_all,
                session_labels=artifact_binary_sessions_all,
                random_state=int(random_state) + 133,
                class_count=2,
                allow_trial_stratified_fallback=bool(allow_trial_level_fallback),
            )
        artifact_split_indices = {
            "train": artifact_train_idx,
            "val": artifact_val_idx,
            "test": artifact_test_idx,
            "trainval": np.concatenate([artifact_train_idx, artifact_val_idx], axis=0),
        }
        if artifact_split_strategy not in STRICT_AUX_SPLIT_STRATEGIES:
            message = (
                "Artifact split resolved to trial-level stratification fallback; "
                "collect more artifact/clean runs for leakage-safe artifact evaluation."
            )
            if allow_trial_level_fallback:
                print(f"warning={message}")
            else:
                artifact_rejector_available = False
                artifact_split_strategy = "strict_split_unavailable"
                artifact_split_block_reason = message
                artifact_split_indices = {
                    "train": np.asarray([], dtype=np.int64),
                    "val": np.asarray([], dtype=np.int64),
                    "test": np.asarray([], dtype=np.int64),
                    "trainval": np.asarray([], dtype=np.int64),
                }

    mi_split_assignments = summarize_split_assignments(
        groups=groups,
        sessions=session_labels,
        split_indices=split_indices,
    )
    gate_split_assignments = summarize_split_assignments(
        groups=gate_groups_all,
        sessions=gate_sessions_all,
        split_indices=gate_split_indices,
    )
    artifact_split_assignments = summarize_split_assignments(
        groups=artifact_binary_groups_all,
        sessions=artifact_binary_sessions_all,
        split_indices=artifact_split_indices,
    )

    train_session_set = set(str(item) for item in mi_split_assignments["train"]["sessions"])
    val_session_set = set(str(item) for item in mi_split_assignments["val"]["sessions"])
    test_session_set = set(str(item) for item in mi_split_assignments["test"]["sessions"])
    continuous_records_by_split = split_continuous_records_by_group_sets(
        list(continuous_loaded.get("records", [])),
        train_groups=train_group_set,
        val_groups=val_group_set,
        test_groups=test_group_set,
        train_sessions=train_session_set,
        val_sessions=val_session_set,
        test_sessions=test_session_set,
    )

    rest_segments_all = [np.asarray(item, dtype=np.float32) for item in X_gate_neg_all] if X_gate_neg_all.ndim == 3 else []
    if gate_neg_sources_all.shape[0] == len(rest_segments_all):
        rest_source_phases_all = np.asarray(gate_neg_sources_all, dtype=object)
    else:
        rest_source_phases_all = np.asarray(["gate_neg"] * len(rest_segments_all), dtype=object)
    if gate_neg_groups_all.size:
        rest_train_mask = np.asarray([str(item) in train_group_set for item in gate_neg_groups_all.tolist()], dtype=bool)
        rest_val_mask = np.asarray([str(item) in val_group_set for item in gate_neg_groups_all.tolist()], dtype=bool)
        rest_test_mask = np.asarray([str(item) in test_group_set for item in gate_neg_groups_all.tolist()], dtype=bool)
        rest_trainval_mask = np.asarray([str(item) in trainval_group_set for item in gate_neg_groups_all.tolist()], dtype=bool)
    else:
        rest_train_mask = np.zeros((0,), dtype=bool)
        rest_val_mask = np.zeros((0,), dtype=bool)
        rest_test_mask = np.zeros((0,), dtype=bool)
        rest_trainval_mask = np.zeros((0,), dtype=bool)

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    member_summaries: list[dict[str, object]] = []
    member_artifacts: list[dict[str, object]] = []
    val_probability_vectors: list[np.ndarray] = []
    test_probability_vectors: list[np.ndarray] = []
    gate_member_summaries: list[dict[str, object]] = []
    gate_member_artifacts: list[dict[str, object]] = []
    gate_val_probability_vectors: list[np.ndarray] = []
    gate_test_probability_vectors: list[np.ndarray] = []
    gate_val_labels_reference: np.ndarray | None = None
    gate_test_labels_reference: np.ndarray | None = None
    artifact_member_summaries: list[dict[str, object]] = []
    artifact_member_artifacts: list[dict[str, object]] = []
    artifact_val_probability_vectors: list[np.ndarray] = []
    artifact_test_probability_vectors: list[np.ndarray] = []
    artifact_val_labels_reference: np.ndarray | None = None
    artifact_test_labels_reference: np.ndarray | None = None

    for window_sec in selected_window_secs:
        processed_by_offset = {}
        for offset_sec in selected_offset_secs:
            cropped = {
                split_name: crop_trials_window(
                    X_raw[indices],
                    sampling_rate,
                    window_sec=window_sec,
                    offset_sec=offset_sec,
                )
                for split_name, indices in split_indices.items()
            }
            processed_by_offset[offset_sec] = {
                split_name: preprocess_trials(split_X, sampling_rate)
                for split_name, split_X in cropped.items()
            }

        gate_control_by_split: dict[str, np.ndarray] = {}
        rest_windows_by_split: dict[str, np.ndarray] = {}
        rest_eval_windows_by_split: dict[str, np.ndarray] = {}
        rest_window_phase_counts: dict[str, dict[str, int]] = {}
        for split_name in split_indices.keys():
            if not gate_available or gate_split_indices.get(split_name) is None or gate_split_indices[split_name].size == 0:
                gate_control_by_split[split_name] = np.empty((0, X_raw.shape[1], 0), dtype=np.float32)
                rest_windows_by_split[split_name] = np.empty((0, X_raw.shape[1], 0), dtype=np.float32)
                rest_eval_windows_by_split[split_name] = np.empty((0, X_raw.shape[1], 0), dtype=np.float32)
                rest_window_phase_counts[split_name] = {}
                continue

            gate_split_X = np.asarray(gate_X_all[gate_split_indices[split_name]], dtype=np.float32)
            gate_split_y = np.asarray(gate_y_all[gate_split_indices[split_name]], dtype=np.int64)
            gate_split_sources = (
                np.asarray(gate_sources_all[gate_split_indices[split_name]], dtype=object)
                if gate_sources_all.shape[0] == gate_X_all.shape[0]
                else np.asarray([], dtype=object)
            )
            control_raw = np.asarray(gate_split_X[gate_split_y == 1], dtype=np.float32)
            clean_raw = np.asarray(gate_split_X[gate_split_y == 0], dtype=np.float32)
            clean_sources = (
                np.asarray(gate_split_sources[gate_split_y == 0], dtype=object)
                if gate_split_sources.shape[0] == gate_split_y.shape[0]
                else np.asarray([], dtype=object)
            )

            control_windows = crop_trials_multi_offset(
                control_raw,
                sampling_rate,
                window_sec=float(window_sec),
                offset_secs=[float(item) for item in selected_offset_secs],
            ) if control_raw.shape[0] > 0 else np.empty((0, X_raw.shape[1], 0), dtype=np.float32)
            clean_windows = crop_trials_multi_offset(
                clean_raw,
                sampling_rate,
                window_sec=float(window_sec),
                offset_secs=[0.0],
            ) if clean_raw.shape[0] > 0 else np.empty((0, X_raw.shape[1], 0), dtype=np.float32)

            gate_control_by_split[split_name] = (
                preprocess_trials(control_windows, sampling_rate)
                if control_windows.shape[0] > 0
                else control_windows
            )
            processed_clean_windows = (
                preprocess_trials(clean_windows, sampling_rate)
                if clean_windows.shape[0] > 0
                else clean_windows
            )
            rest_windows_by_split[split_name] = processed_clean_windows
            rest_eval_windows_by_split[split_name] = processed_clean_windows
            if clean_sources.shape[0] > 0:
                rest_window_phase_counts[split_name] = {
                    str(source): int(np.sum(clean_sources == source))
                    for source in np.unique(clean_sources)
                }
            else:
                rest_window_phase_counts[split_name] = {
                    "gate_negative": int(processed_clean_windows.shape[0]),
                }

        artifact_binary_by_split: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for split_name in split_indices.keys():
            split_idx = artifact_split_indices.get(split_name)
            if (
                not artifact_rejector_available
                or split_idx is None
                or split_idx.size == 0
                or artifact_binary_X_all.shape[0] == 0
            ):
                artifact_binary_by_split[split_name] = (
                    np.empty((0, X_raw.shape[1], 0), dtype=np.float32),
                    np.empty((0,), dtype=np.int64),
                )
                continue
            raw_split_X = np.asarray(artifact_binary_X_all[split_idx], dtype=np.float32)
            raw_split_y = np.asarray(artifact_binary_y_all[split_idx], dtype=np.int64)
            cropped_split_X = crop_trials_window(
                raw_split_X,
                sampling_rate,
                window_sec=float(window_sec),
                offset_sec=0.0,
            )
            processed_split_X = (
                preprocess_trials(cropped_split_X, sampling_rate)
                if cropped_split_X.shape[0] > 0
                else np.empty((0, X_raw.shape[1], 0), dtype=np.float32)
            )
            artifact_binary_by_split[split_name] = (processed_split_X, raw_split_y)

        best_name = None
        best_offset_sec = None
        best_selection_metrics = None
        best_selection_val_probabilities = None
        best_probability_calibration = None
        best_guided_offset_scores = None
        candidate_scores = []
        candidate_errors = []
        candidate_factory = lambda: build_candidates(
            sampling_rate,
            channel_names=loaded["channel_names"],
            candidate_names=selected_candidate_names,
            torch_epochs=torch_epochs,
            torch_batch_size=torch_batch_size,
            torch_learning_rate=torch_learning_rate,
            torch_weight_decay=torch_weight_decay,
            torch_patience=torch_patience,
            torch_validation_split=torch_validation_split,
            torch_device=torch_device,
            deep_stage_pretrain_window_secs=deep_stage_pretrain_window_secs,
            deep_stage_finetune_window_secs=deep_stage_finetune_window_secs,
            central_prior_alpha=central_prior_alpha,
            central_aux_loss_weight=central_aux_loss_weight,
        )
        gate_candidate_factory = lambda: build_candidates(
            sampling_rate,
            channel_names=loaded["channel_names"],
            candidate_names=selected_gate_candidate_names,
            torch_epochs=torch_epochs,
            torch_batch_size=torch_batch_size,
            torch_learning_rate=torch_learning_rate,
            torch_weight_decay=torch_weight_decay,
            torch_patience=torch_patience,
            torch_validation_split=torch_validation_split,
            torch_device=torch_device,
            deep_stage_pretrain_window_secs=deep_stage_pretrain_window_secs,
            deep_stage_finetune_window_secs=deep_stage_finetune_window_secs,
            central_prior_alpha=central_prior_alpha,
            central_aux_loss_weight=central_aux_loss_weight,
        )
        artifact_candidate_factory = lambda: build_candidates(
            sampling_rate,
            channel_names=loaded["channel_names"],
            candidate_names=selected_artifact_candidate_names,
            torch_epochs=torch_epochs,
            torch_batch_size=torch_batch_size,
            torch_learning_rate=torch_learning_rate,
            torch_weight_decay=torch_weight_decay,
            torch_patience=torch_patience,
            torch_validation_split=torch_validation_split,
            torch_device=torch_device,
            deep_stage_pretrain_window_secs=deep_stage_pretrain_window_secs,
            deep_stage_finetune_window_secs=deep_stage_finetune_window_secs,
            central_prior_alpha=central_prior_alpha,
            central_aux_loss_weight=central_aux_loss_weight,
        )

        main_train_X, main_train_y = stack_multi_offset_split(
            processed_by_offset,
            "train",
            selected_offset_secs,
            split_labels["train"],
        )
        main_trainval_X, main_trainval_y = stack_multi_offset_split(
            processed_by_offset,
            "trainval",
            selected_offset_secs,
            split_labels["trainval"],
        )

        for name, pipeline in candidate_factory().items():
            try:
                pipeline.fit(main_train_X, main_train_y)
                raw_val_probabilities = predict_multi_offset_probability_matrix(
                    pipeline,
                    processed_by_offset,
                    "val",
                    selected_offset_secs,
                    len(class_names),
                )
                member_probability_calibration, val_probabilities = calibrate_probability_matrix(
                    raw_val_probabilities,
                    split_labels["val"],
                    role="main_classifier_member",
                    selection_source="main_member_selection_val",
                )
            except Exception as error:
                candidate_errors.append(
                    {
                        "name": name,
                        "offset_sec": None,
                        "training_mode": "multi_offset_augmented",
                        "error": str(error),
                    }
                )
                print(
                    f"window={window_sec:.2f}s candidate={name:<16} "
                    f"mode=multi_offset_augmented failed: {error}"
                )
                continue

            val_predictions = np.argmax(val_probabilities, axis=1)
            val_metrics = classification_metrics(split_labels["val"], val_predictions, class_names)

            guided_offset_scores = []
            guided_best_offset_sec = None
            guided_best_metrics = None
            for offset_sec in selected_offset_secs:
                raw_guided_probabilities = _predict_probability_matrix(
                    pipeline,
                    processed_by_offset[float(offset_sec)]["val"],
                    len(class_names),
                )
                guided_probabilities = apply_probability_calibration(
                    raw_guided_probabilities,
                    member_probability_calibration,
                )
                guided_predictions = np.argmax(guided_probabilities, axis=1)
                guided_metrics = classification_metrics(split_labels["val"], guided_predictions, class_names)
                guided_offset_scores.append(
                    {
                        "offset_sec": float(offset_sec),
                        "val_acc": float(guided_metrics["acc"]),
                        "val_kappa": float(guided_metrics["kappa"]),
                        "val_macro_acc": float(guided_metrics["macro_acc"]),
                    }
                )
                if guided_best_metrics is None or metric_sort_key(guided_metrics) > metric_sort_key(guided_best_metrics):
                    guided_best_metrics = guided_metrics
                    guided_best_offset_sec = float(offset_sec)

            if guided_best_offset_sec is None:
                guided_best_offset_sec = float(selected_offset_secs[0])

            candidate_scores.append(
                {
                    "name": name,
                    "training_mode": "multi_offset_augmented",
                    "guided_offset_sec": float(guided_best_offset_sec),
                    "offset_count": int(len(selected_offset_secs)),
                    "val_acc": float(val_metrics["acc"]),
                    "val_kappa": float(val_metrics["kappa"]),
                    "val_macro_acc": float(val_metrics["macro_acc"]),
                    "calibration_temperature": float(member_probability_calibration["temperature"]),
                    "guided_offset_scores": guided_offset_scores,
                }
            )
            print(
                f"window={window_sec:.2f}s candidate={name:<16} mode=multi_offset_augmented "
                f"val_kappa={float(val_metrics['kappa']):.4f} val_acc={float(val_metrics['acc']):.4f} "
                f"guided_offset={float(guided_best_offset_sec):.2f}s"
            )
            if best_selection_metrics is None or metric_sort_key(val_metrics) > metric_sort_key(best_selection_metrics):
                best_name = name
                best_offset_sec = float(guided_best_offset_sec)
                best_selection_metrics = val_metrics
                best_selection_val_probabilities = val_probabilities
                best_probability_calibration = dict(member_probability_calibration)
                best_guided_offset_scores = guided_offset_scores

        if (
            best_name is None
            or best_offset_sec is None
            or best_selection_metrics is None
            or best_probability_calibration is None
        ):
            raise RuntimeError(
                f"No candidate trained successfully for {window_sec:.2f}s. Errors: {candidate_errors}"
            )

        final_pipeline = candidate_factory()[best_name]
        final_pipeline.fit(main_trainval_X, main_trainval_y)
        test_probabilities = predict_multi_offset_probability_matrix(
            final_pipeline,
            processed_by_offset,
            "test",
            selected_offset_secs,
            len(class_names),
            probability_calibration=best_probability_calibration,
        )
        test_predictions = np.argmax(test_probabilities, axis=1)
        test_metrics = classification_metrics(split_labels["test"], test_predictions, class_names)

        if best_selection_val_probabilities is None:
            raise RuntimeError(f"Missing validation probabilities for {window_sec:.2f}s.")
        val_probability_vectors.append(best_selection_val_probabilities)
        test_probability_vectors.append(test_probabilities)

        member_output_path = make_member_output_path(output_model_path, window_sec)
        member_report_path = make_member_report_path(report_path, window_sec)
        member_output_path.parent.mkdir(parents=True, exist_ok=True)
        member_report_path.parent.mkdir(parents=True, exist_ok=True)

        member_preprocessing = build_preprocessing_config(
            window_sec=float(window_sec),
            window_offset_sec=float(best_offset_sec),
            window_offset_secs_used=[float(item) for item in selected_offset_secs],
        )
        member_preprocessing_hash = preprocessing_fingerprint(member_preprocessing)
        member_artifact = {
            "artifact_type": "single_window",
            "pipeline": final_pipeline,
            "probability_calibration": dict(best_probability_calibration),
            "subject_id": subject_filter or "all_subjects",
            "selected_pipeline": best_name,
            "model_type": "optimized",
            "class_names": class_names,
            "display_class_names": [CLASS_NAME_TO_DISPLAY.get(name, name) for name in class_names],
            "channel_names": loaded["channel_names"],
            "channel_indices": list(range(len(loaded["channel_names"]))),
            "sampling_rate": sampling_rate,
            "window_sec": float(window_sec),
            "window_offset_sec": float(best_offset_sec),
            "window_offset_secs_used": [float(item) for item in selected_offset_secs],
            "training_mode": "multi_offset_augmented",
            "preprocessing": dict(member_preprocessing),
            "preprocessing_fingerprint": member_preprocessing_hash,
            "metrics": {
                "val_acc": float(best_selection_metrics["acc"]),
                "val_kappa": float(best_selection_metrics["kappa"]),
                "val_macro_f1": float(best_selection_metrics["macro_f1"]),
                "val_macro_acc": float(best_selection_metrics["macro_acc"]),
                "test_acc": float(test_metrics["acc"]),
                "kappa": float(test_metrics["kappa"]),
                "test_macro_f1": float(test_metrics["macro_f1"]),
                "test_macro_acc": float(test_metrics["macro_acc"]),
                "per_class_test_acc": test_metrics["per_class_acc"],
                "per_class_recall": test_metrics["per_class_recall"],
                "confusion_matrix": test_metrics["confusion_matrix"],
                "left_right_confusion": test_metrics["left_right_confusion"],
                "left_right_confusion_rate": float(test_metrics["left_right_confusion_rate"]),
                "feet_tongue_confusion": test_metrics["feet_tongue_confusion"],
                "feet_tongue_confusion_rate": float(test_metrics["feet_tongue_confusion_rate"]),
            },
            "selection_protocol": {
                "selection_split": "train_to_val",
                "refit_split": "train_plus_val",
                "ranking": ["val_kappa", "val_macro_acc", "val_acc"],
                "candidate_count": int(len(candidate_scores)),
                "candidate_names": list(selected_candidate_names),
                "training_mode": "multi_offset_augmented",
                "guided_runtime_offset_sec": float(best_offset_sec),
            },
            "dataset_root": str(dataset_root),
            "source_sessions": loaded["session_paths"],
            "source_records": loaded["source_records"],
            "split_strategy": split_strategy,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        joblib.dump(member_artifact, member_output_path)
        member_artifacts.append(member_artifact)

        member_summary = {
            "window_sec": float(window_sec),
            "window_offset_sec": float(best_offset_sec),
            "window_offset_secs_used": [float(item) for item in selected_offset_secs],
            "training_mode": "multi_offset_augmented",
            "selected_pipeline": best_name,
            "probability_calibration": dict(best_probability_calibration),
            "preprocessing": dict(member_preprocessing),
            "preprocessing_fingerprint": member_preprocessing_hash,
            "metrics": member_artifact["metrics"],
            "candidate_scores": candidate_scores,
            "candidate_errors": candidate_errors,
            "guided_offset_scores": best_guided_offset_scores or [],
            "model_path": str(member_output_path),
            "report_path": str(member_report_path),
        }

        gate_summary = {
            "available": False,
            "candidate_names": list(selected_gate_candidate_names),
            "train_rest_windows": int(rest_windows_by_split["train"].shape[0]),
            "val_rest_windows": int(rest_eval_windows_by_split["val"].shape[0]),
            "test_rest_windows": int(rest_eval_windows_by_split["test"].shape[0]),
            "train_phase_counts": rest_window_phase_counts.get("train", {}),
            "val_phase_counts": rest_window_phase_counts.get("val", {}),
            "test_phase_counts": rest_window_phase_counts.get("test", {}),
        }
        if (
            gate_control_by_split["train"].shape[0] > 0
            and gate_control_by_split["val"].shape[0] > 0
            and gate_control_by_split["test"].shape[0] > 0
            and rest_windows_by_split["train"].shape[0] > 0
            and rest_eval_windows_by_split["val"].shape[0] > 0
            and rest_eval_windows_by_split["test"].shape[0] > 0
        ):
            gate_train_X_raw, gate_train_y_raw = build_binary_dataset(
                gate_control_by_split["train"],
                rest_windows_by_split["train"],
            )
            gate_train_X, gate_train_y = balance_binary_dataset(
                gate_train_X_raw,
                gate_train_y_raw,
                random_state=int(random_state),
            )
            gate_val_X, gate_val_y = build_binary_dataset(
                gate_control_by_split["val"],
                rest_eval_windows_by_split["val"],
            )
            gate_test_X, gate_test_y = build_binary_dataset(
                gate_control_by_split["test"],
                rest_eval_windows_by_split["test"],
            )
            gate_trainval_X_raw, gate_trainval_y_raw = build_binary_dataset(
                gate_control_by_split["trainval"],
                rest_windows_by_split["trainval"],
            )
            gate_trainval_X, gate_trainval_y = balance_binary_dataset(
                gate_trainval_X_raw,
                gate_trainval_y_raw,
                random_state=int(random_state),
            )

            best_gate_name = None
            best_gate_metrics = None
            best_gate_val_probabilities = None
            best_gate_probability_calibration = None
            gate_candidate_scores = []
            gate_candidate_errors = []
            for name, pipeline in gate_candidate_factory().items():
                try:
                    pipeline.fit(gate_train_X, gate_train_y)
                    raw_gate_val_probabilities = _predict_probability_matrix(
                        pipeline,
                        gate_val_X,
                        len(GATE_CLASS_NAMES),
                    )
                    gate_probability_calibration, gate_val_probabilities = calibrate_probability_matrix(
                        raw_gate_val_probabilities,
                        gate_val_y,
                        role="control_gate_member",
                        selection_source="control_gate_selection_val",
                    )
                    gate_val_predictions = np.argmax(gate_val_probabilities, axis=1)
                except Exception as error:
                    gate_candidate_errors.append({"name": name, "error": str(error)})
                    print(f"window={window_sec:.2f}s gate candidate={name:<16} failed: {error}")
                    continue

                gate_val_metrics = classification_metrics(gate_val_y, gate_val_predictions, GATE_CLASS_NAMES)
                gate_candidate_scores.append(
                    {
                        "name": name,
                        "val_acc": float(gate_val_metrics["acc"]),
                        "val_kappa": float(gate_val_metrics["kappa"]),
                        "val_macro_acc": float(gate_val_metrics["macro_acc"]),
                        "calibration_temperature": float(gate_probability_calibration["temperature"]),
                    }
                )
                print(
                    f"window={window_sec:.2f}s gate candidate={name:<16} "
                    f"val_kappa={float(gate_val_metrics['kappa']):.4f} val_acc={float(gate_val_metrics['acc']):.4f}"
                )
                if best_gate_metrics is None or metric_sort_key(gate_val_metrics) > metric_sort_key(best_gate_metrics):
                    best_gate_name = name
                    best_gate_metrics = gate_val_metrics
                    best_gate_val_probabilities = gate_val_probabilities
                    best_gate_probability_calibration = dict(gate_probability_calibration)

            if (
                best_gate_name is not None
                and best_gate_metrics is not None
                and best_gate_val_probabilities is not None
                and best_gate_probability_calibration is not None
            ):
                final_gate_pipeline = gate_candidate_factory()[best_gate_name]
                final_gate_pipeline.fit(gate_trainval_X, gate_trainval_y)
                raw_gate_test_probabilities = _predict_probability_matrix(
                    final_gate_pipeline,
                    gate_test_X,
                    len(GATE_CLASS_NAMES),
                )
                gate_test_probabilities = apply_probability_calibration(
                    raw_gate_test_probabilities,
                    best_gate_probability_calibration,
                )
                gate_test_predictions = np.argmax(gate_test_probabilities, axis=1)
                gate_test_metrics = classification_metrics(gate_test_y, gate_test_predictions, GATE_CLASS_NAMES)

                if gate_val_labels_reference is None:
                    gate_val_labels_reference = gate_val_y.copy()
                elif not np.array_equal(gate_val_labels_reference, gate_val_y):
                    raise RuntimeError("Gate validation windows are misaligned across member windows.")
                if gate_test_labels_reference is None:
                    gate_test_labels_reference = gate_test_y.copy()
                elif not np.array_equal(gate_test_labels_reference, gate_test_y):
                    raise RuntimeError("Gate test windows are misaligned across member windows.")
                gate_val_probability_vectors.append(best_gate_val_probabilities)
                gate_test_probability_vectors.append(gate_test_probabilities)

                gate_preprocessing = build_preprocessing_config(
                    window_sec=float(window_sec),
                    window_offset_sec=0.0,
                    window_offset_secs_used=[0.0],
                )
                gate_preprocessing_hash = preprocessing_fingerprint(gate_preprocessing)
                gate_member_artifact = {
                    "artifact_type": "single_window",
                    "pipeline": final_gate_pipeline,
                    "probability_calibration": dict(best_gate_probability_calibration),
                    "subject_id": subject_filter or "all_subjects",
                    "selected_pipeline": best_gate_name,
                    "model_type": "optimized",
                    "class_names": list(GATE_CLASS_NAMES),
                    "display_class_names": list(GATE_DISPLAY_CLASS_NAMES),
                    "channel_names": loaded["channel_names"],
                    "channel_indices": list(range(len(loaded["channel_names"]))),
                    "sampling_rate": sampling_rate,
                    "window_sec": float(window_sec),
                    "window_offset_sec": 0.0,
                    "preprocessing": dict(gate_preprocessing),
                    "preprocessing_fingerprint": gate_preprocessing_hash,
                    "metrics": {
                        "val_acc": float(best_gate_metrics["acc"]),
                        "val_kappa": float(best_gate_metrics["kappa"]),
                        "val_macro_acc": float(best_gate_metrics["macro_acc"]),
                        "test_acc": float(gate_test_metrics["acc"]),
                        "kappa": float(gate_test_metrics["kappa"]),
                        "test_macro_acc": float(gate_test_metrics["macro_acc"]),
                        "per_class_test_acc": gate_test_metrics["per_class_acc"],
                    },
                    "selection_protocol": {
                        "selection_split": "train_to_val",
                        "refit_split": "train_plus_val",
                        "ranking": ["val_kappa", "val_macro_acc", "val_acc"],
                        "candidate_count": int(len(gate_candidate_scores)),
                        "candidate_names": list(selected_gate_candidate_names),
                        "control_offsets_used": [float(item) for item in selected_offset_secs],
                        "balanced_train_windows": int(gate_train_y.shape[0]),
                        "balanced_trainval_windows": int(gate_trainval_y.shape[0]),
                    },
                    "dataset_root": str(dataset_root),
                    "source_sessions": loaded["session_paths"],
                    "source_records": loaded["source_records"],
                    "split_strategy": gate_split_strategy,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
                gate_member_artifacts.append(gate_member_artifact)
                gate_summary = {
                    "available": True,
                    "window_sec": float(window_sec),
                    "selected_pipeline": best_gate_name,
                    "probability_calibration": dict(best_gate_probability_calibration),
                    "preprocessing": dict(gate_preprocessing),
                    "preprocessing_fingerprint": gate_preprocessing_hash,
                    "metrics": gate_member_artifact["metrics"],
                    "candidate_scores": gate_candidate_scores,
                    "candidate_errors": gate_candidate_errors,
                    "candidate_names": list(selected_gate_candidate_names),
                    "control_offsets_used": [float(item) for item in selected_offset_secs],
                    "train_rest_windows": int(rest_windows_by_split["train"].shape[0]),
                    "val_rest_windows": int(rest_eval_windows_by_split["val"].shape[0]),
                    "test_rest_windows": int(rest_eval_windows_by_split["test"].shape[0]),
                    "train_phase_counts": rest_window_phase_counts.get("train", {}),
                    "val_phase_counts": rest_window_phase_counts.get("val", {}),
                    "test_phase_counts": rest_window_phase_counts.get("test", {}),
                }
            else:
                gate_summary["reason"] = "no_gate_candidate_trained"
                gate_summary["candidate_errors"] = gate_candidate_errors
        else:
            gate_summary["reason"] = "insufficient_rest_or_control_windows"

        artifact_summary = {
            "available": False,
            "candidate_names": list(selected_artifact_candidate_names),
            "class_names": list(ARTIFACT_REJECTOR_CLASS_NAMES),
        }
        artifact_train_X, artifact_train_y = artifact_binary_by_split["train"]
        artifact_val_X, artifact_val_y = artifact_binary_by_split["val"]
        artifact_test_X, artifact_test_y = artifact_binary_by_split["test"]
        artifact_trainval_X, artifact_trainval_y = artifact_binary_by_split["trainval"]
        artifact_summary["train_windows"] = int(artifact_train_X.shape[0])
        artifact_summary["val_windows"] = int(artifact_val_X.shape[0])
        artifact_summary["test_windows"] = int(artifact_test_X.shape[0])
        artifact_summary["train_clean_windows"] = int(np.sum(artifact_train_y == 0))
        artifact_summary["train_artifact_windows"] = int(np.sum(artifact_train_y == 1))
        artifact_summary["val_clean_windows"] = int(np.sum(artifact_val_y == 0))
        artifact_summary["val_artifact_windows"] = int(np.sum(artifact_val_y == 1))
        artifact_summary["test_clean_windows"] = int(np.sum(artifact_test_y == 0))
        artifact_summary["test_artifact_windows"] = int(np.sum(artifact_test_y == 1))
        if (
            artifact_train_X.shape[0] > 0
            and artifact_val_X.shape[0] > 0
            and artifact_test_X.shape[0] > 0
            and split_contains_all_classes(artifact_train_y, 2)
            and split_contains_all_classes(artifact_val_y, 2)
            and split_contains_all_classes(artifact_test_y, 2)
        ):
            artifact_train_X_balanced, artifact_train_y_balanced = balance_binary_dataset(
                artifact_train_X,
                artifact_train_y,
                random_state=int(random_state) + 233,
            )
            artifact_trainval_X_balanced, artifact_trainval_y_balanced = balance_binary_dataset(
                artifact_trainval_X,
                artifact_trainval_y,
                random_state=int(random_state) + 239,
            )

            best_artifact_name = None
            best_artifact_metrics = None
            best_artifact_val_probabilities = None
            best_artifact_probability_calibration = None
            artifact_candidate_scores = []
            artifact_candidate_errors = []
            for name, pipeline in artifact_candidate_factory().items():
                try:
                    pipeline.fit(artifact_train_X_balanced, artifact_train_y_balanced)
                    raw_artifact_val_probabilities = _predict_probability_matrix(
                        pipeline,
                        artifact_val_X,
                        len(ARTIFACT_REJECTOR_CLASS_NAMES),
                    )
                    artifact_probability_calibration, artifact_val_probabilities = calibrate_probability_matrix(
                        raw_artifact_val_probabilities,
                        artifact_val_y,
                        role="artifact_rejector_member",
                        selection_source="artifact_rejector_selection_val",
                    )
                    artifact_val_predictions = np.argmax(artifact_val_probabilities, axis=1)
                except Exception as error:
                    artifact_candidate_errors.append({"name": name, "error": str(error)})
                    print(f"window={window_sec:.2f}s artifact candidate={name:<16} failed: {error}")
                    continue

                artifact_val_metrics = classification_metrics(
                    artifact_val_y,
                    artifact_val_predictions,
                    ARTIFACT_REJECTOR_CLASS_NAMES,
                )
                artifact_candidate_scores.append(
                    {
                        "name": name,
                        "val_acc": float(artifact_val_metrics["acc"]),
                        "val_kappa": float(artifact_val_metrics["kappa"]),
                        "val_macro_acc": float(artifact_val_metrics["macro_acc"]),
                        "calibration_temperature": float(artifact_probability_calibration["temperature"]),
                    }
                )
                print(
                    f"window={window_sec:.2f}s artifact candidate={name:<16} "
                    f"val_kappa={float(artifact_val_metrics['kappa']):.4f} "
                    f"val_acc={float(artifact_val_metrics['acc']):.4f}"
                )
                if best_artifact_metrics is None or metric_sort_key(artifact_val_metrics) > metric_sort_key(best_artifact_metrics):
                    best_artifact_name = name
                    best_artifact_metrics = artifact_val_metrics
                    best_artifact_val_probabilities = artifact_val_probabilities
                    best_artifact_probability_calibration = dict(artifact_probability_calibration)

            if (
                best_artifact_name is not None
                and best_artifact_metrics is not None
                and best_artifact_val_probabilities is not None
                and best_artifact_probability_calibration is not None
            ):
                final_artifact_pipeline = artifact_candidate_factory()[best_artifact_name]
                final_artifact_pipeline.fit(artifact_trainval_X_balanced, artifact_trainval_y_balanced)
                raw_artifact_test_probabilities = _predict_probability_matrix(
                    final_artifact_pipeline,
                    artifact_test_X,
                    len(ARTIFACT_REJECTOR_CLASS_NAMES),
                )
                artifact_test_probabilities = apply_probability_calibration(
                    raw_artifact_test_probabilities,
                    best_artifact_probability_calibration,
                )
                artifact_test_predictions = np.argmax(artifact_test_probabilities, axis=1)
                artifact_test_metrics = classification_metrics(
                    artifact_test_y,
                    artifact_test_predictions,
                    ARTIFACT_REJECTOR_CLASS_NAMES,
                )

                if artifact_val_labels_reference is None:
                    artifact_val_labels_reference = artifact_val_y.copy()
                elif not np.array_equal(artifact_val_labels_reference, artifact_val_y):
                    raise RuntimeError("Artifact validation windows are misaligned across member windows.")
                if artifact_test_labels_reference is None:
                    artifact_test_labels_reference = artifact_test_y.copy()
                elif not np.array_equal(artifact_test_labels_reference, artifact_test_y):
                    raise RuntimeError("Artifact test windows are misaligned across member windows.")
                artifact_val_probability_vectors.append(best_artifact_val_probabilities)
                artifact_test_probability_vectors.append(artifact_test_probabilities)

                artifact_preprocessing = build_preprocessing_config(
                    window_sec=float(window_sec),
                    window_offset_sec=0.0,
                    window_offset_secs_used=[0.0],
                )
                artifact_preprocessing_hash = preprocessing_fingerprint(artifact_preprocessing)
                artifact_member_artifact = {
                    "artifact_type": "single_window",
                    "pipeline": final_artifact_pipeline,
                    "probability_calibration": dict(best_artifact_probability_calibration),
                    "subject_id": subject_filter or "all_subjects",
                    "selected_pipeline": best_artifact_name,
                    "model_type": "optimized",
                    "class_names": list(ARTIFACT_REJECTOR_CLASS_NAMES),
                    "display_class_names": list(ARTIFACT_REJECTOR_DISPLAY_CLASS_NAMES),
                    "channel_names": loaded["channel_names"],
                    "channel_indices": list(range(len(loaded["channel_names"]))),
                    "sampling_rate": sampling_rate,
                    "window_sec": float(window_sec),
                    "window_offset_sec": 0.0,
                    "preprocessing": dict(artifact_preprocessing),
                    "preprocessing_fingerprint": artifact_preprocessing_hash,
                    "metrics": {
                        "val_acc": float(best_artifact_metrics["acc"]),
                        "val_kappa": float(best_artifact_metrics["kappa"]),
                        "val_macro_acc": float(best_artifact_metrics["macro_acc"]),
                        "test_acc": float(artifact_test_metrics["acc"]),
                        "kappa": float(artifact_test_metrics["kappa"]),
                        "test_macro_acc": float(artifact_test_metrics["macro_acc"]),
                        "per_class_test_acc": artifact_test_metrics["per_class_acc"],
                    },
                    "selection_protocol": {
                        "selection_split": "train_to_val",
                        "refit_split": "train_plus_val",
                        "ranking": ["val_kappa", "val_macro_acc", "val_acc"],
                        "candidate_count": int(len(artifact_candidate_scores)),
                        "candidate_names": list(selected_artifact_candidate_names),
                        "balanced_train_windows": int(artifact_train_y_balanced.shape[0]),
                        "balanced_trainval_windows": int(artifact_trainval_y_balanced.shape[0]),
                    },
                    "dataset_root": str(dataset_root),
                    "source_sessions": loaded["session_paths"],
                    "source_records": loaded["source_records"],
                    "split_strategy": artifact_split_strategy,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
                artifact_member_artifacts.append(artifact_member_artifact)
                artifact_summary = {
                    "available": True,
                    "window_sec": float(window_sec),
                    "selected_pipeline": best_artifact_name,
                    "probability_calibration": dict(best_artifact_probability_calibration),
                    "candidate_names": list(selected_artifact_candidate_names),
                    "class_names": list(ARTIFACT_REJECTOR_CLASS_NAMES),
                    "preprocessing": dict(artifact_preprocessing),
                    "preprocessing_fingerprint": artifact_preprocessing_hash,
                    "metrics": artifact_member_artifact["metrics"],
                    "candidate_scores": artifact_candidate_scores,
                    "candidate_errors": artifact_candidate_errors,
                    "train_windows": int(artifact_train_X.shape[0]),
                    "val_windows": int(artifact_val_X.shape[0]),
                    "test_windows": int(artifact_test_X.shape[0]),
                    "train_clean_windows": int(np.sum(artifact_train_y == 0)),
                    "train_artifact_windows": int(np.sum(artifact_train_y == 1)),
                    "val_clean_windows": int(np.sum(artifact_val_y == 0)),
                    "val_artifact_windows": int(np.sum(artifact_val_y == 1)),
                    "test_clean_windows": int(np.sum(artifact_test_y == 0)),
                    "test_artifact_windows": int(np.sum(artifact_test_y == 1)),
                }
            else:
                artifact_summary["reason"] = "no_artifact_candidate_trained"
                artifact_summary["candidate_errors"] = artifact_candidate_errors
        else:
            artifact_summary["reason"] = "insufficient_clean_or_artifact_windows"

        member_summary["control_gate"] = gate_summary
        member_summary["artifact_rejector"] = artifact_summary
        member_summaries.append(member_summary)
        if gate_summary.get("available"):
            gate_member_summaries.append(gate_summary)
        if artifact_summary.get("available"):
            artifact_member_summaries.append(artifact_summary)

        with member_report_path.open("w", encoding="utf-8") as file:
            json.dump(member_summary, file, indent=2, ensure_ascii=False)
        with member_output_path.with_suffix(".json").open("w", encoding="utf-8") as file:
            json.dump(member_summary, file, indent=2, ensure_ascii=False)

    default_fusion_weights = aligned_fusion_weights or default_window_weights(selected_window_secs)
    if aligned_fusion_weights is None:
        effective_fusion_weights, fusion_selection = select_fusion_weights(
            val_probability_vectors,
            split_labels["val"],
            class_names,
            fusion_method=fusion_method,
            fallback_weights=default_fusion_weights,
        )
    else:
        effective_fusion_weights = [float(item) for item in aligned_fusion_weights]
        fused_val_for_selection = fuse_probability_stack(
            val_probability_vectors,
            effective_fusion_weights,
            fusion_method=fusion_method,
        )
        fusion_selection = {
            "strategy": "user_provided",
            "candidate_count": 1,
            "metrics": classification_metrics(
                split_labels["val"],
                np.argmax(fused_val_for_selection, axis=1),
                class_names,
            ),
        }

    bank_artifact = build_realtime_artifact_bank(
        member_artifacts,
        fusion_weights=effective_fusion_weights,
        fusion_method=fusion_method,
    )
    fused_val = fuse_probability_stack(
        val_probability_vectors,
        bank_artifact["fusion_weights"],
        fusion_method=fusion_method,
    )
    fused_test = fuse_probability_stack(
        test_probability_vectors,
        bank_artifact["fusion_weights"],
        fusion_method=fusion_method,
    )
    bank_probability_calibration, fused_val = calibrate_probability_matrix(
        fused_val,
        split_labels["val"],
        role="main_classifier_bank",
        selection_source="main_bank_selection_val",
    )
    fused_test = apply_probability_calibration(fused_test, bank_probability_calibration)
    fused_val_predictions = np.argmax(fused_val, axis=1)
    fused_test_predictions = np.argmax(fused_test, axis=1)
    fused_val_metrics = classification_metrics(split_labels["val"], fused_val_predictions, class_names)
    fused_test_metrics = classification_metrics(split_labels["test"], fused_test_predictions, class_names)

    bank_artifact["probability_calibration"] = dict(bank_probability_calibration)
    bank_artifact["metrics"] = {
        "selection_val_acc": float(fused_val_metrics["acc"]),
        "selection_val_kappa": float(fused_val_metrics["kappa"]),
        "selection_val_macro_f1": float(fused_val_metrics["macro_f1"]),
        "selection_val_macro_acc": float(fused_val_metrics["macro_acc"]),
        "bank_test_acc": float(fused_test_metrics["acc"]),
        "bank_kappa": float(fused_test_metrics["kappa"]),
        "bank_macro_f1": float(fused_test_metrics["macro_f1"]),
        "bank_test_macro_acc": float(fused_test_metrics["macro_acc"]),
        "bank_per_class_test_acc": fused_test_metrics["per_class_acc"],
        "bank_per_class_recall": fused_test_metrics["per_class_recall"],
        "bank_confusion_matrix": fused_test_metrics["confusion_matrix"],
        "left_right_confusion": fused_test_metrics["left_right_confusion"],
        "left_right_confusion_rate": float(fused_test_metrics["left_right_confusion_rate"]),
        "feet_tongue_confusion": fused_test_metrics["feet_tongue_confusion"],
        "feet_tongue_confusion_rate": float(fused_test_metrics["feet_tongue_confusion_rate"]),
        "fusion_selection": fusion_selection,
        "members": [
            {
                "window_sec": float(summary["window_sec"]),
                "window_offset_sec": float(summary["window_offset_sec"]),
                "selected_pipeline": summary["selected_pipeline"],
                "metrics": summary["metrics"],
            }
            for summary in member_summaries
        ],
    }
    bank_artifact["created_at"] = datetime.now().isoformat(timespec="seconds")
    bank_artifact["dataset_root"] = str(dataset_root)
    bank_artifact["source_sessions"] = loaded["session_paths"]
    bank_artifact["source_records"] = loaded["source_records"]
    bank_artifact["window_offset_secs"] = [float(item) for item in selected_offset_secs]
    bank_artifact["split_strategy"] = split_strategy
    bank_artifact["split_assignments"] = mi_split_assignments
    bank_artifact["candidate_names"] = list(selected_candidate_names)
    bank_artifact["gate_candidate_names"] = list(selected_gate_candidate_names)
    bank_artifact["artifact_candidate_names"] = list(selected_artifact_candidate_names)
    bank_artifact["torch_available"] = bool(TORCH_AVAILABLE)
    bank_artifact["deep_stage_pretrain_window_secs"] = [
        float(item) for item in (deep_stage_pretrain_window_secs or DEFAULT_DEEP_STAGE_PRETRAIN_WINDOW_SECS)
    ]
    bank_artifact["deep_stage_finetune_window_secs"] = [
        float(item) for item in (deep_stage_finetune_window_secs or DEFAULT_DEEP_STAGE_FINETUNE_WINDOW_SECS)
    ]
    bank_artifact["central_prior_alpha"] = float(central_prior_alpha)
    bank_artifact["central_aux_loss_weight"] = float(central_aux_loss_weight)
    bank_preprocessing = dict(bank_artifact.get("preprocessing", {}))
    bank_artifact["preprocessing"] = bank_preprocessing
    bank_artifact["preprocessing_fingerprint"] = (
        preprocessing_fingerprint(bank_preprocessing) if bank_preprocessing else ""
    )
    bank_artifact["member_preprocessing_fingerprints"] = [
        str(item.get("preprocessing_fingerprint", ""))
        for item in member_artifacts
    ]

    gate_runtime = None
    gate_summary: dict[str, object] = {
        "enabled": False,
        "candidate_names": list(selected_gate_candidate_names),
        "available_member_count": int(len(gate_member_artifacts)),
        "strict_split_block_reason": gate_split_block_reason,
    }
    if (
        gate_member_artifacts
        and gate_val_labels_reference is not None
        and gate_test_labels_reference is not None
        and gate_val_probability_vectors
        and gate_test_probability_vectors
    ):
        gate_window_secs = [float(item["window_sec"]) for item in gate_member_artifacts]
        gate_default_weights = (
            [float(item) for item in aligned_fusion_weights]
            if aligned_fusion_weights is not None and len(aligned_fusion_weights) == len(gate_member_artifacts)
            else default_window_weights(gate_window_secs)
        )
        if aligned_fusion_weights is None or len(aligned_fusion_weights) != len(gate_member_artifacts):
            effective_gate_weights, gate_fusion_selection = select_fusion_weights(
                gate_val_probability_vectors,
                gate_val_labels_reference,
                GATE_CLASS_NAMES,
                fusion_method=fusion_method,
                fallback_weights=gate_default_weights,
            )
        else:
            effective_gate_weights = gate_default_weights
            fused_gate_val_for_selection = fuse_probability_stack(
                gate_val_probability_vectors,
                effective_gate_weights,
                fusion_method=fusion_method,
            )
            gate_fusion_selection = {
                "strategy": "user_provided",
                "candidate_count": 1,
                "metrics": classification_metrics(
                    gate_val_labels_reference,
                    np.argmax(fused_gate_val_for_selection, axis=1),
                    GATE_CLASS_NAMES,
                ),
            }

        gate_bank_artifact = build_realtime_artifact_bank(
            gate_member_artifacts,
            fusion_weights=effective_gate_weights,
            fusion_method=fusion_method,
        )
        fused_gate_val = fuse_probability_stack(
            gate_val_probability_vectors,
            gate_bank_artifact["fusion_weights"],
            fusion_method=fusion_method,
        )
        fused_gate_test = fuse_probability_stack(
            gate_test_probability_vectors,
            gate_bank_artifact["fusion_weights"],
            fusion_method=fusion_method,
        )
        gate_bank_probability_calibration, fused_gate_val = calibrate_probability_matrix(
            fused_gate_val,
            gate_val_labels_reference,
            role="control_gate_bank",
            selection_source="control_gate_bank_selection_val",
        )
        fused_gate_test = apply_probability_calibration(
            fused_gate_test,
            gate_bank_probability_calibration,
        )
        fused_gate_val_predictions = np.argmax(fused_gate_val, axis=1)
        fused_gate_test_predictions = np.argmax(fused_gate_test, axis=1)
        fused_gate_val_metrics = classification_metrics(
            gate_val_labels_reference,
            fused_gate_val_predictions,
            GATE_CLASS_NAMES,
        )
        fused_gate_test_metrics = classification_metrics(
            gate_test_labels_reference,
            fused_gate_test_predictions,
            GATE_CLASS_NAMES,
        )
        gate_threshold_recommendation = select_gate_thresholds(
            fused_gate_val[gate_val_labels_reference == 1],
            fused_gate_val[gate_val_labels_reference == 0],
            target_rest_activation_rate=float(DEFAULT_REST_FALSE_ACTIVATION_TARGET),
            control_class_index=1,
        )
        gate_runtime = {
            "confidence_threshold": float(gate_threshold_recommendation["confidence_threshold"]),
            "margin_threshold": float(gate_threshold_recommendation["margin_threshold"]),
            "recommended_threshold": {
                "confidence": float(gate_threshold_recommendation["confidence_threshold"]),
                "margin": float(gate_threshold_recommendation["margin_threshold"]),
            },
            "selection_source": "control_gate_calibration",
            "calibration": gate_threshold_recommendation,
            "probability_calibration": dict(gate_bank_probability_calibration),
            "decision_role": "control_gate",
            "outputs": {
                "probability": "gate_probabilities",
                "margin": "gate_margin",
                "confidence": "gate_confidence",
            },
            "runtime_output_fields": {
                "probability": "gate_probabilities",
                "margin": "gate_margin",
                "confidence": "gate_confidence",
            },
        }
        gate_test_threshold_eval = evaluate_gate_thresholds(
            fused_gate_test[gate_test_labels_reference == 1],
            fused_gate_test[gate_test_labels_reference == 0],
            confidence_threshold=float(gate_runtime["confidence_threshold"]),
            margin_threshold=float(gate_runtime["margin_threshold"]),
            control_class_index=1,
        )
        gate_bank_artifact["probability_calibration"] = dict(gate_bank_probability_calibration)
        gate_bank_artifact["recommended_runtime"] = gate_runtime
        gate_bank_artifact["metrics"] = {
            "selection_val_acc": float(fused_gate_val_metrics["acc"]),
            "selection_val_kappa": float(fused_gate_val_metrics["kappa"]),
            "selection_val_macro_acc": float(fused_gate_val_metrics["macro_acc"]),
            "bank_test_acc": float(fused_gate_test_metrics["acc"]),
            "bank_kappa": float(fused_gate_test_metrics["kappa"]),
            "bank_test_macro_acc": float(fused_gate_test_metrics["macro_acc"]),
            "bank_per_class_test_acc": fused_gate_test_metrics["per_class_acc"],
            "fusion_selection": gate_fusion_selection,
            "threshold_eval_test": {
                "control_detection_rate": float(gate_test_threshold_eval["control_detection_rate"]),
                "rest_false_activation_rate": float(gate_test_threshold_eval["rest_false_activation_rate"]),
            },
            "members": [
                {
                    "window_sec": float(summary["window_sec"]),
                    "selected_pipeline": str(summary["selected_pipeline"]),
                    "metrics": dict(summary["metrics"]),
                }
                for summary in gate_member_summaries
            ],
        }
        gate_bank_artifact["created_at"] = datetime.now().isoformat(timespec="seconds")
        gate_bank_artifact["dataset_root"] = str(dataset_root)
        gate_bank_artifact["source_sessions"] = loaded["session_paths"]
        gate_bank_artifact["source_records"] = loaded["source_records"]
        gate_bank_artifact["candidate_names"] = list(selected_gate_candidate_names)
        gate_bank_artifact["window_offset_secs"] = [float(item) for item in selected_offset_secs]
        gate_bank_artifact["split_strategy"] = gate_split_strategy
        gate_bank_artifact["split_assignments"] = gate_split_assignments
        gate_bank_preprocessing = dict(gate_bank_artifact.get("preprocessing", {}))
        gate_bank_artifact["preprocessing"] = gate_bank_preprocessing
        gate_bank_artifact["preprocessing_fingerprint"] = (
            preprocessing_fingerprint(gate_bank_preprocessing) if gate_bank_preprocessing else ""
        )
        gate_bank_artifact["member_preprocessing_fingerprints"] = [
            str(item.get("preprocessing_fingerprint", ""))
            for item in gate_member_artifacts
        ]
        gate_calibration = dict(gate_runtime.get("calibration") or {})
        gate_per_class_test_acc = {
            str(name): float(value)
            for name, value in dict(gate_bank_artifact["metrics"].get("bank_per_class_test_acc", {})).items()
        }
        gate_control_acc = float(gate_per_class_test_acc.get("control", 0.0))
        gate_rest_acc = float(gate_per_class_test_acc.get("rest", 0.0))
        gate_control_detection = float(gate_calibration.get("control_detection_rate", 0.0))
        gate_selection_feasible = bool(gate_calibration.get("selection_feasible", True))
        gate_auto_disable_reasons: list[str] = []
        if not gate_selection_feasible:
            gate_auto_disable_reasons.append("threshold_selection_not_feasible")
        if gate_control_detection < float(DEFAULT_MIN_GATE_CONTROL_DETECTION_RATE):
            gate_auto_disable_reasons.append(
                f"control_detection_too_low:{gate_control_detection:.3f}<"
                f"{float(DEFAULT_MIN_GATE_CONTROL_DETECTION_RATE):.3f}"
            )
        if gate_control_acc < float(DEFAULT_MIN_BINARY_STAGE_CLASS_ACCURACY):
            gate_auto_disable_reasons.append(
                f"control_class_acc_too_low:{gate_control_acc:.3f}<"
                f"{float(DEFAULT_MIN_BINARY_STAGE_CLASS_ACCURACY):.3f}"
            )
        if gate_rest_acc < float(DEFAULT_MIN_BINARY_STAGE_CLASS_ACCURACY):
            gate_auto_disable_reasons.append(
                f"rest_class_acc_too_low:{gate_rest_acc:.3f}<"
                f"{float(DEFAULT_MIN_BINARY_STAGE_CLASS_ACCURACY):.3f}"
            )

        if gate_auto_disable_reasons:
            gate_summary = {
                "enabled": False,
                "candidate_names": list(selected_gate_candidate_names),
                "available_member_count": int(len(gate_member_artifacts)),
                "split_strategy": gate_split_strategy,
                "strict_split_block_reason": gate_split_block_reason,
                "auto_disabled": True,
                "auto_disable_reasons": gate_auto_disable_reasons,
                "metrics": gate_bank_artifact["metrics"],
                "recommended_runtime": gate_runtime,
            }
            bank_artifact["control_gate"] = None
        else:
            gate_summary = {
                "enabled": True,
                "candidate_names": list(selected_gate_candidate_names),
                "fusion_weights": list(gate_bank_artifact["fusion_weights"]),
                "fusion_method": fusion_method,
                "probability_calibration": dict(gate_bank_probability_calibration),
                "recommended_runtime": gate_runtime,
                "metrics": gate_bank_artifact["metrics"],
                "member_models": gate_member_summaries,
                "split_strategy": gate_split_strategy,
                "strict_split_block_reason": gate_split_block_reason,
            }
            bank_artifact["control_gate"] = gate_bank_artifact
    else:
        bank_artifact["control_gate"] = None

    artifact_runtime = None
    artifact_rejector_summary: dict[str, object] = {
        "enabled": False,
        "candidate_names": list(selected_artifact_candidate_names),
        "class_names": list(ARTIFACT_REJECTOR_CLASS_NAMES),
        "available_member_count": int(len(artifact_member_artifacts)),
        "split_strategy": artifact_split_strategy,
        "strict_split_block_reason": artifact_split_block_reason,
    }
    if (
        artifact_member_artifacts
        and artifact_val_labels_reference is not None
        and artifact_test_labels_reference is not None
        and artifact_val_probability_vectors
        and artifact_test_probability_vectors
    ):
        artifact_window_secs = [float(item["window_sec"]) for item in artifact_member_artifacts]
        artifact_default_weights = (
            [float(item) for item in aligned_fusion_weights]
            if aligned_fusion_weights is not None and len(aligned_fusion_weights) == len(artifact_member_artifacts)
            else default_window_weights(artifact_window_secs)
        )
        if aligned_fusion_weights is None or len(aligned_fusion_weights) != len(artifact_member_artifacts):
            effective_artifact_weights, artifact_fusion_selection = select_fusion_weights(
                artifact_val_probability_vectors,
                artifact_val_labels_reference,
                ARTIFACT_REJECTOR_CLASS_NAMES,
                fusion_method=fusion_method,
                fallback_weights=artifact_default_weights,
            )
        else:
            effective_artifact_weights = artifact_default_weights
            fused_artifact_val_for_selection = fuse_probability_stack(
                artifact_val_probability_vectors,
                effective_artifact_weights,
                fusion_method=fusion_method,
            )
            artifact_fusion_selection = {
                "strategy": "user_provided",
                "candidate_count": 1,
                "metrics": classification_metrics(
                    artifact_val_labels_reference,
                    np.argmax(fused_artifact_val_for_selection, axis=1),
                    ARTIFACT_REJECTOR_CLASS_NAMES,
                ),
            }

        artifact_bank_artifact = build_realtime_artifact_bank(
            artifact_member_artifacts,
            fusion_weights=effective_artifact_weights,
            fusion_method=fusion_method,
        )
        fused_artifact_val = fuse_probability_stack(
            artifact_val_probability_vectors,
            artifact_bank_artifact["fusion_weights"],
            fusion_method=fusion_method,
        )
        fused_artifact_test = fuse_probability_stack(
            artifact_test_probability_vectors,
            artifact_bank_artifact["fusion_weights"],
            fusion_method=fusion_method,
        )
        artifact_bank_probability_calibration, fused_artifact_val = calibrate_probability_matrix(
            fused_artifact_val,
            artifact_val_labels_reference,
            role="artifact_rejector_bank",
            selection_source="artifact_rejector_bank_selection_val",
        )
        fused_artifact_test = apply_probability_calibration(
            fused_artifact_test,
            artifact_bank_probability_calibration,
        )
        fused_artifact_val_predictions = np.argmax(fused_artifact_val, axis=1)
        fused_artifact_test_predictions = np.argmax(fused_artifact_test, axis=1)
        fused_artifact_val_metrics = classification_metrics(
            artifact_val_labels_reference,
            fused_artifact_val_predictions,
            ARTIFACT_REJECTOR_CLASS_NAMES,
        )
        fused_artifact_test_metrics = classification_metrics(
            artifact_test_labels_reference,
            fused_artifact_test_predictions,
            ARTIFACT_REJECTOR_CLASS_NAMES,
        )
        artifact_threshold_recommendation = select_gate_thresholds(
            fused_artifact_val[artifact_val_labels_reference == 1],
            fused_artifact_val[artifact_val_labels_reference == 0],
            target_rest_activation_rate=float(DEFAULT_REST_FALSE_ACTIVATION_TARGET),
            control_class_index=1,
        )
        artifact_runtime = {
            "confidence_threshold": float(artifact_threshold_recommendation["confidence_threshold"]),
            "margin_threshold": float(artifact_threshold_recommendation["margin_threshold"]),
            "recommended_threshold": {
                "confidence": float(artifact_threshold_recommendation["confidence_threshold"]),
                "margin": float(artifact_threshold_recommendation["margin_threshold"]),
            },
            "selection_source": "artifact_rejector_calibration",
            "calibration": artifact_threshold_recommendation,
            "probability_calibration": dict(artifact_bank_probability_calibration),
            "decision_role": "bad_window_rejector",
            "outputs": {
                "probability": "artifact_probabilities",
                "margin": "artifact_margin",
                "confidence": "artifact_confidence",
            },
            "runtime_output_fields": {
                "probability": "artifact_probabilities",
                "margin": "artifact_margin",
                "confidence": "artifact_confidence",
            },
        }
        artifact_test_threshold_eval = evaluate_gate_thresholds(
            fused_artifact_test[artifact_test_labels_reference == 1],
            fused_artifact_test[artifact_test_labels_reference == 0],
            confidence_threshold=float(artifact_runtime["confidence_threshold"]),
            margin_threshold=float(artifact_runtime["margin_threshold"]),
            control_class_index=1,
        )
        artifact_bank_artifact["probability_calibration"] = dict(artifact_bank_probability_calibration)
        artifact_bank_artifact["recommended_runtime"] = artifact_runtime
        artifact_bank_artifact["metrics"] = {
            "selection_val_acc": float(fused_artifact_val_metrics["acc"]),
            "selection_val_kappa": float(fused_artifact_val_metrics["kappa"]),
            "selection_val_macro_acc": float(fused_artifact_val_metrics["macro_acc"]),
            "bank_test_acc": float(fused_artifact_test_metrics["acc"]),
            "bank_kappa": float(fused_artifact_test_metrics["kappa"]),
            "bank_test_macro_acc": float(fused_artifact_test_metrics["macro_acc"]),
            "bank_per_class_test_acc": fused_artifact_test_metrics["per_class_acc"],
            "fusion_selection": artifact_fusion_selection,
            "threshold_eval_test": {
                "artifact_detection_rate": float(artifact_test_threshold_eval["control_detection_rate"]),
                "clean_false_reject_rate": float(artifact_test_threshold_eval["rest_false_activation_rate"]),
            },
            "members": [
                {
                    "window_sec": float(summary["window_sec"]),
                    "selected_pipeline": str(summary["selected_pipeline"]),
                    "metrics": dict(summary["metrics"]),
                }
                for summary in artifact_member_summaries
            ],
        }
        artifact_bank_artifact["created_at"] = datetime.now().isoformat(timespec="seconds")
        artifact_bank_artifact["dataset_root"] = str(dataset_root)
        artifact_bank_artifact["source_sessions"] = loaded["session_paths"]
        artifact_bank_artifact["source_records"] = loaded["source_records"]
        artifact_bank_artifact["candidate_names"] = list(selected_artifact_candidate_names)
        artifact_bank_artifact["window_offset_secs"] = [0.0]
        artifact_bank_artifact["split_strategy"] = artifact_split_strategy
        artifact_bank_artifact["split_assignments"] = artifact_split_assignments
        artifact_bank_preprocessing = dict(artifact_bank_artifact.get("preprocessing", {}))
        artifact_bank_artifact["preprocessing"] = artifact_bank_preprocessing
        artifact_bank_artifact["preprocessing_fingerprint"] = (
            preprocessing_fingerprint(artifact_bank_preprocessing) if artifact_bank_preprocessing else ""
        )
        artifact_bank_artifact["member_preprocessing_fingerprints"] = [
            str(item.get("preprocessing_fingerprint", ""))
            for item in artifact_member_artifacts
        ]
        artifact_calibration = dict(artifact_runtime.get("calibration") or {})
        artifact_per_class_test_acc = {
            str(name): float(value)
            for name, value in dict(artifact_bank_artifact["metrics"].get("bank_per_class_test_acc", {})).items()
        }
        artifact_detection = float(artifact_calibration.get("control_detection_rate", 0.0))
        clean_acc = float(artifact_per_class_test_acc.get("clean", 0.0))
        artifact_acc = float(artifact_per_class_test_acc.get("artifact", 0.0))
        artifact_selection_feasible = bool(artifact_calibration.get("selection_feasible", True))
        artifact_auto_disable_reasons: list[str] = []
        if not artifact_selection_feasible:
            artifact_auto_disable_reasons.append("threshold_selection_not_feasible")
        if artifact_detection < float(DEFAULT_MIN_ARTIFACT_DETECTION_RATE):
            artifact_auto_disable_reasons.append(
                f"artifact_detection_too_low:{artifact_detection:.3f}<"
                f"{float(DEFAULT_MIN_ARTIFACT_DETECTION_RATE):.3f}"
            )
        if clean_acc < float(DEFAULT_MIN_BINARY_STAGE_CLASS_ACCURACY):
            artifact_auto_disable_reasons.append(
                f"clean_class_acc_too_low:{clean_acc:.3f}<"
                f"{float(DEFAULT_MIN_BINARY_STAGE_CLASS_ACCURACY):.3f}"
            )
        if artifact_acc < float(DEFAULT_MIN_BINARY_STAGE_CLASS_ACCURACY):
            artifact_auto_disable_reasons.append(
                f"artifact_class_acc_too_low:{artifact_acc:.3f}<"
                f"{float(DEFAULT_MIN_BINARY_STAGE_CLASS_ACCURACY):.3f}"
            )

        if artifact_auto_disable_reasons:
            artifact_rejector_summary = {
                "enabled": False,
                "candidate_names": list(selected_artifact_candidate_names),
                "class_names": list(ARTIFACT_REJECTOR_CLASS_NAMES),
                "available_member_count": int(len(artifact_member_artifacts)),
                "split_strategy": artifact_split_strategy,
                "strict_split_block_reason": artifact_split_block_reason,
                "auto_disabled": True,
                "auto_disable_reasons": artifact_auto_disable_reasons,
                "metrics": artifact_bank_artifact["metrics"],
                "recommended_runtime": artifact_runtime,
            }
            bank_artifact["artifact_rejector"] = None
        else:
            artifact_rejector_summary = {
                "enabled": True,
                "candidate_names": list(selected_artifact_candidate_names),
                "class_names": list(ARTIFACT_REJECTOR_CLASS_NAMES),
                "fusion_weights": list(artifact_bank_artifact["fusion_weights"]),
                "fusion_method": fusion_method,
                "probability_calibration": dict(artifact_bank_probability_calibration),
                "recommended_runtime": artifact_runtime,
                "metrics": artifact_bank_artifact["metrics"],
                "member_models": artifact_member_summaries,
                "split_strategy": artifact_split_strategy,
                "strict_split_block_reason": artifact_split_block_reason,
            }
            bank_artifact["artifact_rejector"] = artifact_bank_artifact
    else:
        bank_artifact["artifact_rejector"] = None

    recommended_runtime = None
    rest_calibration_summary = {
        "mode": "rest_calibration_pending",
        "available_rest_segments": int(len(rest_segments_all)),
        "val_rest_segments": int(np.sum(rest_val_mask)) if rest_val_mask.size else 0,
        "test_rest_segments": int(np.sum(rest_test_mask)) if rest_test_mask.size else 0,
        "rest_window_sec": float(max(selected_window_secs)),
        "rest_window_step_sec": float(DEFAULT_REST_WINDOW_STEP_SEC),
    }
    calibration_rest_segments = [rest_segments_all[index] for index, keep in enumerate(rest_val_mask.tolist()) if keep]
    calibration_rest_phases = np.asarray(rest_source_phases_all[rest_val_mask], dtype=object)
    if not calibration_rest_segments and rest_segments_all:
        calibration_rest_segments = list(rest_segments_all)
        calibration_rest_phases = np.asarray(rest_source_phases_all, dtype=object)
        rest_calibration_summary["fallback_pool"] = "all_rest_segments"

    if calibration_rest_segments:
        rest_calibration_summary["mode"] = "rest_segments_available"
        rest_windows_val, rest_window_phases_val = sample_rest_windows(
            calibration_rest_segments,
            calibration_rest_phases,
            sampling_rate,
            window_sec=float(max(selected_window_secs)),
            step_sec=float(DEFAULT_REST_WINDOW_STEP_SEC),
        )
        rest_calibration_summary["val_rest_windows"] = int(rest_windows_val.shape[0])
        if rest_windows_val.shape[0] > 0:
            rest_calibration_summary["val_rest_phase_counts"] = {
                str(phase_name): int(np.sum(rest_window_phases_val == phase_name))
                for phase_name in np.unique(rest_window_phases_val)
            }
            fused_rest_val, _ = evaluate_bank_on_raw_windows(
                rest_windows_val,
                member_artifacts,
                sampling_rate=sampling_rate,
                fusion_weights=bank_artifact["fusion_weights"],
                fusion_method=fusion_method,
                class_count=len(class_names),
                probability_calibration=bank_artifact.get("probability_calibration"),
            )
            threshold_recommendation = select_reject_thresholds(
                fused_val,
                fused_rest_val,
                target_rest_activation_rate=float(DEFAULT_REST_FALSE_ACTIVATION_TARGET),
            )
            recommended_runtime = {
                "confidence_threshold": float(threshold_recommendation["confidence_threshold"]),
                "margin_threshold": float(threshold_recommendation["margin_threshold"]),
                "selection_source": "rest_calibration",
                "calibration": threshold_recommendation,
                "probability_calibration": dict(bank_artifact["probability_calibration"]),
            }

            calibration_test_segments = [rest_segments_all[index] for index, keep in enumerate(rest_test_mask.tolist()) if keep]
            calibration_test_phases = np.asarray(rest_source_phases_all[rest_test_mask], dtype=object)
            if not calibration_test_segments and rest_segments_all:
                calibration_test_segments = list(rest_segments_all)
                calibration_test_phases = np.asarray(rest_source_phases_all, dtype=object)
            if calibration_test_segments:
                rest_windows_test, rest_window_phases_test = sample_rest_windows(
                    calibration_test_segments,
                    calibration_test_phases,
                    sampling_rate,
                    window_sec=float(max(selected_window_secs)),
                    step_sec=float(DEFAULT_REST_WINDOW_STEP_SEC),
                )
                rest_calibration_summary["test_rest_windows"] = int(rest_windows_test.shape[0])
                if rest_windows_test.shape[0] > 0:
                    fused_rest_test, _ = evaluate_bank_on_raw_windows(
                        rest_windows_test,
                        member_artifacts,
                        sampling_rate=sampling_rate,
                        fusion_weights=bank_artifact["fusion_weights"],
                        fusion_method=fusion_method,
                        class_count=len(class_names),
                        probability_calibration=bank_artifact.get("probability_calibration"),
                    )
                    test_recommendation = select_reject_thresholds(
                        fused_test,
                        fused_rest_test,
                        target_rest_activation_rate=float(DEFAULT_REST_FALSE_ACTIVATION_TARGET),
                    )
                    rest_calibration_summary["test_phase_counts"] = {
                        str(phase_name): int(np.sum(rest_window_phases_test == phase_name))
                        for phase_name in np.unique(rest_window_phases_test)
                    }
                    rest_calibration_summary["test_threshold_eval"] = {
                        "control_detection_rate": float(test_recommendation["control_detection_rate"]),
                        "rest_false_activation_rate": float(test_recommendation["rest_false_activation_rate"]),
                        "rest_reject_rate": float(test_recommendation["rest_reject_rate"]),
                    }
    else:
        rest_calibration_summary["mode"] = "imagery_only_no_rest_segments"

    gate_model_artifact = bank_artifact.get("control_gate") if isinstance(bank_artifact.get("control_gate"), dict) else None
    artifact_model_artifact = (
        bank_artifact.get("artifact_rejector")
        if isinstance(bank_artifact.get("artifact_rejector"), dict)
        else None
    )
    continuous_variants = [
        {
            "name": "main_only",
            "use_gate": False,
            "use_artifact": False,
        }
    ]
    if gate_model_artifact is not None:
        continuous_variants.append(
            {
                "name": "main_plus_gate",
                "use_gate": True,
                "use_artifact": False,
            }
        )
    if artifact_model_artifact is not None:
        continuous_variants.append(
            {
                "name": "main_plus_artifact",
                "use_gate": False,
                "use_artifact": True,
            }
        )
    if gate_model_artifact is not None and artifact_model_artifact is not None:
        continuous_variants.append(
            {
                "name": "full_stack",
                "use_gate": True,
                "use_artifact": True,
            }
        )

    continuous_selection_records = list(continuous_records_by_split.get("val", []))
    continuous_selection_source = "val_split"
    if not continuous_selection_records:
        continuous_selection_records = list(continuous_records_by_split.get("test", []))
        continuous_selection_source = "test_split_fallback"
    if not continuous_selection_records:
        continuous_selection_records = list(continuous_records_by_split.get("all", []))
        continuous_selection_source = "all_records_fallback"

    continuous_eval_records = list(continuous_records_by_split.get("test", []))
    continuous_eval_source = "test_split"
    if not continuous_eval_records:
        continuous_eval_records = list(continuous_records_by_split.get("val", []))
        continuous_eval_source = "val_split_fallback"
    if not continuous_eval_records:
        continuous_eval_records = list(continuous_records_by_split.get("all", []))
        continuous_eval_source = "all_records_fallback"

    def _evaluate_continuous_variant(
        *,
        records: list[dict[str, object]],
        use_gate: bool,
        use_artifact: bool,
    ) -> dict[str, object]:
        return evaluate_continuous_online_like(
            continuous_records=records,
            main_member_artifacts=member_artifacts,
            main_fusion_weights=list(bank_artifact["fusion_weights"]),
            main_fusion_method=fusion_method,
            main_probability_calibration=bank_artifact.get("probability_calibration"),
            class_names=class_names,
            sampling_rate=sampling_rate,
            main_runtime=recommended_runtime,
            gate_member_artifacts=gate_member_artifacts if use_gate else None,
            gate_fusion_weights=(
                list(gate_model_artifact["fusion_weights"])
                if use_gate and gate_model_artifact is not None
                else None
            ),
            gate_fusion_method=(
                str(gate_model_artifact.get("fusion_method", fusion_method))
                if use_gate and gate_model_artifact is not None
                else fusion_method
            ),
            gate_probability_calibration=(
                dict(gate_model_artifact.get("probability_calibration") or {})
                if use_gate and gate_model_artifact is not None
                else None
            ),
            gate_runtime=gate_runtime if use_gate else None,
            artifact_member_artifacts=artifact_member_artifacts if use_artifact else None,
            artifact_fusion_weights=(
                list(artifact_model_artifact["fusion_weights"])
                if use_artifact and artifact_model_artifact is not None
                else None
            ),
            artifact_fusion_method=(
                str(artifact_model_artifact.get("fusion_method", fusion_method))
                if use_artifact and artifact_model_artifact is not None
                else fusion_method
            ),
            artifact_probability_calibration=(
                dict(artifact_model_artifact.get("probability_calibration") or {})
                if use_artifact and artifact_model_artifact is not None
                else None
            ),
            artifact_runtime=artifact_runtime if use_artifact else None,
        )

    continuous_variant_results: list[dict[str, object]] = []
    for variant in continuous_variants:
        use_gate = bool(variant["use_gate"])
        use_artifact = bool(variant["use_artifact"])
        selection_summary = _evaluate_continuous_variant(
            records=continuous_selection_records,
            use_gate=use_gate,
            use_artifact=use_artifact,
        )
        objective = compute_selection_objective_score(
            metrics=bank_artifact["metrics"],
            continuous_summary=selection_summary,
        )
        continuous_variant_results.append(
            {
                "name": str(variant["name"]),
                "use_gate": use_gate,
                "use_artifact": use_artifact,
                "selection_continuous": selection_summary,
                "selection_objective": objective,
            }
        )

    best_variant = max(
        continuous_variant_results,
        key=lambda item: (
            float(item["selection_objective"]["score"]),
            float((item["selection_continuous"] or {}).get("mi_prompt_accuracy", 0.0)),
            -float((item["selection_continuous"] or {}).get("no_control_false_activation_rate", 1.0)),
            int(bool(item.get("use_gate"))) + int(bool(item.get("use_artifact"))),
        ),
    )
    selected_variant_name = str(best_variant["name"])
    selected_use_gate = bool(best_variant.get("use_gate", False))
    selected_use_artifact = bool(best_variant.get("use_artifact", False))
    evaluation_summary = _evaluate_continuous_variant(
        records=continuous_eval_records,
        use_gate=selected_use_gate,
        use_artifact=selected_use_artifact,
    )
    continuous_summary = dict(evaluation_summary or best_variant.get("selection_continuous") or {})
    continuous_summary["selection_objective"] = dict(best_variant.get("selection_objective") or {})
    continuous_summary["selected_variant"] = selected_variant_name
    continuous_summary["selection_source"] = continuous_selection_source
    continuous_summary["evaluation_source"] = continuous_eval_source
    continuous_summary["selection_record_count"] = int(len(continuous_selection_records))
    continuous_summary["evaluation_record_count"] = int(len(continuous_eval_records))
    continuous_summary["split_record_counts"] = {
        "train": int(len(continuous_records_by_split.get("train", []))),
        "val": int(len(continuous_records_by_split.get("val", []))),
        "test": int(len(continuous_records_by_split.get("test", []))),
        "unassigned": int(len(continuous_records_by_split.get("unassigned", []))),
        "all": int(len(continuous_records_by_split.get("all", []))),
    }
    continuous_summary["selection_summary"] = dict(best_variant.get("selection_continuous") or {})
    continuous_summary["evaluated_variants"] = [
        {
            "name": str(item["name"]),
            "use_gate": bool(item["use_gate"]),
            "use_artifact": bool(item["use_artifact"]),
            "selection_objective_score": float(item["selection_objective"]["score"]),
            "selection_objective_components": dict(item["selection_objective"]["components"]),
            "selection_continuous_available": bool(item["selection_continuous"].get("available", False)),
            "selection_mi_prompt_accuracy": float(item["selection_continuous"].get("mi_prompt_accuracy", 0.0)),
            "selection_no_control_false_activation_rate": float(
                item["selection_continuous"].get("no_control_false_activation_rate", 0.0)
            ),
            "selection_evaluated_prompt_count": int(
                item["selection_continuous"].get("evaluated_prompt_count", 0)
            ),
        }
        for item in continuous_variant_results
    ]
    continuous_summary["decision_order"] = [
        "bad_window_rejector",
        "control_gate",
        "main_mi_classifier",
    ]

    if gate_model_artifact is not None and not selected_use_gate:
        gate_summary = dict(gate_summary)
        gate_summary["enabled"] = False
        gate_summary["selection_disabled"] = True
        gate_summary["selection_disable_reasons"] = [
            f"continuous_objective_selected_variant:{selected_variant_name}",
        ]
    if artifact_model_artifact is not None and not selected_use_artifact:
        artifact_rejector_summary = dict(artifact_rejector_summary)
        artifact_rejector_summary["enabled"] = False
        artifact_rejector_summary["selection_disabled"] = True
        artifact_rejector_summary["selection_disable_reasons"] = [
            f"continuous_objective_selected_variant:{selected_variant_name}",
        ]

    bank_artifact["control_gate"] = gate_model_artifact if selected_use_gate else None
    bank_artifact["artifact_rejector"] = artifact_model_artifact if selected_use_artifact else None

    bank_artifact["recommended_runtime"] = recommended_runtime
    bank_artifact["rest_calibration"] = rest_calibration_summary
    bank_artifact["control_gate_summary"] = gate_summary
    bank_artifact["artifact_rejector_summary"] = artifact_rejector_summary
    bank_artifact["continuous_online_like_eval"] = continuous_summary
    bank_artifact["selection_objective"] = dict(continuous_summary.get("selection_objective") or {})
    bank_artifact["selected_runtime_variant"] = selected_variant_name
    joblib.dump(bank_artifact, output_model_path)

    summary = {
        "created_at": bank_artifact["created_at"],
        "artifact_type": bank_artifact["artifact_type"],
        "dataset_root": str(dataset_root),
        "subject_filter": subject_filter,
        "split_strategy": split_strategy,
        "split_strategies": {
            "mi": split_strategy,
            "gate": gate_split_strategy,
            "artifact": artifact_split_strategy,
        },
        "split_assignments": {
            "mi": mi_split_assignments,
            "gate": gate_split_assignments,
            "artifact": artifact_split_assignments,
        },
        "train_groups": list(mi_split_assignments["train"]["groups"]),
        "val_groups": list(mi_split_assignments["val"]["groups"]),
        "test_groups": list(mi_split_assignments["test"]["groups"]),
        "train_sessions": list(mi_split_assignments["train"]["sessions"]),
        "val_sessions": list(mi_split_assignments["val"]["sessions"]),
        "test_sessions": list(mi_split_assignments["test"]["sessions"]),
        "candidate_names": list(selected_candidate_names),
        "gate_candidate_names": list(selected_gate_candidate_names),
        "artifact_candidate_names": list(selected_artifact_candidate_names),
        "torch_available": bool(TORCH_AVAILABLE),
        "deep_stage_pretrain_window_secs": [
            float(item) for item in (deep_stage_pretrain_window_secs or DEFAULT_DEEP_STAGE_PRETRAIN_WINDOW_SECS)
        ],
        "deep_stage_finetune_window_secs": [
            float(item) for item in (deep_stage_finetune_window_secs or DEFAULT_DEEP_STAGE_FINETUNE_WINDOW_SECS)
        ],
        "central_prior_alpha": float(central_prior_alpha),
        "central_aux_loss_weight": float(central_aux_loss_weight),
        "fusion_method": fusion_method,
        "fusion_weights": bank_artifact["fusion_weights"],
        "probability_calibration": bank_artifact.get("probability_calibration"),
        "preprocessing": dict(bank_artifact.get("preprocessing", {})),
        "preprocessing_fingerprint": str(bank_artifact.get("preprocessing_fingerprint", "")),
        "member_preprocessing_fingerprints": list(bank_artifact.get("member_preprocessing_fingerprints", [])),
        "window_secs": [float(item) for item in bank_artifact["window_secs"]],
        "window_offset_secs": [float(item) for item in selected_offset_secs],
        "metrics": bank_artifact["metrics"],
        "overall_acc": float(bank_artifact["metrics"]["bank_test_acc"]),
        "macro_f1": float(bank_artifact["metrics"]["bank_macro_f1"]),
        "kappa": float(bank_artifact["metrics"]["bank_kappa"]),
        "confusion_matrix": bank_artifact["metrics"]["bank_confusion_matrix"],
        "per_class_recall": bank_artifact["metrics"]["bank_per_class_recall"],
        "left_right_confusion_rate": float(bank_artifact["metrics"]["left_right_confusion_rate"]),
        "feet_tongue_confusion_rate": float(bank_artifact["metrics"]["feet_tongue_confusion_rate"]),
        "class_names": class_names,
        "display_class_names": [CLASS_NAME_TO_DISPLAY.get(name, name) for name in class_names],
        "channel_names": loaded["channel_names"],
        "sampling_rate": sampling_rate,
        "full_window_sec": full_window_sec,
        "reference_epoch_samples": mi_loaded.get("reference_epoch_samples"),
        "target_epoch_samples": mi_loaded.get("target_epoch_samples"),
        "dropped_short_runs": mi_loaded.get("dropped_short_runs", []),
        "total_trials": int(X_raw.shape[0]),
        "train_trials": int(train_idx.shape[0]),
        "val_trials": int(val_idx.shape[0]),
        "test_trials": int(test_idx.shape[0]),
        "label_distribution": label_distribution,
        "dataset_readiness": dataset_readiness,
        "enforce_readiness": bool(enforce_readiness),
        "allow_trial_level_fallback": bool(allow_trial_level_fallback),
        "recommended_runtime": recommended_runtime,
        "rest_calibration": rest_calibration_summary,
        "control_gate": gate_summary,
        "artifact_rejector": artifact_rejector_summary,
        "continuous_online_like_eval": continuous_summary,
        "selection_objective": dict(continuous_summary.get("selection_objective") or {}),
        "selected_runtime_variant": selected_variant_name,
        "continuous_prompt_count": int(continuous_loaded.get("prompt_count", 0)),
        "continuous_record_count": int(continuous_loaded.get("record_count", 0)),
        "continuous_split_records": {
            "train": int(len(continuous_records_by_split.get("train", []))),
            "val": int(len(continuous_records_by_split.get("val", []))),
            "test": int(len(continuous_records_by_split.get("test", []))),
            "unassigned": int(len(continuous_records_by_split.get("unassigned", []))),
        },
        "gate_dataset": {
            "available": bool(gate_available),
            "split_strategy": gate_split_strategy,
            "split_assignments": gate_split_assignments,
            "strict_split_block_reason": gate_split_block_reason,
            "train_count": int(gate_split_indices["train"].shape[0]) if gate_available else 0,
            "val_count": int(gate_split_indices["val"].shape[0]) if gate_available else 0,
            "test_count": int(gate_split_indices["test"].shape[0]) if gate_available else 0,
            "positive_count": int(X_gate_pos_all.shape[0]),
            "negative_count": int(X_gate_neg_all.shape[0] + X_gate_hard_neg_all.shape[0]),
            "negative_clean_count": int(X_gate_neg_all.shape[0]),
            "negative_hard_count": int(X_gate_hard_neg_all.shape[0]),
            "required_window_sec": float(gate_required_samples / float(sampling_rate)),
            "required_epoch_samples": int(gate_required_samples),
            "max_control_window_sec": float(int(gate_branch_loaded.get("pos_max_samples", 0)) / float(sampling_rate)),
            "max_clean_window_sec": float(int(gate_branch_loaded.get("neg_max_samples", 0)) / float(sampling_rate)),
            "max_hard_window_sec": float(
                int(gate_branch_loaded.get("hard_neg_max_samples", 0)) / float(sampling_rate)
            ),
            "negative_source_counts": {
                str(source): int(np.sum(np.asarray(gate_sources_all[gate_y_all == 0], dtype=object) == source))
                for source in np.unique(np.asarray(gate_sources_all[gate_y_all == 0], dtype=object))
            },
        },
        "artifact_dataset": {
            "available": bool(artifact_rejector_available),
            "split_strategy": artifact_split_strategy,
            "split_assignments": artifact_split_assignments,
            "strict_split_block_reason": artifact_split_block_reason,
            "train_count": int(artifact_split_indices["train"].shape[0]) if artifact_rejector_available else 0,
            "val_count": int(artifact_split_indices["val"].shape[0]) if artifact_rejector_available else 0,
            "test_count": int(artifact_split_indices["test"].shape[0]) if artifact_rejector_available else 0,
            "artifact_positive_count": int(X_artifact_all.shape[0]),
            "clean_negative_count": int(X_clean_negative_all.shape[0]),
            "required_window_sec": float(artifact_required_samples / float(sampling_rate)),
            "required_epoch_samples": int(artifact_required_samples),
            "max_artifact_window_sec": float(
                int(artifact_branch_loaded.get("artifact_max_samples", 0)) / float(sampling_rate)
            ),
            "max_clean_window_sec": float(
                int(artifact_branch_loaded.get("clean_negative_max_samples", 0)) / float(sampling_rate)
            ),
        },
        "source_sessions": loaded["session_paths"],
        "source_records": loaded["source_records"],
        "model_path": str(output_model_path),
        "member_models": member_summaries,
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    with output_model_path.with_suffix(".json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Train MI classifier from custom collected datasets.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Root directory of custom dataset files.")
    parser.add_argument("--subject", type=str, default="", help="Optional subject filter, e.g. 001 or sub-001.")
    parser.add_argument("--output-model", type=Path, default=DEFAULT_MODEL_PATH, help="Output realtime model artifact path.")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH, help="Output training summary JSON path.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--min-class-trials", type=int, default=5, help="Minimum accepted MI trials per class.")
    parser.add_argument(
        "--recommended-total-class-trials",
        type=int,
        default=DEFAULT_RECOMMENDED_TOTAL_CLASS_TRIALS,
        help="Advisory target for per-class accepted MI trials across all loaded runs (default: 30).",
    )
    parser.add_argument(
        "--recommended-run-class-trials",
        type=int,
        default=DEFAULT_RECOMMENDED_RUN_CLASS_TRIALS,
        help="Advisory target for per-class accepted MI trials inside one run (default: 8).",
    )
    parser.add_argument(
        "--enforce-readiness",
        action="store_true",
        help="Fail training when advisory readiness checks are not met.",
    )
    parser.add_argument(
        "--allow-trial-level-fallback",
        action="store_true",
        help=(
            "Allow trial-level stratified fallback when run/session split is not feasible. "
            "Disabled by default to enforce leakage-safe group/session holdout."
        ),
    )
    parser.add_argument(
        "--window-secs",
        type=str,
        default=",".join(str(item) for item in DEFAULT_WINDOW_SECS),
        help="Candidate training/inference window lengths in seconds, e.g. 2.0,2.5,3.0.",
    )
    parser.add_argument(
        "--window-offset-sec",
        type=float,
        default=None,
        help="Deprecated compatibility option for one offset from imagery onset (seconds).",
    )
    parser.add_argument(
        "--window-offset-secs",
        type=str,
        default="",
        help="Candidate window offsets in seconds, e.g. 0.5,0.75.",
    )
    parser.add_argument(
        "--fusion-method",
        type=str,
        default=DEFAULT_FUSION_METHOD,
        choices=["weighted_mean", "log_weighted_mean"],
        help="Fusion method used across member windows.",
    )
    parser.add_argument(
        "--fusion-weights",
        type=str,
        default="",
        help="Optional fusion weights, e.g. 0.45,0.35,0.20.",
    )
    parser.add_argument(
        "--candidate-names",
        type=str,
        default="",
        help=(
            "Optional candidate list such as central_fbcsp_lda,"
            "central_prior_dual_branch_fblight_tcn,riemann+lda"
        ),
    )
    parser.add_argument(
        "--gate-candidate-names",
        type=str,
        default="",
        help=(
            "Optional gate-only candidates. Defaults to central_gate_fblight,"
            "central_prior_gate_fblight when torch is available."
        ),
    )
    parser.add_argument(
        "--artifact-candidate-names",
        type=str,
        default="",
        help=(
            "Optional bad-window rejector candidates. Defaults to full8_fblight "
            "when torch is available."
        ),
    )
    parser.add_argument(
        "--torch-epochs",
        type=int,
        default=DEFAULT_TORCH_EPOCHS,
        help="Epoch count used by EEGNet-style candidates",
    )
    parser.add_argument(
        "--torch-batch-size",
        type=int,
        default=DEFAULT_TORCH_BATCH_SIZE,
        help="Batch size used by EEGNet-style candidates",
    )
    parser.add_argument(
        "--torch-learning-rate",
        type=float,
        default=DEFAULT_TORCH_LEARNING_RATE,
        help="Learning rate used by EEGNet-style candidates",
    )
    parser.add_argument(
        "--torch-weight-decay",
        type=float,
        default=DEFAULT_TORCH_WEIGHT_DECAY,
        help="Weight decay used by EEGNet-style candidates",
    )
    parser.add_argument(
        "--torch-patience",
        type=int,
        default=DEFAULT_TORCH_PATIENCE,
        help="Early-stop patience used by EEGNet-style candidates",
    )
    parser.add_argument(
        "--torch-validation-split",
        type=float,
        default=DEFAULT_TORCH_VALIDATION_SPLIT,
        help="Internal validation split used by EEGNet-style candidates",
    )
    parser.add_argument(
        "--torch-device",
        type=str,
        default="",
        help="Optional torch device override such as cpu or cuda",
    )
    parser.add_argument(
        "--deep-stage-pretrain-window-secs",
        type=str,
        default=",".join(str(item) for item in DEFAULT_DEEP_STAGE_PRETRAIN_WINDOW_SECS),
        help="Deep-model stage-1 windows in seconds, e.g. 2.5,2.0",
    )
    parser.add_argument(
        "--deep-stage-finetune-window-secs",
        type=str,
        default=",".join(str(item) for item in DEFAULT_DEEP_STAGE_FINETUNE_WINDOW_SECS),
        help="Deep-model stage-2 windows in seconds, e.g. 2.0,1.5",
    )
    parser.add_argument(
        "--central-prior-alpha",
        type=float,
        default=DEFAULT_CENTRAL_PRIOR_ALPHA,
        help="Dual-branch late-fusion alpha: logits = alpha*central + (1-alpha)*full",
    )
    parser.add_argument(
        "--central-aux-loss-weight",
        type=float,
        default=DEFAULT_CENTRAL_AUX_LOSS_WEIGHT,
        help="Auxiliary central-branch CE loss weight in dual-branch training",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    summary = train_custom_model(
        dataset_root=args.dataset_root.resolve(),
        subject_filter=normalize_subject_filter(args.subject),
        output_model_path=args.output_model.resolve(),
        report_path=args.report_path.resolve(),
        random_state=int(args.random_state),
        min_class_trials=int(args.min_class_trials),
        recommended_total_class_trials=int(args.recommended_total_class_trials),
        recommended_run_class_trials=int(args.recommended_run_class_trials),
        enforce_readiness=bool(args.enforce_readiness),
        allow_trial_level_fallback=bool(args.allow_trial_level_fallback),
        window_secs=parse_float_list(args.window_secs),
        window_offset_secs=(
            parse_float_list(args.window_offset_secs)
            if str(args.window_offset_secs).strip()
            else (
                [float(args.window_offset_sec)]
                if args.window_offset_sec is not None
                else default_window_offsets()
            )
        ),
        fusion_method=str(args.fusion_method),
        fusion_weights=parse_float_list(args.fusion_weights) or None,
        candidate_names=parse_string_list(args.candidate_names) or None,
        gate_candidate_names=parse_string_list(args.gate_candidate_names) or None,
        artifact_candidate_names=parse_string_list(args.artifact_candidate_names) or None,
        torch_epochs=int(args.torch_epochs),
        torch_batch_size=int(args.torch_batch_size),
        torch_learning_rate=float(args.torch_learning_rate),
        torch_weight_decay=float(args.torch_weight_decay),
        torch_patience=int(args.torch_patience),
        torch_validation_split=float(args.torch_validation_split),
        torch_device=str(args.torch_device).strip() or None,
        deep_stage_pretrain_window_secs=parse_float_list(args.deep_stage_pretrain_window_secs)
        or list(DEFAULT_DEEP_STAGE_PRETRAIN_WINDOW_SECS),
        deep_stage_finetune_window_secs=parse_float_list(args.deep_stage_finetune_window_secs)
        or list(DEFAULT_DEEP_STAGE_FINETUNE_WINDOW_SECS),
        central_prior_alpha=float(args.central_prior_alpha),
        central_aux_loss_weight=float(args.central_aux_loss_weight),
    )
    print(f"candidate_names={summary['candidate_names']}")
    print(f"gate_candidate_names={summary['gate_candidate_names']}")
    print(f"artifact_candidate_names={summary['artifact_candidate_names']}")
    print(f"torch_available={summary['torch_available']}")
    print(f"deep_stage_pretrain_window_secs={summary['deep_stage_pretrain_window_secs']}")
    print(f"deep_stage_finetune_window_secs={summary['deep_stage_finetune_window_secs']}")
    print(
        "central_prior="
        f"alpha:{float(summary['central_prior_alpha']):.2f},"
        f"aux_loss_weight:{float(summary['central_aux_loss_weight']):.2f}"
    )
    print(f"window_secs={summary['window_secs']}")
    print(f"bank_test_acc={summary['metrics']['bank_test_acc']:.4f}")
    print(f"bank_macro_f1={summary['metrics']['bank_macro_f1']:.4f}")
    print(f"bank_kappa={summary['metrics']['bank_kappa']:.4f}")
    print(f"left_right_confusion_rate={float(summary['metrics']['left_right_confusion_rate']):.4f}")
    print(f"feet_tongue_confusion_rate={float(summary['metrics']['feet_tongue_confusion_rate']):.4f}")
    print(f"confusion_matrix={summary['metrics']['bank_confusion_matrix']}")
    control_gate_summary = dict(summary.get("control_gate") or {})
    artifact_rejector_summary = dict(summary.get("artifact_rejector") or {})
    continuous_summary = dict(summary.get("continuous_online_like_eval") or {})
    dataset_readiness = dict(summary.get("dataset_readiness") or {})
    print(f"dataset_readiness_ready={bool(dataset_readiness.get('ready_for_stable_comparison', False))}")
    if dataset_readiness.get("warnings"):
        print(f"dataset_readiness_warnings={dataset_readiness['warnings']}")
    print(f"control_gate_enabled={bool(control_gate_summary.get('enabled', False))}")
    print(f"artifact_rejector_enabled={bool(artifact_rejector_summary.get('enabled', False))}")
    print(f"continuous_eval_available={bool(continuous_summary.get('available', False))}")
    if summary.get("recommended_runtime"):
        recommended_runtime = summary["recommended_runtime"]
        print(
            "recommended_thresholds="
            f"conf:{float(recommended_runtime['confidence_threshold']):.3f},"
            f"margin:{float(recommended_runtime['margin_threshold']):.3f}"
        )
    else:
        print(
            "recommended_thresholds=unavailable "
            f"(mode={summary['rest_calibration']['mode']}; keep manual realtime thresholds)"
        )
    if control_gate_summary.get("enabled") and control_gate_summary.get("recommended_runtime"):
        gate_runtime = control_gate_summary["recommended_runtime"]
        print(
            "recommended_gate_thresholds="
            f"conf:{float(gate_runtime['confidence_threshold']):.3f},"
            f"margin:{float(gate_runtime['margin_threshold']):.3f}"
        )
    elif control_gate_summary:
        print("recommended_gate_thresholds=unavailable")
    if artifact_rejector_summary.get("enabled") and artifact_rejector_summary.get("recommended_runtime"):
        artifact_runtime = artifact_rejector_summary["recommended_runtime"]
        print(
            "recommended_artifact_thresholds="
            f"conf:{float(artifact_runtime['confidence_threshold']):.3f},"
            f"margin:{float(artifact_runtime['margin_threshold']):.3f}"
        )
    elif artifact_rejector_summary:
        print("recommended_artifact_thresholds=unavailable")
    if continuous_summary:
        print(
            "continuous_online_like="
            f"evaluated:{int(continuous_summary.get('evaluated_prompt_count', 0))},"
            f"mi_acc:{float(continuous_summary.get('mi_prompt_accuracy', 0.0)):.4f},"
            f"no_control_fa:{float(continuous_summary.get('no_control_false_activation_rate', 0.0)):.4f}"
        )
    selection_objective = dict(summary.get("selection_objective") or {})
    if selection_objective:
        print(
            "selection_objective="
            f"score:{float(selection_objective.get('score', 0.0)):.4f},"
            f"selected_variant:{str(continuous_summary.get('selected_variant', 'main_only'))}"
        )
    print(f"model_path={summary['model_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

