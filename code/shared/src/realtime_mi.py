"""Realtime motor-imagery helpers built on top of the optimized offline pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import resample
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split

from src.models import apply_probability_calibration, fit_probability_calibration, predict_probability_matrix
from src.preprocessing import preprocess


BCI_IV_2A_CHANNEL_NAMES = [
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
]

DEFAULT_REALTIME_CHANNEL_NAMES = [
    "FC3",
    "FCz",
    "FC4",
    "C3",
    "Cz",
    "C4",
    "CP3",
    "CP4",
]

DISPLAY_NAME_MAP = {
    "left_hand": "LEFT HAND",
    "right_hand": "RIGHT HAND",
    "feet": "FEET",
    "tongue": "TONGUE",
}
REQUIRED_PREPROCESSING_KEYS = (
    "bandpass",
    "optimized_input_bandpass",
    "notch",
    "apply_car",
    "standardize",
    "epoch_window",
    "window_offset_sec",
)

LEGACY_PREPROCESSING_DEFAULTS = {
    "bandpass": [4.0, 40.0],
    "optimized_input_bandpass": [4.0, 40.0],
    "notch": 50.0,
    "apply_car": True,
    "standardize": False,
}


def _as_float_pair(value: object, *, field_name: str) -> list[float]:
    """Validate and normalize two-element numeric lists used by preprocessing config."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Artifact preprocessing field '{field_name}' must be a length-2 list.")
    low = float(value[0])
    high = float(value[1])
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"Artifact preprocessing field '{field_name}' must be finite numbers.")
    if low >= high:
        raise ValueError(f"Artifact preprocessing field '{field_name}' must satisfy low < high.")
    return [low, high]


def _preprocessing_fingerprint(preprocessing: dict[str, object]) -> str:
    """Build a deterministic fingerprint for artifact preprocessing payloads."""
    serialized = json.dumps(preprocessing, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _upgrade_legacy_preprocessing_config(artifact: dict, preproc_cfg: object) -> dict[str, object] | None:
    """Backfill legacy realtime artifacts that predate the current preprocessing schema."""
    if not isinstance(preproc_cfg, dict):
        return None

    upgraded = dict(preproc_cfg)
    window_sec = artifact.get("window_sec")
    raw_offset_sec = artifact.get("window_offset_sec", 0.0)
    raw_offset_secs_used = artifact.get("window_offset_secs_used")

    if "bandpass" not in upgraded:
        upgraded["bandpass"] = list(LEGACY_PREPROCESSING_DEFAULTS["bandpass"])
    if "optimized_input_bandpass" not in upgraded:
        upgraded["optimized_input_bandpass"] = list(LEGACY_PREPROCESSING_DEFAULTS["optimized_input_bandpass"])
    if "notch" not in upgraded:
        upgraded["notch"] = LEGACY_PREPROCESSING_DEFAULTS["notch"]
    if "apply_car" not in upgraded:
        upgraded["apply_car"] = LEGACY_PREPROCESSING_DEFAULTS["apply_car"]
    if "standardize" not in upgraded:
        upgraded["standardize"] = LEGACY_PREPROCESSING_DEFAULTS["standardize"]
    if "epoch_window" not in upgraded and window_sec is not None:
        upgraded["epoch_window"] = [0.0, float(window_sec)]
    if "window_offset_sec" not in upgraded:
        upgraded["window_offset_sec"] = float(raw_offset_sec)
    if "window_offset_secs_used" not in upgraded:
        if isinstance(raw_offset_secs_used, (list, tuple)) and raw_offset_secs_used:
            upgraded["window_offset_secs_used"] = [float(item) for item in raw_offset_secs_used]
        else:
            upgraded["window_offset_secs_used"] = [float(upgraded.get("window_offset_sec", 0.0))]
    return upgraded


def _resolve_required_preprocessing_config(artifact: dict) -> dict[str, object]:
    """Return a validated preprocessing config; raise when mandatory fields are missing."""
    preproc_cfg = _upgrade_legacy_preprocessing_config(artifact, artifact.get("preprocessing"))
    if not isinstance(preproc_cfg, dict):
        raise ValueError("Realtime artifact is missing dict field 'preprocessing'.")
    missing = [field for field in REQUIRED_PREPROCESSING_KEYS if field not in preproc_cfg]
    if missing:
        raise ValueError(
            "Realtime artifact preprocessing is incomplete. Missing fields: "
            f"{missing}. Re-export model from latest training pipeline."
        )

    normalized = {
        "bandpass": _as_float_pair(preproc_cfg["bandpass"], field_name="bandpass"),
        "optimized_input_bandpass": _as_float_pair(
            preproc_cfg["optimized_input_bandpass"],
            field_name="optimized_input_bandpass",
        ),
        "notch": (None if preproc_cfg["notch"] is None else float(preproc_cfg["notch"])),
        "apply_car": bool(preproc_cfg["apply_car"]),
        "standardize": bool(preproc_cfg["standardize"]),
        "epoch_window": _as_float_pair(preproc_cfg["epoch_window"], field_name="epoch_window"),
        "window_offset_sec": float(preproc_cfg["window_offset_sec"]),
        "window_offset_secs_used": [
            float(item)
            for item in (
                preproc_cfg.get("window_offset_secs_used")
                if isinstance(preproc_cfg.get("window_offset_secs_used"), (list, tuple))
                else [float(preproc_cfg["window_offset_sec"])]
            )
        ],
    }
    if normalized["notch"] is not None and normalized["notch"] <= 0.0:
        raise ValueError("Artifact preprocessing field 'notch' must be positive when provided.")
    artifact["preprocessing"] = dict(normalized)
    artifact["preprocessing_fingerprint"] = _preprocessing_fingerprint(normalized)
    return normalized


def _preprocess_trials_for_runtime(
    X: np.ndarray,
    *,
    sampling_rate: float,
    optimized_input_bandpass: list[float] | tuple[float, float],
    notch: float | None,
    apply_car: bool,
    standardize_data: bool,
) -> np.ndarray:
    """Run the optimized realtime preprocessing without importing training-only modules."""
    return preprocess(
        X,
        fs=float(sampling_rate),
        bandpass=optimized_input_bandpass,
        notch=notch,
        apply_car=apply_car,
        standardize_data=standardize_data,
    )


def parse_channel_names(raw_value: str | list[str] | None) -> list[str]:
    """Parse a comma-separated channel-name list."""
    if raw_value is None:
        return list(DEFAULT_REALTIME_CHANNEL_NAMES)
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]


def channel_indices_from_names(channel_names: list[str]) -> list[int]:
    """Map channel names to the BCI IV 2a channel indices."""
    missing = [name for name in channel_names if name not in BCI_IV_2A_CHANNEL_NAMES]
    if missing:
        raise ValueError(f"Unknown channel names: {missing}")
    return [BCI_IV_2A_CHANNEL_NAMES.index(name) for name in channel_names]


def class_display_name(class_name: str) -> str:
    """Convert a raw class label into a UI-friendly label."""
    return DISPLAY_NAME_MAP.get(class_name, class_name.replace("_", " ").upper())


def _flatten_realtime_artifact_members(artifact: dict) -> list[dict]:
    """Return single-window member artifacts from either a single artifact or a bank artifact."""
    artifact_type = str(artifact.get("artifact_type", "single_window"))
    if artifact_type == "multi_window_bank":
        members = artifact.get("members", [])
        if not members:
            raise ValueError("Multi-window artifact does not contain any members.")
        return [dict(member) for member in members]
    return [dict(artifact)]


def _normalize_fusion_weights(raw_weights: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    """Normalize fusion weights and fall back to uniform weights when needed."""
    weights = np.asarray(raw_weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("Fusion weights must be a non-empty 1D sequence.")
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.clip(weights, a_min=0.0, a_max=None)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(weights.shape[0], 1.0 / weights.shape[0], dtype=np.float64)
    return weights / total


def _fuse_probability_vectors(
    probability_vectors: list[np.ndarray],
    fusion_weights: np.ndarray,
    *,
    method: str,
) -> np.ndarray:
    """Fuse multiple probability vectors into one class-probability vector."""
    if not probability_vectors:
        raise ValueError("At least one probability vector is required for fusion.")

    probabilities = np.asarray(probability_vectors, dtype=np.float64)
    weights = _normalize_fusion_weights(fusion_weights)
    if probabilities.ndim != 2:
        raise ValueError("Probability vectors must form a 2D matrix.")
    if probabilities.shape[0] != weights.shape[0]:
        raise ValueError("Fusion weights do not match the number of probability vectors.")

    normalized_method = str(method).strip().lower()
    if normalized_method in {"weighted_mean", "weighted_average", "mean"}:
        fused = np.sum(probabilities * weights[:, np.newaxis], axis=0)
        total = float(np.sum(fused))
        if total > 0.0:
            return fused / total
        return np.full(probabilities.shape[1], 1.0 / probabilities.shape[1], dtype=np.float64)

    if normalized_method in {"log_weighted_mean", "log_average", "log_mean"}:
        clipped = np.clip(probabilities, 1e-8, 1.0)
        log_scores = np.sum(weights[:, np.newaxis] * np.log(clipped), axis=0)
        shifted = log_scores - np.max(log_scores)
        exponentiated = np.exp(shifted)
        total = float(np.sum(exponentiated))
        if total > 0.0:
            return exponentiated / total
        return np.full(probabilities.shape[1], 1.0 / probabilities.shape[1], dtype=np.float64)

    raise ValueError(f"Unsupported fusion method: {method}")


def build_realtime_artifact_bank(
    artifacts: list[dict],
    *,
    fusion_weights: list[float] | tuple[float, ...] | np.ndarray | None = None,
    fusion_method: str = "log_weighted_mean",
) -> dict:
    """Combine multiple single-window artifacts into one multi-window realtime bank."""
    if not artifacts:
        raise ValueError("At least one artifact is required to build a model bank.")

    members = []
    for artifact in artifacts:
        members.extend(_flatten_realtime_artifact_members(artifact))
    if not members:
        raise ValueError("No single-window members were found in the provided artifacts.")
    for member in members:
        normalized_preprocessing = _resolve_required_preprocessing_config(member)
        member["preprocessing"] = normalized_preprocessing
        member["preprocessing_fingerprint"] = _preprocessing_fingerprint(normalized_preprocessing)

    first = members[0]
    required_common_fields = [
        "class_names",
        "display_class_names",
        "channel_names",
        "channel_indices",
        "sampling_rate",
        "model_type",
    ]
    for index, member in enumerate(members[1:], start=1):
        for field in required_common_fields:
            if member.get(field) != first.get(field):
                raise ValueError(
                    f"Cannot combine artifact member #{index}: field '{field}' does not match the first member."
                )

    if fusion_weights is None:
        sorted_members = sorted(members, key=lambda member: float(member["window_sec"]))
        descending = np.arange(len(sorted_members), 0, -1, dtype=np.float64)
        normalized_weights = _normalize_fusion_weights(descending)
    else:
        if len(fusion_weights) != len(members):
            raise ValueError(
                f"Fusion weight count mismatch: expected {len(members)}, got {len(fusion_weights)}."
            )
        weighted_members = list(zip(members, fusion_weights))
        weighted_members.sort(key=lambda item: float(item[0]["window_sec"]))
        sorted_members = [member for member, _ in weighted_members]
        normalized_weights = _normalize_fusion_weights([weight for _, weight in weighted_members])

    window_secs = [float(member["window_sec"]) for member in sorted_members]
    member_pipelines = [str(member.get("selected_pipeline", "unknown")) for member in sorted_members]
    window_label = ", ".join(f"{window_sec:.2f}s" for window_sec in window_secs)

    return {
        "artifact_type": "multi_window_bank",
        "pipeline": None,
        "members": sorted_members,
        "member_selected_pipelines": member_pipelines,
        "selected_pipeline": f"{fusion_method}[{window_label}]",
        "model_type": first["model_type"],
        "class_names": list(first["class_names"]),
        "display_class_names": list(first["display_class_names"]),
        "channel_names": list(first["channel_names"]),
        "channel_indices": list(first["channel_indices"]),
        "sampling_rate": float(first["sampling_rate"]),
        "window_sec": float(max(window_secs)),
        "min_window_sec": float(min(window_secs)),
        "window_secs": window_secs,
        "fusion_weights": normalized_weights.astype(np.float64).tolist(),
        "fusion_method": str(fusion_method),
        "preprocessing": dict(first.get("preprocessing", {})),
        "preprocessing_fingerprint": str(first.get("preprocessing_fingerprint", "")),
        "member_preprocessing_fingerprints": [str(member.get("preprocessing_fingerprint", "")) for member in sorted_members],
        "subject_id": first.get("subject_id"),
        "metrics": {
            "members": [
                {
                    "window_sec": float(member["window_sec"]),
                    "selected_pipeline": str(member.get("selected_pipeline", "unknown")),
                    "metrics": dict(member.get("metrics", {})),
                }
                for member in sorted_members
            ]
        },
        "config_path": first.get("config_path"),
    }


def _decision_to_probabilities(
    model,
    X: np.ndarray,
    n_classes: int,
    probability_calibration: dict[str, object] | None = None,
) -> np.ndarray:
    """Convert model outputs to comparable confidence scores."""
    probabilities = predict_probability_matrix(
        model,
        X,
        n_classes,
        probability_calibration=probability_calibration,
    )
    return np.asarray(probabilities[0], dtype=np.float64)


def _preprocess_single_window(raw_window: np.ndarray, artifact: dict, live_sampling_rate: float) -> np.ndarray:
    """Resample and preprocess one realtime EEG window."""
    model_sampling_rate = float(artifact["sampling_rate"])
    window_sec = float(artifact["window_sec"])
    expected_samples = int(round(window_sec * model_sampling_rate))

    working = np.asarray(raw_window, dtype=np.float32)
    if working.ndim != 2:
        raise ValueError("Realtime window must have shape (channels, samples).")
    if working.shape[0] <= 0 or working.shape[1] <= 0:
        raise ValueError("Realtime window cannot be empty.")
    if not np.all(np.isfinite(working)):
        working = np.nan_to_num(working, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if abs(float(live_sampling_rate) - model_sampling_rate) > 1e-6:
        working = resample(working, expected_samples, axis=-1).astype(np.float32)
        effective_fs = model_sampling_rate
    else:
        if working.shape[1] != expected_samples:
            working = resample(working, expected_samples, axis=-1).astype(np.float32)
        effective_fs = model_sampling_rate

    preproc_cfg = _resolve_required_preprocessing_config(artifact)
    processed = _preprocess_trials_for_runtime(
        working[np.newaxis, ...],
        sampling_rate=float(effective_fs),
        optimized_input_bandpass=list(preproc_cfg["optimized_input_bandpass"]),
        notch=preproc_cfg["notch"],
        apply_car=bool(preproc_cfg["apply_car"]),
        standardize_data=bool(preproc_cfg["standardize"]),
    )
    return processed


def fit_realtime_model(
    *,
    config_path: str | Path = "config.yaml",
    subject_id: int = 1,
    channel_names: list[str] | None = None,
    output_path: str | Path | None = None,
) -> dict:
    """Train and optionally persist a realtime-compatible MI classifier."""
    from src.train import (
        build_optimized_model_candidates,
        ensure_dataset_available,
        load_config,
        load_subject_data,
        seed_everything,
    )

    config_path = Path(config_path).resolve()
    project_root = config_path.parent
    config = load_config(config_path)
    seed_everything(int(config["training"]["random_state"]))
    config["model"]["type"] = "optimized"

    channel_names = channel_names or list(DEFAULT_REALTIME_CHANNEL_NAMES)
    channel_indices = channel_indices_from_names(channel_names)

    dataset_dir = ensure_dataset_available(config, project_root)
    X_train, y_train, X_test, y_test = load_subject_data(config, dataset_dir, subject_id)

    X_train = X_train[:, channel_indices, :]
    X_test = X_test[:, channel_indices, :]

    optimized_input_bandpass = [
        float(item) for item in list(config["model"].get("optimized_input_bandpass", [4.0, 40.0]))
    ]
    notch = config["preprocessing"].get("notch")
    X_train = _preprocess_trials_for_runtime(
        X_train,
        sampling_rate=float(config["dataset"]["sampling_rate"]),
        optimized_input_bandpass=optimized_input_bandpass,
        notch=None if notch is None else float(notch),
        apply_car=bool(config["preprocessing"].get("apply_car", True)),
        standardize_data=bool(config["preprocessing"].get("standardize", False)),
    )
    X_test = _preprocess_trials_for_runtime(
        X_test,
        sampling_rate=float(config["dataset"]["sampling_rate"]),
        optimized_input_bandpass=optimized_input_bandpass,
        notch=None if notch is None else float(notch),
        apply_car=bool(config["preprocessing"].get("apply_car", True)),
        standardize_data=bool(config["preprocessing"].get("standardize", False)),
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=float(config["training"]["validation_split"]),
        random_state=int(config["training"]["random_state"]),
        stratify=y_train,
    )

    best_name = None
    best_val_acc = -1.0
    best_pipeline = None
    best_probability_calibration = None

    for name, pipeline in build_optimized_model_candidates(config).items():
        pipeline.fit(X_tr, y_tr)
        raw_val_probabilities = predict_probability_matrix(
            pipeline,
            X_val,
            len(config["dataset"]["class_names"]),
        )
        probability_calibration = fit_probability_calibration(
            raw_val_probabilities,
            y_val,
            role="realtime_main_member",
            selection_source="realtime_selection_val",
        )
        val_probabilities = apply_probability_calibration(raw_val_probabilities, probability_calibration)
        val_predictions = np.argmax(val_probabilities, axis=1)
        val_acc = accuracy_score(y_val, val_predictions)
        if val_acc > best_val_acc:
            best_name = name
            best_val_acc = val_acc
            best_pipeline = pipeline
            best_probability_calibration = dict(probability_calibration)

    if best_pipeline is None or best_probability_calibration is None:
        raise RuntimeError("Failed to select a realtime pipeline.")

    best_pipeline.fit(X_train, y_train)
    test_probabilities = predict_probability_matrix(
        best_pipeline,
        X_test,
        len(config["dataset"]["class_names"]),
        probability_calibration=best_probability_calibration,
    )
    test_predictions = np.argmax(test_probabilities, axis=1)

    class_names = list(config["dataset"]["class_names"])
    epoch_window = [float(item) for item in list(config["preprocessing"]["epoch_window"])]
    preprocessing_payload = {
        "bandpass": [float(item) for item in list(config["preprocessing"]["bandpass"])],
        "optimized_input_bandpass": [float(item) for item in list(config["model"]["optimized_input_bandpass"])],
        "notch": None if config["preprocessing"]["notch"] is None else float(config["preprocessing"]["notch"]),
        "apply_car": bool(config["preprocessing"]["apply_car"]),
        "standardize": bool(config["preprocessing"]["standardize"]),
        "epoch_window": epoch_window,
        "window_offset_sec": 0.0,
        "window_offset_secs_used": [0.0],
    }
    artifact = {
        "pipeline": best_pipeline,
        "probability_calibration": dict(best_probability_calibration),
        "subject_id": int(subject_id),
        "selected_pipeline": best_name,
        "model_type": "optimized",
        "class_names": class_names,
        "display_class_names": [class_display_name(name) for name in class_names],
        "channel_names": channel_names,
        "channel_indices": channel_indices,
        "sampling_rate": float(config["dataset"]["sampling_rate"]),
        "window_sec": float(epoch_window[1] - epoch_window[0]),
        "preprocessing": dict(preprocessing_payload),
        "preprocessing_fingerprint": _preprocessing_fingerprint(preprocessing_payload),
        "metrics": {
            "val_acc": float(best_val_acc),
            "test_acc": float(accuracy_score(y_test, test_predictions)),
            "kappa": float(cohen_kappa_score(y_test, test_predictions)),
        },
        "config_path": str(config_path),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, output_path)

        summary_path = output_path.with_suffix(".json")
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "subject_id": artifact["subject_id"],
                    "selected_pipeline": artifact["selected_pipeline"],
                    "probability_calibration": artifact.get("probability_calibration"),
                    "channel_names": artifact["channel_names"],
                    "class_names": artifact["class_names"],
                    "metrics": artifact["metrics"],
                },
                file,
                indent=2,
            )

    return artifact


def load_realtime_model(
    model_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    fusion_weights: list[float] | tuple[float, ...] | np.ndarray | None = None,
    fusion_method: str = "log_weighted_mean",
) -> dict:
    """Load one or more saved realtime MI artifacts."""
    if isinstance(model_path, (list, tuple)):
        loaded_artifacts = [joblib.load(Path(path)) for path in model_path]
        return build_realtime_artifact_bank(
            loaded_artifacts,
            fusion_weights=fusion_weights,
            fusion_method=fusion_method,
        )

    artifact = joblib.load(Path(model_path))
    if str(artifact.get("artifact_type", "single_window")) == "multi_window_bank":
        rebuilt = build_realtime_artifact_bank(
            [artifact],
            fusion_weights=artifact.get("fusion_weights") if fusion_weights is None else fusion_weights,
            fusion_method=str(artifact.get("fusion_method", fusion_method)),
        )
        for key, value in artifact.items():
            if key not in rebuilt:
                rebuilt[key] = value
        return rebuilt
    _resolve_required_preprocessing_config(artifact)
    return artifact


class RealtimeMIPredictor:
    """Sliding-window realtime predictor with multi-window fusion, smoothing, and hysteresis."""

    def __init__(
        self,
        artifact: dict,
        *,
        history_len: int = 5,
        confidence_threshold: float = 0.45,
        gate_confidence_threshold: float | None = None,
        probability_smoothing: float = 0.35,
        margin_threshold: float = 0.12,
        gate_margin_threshold: float | None = None,
        artifact_rejector_confidence_threshold: float | None = None,
        artifact_rejector_margin_threshold: float | None = None,
        switch_delta: float = 0.08,
        hold_confidence_drop: float = 0.10,
        hold_margin_drop: float = 0.04,
        release_windows: int = 2,
        min_stable_windows: int = 2,
        flatline_std_threshold: float = 1e-7,
        dominant_channel_ratio_threshold: float = 8.0,
        max_bad_channels: int = 1,
        artifact_freeze_windows: int = 2,
        gate_release_windows: int | None = None,
    ) -> None:
        self.artifact = artifact
        self.model_entries = self._prepare_model_entries(artifact)
        self.pipeline = self.model_entries[-1]["pipeline"]
        self.fusion_method = str(artifact.get("fusion_method", "weighted_mean"))
        self.control_gate_artifact = artifact.get("control_gate")
        self.gate_model_entries = (
            self._prepare_model_entries(self.control_gate_artifact)
            if isinstance(self.control_gate_artifact, dict)
            else []
        )
        self.gate_enabled = bool(self.gate_model_entries)
        self.artifact_rejector_artifact = artifact.get("artifact_rejector")
        self.artifact_rejector_model_entries = (
            self._prepare_model_entries(self.artifact_rejector_artifact)
            if isinstance(self.artifact_rejector_artifact, dict)
            else []
        )
        self.artifact_rejector_enabled = bool(self.artifact_rejector_model_entries)
        self.confirmation_windows = max(1, int(history_len))
        self.confidence_threshold = float(min(max(confidence_threshold, 0.0), 1.0))
        self.probability_smoothing = float(min(max(probability_smoothing, 0.0), 1.0))
        self.margin_threshold = float(max(margin_threshold, 0.0))
        gate_runtime = (
            dict(self.control_gate_artifact.get("recommended_runtime") or {})
            if self.gate_enabled
            else {}
        )
        resolved_gate_confidence = gate_runtime.get("confidence_threshold", 0.60)
        resolved_gate_margin = gate_runtime.get("margin_threshold", 0.00)
        self.gate_confidence_threshold = float(
            min(
                max(
                    resolved_gate_confidence if gate_confidence_threshold is None else gate_confidence_threshold,
                    0.0,
                ),
                1.0,
            )
        )
        self.gate_margin_threshold = float(
            max(resolved_gate_margin if gate_margin_threshold is None else gate_margin_threshold, 0.0)
        )
        artifact_runtime = (
            dict(self.artifact_rejector_artifact.get("recommended_runtime") or {})
            if self.artifact_rejector_enabled
            else {}
        )
        resolved_artifact_confidence = artifact_runtime.get("confidence_threshold", 0.60)
        resolved_artifact_margin = artifact_runtime.get("margin_threshold", 0.00)
        self.artifact_rejector_confidence_threshold = float(
            min(
                max(
                    (
                        resolved_artifact_confidence
                        if artifact_rejector_confidence_threshold is None
                        else artifact_rejector_confidence_threshold
                    ),
                    0.0,
                ),
                1.0,
            )
        )
        self.artifact_rejector_margin_threshold = float(
            max(
                resolved_artifact_margin
                if artifact_rejector_margin_threshold is None
                else artifact_rejector_margin_threshold,
                0.0,
            )
        )
        self.switch_delta = float(max(switch_delta, 0.0))
        self.hold_confidence_threshold = float(max(self.confidence_threshold - hold_confidence_drop, 0.0))
        self.hold_margin_threshold = float(max(self.margin_threshold - hold_margin_drop, 0.0))
        self.release_windows = max(1, int(release_windows))
        self.gate_release_windows = max(
            1,
            int(self.release_windows if gate_release_windows is None else gate_release_windows),
        )
        self.min_stable_windows = max(1, int(min_stable_windows))
        self.flatline_std_threshold = float(max(flatline_std_threshold, 0.0))
        self.dominant_channel_ratio_threshold = float(max(dominant_channel_ratio_threshold, 1.0))
        self.max_bad_channels = max(0, int(max_bad_channels))
        self.artifact_freeze_windows = max(1, int(artifact_freeze_windows))
        self.minimum_window_samples = min(entry["window_samples"] for entry in self.model_entries)
        self.maximum_window_samples = max(entry["window_samples"] for entry in self.model_entries)
        self.minimum_window_sec = min(entry["window_sec"] for entry in self.model_entries)
        self.maximum_window_sec = max(entry["window_sec"] for entry in self.model_entries)
        self.minimum_guided_window_samples = min(entry["guided_end_samples"] for entry in self.model_entries)
        self.maximum_guided_window_samples = max(entry["guided_end_samples"] for entry in self.model_entries)
        self.minimum_guided_window_sec = min(entry["guided_end_sec"] for entry in self.model_entries)
        self.maximum_guided_window_sec = max(entry["guided_end_sec"] for entry in self.model_entries)
        self.stream_buffer_samples = max(self.maximum_window_samples, self.maximum_guided_window_samples)
        self.window_samples = self.maximum_window_samples
        self._smoothed_probabilities: np.ndarray | None = None
        self._gate_smoothed_probabilities: np.ndarray | None = None
        self._artifact_rejector_smoothed_probabilities: np.ndarray | None = None
        self._stable_prediction: int | None = None
        self._pending_prediction: int | None = None
        self._pending_count = 0
        self._release_count = 0
        self._gate_reject_count = 0
        self._stable_age = 0
        self._artifact_count = 0
        if self.window_samples <= 0:
            raise ValueError("window_sec and sampling_rate produce an invalid window length.")

    @property
    def expected_channel_count(self) -> int:
        return len(self.artifact["channel_names"])

    @property
    def minimum_required_samples(self) -> int:
        return int(self.minimum_window_samples)

    @property
    def maximum_required_samples(self) -> int:
        return int(self.maximum_window_samples)

    @property
    def minimum_guided_required_samples(self) -> int:
        return int(self.minimum_guided_window_samples)

    @property
    def maximum_guided_required_samples(self) -> int:
        return int(self.maximum_guided_window_samples)

    def _prepare_model_entries(self, artifact: dict) -> list[dict[str, object]]:
        """Normalize either a single artifact or a bank artifact into fusion-ready entries."""
        members = _flatten_realtime_artifact_members(artifact)
        raw_weights = artifact.get("fusion_weights")
        if raw_weights is None:
            raw_weights = np.ones(len(members), dtype=np.float64)
        normalized_weights = _normalize_fusion_weights(raw_weights)

        entries = []
        for member, weight in zip(members, normalized_weights):
            window_sec = float(member["window_sec"])
            window_offset_sec = float(member.get("window_offset_sec", 0.0))
            sampling_rate = float(member["sampling_rate"])
            window_samples = int(round(window_sec * sampling_rate))
            guided_end_samples = int(round((window_offset_sec + window_sec) * sampling_rate))
            if window_samples <= 0:
                raise ValueError(f"Invalid member window size: {window_sec} seconds.")
            if guided_end_samples <= 0:
                raise ValueError(
                    f"Invalid guided member timing: offset={window_offset_sec}, window={window_sec}."
                )
            entries.append(
                {
                    "artifact": member,
                    "pipeline": member["pipeline"],
                    "window_sec": window_sec,
                    "window_offset_sec": window_offset_sec,
                    "window_samples": window_samples,
                    "guided_end_sec": float(window_offset_sec + window_sec),
                    "guided_end_samples": guided_end_samples,
                    "weight": float(weight),
                    "selected_pipeline": str(member.get("selected_pipeline", "unknown")),
                }
            )
        return sorted(entries, key=lambda entry: float(entry["window_sec"]))

    def reset_state(self) -> None:
        """Reset smoothing / hysteresis state between guided MI trials."""
        self._smoothed_probabilities = None
        self._gate_smoothed_probabilities = None
        self._artifact_rejector_smoothed_probabilities = None
        self._stable_prediction = None
        self._pending_prediction = None
        self._pending_count = 0
        self._release_count = 0
        self._gate_reject_count = 0
        self._stable_age = 0
        self._artifact_count = 0

    def _normalize_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """Normalize class probabilities and fall back to uniform uncertainty when invalid."""
        probabilities = np.asarray(probabilities, dtype=np.float64)
        probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
        total = float(np.sum(probabilities))
        if total <= 0.0:
            return np.full(len(self.artifact["class_names"]), 1.0 / len(self.artifact["class_names"]), dtype=np.float64)
        return probabilities / total

    def _smooth_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply EMA smoothing over probability vectors."""
        probabilities = self._normalize_probabilities(probabilities)
        if self._smoothed_probabilities is None:
            self._smoothed_probabilities = probabilities
        else:
            alpha = self.probability_smoothing
            self._smoothed_probabilities = (alpha * probabilities) + ((1.0 - alpha) * self._smoothed_probabilities)
            self._smoothed_probabilities = self._normalize_probabilities(self._smoothed_probabilities)
        return self._smoothed_probabilities.copy()

    def _smooth_gate_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply EMA smoothing over gate probabilities."""
        probabilities = np.asarray(probabilities, dtype=np.float64)
        probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
        total = float(np.sum(probabilities))
        if total <= 0.0:
            probabilities = np.full(2, 0.5, dtype=np.float64)
        else:
            probabilities = probabilities / total
        if self._gate_smoothed_probabilities is None:
            self._gate_smoothed_probabilities = probabilities
        else:
            alpha = self.probability_smoothing
            self._gate_smoothed_probabilities = (alpha * probabilities) + ((1.0 - alpha) * self._gate_smoothed_probabilities)
            total = float(np.sum(self._gate_smoothed_probabilities))
            if total <= 0.0:
                self._gate_smoothed_probabilities = np.full(2, 0.5, dtype=np.float64)
            else:
                self._gate_smoothed_probabilities = self._gate_smoothed_probabilities / total
        return self._gate_smoothed_probabilities.copy()

    def _smooth_artifact_rejector_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply EMA smoothing over bad-window rejector probabilities."""
        probabilities = np.asarray(probabilities, dtype=np.float64)
        probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
        total = float(np.sum(probabilities))
        if total <= 0.0:
            probabilities = np.full(2, 0.5, dtype=np.float64)
        else:
            probabilities = probabilities / total
        if self._artifact_rejector_smoothed_probabilities is None:
            self._artifact_rejector_smoothed_probabilities = probabilities
        else:
            alpha = self.probability_smoothing
            self._artifact_rejector_smoothed_probabilities = (
                (alpha * probabilities)
                + ((1.0 - alpha) * self._artifact_rejector_smoothed_probabilities)
            )
            total = float(np.sum(self._artifact_rejector_smoothed_probabilities))
            if total <= 0.0:
                self._artifact_rejector_smoothed_probabilities = np.full(2, 0.5, dtype=np.float64)
            else:
                self._artifact_rejector_smoothed_probabilities = (
                    self._artifact_rejector_smoothed_probabilities / total
                )
        return self._artifact_rejector_smoothed_probabilities.copy()

    @staticmethod
    def _uniform_probabilities(class_count: int) -> np.ndarray:
        """Return a uniform class-probability vector."""
        return np.full(int(class_count), 1.0 / max(int(class_count), 1), dtype=np.float64)

    def _collect_window_probabilities(
        self,
        raw_window: np.ndarray,
        live_sampling_rate: float,
        *,
        model_entries: list[dict[str, object]] | None = None,
        artifact: dict | None = None,
        fusion_method: str | None = None,
    ) -> tuple[np.ndarray, list[dict[str, object]]]:
        """Run all currently available window models and fuse their probability outputs."""
        model_entries = self.model_entries if model_entries is None else model_entries
        target_artifact = self.artifact if artifact is None else artifact
        target_fusion_method = self.fusion_method if fusion_method is None else str(fusion_method)
        probability_vectors: list[np.ndarray] = []
        active_weights: list[float] = []
        window_evidence: list[dict[str, object]] = []
        available_samples = int(raw_window.shape[1])
        class_count = len(target_artifact["class_names"])

        for entry in model_entries:
            window_sec = float(entry["window_sec"])
            required_live_samples = max(8, int(round(window_sec * float(live_sampling_rate))))
            if available_samples < required_live_samples:
                continue

            member_window = np.ascontiguousarray(raw_window[:, -required_live_samples:], dtype=np.float32)
            member_artifact = entry["artifact"]
            processed = _preprocess_single_window(member_window, member_artifact, live_sampling_rate)
            probabilities = _decision_to_probabilities(
                entry["pipeline"],
                processed,
                n_classes=class_count,
                probability_calibration=member_artifact.get("probability_calibration"),
            )
            probabilities = self._normalize_probabilities(probabilities)
            probability_vectors.append(probabilities)
            active_weights.append(float(entry["weight"]))

            prediction, confidence, runner_up_index, runner_up_confidence, margin = self._top_two(probabilities)
            window_evidence.append(
                {
                    "window_sec": window_sec,
                    "window_samples": int(entry["window_samples"]),
                    "weight": float(entry["weight"]),
                    "selected_pipeline": str(entry["selected_pipeline"]),
                    "prediction_index": prediction,
                    "prediction_name": target_artifact["class_names"][prediction],
                    "prediction_display_name": target_artifact["display_class_names"][prediction],
                    "confidence": float(confidence),
                    "margin": float(margin),
                    "runner_up_index": runner_up_index,
                    "runner_up_name": (
                        None if runner_up_index is None else target_artifact["class_names"][runner_up_index]
                    ),
                    "runner_up_display_name": (
                        None if runner_up_index is None else target_artifact["display_class_names"][runner_up_index]
                    ),
                    "runner_up_confidence": float(runner_up_confidence),
                    "probabilities": probabilities.tolist(),
                }
            )

        if not probability_vectors:
            raise ValueError(
                f"Realtime input is shorter than the minimum configured window ({self.minimum_window_sec:.3f}s)."
            )

        normalized_weights = _normalize_fusion_weights(active_weights)
        fused_probabilities = _fuse_probability_vectors(
            probability_vectors,
            normalized_weights,
            method=target_fusion_method,
        )
        fused_probabilities = np.asarray(
            apply_probability_calibration(
                fused_probabilities,
                target_artifact.get("probability_calibration"),
            ),
            dtype=np.float64,
        )

        for evidence, normalized_weight in zip(window_evidence, normalized_weights):
            evidence["normalized_weight"] = float(normalized_weight)

        return fused_probabilities, window_evidence

    def _collect_guided_window_probabilities(
        self,
        raw_imagery_window: np.ndarray,
        live_sampling_rate: float,
        imagery_elapsed_sec: float,
        *,
        model_entries: list[dict[str, object]] | None = None,
        artifact: dict | None = None,
        fusion_method: str | None = None,
    ) -> tuple[np.ndarray | None, list[dict[str, object]]]:
        """Run member models on cue-aligned imagery slices using each member's trained offset."""
        model_entries = self.model_entries if model_entries is None else model_entries
        target_artifact = self.artifact if artifact is None else artifact
        target_fusion_method = self.fusion_method if fusion_method is None else str(fusion_method)
        probability_vectors: list[np.ndarray] = []
        active_weights: list[float] = []
        window_evidence: list[dict[str, object]] = []
        available_samples = int(raw_imagery_window.shape[1])
        class_count = len(target_artifact["class_names"])

        for entry in model_entries:
            window_sec = float(entry["window_sec"])
            offset_sec = float(entry["window_offset_sec"])
            guided_end_sec = float(entry["guided_end_sec"])
            guided_end_live_samples = max(8, int(round(guided_end_sec * float(live_sampling_rate))))
            if available_samples < guided_end_live_samples:
                continue

            offset_live_samples = int(round(offset_sec * float(live_sampling_rate)))
            window_live_samples = int(round(window_sec * float(live_sampling_rate)))
            start_index = offset_live_samples
            stop_index = start_index + window_live_samples
            if stop_index > available_samples:
                continue

            member_window = np.ascontiguousarray(
                raw_imagery_window[:, start_index:stop_index],
                dtype=np.float32,
            )
            member_artifact = entry["artifact"]
            processed = _preprocess_single_window(member_window, member_artifact, live_sampling_rate)
            probabilities = _decision_to_probabilities(
                entry["pipeline"],
                processed,
                n_classes=class_count,
                probability_calibration=member_artifact.get("probability_calibration"),
            )
            probabilities = self._normalize_probabilities(probabilities)
            probability_vectors.append(probabilities)
            active_weights.append(float(entry["weight"]))

            prediction, confidence, runner_up_index, runner_up_confidence, margin = self._top_two(probabilities)
            window_evidence.append(
                {
                    "window_sec": window_sec,
                    "window_offset_sec": offset_sec,
                    "guided_end_sec": guided_end_sec,
                    "window_samples": int(entry["window_samples"]),
                    "weight": float(entry["weight"]),
                    "selected_pipeline": str(entry["selected_pipeline"]),
                    "prediction_index": prediction,
                    "prediction_name": target_artifact["class_names"][prediction],
                    "prediction_display_name": target_artifact["display_class_names"][prediction],
                    "confidence": float(confidence),
                    "margin": float(margin),
                    "runner_up_index": runner_up_index,
                    "runner_up_name": (
                        None if runner_up_index is None else target_artifact["class_names"][runner_up_index]
                    ),
                    "runner_up_display_name": (
                        None if runner_up_index is None else target_artifact["display_class_names"][runner_up_index]
                    ),
                    "runner_up_confidence": float(runner_up_confidence),
                    "probabilities": probabilities.tolist(),
                }
            )

        if not probability_vectors:
            return None, []

        normalized_weights = _normalize_fusion_weights(active_weights)
        fused_probabilities = _fuse_probability_vectors(
            probability_vectors,
            normalized_weights,
            method=target_fusion_method,
        )
        fused_probabilities = np.asarray(
            apply_probability_calibration(
                fused_probabilities,
                target_artifact.get("probability_calibration"),
            ),
            dtype=np.float64,
        )
        for evidence, normalized_weight in zip(window_evidence, normalized_weights):
            evidence["normalized_weight"] = float(normalized_weight)
            evidence["imagery_elapsed_sec"] = float(imagery_elapsed_sec)

        return fused_probabilities, window_evidence

    @staticmethod
    def _top_two(probabilities: np.ndarray) -> tuple[int, float, int | None, float, float]:
        """Return top-1 / top-2 indices, confidences, and margin."""
        order = np.argsort(probabilities)[::-1]
        top_index = int(order[0])
        top_confidence = float(probabilities[top_index])
        if len(order) <= 1:
            return top_index, top_confidence, None, 0.0, top_confidence
        second_index = int(order[1])
        second_confidence = float(probabilities[second_index])
        return top_index, top_confidence, second_index, second_confidence, top_confidence - second_confidence

    def _stable_evidence(self, probabilities: np.ndarray) -> tuple[float, float]:
        """Return probability and margin for the current stable class."""
        if self._stable_prediction is None:
            return 0.0, 0.0
        stable_probability = float(probabilities[self._stable_prediction])
        if len(probabilities) <= 1:
            return stable_probability, stable_probability
        competing = np.delete(probabilities, self._stable_prediction)
        competitor_probability = float(np.max(competing)) if competing.size else 0.0
        return stable_probability, stable_probability - competitor_probability

    def _reset_pending(self) -> None:
        self._pending_prediction = None
        self._pending_count = 0

    def _reset_release(self) -> None:
        self._release_count = 0

    def _reset_artifact(self) -> None:
        self._artifact_count = 0

    def _advance_artifact(self) -> int:
        self._artifact_count += 1
        return self._artifact_count

    def _clear_stable(self) -> None:
        self._stable_prediction = None
        self._stable_age = 0
        self._reset_release()

    def _set_stable(self, prediction: int) -> None:
        self._stable_prediction = int(prediction)
        self._stable_age = 1
        self._reset_pending()
        self._reset_release()

    def _hold_current(self) -> None:
        if self._stable_prediction is None:
            return
        self._stable_age = max(1, self._stable_age + 1)
        self._reset_release()

    def _advance_release(self) -> int:
        self._release_count += 1
        return self._release_count

    def _advance_pending(self, candidate_index: int) -> int:
        if self._pending_prediction == candidate_index:
            self._pending_count += 1
        else:
            self._pending_prediction = candidate_index
            self._pending_count = 1
        return self._pending_count

    def _evaluate_control_gate(
        self,
        raw_window: np.ndarray,
        live_sampling_rate: float,
        *,
        guided: bool = False,
        imagery_elapsed_sec: float = 0.0,
    ) -> dict[str, object]:
        """Evaluate the explicit control-vs-rest gate for the current realtime window."""
        if not self.gate_enabled:
            return {
                "enabled": False,
                "passed": True,
                "decision_state": "disabled",
                "raw_probabilities": np.asarray([0.0, 1.0], dtype=np.float64),
                "smoothed_probabilities": np.asarray([0.0, 1.0], dtype=np.float64),
                "window_evidence": [],
                "prediction_index": 1,
                "prediction_name": "control",
                "prediction_display_name": "CONTROL",
                "confidence": 1.0,
                "margin": 1.0,
                "thresholds": {
                    "confidence_threshold": float(self.gate_confidence_threshold),
                    "margin_threshold": float(self.gate_margin_threshold),
                },
            }

        if guided:
            probabilities, window_evidence = self._collect_guided_window_probabilities(
                raw_window,
                live_sampling_rate,
                imagery_elapsed_sec,
                model_entries=self.gate_model_entries,
                artifact=self.control_gate_artifact,
                fusion_method=str(self.control_gate_artifact.get("fusion_method", self.fusion_method)),
            )
        else:
            probabilities, window_evidence = self._collect_window_probabilities(
                raw_window,
                live_sampling_rate,
                model_entries=self.gate_model_entries,
                artifact=self.control_gate_artifact,
                fusion_method=str(self.control_gate_artifact.get("fusion_method", self.fusion_method)),
            )

        if probabilities is None or not window_evidence:
            fallback = (
                self._uniform_probabilities(len(self.control_gate_artifact["class_names"]))
                if self._gate_smoothed_probabilities is None
                else self._gate_smoothed_probabilities.copy()
            )
            return {
                "enabled": True,
                "passed": False,
                "decision_state": "warming_up",
                "raw_probabilities": fallback,
                "smoothed_probabilities": fallback,
                "window_evidence": [],
                "prediction_index": int(np.argmax(fallback)),
                "prediction_name": self.control_gate_artifact["class_names"][int(np.argmax(fallback))],
                "prediction_display_name": self.control_gate_artifact["display_class_names"][int(np.argmax(fallback))],
                "confidence": float(np.max(fallback)),
                "margin": float(np.sort(fallback)[-1] - (np.sort(fallback)[-2] if fallback.shape[0] > 1 else 0.0)),
                "thresholds": {
                    "confidence_threshold": float(self.gate_confidence_threshold),
                    "margin_threshold": float(self.gate_margin_threshold),
                },
            }

        raw_probabilities = np.asarray(probabilities, dtype=np.float64)
        smoothed_probabilities = self._smooth_gate_probabilities(raw_probabilities)
        prediction, confidence, _, _, margin = self._top_two(smoothed_probabilities)
        control_index = int(len(self.control_gate_artifact["class_names"]) - 1)
        passed = (
            prediction == control_index
            and confidence >= self.gate_confidence_threshold
            and margin >= self.gate_margin_threshold
        )
        return {
            "enabled": True,
            "passed": bool(passed),
            "decision_state": "passed" if passed else "blocked",
            "raw_probabilities": raw_probabilities,
            "smoothed_probabilities": smoothed_probabilities,
            "window_evidence": window_evidence,
            "prediction_index": int(prediction),
            "prediction_name": self.control_gate_artifact["class_names"][prediction],
            "prediction_display_name": self.control_gate_artifact["display_class_names"][prediction],
            "confidence": float(confidence),
            "margin": float(margin),
            "thresholds": {
                "confidence_threshold": float(self.gate_confidence_threshold),
                "margin_threshold": float(self.gate_margin_threshold),
            },
        }

    def _evaluate_artifact_rejector(
        self,
        raw_window: np.ndarray,
        live_sampling_rate: float,
        *,
        guided: bool = False,
        imagery_elapsed_sec: float = 0.0,
    ) -> dict[str, object]:
        """Evaluate learned bad-window rejector before control gate and main classifier."""
        if not self.artifact_rejector_enabled:
            return {
                "enabled": False,
                "rejected": False,
                "decision_state": "disabled",
                "raw_probabilities": np.asarray([1.0, 0.0], dtype=np.float64),
                "smoothed_probabilities": np.asarray([1.0, 0.0], dtype=np.float64),
                "window_evidence": [],
                "prediction_index": 0,
                "prediction_name": "clean",
                "prediction_display_name": "CLEAN",
                "confidence": 1.0,
                "margin": 1.0,
                "thresholds": {
                    "confidence_threshold": float(self.artifact_rejector_confidence_threshold),
                    "margin_threshold": float(self.artifact_rejector_margin_threshold),
                },
            }

        if guided:
            probabilities, window_evidence = self._collect_guided_window_probabilities(
                raw_window,
                live_sampling_rate,
                imagery_elapsed_sec,
                model_entries=self.artifact_rejector_model_entries,
                artifact=self.artifact_rejector_artifact,
                fusion_method=str(self.artifact_rejector_artifact.get("fusion_method", self.fusion_method)),
            )
        else:
            probabilities, window_evidence = self._collect_window_probabilities(
                raw_window,
                live_sampling_rate,
                model_entries=self.artifact_rejector_model_entries,
                artifact=self.artifact_rejector_artifact,
                fusion_method=str(self.artifact_rejector_artifact.get("fusion_method", self.fusion_method)),
            )

        if probabilities is None or not window_evidence:
            fallback = (
                self._uniform_probabilities(len(self.artifact_rejector_artifact["class_names"]))
                if self._artifact_rejector_smoothed_probabilities is None
                else self._artifact_rejector_smoothed_probabilities.copy()
            )
            return {
                "enabled": True,
                "rejected": False,
                "decision_state": "warming_up",
                "raw_probabilities": fallback,
                "smoothed_probabilities": fallback,
                "window_evidence": [],
                "prediction_index": int(np.argmax(fallback)),
                "prediction_name": self.artifact_rejector_artifact["class_names"][int(np.argmax(fallback))],
                "prediction_display_name": self.artifact_rejector_artifact["display_class_names"][int(np.argmax(fallback))],
                "confidence": float(np.max(fallback)),
                "margin": float(np.sort(fallback)[-1] - (np.sort(fallback)[-2] if fallback.shape[0] > 1 else 0.0)),
                "thresholds": {
                    "confidence_threshold": float(self.artifact_rejector_confidence_threshold),
                    "margin_threshold": float(self.artifact_rejector_margin_threshold),
                },
            }

        raw_probabilities = np.asarray(probabilities, dtype=np.float64)
        smoothed_probabilities = self._smooth_artifact_rejector_probabilities(raw_probabilities)
        prediction, confidence, _, _, margin = self._top_two(smoothed_probabilities)
        artifact_index = int(len(self.artifact_rejector_artifact["class_names"]) - 1)
        rejected = (
            prediction == artifact_index
            and confidence >= self.artifact_rejector_confidence_threshold
            and margin >= self.artifact_rejector_margin_threshold
        )
        return {
            "enabled": True,
            "rejected": bool(rejected),
            "decision_state": "rejected" if rejected else "clean",
            "raw_probabilities": raw_probabilities,
            "smoothed_probabilities": smoothed_probabilities,
            "window_evidence": window_evidence,
            "prediction_index": int(prediction),
            "prediction_name": self.artifact_rejector_artifact["class_names"][prediction],
            "prediction_display_name": self.artifact_rejector_artifact["display_class_names"][prediction],
            "confidence": float(confidence),
            "margin": float(margin),
            "thresholds": {
                "confidence_threshold": float(self.artifact_rejector_confidence_threshold),
                "margin_threshold": float(self.artifact_rejector_margin_threshold),
            },
        }

    def _assess_window_quality(self, raw_window: np.ndarray) -> dict[str, object]:
        """Detect obvious bad windows before they reach the classifier."""
        channel_std = np.std(raw_window, axis=1, dtype=np.float64)
        channel_ptp = np.ptp(raw_window, axis=1)
        finite_std = channel_std[np.isfinite(channel_std)]
        median_std = float(np.median(finite_std)) if finite_std.size else 0.0

        flatline_mask = channel_std <= self.flatline_std_threshold
        dominant_mask = np.zeros_like(flatline_mask, dtype=bool)
        if median_std > 0.0:
            dominant_mask = channel_std >= (median_std * self.dominant_channel_ratio_threshold)

        bad_mask = flatline_mask | dominant_mask
        bad_indices = [int(index) for index in np.flatnonzero(bad_mask)]
        bad_names = [self.artifact["channel_names"][index] for index in bad_indices]

        reasons = []
        if np.any(flatline_mask):
            reasons.append("flatline")
        if np.any(dominant_mask):
            reasons.append("dominant_channel")

        quality_ok = len(bad_indices) <= self.max_bad_channels
        return {
            "quality_ok": bool(quality_ok),
            "reason": "ok" if quality_ok else ",".join(reasons) or "bad_window",
            "bad_channel_indices": bad_indices,
            "bad_channel_names": bad_names,
            "bad_channel_count": len(bad_indices),
            "channel_std": channel_std.astype(np.float64).tolist(),
            "channel_ptp": np.asarray(channel_ptp, dtype=np.float64).tolist(),
            "median_channel_std": median_std,
        }

    def _resolve_stable_prediction(
        self,
        probabilities: np.ndarray,
        candidate_index: int,
        candidate_confidence: float,
        candidate_margin: float,
    ) -> tuple[int | None, str]:
        """Apply margin thresholds and hysteresis to produce the stable realtime output."""
        candidate_ready = (
            candidate_confidence >= self.confidence_threshold
            and candidate_margin >= self.margin_threshold
        )
        stable_probability, stable_margin = self._stable_evidence(probabilities)
        hold_ready = (
            stable_probability >= self.hold_confidence_threshold
            and stable_margin >= self.hold_margin_threshold
        )

        if self._stable_prediction is None:
            self._stable_age = 0
            self._reset_release()
            if not candidate_ready:
                self._reset_pending()
                return None, "uncertain"

            confirmation_count = self._advance_pending(candidate_index)
            if confirmation_count >= self.confirmation_windows:
                self._set_stable(candidate_index)
                return self._stable_prediction, "entered"
            return None, "confirming"

        in_min_hold = self._stable_age < self.min_stable_windows

        if candidate_index == self._stable_prediction:
            self._reset_pending()
            if hold_ready:
                self._hold_current()
                return self._stable_prediction, "holding"
            if in_min_hold:
                self._hold_current()
                return self._stable_prediction, "cooldown"
            release_count = self._advance_release()
            if release_count >= self.release_windows:
                self._clear_stable()
                return None, "released"
            self._stable_age += 1
            return self._stable_prediction, "fading"

        switch_ready = (
            candidate_ready
            and (candidate_confidence - stable_probability) >= self.switch_delta
        )
        if in_min_hold:
            self._reset_pending()
            self._hold_current()
            return self._stable_prediction, "cooldown"
        if switch_ready:
            confirmation_count = self._advance_pending(candidate_index)
            if confirmation_count >= self.confirmation_windows:
                self._set_stable(candidate_index)
                return self._stable_prediction, "switched"
            self._stable_age += 1
            self._reset_release()
            return self._stable_prediction, "switching"

        self._reset_pending()
        if hold_ready:
            self._hold_current()
            return self._stable_prediction, "holding"

        release_count = self._advance_release()
        if release_count >= self.release_windows:
            self._clear_stable()
            return None, "released"
        self._stable_age += 1
        return self._stable_prediction, "fading"

    def _build_result(
        self,
        *,
        raw_probabilities: np.ndarray,
        smoothed_probabilities: np.ndarray,
        stable_prediction: int | None,
        decision_state: str,
        quality: dict[str, object],
        window_evidence: list[dict[str, object]],
        input_duration_sec: float,
        gate_summary: dict[str, object] | None = None,
        artifact_summary: dict[str, object] | None = None,
        override_stable_display_name: str | None = None,
    ) -> dict[str, object]:
        """Build a UI-friendly result dictionary for one realtime step."""
        raw_prediction, raw_confidence, raw_second_index, raw_second_confidence, raw_margin = self._top_two(
            raw_probabilities
        )
        prediction, confidence, second_index, second_confidence, margin = self._top_two(smoothed_probabilities)

        stable_class_name = None
        stable_display_name = "UNCERTAIN"
        stable_confidence = 0.0
        stable_margin = 0.0
        if stable_prediction is not None:
            stable_class_name = self.artifact["class_names"][stable_prediction]
            stable_display_name = self.artifact["display_class_names"][stable_prediction]
            stable_confidence, stable_margin = self._stable_evidence(smoothed_probabilities)
        elif override_stable_display_name:
            stable_display_name = str(override_stable_display_name)

        pending_class_name = None
        pending_display_name = None
        if self._pending_prediction is not None:
            pending_class_name = self.artifact["class_names"][self._pending_prediction]
            pending_display_name = self.artifact["display_class_names"][self._pending_prediction]

        gate_summary = dict(gate_summary or {})
        gate_enabled = bool(gate_summary.get("enabled", False))
        gate_prediction_index = gate_summary.get("prediction_index")
        gate_prediction_name = gate_summary.get("prediction_name")
        gate_prediction_display_name = gate_summary.get("prediction_display_name")
        gate_passed = bool(gate_summary.get("passed", True)) if gate_enabled else True
        gate_raw_probabilities = np.asarray(
            gate_summary.get("raw_probabilities", np.asarray([0.0, 1.0], dtype=np.float64)),
            dtype=np.float64,
        )
        gate_smoothed_probabilities = np.asarray(
            gate_summary.get("smoothed_probabilities", gate_raw_probabilities),
            dtype=np.float64,
        )
        gate_thresholds = dict(gate_summary.get("thresholds", {}))

        artifact_summary = dict(artifact_summary or {})
        artifact_enabled = bool(artifact_summary.get("enabled", False))
        artifact_prediction_index = artifact_summary.get("prediction_index")
        artifact_prediction_name = artifact_summary.get("prediction_name")
        artifact_prediction_display_name = artifact_summary.get("prediction_display_name")
        artifact_rejected = bool(artifact_summary.get("rejected", False)) if artifact_enabled else False
        artifact_raw_probabilities = np.asarray(
            artifact_summary.get("raw_probabilities", np.asarray([1.0, 0.0], dtype=np.float64)),
            dtype=np.float64,
        )
        artifact_smoothed_probabilities = np.asarray(
            artifact_summary.get("smoothed_probabilities", artifact_raw_probabilities),
            dtype=np.float64,
        )
        artifact_thresholds = dict(artifact_summary.get("thresholds", {}))

        return {
            "prediction_index": prediction,
            "prediction_name": self.artifact["class_names"][prediction],
            "prediction_display_name": self.artifact["display_class_names"][prediction],
            "confidence": confidence,
            "margin": margin,
            "runner_up_index": second_index,
            "runner_up_name": None if second_index is None else self.artifact["class_names"][second_index],
            "runner_up_display_name": (
                None if second_index is None else self.artifact["display_class_names"][second_index]
            ),
            "runner_up_confidence": second_confidence,
            "raw_prediction_index": raw_prediction,
            "raw_prediction_name": self.artifact["class_names"][raw_prediction],
            "raw_prediction_display_name": self.artifact["display_class_names"][raw_prediction],
            "raw_confidence": raw_confidence,
            "raw_margin": raw_margin,
            "raw_runner_up_index": raw_second_index,
            "raw_runner_up_name": None if raw_second_index is None else self.artifact["class_names"][raw_second_index],
            "raw_runner_up_display_name": (
                None if raw_second_index is None else self.artifact["display_class_names"][raw_second_index]
            ),
            "raw_runner_up_confidence": raw_second_confidence,
            "stable_prediction_index": stable_prediction,
            "stable_prediction_name": stable_class_name,
            "stable_prediction_display_name": stable_display_name,
            "stable_confidence": stable_confidence,
            "stable_margin": stable_margin,
            "decision_state": decision_state,
            "pending_prediction_index": self._pending_prediction,
            "pending_prediction_name": pending_class_name,
            "pending_prediction_display_name": pending_display_name,
            "pending_count": int(self._pending_count),
            "confirmation_windows": int(self.confirmation_windows),
            "release_count": int(self._release_count),
            "release_windows": int(self.release_windows),
            "stable_age": int(self._stable_age),
            "min_stable_windows": int(self.min_stable_windows),
            "artifact_count": int(self._artifact_count),
            "artifact_freeze_windows": int(self.artifact_freeze_windows),
            "quality_ok": bool(quality["quality_ok"]),
            "quality_reason": str(quality["reason"]),
            "quality_bad_channel_indices": list(quality["bad_channel_indices"]),
            "quality_bad_channel_names": list(quality["bad_channel_names"]),
            "quality_bad_channel_count": int(quality["bad_channel_count"]),
            "quality_channel_std": list(quality["channel_std"]),
            "quality_channel_ptp": list(quality["channel_ptp"]),
            "quality_median_channel_std": float(quality["median_channel_std"]),
            "gate_enabled": gate_enabled,
            "gate_passed": bool(gate_passed),
            "gate_decision_state": str(gate_summary.get("decision_state", "disabled" if not gate_enabled else "passed")),
            "gate_prediction_index": gate_prediction_index,
            "gate_prediction_name": gate_prediction_name,
            "gate_prediction_display_name": gate_prediction_display_name,
            "gate_confidence": float(gate_summary.get("confidence", 1.0 if not gate_enabled else 0.0)),
            "gate_margin": float(gate_summary.get("margin", 1.0 if not gate_enabled else 0.0)),
            "gate_thresholds": gate_thresholds,
            "gate_raw_probabilities": gate_raw_probabilities.tolist(),
            "gate_probabilities": gate_smoothed_probabilities.tolist(),
            "gate_window_evidence": list(gate_summary.get("window_evidence", [])),
            "gate_reject_count": int(self._gate_reject_count),
            "gate_release_windows": int(self.gate_release_windows),
            "artifact_rejector_enabled": artifact_enabled,
            "artifact_rejected": bool(artifact_rejected),
            "artifact_decision_state": str(
                artifact_summary.get("decision_state", "disabled" if not artifact_enabled else "clean")
            ),
            "artifact_prediction_index": artifact_prediction_index,
            "artifact_prediction_name": artifact_prediction_name,
            "artifact_prediction_display_name": artifact_prediction_display_name,
            "artifact_confidence": float(artifact_summary.get("confidence", 0.0)),
            "artifact_margin": float(artifact_summary.get("margin", 0.0)),
            "artifact_thresholds": artifact_thresholds,
            "artifact_raw_probabilities": artifact_raw_probabilities.tolist(),
            "artifact_probabilities": artifact_smoothed_probabilities.tolist(),
            "artifact_window_evidence": list(artifact_summary.get("window_evidence", [])),
            "fusion_method": self.fusion_method,
            "input_duration_sec": float(input_duration_sec),
            "active_window_count": int(len(window_evidence)),
            "active_window_secs": [float(item["window_sec"]) for item in window_evidence],
            "configured_window_secs": [float(entry["window_sec"]) for entry in self.model_entries],
            "configured_window_offset_secs": [float(entry["window_offset_sec"]) for entry in self.model_entries],
            "window_evidence": window_evidence,
            "probabilities": smoothed_probabilities.tolist(),
            "raw_probabilities": raw_probabilities.tolist(),
            "class_names": list(self.artifact["class_names"]),
            "display_class_names": list(self.artifact["display_class_names"]),
        }

    def _handle_gate_block(
        self,
        *,
        quality: dict[str, object],
        input_duration_sec: float,
        gate_summary: dict[str, object],
        artifact_summary: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Return a no-control result when the explicit gate rejects the current window."""
        self._reset_pending()
        self._gate_reject_count += 1
        classifier_probabilities = (
            self._uniform_probabilities(len(self.artifact["class_names"]))
            if self._smoothed_probabilities is None
            else self._smoothed_probabilities.copy()
        )
        stable_prediction = self._stable_prediction
        gate_state = str(gate_summary.get("decision_state", "blocked"))

        if stable_prediction is not None and self._gate_reject_count <= self.gate_release_windows:
            self._stable_age = max(1, self._stable_age + 1)
            return self._build_result(
                raw_probabilities=classifier_probabilities,
                smoothed_probabilities=classifier_probabilities,
                stable_prediction=stable_prediction,
                decision_state="gate_holding",
                quality=quality,
                window_evidence=[],
                input_duration_sec=float(input_duration_sec),
                gate_summary=gate_summary,
                artifact_summary=artifact_summary,
            )

        self._clear_stable()
        if gate_state != "warming_up":
            self._smoothed_probabilities = None
        decision_state = "warming_up" if gate_state == "warming_up" else "no_control"
        return self._build_result(
            raw_probabilities=classifier_probabilities,
            smoothed_probabilities=classifier_probabilities,
            stable_prediction=None,
            decision_state=decision_state,
            quality=quality,
            window_evidence=[],
            input_duration_sec=float(input_duration_sec),
            gate_summary=gate_summary,
            artifact_summary=artifact_summary,
            override_stable_display_name=None if decision_state == "warming_up" else "NO CONTROL",
        )

    def analyze_guided_imagery(
        self,
        raw_imagery_window: np.ndarray,
        live_sampling_rate: float,
        *,
        imagery_elapsed_sec: float,
    ) -> dict:
        """Predict one cue-aligned imagery segment in the synchronous MI protocol."""
        raw_imagery_window = np.asarray(raw_imagery_window, dtype=np.float32)
        if raw_imagery_window.ndim != 2:
            raise ValueError("Guided realtime input must have shape (channels, samples).")
        if raw_imagery_window.shape[0] != self.expected_channel_count:
            raise ValueError(
                f"Expected {self.expected_channel_count} channels, got {raw_imagery_window.shape[0]}."
            )
        if raw_imagery_window.shape[1] < 8:
            raise ValueError("Guided imagery input window is too short.")

        quality = self._assess_window_quality(raw_imagery_window)
        if not quality["quality_ok"]:
            self._reset_pending()
            artifact_count = self._advance_artifact()
            if self._smoothed_probabilities is None:
                frozen_probabilities = np.full(
                    len(self.artifact["class_names"]),
                    1.0 / len(self.artifact["class_names"]),
                    dtype=np.float64,
                )
            else:
                frozen_probabilities = self._smoothed_probabilities.copy()

            stable_prediction = self._stable_prediction
            if stable_prediction is not None and artifact_count <= self.artifact_freeze_windows:
                self._stable_age = max(1, self._stable_age + 1)
                return self._build_result(
                    raw_probabilities=frozen_probabilities,
                    smoothed_probabilities=frozen_probabilities,
                    stable_prediction=stable_prediction,
                    decision_state="frozen",
                    quality=quality,
                    window_evidence=[],
                    input_duration_sec=float(imagery_elapsed_sec),
                )

            self._clear_stable()
            return self._build_result(
                raw_probabilities=frozen_probabilities,
                smoothed_probabilities=frozen_probabilities,
                stable_prediction=None,
                decision_state="artifact_released" if artifact_count > self.artifact_freeze_windows else "bad_window",
                quality=quality,
                window_evidence=[],
                input_duration_sec=float(imagery_elapsed_sec),
            )

        artifact_summary = self._evaluate_artifact_rejector(
            raw_imagery_window,
            live_sampling_rate,
            guided=True,
            imagery_elapsed_sec=float(imagery_elapsed_sec),
        )
        if artifact_summary.get("rejected", False):
            self._reset_pending()
            artifact_count = self._advance_artifact()
            frozen_probabilities = (
                np.full(
                    len(self.artifact["class_names"]),
                    1.0 / len(self.artifact["class_names"]),
                    dtype=np.float64,
                )
                if self._smoothed_probabilities is None
                else self._smoothed_probabilities.copy()
            )
            stable_prediction = self._stable_prediction
            if stable_prediction is not None and artifact_count <= self.artifact_freeze_windows:
                self._stable_age = max(1, self._stable_age + 1)
                return self._build_result(
                    raw_probabilities=frozen_probabilities,
                    smoothed_probabilities=frozen_probabilities,
                    stable_prediction=stable_prediction,
                    decision_state="frozen",
                    quality=quality,
                    window_evidence=[],
                    input_duration_sec=float(imagery_elapsed_sec),
                    artifact_summary=artifact_summary,
                )
            self._clear_stable()
            return self._build_result(
                raw_probabilities=frozen_probabilities,
                smoothed_probabilities=frozen_probabilities,
                stable_prediction=None,
                decision_state=(
                    "artifact_rejector_released"
                    if artifact_count > self.artifact_freeze_windows
                    else "artifact_rejected"
                ),
                quality=quality,
                window_evidence=[],
                input_duration_sec=float(imagery_elapsed_sec),
                artifact_summary=artifact_summary,
            )

        self._reset_artifact()

        gate_summary = self._evaluate_control_gate(
            raw_imagery_window,
            live_sampling_rate,
            guided=True,
            imagery_elapsed_sec=float(imagery_elapsed_sec),
        )
        if not gate_summary.get("passed", True):
            return self._handle_gate_block(
                quality=quality,
                input_duration_sec=float(imagery_elapsed_sec),
                gate_summary=gate_summary,
                artifact_summary=artifact_summary,
            )
        self._gate_reject_count = 0

        probabilities, window_evidence = self._collect_guided_window_probabilities(
            raw_imagery_window,
            live_sampling_rate,
            imagery_elapsed_sec,
        )
        if probabilities is None or not window_evidence:
            waiting_probabilities = (
                np.full(len(self.artifact["class_names"]), 1.0 / len(self.artifact["class_names"]), dtype=np.float64)
                if self._smoothed_probabilities is None
                else self._smoothed_probabilities.copy()
            )
            return self._build_result(
                raw_probabilities=waiting_probabilities,
                smoothed_probabilities=waiting_probabilities,
                stable_prediction=self._stable_prediction,
                decision_state="warming_up",
                quality=quality,
                window_evidence=[],
                input_duration_sec=float(imagery_elapsed_sec),
                gate_summary=gate_summary,
                artifact_summary=artifact_summary,
            )

        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.shape[0] != len(self.artifact["class_names"]):
            raise ValueError("Guided model output probability shape does not match class count.")
        raw_probabilities = self._normalize_probabilities(probabilities)
        smoothed_probabilities = self._smooth_probabilities(raw_probabilities)
        prediction, confidence, _, _, margin = self._top_two(smoothed_probabilities)

        stable_prediction, decision_state = self._resolve_stable_prediction(
            smoothed_probabilities,
            prediction,
            confidence,
            margin,
        )
        return self._build_result(
            raw_probabilities=raw_probabilities,
            smoothed_probabilities=smoothed_probabilities,
            stable_prediction=stable_prediction,
            decision_state=decision_state,
            quality=quality,
            window_evidence=window_evidence,
            input_duration_sec=float(imagery_elapsed_sec),
            gate_summary=gate_summary,
            artifact_summary=artifact_summary,
        )

    def analyze_window(self, raw_window: np.ndarray, live_sampling_rate: float) -> dict:
        """Predict one sliding realtime window."""
        raw_window = np.asarray(raw_window, dtype=np.float32)
        if raw_window.ndim != 2:
            raise ValueError("Realtime input must have shape (channels, samples).")
        if raw_window.shape[0] != self.expected_channel_count:
            raise ValueError(
                f"Expected {self.expected_channel_count} channels, got {raw_window.shape[0]}."
            )
        if raw_window.shape[1] < 8:
            raise ValueError("Realtime input window is too short.")

        quality = self._assess_window_quality(raw_window)
        if not quality["quality_ok"]:
            self._reset_pending()
            artifact_count = self._advance_artifact()
            if self._smoothed_probabilities is None:
                frozen_probabilities = np.full(
                    len(self.artifact["class_names"]),
                    1.0 / len(self.artifact["class_names"]),
                    dtype=np.float64,
                )
            else:
                frozen_probabilities = self._smoothed_probabilities.copy()

            stable_prediction = self._stable_prediction
            if stable_prediction is not None and artifact_count <= self.artifact_freeze_windows:
                self._stable_age = max(1, self._stable_age + 1)
                return self._build_result(
                    raw_probabilities=frozen_probabilities,
                    smoothed_probabilities=frozen_probabilities,
                    stable_prediction=stable_prediction,
                    decision_state="frozen",
                    quality=quality,
                    window_evidence=[],
                    input_duration_sec=float(raw_window.shape[1] / max(float(live_sampling_rate), 1e-6)),
                )

            self._clear_stable()
            return self._build_result(
                raw_probabilities=frozen_probabilities,
                smoothed_probabilities=frozen_probabilities,
                stable_prediction=None,
                decision_state="artifact_released" if artifact_count > self.artifact_freeze_windows else "bad_window",
                quality=quality,
                window_evidence=[],
                input_duration_sec=float(raw_window.shape[1] / max(float(live_sampling_rate), 1e-6)),
            )

        artifact_summary = self._evaluate_artifact_rejector(
            raw_window,
            live_sampling_rate,
            guided=False,
        )
        if artifact_summary.get("rejected", False):
            self._reset_pending()
            artifact_count = self._advance_artifact()
            frozen_probabilities = (
                np.full(
                    len(self.artifact["class_names"]),
                    1.0 / len(self.artifact["class_names"]),
                    dtype=np.float64,
                )
                if self._smoothed_probabilities is None
                else self._smoothed_probabilities.copy()
            )
            stable_prediction = self._stable_prediction
            if stable_prediction is not None and artifact_count <= self.artifact_freeze_windows:
                self._stable_age = max(1, self._stable_age + 1)
                return self._build_result(
                    raw_probabilities=frozen_probabilities,
                    smoothed_probabilities=frozen_probabilities,
                    stable_prediction=stable_prediction,
                    decision_state="frozen",
                    quality=quality,
                    window_evidence=[],
                    input_duration_sec=float(raw_window.shape[1] / max(float(live_sampling_rate), 1e-6)),
                    artifact_summary=artifact_summary,
                )
            self._clear_stable()
            return self._build_result(
                raw_probabilities=frozen_probabilities,
                smoothed_probabilities=frozen_probabilities,
                stable_prediction=None,
                decision_state=(
                    "artifact_rejector_released"
                    if artifact_count > self.artifact_freeze_windows
                    else "artifact_rejected"
                ),
                quality=quality,
                window_evidence=[],
                input_duration_sec=float(raw_window.shape[1] / max(float(live_sampling_rate), 1e-6)),
                artifact_summary=artifact_summary,
            )

        self._reset_artifact()

        gate_summary = self._evaluate_control_gate(
            raw_window,
            live_sampling_rate,
            guided=False,
        )
        if not gate_summary.get("passed", True):
            return self._handle_gate_block(
                quality=quality,
                input_duration_sec=float(raw_window.shape[1] / max(float(live_sampling_rate), 1e-6)),
                gate_summary=gate_summary,
                artifact_summary=artifact_summary,
            )
        self._gate_reject_count = 0

        probabilities, window_evidence = self._collect_window_probabilities(raw_window, live_sampling_rate)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.shape[0] != len(self.artifact["class_names"]):
            raise ValueError("Model output probability shape does not match class count.")
        raw_probabilities = self._normalize_probabilities(probabilities)

        smoothed_probabilities = self._smooth_probabilities(raw_probabilities)
        prediction, confidence, _, _, margin = self._top_two(smoothed_probabilities)

        stable_prediction, decision_state = self._resolve_stable_prediction(
            smoothed_probabilities,
            prediction,
            confidence,
            margin,
        )
        return self._build_result(
            raw_probabilities=raw_probabilities,
            smoothed_probabilities=smoothed_probabilities,
            stable_prediction=stable_prediction,
            decision_state=decision_state,
            quality=quality,
            window_evidence=window_evidence,
            input_duration_sec=float(raw_window.shape[1] / max(float(live_sampling_rate), 1e-6)),
            gate_summary=gate_summary,
            artifact_summary=artifact_summary,
        )
