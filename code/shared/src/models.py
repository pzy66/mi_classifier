"""Model definitions and pipeline builders for the MI classifier."""

from __future__ import annotations

import copy

import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.custom_transformers import (
    ChannelSelector,
    FlattenTransformer,
    HybridFeatureExtractor,
    HierarchicalMIClassifier,
    OneVsRestFBCSP,
    RiemannFeatureExtractor,
)


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

DEFAULT_CENTRAL_CHANNEL_NAMES = ["C3", "Cz", "C4"]
DEFAULT_CENTRAL_BRANCH_WEIGHT = 0.65
CENTRAL_FBCSP_LDA_ALIASES = {"central_fbcsp_lda", "central-fbcsp-lda"}
CENTRAL_FBCSP_LDA_BANDS = [
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 28.0),
    (28.0, 32.0),
]
CENTRAL_FBCSP_LDA_COMPONENTS = 2
DEFAULT_DEEP_STAGE_PRETRAIN_WINDOW_SECS = [2.5, 2.0]
DEFAULT_DEEP_STAGE_FINETUNE_WINDOW_SECS = [2.0, 1.5]
DEFAULT_CENTRAL_PRIOR_LOGIT_ALPHA = 0.75
DEFAULT_CENTRAL_PRIOR_AUX_LOSS_WEIGHT = 0.3
DEEP_FBLIGHT_ALIASES = {"fblight", "fblight_tcn"}
CENTRAL_ONLY_FBLIGHT_ALIASES = {"central_only_fblight", "central-only-fblight"}
FULL8_FBLIGHT_ALIASES = {"full8_fblight", "full8-fblight"}
CENTRAL_PRIOR_DUAL_BRANCH_ALIASES = {
    "central_prior_dual_branch_fblight_tcn",
    "central-prior-dual-branch-fblight-tcn",
}
CENTRAL_GATE_FBLIGHT_ALIASES = {"central_gate_fblight", "central-gate-fblight"}
CENTRAL_PRIOR_GATE_FBLIGHT_ALIASES = {
    "central_prior_gate_fblight",
    "central-prior-gate-fblight",
}

TORCH_AVAILABLE = torch is not None


def _coerce_bands(raw_bands: list[list[float]] | list[tuple[float, float]] | None) -> list[tuple[float, float]]:
    """Normalize YAML-style band definitions into float tuples."""
    if raw_bands is None:
        return list(DEFAULT_FBCSP_BANDS)
    return [tuple(float(value) for value in band) for band in raw_bands]


def _resolve_channel_indices(
    channel_names: list[str] | None,
    requested_names: list[str],
) -> list[int]:
    """Resolve requested channel names into dataset-channel indices."""
    if channel_names is None:
        raise ValueError(
            f"Candidate requires explicit channel_names, but none were provided. Required={requested_names}"
        )
    normalized_lookup = {
        str(name).strip().lower(): index for index, name in enumerate(channel_names)
    }
    resolved: list[int] = []
    missing: list[str] = []
    for name in requested_names:
        token = str(name).strip().lower()
        if token not in normalized_lookup:
            missing.append(str(name))
            continue
        resolved.append(int(normalized_lookup[token]))
    if missing:
        raise ValueError(
            f"Missing required channels for central-priority candidate: {missing}. "
            f"Available channels: {list(channel_names)}"
        )
    return resolved


def _normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    """Normalize a probability matrix row-wise and replace invalid rows with uniform uncertainty."""
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D probability matrix, got shape {matrix.shape}.")
    if matrix.shape[1] <= 0:
        raise ValueError("Probability matrix must contain at least one class column.")

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.clip(matrix, a_min=0.0, a_max=None)
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    invalid_rows = row_sums <= 0.0
    if np.any(invalid_rows):
        matrix[invalid_rows[:, 0]] = 1.0
        row_sums = np.sum(matrix, axis=1, keepdims=True)
    return matrix / row_sums


def _softmax_rows(scores: np.ndarray) -> np.ndarray:
    """Apply a numerically stable row-wise softmax."""
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D score matrix, got shape {matrix.shape}.")
    if matrix.shape[1] <= 0:
        raise ValueError("Score matrix must contain at least one class column.")

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    shifted = matrix - np.max(matrix, axis=1, keepdims=True)
    exponentiated = np.exp(np.clip(shifted, a_min=-60.0, a_max=60.0))
    row_sums = np.sum(exponentiated, axis=1, keepdims=True)
    invalid_rows = row_sums <= 0.0
    if np.any(invalid_rows):
        exponentiated[invalid_rows[:, 0]] = 1.0
        row_sums = np.sum(exponentiated, axis=1, keepdims=True)
    return exponentiated / row_sums


def _coerce_decision_matrix(decision_values: np.ndarray, class_count: int) -> np.ndarray | None:
    """Normalize raw decision outputs into an (n_samples, n_classes) score matrix."""
    matrix = np.asarray(decision_values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        return None
    if matrix.shape[1] == int(class_count):
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if int(class_count) == 2 and matrix.shape[1] == 1:
        zeros = np.zeros((matrix.shape[0], 1), dtype=np.float64)
        return np.concatenate([zeros, matrix], axis=1)
    return None


def _apply_temperature_to_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to a probability matrix via log-probability reparameterization."""
    normalized = _normalize_probability_rows(probabilities)
    safe_temperature = max(float(temperature), 1e-6)
    log_probabilities = np.log(np.clip(normalized, 1e-8, 1.0))
    return _softmax_rows(log_probabilities / safe_temperature)


def _negative_log_likelihood(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    """Compute multiclass negative log-likelihood from class probabilities."""
    normalized = _normalize_probability_rows(probabilities)
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if normalized.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Probability rows ({normalized.shape[0]}) do not match labels ({labels.shape[0]})."
        )
    if normalized.shape[0] == 0:
        return 0.0
    clipped = np.clip(normalized[np.arange(labels.shape[0]), labels], 1e-8, 1.0)
    return float(-np.mean(np.log(clipped)))


def fit_probability_calibration(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    *,
    calibration_type: str = "temperature_scaling",
    role: str | None = None,
    selection_source: str | None = None,
) -> dict[str, object]:
    """Fit a lightweight post-hoc probability calibration object from validation probabilities."""
    normalized = _normalize_probability_rows(probabilities)
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if normalized.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Calibration probabilities ({normalized.shape[0]}) do not match labels ({labels.shape[0]})."
        )

    calibration_name = str(calibration_type).strip().lower()
    if calibration_name != "temperature_scaling":
        raise ValueError(f"Unsupported probability calibration type: {calibration_type}")

    baseline_nll = _negative_log_likelihood(normalized, labels)
    best_temperature = 1.0
    best_nll = baseline_nll

    if normalized.shape[0] > 0 and normalized.shape[1] > 1:
        lower = 0.25
        upper = 4.0
        for _ in range(3):
            grid = np.geomspace(lower, upper, num=25)
            losses = [
                _negative_log_likelihood(
                    _apply_temperature_to_probabilities(normalized, float(temperature)),
                    labels,
                )
                for temperature in grid
            ]
            best_index = int(np.argmin(losses))
            candidate_temperature = float(grid[best_index])
            candidate_nll = float(losses[best_index])
            if candidate_nll < best_nll:
                best_temperature = candidate_temperature
                best_nll = candidate_nll

            if best_index == 0:
                lower = max(lower / 2.0, 0.05)
                upper = float(grid[min(1, len(grid) - 1)])
            elif best_index == len(grid) - 1:
                lower = float(grid[max(len(grid) - 2, 0)])
                upper = min(upper * 2.0, 20.0)
            else:
                lower = float(grid[best_index - 1])
                upper = float(grid[best_index + 1])

    calibrated = _apply_temperature_to_probabilities(normalized, best_temperature)
    summary = {
        "type": calibration_name,
        "temperature": float(best_temperature),
        "sample_count": int(labels.shape[0]),
        "class_count": int(normalized.shape[1]),
        "nll_before": float(baseline_nll),
        "nll_after": float(_negative_log_likelihood(calibrated, labels)),
        "mean_confidence_before": float(np.mean(np.max(normalized, axis=1))) if normalized.shape[0] else 0.0,
        "mean_confidence_after": float(np.mean(np.max(calibrated, axis=1))) if calibrated.shape[0] else 0.0,
    }
    if role is not None:
        summary["role"] = str(role)
    if selection_source is not None:
        summary["selection_source"] = str(selection_source)
    return summary


def apply_probability_calibration(
    probabilities: np.ndarray,
    probability_calibration: dict[str, object] | None = None,
) -> np.ndarray:
    """Apply an exported probability calibration object to a 1D or 2D probability matrix."""
    matrix = np.asarray(probabilities, dtype=np.float64)
    squeeze_output = matrix.ndim == 1
    normalized = _normalize_probability_rows(matrix)

    if not probability_calibration:
        return normalized[0] if squeeze_output else normalized

    calibration_type = str(probability_calibration.get("type", "")).strip().lower()
    if calibration_type == "temperature_scaling":
        temperature = float(probability_calibration.get("temperature", 1.0))
        calibrated = _apply_temperature_to_probabilities(normalized, temperature)
    else:
        calibrated = normalized

    return calibrated[0] if squeeze_output else calibrated


def _probability_like_matrix(
    model,
    X: np.ndarray,
    class_count: int,
    probability_calibration: dict[str, object] | None = None,
) -> np.ndarray:
    """Convert model outputs into an (n_samples, n_classes) probability-like matrix."""
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
        if probabilities.ndim == 2 and probabilities.shape[1] == class_count:
            return apply_probability_calibration(probabilities, probability_calibration)

    if hasattr(model, "decision_function"):
        decision_matrix = _coerce_decision_matrix(model.decision_function(X), class_count)
        if decision_matrix is not None:
            probabilities = _softmax_rows(decision_matrix)
            return apply_probability_calibration(probabilities, probability_calibration)

    predictions = np.asarray(model.predict(X), dtype=np.int64).reshape(-1)
    fallback = np.zeros((predictions.shape[0], class_count), dtype=np.float64)
    fallback[np.arange(predictions.shape[0]), predictions] = 1.0
    return apply_probability_calibration(fallback, probability_calibration)


def predict_probability_matrix(
    model,
    X: np.ndarray,
    class_count: int,
    probability_calibration: dict[str, object] | None = None,
) -> np.ndarray:
    """Public wrapper used by training / realtime code to obtain calibrated class probabilities."""
    return _probability_like_matrix(
        model,
        X,
        class_count,
        probability_calibration=probability_calibration,
    )


def _align_probability_columns(
    probabilities: np.ndarray,
    source_classes: np.ndarray,
    target_classes: np.ndarray,
) -> np.ndarray:
    """Align class-probability columns from source class order to target class order."""
    aligned = np.zeros((probabilities.shape[0], target_classes.shape[0]), dtype=np.float64)
    source_lookup = {int(label): column for column, label in enumerate(source_classes.tolist())}
    for target_column, label in enumerate(target_classes.tolist()):
        source_column = source_lookup.get(int(label))
        if source_column is not None:
            aligned[:, target_column] = probabilities[:, source_column]
    row_sums = np.sum(aligned, axis=1, keepdims=True)
    row_sums[row_sums <= 0.0] = 1.0
    return aligned / row_sums


def _build_classifier(
    classifier_name: str,
    *,
    kernel: str = "rbf",
    C: float = 1.0,
    probability: bool = False,
):
    """Build a sklearn classifier used by the classical pipelines."""
    classifier_name = classifier_name.lower()
    if classifier_name == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if classifier_name == "svm":
        return SVC(kernel=kernel, C=C, gamma="scale", probability=probability)
    raise ValueError(f"Unsupported classifier: {classifier_name}")


def _resolve_torch_device(device_name: str | None = None) -> str:
    """Return a safe torch device string for training and inference."""
    if torch is None:
        return "cpu"
    requested = str(device_name or "").strip().lower()
    if requested:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_classical_pipeline(model_name: str, *, kernel: str = "rbf", C: float = 1.0):
    """Build the flatten-and-classify baseline pipelines."""
    classifier = _build_classifier(model_name, kernel=kernel, C=C)
    return make_pipeline(FlattenTransformer(), StandardScaler(), classifier)


def build_feature_extractor(
    feature_name: str,
    *,
    fs: float = 250.0,
    bands: list[list[float]] | list[tuple[float, float]] | None = None,
    n_components: int = 4,
    riemann_band: tuple[float, float] | list[float] = (4.0, 40.0),
    estimator: str = "lwf",
    metric: str = "riemann",
):
    """Build an EEG feature extractor used by the optimized classical pipelines."""
    feature_name = feature_name.lower()
    normalized_bands = _coerce_bands(bands)
    normalized_riemann_band = tuple(float(value) for value in riemann_band)

    if feature_name == "fbcsp":
        return OneVsRestFBCSP(normalized_bands, n_components=n_components, fs=fs)
    if feature_name == "riemann":
        return RiemannFeatureExtractor(
            band=normalized_riemann_band,
            fs=fs,
            estimator=estimator,
            metric=metric,
        )
    if feature_name == "hybrid":
        return HybridFeatureExtractor(
            normalized_bands,
            n_components=n_components,
            fs=fs,
            riemann_band=normalized_riemann_band,
            estimator=estimator,
            metric=metric,
        )
    raise ValueError(f"Unsupported feature extractor: {feature_name}")


def _build_torch_candidate(
    architecture: str,
    *,
    torch_epochs: int,
    torch_batch_size: int,
    torch_learning_rate: float,
    torch_weight_decay: float,
    torch_patience: int,
    torch_validation_split: float,
    torch_device: str | None,
    sampling_rate: float | None = None,
    stage_pretrain_window_secs: list[float] | tuple[float, ...] | None = None,
    stage_finetune_window_secs: list[float] | tuple[float, ...] | None = None,
    central_prior_alpha: float = DEFAULT_CENTRAL_PRIOR_LOGIT_ALPHA,
    central_aux_loss_weight: float = DEFAULT_CENTRAL_PRIOR_AUX_LOSS_WEIGHT,
    dual_branch_central_indices: list[int] | tuple[int, ...] | None = None,
):
    """Build one torch EEG candidate wrapper."""
    return TorchEEGClassifier(
        architecture=architecture,
        n_channels=None,
        n_classes=None,
        epochs=torch_epochs,
        batch_size=torch_batch_size,
        learning_rate=torch_learning_rate,
        weight_decay=torch_weight_decay,
        validation_split=torch_validation_split,
        patience=torch_patience,
        random_state=42,
        device=torch_device,
        sampling_rate=sampling_rate,
        stage_pretrain_window_secs=stage_pretrain_window_secs,
        stage_finetune_window_secs=stage_finetune_window_secs,
        central_prior_alpha=central_prior_alpha,
        central_aux_loss_weight=central_aux_loss_weight,
        dual_branch_central_indices=dual_branch_central_indices,
    )


def _build_standard_candidate(
    *,
    feature_name: str,
    classifier_name: str,
    fs: float,
    bands: list[list[float]] | list[tuple[float, float]] | None,
    n_components: int,
    riemann_band: tuple[float, float] | list[float],
    estimator: str,
    metric: str,
    kernel: str,
    C: float,
):
    """Build one non-deep candidate used by either full or central branches."""
    if feature_name == "hierarchical":
        return HierarchicalMIClassifier(
            bands=_coerce_bands(bands),
            n_components=n_components,
            fs=fs,
            riemann_band=tuple(float(value) for value in riemann_band),
            estimator=estimator,
            metric=metric,
            classifier_name=classifier_name,
            kernel=kernel,
            C=C,
        )

    return make_pipeline(
        build_feature_extractor(
            feature_name,
            fs=fs,
            bands=bands,
            n_components=n_components,
            riemann_band=riemann_band,
            estimator=estimator,
            metric=metric,
        ),
        StandardScaler(),
        _build_classifier(classifier_name, kernel=kernel, C=C),
    )


def build_optimized_candidates(
    *,
    candidate_names: list[str] | None = None,
    fs: float = 250.0,
    bands: list[list[float]] | list[tuple[float, float]] | None = None,
    n_components: int = 4,
    riemann_band: tuple[float, float] | list[float] = (4.0, 40.0),
    estimator: str = "lwf",
    metric: str = "riemann",
    kernel: str = "rbf",
    C: float = 1.0,
    torch_epochs: int = 80,
    torch_batch_size: int = 32,
    torch_learning_rate: float = 1e-3,
    torch_weight_decay: float = 1e-4,
    torch_patience: int = 12,
    torch_validation_split: float = 0.15,
    torch_device: str | None = None,
    channel_names: list[str] | None = None,
    deep_stage_pretrain_window_secs: list[float] | tuple[float, ...] | None = None,
    deep_stage_finetune_window_secs: list[float] | tuple[float, ...] | None = None,
    central_prior_alpha: float = DEFAULT_CENTRAL_PRIOR_LOGIT_ALPHA,
    central_aux_loss_weight: float = DEFAULT_CENTRAL_PRIOR_AUX_LOSS_WEIGHT,
) -> dict[str, object]:
    """Build the optimized candidate pipelines used for validation-time model selection."""
    candidate_names = candidate_names or ["hierarchical+lda", "hierarchical+svm", "hybrid+lda", "hybrid+svm"]
    normalized_candidate_names = [str(name).strip().lower() for name in candidate_names]
    stage_pretrain_window_secs = (
        [float(value) for value in (deep_stage_pretrain_window_secs or DEFAULT_DEEP_STAGE_PRETRAIN_WINDOW_SECS)]
    )
    stage_finetune_window_secs = (
        [float(value) for value in (deep_stage_finetune_window_secs or DEFAULT_DEEP_STAGE_FINETUNE_WINDOW_SECS)]
    )
    pipelines = {}
    deep_aliases = {"eegnet", "shallow", "deep"} | DEEP_FBLIGHT_ALIASES
    full_channel_indices = (
        [int(index) for index in range(len(channel_names))]
        if channel_names is not None
        else None
    )
    central_channel_indices: list[int] | None = None
    if any(
        name.startswith("central-")
        or name.startswith("latefusion-")
        or name in CENTRAL_FBCSP_LDA_ALIASES
        or name in CENTRAL_ONLY_FBLIGHT_ALIASES
        or name in CENTRAL_GATE_FBLIGHT_ALIASES
        or name in CENTRAL_PRIOR_DUAL_BRANCH_ALIASES
        or name in CENTRAL_PRIOR_GATE_FBLIGHT_ALIASES
        for name in normalized_candidate_names
    ):
        central_channel_indices = _resolve_channel_indices(channel_names, DEFAULT_CENTRAL_CHANNEL_NAMES)
        if full_channel_indices is None:
            raise ValueError("Central/late-fusion candidates require explicit channel_names.")

    for candidate_name in candidate_names:
        normalized_name = candidate_name.lower().strip()

        if normalized_name in CENTRAL_FBCSP_LDA_ALIASES:
            if central_channel_indices is None:
                raise ValueError("central_fbcsp_lda requires channel_names with C3/Cz/C4.")
            pipelines[candidate_name] = make_pipeline(
                ChannelSelector(central_channel_indices),
                _build_standard_candidate(
                    feature_name="fbcsp",
                    classifier_name="lda",
                    fs=fs,
                    bands=CENTRAL_FBCSP_LDA_BANDS,
                    n_components=int(CENTRAL_FBCSP_LDA_COMPONENTS),
                    riemann_band=riemann_band,
                    estimator=estimator,
                    metric=metric,
                    kernel=kernel,
                    C=C,
                ),
            )
            continue

        if normalized_name in (CENTRAL_ONLY_FBLIGHT_ALIASES | CENTRAL_GATE_FBLIGHT_ALIASES):
            if central_channel_indices is None:
                raise ValueError("central_only_fblight requires channel_names with C3/Cz/C4.")
            pipelines[candidate_name] = make_pipeline(
                ChannelSelector(central_channel_indices),
                TorchEEGClassifier(
                    architecture="fblight_tcn",
                    n_channels=len(central_channel_indices),
                    n_classes=None,
                    epochs=torch_epochs,
                    batch_size=torch_batch_size,
                    learning_rate=torch_learning_rate,
                    weight_decay=torch_weight_decay,
                    validation_split=torch_validation_split,
                    patience=torch_patience,
                    random_state=42,
                    device=torch_device,
                    sampling_rate=fs,
                    stage_pretrain_window_secs=stage_pretrain_window_secs,
                    stage_finetune_window_secs=stage_finetune_window_secs,
                ),
            )
            continue

        if normalized_name in FULL8_FBLIGHT_ALIASES:
            if full_channel_indices is None:
                raise ValueError("full8_fblight requires explicit channel_names.")
            if len(full_channel_indices) != 8:
                raise ValueError(
                    f"full8_fblight expects exactly 8 channels, got {len(full_channel_indices)}: {list(channel_names or [])}"
                )
            pipelines[candidate_name] = TorchEEGClassifier(
                architecture="fblight_tcn",
                n_channels=len(full_channel_indices),
                n_classes=None,
                epochs=torch_epochs,
                batch_size=torch_batch_size,
                learning_rate=torch_learning_rate,
                weight_decay=torch_weight_decay,
                validation_split=torch_validation_split,
                patience=torch_patience,
                random_state=42,
                device=torch_device,
                sampling_rate=fs,
                stage_pretrain_window_secs=stage_pretrain_window_secs,
                stage_finetune_window_secs=stage_finetune_window_secs,
            )
            continue

        if normalized_name in (CENTRAL_PRIOR_DUAL_BRANCH_ALIASES | CENTRAL_PRIOR_GATE_FBLIGHT_ALIASES):
            if central_channel_indices is None or full_channel_indices is None:
                raise ValueError("central_prior_dual_branch_fblight_tcn requires explicit channel_names.")
            if len(full_channel_indices) != 8:
                raise ValueError(
                    "central_prior_dual_branch_fblight_tcn expects exactly 8 full-channel inputs."
                )
            pipelines[candidate_name] = TorchEEGClassifier(
                architecture="dual_branch_fblight_tcn",
                n_channels=len(full_channel_indices),
                n_classes=None,
                epochs=torch_epochs,
                batch_size=torch_batch_size,
                learning_rate=torch_learning_rate,
                weight_decay=torch_weight_decay,
                validation_split=torch_validation_split,
                patience=torch_patience,
                random_state=42,
                device=torch_device,
                sampling_rate=fs,
                stage_pretrain_window_secs=stage_pretrain_window_secs,
                stage_finetune_window_secs=stage_finetune_window_secs,
                central_prior_alpha=float(central_prior_alpha),
                central_aux_loss_weight=float(central_aux_loss_weight),
                dual_branch_central_indices=[int(index) for index in central_channel_indices],
            )
            continue

        if normalized_name in deep_aliases:
            pipelines[candidate_name] = _build_torch_candidate(
                normalized_name,
                torch_epochs=torch_epochs,
                torch_batch_size=torch_batch_size,
                torch_learning_rate=torch_learning_rate,
                torch_weight_decay=torch_weight_decay,
                torch_patience=torch_patience,
                torch_validation_split=torch_validation_split,
                torch_device=torch_device,
                sampling_rate=fs,
                stage_pretrain_window_secs=stage_pretrain_window_secs,
                stage_finetune_window_secs=stage_finetune_window_secs,
                central_prior_alpha=float(central_prior_alpha),
                central_aux_loss_weight=float(central_aux_loss_weight),
            )
            continue

        if normalized_name.startswith("central-"):
            if central_channel_indices is None:
                raise ValueError("Central-only candidates require channel_names.")
            branch_name = normalized_name[len("central-") :]
            if branch_name in deep_aliases:
                pipelines[candidate_name] = make_pipeline(
                    ChannelSelector(central_channel_indices),
                    _build_torch_candidate(
                        branch_name,
                        torch_epochs=torch_epochs,
                        torch_batch_size=torch_batch_size,
                        torch_learning_rate=torch_learning_rate,
                        torch_weight_decay=torch_weight_decay,
                        torch_patience=torch_patience,
                        torch_validation_split=torch_validation_split,
                        torch_device=torch_device,
                        sampling_rate=fs,
                        stage_pretrain_window_secs=stage_pretrain_window_secs,
                        stage_finetune_window_secs=stage_finetune_window_secs,
                        central_prior_alpha=float(central_prior_alpha),
                        central_aux_loss_weight=float(central_aux_loss_weight),
                    ),
                )
                continue

            if "+" not in branch_name:
                raise ValueError(
                    f"Invalid central candidate name '{candidate_name}'. Expected format central-<feature>+<classifier>."
                )
            feature_name, classifier_name = branch_name.split("+", 1)
            pipelines[candidate_name] = make_pipeline(
                ChannelSelector(central_channel_indices),
                _build_standard_candidate(
                    feature_name=feature_name,
                    classifier_name=classifier_name,
                    fs=fs,
                    bands=bands,
                    n_components=n_components,
                    riemann_band=riemann_band,
                    estimator=estimator,
                    metric=metric,
                    kernel=kernel,
                    C=C,
                ),
            )
            continue

        if normalized_name.startswith("latefusion-"):
            if central_channel_indices is None or full_channel_indices is None:
                raise ValueError("Late-fusion candidates require channel_names.")
            branch_name = normalized_name[len("latefusion-") :]
            if branch_name in deep_aliases:
                central_estimator = _build_torch_candidate(
                    branch_name,
                    torch_epochs=torch_epochs,
                    torch_batch_size=torch_batch_size,
                    torch_learning_rate=torch_learning_rate,
                    torch_weight_decay=torch_weight_decay,
                    torch_patience=torch_patience,
                    torch_validation_split=torch_validation_split,
                    torch_device=torch_device,
                    sampling_rate=fs,
                    stage_pretrain_window_secs=stage_pretrain_window_secs,
                    stage_finetune_window_secs=stage_finetune_window_secs,
                    central_prior_alpha=float(central_prior_alpha),
                    central_aux_loss_weight=float(central_aux_loss_weight),
                )
                full_estimator = _build_torch_candidate(
                    branch_name,
                    torch_epochs=torch_epochs,
                    torch_batch_size=torch_batch_size,
                    torch_learning_rate=torch_learning_rate,
                    torch_weight_decay=torch_weight_decay,
                    torch_patience=torch_patience,
                    torch_validation_split=torch_validation_split,
                    torch_device=torch_device,
                    sampling_rate=fs,
                    stage_pretrain_window_secs=stage_pretrain_window_secs,
                    stage_finetune_window_secs=stage_finetune_window_secs,
                    central_prior_alpha=float(central_prior_alpha),
                    central_aux_loss_weight=float(central_aux_loss_weight),
                )
            else:
                if "+" not in branch_name:
                    raise ValueError(
                        "Invalid late-fusion candidate name "
                        f"'{candidate_name}'. Expected latefusion-<feature>+<classifier> or latefusion-<deep_model>."
                    )
                feature_name, classifier_name = branch_name.split("+", 1)
                central_estimator = _build_standard_candidate(
                    feature_name=feature_name,
                    classifier_name=classifier_name,
                    fs=fs,
                    bands=bands,
                    n_components=n_components,
                    riemann_band=riemann_band,
                    estimator=estimator,
                    metric=metric,
                    kernel=kernel,
                    C=C,
                )
                full_estimator = _build_standard_candidate(
                    feature_name=feature_name,
                    classifier_name=classifier_name,
                    fs=fs,
                    bands=bands,
                    n_components=n_components,
                    riemann_band=riemann_band,
                    estimator=estimator,
                    metric=metric,
                    kernel=kernel,
                    C=C,
                )
            pipelines[candidate_name] = LateFusionMIClassifier(
                central_estimator=central_estimator,
                full_estimator=full_estimator,
                central_indices=central_channel_indices,
                full_indices=full_channel_indices,
                central_weight=DEFAULT_CENTRAL_BRANCH_WEIGHT,
                fusion_method="log_weighted_mean",
            )
            continue

        if "+" not in normalized_name:
            raise ValueError(
                f"Invalid candidate name '{candidate_name}'. Expected <feature>+<classifier> or a deep alias."
            )
        feature_name, classifier_name = normalized_name.split("+", 1)
        pipelines[candidate_name] = _build_standard_candidate(
            feature_name=feature_name,
            classifier_name=classifier_name,
            fs=fs,
            bands=bands,
            n_components=n_components,
            riemann_band=riemann_band,
            estimator=estimator,
            metric=metric,
            kernel=kernel,
            C=C,
        )

    return pipelines


class LateFusionMIClassifier(BaseEstimator, ClassifierMixin):
    """Two-branch late fusion classifier: central channels + full channels."""

    def __init__(
        self,
        *,
        central_estimator,
        full_estimator,
        central_indices: list[int] | tuple[int, ...],
        full_indices: list[int] | tuple[int, ...],
        central_weight: float = DEFAULT_CENTRAL_BRANCH_WEIGHT,
        fusion_method: str = "log_weighted_mean",
    ) -> None:
        self.central_estimator = central_estimator
        self.full_estimator = full_estimator
        self.central_indices = central_indices
        self.full_indices = full_indices
        self.central_weight = central_weight
        self.fusion_method = fusion_method
        self.central_model_ = None
        self.full_model_ = None
        self.classes_ = None

    @staticmethod
    def _validate_indices(indices: list[int] | tuple[int, ...], channel_count: int, branch_name: str) -> tuple[int, ...]:
        resolved = tuple(int(index) for index in indices)
        if not resolved:
            raise ValueError(f"{branch_name} requires at least one channel index.")
        invalid = [index for index in resolved if index < 0 or index >= channel_count]
        if invalid:
            raise ValueError(f"{branch_name} indices out of range for channel_count={channel_count}: {invalid}")
        return resolved

    @staticmethod
    def _normalize_branch_weights(central_weight: float) -> tuple[float, float]:
        central = float(np.clip(float(central_weight), 0.0, 1.0))
        full = float(1.0 - central)
        if central <= 0.0 and full <= 0.0:
            return 0.5, 0.5
        if central <= 0.0:
            return 0.0, 1.0
        if full <= 0.0:
            return 1.0, 0.0
        return central, full

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        if X.ndim != 3:
            raise ValueError("LateFusionMIClassifier expects X with shape (trials, channels, samples).")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of trials.")

        channel_count = int(X.shape[1])
        self.central_indices_ = self._validate_indices(self.central_indices, channel_count, "central branch")
        self.full_indices_ = self._validate_indices(self.full_indices, channel_count, "full branch")
        self.classes_ = np.unique(y).astype(np.int64)
        class_count = int(self.classes_.shape[0])
        if class_count < 2:
            raise ValueError("LateFusionMIClassifier requires at least two classes.")

        self.central_model_ = copy.deepcopy(self.central_estimator)
        self.full_model_ = copy.deepcopy(self.full_estimator)
        self.central_model_.fit(np.asarray(X[:, self.central_indices_, :], dtype=np.float32), y)
        self.full_model_.fit(np.asarray(X[:, self.full_indices_, :], dtype=np.float32), y)
        return self

    def predict_proba(self, X):
        if self.central_model_ is None or self.full_model_ is None or self.classes_ is None:
            raise RuntimeError("LateFusionMIClassifier must be fitted before calling predict_proba().")

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError("LateFusionMIClassifier expects X with shape (trials, channels, samples).")

        class_count = int(self.classes_.shape[0])
        central_probs = _probability_like_matrix(
            self.central_model_,
            np.asarray(X[:, self.central_indices_, :], dtype=np.float32),
            class_count,
        )
        full_probs = _probability_like_matrix(
            self.full_model_,
            np.asarray(X[:, self.full_indices_, :], dtype=np.float32),
            class_count,
        )

        central_classes = getattr(self.central_model_, "classes_", self.classes_)
        full_classes = getattr(self.full_model_, "classes_", self.classes_)
        central_aligned = _align_probability_columns(
            central_probs,
            np.asarray(central_classes, dtype=np.int64),
            self.classes_,
        )
        full_aligned = _align_probability_columns(
            full_probs,
            np.asarray(full_classes, dtype=np.int64),
            self.classes_,
        )

        central_weight, full_weight = self._normalize_branch_weights(self.central_weight)
        method = str(self.fusion_method).strip().lower()
        if method == "log_weighted_mean":
            clipped_central = np.clip(central_aligned, 1e-8, 1.0)
            clipped_full = np.clip(full_aligned, 1e-8, 1.0)
            fused = np.exp((central_weight * np.log(clipped_central)) + (full_weight * np.log(clipped_full)))
        else:
            fused = (central_weight * central_aligned) + (full_weight * full_aligned)

        fused = np.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
        row_sums = np.sum(fused, axis=1, keepdims=True)
        row_sums[row_sums <= 0.0] = 1.0
        return fused / row_sums

    def predict(self, X):
        probabilities = self.predict_proba(X)
        indices = np.argmax(probabilities, axis=1)
        return self.classes_[indices]


class LDAModel:
    """LDA baseline on flattened trials."""

    def __init__(self) -> None:
        self.model = build_classical_pipeline("lda")

    def fit(self, X, y) -> None:
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class SVMModel:
    """SVM baseline on flattened trials."""

    def __init__(self, *, kernel: str = "rbf", C: float = 1.0) -> None:
        self.model = make_pipeline(
            FlattenTransformer(),
            StandardScaler(),
            _build_classifier("svm", kernel=kernel, C=C, probability=True),
        )

    def fit(self, X, y) -> None:
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


if nn is not None:

    class EEGNet(nn.Module):
        """Compact EEGNet-style model."""

        def __init__(
            self,
            n_channels: int = 22,
            n_classes: int = 4,
            *,
            F1: int = 8,
            D: int = 2,
            F2: int = 16,
            kernel_length: int = 64,
            dropout_rate: float = 0.5,
        ) -> None:
            super().__init__()
            self.temporal = nn.Sequential(
                nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
                nn.BatchNorm2d(F1),
            )
            self.spatial = nn.Sequential(
                nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
                nn.BatchNorm2d(F1 * D),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4)),
                nn.Dropout(dropout_rate),
            )
            self.separable = nn.Sequential(
                nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
                nn.Conv2d(F1 * D, F2, kernel_size=1, bias=False),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 8)),
                nn.Dropout(dropout_rate),
                nn.AdaptiveAvgPool2d((1, 16)),
            )
            self.classifier = nn.Linear(F2 * 16, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.unsqueeze(1)
            x = self.temporal(x)
            x = self.spatial(x)
            x = self.separable(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)


    class ShallowConvNet(nn.Module):
        """Shallow ConvNet inspired by Schirrmeister et al."""

        def __init__(self, n_channels: int = 22, n_classes: int = 4, *, dropout_rate: float = 0.5) -> None:
            super().__init__()
            self.conv_time = nn.Conv2d(1, 40, kernel_size=(1, 25), bias=False)
            self.conv_spatial = nn.Conv2d(40, 40, kernel_size=(n_channels, 1), bias=False)
            self.bn = nn.BatchNorm2d(40)
            self.pool = nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7))
            self.dropout = nn.Dropout(dropout_rate)
            self.summary_pool = nn.AdaptiveAvgPool2d((1, 16))
            self.classifier = nn.Linear(40 * 16, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.unsqueeze(1)
            x = self.conv_time(x)
            x = self.conv_spatial(x)
            x = self.bn(x)
            x = x.square()
            x = self.pool(x)
            x = torch.log(torch.clamp(x, min=1e-6))
            x = self.dropout(x)
            x = self.summary_pool(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)


    class DeepConvNet(nn.Module):
        """A deeper convolutional baseline."""

        def __init__(self, n_channels: int = 22, n_classes: int = 4, *, dropout_rate: float = 0.5) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, 25, kernel_size=(1, 10), bias=False),
                nn.Conv2d(25, 25, kernel_size=(n_channels, 1), bias=False),
                nn.BatchNorm2d(25),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3)),
                nn.Dropout(dropout_rate),
            )
            self.blocks = nn.Sequential(
                nn.Conv2d(25, 50, kernel_size=(1, 10), bias=False),
                nn.BatchNorm2d(50),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3)),
                nn.Dropout(dropout_rate),
                nn.Conv2d(50, 100, kernel_size=(1, 10), bias=False),
                nn.BatchNorm2d(100),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3)),
                nn.Dropout(dropout_rate),
                nn.AdaptiveAvgPool2d((1, 16)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(100 * 16, 128),
                nn.ELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.unsqueeze(1)
            x = self.stem(x)
            x = self.blocks(x)
            return self.classifier(x)


    class TemporalResidualBlock(nn.Module):
        """Lightweight residual temporal block used by FBLight backbones."""

        def __init__(self, channels: int, *, dilation: int, dropout_rate: float) -> None:
            super().__init__()
            padding = int(max(1, dilation))
            self.block = nn.Sequential(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=3,
                    dilation=int(dilation),
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm1d(channels),
                nn.ELU(),
                nn.Dropout(dropout_rate),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=3,
                    dilation=int(dilation),
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm1d(channels),
            )
            self.activation = nn.ELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            output = self.block(x)
            if output.shape[-1] != residual.shape[-1]:
                target_length = int(min(output.shape[-1], residual.shape[-1]))
                output = output[..., :target_length]
                residual = residual[..., :target_length]
            return self.activation(output + residual)


    class FBLightTCNBackbone(nn.Module):
        """Filter-bank-light temporal-spatial backbone with a tiny TCN head."""

        def __init__(
            self,
            *,
            n_channels: int,
            temporal_kernels: tuple[int, ...] = (15, 31, 63),
            branch_width: int = 12,
            tcn_channels: int = 32,
            dropout_rate: float = 0.35,
        ) -> None:
            super().__init__()
            if int(n_channels) <= 0:
                raise ValueError(f"n_channels must be positive, got {n_channels}.")
            self.n_channels = int(n_channels)
            self.temporal_branches = nn.ModuleList(
                [
                    nn.Conv2d(
                        1,
                        int(branch_width),
                        kernel_size=(1, int(kernel_size)),
                        padding=(0, int(kernel_size) // 2),
                        bias=False,
                    )
                    for kernel_size in temporal_kernels
                ]
            )
            temporal_out = int(branch_width) * len(temporal_kernels)
            self.temporal_bn = nn.BatchNorm2d(temporal_out)
            self.spatial_depthwise = nn.Conv2d(
                temporal_out,
                temporal_out,
                kernel_size=(self.n_channels, 1),
                groups=temporal_out,
                bias=False,
            )
            self.spatial_pointwise = nn.Conv2d(temporal_out, int(tcn_channels), kernel_size=1, bias=False)
            self.spatial_bn = nn.BatchNorm2d(int(tcn_channels))
            self.energy_pool = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 2))
            self.dropout = nn.Dropout(dropout_rate)
            self.tcn = nn.Sequential(
                TemporalResidualBlock(int(tcn_channels), dilation=1, dropout_rate=float(dropout_rate)),
                TemporalResidualBlock(int(tcn_channels), dilation=2, dropout_rate=float(dropout_rate)),
            )
            self.output_dim = int(tcn_channels) * 2

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.unsqueeze(1)
            temporal_features = [branch(x) for branch in self.temporal_branches]
            x = torch.cat(temporal_features, dim=1)
            x = self.temporal_bn(x)
            x = F.elu(x)
            x = self.spatial_depthwise(x)
            x = self.spatial_pointwise(x)
            x = self.spatial_bn(x)
            x = F.elu(x)
            x = x.square()
            x = self.energy_pool(x)
            x = self.dropout(x)
            x = x.squeeze(2)
            x = self.tcn(x)
            pooled_mean = x.mean(dim=-1)
            pooled_max = x.amax(dim=-1)
            return torch.cat([pooled_mean, pooled_max], dim=1)


    class FBLightTCNClassifier(nn.Module):
        """Single-branch FBLight+TCN classifier."""

        def __init__(self, *, n_channels: int, n_classes: int, dropout_rate: float = 0.35) -> None:
            super().__init__()
            self.backbone = FBLightTCNBackbone(
                n_channels=int(n_channels),
                branch_width=12,
                tcn_channels=32,
                dropout_rate=float(dropout_rate),
            )
            self.classifier = nn.Linear(self.backbone.output_dim, int(n_classes))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.backbone(x)
            return self.classifier(features)


    class CentralPriorDualBranchFBLightTCN(nn.Module):
        """Central-prior dual-branch network with late-logit fusion and central auxiliary head."""

        def __init__(
            self,
            *,
            n_channels: int,
            n_classes: int,
            central_indices: list[int] | tuple[int, ...],
            central_alpha: float = DEFAULT_CENTRAL_PRIOR_LOGIT_ALPHA,
            dropout_rate: float = 0.35,
        ) -> None:
            super().__init__()
            self.n_channels = int(n_channels)
            self.central_indices = tuple(int(index) for index in central_indices)
            if not self.central_indices:
                raise ValueError("central_indices cannot be empty for dual-branch model.")
            invalid = [index for index in self.central_indices if index < 0 or index >= self.n_channels]
            if invalid:
                raise ValueError(f"central_indices out of range for n_channels={self.n_channels}: {invalid}")
            self.central_alpha = float(np.clip(float(central_alpha), 0.0, 1.0))
            self.central_backbone = FBLightTCNBackbone(
                n_channels=len(self.central_indices),
                branch_width=12,
                tcn_channels=32,
                dropout_rate=float(dropout_rate),
            )
            self.full_backbone = FBLightTCNBackbone(
                n_channels=self.n_channels,
                branch_width=6,
                tcn_channels=16,
                dropout_rate=float(dropout_rate),
            )
            self.central_head = nn.Linear(self.central_backbone.output_dim, int(n_classes))
            self.full_head = nn.Linear(self.full_backbone.output_dim, int(n_classes))

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            central_input = x[:, self.central_indices, :]
            central_features = self.central_backbone(central_input)
            full_features = self.full_backbone(x)
            central_logits = self.central_head(central_features)
            full_logits = self.full_head(full_features)
            fused_logits = (self.central_alpha * central_logits) + ((1.0 - self.central_alpha) * full_logits)
            return {
                "logits": fused_logits,
                "central_logits": central_logits,
                "full_logits": full_logits,
            }


    class TorchEEGClassifier(BaseEstimator, ClassifierMixin):
        """sklearn-style wrapper around compact torch EEG classifiers."""

        def __init__(
            self,
            *,
            architecture: str = "eegnet",
            n_channels: int | None = None,
            n_classes: int | None = None,
            epochs: int = 80,
            batch_size: int = 32,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            dropout_rate: float = 0.5,
            validation_split: float = 0.15,
            patience: int = 12,
            random_state: int = 42,
            device: str | None = None,
            sampling_rate: float | None = None,
            stage_pretrain_window_secs: list[float] | tuple[float, ...] | None = None,
            stage_finetune_window_secs: list[float] | tuple[float, ...] | None = None,
            central_prior_alpha: float = DEFAULT_CENTRAL_PRIOR_LOGIT_ALPHA,
            central_aux_loss_weight: float = DEFAULT_CENTRAL_PRIOR_AUX_LOSS_WEIGHT,
            dual_branch_central_indices: list[int] | tuple[int, ...] | None = None,
            verbose: bool = False,
        ) -> None:
            self.architecture = str(architecture).strip().lower()
            self.n_channels = n_channels
            self.n_classes = n_classes
            self.epochs = int(epochs)
            self.batch_size = int(batch_size)
            self.learning_rate = float(learning_rate)
            self.weight_decay = float(weight_decay)
            self.dropout_rate = float(dropout_rate)
            self.validation_split = float(validation_split)
            self.patience = int(patience)
            self.random_state = int(random_state)
            self.device = device
            self.sampling_rate = None if sampling_rate is None else float(sampling_rate)
            self.stage_pretrain_window_secs = (
                tuple(float(value) for value in (stage_pretrain_window_secs or ()))
            )
            self.stage_finetune_window_secs = (
                tuple(float(value) for value in (stage_finetune_window_secs or ()))
            )
            self.central_prior_alpha = float(central_prior_alpha)
            self.central_aux_loss_weight = float(central_aux_loss_weight)
            self.dual_branch_central_indices = (
                tuple(int(index) for index in dual_branch_central_indices)
                if dual_branch_central_indices is not None
                else None
            )
            self.verbose = bool(verbose)
            self.model_ = None
            self.classes_ = None

        def _build_network(self, *, n_channels: int, n_classes: int):
            if self.architecture == "eegnet":
                return EEGNet(
                    n_channels=n_channels,
                    n_classes=n_classes,
                    dropout_rate=self.dropout_rate,
                )
            if self.architecture == "shallow":
                return ShallowConvNet(
                    n_channels=n_channels,
                    n_classes=n_classes,
                    dropout_rate=self.dropout_rate,
                )
            if self.architecture == "deep":
                return DeepConvNet(
                    n_channels=n_channels,
                    n_classes=n_classes,
                    dropout_rate=self.dropout_rate,
                )
            if self.architecture in DEEP_FBLIGHT_ALIASES:
                return FBLightTCNClassifier(
                    n_channels=n_channels,
                    n_classes=n_classes,
                    dropout_rate=self.dropout_rate,
                )
            if self.architecture == "dual_branch_fblight_tcn":
                if self.dual_branch_central_indices is None:
                    raise ValueError("dual_branch_fblight_tcn requires dual_branch_central_indices.")
                return CentralPriorDualBranchFBLightTCN(
                    n_channels=n_channels,
                    n_classes=n_classes,
                    central_indices=list(self.dual_branch_central_indices),
                    central_alpha=float(self.central_prior_alpha),
                    dropout_rate=self.dropout_rate,
                )
            raise ValueError(f"Unsupported torch EEG architecture: {self.architecture}")

        @staticmethod
        def _tail_crop(X: np.ndarray, window_samples: int) -> np.ndarray:
            target = int(max(1, window_samples))
            if int(X.shape[-1]) <= target:
                return np.asarray(X, dtype=np.float32)
            return np.asarray(X[:, :, -target:], dtype=np.float32)

        @staticmethod
        def _extract_logits(model_output):
            if isinstance(model_output, dict):
                return model_output["logits"]
            return model_output

        def _compute_loss(self, model_output, labels: torch.Tensor, criterion) -> torch.Tensor:
            logits = self._extract_logits(model_output)
            loss = criterion(logits, labels)
            if isinstance(model_output, dict) and "central_logits" in model_output:
                loss = loss + float(self.central_aux_loss_weight) * criterion(model_output["central_logits"], labels)
            return loss

        def _resolve_stage_window_groups(self, n_samples: int) -> list[list[int]]:
            if self.sampling_rate is None or float(self.sampling_rate) <= 0.0:
                return [[int(n_samples)]]

            def _to_stage(samples_or_secs: tuple[float, ...]) -> list[int]:
                values: list[int] = []
                for sec in samples_or_secs:
                    if float(sec) <= 0.0:
                        continue
                    samples = int(round(float(sec) * float(self.sampling_rate)))
                    samples = int(max(1, min(int(n_samples), samples)))
                    values.append(samples)
                return values

            groups: list[list[int]] = []
            pretrain_group = _to_stage(self.stage_pretrain_window_secs)
            finetune_group = _to_stage(self.stage_finetune_window_secs)
            if pretrain_group:
                groups.append(pretrain_group)
            if finetune_group:
                groups.append(finetune_group)
            if not groups:
                groups = [[int(n_samples)]]
            return groups

        def fit(self, X, y):
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.int64).reshape(-1)
            if X.ndim != 3:
                raise ValueError("TorchEEGClassifier expects X with shape (trials, channels, samples).")
            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y must contain the same number of trials.")

            self.classes_, y_indices = np.unique(y, return_inverse=True)
            class_count = int(self.classes_.shape[0])
            if class_count < 2:
                raise ValueError("TorchEEGClassifier requires at least two classes.")

            n_channels = int(self.n_channels or X.shape[1])
            if X.shape[1] != n_channels:
                raise ValueError(f"Expected {n_channels} channels, got {X.shape[1]}.")

            device_name = _resolve_torch_device(self.device)
            torch.manual_seed(int(self.random_state))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.random_state))

            train_X = X
            train_y = y_indices
            val_X = None
            val_y = None

            unique_labels, label_counts = np.unique(y_indices, return_counts=True)
            proposed_val_size = int(np.ceil(X.shape[0] * float(self.validation_split)))
            can_split = (
                unique_labels.shape[0] == class_count
                and np.min(label_counts) >= 2
                and X.shape[0] >= max(class_count * 2, 8)
                and self.validation_split > 0.0
                and proposed_val_size >= class_count
                and (X.shape[0] - proposed_val_size) >= class_count
            )
            if can_split:
                train_X, val_X, train_y, val_y = train_test_split(
                    X,
                    y_indices,
                    test_size=float(self.validation_split),
                    random_state=int(self.random_state),
                    stratify=y_indices,
                )

            model = self._build_network(n_channels=n_channels, n_classes=class_count)
            model.to(device_name)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(self.learning_rate),
                weight_decay=float(self.weight_decay),
            )
            criterion = nn.CrossEntropyLoss()

            best_state = None
            best_score = None
            bad_epochs = 0
            stage_groups = self._resolve_stage_window_groups(int(X.shape[-1]))
            self.stage_window_groups_samples_ = [[int(value) for value in group] for group in stage_groups]
            schedule_steps: list[tuple[int, int]] = []
            for stage_index, group in enumerate(stage_groups):
                for window_samples in group:
                    schedule_steps.append((int(stage_index), int(window_samples)))
            if not schedule_steps:
                schedule_steps = [(0, int(X.shape[-1]))]
            epochs_total = max(1, int(self.epochs))
            per_step_epochs = max(1, int(np.ceil(epochs_total / len(schedule_steps))))
            epoch_plan: list[tuple[int, int]] = []
            for step in schedule_steps:
                epoch_plan.extend([step] * per_step_epochs)
            epoch_plan = epoch_plan[:epochs_total]

            current_stage_index = 0
            for epoch_index, (stage_index, window_samples) in enumerate(epoch_plan):
                if stage_index != current_stage_index:
                    for group in optimizer.param_groups:
                        group["lr"] = max(float(self.learning_rate) * 0.1, float(group["lr"]) * 0.5)
                    current_stage_index = stage_index

                current_train_X = self._tail_crop(train_X, int(window_samples))
                train_dataset = TensorDataset(
                    torch.from_numpy(np.asarray(current_train_X, dtype=np.float32)),
                    torch.from_numpy(np.asarray(train_y, dtype=np.int64)),
                )
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=max(1, min(int(self.batch_size), len(train_dataset))),
                    shuffle=True,
                )

                model.train()
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device_name)
                    batch_y = batch_y.to(device_name)
                    optimizer.zero_grad()
                    model_output = model(batch_X)
                    loss = self._compute_loss(model_output, batch_y, criterion)
                    loss.backward()
                    optimizer.step()

                score = None
                if val_X is not None and val_y is not None and len(val_y) > 0:
                    current_val_X = self._tail_crop(val_X, int(window_samples))
                    val_dataset = TensorDataset(
                        torch.from_numpy(np.asarray(current_val_X, dtype=np.float32)),
                        torch.from_numpy(np.asarray(val_y, dtype=np.int64)),
                    )
                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=max(1, min(int(self.batch_size), len(val_dataset))),
                        shuffle=False,
                    )
                    model.eval()
                    total_loss = 0.0
                    batch_count = 0
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            batch_X = batch_X.to(device_name)
                            batch_y = batch_y.to(device_name)
                            model_output = model(batch_X)
                            loss = self._compute_loss(model_output, batch_y, criterion)
                            total_loss += float(loss.item())
                            batch_count += 1
                    score = total_loss / max(batch_count, 1)
                else:
                    score = 0.0

                if best_score is None or float(score) < float(best_score):
                    best_score = float(score)
                    best_state = copy.deepcopy(model.state_dict())
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if val_X is not None and val_y is not None and bad_epochs >= max(1, int(self.patience)):
                        break

                if self.verbose:
                    print(
                        f"[TorchEEGClassifier] epoch={epoch_index + 1}/{epochs_total} "
                        f"stage={stage_index + 1} window_samples={int(window_samples)} score={float(score):.6f}"
                    )

            if best_state is not None:
                model.load_state_dict(best_state)
            model.to("cpu")
            model.eval()

            self.model_ = model
            self.n_channels_ = n_channels
            self.n_classes_ = class_count
            self.device_name_ = "cpu"
            return self

        def predict_proba(self, X):
            if self.model_ is None or self.classes_ is None:
                raise RuntimeError("TorchEEGClassifier must be fitted before calling predict_proba().")
            X = np.asarray(X, dtype=np.float32)
            if X.ndim != 3:
                raise ValueError("TorchEEGClassifier expects X with shape (trials, channels, samples).")
            self.model_.eval()
            with torch.no_grad():
                inputs = torch.from_numpy(X.astype(np.float32))
                model_output = self.model_(inputs)
                logits = self._extract_logits(model_output)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            return np.asarray(probabilities, dtype=np.float64)

        def predict(self, X):
            probabilities = self.predict_proba(X)
            predictions = np.argmax(probabilities, axis=1)
            return self.classes_[predictions]

else:

    def _raise_missing_torch() -> None:
        raise ModuleNotFoundError(
            "PyTorch is required for deep models. Install `torch` to use eegnet, shallow, deep, or fblight variants."
        )


    class EEGNet:
        """Fallback used when PyTorch is not installed."""

        def __init__(self, *args, **kwargs) -> None:
            _raise_missing_torch()


    class ShallowConvNet:
        """Fallback used when PyTorch is not installed."""

        def __init__(self, *args, **kwargs) -> None:
            _raise_missing_torch()


    class DeepConvNet:
        """Fallback used when PyTorch is not installed."""

        def __init__(self, *args, **kwargs) -> None:
            _raise_missing_torch()


    class FBLightTCNClassifier:
        """Fallback used when PyTorch is not installed."""

        def __init__(self, *args, **kwargs) -> None:
            _raise_missing_torch()


    class CentralPriorDualBranchFBLightTCN:
        """Fallback used when PyTorch is not installed."""

        def __init__(self, *args, **kwargs) -> None:
            _raise_missing_torch()


    class TorchEEGClassifier:
        """Fallback wrapper used when PyTorch is not installed."""

        def __init__(self, *args, **kwargs) -> None:
            _raise_missing_torch()


def get_model(model_name: str, **kwargs):
    """Factory for supported models."""
    model_name = model_name.lower()
    if model_name == "lda":
        return LDAModel()
    if model_name == "svm":
        return SVMModel(kernel=kwargs.get("kernel", "rbf"), C=kwargs.get("C", 1.0))
    if model_name == "eegnet":
        return EEGNet(n_channels=kwargs.get("n_channels", 22), n_classes=kwargs.get("n_classes", 4))
    if model_name == "shallow":
        return ShallowConvNet(n_channels=kwargs.get("n_channels", 22), n_classes=kwargs.get("n_classes", 4))
    if model_name == "deep":
        return DeepConvNet(n_channels=kwargs.get("n_channels", 22), n_classes=kwargs.get("n_classes", 4))
    if model_name in (DEEP_FBLIGHT_ALIASES | CENTRAL_GATE_FBLIGHT_ALIASES | CENTRAL_ONLY_FBLIGHT_ALIASES):
        return FBLightTCNClassifier(
            n_channels=kwargs.get("n_channels", 22),
            n_classes=kwargs.get("n_classes", 4),
            dropout_rate=kwargs.get("dropout_rate", 0.35),
        )
    raise ValueError(f"Unsupported model type: {model_name}")
