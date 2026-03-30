"""Dataset loading helpers for BCI Competition IV 2a."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASETS_ROOT = PROJECT_ROOT / "datasets"
CACHE_ROOT = DATASETS_ROOT / "cache"
MNE_HOME_DIR = CACHE_ROOT / "mne_home"
MOABB_DATA_DIR = CACHE_ROOT / "moabb_data"

os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(MNE_HOME_DIR))
os.environ.setdefault("MNE_DATA", str(MOABB_DATA_DIR))
(MNE_HOME_DIR / ".mne").mkdir(parents=True, exist_ok=True)
MOABB_DATA_DIR.mkdir(parents=True, exist_ok=True)

import mne
import numpy as np

try:
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    BNCI2014_001 = None
    MotorImagery = None


GDF_EVENT_CODES = {
    "769": 0,
    "770": 1,
    "771": 2,
    "772": 3,
}


def extract_dataset(archive_path: str | Path, extract_dir: str | Path) -> Path:
    """Extract the dataset archive into the target directory."""
    archive_path = Path(archive_path)
    extract_dir = Path(extract_dir)

    if not archive_path.exists():
        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zip_file:
        zip_file.extractall(extract_dir)

    return extract_dir


def load_gdf_file(
    gdf_path: str | Path,
    *,
    tmin: float = 0.5,
    tmax: float = 2.5,
    eeg_channels: int = 22,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one GDF file and return trials and labels."""
    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose=False)
    raw.pick(raw.ch_names[:eeg_channels])

    annotation_map = {description: int(description) for description in GDF_EVENT_CODES}
    events, _ = mne.events_from_annotations(raw, event_id=annotation_map, verbose=False)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=annotation_map,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )

    X = epochs.get_data().astype(np.float32)
    y = np.array([GDF_EVENT_CODES[str(code)] for code in epochs.events[:, 2]], dtype=np.int64)
    return X, y


def load_subject(
    dataset_dir: str | Path,
    subject_id: int,
    *,
    tmin: float = 0.5,
    tmax: float = 2.5,
    eeg_channels: int = 22,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train and test recordings for one subject."""
    dataset_dir = Path(dataset_dir)
    subject_prefix = f"A{subject_id:02d}"

    train_file = dataset_dir / f"{subject_prefix}T.gdf"
    test_file = dataset_dir / f"{subject_prefix}E.gdf"

    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    X_train, y_train = load_gdf_file(train_file, tmin=tmin, tmax=tmax, eeg_channels=eeg_channels)
    X_test, y_test = load_gdf_file(test_file, tmin=tmin, tmax=tmax, eeg_channels=eeg_channels)
    return X_train, y_train, X_test, y_test


def load_subject_moabb(
    subject_id: int,
    *,
    tmin: float = 0.5,
    tmax: float = 2.5,
    n_classes: int = 4,
    fmin: float = 0.5,
    fmax: float = 100.0,
    class_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one subject from MOABB with official train/test session labels."""
    if BNCI2014_001 is None or MotorImagery is None:
        raise ModuleNotFoundError("MOABB is required for data.source='moabb'. Install `moabb` first.")

    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=n_classes, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject_id], return_epochs=False)

    y = np.asarray(y)
    unique_labels = list(np.unique(y))
    if class_names is not None and all(label in class_names for label in unique_labels):
        ordered_labels = [label for label in class_names if label in unique_labels]
    else:
        ordered_labels = unique_labels
    label_to_index = {label: index for index, label in enumerate(ordered_labels)}
    y_encoded = np.array([label_to_index[label] for label in y], dtype=np.int64)

    sessions = meta["session"].to_numpy()
    train_mask = sessions == "0train"
    test_mask = sessions == "1test"

    return (
        X[train_mask].astype(np.float32),
        y_encoded[train_mask],
        X[test_mask].astype(np.float32),
        y_encoded[test_mask],
    )


def load_all_subjects(
    dataset_dir: str | Path,
    *,
    subjects: list[int] | None = None,
    tmin: float = 0.5,
    tmax: float = 2.5,
    eeg_channels: int = 22,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load all requested subjects into memory."""
    if subjects is None:
        subjects = list(range(1, 10))

    loaded = {}
    for subject_id in subjects:
        loaded[subject_id] = load_subject(
            dataset_dir,
            subject_id,
            tmin=tmin,
            tmax=tmax,
            eeg_channels=eeg_channels,
        )
    return loaded
