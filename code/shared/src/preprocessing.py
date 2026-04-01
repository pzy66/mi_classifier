"""Preprocessing utilities for EEG trials."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import signal

DEFAULT_PREPROCESS_BANDPASS = (4.0, 40.0)
DEFAULT_PREPROCESS_NOTCH = 50.0
DEFAULT_PREPROCESS_APPLY_CAR = True
DEFAULT_PREPROCESS_STANDARDIZE = False


@lru_cache(maxsize=64)
def _cached_bandpass_sos(order: int, lowcut: float, highcut: float, fs: float) -> np.ndarray:
    """Cache Butterworth SOS coefficients to avoid redesigning filters repeatedly."""
    return signal.butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")


@lru_cache(maxsize=64)
def _cached_notch_ba(freq: float, quality: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Cache notch filter coefficients for repeated realtime calls."""
    b, a = signal.iirnotch(freq, quality, fs=fs)
    return b, a


def bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    *,
    order: int = 4,
) -> np.ndarray:
    """Apply a stable Butterworth bandpass filter."""
    if not 0 < lowcut < highcut < fs / 2:
        raise ValueError("Bandpass frequencies must satisfy 0 < lowcut < highcut < fs/2.")

    sos = _cached_bandpass_sos(int(order), float(lowcut), float(highcut), float(fs))
    return signal.sosfiltfilt(sos, data, axis=-1).astype(np.float32)


def notch_filter(
    data: np.ndarray,
    freq: float,
    fs: float,
    *,
    quality: float = 30.0,
) -> np.ndarray:
    """Apply a notch filter to remove line noise."""
    if not 0 < freq < fs / 2:
        raise ValueError("Notch frequency must satisfy 0 < freq < fs/2.")
    b, a = _cached_notch_ba(float(freq), float(quality), float(fs))
    return signal.filtfilt(b, a, data, axis=-1).astype(np.float32)


def common_average_reference(data: np.ndarray) -> np.ndarray:
    """Apply common average reference across channels."""
    return (data - np.mean(data, axis=-2, keepdims=True)).astype(np.float32)


def standardize(data: np.ndarray) -> np.ndarray:
    """Standardize each channel in each trial across time."""
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((data - mean) / std).astype(np.float32)


def downsample_data(data: np.ndarray, factor: int) -> np.ndarray:
    """Downsample along the time axis using striding."""
    if factor < 1:
        raise ValueError("Downsample factor must be >= 1.")
    return data[..., ::factor].astype(np.float32)


def preprocess(
    X: np.ndarray,
    *,
    fs: float = 250.0,
    bandpass: tuple[float, float] | list[float] | None = DEFAULT_PREPROCESS_BANDPASS,
    notch: float | None = DEFAULT_PREPROCESS_NOTCH,
    apply_car: bool = DEFAULT_PREPROCESS_APPLY_CAR,
    standardize_data: bool = DEFAULT_PREPROCESS_STANDARDIZE,
) -> np.ndarray:
    """Run the baseline preprocessing pipeline."""
    X = np.asarray(X, dtype=np.float32)
    if not np.all(np.isfinite(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if bandpass is not None:
        X = bandpass_filter(X, bandpass[0], bandpass[1], fs)
    if notch is not None:
        X = notch_filter(X, notch, fs)
    if apply_car:
        X = common_average_reference(X)
    if standardize_data:
        X = standardize(X)

    return X.astype(np.float32)
