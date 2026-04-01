"""Feature extraction helpers used by experiment scripts."""

from __future__ import annotations

import numpy as np
from scipy import linalg, signal


def _normalized_covariance(trial: np.ndarray, regularization: float) -> np.ndarray:
    covariance = np.cov(trial)
    covariance += regularization * np.eye(covariance.shape[0], dtype=covariance.dtype)
    return covariance / np.trace(covariance)


class CSP:
    """Common Spatial Patterns for binary classification."""

    def __init__(self, n_components: int = 4, regularization: float = 1e-6) -> None:
        self.n_components = n_components
        self.regularization = regularization
        self.filters_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSP":
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP only supports binary labels.")

        class_covariances = []
        for label in classes:
            class_trials = X[y == label]
            averaged_covariance = np.mean(
                [_normalized_covariance(trial, self.regularization) for trial in class_trials],
                axis=0,
            )
            class_covariances.append(averaged_covariance)

        cov_a, cov_b = class_covariances
        eigenvalues, eigenvectors = linalg.eigh(cov_a, cov_a + cov_b)

        lower_count = self.n_components // 2
        upper_count = self.n_components - lower_count
        selected = np.concatenate(
            [
                np.arange(lower_count),
                np.arange(len(eigenvalues) - upper_count, len(eigenvalues)),
            ]
        )
        self.filters_ = eigenvectors[:, selected]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.filters_ is None:
            raise RuntimeError("CSP must be fitted before calling transform().")

        projected = np.einsum("nct,cf->nft", X, self.filters_)
        variances = np.var(projected, axis=2)
        variances = variances / np.sum(variances, axis=1, keepdims=True)
        return np.log(variances + 1e-12)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)


class FBCSP:
    """Filter-bank CSP for binary classification."""

    def __init__(
        self,
        n_components: int = 4,
        *,
        fs: float = 250.0,
        freq_bands: list[tuple[float, float]] | None = None,
    ) -> None:
        self.n_components = n_components
        self.fs = fs
        self.freq_bands = freq_bands or [
            (4, 8),
            (8, 12),
            (12, 16),
            (16, 20),
            (20, 24),
            (24, 28),
            (28, 32),
            (32, 36),
            (36, 40),
        ]
        self.band_models_: list[tuple[tuple[float, float], CSP]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FBCSP":
        self.band_models_ = []
        for band in self.freq_bands:
            filtered = self._filter_band(X, band)
            csp = CSP(n_components=self.n_components)
            csp.fit(filtered, y)
            self.band_models_.append((band, csp))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.band_models_:
            raise RuntimeError("FBCSP must be fitted before calling transform().")

        features = []
        for band, csp in self.band_models_:
            filtered = self._filter_band(X, band)
            features.append(csp.transform(filtered))
        return np.hstack(features)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def _filter_band(self, X: np.ndarray, band: tuple[float, float]) -> np.ndarray:
        sos = signal.butter(4, band, btype="bandpass", fs=self.fs, output="sos")
        return signal.sosfiltfilt(sos, X, axis=-1)


def extract_fft_features(X: np.ndarray, fs: float = 250.0, n_bands: int = 5) -> np.ndarray:
    """Extract simple FFT band-power features."""
    _, _, n_times = X.shape
    frequencies = np.fft.rfftfreq(n_times, d=1.0 / fs)
    band_edges = np.linspace(0.0, fs / 2.0, n_bands + 1)

    all_features = []
    for trial in X:
        trial_features = []
        for channel in trial:
            fft_values = np.abs(np.fft.rfft(channel))
            for index in range(n_bands):
                mask = (frequencies >= band_edges[index]) & (frequencies < band_edges[index + 1])
                trial_features.append(float(np.mean(fft_values[mask])) if np.any(mask) else 0.0)
        all_features.append(trial_features)

    return np.asarray(all_features, dtype=np.float32)
