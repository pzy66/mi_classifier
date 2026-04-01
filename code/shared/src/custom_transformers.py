"""Custom sklearn-compatible transformers for EEG feature extraction."""

from __future__ import annotations

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy import signal
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.features import CSP, FBCSP


class FlattenTransformer(BaseEstimator, TransformerMixin):
    """Flatten EEG trials into 2D feature vectors."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X.reshape(X.shape[0], -1)


class ChannelSelector(BaseEstimator, TransformerMixin):
    """Select a fixed subset of EEG channels."""

    def __init__(self, channel_indices: list[int] | tuple[int, ...]) -> None:
        self.channel_indices = channel_indices
        self.channel_indices_: tuple[int, ...] | None = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("ChannelSelector expects X with shape (trials, channels, samples).")

        resolved = tuple(int(index) for index in self.channel_indices)
        if not resolved:
            raise ValueError("ChannelSelector requires at least one channel index.")

        channel_count = int(X.shape[1])
        invalid = [index for index in resolved if index < 0 or index >= channel_count]
        if invalid:
            raise ValueError(
                f"ChannelSelector indices out of range for channel_count={channel_count}: {invalid}"
            )

        self.channel_indices_ = resolved
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError("ChannelSelector expects X with shape (trials, channels, samples).")
        if self.channel_indices_ is None:
            raise RuntimeError("ChannelSelector must be fitted before calling transform().")
        return np.ascontiguousarray(X[:, self.channel_indices_, :], dtype=np.float32)


class OneVsRestFBCSP(BaseEstimator, TransformerMixin):
    """Filter-bank CSP using one-vs-rest binary decompositions."""

    def __init__(
        self,
        bands: list[tuple[float, float]] | None = None,
        *,
        n_components: int = 4,
        fs: float = 250.0,
    ) -> None:
        self.bands = bands or [
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
        self.n_components = n_components
        self.fs = fs
        self.models_: list[tuple[np.ndarray, CSP]] = []
        self.classes_: np.ndarray | None = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.models_ = []

        for band in self.bands:
            sos = signal.butter(4, band, btype="bandpass", fs=self.fs, output="sos")
            X_band = signal.sosfiltfilt(sos, X, axis=-1).astype(np.float32)
            for class_label in self.classes_:
                binary_y = (y == class_label).astype(int)
                csp = CSP(n_components=self.n_components)
                csp.fit(X_band, binary_y)
                self.models_.append((sos, csp))

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        features = []
        for sos, csp in self.models_:
            X_band = signal.sosfiltfilt(sos, X, axis=-1).astype(np.float32)
            features.append(csp.transform(X_band))
        return np.hstack(features).astype(np.float32)


def _build_binary_classifier(
    classifier_name: str,
    *,
    kernel: str = "rbf",
    C: float = 1.0,
):
    """Build a binary classifier for the hierarchical MI branches."""
    classifier_name = classifier_name.lower()
    if classifier_name == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if classifier_name == "svm":
        return SVC(kernel=kernel, C=C, gamma="scale", probability=True)
    raise ValueError(f"Unsupported hierarchical classifier: {classifier_name}")


class BinaryFBCSPFeatureExtractor(BaseEstimator, TransformerMixin):
    """Binary FBCSP features for one branch of the hierarchical MI classifier."""

    def __init__(
        self,
        bands: list[tuple[float, float]] | None = None,
        *,
        n_components: int = 4,
        fs: float = 250.0,
    ) -> None:
        self.bands = bands or [
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
        self.n_components = n_components
        self.fs = fs
        self.model_: FBCSP | None = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("BinaryFBCSPFeatureExtractor requires exactly 2 classes.")

        self.model_ = FBCSP(
            n_components=self.n_components,
            fs=self.fs,
            freq_bands=list(self.bands),
        )
        self.model_.fit(X, y)
        return self

    def transform(self, X):
        if self.model_ is None:
            raise RuntimeError("BinaryFBCSPFeatureExtractor must be fitted before calling transform().")
        X = np.asarray(X, dtype=np.float32)
        return self.model_.transform(X).astype(np.float32)


class RiemannFeatureExtractor(BaseEstimator, TransformerMixin):
    """Riemannian tangent-space features on a selected frequency band."""

    def __init__(
        self,
        *,
        band: tuple[float, float] = (4.0, 40.0),
        fs: float = 250.0,
        estimator: str = "lwf",
        metric: str = "riemann",
    ) -> None:
        self.band = band
        self.fs = fs
        self.estimator = estimator
        self.metric = metric
        self.cov_estimator_: Covariances | None = None
        self.tangent_space_: TangentSpace | None = None
        self.sos_: np.ndarray | None = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        self.sos_ = signal.butter(4, self.band, btype="bandpass", fs=self.fs, output="sos")
        X_band = signal.sosfiltfilt(self.sos_, X, axis=-1).astype(np.float32)
        self.cov_estimator_ = Covariances(estimator=self.estimator)
        covariances = self.cov_estimator_.fit_transform(X_band)
        self.tangent_space_ = TangentSpace(metric=self.metric)
        self.tangent_space_.fit(covariances, y)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_band = signal.sosfiltfilt(self.sos_, X, axis=-1).astype(np.float32)
        covariances = self.cov_estimator_.transform(X_band)
        return self.tangent_space_.transform(covariances).astype(np.float32)


class HybridFeatureExtractor(BaseEstimator, TransformerMixin):
    """Concatenate FBCSP and Riemannian features."""

    def __init__(
        self,
        bands: list[tuple[float, float]] | None = None,
        *,
        n_components: int = 4,
        fs: float = 250.0,
        riemann_band: tuple[float, float] = (4.0, 40.0),
        estimator: str = "lwf",
        metric: str = "riemann",
    ) -> None:
        self.bands = bands
        self.n_components = n_components
        self.fs = fs
        self.riemann_band = riemann_band
        self.estimator = estimator
        self.metric = metric
        self.fbcsp_: OneVsRestFBCSP | None = None
        self.riemann_: RiemannFeatureExtractor | None = None

    def fit(self, X, y):
        self.fbcsp_ = OneVsRestFBCSP(self.bands, n_components=self.n_components, fs=self.fs)
        self.riemann_ = RiemannFeatureExtractor(
            band=self.riemann_band,
            fs=self.fs,
            estimator=self.estimator,
            metric=self.metric,
        )
        self.fbcsp_.fit(X, y)
        self.riemann_.fit(X, y)
        return self

    def transform(self, X):
        fbcsp_features = self.fbcsp_.transform(X)
        riemann_features = self.riemann_.transform(X)
        return np.hstack([fbcsp_features, riemann_features]).astype(np.float32)


class BinaryHybridFeatureExtractor(BaseEstimator, TransformerMixin):
    """Binary hybrid features for one branch of the hierarchical MI classifier."""

    def __init__(
        self,
        bands: list[tuple[float, float]] | None = None,
        *,
        n_components: int = 4,
        fs: float = 250.0,
        riemann_band: tuple[float, float] = (4.0, 40.0),
        estimator: str = "lwf",
        metric: str = "riemann",
    ) -> None:
        self.bands = bands
        self.n_components = n_components
        self.fs = fs
        self.riemann_band = riemann_band
        self.estimator = estimator
        self.metric = metric
        self.fbcsp_: BinaryFBCSPFeatureExtractor | None = None
        self.riemann_: RiemannFeatureExtractor | None = None

    def fit(self, X, y):
        self.fbcsp_ = BinaryFBCSPFeatureExtractor(self.bands, n_components=self.n_components, fs=self.fs)
        self.riemann_ = RiemannFeatureExtractor(
            band=self.riemann_band,
            fs=self.fs,
            estimator=self.estimator,
            metric=self.metric,
        )
        self.fbcsp_.fit(X, y)
        self.riemann_.fit(X, y)
        return self

    def transform(self, X):
        if self.fbcsp_ is None or self.riemann_ is None:
            raise RuntimeError("BinaryHybridFeatureExtractor must be fitted before calling transform().")
        fbcsp_features = self.fbcsp_.transform(X)
        riemann_features = self.riemann_.transform(X)
        return np.hstack([fbcsp_features, riemann_features]).astype(np.float32)


class HierarchicalMIClassifier(BaseEstimator, ClassifierMixin):
    """Hierarchical 4-class MI classifier with binary branch-specific CSP features."""

    def __init__(
        self,
        *,
        hands_classes: tuple[int, int] = (0, 1),
        midline_classes: tuple[int, int] = (2, 3),
        bands: list[tuple[float, float]] | None = None,
        n_components: int = 4,
        fs: float = 250.0,
        riemann_band: tuple[float, float] = (4.0, 40.0),
        estimator: str = "lwf",
        metric: str = "riemann",
        classifier_name: str = "lda",
        kernel: str = "rbf",
        C: float = 1.0,
    ) -> None:
        self.hands_classes = hands_classes
        self.midline_classes = midline_classes
        self.bands = bands
        self.n_components = n_components
        self.fs = fs
        self.riemann_band = riemann_band
        self.estimator = estimator
        self.metric = metric
        self.classifier_name = classifier_name
        self.kernel = kernel
        self.C = C
        self.stage_1_model_ = None
        self.hands_model_ = None
        self.midline_model_ = None
        self.classes_ = None

    def _build_branch_pipeline(self):
        return make_pipeline(
            BinaryHybridFeatureExtractor(
                bands=self.bands,
                n_components=self.n_components,
                fs=self.fs,
                riemann_band=self.riemann_band,
                estimator=self.estimator,
                metric=self.metric,
            ),
            StandardScaler(),
            _build_binary_classifier(
                self.classifier_name,
                kernel=self.kernel,
                C=self.C,
            ),
        )

    @staticmethod
    def _binary_labels(y: np.ndarray, positive_labels: tuple[int, int] | tuple[int, ...]) -> np.ndarray:
        positive_set = set(int(label) for label in positive_labels)
        return np.asarray([1 if int(label) in positive_set else 0 for label in y], dtype=np.int64)

    @staticmethod
    def _ensure_binary_branch(X: np.ndarray, y: np.ndarray, branch_name: str) -> tuple[np.ndarray, np.ndarray]:
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"{branch_name} expects exactly 2 labels, got {classes.tolist()}.")
        return X, y

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        expected_classes = np.asarray([*self.hands_classes, *self.midline_classes], dtype=np.int64)
        if sorted(np.unique(y).tolist()) != sorted(expected_classes.tolist()):
            raise ValueError(
                f"HierarchicalMIClassifier expects labels {expected_classes.tolist()}, got {np.unique(y).tolist()}."
            )

        y_stage_1 = self._binary_labels(y, tuple(int(label) for label in self.midline_classes))
        self.stage_1_model_ = self._build_branch_pipeline()
        self.stage_1_model_.fit(X, y_stage_1)

        hands_mask = np.isin(y, np.asarray(self.hands_classes, dtype=np.int64))
        X_hands = X[hands_mask]
        y_hands = self._binary_labels(y[hands_mask], (int(self.hands_classes[1]),))
        X_hands, y_hands = self._ensure_binary_branch(X_hands, y_hands, "hands branch")
        self.hands_model_ = self._build_branch_pipeline()
        self.hands_model_.fit(X_hands, y_hands)

        midline_mask = np.isin(y, np.asarray(self.midline_classes, dtype=np.int64))
        X_midline = X[midline_mask]
        y_midline = self._binary_labels(y[midline_mask], (int(self.midline_classes[1]),))
        X_midline, y_midline = self._ensure_binary_branch(X_midline, y_midline, "midline branch")
        self.midline_model_ = self._build_branch_pipeline()
        self.midline_model_.fit(X_midline, y_midline)

        self.classes_ = expected_classes.astype(np.int64)
        return self

    def predict_proba(self, X):
        if self.stage_1_model_ is None or self.hands_model_ is None or self.midline_model_ is None:
            raise RuntimeError("HierarchicalMIClassifier must be fitted before calling predict_proba().")

        X = np.asarray(X, dtype=np.float32)
        stage_1_proba = np.asarray(self.stage_1_model_.predict_proba(X), dtype=np.float64)
        hands_proba = np.asarray(self.hands_model_.predict_proba(X), dtype=np.float64)
        midline_proba = np.asarray(self.midline_model_.predict_proba(X), dtype=np.float64)

        final = np.column_stack(
            [
                stage_1_proba[:, 0] * hands_proba[:, 0],
                stage_1_proba[:, 0] * hands_proba[:, 1],
                stage_1_proba[:, 1] * midline_proba[:, 0],
                stage_1_proba[:, 1] * midline_proba[:, 1],
            ]
        )
        final = np.nan_to_num(final, nan=0.0, posinf=0.0, neginf=0.0)
        totals = np.sum(final, axis=1, keepdims=True)
        totals = np.where(totals <= 0.0, 1.0, totals)
        return (final / totals).astype(np.float64)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        indices = np.argmax(probabilities, axis=1)
        return self.classes_[indices]
