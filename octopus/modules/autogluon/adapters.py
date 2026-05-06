"""Sklearn-compatible inference wrappers around a fitted AutoGluon TabularPredictor.

Persistence
-----------
`TabularPredictor` owns native artifacts that are not picklable. We override
`__reduce__` to pickle only the AG store path so in-process pickling
(`ModuleResult.save`, ray, multiprocessing) round-trips cheaply within a
single study tree.

Removing the on-disk AG store at `predictor.path` invalidates every wrapper
that pickled against it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from octopus._optional.autogluon import TabularPredictor


def _load_sklearn_classifier(path: str) -> SklearnClassifier:
    return SklearnClassifier(TabularPredictor.load(path))


def _load_sklearn_regressor(path: str) -> SklearnRegressor:
    return SklearnRegressor(TabularPredictor.load(path))


class SklearnClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn classifier wrapper for an AutoGluon TabularPredictor.

    Inference-only. Calling `fit()` raises NotImplementedError; the wrapped
    predictor is already trained.

    Pickling persists the AG store path, not the predictor itself. Unpickling
    reads the predictor back via `TabularPredictor.load(path)`.
    """

    def __init__(self, predictor: TabularPredictor) -> None:
        self.predictor = predictor
        self.classes_ = np.asarray(self.predictor.class_labels)
        self.n_classes_ = len(self.classes_)
        self.feature_names_in_ = self.predictor.original_features
        self.n_features_in_ = len(self.feature_names_in_)

    def predict(self, x: Any) -> np.ndarray:
        """Return hard class-label predictions as a numpy array."""
        return np.asarray(self.predictor.predict(x, as_pandas=False))

    def predict_proba(self, x: Any) -> np.ndarray:
        """Return class-probability predictions of shape (n_samples, n_classes)."""
        return np.asarray(self.predictor.predict_proba(x, as_pandas=False, as_multiclass=True))

    def fit(self, x: Any, y: Any) -> None:
        """Refuse to refit; the wrapped predictor is already trained."""
        raise NotImplementedError("This classifier is already fitted. Only for inference use.")

    def __reduce__(self) -> tuple[Any, tuple[str]]:
        return (_load_sklearn_classifier, (self.predictor.path,))


class SklearnRegressor(BaseEstimator, RegressorMixin):
    """Sklearn regressor wrapper for an AutoGluon TabularPredictor.

    Inference-only. Pickle round-trips through the AG on-disk store path; see
    the module docstring for details.
    """

    def __init__(self, predictor: TabularPredictor) -> None:
        self.predictor = predictor
        self.feature_names_in_ = self.predictor.original_features
        self.n_features_in_ = len(self.feature_names_in_)
        self.n_outputs_ = 1

    def predict(self, x: Any) -> np.ndarray:
        """Return numeric predictions as a numpy array."""
        return np.asarray(self.predictor.predict(x, as_pandas=False))

    def fit(self, x: Any, y: Any) -> None:
        """Refuse to refit; the wrapped predictor is already trained."""
        raise NotImplementedError("This regressor is already fitted. Only for inference use.")

    def __reduce__(self) -> tuple[Any, tuple[str]]:
        return (_load_sklearn_regressor, (self.predictor.path,))
