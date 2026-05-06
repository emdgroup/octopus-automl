"""Persistence tests for the SklearnClassifier / SklearnRegressor adapters.

The on-disk contract is:
  * AG persists itself under `<results_dir>/best/model/ag_predictor/`.
  * `ModuleResult.save` joblib-pickles the wrapper into the same `model/` dir.
  * Pickling the wrapper persists ONLY the AG store path; loading reads the
    AG predictor back via `TabularPredictor.load(path)`.

These tests fit a tiny AG predictor, build the wrapper, round-trip it through
joblib, and verify the reloaded wrapper produces predictions that match the
original (and exposes the right `classes_` / `predict_proba` contract).
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
import ray
from sklearn.datasets import make_classification, make_regression

from octopus._optional.autogluon import TabularPredictor
from octopus.modules.autogluon.adapters import SklearnClassifier, SklearnRegressor


def _binary_data(n: int = 60) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n,
        n_features=5,
        n_informative=3,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = y
    return df


def _regression_data(n: int = 60) -> pd.DataFrame:
    X, y = make_regression(n_samples=n, n_features=5, noise=0.1, random_state=42, coef=False)  # type: ignore[misc]
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = y
    return df


def _fit(train: pd.DataFrame, *, eval_metric: str, path) -> TabularPredictor:
    feature_cols = [f"feat_{i}" for i in range(5)]
    train_data = train[[*feature_cols, "target"]]
    predictor = TabularPredictor(
        label="target",
        eval_metric=eval_metric,
        verbosity=0,
        log_to_file=False,
        path=str(path),
    )
    predictor.fit(train_data, time_limit=30, num_bag_folds=2, num_bag_sets=1)
    if ray.is_initialized():
        ray.shutdown()
    return predictor


@pytest.fixture(scope="module")
def binary_wrapper(tmp_path_factory):
    """Return (wrapper, x_test, classes) for a fitted binary AG predictor."""
    df = _binary_data()
    train, test = df.iloc[:40].reset_index(drop=True), df.iloc[40:].reset_index(drop=True)
    ag_path = tmp_path_factory.mktemp("persist_binary") / "ag_predictor"
    predictor = _fit(train, eval_metric="balanced_accuracy", path=ag_path)
    wrapper = SklearnClassifier(predictor)
    feature_cols = [f"feat_{i}" for i in range(5)]
    return {
        "wrapper": wrapper,
        "x_test": test[feature_cols],
        "y_test": test["target"].to_numpy(),
        "class_labels": list(predictor.class_labels),
    }


@pytest.fixture(scope="module")
def regression_wrapper(tmp_path_factory):
    """Return (wrapper, x_test) for a fitted regression AG predictor."""
    df = _regression_data()
    train, test = df.iloc[:40].reset_index(drop=True), df.iloc[40:].reset_index(drop=True)
    ag_path = tmp_path_factory.mktemp("persist_regression") / "ag_predictor"
    predictor = _fit(train, eval_metric="r2", path=ag_path)
    wrapper = SklearnRegressor(predictor)
    feature_cols = [f"feat_{i}" for i in range(5)]
    return {"wrapper": wrapper, "x_test": test[feature_cols]}


def _roundtrip(wrapper, tmp_path):
    """Pickle wrapper to disk via joblib, then load back."""
    blob = tmp_path / "model.joblib"
    joblib.dump(wrapper, blob)
    return joblib.load(blob)


class TestClassifierPersistence:
    """Joblib round-trip yields a wrapper that produces identical predictions."""

    def test_predict_matches_after_reload(self, binary_wrapper, tmp_path) -> None:
        """Hard-label predictions are bit-exact after pickle round-trip."""
        wrapper = binary_wrapper["wrapper"]
        x_test = binary_wrapper["x_test"]
        original = wrapper.predict(x_test)
        reloaded_wrapper = _roundtrip(wrapper, tmp_path)
        reloaded = reloaded_wrapper.predict(x_test)
        np.testing.assert_array_equal(original, reloaded)

    def test_predict_proba_matches_after_reload(self, binary_wrapper, tmp_path) -> None:
        """Probability predictions match exactly after round-trip."""
        wrapper = binary_wrapper["wrapper"]
        x_test = binary_wrapper["x_test"]
        original = wrapper.predict_proba(x_test)
        reloaded = _roundtrip(wrapper, tmp_path).predict_proba(x_test)
        np.testing.assert_allclose(original, reloaded, rtol=0, atol=0)

    def test_classes_preserved(self, binary_wrapper, tmp_path) -> None:
        """The reloaded wrapper exposes the same class label array.

        OctoPredictor relies on `self.classes_[np.argmax(...)]` to map argmax
        positions back to actual labels; this contract must survive reload.
        """
        wrapper = binary_wrapper["wrapper"]
        reloaded = _roundtrip(wrapper, tmp_path)
        np.testing.assert_array_equal(reloaded.classes_, np.asarray(binary_wrapper["class_labels"]))

    def test_predict_proba_shape_and_normalization(self, binary_wrapper, tmp_path) -> None:
        """`predict_proba` returns ndarray of shape (n_samples, n_classes), rows sum to 1."""
        wrapper = binary_wrapper["wrapper"]
        x_test = binary_wrapper["x_test"]
        proba = _roundtrip(wrapper, tmp_path).predict_proba(x_test)
        assert isinstance(proba, np.ndarray)
        assert proba.shape == (len(x_test), len(binary_wrapper["class_labels"]))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_feature_names_preserved(self, binary_wrapper, tmp_path) -> None:
        """`feature_names_in_` survives round-trip."""
        wrapper = binary_wrapper["wrapper"]
        reloaded = _roundtrip(wrapper, tmp_path)
        assert list(reloaded.feature_names_in_) == list(wrapper.feature_names_in_)


class TestRegressorPersistence:
    """Regressor round-trip preserves predictions."""

    def test_predict_matches_after_reload(self, regression_wrapper, tmp_path) -> None:
        """Numeric predictions match exactly after pickle round-trip."""
        wrapper = regression_wrapper["wrapper"]
        x_test = regression_wrapper["x_test"]
        original = wrapper.predict(x_test)
        reloaded = _roundtrip(wrapper, tmp_path).predict(x_test)
        np.testing.assert_allclose(original, reloaded, rtol=0, atol=0)

    def test_feature_names_preserved(self, regression_wrapper, tmp_path) -> None:
        """`feature_names_in_` survives round-trip."""
        wrapper = regression_wrapper["wrapper"]
        reloaded = _roundtrip(wrapper, tmp_path)
        assert list(reloaded.feature_names_in_) == list(wrapper.feature_names_in_)


class TestFitRefused:
    """The wrappers are inference-only; calling fit must raise."""

    def test_classifier_fit_raises(self, binary_wrapper) -> None:
        """SklearnClassifier.fit raises NotImplementedError."""
        wrapper = binary_wrapper["wrapper"]
        with pytest.raises(NotImplementedError, match="already fitted"):
            wrapper.fit(binary_wrapper["x_test"], binary_wrapper["y_test"])

    def test_regressor_fit_raises(self, regression_wrapper) -> None:
        """SklearnRegressor.fit raises NotImplementedError."""
        wrapper = regression_wrapper["wrapper"]
        with pytest.raises(NotImplementedError, match="already fitted"):
            wrapper.fit(regression_wrapper["x_test"], np.zeros(len(regression_wrapper["x_test"])))
