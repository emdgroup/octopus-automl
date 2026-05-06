"""Unit tests for octopus.modules.autogluon.feature_importance.

Covers:
  * OOF success on numeric data: non-empty FI, training_id is set to the
    actual L1 model name, feature index is a subset of feature_cols.
  * OOF success on text-feature data: text-derived transformed columns
    (`txt.char_count`, `txt.word_count`, ...) are aggregated back to the
    original text column name and never leak into the published FI frame.
  * No qualifying L1 child: fitting with only `included_model_types=["RF"]`
    produces children with `_bagged_mode=False`, so `compute_fi` returns
    an empty DataFrame.
  * Group-FI warning: passing non-empty feature_groups logs a single warning.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import ray
from sklearn.datasets import make_classification

from octopus._optional.autogluon import TabularPredictor
from octopus.modules.autogluon.feature_importance import compute_fi


def _binary_numeric_data(n: int = 80) -> pd.DataFrame:
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


def _binary_text_data(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    X, y = make_classification(
        n_samples=n,
        n_features=3,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(3)])
    sentences = ["alpha bravo charlie delta", "echo foxtrot golf hotel", "india juliet kilo lima"]
    df["txt"] = rng.choice(sentences, size=n)
    df["target"] = y
    return df


def _fit(
    train: pd.DataFrame,
    *,
    feature_cols: list[str],
    eval_metric: str,
    path,
    included_model_types: list[str] | None = None,
) -> TabularPredictor:
    train_data = train[[*feature_cols, "target"]]
    predictor = TabularPredictor(
        label="target",
        eval_metric=eval_metric,
        verbosity=0,
        log_to_file=False,
        path=str(path),
    )
    fit_kwargs: dict = {"time_limit": 30, "num_bag_folds": 2, "num_bag_sets": 1}
    if included_model_types is not None:
        fit_kwargs["included_model_types"] = included_model_types
    predictor.fit(train_data, **fit_kwargs)
    if ray.is_initialized():
        ray.shutdown()
    return predictor


def _bundle(predictor: TabularPredictor, feature_cols: list[str]) -> dict:
    return {
        "predictor": predictor,
        "feature_cols": feature_cols,
        "leaderboard": predictor.leaderboard(silent=True),
    }


@pytest.fixture(scope="module")
def numeric_predictor(tmp_path_factory):
    """Fit a tiny binary AG predictor on numeric features only."""
    df = _binary_numeric_data()
    feature_cols = [f"feat_{i}" for i in range(5)]
    predictor = _fit(
        df,
        feature_cols=feature_cols,
        eval_metric="balanced_accuracy",
        path=tmp_path_factory.mktemp("fi_numeric") / "ag_predictor",
    )
    return _bundle(predictor, feature_cols)


@pytest.fixture(scope="module")
def text_predictor(tmp_path_factory):
    """Fit a tiny binary AG predictor with both numeric and text features."""
    df = _binary_text_data()
    feature_cols = [*[f"feat_{i}" for i in range(3)], "txt"]
    predictor = _fit(
        df,
        feature_cols=feature_cols,
        eval_metric="balanced_accuracy",
        path=tmp_path_factory.mktemp("fi_text") / "ag_predictor",
    )
    return _bundle(predictor, feature_cols)


@pytest.fixture(scope="module")
def rf_only_predictor(tmp_path_factory):
    """Fit AG with only RF: no L1 child should qualify for OOF FI."""
    df = _binary_numeric_data()
    feature_cols = [f"feat_{i}" for i in range(5)]
    predictor = _fit(
        df,
        feature_cols=feature_cols,
        eval_metric="balanced_accuracy",
        path=tmp_path_factory.mktemp("fi_rf_only") / "ag_predictor",
        included_model_types=["RF"],
    )
    return _bundle(predictor, feature_cols)


def _call_fi(fixture: dict, *, feature_groups: dict | None = None) -> pd.DataFrame:
    return compute_fi(
        fixture["predictor"],
        feature_cols=fixture["feature_cols"],
        feature_groups=feature_groups or {},
        leaderboard=fixture["leaderboard"],
    )


class TestNumericFI:
    """OOF FI on a numeric-only dataset."""

    def test_returns_non_empty(self, numeric_predictor) -> None:
        """`compute_fi` produces a non-empty importance frame."""
        fi = _call_fi(numeric_predictor)
        assert not fi.empty
        assert "importance" in fi.columns

    def test_index_matches_feature_cols(self, numeric_predictor) -> None:
        """Returned features form exactly the original feature_cols set."""
        fi = _call_fi(numeric_predictor)
        assert set(fi.index) == set(numeric_predictor["feature_cols"])

    def test_training_id_is_actual_model_name(self, numeric_predictor) -> None:
        """`attrs['training_id']` holds the AG model name used for FI, not 'mean'."""
        fi = _call_fi(numeric_predictor)
        training_id = fi.attrs["training_id"]
        assert training_id != "mean"
        assert training_id != "autogluon"
        assert training_id in numeric_predictor["predictor"].model_names(level=1)

    def test_sorted_descending(self, numeric_predictor) -> None:
        """FI is sorted by importance descending."""
        fi = _call_fi(numeric_predictor)
        importances = fi["importance"].to_numpy()
        assert (importances[:-1] >= importances[1:]).all()


class TestTextFI:
    """OOF FI when AG generates transformed text features must aggregate back to originals."""

    def test_no_transformed_names_in_published_fi(self, text_predictor) -> None:
        """`txt.char_count`, `txt.word_count`, etc. must not leak into the result."""
        fi = _call_fi(text_predictor)
        assert set(fi.index).issubset(set(text_predictor["feature_cols"]))
        for name in fi.index:
            assert "." not in str(name), f"Transformed name {name!r} leaked into FI"

    def test_txt_appears_in_index(self, text_predictor) -> None:
        """The original `txt` column has an entry in the published FI frame."""
        fi = _call_fi(text_predictor)
        assert "txt" in fi.index


class TestEmptyFallback:
    """No qualifying L1 child means an empty DataFrame, never a hard failure."""

    def test_rf_only_returns_empty(self, rf_only_predictor) -> None:
        """RandomForest L1 children have `_child_oof=True` and so don't qualify."""
        assert _call_fi(rf_only_predictor).empty


class TestGroupFIWarning:
    """Passing feature_groups logs a warning but does not change the result shape."""

    def test_warns_when_groups_passed(self, numeric_predictor, caplog) -> None:
        """A WARNING-level log fires explaining group FI is unsupported."""
        feature_groups = {"all": numeric_predictor["feature_cols"]}
        octo_logger = logging.getLogger("OctoManager")
        octo_logger.addHandler(caplog.handler)
        caplog.set_level(logging.WARNING)
        try:
            _call_fi(numeric_predictor, feature_groups=feature_groups)
        finally:
            octo_logger.removeHandler(caplog.handler)
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("group feature importances are not supported" in msg.lower() for msg in warning_messages), (
            f"Expected group-FI warning, got: {warning_messages}"
        )

    def test_per_feature_fi_still_produced(self, numeric_predictor) -> None:
        """Passing feature_groups must not change the result shape."""
        fi = _call_fi(numeric_predictor, feature_groups={"all": numeric_predictor["feature_cols"]})
        assert set(fi.index) == set(numeric_predictor["feature_cols"])
