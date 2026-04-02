"""Tests for octopus.predict module.

Self-contained: runs a minimal classification study in a session-scoped
fixture so that tests do not depend on pre-existing study directories.

Covers:
- study_io.py (_extract_metadata_from_config, load_config)
- task_predictor.py (TaskPredictor)
- task_predictor_test.py (TaskPredictorTest)
- feature_importance.py (permutation FI)
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest

from octopus.example_data import load_breast_cancer_data
from octopus.modules import Octo
from octopus.predict.study_io import load_config, load_study
from octopus.predict.task_predictor import TaskPredictor
from octopus.predict.task_predictor_test import TaskPredictorTest
from octopus.study import OctoClassification
from octopus.types import FIComputeMethod, FIType, MLType, ModelName
from octopus.utils import parquet_load


# ── Shared study fixture ────────────────────────────────────────

_STUDY_TMPDIR: tempfile.TemporaryDirectory | None = None


def _create_classification_study(tmp_path: str) -> tuple[OctoClassification, pd.DataFrame]:
    """Create a minimal classification study and data without calling fit().

    Parameters:
        tmp_path: Path to temporary directory for study output.

    Returns:
        Tuple of (study, data) for use in tests.
    """
    # Use breast cancer dataset (enough samples for nonzero FI)
    df, features, _targets = load_breast_cancer_data()
    features = features[:5]  # Use only first 5 features for faster testing

    study = OctoClassification(
        name="predict_test_study",
        target_metric="ACCBAL",
        feature_cols=features,
        target_col="target",
        sample_id_col="index",
        stratification_col="target",
        datasplit_seed_outer=1234,
        n_folds_outer=2,
        path=tmp_path,
        ignore_data_health_warning=True,
        workflow=[
            Octo(
                description="step_1_octo",
                task_id=0,
                depends_on=None,
                n_folds_inner=3,
                models=[ModelName.ExtraTreesClassifier],
                model_seed=0,
                max_outl=0,
                fi_methods_bestbag=[FIComputeMethod.PERMUTATION],
                optuna_seed=0,
                n_optuna_startup_trials=3,
                n_trials=5,
                max_features=5,
                penalty_factor=1.0,
                ensemble_selection=True,
                ensel_n_save_trials=5,
            ),
        ],
    )
    return study, df


def _run_classification_study() -> str:
    """Run a minimal classification study and return the study path."""
    global _STUDY_TMPDIR  # noqa: PLW0603
    _STUDY_TMPDIR = tempfile.TemporaryDirectory()
    tmp = _STUDY_TMPDIR.name

    study, df = _create_classification_study(tmp)
    study.fit(data=df)
    return str(study.output_path)


@pytest.fixture(scope="module")
def study_path():
    """Run a study once per module and yield its path."""
    path = _run_classification_study()
    yield path
    if _STUDY_TMPDIR is not None:
        _STUDY_TMPDIR.cleanup()


@pytest.fixture(scope="module")
def study(study_path):
    """Module-scoped validated StudyInfo."""
    return load_study(study_path)


@pytest.fixture(scope="module")
def tpt(study):
    """Module-scoped TaskPredictorTest."""
    return TaskPredictorTest(study=study, task_id=0)


@pytest.fixture(scope="module")
def tp(study):
    """Module-scoped TaskPredictor."""
    return TaskPredictor(study=study, task_id=0)


# ═══════════════════════════════════════════════════════════════
# study_io tests
# ═══════════════════════════════════════════════════════════════


class TestStudyIO:
    """Tests for standalone I/O functions."""

    def test_load_config(self, study_path):
        """Verify load_config returns correct ml_type and folds."""
        cfg = load_config(study_path)
        assert cfg["ml_type"] == MLType.BINARY
        assert cfg["n_folds_outer"] == 2
        assert cfg["target_metric"] == "ACCBAL"
        assert len(cfg["feature_cols"]) == 5



# ═══════════════════════════════════════════════════════════════
# TaskPredictorTest tests
# ═══════════════════════════════════════════════════════════════


class TestTaskPredictorTestProperties:
    """Test TaskPredictorTest attributes."""

    def test_ml_type(self, tpt):
        """Verify ml_type is classification."""
        assert MLType(tpt._config["ml_type"]) == MLType.BINARY

    def test_n_outersplits(self, tpt):
        """Verify n_outersplits matches study configuration."""
        assert len(tpt._models) == 2

    def test_outersplits(self, tpt):
        """Verify outersplits returns correct split indices."""
        assert list(tpt._models.keys()) == [0, 1]

    def test_feature_cols(self, tpt):
        """Verify feature_cols is non-empty."""
        assert len(tpt._feature_cols) > 0

    def test_classes(self, tpt):
        """Verify classes_ contains two classes for binary classification."""
        assert len(next(iter(tpt._models.values())).classes_) == 2


class TestTaskPredictorTestPredict:
    """Test TaskPredictorTest.predict()."""

    def test_predict_array(self, tpt):
        """Verify predict returns a non-empty numpy array."""
        result = tpt.predict(df=False)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_predict_df(self, tpt):
        """Verify predict with df=True returns DataFrame with expected columns."""
        result = tpt.predict(df=True)
        assert isinstance(result, pd.DataFrame)
        for col in ("outersplit", "row_id", "prediction", "target"):
            assert col in result.columns
        assert result["outersplit"].nunique() == 2


class TestTaskPredictorTestPredictProba:
    """Test TaskPredictorTest.predict_proba()."""

    def test_predict_proba_array(self, tpt):
        """Verify predict_proba returns a 2D array with correct shape."""
        result = tpt.predict_proba(df=False)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_predict_proba_df(self, tpt):
        """Verify predict_proba with df=True returns DataFrame with outersplit and target."""
        result = tpt.predict_proba(df=True)
        assert isinstance(result, pd.DataFrame)
        assert "outersplit" in result.columns
        assert "target" in result.columns

    def test_predict_proba_sums_to_one(self, tpt):
        """Verify predicted probabilities sum to 1 for each sample."""
        result = tpt.predict_proba(df=False)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)


class TestTaskPredictorTestPerformance:
    """Test TaskPredictorTest.performance()."""

    def test_performance_default(self, tpt):
        """Verify default performance returns one row per outersplit."""
        result = tpt.performance()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # one per outersplit

    def test_performance_multiple_metrics(self, tpt):
        """Verify performance with multiple metrics returns correct number of rows."""
        result = tpt.performance(metrics=["ACCBAL", "AUCROC"])
        assert len(result) == 4  # 2 splits x 2 metrics


class TestTaskPredictorTestFI:
    """Test TaskPredictorTest.calculate_fi()."""

    def test_fi_permutation(self, tpt):
        """Verify permutation FI returns DataFrame with per-split and ensemble rows."""
        fi = tpt.calculate_fi(fi_type=FIType.PERMUTATION, n_repeats=2)
        assert isinstance(fi, pd.DataFrame)
        assert "feature" in fi.columns
        assert "importance_mean" in fi.columns
        assert "fi_source" in fi.columns
        # Should have per-split + ensemble rows
        sources = fi["fi_source"].unique()
        assert "ensemble" in sources

    def test_fi_invalid_type(self, tpt):
        """Verify invalid fi_type raises ValueError."""
        with pytest.raises(ValueError, match="is not a valid FIType"):
            tpt.calculate_fi(fi_type="invalid_method")


class TestGetTargetColumns:
    """Test TaskPredictorTest._get_target_columns helper."""

    def test_single_target_returns_target_key(self, tpt):
        """Verify single-target tasks produce {'target': ...} dict."""
        test_df = pd.DataFrame({"target": [0, 1, 0], "other": [1, 2, 3]})
        original = tpt._config.get("prepared", {}).get("target_assignments", {})
        try:
            tpt._config.setdefault("prepared", {})["target_assignments"] = {"default": "target"}
            result = tpt._get_target_columns(test_df)
            assert list(result.keys()) == ["target"]
            np.testing.assert_array_equal(result["target"], [0, 1, 0])
        finally:
            tpt._config["prepared"]["target_assignments"] = original

    def test_multi_target_returns_prefixed_keys(self, tpt):
        """Verify multi-target tasks produce {'target_role': ...} dict for each role."""
        test_df = pd.DataFrame(
            {
                "time_col": [10.0, 20.0, 30.0],
                "event_col": [1, 0, 1],
            }
        )
        original_ml_type = tpt._config["ml_type"]
        original_target_assignments = tpt._config.get("prepared", {}).get("target_assignments", {})
        try:
            tpt._config["ml_type"] = MLType.TIMETOEVENT
            tpt._config.setdefault("prepared", {})["target_assignments"] = {"duration": "time_col", "event": "event_col"}
            result = tpt._get_target_columns(test_df)
            assert "target_duration" in result
            assert "target_event" in result
            assert "target" not in result
            np.testing.assert_array_equal(result["target_duration"], [10.0, 20.0, 30.0])
            np.testing.assert_array_equal(result["target_event"], [1, 0, 1])
        finally:
            tpt._config["ml_type"] = original_ml_type
            tpt._config["prepared"]["target_assignments"] = original_target_assignments

    def test_predict_df_single_target_has_target_column(self, tpt):
        """Verify predict(df=True) includes 'target' column for single-target tasks."""
        result = tpt.predict(df=True)
        assert "target" in result.columns

    def test_predict_proba_df_single_target_has_target_column(self, tpt):
        """Verify predict_proba(df=True) includes 'target' column for single-target tasks."""
        result = tpt.predict_proba(df=True)
        assert "target" in result.columns



# ═══════════════════════════════════════════════════════════════
# TaskPredictor tests
# ═══════════════════════════════════════════════════════════════


class TestTaskPredictorPredict:
    """Test TaskPredictor predict and predict_proba."""

    def test_predict_array(self, tp, study_path):
        """Verify TaskPredictor predict returns array matching input length."""
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        result = tp.predict(data, df=False)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)

    def test_predict_df(self, tp, study_path):
        """Verify TaskPredictor predict with df=True includes ensemble predictions."""
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        result = tp.predict(data, df=True)
        assert isinstance(result, pd.DataFrame)
        assert "prediction" in result.columns
        # Should have per-split + ensemble rows
        assert "ensemble" in result["outersplit"].values

    def test_predict_proba(self, tp, study_path):
        """Verify TaskPredictor predict_proba returns valid probabilities."""
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        result = tp.predict_proba(data, df=False)
        assert result.ndim == 2
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)


class TestTaskPredictorSaveLoad:
    """Test TaskPredictor save/load round-trip."""

    def test_roundtrip(self, tp, tmp_path):
        """Verify save/load preserves config, n_outersplits, and feature_cols."""
        tp.save(tmp_path / "saved")
        loaded = TaskPredictor.load(tmp_path / "saved")
        assert loaded._config["ml_type"] == tp._config["ml_type"]
        assert len(loaded._models) == len(tp._models)
        assert loaded._feature_cols == tp._feature_cols

    def test_loaded_predicts(self, tp, study_path, tmp_path):
        """Verify a loaded TaskPredictor can still produce predictions."""
        tp.save(tmp_path / "saved2")
        loaded = TaskPredictor.load(tmp_path / "saved2")
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        result = loaded.predict(data)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)


class TestPredictProbaMLTypeGuard:
    """Test predict_proba raises TypeError for non-classification."""

    def test_mock_regression_guard(self, tp):
        """Temporarily override ml_type to test the guard."""
        original = tp._config["ml_type"]
        try:
            tp._config["ml_type"] = MLType.REGRESSION
            data = pd.DataFrame({"f0": [1], "f1": [2], "f2": [3], "f3": [4], "f4": [5]})
            with pytest.raises(TypeError, match=r"predict_proba.*only available"):
                tp.predict_proba(data)
        finally:
            tp._config["ml_type"] = original
