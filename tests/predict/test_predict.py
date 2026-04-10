"""Tests for octopus.predict module.

Self-contained: runs a minimal classification study in a session-scoped
fixture so that tests do not depend on pre-existing study directories.

Covers:
- study_io.py (_extract_metadata_from_config, load_config)
- predictor.py (OctoPredictor)
- test_evaluator.py (OctoTestEvaluator)
- feature_importance.py (permutation FI)
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest

from octopus.analysis.test_evaluator import OctoTestEvaluator
from octopus.example_data import load_breast_cancer_data
from octopus.modules import Octo
from octopus.predict.predictor import OctoPredictor
from octopus.predict.study_io import load_config, load_study_info
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
        study_name="predict_test_study",
        target_metric="ACCBAL",
        feature_cols=features,
        target_col="target",
        sample_id_col="index",
        stratification_col="target",
        outer_split_seed=1234,
        n_outer_splits=2,
        studies_directory=tmp_path,
        workflow=[
            Octo(
                description="step_1_octo",
                task_id=0,
                depends_on=None,
                n_inner_splits=3,
                models=[ModelName.ExtraTreesClassifier],
                max_outliers=0,
                fi_methods=[FIComputeMethod.PERMUTATION],
                n_startup_trials=3,
                n_trials=5,
                max_features=5,
                ensemble_selection=True,
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
    return load_study_info(study_path)


@pytest.fixture(scope="module")
def tpt(study):
    """Module-scoped OctoTestEvaluator."""
    return OctoTestEvaluator(study=study, task_id=0)


@pytest.fixture(scope="module")
def tp(study):
    """Module-scoped OctoPredictor."""
    return OctoPredictor(study=study, task_id=0)


# ═══════════════════════════════════════════════════════════════
# study_io tests
# ═══════════════════════════════════════════════════════════════


class TestStudyIO:
    """Tests for standalone I/O functions."""

    def test_load_config(self, study_path):
        """Verify load_config returns correct ml_type and folds."""
        cfg = load_config(study_path)
        assert cfg["ml_type"] == MLType.BINARY
        assert cfg["n_outer_splits"] == 2
        assert cfg["target_metric"] == "ACCBAL"
        assert len(cfg["feature_cols"]) == 5


# ═══════════════════════════════════════════════════════════════
# OctoTestEvaluator tests
# ═══════════════════════════════════════════════════════════════


class TestOctoTestEvaluatorProperties:
    """Test OctoTestEvaluator attributes."""

    def test_ml_type(self, tpt):
        """Verify ml_type is classification."""
        assert MLType(tpt._config["ml_type"]) == MLType.BINARY

    def test_n_outer_splits(self, tpt):
        """Verify n_outer_splits matches study configuration."""
        assert len(tpt._models) == 2

    def test_outer_splits(self, tpt):
        """Verify outer_splits returns correct split indices."""
        assert list(tpt._models.keys()) == [0, 1]

    def test_feature_cols(self, tpt):
        """Verify feature_cols is non-empty."""
        assert len(tpt._feature_cols) > 0

    def test_classes(self, tpt):
        """Verify classes_ contains two classes for binary classification."""
        assert len(next(iter(tpt._models.values())).classes_) == 2


class TestOctoTestEvaluatorPredict:
    """Test OctoTestEvaluator.predict()."""

    def test_predict(self, tpt):
        """Verify predict returns a DataFrame with expected columns."""
        result = tpt.predict()
        assert isinstance(result, pd.DataFrame)
        for col in ("outer_split", "row_id", "prediction", "target"):
            assert col in result.columns
        assert result["outer_split"].nunique() == 2


class TestOctoTestEvaluatorPredictProba:
    """Test OctoTestEvaluator.predict_proba()."""

    def test_predict_proba(self, tpt):
        """Verify predict_proba returns DataFrame with outer_split and target."""
        result = tpt.predict_proba()
        assert isinstance(result, pd.DataFrame)
        assert "outer_split" in result.columns
        assert "target" in result.columns

    def test_predict_proba_sums_to_one(self, tpt):
        """Verify predicted probabilities sum to 1 for each sample."""
        result = tpt.predict_proba()
        class_cols = [c for c in result.columns if c not in ("outer_split", "row_id", "target")]
        np.testing.assert_allclose(result[class_cols].sum(axis=1), 1.0, atol=1e-6)


class TestOctoTestEvaluatorPerformance:
    """Test OctoTestEvaluator.performance()."""

    def test_performance_default(self, tpt):
        """Verify default performance returns one row per outer split."""
        result = tpt.performance()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # one per outer split

    def test_performance_multiple_metrics(self, tpt):
        """Verify performance with multiple metrics returns correct number of rows."""
        result = tpt.performance(metrics=["ACCBAL", "AUCROC"])
        assert len(result) == 4  # 2 splits x 2 metrics


class TestOctoTestEvaluatorFI:
    """Test OctoTestEvaluator.calculate_fi()."""

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
    """Test OctoTestEvaluator._get_target_columns helper."""

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
            tpt._config.setdefault("prepared", {})["target_assignments"] = {
                "duration": "time_col",
                "event": "event_col",
            }
            result = tpt._get_target_columns(test_df)
            assert "target_duration" in result
            assert "target_event" in result
            assert "target" not in result
            np.testing.assert_array_equal(result["target_duration"], [10.0, 20.0, 30.0])
            np.testing.assert_array_equal(result["target_event"], [1, 0, 1])
        finally:
            tpt._config["ml_type"] = original_ml_type
            tpt._config["prepared"]["target_assignments"] = original_target_assignments

    def test_predict_single_target_has_target_column(self, tpt):
        """Verify predict includes 'target' column for single-target tasks."""
        result = tpt.predict()
        assert "target" in result.columns

    def test_predict_proba_single_target_has_target_column(self, tpt):
        """Verify predict_proba includes 'target' column for single-target tasks."""
        result = tpt.predict_proba()
        assert "target" in result.columns


# ═══════════════════════════════════════════════════════════════
# OctoPredictor tests
# ═══════════════════════════════════════════════════════════════


class TestBuildPoolData:
    """Test OctoPredictor._build_pool_data fallback behavior."""

    def test_keyerror_fallback(self, tp, study_path, monkeypatch):
        """Verify _build_pool_data falls back to data on KeyError."""
        import octopus.predict.predictor as pred_mod  # noqa: PLC0415

        monkeypatch.setattr(
            pred_mod, "load_split_data", lambda *a, **kw: (_ for _ in ()).throw(KeyError("missing row"))
        )
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        pool = tp._build_pool_data(data)
        for split_id in tp._models:
            assert pool[split_id] is data

    def test_filenotfounderror_fallback(self, tp, study_path, monkeypatch):
        """Verify _build_pool_data falls back to data on FileNotFoundError."""
        import octopus.predict.predictor as pred_mod  # noqa: PLC0415

        monkeypatch.setattr(pred_mod, "load_split_data", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()))
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        pool = tp._build_pool_data(data)
        for split_id in tp._models:
            assert pool[split_id] is data


class TestOctoPredictorPredict:
    """Test OctoPredictor predict and predict_proba."""

    def test_predict(self, tp, study_path):
        """Verify OctoPredictor predict returns DataFrame with row_id and prediction."""
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        result = tp.predict(data)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["row_id", "prediction"]
        assert len(result) == len(data)

    def test_predict_per_split(self, tp, study_path):
        """Verify per_split=True adds individual split columns."""
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        result = tp.predict(data, per_split=True)
        assert "row_id" in result.columns
        assert "prediction" in result.columns
        split_cols = [c for c in result.columns if c.startswith("split_")]
        assert len(split_cols) == len(tp._models)
        assert len(result) == len(data)

    def test_predict_proba(self, tp, study_path):
        """Verify OctoPredictor predict_proba returns valid probabilities."""
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        result = tp.predict_proba(data)
        assert isinstance(result, pd.DataFrame)
        assert "row_id" in result.columns
        class_cols = [c for c in result.columns if c != "row_id"]
        np.testing.assert_allclose(result[class_cols].sum(axis=1), 1.0, atol=1e-6)
        assert len(result) == len(data)

    def test_predict_proba_per_split(self, tp, study_path):
        """Verify per_split=True adds individual split probability columns."""
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        result = tp.predict_proba(data, per_split=True)
        assert "row_id" in result.columns
        split_proba_cols = [c for c in result.columns if "_split_" in str(c)]
        assert len(split_proba_cols) > 0


class TestOctoPredictorSaveLoad:
    """Test OctoPredictor save/load round-trip."""

    def test_roundtrip(self, tp, tmp_path):
        """Verify save/load preserves config, n_outer_splits, and feature_cols."""
        tp.save(tmp_path / "saved")
        loaded = OctoPredictor.load(tmp_path / "saved")
        assert loaded._config["ml_type"] == tp._config["ml_type"]
        assert len(loaded._models) == len(tp._models)
        assert loaded._feature_cols == tp._feature_cols

    def test_loaded_predicts(self, tp, study_path, tmp_path):
        """Verify a loaded OctoPredictor can still produce predictions."""
        tp.save(tmp_path / "saved2")
        loaded = OctoPredictor.load(tmp_path / "saved2")
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        result = loaded.predict(data)
        assert isinstance(result, pd.DataFrame)
        assert "prediction" in result.columns


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
