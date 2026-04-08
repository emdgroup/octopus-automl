"""Tests for octopus.predict module.

Self-contained: runs a minimal classification study in a session-scoped
fixture so that tests do not depend on pre-existing study directories.

Covers:
- study_io.py (StudyLoader, StudyMetadata)
- task_predictor.py (TaskPredictor)
- task_predictor_test.py (TaskPredictorTest)
- feature_importance.py (permutation FI)
- notebook_utils.py (show_study_details, show_target_metric_performance,
  show_selected_features, show_testset_performance, show_overall_fi_table,
  show_overall_fi_plot, show_confusionmatrix, show_aucroc_plots)
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from octopus.example_data import load_breast_cancer_data
from octopus.modules import Octo
from octopus.predict.notebook_utils import (
    display_table,
    show_aucroc_plots,
    show_confusionmatrix,
    show_overall_fi_plot,
    show_overall_fi_table,
    show_selected_features,
    show_study_details,
    show_target_metric_performance,
    show_testset_performance,
)
from octopus.predict.study_io import StudyLoader, StudyMetadata
from octopus.predict.task_predictor import TaskPredictor
from octopus.predict.task_predictor_test import TaskPredictorTest
from octopus.study import OctoClassification
from octopus.types import FIComputeMethod, FIType, MLType, ModelName
from octopus.utils import parquet_load

# ── Prevent plotly from opening browser windows ─────────────────


@pytest.fixture(autouse=True)
def _no_plotly_show(monkeypatch):
    """Patch plotly Figure.show() to prevent opening browser windows."""
    monkeypatch.setattr(go.Figure, "show", lambda *_args, **_kwargs: None)


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
def tpt(study_path):
    """Module-scoped TaskPredictorTest."""
    return TaskPredictorTest(study_path=study_path, task_id=0)


@pytest.fixture(scope="module")
def tp(study_path):
    """Module-scoped TaskPredictor."""
    return TaskPredictor(study_path=study_path, task_id=0)


# ═══════════════════════════════════════════════════════════════
# study_io tests
# ═══════════════════════════════════════════════════════════════


class TestStudyIO:
    """Tests for StudyLoader and StudyMetadata."""

    def test_load_config(self, study_path):
        """Verify StudyLoader loads config with correct ml_type and folds."""
        loader = StudyLoader(study_path)
        cfg = loader.load_config()
        assert cfg["ml_type"] == MLType.BINARY
        assert cfg["n_outer_splits"] == 2

    def test_extract_metadata(self, study_path):
        """Verify extracted metadata matches expected study properties."""
        loader = StudyLoader(study_path)
        cfg = loader.load_config()
        meta = loader.extract_metadata(cfg)
        assert isinstance(meta, StudyMetadata)
        assert meta.ml_type == MLType.BINARY
        assert meta.target_metric == "ACCBAL"
        assert len(meta.feature_cols) == 5

    def test_validate_task_id_valid(self, study_path):
        """Verify valid task_id passes validation without error."""
        loader = StudyLoader(study_path)
        cfg = loader.load_config()
        loader.validate_task_id(0, cfg)  # should not raise

    def test_validate_task_id_invalid(self, study_path):
        """Verify invalid task_id raises ValueError."""
        loader = StudyLoader(study_path)
        cfg = loader.load_config()
        with pytest.raises(ValueError):
            loader.validate_task_id(-1, cfg)

    def test_build_performance_summary(self, study_path):
        """Verify performance summary returns a non-empty DataFrame with Task column."""
        loader = StudyLoader(study_path)
        df = loader.build_performance_summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "Task" in df.columns

    def test_build_feature_summary(self, study_path):
        """Verify feature summary returns feature and frequency DataFrames."""
        loader = StudyLoader(study_path)
        feat_table, freq_table = loader.build_feature_summary()
        assert isinstance(feat_table, pd.DataFrame)
        assert isinstance(freq_table, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════
# TaskPredictorTest tests
# ═══════════════════════════════════════════════════════════════


class TestTaskPredictorTestProperties:
    """Test TaskPredictorTest properties."""

    def test_ml_type(self, tpt):
        """Verify ml_type is classification."""
        assert tpt.ml_type == MLType.BINARY

    def test_n_outersplits(self, tpt):
        """Verify n_outersplits matches study configuration."""
        assert tpt.n_outersplits == 2

    def test_outersplits(self, tpt):
        """Verify outersplits returns correct split indices."""
        assert tpt.outersplits == [0, 1]

    def test_feature_cols(self, tpt):
        """Verify feature_cols is non-empty."""
        assert len(tpt.feature_cols) > 0

    def test_classes(self, tpt):
        """Verify classes_ contains two classes for binary classification."""
        assert len(tpt.classes_) == 2


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
        # Binary classification has a single target assignment
        test_df = pd.DataFrame({"target": [0, 1, 0], "other": [1, 2, 3]})
        # Override target_assignments to a known single-target mapping
        original = tpt._metadata.target_assignments
        try:
            tpt._metadata = tpt._metadata.__class__(
                ml_type=tpt._metadata.ml_type,
                target_metric=tpt._metadata.target_metric,
                target_col=tpt._metadata.target_col,
                target_assignments={"default": "target"},
                positive_class=tpt._metadata.positive_class,
                row_id_col=tpt._metadata.row_id_col,
                feature_cols=tpt._metadata.feature_cols,
                n_outersplits=tpt._metadata.n_outersplits,
            )
            result = tpt._get_target_columns(test_df)
            assert list(result.keys()) == ["target"]
            np.testing.assert_array_equal(result["target"], [0, 1, 0])
        finally:
            tpt._metadata = tpt._metadata.__class__(
                ml_type=tpt._metadata.ml_type,
                target_metric=tpt._metadata.target_metric,
                target_col=tpt._metadata.target_col,
                target_assignments=original,
                positive_class=tpt._metadata.positive_class,
                row_id_col=tpt._metadata.row_id_col,
                feature_cols=tpt._metadata.feature_cols,
                n_outersplits=tpt._metadata.n_outersplits,
            )

    def test_multi_target_returns_prefixed_keys(self, tpt):
        """Verify multi-target tasks produce {'target_role': ...} dict for each role."""
        test_df = pd.DataFrame(
            {
                "time_col": [10.0, 20.0, 30.0],
                "event_col": [1, 0, 1],
            }
        )
        original_ml_type = tpt._metadata.ml_type
        original_target_assignments = tpt._metadata.target_assignments
        try:
            tpt._metadata = tpt._metadata.__class__(
                ml_type=MLType.TIMETOEVENT,
                target_metric=tpt._metadata.target_metric,
                target_col=tpt._metadata.target_col,
                target_assignments={"duration": "time_col", "event": "event_col"},
                positive_class=tpt._metadata.positive_class,
                row_id_col=tpt._metadata.row_id_col,
                feature_cols=tpt._metadata.feature_cols,
                n_outersplits=tpt._metadata.n_outersplits,
            )
            result = tpt._get_target_columns(test_df)
            assert "target_duration" in result
            assert "target_event" in result
            assert "target" not in result
            np.testing.assert_array_equal(result["target_duration"], [10.0, 20.0, 30.0])
            np.testing.assert_array_equal(result["target_event"], [1, 0, 1])
        finally:
            tpt._metadata = tpt._metadata.__class__(
                ml_type=original_ml_type,
                target_metric=tpt._metadata.target_metric,
                target_col=tpt._metadata.target_col,
                target_assignments=original_target_assignments,
                positive_class=tpt._metadata.positive_class,
                row_id_col=tpt._metadata.row_id_col,
                feature_cols=tpt._metadata.feature_cols,
                n_outersplits=tpt._metadata.n_outersplits,
            )

    def test_predict_df_single_target_has_target_column(self, tpt):
        """Verify predict(df=True) includes 'target' column for single-target tasks."""
        result = tpt.predict(df=True)
        assert "target" in result.columns

    def test_predict_proba_df_single_target_has_target_column(self, tpt):
        """Verify predict_proba(df=True) includes 'target' column for single-target tasks."""
        result = tpt.predict_proba(df=True)
        assert "target" in result.columns


class TestTaskPredictorTestGuards:
    """Test TaskPredictorTest serialization guards."""

    def test_save_raises(self, tpt, tmp_path):
        """Verify save raises NotImplementedError for test predictor."""
        with pytest.raises(NotImplementedError):
            tpt.save(tmp_path / "nope")

    def test_load_raises(self):
        """Verify load raises NotImplementedError for test predictor."""
        with pytest.raises(NotImplementedError):
            TaskPredictorTest.load("dummy")


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
        """Verify save/load preserves ml_type, n_outersplits, and feature_cols."""
        tp.save(tmp_path / "saved")
        loaded = TaskPredictor.load(tmp_path / "saved")
        assert loaded.ml_type == tp.ml_type
        assert loaded.n_outersplits == tp.n_outersplits
        assert loaded.feature_cols == tp.feature_cols

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
        original = tp._metadata.ml_type
        try:
            tp._metadata = tp._metadata.__class__(
                ml_type=MLType.REGRESSION,
                target_metric=tp._metadata.target_metric,
                target_col=tp._metadata.target_col,
                target_assignments=tp._metadata.target_assignments,
                positive_class=tp._metadata.positive_class,
                row_id_col=tp._metadata.row_id_col,
                feature_cols=tp._metadata.feature_cols,
                n_outersplits=tp._metadata.n_outersplits,
            )
            data = pd.DataFrame({"f0": [1], "f1": [2], "f2": [3], "f3": [4], "f4": [5]})
            with pytest.raises(TypeError, match=r"predict_proba.*only available"):
                tp.predict_proba(data)
        finally:
            tp._metadata = tp._metadata.__class__(
                ml_type=original,
                target_metric=tp._metadata.target_metric,
                target_col=tp._metadata.target_col,
                target_assignments=tp._metadata.target_assignments,
                positive_class=tp._metadata.positive_class,
                row_id_col=tp._metadata.row_id_col,
                feature_cols=tp._metadata.feature_cols,
                n_outersplits=tp._metadata.n_outersplits,
            )


# ═══════════════════════════════════════════════════════════════
# notebook_utils tests
# ═══════════════════════════════════════════════════════════════


class TestNotebookUtilsStudyLevel:
    """Test study-level notebook utils (show_study_details, etc.)."""

    def test_show_study_details(self, study_path):
        """Verify show_study_details returns correct study info dict."""
        info = show_study_details(study_path, verbose=False)
        assert info["ml_type"] == MLType.BINARY
        assert info["n_outer_splits"] == 2
        assert len(info["outersplit_dirs"]) == 2
        assert len(info["missing_outersplits"]) == 0

    def test_show_study_details_verbose(self, study_path, capsys):
        """Verify verbose mode prints ML type to stdout."""
        show_study_details(study_path, verbose=True)
        captured = capsys.readouterr()
        assert "ML Type: binary" in captured.out

    def test_show_study_details_missing_path(self):
        """Verify FileNotFoundError is raised for nonexistent path."""
        with pytest.raises(FileNotFoundError):
            show_study_details("/nonexistent/path")

    def test_show_target_metric_performance(self, study_path):
        """Verify target metric performance returns at least one DataFrame."""
        info = show_study_details(study_path, verbose=False)
        tables = show_target_metric_performance(info)
        assert len(tables) >= 1
        assert isinstance(tables[0], pd.DataFrame)

    def test_show_selected_features(self, study_path):
        """Verify selected features returns feature and frequency DataFrames."""
        info = show_study_details(study_path, verbose=False)
        feat_table, freq_table, _ = show_selected_features(info)
        assert isinstance(feat_table, pd.DataFrame)
        assert isinstance(freq_table, pd.DataFrame)


class TestNotebookUtilsTaskLevel:
    """Test task-level notebook utils."""

    def test_show_testset_performance(self, tpt):
        """Verify testset performance overview includes a Mean row."""
        df = show_testset_performance(tpt, metrics=["ACCBAL", "ACC"])
        assert isinstance(df, pd.DataFrame)
        assert "Mean" in df.index

    def test_display_table(self, capsys):
        """Verify display_table runs without error."""
        df = pd.DataFrame({"a": [1, 2]})
        display_table(df)
        # Should not raise; output goes to stdout or IPython display

    def test_show_overall_fi_table(self, tpt):
        """Verify overall FI table contains only ensemble rows."""
        fi = tpt.calculate_fi(fi_type=FIType.PERMUTATION, n_repeats=2)
        ensemble_df = show_overall_fi_table(fi)
        assert isinstance(ensemble_df, pd.DataFrame)
        # All rows should be ensemble
        if "fi_source" in ensemble_df.columns:
            assert (ensemble_df["fi_source"] == "ensemble").all()

    def test_show_overall_fi_plot(self, tpt):
        """Verify overall FI plot renders without error."""
        fi = tpt.calculate_fi(fi_type=FIType.PERMUTATION, n_repeats=2)
        # Should not raise (plotly fig.show() is a no-op in test)
        show_overall_fi_plot(fi, top_n=3)

    def test_show_confusionmatrix(self, tpt):
        """Verify confusion matrix renders without error."""
        # Should not raise
        show_confusionmatrix(tpt, threshold=0.5, metrics=["ACCBAL", "ACC"])

    def test_show_aucroc_plots(self, tpt):
        """Verify AUC-ROC plots render without error."""
        # Should not raise
        show_aucroc_plots(tpt, show_individual=False)


# ═══════════════════════════════════════════════════════════════
# Notebook workflow integration test
# ═══════════════════════════════════════════════════════════════


class TestNotebookWorkflow:
    """Mimic the full analyse_study_classification.ipynb workflow."""

    def test_full_notebook_workflow(self, study_path):
        """Run all notebook steps end-to-end."""
        # Cell: show_study_details
        study_info = show_study_details(study_path, verbose=True)
        assert study_info["ml_type"] == MLType.BINARY

        # Cell: show_target_metric_performance
        perf_tables = show_target_metric_performance(study_info, details=False)
        assert len(perf_tables) >= 1

        # Cell: show_selected_features
        feat_table, _, _ = show_selected_features(study_info)
        assert len(feat_table) > 0

        # Cell: create TaskPredictorTest
        tpt_local = TaskPredictorTest(
            study_path=study_info["path"],
            task_id=0,
            result_type="best",
        )

        # Cell: show_testset_performance
        metrics = ["AUCROC", "ACCBAL", "ACC"]
        perf_df = show_testset_performance(tpt_local, metrics=metrics)
        assert "Mean" in perf_df.index

        # Cell: show_aucroc_plots
        show_aucroc_plots(tpt_local, show_individual=True)

        # Cell: show_confusionmatrix
        show_confusionmatrix(tpt_local, threshold=0.5, metrics=metrics)

        # Cell: permutation FI
        fi_perm = tpt_local.calculate_fi(
            fi_type=FIType.PERMUTATION,
            n_repeats=2,
        )
        assert isinstance(fi_perm, pd.DataFrame)

        # Cell: show_overall_fi_table
        fi_ensemble = show_overall_fi_table(fi_perm)
        assert len(fi_ensemble) > 0

        # Cell: show_overall_fi_plot
        show_overall_fi_plot(fi_perm)

        # Cell: per-split FI access
        for split_id in tpt_local.outersplits:
            split_fi = fi_perm[fi_perm["fi_source"] == split_id]
            assert len(split_fi) > 0
