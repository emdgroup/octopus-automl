"""Tests for octopus.diagnostics module.

Tests cover:
- _extract_id_from_dirname: directory name parsing
- load_parquet_glob: generic glob-based parquet loading
- load_optuna: Optuna-specific loader
- load_feature_importances: FI loader
- StudyDiagnostics: integration tests
- Optuna chart functions: plot generation
- Feature importance chart: plot generation
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest
from upath import UPath

from octopus.diagnostics._data_loader import (
    _extract_id_from_dirname,
    load_feature_importances,
    load_optuna,
    load_parquet_glob,
)
from octopus.diagnostics._plots import (
    plot_feature_importance_chart,
    plot_optuna_hyperparameters_chart,
    plot_optuna_trial_counts_chart,
    plot_optuna_trials_chart,
)
from octopus.diagnostics.core import StudyDiagnostics
from octopus.predict.notebook_utils import find_latest_study
from octopus.types import MetricDirection, MLType

# ── Fixtures ────────────────────────────────────────────────────

STUDIES_ROOT = "studies"
STUDY_PREFIX = "wf_octo_mrmr_octo"


@pytest.fixture(scope="module")
def study_path() -> UPath:
    """Resolve the latest study path for the test suite."""
    return UPath(find_latest_study(STUDIES_ROOT, STUDY_PREFIX))


@pytest.fixture(scope="module")
def optuna_df(study_path: UPath) -> pd.DataFrame:
    """Load Optuna data once for the module."""
    return load_optuna(study_path)

@pytest.fixture(scope="module")
def fi_df(study_path: UPath) -> pd.DataFrame:
    """Load feature importances once for the module."""
    return load_feature_importances(study_path)


# ── TestExtractIdFromDirname ────────────────────────────────────


class TestExtractIdFromDirname:
    """Unit tests for _extract_id_from_dirname."""

    def test_outersplit_valid(self) -> None:
        """Test extracting outersplit ID from valid dirname."""
        assert _extract_id_from_dirname("outersplit0", "outersplit") == 0

    def test_outersplit_multidigit(self) -> None:
        """Test extracting multi-digit outersplit ID."""
        assert _extract_id_from_dirname("outersplit12", "outersplit") == 12

    def test_task_valid(self) -> None:
        """Test extracting task ID from valid dirname."""
        assert _extract_id_from_dirname("task0", "task") == 0

    def test_task_multidigit(self) -> None:
        """Test extracting multi-digit task ID."""
        assert _extract_id_from_dirname("task5", "task") == 5

    def test_wrong_prefix(self) -> None:
        """Test that wrong prefix returns None."""
        assert _extract_id_from_dirname("task5", "outersplit") is None

    def test_invalid_name(self) -> None:
        """Test that invalid dirname returns None."""
        assert _extract_id_from_dirname("invalid", "outersplit") is None

    def test_no_number(self) -> None:
        """Test that prefix without number returns None."""
        assert _extract_id_from_dirname("outersplit", "outersplit") is None


# ── TestLoadParquetGlob ─────────────────────────────────────────


class TestLoadParquetGlob:
    """Tests for load_parquet_glob with real study data."""

    def test_load_optuna_pattern(self, study_path: UPath) -> None:
        """Test loading optuna files with glob pattern."""
        df = load_parquet_glob(study_path, "outersplit*/task*/results/optuna_results.parquet")
        assert not df.empty

    def test_injects_outersplit_id(self, study_path: UPath) -> None:
        """Test that outersplit_id column is injected from directory names."""
        df = load_parquet_glob(study_path, "outersplit*/task*/results/optuna_results.parquet")
        assert "outersplit_id" in df.columns

    def test_injects_task_id(self, study_path: UPath) -> None:
        """Test that task_id column is injected from directory names."""
        df = load_parquet_glob(study_path, "outersplit*/task*/results/optuna_results.parquet")
        assert "task_id" in df.columns

    def test_no_match(self, tmp_path: UPath) -> None:
        """Test that non-matching pattern returns empty DataFrame."""
        df = load_parquet_glob(UPath(tmp_path), "nonexistent/*.parquet")
        assert df.empty

    def test_columns_not_duplicated(self, study_path: UPath) -> None:
        """Test that outersplit_id is not duplicated when already in parquet."""
        df = load_parquet_glob(study_path, "outersplit*/task*/results/optuna_results.parquet")
        assert list(df.columns).count("outersplit_id") == 1


# ── TestLoadOptuna ──────────────────────────────────────────────


class TestLoadOptuna:
    """Tests for load_optuna."""

    def test_returns_dataframe(self, study_path: UPath) -> None:
        """Test that load_optuna returns a non-empty DataFrame."""
        df = load_optuna(study_path)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_expected_columns(self, study_path: UPath) -> None:
        """Test that expected columns are present in Optuna results."""
        df = load_optuna(study_path)
        expected = {"outersplit_id", "task_id", "trial", "value", "model_type", "hyper_param", "param_value"}
        assert expected.issubset(set(df.columns))

    def test_empty_for_fake_path(self, tmp_path: UPath) -> None:
        """Test that fake path returns empty DataFrame."""
        df = load_optuna(UPath(tmp_path))
        assert df.empty


# ── TestStudyDiagnostics ────────────────────────────────────────


class TestStudyDiagnostics:
    """Integration tests for StudyDiagnostics."""

    def test_init_valid_path(self, study_path: UPath) -> None:
        """Test creating StudyDiagnostics with valid path."""
        diag = StudyDiagnostics(study_path)
        assert diag.study_path.exists()

    def test_init_missing_path(self) -> None:
        """Test that missing path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            StudyDiagnostics("/nonexistent/path/to/study")

    def test_config_loaded(self, study_path: UPath) -> None:
        """Test that config is loaded as non-empty dict with ml_type."""
        diag = StudyDiagnostics(study_path)
        assert isinstance(diag.config, dict)
        assert "ml_type" in diag.config

    def test_ml_type(self, study_path: UPath) -> None:
        """Test that ml_type returns MLType.BINARY for test study."""
        diag = StudyDiagnostics(study_path)
        assert diag.ml_type == MLType.BINARY

    def test_optuna_trials_lazy(self, study_path: UPath) -> None:
        """Test that optuna data is not loaded until accessed."""
        diag = StudyDiagnostics(study_path)
        assert diag._optuna is None

    def test_optuna_trials_loaded(self, study_path: UPath) -> None:
        """Test that optuna data is loaded after first access."""
        diag = StudyDiagnostics(study_path)
        df = diag.optuna_trials
        assert not df.empty
        assert diag._optuna is not None

    def test_feature_importances_lazy(self, study_path: UPath) -> None:
        """Test that FI data is not loaded until accessed."""
        diag = StudyDiagnostics(study_path)
        assert diag._feature_importances is None  # noqa: SLF001

    def test_feature_importances_loaded(self, study_path: UPath) -> None:
        """Test that FI data is loaded after first access."""
        diag = StudyDiagnostics(study_path)
        df = diag.feature_importances
        assert not df.empty
        assert diag._feature_importances is not None  # noqa: SLF001

    def test_get_feature_importances_task_filter(self, study_path: UPath) -> None:
        """Test get_feature_importances filters by task_id."""
        diag = StudyDiagnostics(study_path)
        df = diag.get_feature_importances(task_id=0)
        assert not df.empty
        assert (df["task_id"] == 0).all()

    def test_get_feature_importances_result_type_filter(self, study_path: UPath) -> None:
        """Test get_feature_importances filters by result_type."""
        diag = StudyDiagnostics(study_path)
        df = diag.get_feature_importances(result_type="best")
        assert not df.empty
        assert (df["result_type"] == "best").all()

    def test_get_feature_importances_all(self, study_path: UPath) -> None:
        """Test get_feature_importances with no filters returns all data."""
        diag = StudyDiagnostics(study_path)
        df = diag.get_feature_importances()
        assert not df.empty
        assert len(df) == len(diag.feature_importances)


# ── TestOptunaPlots ─────────────────────────────────────────────


class TestOptunaPlots:
    """Tests for Optuna chart functions."""

    def test_trial_counts_chart(self, optuna_df: pd.DataFrame) -> None:
        """Test trial counts chart returns a Figure."""
        fig = plot_optuna_trial_counts_chart(optuna_df)
        assert isinstance(fig, go.Figure)

    def test_trial_counts_empty(self) -> None:
        """Test trial counts chart with empty DataFrame."""
        fig = plot_optuna_trial_counts_chart(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_trials_chart(self, optuna_df: pd.DataFrame) -> None:
        """Test trials chart returns a Figure."""
        fig = plot_optuna_trials_chart(optuna_df, outersplit_id=0, task_id=0)
        assert isinstance(fig, go.Figure)

    def test_trials_chart_maximize(self, optuna_df: pd.DataFrame) -> None:
        """Test trials chart with maximize direction."""
        fig = plot_optuna_trials_chart(
            optuna_df, outersplit_id=0, task_id=0, direction=MetricDirection.MAXIMIZE
        )
        assert isinstance(fig, go.Figure)

    def test_trials_chart_empty(self) -> None:
        """Test trials chart with empty DataFrame."""
        fig = plot_optuna_trials_chart(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_hyperparameters_chart(self, optuna_df: pd.DataFrame) -> None:
        """Test hyperparameters chart returns a Figure."""
        fig = plot_optuna_hyperparameters_chart(optuna_df, outersplit_id=0, task_id=0)
        assert isinstance(fig, go.Figure)

    def test_hyperparameters_chart_empty(self) -> None:
        """Test hyperparameters chart with empty DataFrame."""
        fig = plot_optuna_hyperparameters_chart(pd.DataFrame())
        assert isinstance(fig, go.Figure)

# ── TestLoadFeatureImportances ──────────────────────────────────

class TestLoadFeatureImportances:
    """Tests for load_feature_importances."""

    def test_returns_nonempty(self, study_path: UPath) -> None:
        """Test that load_feature_importances returns non-empty DataFrame."""
        df = load_feature_importances(study_path)
        assert not df.empty

    def test_expected_columns(self, study_path: UPath) -> None:
        """Test expected columns in FI DataFrame."""
        df = load_feature_importances(study_path)
        expected = {"feature", "importance", "fi_method", "fi_dataset", "training_id", "module", "result_type"}
        assert expected.issubset(set(df.columns))

    def test_empty_for_fake_path(self, tmp_path: UPath) -> None:
        """Test that fake path returns empty DataFrame."""
        df = load_feature_importances(UPath(tmp_path))
        assert df.empty

# ── TestFeatureImportancePlots ──────────────────────────────────

class TestFeatureImportancePlots:
    """Tests for feature importance chart function."""

    def test_fi_chart_returns_figure(self, fi_df: pd.DataFrame) -> None:
        """Test FI chart returns a Figure."""
        fig = plot_feature_importance_chart(fi_df, outersplit_id=0, task_id=0)
        assert isinstance(fig, go.Figure)

    def test_fi_chart_with_method_filter(self, fi_df: pd.DataFrame) -> None:
        """Test FI chart with fi_method filter."""
        fig = plot_feature_importance_chart(fi_df, outersplit_id=0, task_id=0, fi_method="internal")
        assert isinstance(fig, go.Figure)

    def test_fi_chart_with_dataset_filter(self, fi_df: pd.DataFrame) -> None:
        """Test FI chart with fi_dataset filter."""
        fig = plot_feature_importance_chart(fi_df, outersplit_id=0, task_id=0, fi_dataset="train")
        assert isinstance(fig, go.Figure)

    def test_fi_chart_with_result_type_filter(self, fi_df: pd.DataFrame) -> None:
        """Test FI chart with result_type filter."""
        fig = plot_feature_importance_chart(fi_df, result_type="best")
        assert isinstance(fig, go.Figure)

    def test_fi_chart_no_filters(self, fi_df: pd.DataFrame) -> None:
        """Test FI chart without any filters (aggregates all)."""
        fig = plot_feature_importance_chart(fi_df)
        assert isinstance(fig, go.Figure)

    def test_fi_chart_empty(self) -> None:
        """Test FI chart with empty DataFrame."""
        fig = plot_feature_importance_chart(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_fi_chart_top_n(self, fi_df: pd.DataFrame) -> None:
        """Test FI chart with top_n parameter."""
        fig = plot_feature_importance_chart(fi_df, outersplit_id=0, task_id=0, top_n=5)
        assert isinstance(fig, go.Figure)


# ── TestStudyDiagnosticsPlotMethods ─────────────────────────────


class TestStudyDiagnosticsPlotMethods:
    """Integration: calling plot methods on StudyDiagnostics (no display)."""

    def test_plot_optuna_trial_counts(self, study_path: UPath, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test plot_optuna_trial_counts does not raise."""
        monkeypatch.setattr("plotly.graph_objects.Figure.show", lambda self: None)
        diag = StudyDiagnostics(study_path)
        diag.plot_optuna_trial_counts()

    def test_plot_optuna_trials(self, study_path: UPath, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test plot_optuna_trials does not raise."""
        monkeypatch.setattr("plotly.graph_objects.Figure.show", lambda self: None)
        diag = StudyDiagnostics(study_path)
        diag.plot_optuna_trials(outersplit_id=0, task_id=0)

    def test_plot_optuna_hyperparameters(self, study_path: UPath, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test plot_optuna_hyperparameters does not raise."""
        monkeypatch.setattr("plotly.graph_objects.Figure.show", lambda self: None)
        diag = StudyDiagnostics(study_path)
        diag.plot_optuna_hyperparameters(outersplit_id=0, task_id=0)

    def test_plot_feature_importance_auto_load(self, study_path: UPath, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test plot_feature_importance without fi_table (auto-loads)."""
        monkeypatch.setattr("plotly.graph_objects.Figure.show", lambda self: None)
        diag = StudyDiagnostics(study_path)
        diag.plot_feature_importance(outersplit_id=0, task_id=0, fi_method="internal")

    def test_plot_feature_importance_with_fi_table(self, study_path: UPath, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test plot_feature_importance with pre-filtered fi_table."""
        monkeypatch.setattr("plotly.graph_objects.Figure.show", lambda self: None)
        diag = StudyDiagnostics(study_path)
        fi_table = diag.get_feature_importances(task_id=0, result_type="best")
        diag.plot_feature_importance(fi_table, fi_method="internal", fi_dataset="train")

    def test_plot_feature_importance_permutation(self, study_path: UPath, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test plot_feature_importance with permutation method via fi_table."""
        monkeypatch.setattr("plotly.graph_objects.Figure.show", lambda self: None)
        diag = StudyDiagnostics(study_path)
        fi_table = diag.get_feature_importances(task_id=0)
        diag.plot_feature_importance(fi_table, fi_method="permutation", fi_dataset="dev")
