"""Tests for the diagnostics package — Optuna-only functionality.

Covers:
- _data_loader: _extract_id_from_dirname, load_parquet_glob, load_optuna
- _plots: plot_optuna_trial_counts_chart, plot_optuna_trials_chart,
          plot_optuna_hyperparameters_chart
- core: StudyDiagnostics init, properties, plot methods
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from octopus.diagnostics._data_loader import (
    _extract_id_from_dirname,
    load_optuna,
    load_parquet_glob,
)
from octopus.diagnostics._plots import (
    plot_optuna_hyperparameters_chart,
    plot_optuna_trial_counts_chart,
    plot_optuna_trials_chart,
)
from octopus.diagnostics.core import StudyDiagnostics

# ── Fixtures ────────────────────────────────────────────────────

_REAL_STUDY = Path("studies/wf_octo_mrmr_octo-20260315_110046")


def _has_real_study() -> bool:
    """Check if the real study directory with optuna data exists."""
    return (_REAL_STUDY / "study_config.json").exists()


_skip_no_study = pytest.mark.skipif(not _has_real_study(), reason="Real study not available")


@pytest.fixture()
def study_path() -> Path:
    """Path to the real study directory."""
    return _REAL_STUDY


@pytest.fixture()
def optuna_df(study_path: Path) -> pd.DataFrame:
    """Optuna results DataFrame from the real study."""
    return load_optuna(study_path)


@pytest.fixture()
def minimal_study(tmp_path: Path) -> Path:
    """Create a minimal study directory with a config and synthetic Optuna data."""
    study_dir = tmp_path / "test_study"
    study_dir.mkdir()

    # Write a minimal config
    config = {"ml_type": "binary", "n_folds_outer": 2}
    (study_dir / "study_config.json").write_text(json.dumps(config))

    # Create synthetic optuna results in outersplit0/task0
    results_dir = study_dir / "outersplit0" / "task0" / "results"
    results_dir.mkdir(parents=True)

    optuna_data = pd.DataFrame(
        {
            "trial": [0, 0, 1, 1],
            "value": [0.5, 0.5, 0.3, 0.3],
            "model_type": ["RandomForest", "RandomForest", "XGBoost", "XGBoost"],
            "hyper_param": ["max_depth", "n_estimators", "max_depth", "learning_rate"],
            "param_value": ["10", "100", "5", "0.1"],
        }
    )
    optuna_data.to_parquet(results_dir / "optuna_results.parquet", index=False)

    return study_dir


# ═══════════════════════════════════════════════════════════════
# _data_loader tests
# ═══════════════════════════════════════════════════════════════


class TestExtractIdFromDirname:
    """Tests for _extract_id_from_dirname."""

    def test_outersplit_zero(self) -> None:
        """Extract 0 from 'outersplit0'."""
        assert _extract_id_from_dirname("outersplit0", "outersplit") == 0

    def test_outersplit_multi_digit(self) -> None:
        """Extract 12 from 'outersplit12'."""
        assert _extract_id_from_dirname("outersplit12", "outersplit") == 12

    def test_task_zero(self) -> None:
        """Extract 0 from 'task0'."""
        assert _extract_id_from_dirname("task0", "task") == 0

    def test_task_multi_digit(self) -> None:
        """Extract 5 from 'task5'."""
        assert _extract_id_from_dirname("task5", "task") == 5

    def test_wrong_prefix_returns_none(self) -> None:
        """Return None when prefix does not match."""
        assert _extract_id_from_dirname("task5", "outersplit") is None

    def test_invalid_name_returns_none(self) -> None:
        """Return None for names that don't match the pattern."""
        assert _extract_id_from_dirname("results", "outersplit") is None

    def test_empty_string_returns_none(self) -> None:
        """Return None for empty string."""
        assert _extract_id_from_dirname("", "outersplit") is None


class TestLoadParquetGlob:
    """Tests for load_parquet_glob."""

    def test_returns_dataframe_with_ids(self, minimal_study: Path) -> None:
        """Loaded DataFrame should have outersplit_id and task_id columns."""
        df = load_parquet_glob(minimal_study, "outersplit*/task*/results/optuna_results.parquet")
        assert not df.empty
        assert "outersplit_id" in df.columns
        assert "task_id" in df.columns
        assert df["outersplit_id"].iloc[0] == 0
        assert df["task_id"].iloc[0] == 0

    def test_no_match_returns_empty(self, tmp_path: Path) -> None:
        """Return empty DataFrame when no files match the glob pattern."""
        df = load_parquet_glob(tmp_path, "nonexistent/*.parquet")
        assert df.empty


class TestLoadOptuna:
    """Tests for load_optuna."""

    def test_load_optuna_minimal(self, minimal_study: Path) -> None:
        """Load Optuna data from minimal synthetic study."""
        df = load_optuna(minimal_study)
        assert not df.empty
        assert "trial" in df.columns
        assert "value" in df.columns
        assert "model_type" in df.columns
        assert "hyper_param" in df.columns

    @_skip_no_study
    def test_load_optuna_real_study(self, study_path: Path) -> None:
        """Load Optuna data from the real study directory."""
        df = load_optuna(study_path)
        assert not df.empty
        assert "outersplit_id" in df.columns
        assert "task_id" in df.columns


# ═══════════════════════════════════════════════════════════════
# _plots tests
# ═══════════════════════════════════════════════════════════════


class TestOptunaPlots:
    """Tests for the three Optuna chart functions."""

    def test_trial_counts_chart(self, minimal_study: Path) -> None:
        """plot_optuna_trial_counts_chart returns a Figure."""
        df = load_optuna(minimal_study)
        fig = plot_optuna_trial_counts_chart(df)
        assert isinstance(fig, go.Figure)

    def test_trial_counts_chart_empty(self) -> None:
        """Handles empty DataFrame gracefully."""
        fig = plot_optuna_trial_counts_chart(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_trials_chart(self, minimal_study: Path) -> None:
        """plot_optuna_trials_chart returns a Figure."""
        df = load_optuna(minimal_study)
        fig = plot_optuna_trials_chart(df, outersplit_id=0, task_id=0)
        assert isinstance(fig, go.Figure)

    def test_trials_chart_empty(self) -> None:
        """Handles empty DataFrame gracefully."""
        fig = plot_optuna_trials_chart(pd.DataFrame(columns=["outersplit_id", "task_id"]), outersplit_id=0, task_id=0)
        assert isinstance(fig, go.Figure)

    def test_hyperparameters_chart(self, minimal_study: Path) -> None:
        """plot_optuna_hyperparameters_chart returns a Figure."""
        df = load_optuna(minimal_study)
        fig = plot_optuna_hyperparameters_chart(df, outersplit_id=0, task_id=0)
        assert isinstance(fig, go.Figure)

    def test_hyperparameters_chart_empty(self) -> None:
        """Handles empty DataFrame gracefully."""
        fig = plot_optuna_hyperparameters_chart(
            pd.DataFrame(columns=["outersplit_id", "task_id"]), outersplit_id=0, task_id=0
        )
        assert isinstance(fig, go.Figure)

    def test_trials_chart_maximize_direction(self, minimal_study: Path) -> None:
        """plot_optuna_trials_chart works with maximize direction."""
        df = load_optuna(minimal_study)
        fig = plot_optuna_trials_chart(df, outersplit_id=0, task_id=0, direction="maximize")
        assert isinstance(fig, go.Figure)

    def test_hyperparameters_chart_with_model_filter(self, minimal_study: Path) -> None:
        """plot_optuna_hyperparameters_chart filters by model_type."""
        df = load_optuna(minimal_study)
        fig = plot_optuna_hyperparameters_chart(df, outersplit_id=0, task_id=0, model_type="RandomForest")
        assert isinstance(fig, go.Figure)

    @_skip_no_study
    def test_trial_counts_chart_real(self, optuna_df: pd.DataFrame) -> None:
        """Trial counts chart with real study data."""
        fig = plot_optuna_trial_counts_chart(optuna_df)
        assert isinstance(fig, go.Figure)

    @_skip_no_study
    def test_trials_chart_real(self, optuna_df: pd.DataFrame) -> None:
        """Trials chart with real study data."""
        fig = plot_optuna_trials_chart(optuna_df, outersplit_id=0, task_id=0)
        assert isinstance(fig, go.Figure)

    @_skip_no_study
    def test_hyperparameters_chart_real(self, optuna_df: pd.DataFrame) -> None:
        """Hyperparameters chart with real study data."""
        fig = plot_optuna_hyperparameters_chart(optuna_df, outersplit_id=0, task_id=0)
        assert isinstance(fig, go.Figure)


# ═══════════════════════════════════════════════════════════════
# StudyDiagnostics tests
# ═══════════════════════════════════════════════════════════════


class TestStudyDiagnostics:
    """Tests for the StudyDiagnostics class."""

    def test_init_valid_path(self, minimal_study: Path) -> None:
        """StudyDiagnostics initializes with a valid study path."""
        diag = StudyDiagnostics(minimal_study)
        assert diag.study_path == minimal_study

    def test_init_missing_path(self) -> None:
        """StudyDiagnostics raises FileNotFoundError for missing path."""
        with pytest.raises(FileNotFoundError):
            StudyDiagnostics("/nonexistent/path/to/study")

    def test_config_loaded(self, minimal_study: Path) -> None:
        """Config dict is loaded from study_config.json."""
        diag = StudyDiagnostics(minimal_study)
        assert isinstance(diag.config, dict)
        assert diag.config["ml_type"] == "binary"

    def test_config_missing_file(self, tmp_path: Path) -> None:
        """Config is empty dict when study_config.json is absent."""
        diag = StudyDiagnostics(tmp_path)
        assert diag.config == {}

    def test_ml_type(self, minimal_study: Path) -> None:
        """ml_type property returns correct MLType."""
        from octopus.types import MLType

        diag = StudyDiagnostics(minimal_study)
        assert diag.ml_type == MLType.BINARY

    def test_optuna_trials_lazy_loading(self, minimal_study: Path) -> None:
        """optuna_trials is not loaded until accessed."""
        diag = StudyDiagnostics(minimal_study)
        assert diag._optuna is None  # noqa: SLF001
        _ = diag.optuna_trials
        assert diag._optuna is not None  # noqa: SLF001

    def test_optuna_trials_returns_dataframe(self, minimal_study: Path) -> None:
        """optuna_trials returns a non-empty DataFrame."""
        diag = StudyDiagnostics(minimal_study)
        df = diag.optuna_trials
        assert isinstance(df, pd.DataFrame)
        assert not df.empty


class TestStudyDiagnosticsPlotMethods:
    """Integration tests for StudyDiagnostics plot methods."""

    def test_plot_optuna_trial_counts(self, minimal_study: Path) -> None:
        """plot_optuna_trial_counts returns a Figure."""
        diag = StudyDiagnostics(minimal_study)
        fig = diag.plot_optuna_trial_counts()
        assert isinstance(fig, go.Figure)

    def test_plot_optuna_trials(self, minimal_study: Path) -> None:
        """plot_optuna_trials returns a Figure."""
        diag = StudyDiagnostics(minimal_study)
        fig = diag.plot_optuna_trials(outersplit_id=0, task_id=0)
        assert isinstance(fig, go.Figure)

    def test_plot_optuna_hyperparameters(self, minimal_study: Path) -> None:
        """plot_optuna_hyperparameters returns a Figure."""
        diag = StudyDiagnostics(minimal_study)
        fig = diag.plot_optuna_hyperparameters(outersplit_id=0, task_id=0)
        assert isinstance(fig, go.Figure)

    def test_plot_methods_empty_study(self, tmp_path: Path) -> None:
        """Plot methods handle empty Optuna data gracefully."""
        diag = StudyDiagnostics(tmp_path)
        fig1 = diag.plot_optuna_trial_counts()
        fig2 = diag.plot_optuna_trials()
        fig3 = diag.plot_optuna_hyperparameters()
        assert isinstance(fig1, go.Figure)
        assert isinstance(fig2, go.Figure)
        assert isinstance(fig3, go.Figure)

    @_skip_no_study
    def test_plot_methods_real_study(self, study_path: Path) -> None:
        """Plot methods work with the real study directory."""
        diag = StudyDiagnostics(study_path)
        assert isinstance(diag.plot_optuna_trial_counts(), go.Figure)
        assert isinstance(diag.plot_optuna_trials(outersplit_id=0, task_id=0), go.Figure)
        assert isinstance(diag.plot_optuna_hyperparameters(outersplit_id=0, task_id=0), go.Figure)