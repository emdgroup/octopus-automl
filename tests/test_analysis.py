"""Tests for octopus.poststudy analysis functions.

Uses the same session-scoped study fixture as test_predict.py.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest
from upath import UPath

from octopus.example_data import load_breast_cancer_data
from octopus.modules import Octo
from octopus.poststudy import OctoTestEvaluator, StudyInfo, load_study_information
from octopus.poststudy.notebook import (
    display_table,
    show_performance_tables,
    show_study_overview,
)
from octopus.poststudy.plots import (
    feature_count_plot,
    feature_frequency_plot,
    performance_plot,
)
from octopus.poststudy.study_io import find_latest_study
from octopus.poststudy.tables import (
    aucroc_data,
    confusion_matrix_data,
    fi_ensemble_table,
    get_performance,
    get_selected_features,
    workflow_graph,
)
from octopus.poststudy.tables import (
    testset_performance_table as _testset_perf_table,
)
from octopus.study import OctoClassification
from octopus.types import FIType, ModelName


@pytest.fixture(scope="session")
def study_path(tmp_path_factory):
    """Create a minimal classification study for testing."""
    df, features, _ = load_breast_cancer_data()
    path = str(tmp_path_factory.mktemp("studies"))

    study = OctoClassification(
        study_name="test_analysis",
        studies_directory=path,
        target_metric="ACCBAL",
        feature_cols=features[:5],
        target_col="target",
        sample_id_col="index",
        stratification_col="target",
        n_outer_splits=2,
        n_cpus=1,
        workflow=[
            Octo(
                description="step1",
                task_id=0,
                depends_on=None,
                models=[ModelName.ExtraTreesClassifier],
                n_trials=3,
                n_inner_splits=2,
            ),
        ],
    )
    study.fit(data=df)
    return str(study.output_path)


@pytest.fixture(scope="session")
def study_info(study_path):
    """Load study information."""
    return load_study_information(study_path)


@pytest.fixture(scope="session")
def tpt(study_info):
    """Create a OctoTestEvaluator."""
    return OctoTestEvaluator(study_info=study_info, task_id=0)


@pytest.fixture(autouse=True)
def _suppress_plotly_show(monkeypatch):
    """Suppress plotly fig.show() in tests."""
    monkeypatch.setattr(go.Figure, "show", lambda *_args, **_kwargs: None)


class TestStudyInfo:
    """Tests for study loading and StudyInfo."""

    def test_load_returns_study_info(self, study_info):
        """Verify load_study_information returns a StudyInfo instance."""
        assert isinstance(study_info, StudyInfo)

    def test_study_info_fields(self, study_info):
        """Verify StudyInfo has expected typed fields."""
        assert study_info.ml_type == "binary"
        assert study_info.target_metric == "ACCBAL"
        assert study_info.n_folds_outer == 2
        assert len(study_info.outersplit_dirs) == 2
        assert len(study_info.workflow_tasks) >= 1
        assert study_info.target_col == "target"
        assert study_info.row_id_col is not None
        assert isinstance(study_info.feature_cols, list)
        assert study_info.positive_class is not None

    def test_study_info_is_frozen(self, study_info):
        """Verify StudyInfo is immutable."""
        with pytest.raises(AttributeError):
            study_info.ml_type = "regression"

    def test_missing_path_raises(self):
        """Verify FileNotFoundError for nonexistent path."""
        with pytest.raises(FileNotFoundError):
            load_study_information("/nonexistent/path")

    def test_find_latest_study(self, study_path):
        """Verify find_latest_study finds the study."""
        parent = str(UPath(study_path).parent)
        prefix = UPath(study_path).name.rsplit("-", 1)[0]
        result = find_latest_study(parent, prefix)
        assert result == study_path


class TestPerformance:
    """Tests for get_performance and related functions."""

    def test_get_performance_returns_dataframe(self, study_info):
        """Verify get_performance returns a non-empty DataFrame."""
        perf = get_performance(study_info)
        assert isinstance(perf, pd.DataFrame)
        assert len(perf) > 0

    def test_get_performance_has_metric_column(self, study_info):
        """Verify DataFrame has a metric column with multiple metrics."""
        perf = get_performance(study_info)
        assert "metric" in perf.columns
        assert len(perf["metric"].unique()) > 1

    def test_get_performance_has_target_metric_attr(self, study_info):
        """Verify attrs contains target_metric."""
        perf = get_performance(study_info)
        assert perf.attrs["target_metric"] == "ACCBAL"

    def test_get_performance_no_test_by_default(self, study_info):
        """Verify test columns are excluded by default."""
        perf = get_performance(study_info)
        test_cols = [c for c in perf.columns if "_test_" in c or c.startswith("test_")]
        assert len(test_cols) == 0

    def test_get_performance_with_test(self, study_info):
        """Verify test columns are included when requested."""
        perf = get_performance(study_info, report_test=True)
        test_cols = [c for c in perf.columns if "_test_" in c or c.startswith("test_")]
        assert len(test_cols) > 0

    def test_get_performance_has_mean_rows(self, study_info):
        """Verify Mean rows are present."""
        perf = get_performance(study_info)
        assert "Mean" in perf.index

    def test_show_performance_tables(self, study_info, capsys):
        """Verify show_performance_tables prints output."""
        perf = get_performance(study_info)
        show_performance_tables(perf)
        captured = capsys.readouterr()
        assert "Workflow task" in captured.out

    def test_performance_plot_returns_figure(self, study_info):
        """Verify performance_plot returns a Plotly Figure."""
        perf = get_performance(study_info)
        fig = performance_plot(perf)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestSelectedFeatures:
    """Tests for get_selected_features and related functions."""

    def test_returns_two_dataframes(self, study_info):
        """Verify returns a tuple of two DataFrames."""
        feat, freq = get_selected_features(study_info)
        assert isinstance(feat, pd.DataFrame)
        assert isinstance(freq, pd.DataFrame)

    def test_feature_count_plot_returns_figure(self, study_info):
        """Verify feature_count_plot returns a Plotly Figure."""
        feat, _ = get_selected_features(study_info)
        fig = feature_count_plot(feat)
        assert isinstance(fig, go.Figure)

    def test_feature_frequency_plot_returns_figure(self, study_info):
        """Verify feature_frequency_plot returns a Plotly Figure."""
        _, freq = get_selected_features(study_info)
        fig = feature_frequency_plot(freq)
        assert isinstance(fig, go.Figure)


class TestWorkflowGraph:
    """Tests for workflow_graph."""

    def test_returns_string(self, study_info):
        """Verify workflow_graph returns a non-empty string."""
        result = workflow_graph(study_info)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_task_info(self, study_info):
        """Verify output contains task ID and module type."""
        result = workflow_graph(study_info)
        assert "[task 0]" in result
        assert "OCTO" in result


class TestTaskLevel:
    """Tests for task-level analysis functions (require OctoTestEvaluator)."""

    def test_testset_performance_table(self, tpt):
        """Verify testset performance table has Mean row."""
        df = _testset_perf_table(tpt, metrics=["ACCBAL", "ACC"])
        assert isinstance(df, pd.DataFrame)
        assert "Mean" in df.index

    def test_aucroc_data(self, tpt):
        """Verify aucroc_data returns dict with expected keys."""
        data = aucroc_data(tpt)
        assert isinstance(data, dict)
        assert "merged_auc" in data
        assert "mean_auc" in data
        assert "per_split" in data
        assert len(data["per_split"]) == tpt.study_info.n_outersplits

    def test_confusion_matrix_data(self, tpt):
        """Verify confusion_matrix_data returns dict with per_split data."""
        data = confusion_matrix_data(tpt, threshold=0.5)
        assert isinstance(data, dict)
        assert "per_split" in data
        assert len(data["per_split"]) == tpt.study_info.n_outersplits

    def test_fi_ensemble_table(self, tpt):
        """Verify fi_ensemble_table filters to ensemble rows."""
        fi = tpt.calculate_fi(fi_type=FIType.PERMUTATION, n_repeats=2)
        ensemble = fi_ensemble_table(fi)
        assert isinstance(ensemble, pd.DataFrame)
        if "fi_source" in ensemble.columns:
            assert (ensemble["fi_source"] == "ensemble").all()


class TestPlots:
    """Smoke tests for Level 2 plot functions."""

    def test_aucroc_merged_plot(self, tpt):
        """Verify aucroc_merged_plot returns a Figure."""
        from octopus.poststudy.plots import aucroc_merged_plot  # noqa: PLC0415

        data = aucroc_data(tpt)
        fig = aucroc_merged_plot(data)
        assert isinstance(fig, go.Figure)

    def test_aucroc_averaged_plot(self, tpt):
        """Verify aucroc_averaged_plot returns a Figure."""
        from octopus.poststudy.plots import aucroc_averaged_plot  # noqa: PLC0415

        data = aucroc_data(tpt)
        fig = aucroc_averaged_plot(data)
        assert isinstance(fig, go.Figure)

    def test_confusion_matrix_plot(self, tpt):
        """Verify confusion_matrix_plot returns a Figure."""
        from octopus.poststudy.plots import confusion_matrix_plot  # noqa: PLC0415

        data = confusion_matrix_data(tpt, threshold=0.5)
        split_data = data["per_split"][0]
        fig = confusion_matrix_plot(split_data["cm_abs"], split_data["cm_rel"], split_data["class_names"], "Split 0")
        assert isinstance(fig, go.Figure)

    def test_fi_plot(self, tpt):
        """Verify fi_plot returns a Figure."""
        from octopus.poststudy.plots import fi_plot  # noqa: PLC0415

        fi = tpt.calculate_fi(fi_type=FIType.PERMUTATION, n_repeats=2)
        ensemble = fi_ensemble_table(fi)
        fig = fi_plot(ensemble)
        assert isinstance(fig, go.Figure)

    def test_testset_performance_plot(self, tpt):
        """Verify testset_performance_plot returns a Figure."""
        from octopus.poststudy.plots import testset_performance_plot  # noqa: PLC0415
        from octopus.poststudy.tables import testset_performance_table  # noqa: PLC0415

        perf = testset_performance_table(tpt, metrics=["ACCBAL", "ACC"])
        fig = testset_performance_plot(perf)
        assert isinstance(fig, go.Figure)


class TestNotebookWrappers:
    """Smoke tests for Level 3 notebook wrappers."""

    def test_show_testset_performance(self, tpt):
        """Verify show_testset_performance runs without error."""
        from octopus.poststudy.notebook import show_testset_performance  # noqa: PLC0415

        show_testset_performance(tpt, metrics=["ACCBAL"])

    def test_show_aucroc(self, tpt):
        """Verify show_aucroc runs without error."""
        from octopus.poststudy.notebook import show_aucroc  # noqa: PLC0415

        show_aucroc(tpt)

    def test_show_confusionmatrix(self, tpt):
        """Verify show_confusionmatrix runs without error."""
        from octopus.poststudy.notebook import show_confusionmatrix  # noqa: PLC0415

        show_confusionmatrix(tpt, threshold=0.5, metrics=["ACCBAL"])

    def test_show_fi(self, tpt):
        """Verify show_fi runs without error."""
        from octopus.poststudy.notebook import show_fi  # noqa: PLC0415

        fi = tpt.calculate_fi(fi_type=FIType.PERMUTATION, n_repeats=2)
        show_fi(fi)

    def test_show_selected_features_tables(self, study_info):
        """Verify show_selected_features_tables runs without error."""
        from octopus.poststudy.notebook import show_selected_features_tables  # noqa: PLC0415

        feat, freq = get_selected_features(study_info)
        show_selected_features_tables(feat, freq)


class TestUtilities:
    """Tests for utility functions."""

    def test_display_table(self, capsys):
        """Verify display_table runs without error."""
        df = pd.DataFrame({"a": [1, 2]})
        display_table(df)

    def test_show_study_overview(self, study_info, capsys):
        """Verify show_study_overview prints expected info."""
        show_study_overview(study_info)
        captured = capsys.readouterr()
        assert "ML type: binary" in captured.out
        assert "Target metric: ACCBAL" in captured.out
