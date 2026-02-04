"""Test OctoStudy core class."""

import tempfile

import numpy as np
import pandas as pd
import pytest
from upath import UPath

from octopus.modules import Octo
from octopus.study import OctoClassification, OctoRegression
from octopus.study.core import _RUNNING_IN_TESTSUITE
from octopus.study.types import DatasplitType, ImputationMethod, MLType
from octopus.task import Task


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "sample_id_col": [f"S{i}" for i in range(100)],
            "feature1": np.random.rand(100),
            "feature2": np.random.randint(0, 10, 100),
            "feature3": np.random.choice(["A", "B", "C"], 100),
            "target": np.random.randint(0, 2, 100),
        }
    ).astype({"feature3": "category"})


@pytest.fixture
def basic_study():
    """Create a basic OctoClassification instance."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield OctoClassification(
            name="test_study",
            target_metric="AUCROC",
            feature_cols=["feature1", "feature2", "feature3"],
            target="target",
            sample_id_col="sample_id_col",
            path=temp_dir,
            ignore_data_health_warning=True,
        )


def test_initialization(basic_study):
    """Test OctoStudy initialization."""
    assert basic_study.name == "test_study"
    assert basic_study.ml_type == MLType.CLASSIFICATION
    assert basic_study.target_metric == "AUCROC"
    assert basic_study.feature_cols == ["feature1", "feature2", "feature3"]
    assert basic_study.target == "target"
    assert basic_study.target_cols == ["target"]  # Property should return list
    assert basic_study.sample_id_col == "sample_id_col"


@pytest.mark.parametrize(
    "study_class,param_name,param_value,expected_enum,kwargs",
    [
        (OctoRegression, "ml_type", "regression", MLType.REGRESSION, {"target_metric": "R2", "target": "target"}),
        (
            OctoClassification,
            "datasplit_type",
            "group_features",
            DatasplitType.GROUP_FEATURES,
            {"target_metric": "AUCROC", "datasplit_type": "group_features", "target": "target"},
        ),
        (
            OctoClassification,
            "imputation_method",
            "halfmin",
            ImputationMethod.HALFMIN,
            {"target_metric": "AUCROC", "imputation_method": "halfmin", "target": "target"},
        ),
    ],
)
def test_string_to_enum_conversion(study_class, param_name, param_value, expected_enum, kwargs):
    """Test that parameters accept strings and convert to enum types."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = study_class(
            name="test",
            feature_cols=["f1"],
            sample_id_col="id",
            path=temp_dir,
            **kwargs,
        )
        assert getattr(study, param_name) == expected_enum


def test_output_path_property():
    """Test that output_path is correctly computed."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoClassification(
            name="my_study",
            target_metric="AUCROC",
            feature_cols=["f1"],
            target="target",
            sample_id_col="id",
            path=temp_dir,
        )
        assert study.output_path == UPath(temp_dir) / "my_study"


def test_default_workflow():
    """Test that default workflow is a single Octo task."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoClassification(
            name="test",
            target_metric="AUCROC",
            feature_cols=["f1"],
            target="target",
            sample_id_col="id",
            path=temp_dir,
        )
        assert len(study.workflow) == 1
        assert isinstance(study.workflow[0], Octo)
        assert study.workflow[0].task_id == 0


@pytest.mark.parametrize(
    "metrics_input,expected_metrics",
    [
        (None, ["AUCROC"]),  # default metrics
        (["AUCROC", "ACCBAL", "F1"], ["AUCROC", "ACCBAL", "F1"]),  # custom metrics
    ],
)
def test_metrics(metrics_input, expected_metrics):
    """Test metrics list with default and custom values."""
    with tempfile.TemporaryDirectory() as temp_dir:
        kwargs = {
            "name": "test",
            "target_metric": "AUCROC",
            "feature_cols": ["f1"],
            "target": "target",
            "sample_id_col": "id",
            "path": temp_dir,
        }
        if metrics_input is not None:
            kwargs["metrics"] = metrics_input

        study = OctoClassification(**kwargs)
        assert study.metrics == expected_metrics


def test_default_values():
    """Test default values are set correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoClassification(
            name="test",
            target_metric="AUCROC",
            feature_cols=["f1"],
            target="target",
            sample_id_col="id",
            path=temp_dir,
        )
        assert study.datasplit_type == DatasplitType.SAMPLE
        assert study.row_id_col is None
        assert study.stratification_column is None
        assert study.positive_class == 1
        assert study.n_folds_outer == 5 if not _RUNNING_IN_TESTSUITE else 2
        assert study.datasplit_seed_outer == 0
        assert study.imputation_method == ImputationMethod.MEDIAN
        assert study.ignore_data_health_warning is False
        assert study.outer_parallelization is True
        assert study.run_single_experiment_num == -1


def test_ml_type_values():
    """Test all valid ml_type values with appropriate classes."""
    test_cases = [
        (MLType.CLASSIFICATION, OctoClassification, "AUCROC", {"target": "target"}),
        (MLType.REGRESSION, OctoRegression, "R2", {"target": "target"}),
    ]
    for expected_ml_type, study_class, metric, extra_kwargs in test_cases:
        with tempfile.TemporaryDirectory() as temp_dir:
            study = study_class(
                name="test",
                target_metric=metric,
                feature_cols=["f1"],
                sample_id_col="id",
                path=temp_dir,
                **extra_kwargs,
            )
            assert study.ml_type == expected_ml_type


def test_start_with_empty_study_valid():
    """Test that start_with_empty_study=True works with tasks that don't have load_task=True."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoClassification(
            name="test",
            target_metric="AUCROC",
            feature_cols=["f1"],
            target="target",
            sample_id_col="id",
            path=temp_dir,
            start_with_empty_study=True,
            workflow=[Octo(task_id=0), Task(task_id=1, depends_on_task=0, load_task=False)],
        )
        assert study.start_with_empty_study is True


def test_start_with_empty_study_invalid():
    """Test that start_with_empty_study=True raises error when workflow has tasks with load_task=True."""
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        pytest.raises(
            ValueError, match="Cannot set start_with_empty_study=True when workflow contains tasks with load_task=True"
        ),
    ):
        OctoClassification(
            name="test",
            target_metric="AUCROC",
            feature_cols=["f1"],
            target="target",
            sample_id_col="id",
            path=temp_dir,
            start_with_empty_study=True,
            workflow=[Octo(task_id=0), Task(task_id=1, depends_on_task=0, load_task=True)],
        )


def test_start_with_empty_study_false_with_load_task():
    """Test that start_with_empty_study=False allows tasks with load_task=True."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoClassification(
            name="test",
            target_metric="AUCROC",
            feature_cols=["f1"],
            target="target",
            sample_id_col="id",
            path=temp_dir,
            start_with_empty_study=False,
            workflow=[Octo(task_id=0), Task(task_id=1, depends_on_task=0, load_task=True)],
        )
        assert study.start_with_empty_study is False
