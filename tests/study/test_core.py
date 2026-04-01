"""Test OctoStudy core class."""

import tempfile

import numpy as np
import pandas as pd
import pytest

from octopus.modules import Octo
from octopus.study import OctoClassification, OctoRegression
from octopus.study.core import _RUNNING_IN_TESTSUITE
from octopus.types import MLType


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
            study_name="test_study",
            target_metric="AUCROC",
            feature_cols=["feature1", "feature2", "feature3"],
            target_col="target",
            sample_id_col="sample_id_col",
            studies_directory=temp_dir,
        )


def test_initialization(basic_study):
    """Test OctoStudy initialization."""
    assert basic_study.study_name == "test_study"
    assert basic_study.ml_type is None  # ml_type is determined during data validation
    assert basic_study.target_metric == "AUCROC"
    assert basic_study.feature_cols == ["feature1", "feature2", "feature3"]
    assert basic_study.target_col == "target"
    assert basic_study.sample_id_col == "sample_id_col"


def test_regression_ml_type():
    """Test that OctoRegression sets ml_type to regression."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoRegression(
            study_name="test",
            target_metric="R2",
            feature_cols=["f1"],
            target_col="target",
            sample_id_col="id",
            studies_directory=temp_dir,
        )
        assert study.ml_type == MLType.REGRESSION


def test_default_workflow():
    """Test that default workflow is a single Octo task."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoClassification(
            study_name="test",
            target_metric="AUCROC",
            feature_cols=["f1"],
            target_col="target",
            sample_id_col="id",
            studies_directory=temp_dir,
        )
        assert len(study.workflow) == 1
        assert isinstance(study.workflow[0], Octo)
        assert study.workflow[0].task_id == 0


def test_default_values():
    """Test default values are set correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoClassification(
            study_name="test",
            target_metric="AUCROC",
            feature_cols=["f1"],
            target_col="target",
            sample_id_col="id",
            studies_directory=temp_dir,
        )
        assert study.row_id_col is None
        assert study.stratification_col is None
        assert study.positive_class is None  # positive_class is determined during data validation
        assert study.n_outer_splits == 5 if not _RUNNING_IN_TESTSUITE else 2
        assert study.outer_split_seed == 0
        assert study.single_outer_split is None


def test_ml_type_values():
    """Test all valid ml_type values with appropriate classes."""
    test_cases = [
        (None, OctoClassification, "AUCROC", {"target_col": "target"}),  # ml_type determined during data validation
        (MLType.REGRESSION, OctoRegression, "R2", {"target_col": "target"}),
    ]
    for expected_ml_type, study_class, metric, extra_kwargs in test_cases:
        with tempfile.TemporaryDirectory() as temp_dir:
            study = study_class(
                study_name="test",
                target_metric=metric,
                feature_cols=["f1"],
                sample_id_col="id",
                studies_directory=temp_dir,
                **extra_kwargs,
            )
            assert study.ml_type == expected_ml_type
