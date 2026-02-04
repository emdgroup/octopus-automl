"""Test experiment initialization."""

import pandas as pd
import pytest
from upath import UPath

from octopus.experiment import OctoExperiment


@pytest.fixture
def sample_data():
    """Fixture to provide sample data for testing."""
    data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [5, 4, 3, 2, 1],
        "feature3": [1, 2, 3, 4, 5],
        "target": [3, 1, 3, 1, 3],
    }
    return pd.DataFrame(data)


@pytest.fixture
def octo_experiment(sample_data):
    """Fixture to create an instance of OctoExperiment."""
    return OctoExperiment(
        id="experiment_1",
        experiment_id=1,
        task_id=1,
        depends_on_task=1,
        task_path=UPath("/path/to/sequence_item"),
        study_path="./studies/",
        study_name="test",
        ml_type="regression",
        target_metric="R2",
        positive_class=1,
        metrics=["R2"],
        imputation_method="median",
        datasplit_column="target",
        row_column="row_id",
        feature_cols=["feature1", "feature2", "feature3"],
        target_assignments={"target": [0, 1]},
        data_traindev=sample_data,
        data_test=sample_data,
    )


def test_initialization(octo_experiment):
    """Test the initialization of the OctoExperiment class."""
    assert octo_experiment.id == "experiment_1"
    assert octo_experiment.experiment_id == 1
    assert octo_experiment.task_id == 1
    assert octo_experiment.depends_on_task == 1
    assert octo_experiment.task_path == UPath("/path/to/sequence_item")
    assert octo_experiment.datasplit_column == "target"
    assert octo_experiment.row_column == "row_id"
    assert octo_experiment.feature_cols == ["feature1", "feature2", "feature3"]
    assert octo_experiment.target_assignments == {"target": [0, 1]}
    assert isinstance(octo_experiment.data_traindev, pd.DataFrame)
    assert isinstance(octo_experiment.data_test, pd.DataFrame)


def test_calculate_feature_groups(octo_experiment):
    """Test the feature group calculation."""
    feature_groups = octo_experiment.calculate_feature_groups(octo_experiment.feature_cols)
    assert isinstance(feature_groups, dict)
    assert len(feature_groups) > 0


def test_path_study(octo_experiment):
    """Test the path_study property."""
    expected_path = UPath(octo_experiment.study_path, octo_experiment.study_name)
    assert octo_experiment.path_study == expected_path


def test_ml_type(octo_experiment):
    """Test the ml_type property."""
    assert octo_experiment.ml_type == "regression"
