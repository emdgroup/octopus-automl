"""Tests for octopus/predict.py."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from upath import UPath

from octopus.experiment import OctoExperiment
from octopus.predict import OctoPredict
from octopus.results import ModuleResults


@pytest.fixture
def sample_data():
    """Fixture to provide sample data for testing."""
    np.random.seed(42)
    data = {
        "row_id": [f"row_{i}" for i in range(100)],
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "feature3": np.random.randn(100),
        "target": np.random.choice([0, 1], 100),
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_model():
    """Fixture to create a mock model that returns predictions based on input size."""
    model = Mock()

    # Create side_effect functions that return arrays matching input size
    def predict_side_effect(X):
        n_samples = len(X)
        np.random.seed(42)
        return np.random.choice([0, 1], n_samples)

    def predict_proba_side_effect(X):
        n_samples = len(X)
        np.random.seed(42)
        # Generate random probabilities that sum to 1
        probs = np.random.rand(n_samples, 2)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    model.predict = Mock(side_effect=predict_side_effect)
    model.predict_proba = Mock(side_effect=predict_proba_side_effect)
    model.classes_ = np.array([0, 1])  # Required for classification tasks
    return model


@pytest.fixture
def mock_experiment(sample_data, mock_model):
    """Fixture to create a mock OctoExperiment."""
    experiment = Mock(spec=OctoExperiment)
    experiment.id = "experiment_0"
    experiment.experiment_id = 0
    experiment.task_id = 0
    experiment.depends_on_task = -1
    experiment._task_path = UPath("outersplit0/workflowtask0")
    experiment.datasplit_column = "target"
    experiment.row_column = "row_id"
    experiment.feature_cols = ["feature1", "feature2", "feature3"]
    experiment.target_assignments = {"target": "target"}
    experiment.target_metric = "AUCROC"
    experiment.ml_type = "classification"
    experiment.positive_class = 1
    experiment.data_traindev = sample_data.iloc[:80]
    experiment.data_test = sample_data.iloc[80:]
    experiment.feature_groups = {"group0": ["feature1", "feature2"]}

    # Create mock results
    mock_result = Mock(spec=ModuleResults)
    mock_result.model = mock_model
    experiment.results = {"best": mock_result}

    return experiment


@pytest.fixture
def mock_study_path(tmp_path):
    """Fixture to create a mock study directory structure."""
    study_path = tmp_path / "test_study"
    study_path.mkdir()

    # Create config.json file (new architecture)
    config_data = {
        "name": "test_study",
        "ml_type": "classification",
        "feature_cols": ["feature1", "feature2", "feature3"],
        "row_id": "row_id",
        "sample_id": "row_id",
        "target_cols": ["target"],
        "target_metric": "AUCROC",
        "n_folds_outer": 3,
        "path": str(study_path),
        "workflow": [
            {
                "task_id": 0,
                "depends_on_task": -1,
                "description": "step_1_octo",
                "models": ["RandomForestClassifier"],
                "n_trials": 1,
            }
        ],
    }
    config_path = study_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)

    # Create experiment directories with pickle files (using outersplit naming)
    for exp_id in range(3):
        exp_dir = study_path / f"outersplit{exp_id}" / "workflowtask0"
        exp_dir.mkdir(parents=True)

        # Create a dummy pickle file
        pkl_file = exp_dir / f"exp{exp_id}_0.pkl"
        pkl_file.touch()

        # Create results directory
        results_dir = exp_dir / "results"
        results_dir.mkdir()

    return study_path


@pytest.fixture
def predictor_with_experiments(mock_study_path, mock_experiment, sample_data, mock_model):
    """Fixture to create an OctoPredict instance with loaded experiments."""
    with patch("octopus.predict.OctoExperiment.from_pickle", return_value=mock_experiment):
        predictor = OctoPredict(study_path=mock_study_path)

        # Manually populate experiments since mocking file system is complex
        # Using SimpleNamespace for attribute access instead of custom dict class
        for exp_id in range(3):
            predictor.experiments[exp_id] = SimpleNamespace(
                id=exp_id,
                model=mock_model,
                data_traindev=sample_data.iloc[:80],
                data_test=sample_data.iloc[80:],
                feature_cols=["feature1", "feature2", "feature3"],
                row_column="row_id",
                target_assignments={"target": "target"},
                target_metric="AUCROC",
                ml_type="classification",
                feature_group_dict={"group0": ["feature1", "feature2"]},
                positive_class=1,
            )

        return predictor


class TestOctoPredictInitialization:
    """Tests for OctoPredict initialization."""

    def test_initialization_with_default_task_id(self, mock_study_path, mock_experiment):
        """Test initialization with default task_id."""
        with patch("octopus.predict.OctoExperiment.from_pickle", return_value=mock_experiment):
            predictor = OctoPredict(study_path=mock_study_path)

            assert predictor.study_path == mock_study_path
            assert predictor.task_id == 0  # Should use last workflow task
            assert predictor.results_key == "best"
            assert isinstance(predictor.experiments, dict)

    def test_initialization_with_custom_task_id(self, mock_study_path, mock_experiment):
        """Test initialization with custom task_id."""
        with patch("octopus.predict.OctoExperiment.from_pickle", return_value=mock_experiment):
            predictor = OctoPredict(study_path=mock_study_path, task_id=0)

            assert predictor.task_id == 0

    def test_config_property(self, mock_study_path, mock_experiment):
        """Test config property returns dict from config.json."""
        with patch("octopus.predict.OctoExperiment.from_pickle", return_value=mock_experiment):
            predictor = OctoPredict(study_path=mock_study_path)

            assert isinstance(predictor.config, dict)
            assert predictor.config["name"] == "test_study"
            assert predictor.config["ml_type"] == "classification"

    def test_n_experiments_property(self, mock_study_path, mock_experiment):
        """Test n_experiments property."""
        with patch("octopus.predict.OctoExperiment.from_pickle", return_value=mock_experiment):
            predictor = OctoPredict(study_path=mock_study_path)

            assert predictor.n_experiments == 3


class TestOctoPredictPredictionMethods:
    """Tests for OctoPredict prediction methods."""

    def test_predict_return_array(self, predictor_with_experiments, sample_data):
        """Test predict method returning numpy array."""
        result = predictor_with_experiments.predict(sample_data.iloc[:5], return_df=False)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 5

    def test_predict_return_dataframe(self, predictor_with_experiments, sample_data):
        """Test predict method returning DataFrame."""
        result = predictor_with_experiments.predict(sample_data.iloc[:5], return_df=True)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert "prediction" in result.columns

    def test_predict_missing_features(self, predictor_with_experiments):
        """Test predict method with missing features."""
        data = pd.DataFrame({"feature1": [1, 2, 3]})

        with pytest.raises(ValueError, match="Features missing in provided dataset"):
            predictor_with_experiments.predict(data)

    def test_predict_proba_return_array(self, predictor_with_experiments, sample_data):
        """Test predict_proba method returning numpy array."""
        result = predictor_with_experiments.predict_proba(sample_data.iloc[:5], return_df=False)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 5

    def test_predict_proba_return_dataframe(self, predictor_with_experiments, sample_data):
        """Test predict_proba method returning DataFrame."""
        result = predictor_with_experiments.predict_proba(sample_data.iloc[:5], return_df=True)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_predict_test(self, predictor_with_experiments):
        """Test predict_test method."""
        result = predictor_with_experiments.predict_test()

        assert isinstance(result, pd.DataFrame)
        assert "prediction" in result.columns
        assert "prediction_std" in result.columns
        assert "n" in result.columns
        assert len(result) > 0

    def test_predict_proba_test(self, predictor_with_experiments):
        """Test predict_proba_test method."""
        result = predictor_with_experiments.predict_proba_test()

        assert isinstance(result, pd.DataFrame)
        assert "probability" in result.columns
        assert "probability_std" in result.columns
        assert "n" in result.columns
        assert len(result) > 0


class TestOctoPredictFeatureImportance:
    """Tests for OctoPredict feature importance methods."""

    @pytest.fixture
    def mock_fi_results(self):
        """Fixture to create mock feature importance results."""
        return pd.DataFrame(
            {
                "feature": ["feature1", "feature2", "feature3"],
                "importance": [0.5, 0.3, 0.2],
                "stddev": [0.1, 0.05, 0.03],
                "p-value": [0.01, 0.05, 0.1],
                "n": [10, 10, 10],
                "ci_low_95": [0.4, 0.25, 0.17],
                "ci_high_95": [0.6, 0.35, 0.23],
            }
        )

    def test_calculate_fi_permutation(self, predictor_with_experiments, sample_data, mock_fi_results):
        """Test calculate_fi with permutation method."""
        with (
            patch("octopus.predict.get_fi_permutation", return_value=mock_fi_results),
            patch("octopus.predict.OctoPredict._plot_permutation_fi"),
        ):
            predictor_with_experiments.calculate_fi(sample_data, n_repeat=10, fi_type="permutation")

            assert "fi_table_permutation_ensemble" in predictor_with_experiments.results
            assert isinstance(predictor_with_experiments.results["fi_table_permutation_ensemble"], pd.DataFrame)

    def test_calculate_fi_invalid_shap_type(self, predictor_with_experiments, sample_data):
        """Test calculate_fi with invalid shap_type."""
        with pytest.raises(ValueError, match="Specified shap_type not supported"):
            predictor_with_experiments.calculate_fi(sample_data, fi_type="shap", shap_type="invalid_shap")

    def test_calculate_fi_test_invalid_shap_type(self, predictor_with_experiments):
        """Test calculate_fi_test with invalid shap_type."""
        with pytest.raises(ValueError, match="Specified shap_type not supported"):
            predictor_with_experiments.calculate_fi_test(fi_type="shap", shap_type="invalid_shap")


class TestOctoPredictPlotting:
    """Tests for OctoPredict plotting methods."""

    def test_plot_permutation_fi(self, predictor_with_experiments, mock_study_path):
        """Test _plot_permutation_fi method."""
        df = pd.DataFrame(
            {
                "feature": ["feature1", "feature2", "feature3"],
                "importance": [0.5, 0.3, 0.2],
                "ci_low_95": [0.4, 0.25, 0.17],
                "ci_high_95": [0.6, 0.35, 0.23],
            }
        )

        with patch("octopus.predict.PdfPages"), patch("octopus.predict.plt"):
            predictor_with_experiments._plot_permutation_fi(0, df)

            # Check that the results directory would be created
            expected_path = mock_study_path / "outersplit0" / "workflowtask0" / "results"
            assert expected_path.exists()

    def test_plot_shap_fi(self, predictor_with_experiments, mock_study_path):
        """Test _plot_shap_fi method."""
        df = pd.DataFrame(
            {
                "feature": ["feature1", "feature2", "feature3"],
                "importance": [0.5, 0.3, 0.2],
            }
        )
        shap_values = np.random.randn(100, 3)
        data = pd.DataFrame(np.random.randn(100, 3), columns=["feature1", "feature2", "feature3"])

        with (
            patch("octopus.predict.PdfPages"),
            patch("octopus.predict.shap.summary_plot"),
            patch("octopus.predict.plt"),
        ):
            predictor_with_experiments._plot_shap_fi(0, df, shap_values, data)

            # Check that the results directory would be created
            expected_path = mock_study_path / "outersplit0" / "workflowtask0" / "results"
            assert expected_path.exists()
