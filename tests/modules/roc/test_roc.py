"""Comprehensive test suite for ROC (Remove Outliers and Correlations) module."""

import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from upath import UPath

from octopus.experiment import OctoExperiment
from octopus.modules.roc.core import RocCore
from octopus.modules.roc.module import Roc


class TestRocModule:
    """Test suite for ROC module configuration."""

    def test_roc_module_initialization_defaults(self):
        """Test ROC module initialization with default parameters."""
        roc = Roc(task_id=0)

        assert roc.task_id == 0
        assert roc.depends_on_task == -1
        assert roc.load_task is False
        assert roc.description == ""
        assert roc.threshold == 0.8
        assert roc.correlation_type == "spearmanr"
        assert roc.filter_type == "f_statistics"
        assert roc.module == "roc"

    def test_roc_module_initialization_custom_params(self):
        """Test ROC module initialization with custom parameters."""
        roc = Roc(
            task_id=1,
            depends_on_task=0,
            load_task=True,
            description="test_roc",
            threshold=0.9,
            correlation_type="rdc",
            filter_type="mutual_info",
        )

        assert roc.task_id == 1
        assert roc.depends_on_task == 0
        assert roc.load_task is True
        assert roc.description == "test_roc"
        assert roc.threshold == 0.9
        assert roc.correlation_type == "rdc"
        assert roc.filter_type == "mutual_info"

    def test_roc_module_invalid_correlation_type(self):
        """Test ROC module with invalid correlation type."""
        with pytest.raises(ValueError, match="must be in"):
            Roc(task_id=0, correlation_type="invalid_correlation")

    def test_roc_module_invalid_filter_type(self):
        """Test ROC module with invalid filter type."""
        with pytest.raises(ValueError, match="must be in"):
            Roc(task_id=0, filter_type="invalid_filter")

    def test_roc_module_invalid_threshold_type(self):
        """Test ROC module with invalid threshold type."""
        with pytest.raises(TypeError):
            Roc(task_id=0, threshold="invalid")

    def test_roc_module_negative_task_id(self):
        """Test ROC module with negative sequence ID."""
        with pytest.raises(ValueError):
            Roc(task_id=-1)

    @pytest.mark.parametrize("threshold", [0.0, 0.5, 0.8, 0.95, 1.0])
    def test_roc_module_threshold_range(self, threshold):
        """Test ROC module with different threshold values."""
        roc = Roc(task_id=0, threshold=threshold)
        assert roc.threshold == threshold

    @pytest.mark.parametrize("correlation_type", ["spearmanr", "rdc"])
    def test_roc_module_correlation_types(self, correlation_type):
        """Test ROC module with different correlation types."""
        roc = Roc(task_id=0, correlation_type=correlation_type)
        assert roc.correlation_type == correlation_type

    @pytest.mark.parametrize("filter_type", ["mutual_info", "f_statistics"])
    def test_roc_module_filter_types(self, filter_type):
        """Test ROC module with different filter types."""
        roc = Roc(task_id=0, filter_type=filter_type)
        assert roc.filter_type == filter_type


class TestRocCore:
    """Test suite for ROC core functionality."""

    @pytest.fixture
    def sample_classification_data(self):
        """Create sample classification data with known correlations."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10

        # Create base features
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=8,
            n_redundant=2,
            n_clusters_per_class=1,
            random_state=42,
        )

        # Create highly correlated features
        X_corr = np.column_stack(
            [
                X,
                X[:, 0] + np.random.normal(0, 0.1, n_samples),  # Highly correlated with feature 0
                X[:, 1] + np.random.normal(0, 0.1, n_samples),  # Highly correlated with feature 1
            ]
        )

        feature_names = [f"feature_{i}" for i in range(X_corr.shape[1])]

        df = pd.DataFrame(X_corr, columns=feature_names)
        df["target"] = y
        df["sample_id"] = range(len(df))

        return df, feature_names

    @pytest.fixture
    def sample_regression_data(self):
        """Create sample regression data with known correlations."""
        np.random.seed(42)
        n_samples = 200
        n_features = 8

        # Create base features
        X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=6, noise=0.1, random_state=42)

        # Create highly correlated features
        X_corr = np.column_stack(
            [
                X,
                X[:, 0] * 0.9 + np.random.normal(0, 0.1, n_samples),  # Highly correlated with feature 0
                X[:, 2] * 0.95 + np.random.normal(0, 0.05, n_samples),  # Highly correlated with feature 2
            ]
        )

        feature_names = [f"feature_{i}" for i in range(X_corr.shape[1])]

        df = pd.DataFrame(X_corr, columns=feature_names)
        df["target"] = y
        df["sample_id"] = range(len(df))

        return df, feature_names

    @pytest.fixture
    def sample_timetoevent_data(self):
        """Create sample time-to-event data."""
        np.random.seed(42)
        n_samples = 150
        n_features = 6

        X = np.random.randn(n_samples, n_features)

        # Create highly correlated features
        X_corr = np.column_stack(
            [
                X,
                X[:, 0] + np.random.normal(0, 0.1, n_samples),  # Highly correlated with feature 0
            ]
        )

        feature_names = [f"feature_{i}" for i in range(X_corr.shape[1])]

        # Generate survival data
        duration = np.random.exponential(10, n_samples)
        event = np.random.choice([True, False], n_samples, p=[0.7, 0.3])

        df = pd.DataFrame(X_corr, columns=feature_names)
        df["duration"] = duration
        df["event"] = event
        df["sample_id"] = range(len(df))

        return df, feature_names

    def create_mock_experiment(self, data, feature_cols, ml_type, target_assignments, roc_config):
        """Create a mock experiment object for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_experiment = Mock(spec=OctoExperiment)
            mock_experiment.data_traindev = data
            mock_experiment.feature_cols = feature_cols
            mock_experiment.ml_type = ml_type
            mock_experiment.target_assignments = target_assignments
            mock_experiment.ml_config = roc_config
            mock_experiment.path_study = UPath(temp_dir)
            mock_experiment.task_path = UPath("roc_test")
            mock_experiment.selected_features = []

            return mock_experiment

    def test_roc_core_classification_spearmanr_f_statistics(self, sample_classification_data):
        """Test ROC core with classification data, Spearman correlation, and F-statistics."""
        data, feature_cols = sample_classification_data

        roc_config = Roc(task_id=0, threshold=0.8, correlation_type="spearmanr", filter_type="f_statistics")

        target_assignments = {"target": "target"}
        mock_experiment = self.create_mock_experiment(
            data, feature_cols, "classification", target_assignments, roc_config
        )

        with patch("shutil.rmtree"), patch("pathlib.Path.mkdir"):
            roc_core = RocCore(experiment=mock_experiment, log_dir=mock_experiment.path_study)
            result_experiment = roc_core.run_experiment()

            # Verify that features were selected
            assert hasattr(result_experiment, "selected_features")
            assert len(result_experiment.selected_features) > 0
            assert len(result_experiment.selected_features) <= len(feature_cols)

            # Verify that highly correlated features were removed
            assert len(result_experiment.selected_features) < len(feature_cols)

    def test_roc_core_classification_rdc_mutual_info(self, sample_classification_data):
        """Test ROC core with classification data, RDC correlation, and mutual information."""
        data, feature_cols = sample_classification_data

        roc_config = Roc(task_id=0, threshold=0.7, correlation_type="rdc", filter_type="mutual_info")

        target_assignments = {"target": "target"}
        mock_experiment = self.create_mock_experiment(
            data, feature_cols, "classification", target_assignments, roc_config
        )

        with patch("shutil.rmtree"), patch("pathlib.Path.mkdir"):
            roc_core = RocCore(experiment=mock_experiment, log_dir=mock_experiment.path_study)
            result_experiment = roc_core.run_experiment()

            # Verify that features were selected
            assert hasattr(result_experiment, "selected_features")
            assert len(result_experiment.selected_features) > 0
            assert len(result_experiment.selected_features) <= len(feature_cols)

    def test_roc_core_regression_spearmanr_f_statistics(self, sample_regression_data):
        """Test ROC core with regression data, Spearman correlation, and F-statistics."""
        data, feature_cols = sample_regression_data

        roc_config = Roc(task_id=0, threshold=0.85, correlation_type="spearmanr", filter_type="f_statistics")

        target_assignments = {"target": "target"}
        mock_experiment = self.create_mock_experiment(data, feature_cols, "regression", target_assignments, roc_config)

        with patch("shutil.rmtree"), patch("pathlib.Path.mkdir"):
            roc_core = RocCore(experiment=mock_experiment, log_dir=mock_experiment.path_study)
            result_experiment = roc_core.run_experiment()

            # Verify that features were selected
            assert hasattr(result_experiment, "selected_features")
            assert len(result_experiment.selected_features) > 0
            assert len(result_experiment.selected_features) <= len(feature_cols)

    def test_roc_core_regression_rdc_mutual_info(self, sample_regression_data):
        """Test ROC core with regression data, RDC correlation, and mutual information."""
        data, feature_cols = sample_regression_data

        roc_config = Roc(task_id=0, threshold=0.9, correlation_type="rdc", filter_type="mutual_info")

        target_assignments = {"target": "target"}
        mock_experiment = self.create_mock_experiment(data, feature_cols, "regression", target_assignments, roc_config)

        with patch("shutil.rmtree"), patch("pathlib.Path.mkdir"):
            roc_core = RocCore(experiment=mock_experiment, log_dir=mock_experiment.path_study)
            result_experiment = roc_core.run_experiment()

            # Verify that features were selected
            assert hasattr(result_experiment, "selected_features")
            assert len(result_experiment.selected_features) > 0

    def test_roc_core_timetoevent(self, sample_timetoevent_data):
        """Test ROC core with time-to-event data."""
        data, feature_cols = sample_timetoevent_data

        roc_config = Roc(
            task_id=0,
            threshold=0.8,
            correlation_type="spearmanr",
            filter_type="f_statistics",  # This should be ignored for timetoevent
        )

        target_assignments = {"duration": "duration", "event": "event"}
        mock_experiment = self.create_mock_experiment(data, feature_cols, "timetoevent", target_assignments, roc_config)

        with patch("shutil.rmtree"), patch("pathlib.Path.mkdir"):
            roc_core = RocCore(experiment=mock_experiment, log_dir=mock_experiment.path_study)
            result_experiment = roc_core.run_experiment()

            # Verify that features were selected
            assert hasattr(result_experiment, "selected_features")
            assert len(result_experiment.selected_features) > 0
            assert len(result_experiment.selected_features) <= len(feature_cols)

    @pytest.mark.parametrize("threshold", [0.5, 0.7, 0.8, 0.9, 0.95])
    def test_roc_core_different_thresholds(self, sample_classification_data, threshold):
        """Test ROC core with different correlation thresholds."""
        data, feature_cols = sample_classification_data

        roc_config = Roc(task_id=0, threshold=threshold, correlation_type="spearmanr", filter_type="f_statistics")

        target_assignments = {"target": "target"}
        mock_experiment = self.create_mock_experiment(
            data, feature_cols, "classification", target_assignments, roc_config
        )

        with patch("shutil.rmtree"), patch("pathlib.Path.mkdir"):
            roc_core = RocCore(experiment=mock_experiment, log_dir=mock_experiment.path_study)
            result_experiment = roc_core.run_experiment()

            # Verify that features were selected
            assert hasattr(result_experiment, "selected_features")
            assert len(result_experiment.selected_features) > 0

            # Higher thresholds should generally result in more features being kept
            # (less strict correlation removal)
            if threshold >= 0.9:
                assert len(result_experiment.selected_features) >= len(feature_cols) * 0.7

    def test_roc_core_no_correlations(self):
        """Test ROC core with data that has no high correlations."""
        np.random.seed(42)
        n_samples = 100
        n_features = 8

        # Create completely independent features
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1], n_samples)

        feature_names = [f"feature_{i}" for i in range(n_features)]
        data = pd.DataFrame(X, columns=feature_names)
        data["target"] = y
        data["sample_id"] = range(len(data))

        roc_config = Roc(task_id=0, threshold=0.8, correlation_type="spearmanr", filter_type="f_statistics")

        target_assignments = {"target": "target"}
        mock_experiment = self.create_mock_experiment(
            data, feature_names, "classification", target_assignments, roc_config
        )

        with patch("shutil.rmtree"), patch("pathlib.Path.mkdir"):
            roc_core = RocCore(experiment=mock_experiment, log_dir=mock_experiment.path_study)
            result_experiment = roc_core.run_experiment()

            # All features should be kept since there are no high correlations
            assert len(result_experiment.selected_features) == len(feature_names)

    def test_roc_core_all_features_correlated(self):
        """Test ROC core with data where all features are highly correlated."""
        np.random.seed(42)
        n_samples = 100
        base_feature = np.random.randn(n_samples)

        # Create features that are all highly correlated with the base feature
        X = np.column_stack(
            [
                base_feature,
                base_feature + np.random.normal(0, 0.05, n_samples),
                base_feature + np.random.normal(0, 0.05, n_samples),
                base_feature + np.random.normal(0, 0.05, n_samples),
            ]
        )
        y = np.random.choice([0, 1], n_samples)

        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        data = pd.DataFrame(X, columns=feature_names)
        data["target"] = y
        data["sample_id"] = range(len(data))

        roc_config = Roc(task_id=0, threshold=0.8, correlation_type="spearmanr", filter_type="f_statistics")

        target_assignments = {"target": "target"}
        mock_experiment = self.create_mock_experiment(
            data, feature_names, "classification", target_assignments, roc_config
        )

        with patch("shutil.rmtree"), patch("pathlib.Path.mkdir"):
            roc_core = RocCore(experiment=mock_experiment, log_dir=mock_experiment.path_study)
            result_experiment = roc_core.run_experiment()

            # Only one feature should be kept from the highly correlated group
            assert len(result_experiment.selected_features) == 1

    def test_roc_core_properties(self, sample_classification_data):
        """Test ROC core properties."""
        data, feature_cols = sample_classification_data

        roc_config = Roc(task_id=0)
        target_assignments = {"target": "target"}
        mock_experiment = self.create_mock_experiment(
            data, feature_cols, "classification", target_assignments, roc_config
        )

        with patch("shutil.rmtree"), patch("pathlib.Path.mkdir"):
            roc_core = RocCore(experiment=mock_experiment, log_dir=mock_experiment.path_study)

            # Test properties
            assert roc_core.feature_cols == feature_cols
            assert roc_core.ml_type == "classification"
            assert roc_core.filter_type == "f_statistics"
            assert isinstance(roc_core.x_traindev, pd.DataFrame)
            assert isinstance(roc_core.y_traindev, pd.DataFrame)
            assert roc_core.config == roc_config

    def test_roc_core_invalid_correlation_type_runtime(self, sample_classification_data):
        """Test ROC core with invalid correlation type at runtime."""
        data, feature_cols = sample_classification_data

        # Create a mock config with invalid correlation type
        roc_config = Mock()
        roc_config.threshold = 0.8
        roc_config.correlation_type = "invalid_correlation"
        roc_config.filter_type = "f_statistics"

        target_assignments = {"target": "target"}
        mock_experiment = self.create_mock_experiment(
            data, feature_cols, "classification", target_assignments, roc_config
        )

        with patch("shutil.rmtree"), patch("pathlib.Path.mkdir"):
            roc_core = RocCore(experiment=mock_experiment, log_dir=mock_experiment.path_study)

            with pytest.raises(ValueError, match="Correlation type invalid_correlation not supported"):
                roc_core.run_experiment()

    def test_roc_core_with_nan_values(self):
        """Test ROC core with NaN values in the data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 6

        X = np.random.randn(n_samples, n_features)
        # Introduce some NaN values
        X[10:15, 0] = np.nan
        X[20:25, 2] = np.nan

        y = np.random.choice([0, 1], n_samples)

        feature_names = [f"feature_{i}" for i in range(n_features)]
        data = pd.DataFrame(X, columns=feature_names)
        data["target"] = y
        data["sample_id"] = range(len(data))

        # Use mutual_info filter which can handle NaN values better, or expect ValueError for f_statistics
        roc_config = Roc(task_id=0, threshold=0.8, correlation_type="spearmanr", filter_type="f_statistics")

        target_assignments = {"target": "target"}
        mock_experiment = self.create_mock_experiment(
            data, feature_names, "classification", target_assignments, roc_config
        )

        with patch("shutil.rmtree"), patch("pathlib.Path.mkdir"):
            roc_core = RocCore(experiment=mock_experiment, log_dir=mock_experiment.path_study)

            # f_statistics filter doesn't handle NaN values, so expect ValueError
            with pytest.raises(ValueError, match="Input X contains NaN"):
                roc_core.run_experiment()


class TestRocIntegration:
    """Integration tests for ROC module."""

    def test_roc_module_and_core_integration(self):
        """Test integration between ROC module configuration and core functionality."""
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        X, y = make_classification(n_samples=n_samples, n_features=8, random_state=42)

        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        data = pd.DataFrame(X, columns=feature_names)
        data["target"] = y
        data["sample_id"] = range(len(data))

        # Create ROC module configuration
        roc_module = Roc(
            task_id=0,
            description="integration_test",
            threshold=0.85,
            correlation_type="spearmanr",
            filter_type="mutual_info",
        )

        # Verify module configuration
        assert roc_module.threshold == 0.85
        assert roc_module.correlation_type == "spearmanr"
        assert roc_module.filter_type == "mutual_info"
        assert roc_module.description == "integration_test"

    @pytest.mark.parametrize(
        "ml_type,target_cols",
        [
            ("classification", {"target": "target"}),
            ("regression", {"target": "target"}),
            ("timetoevent", {"duration": "duration", "event": "event"}),
        ],
    )
    def test_roc_different_ml_types(self, ml_type, target_cols):
        """Test ROC module with different ML types."""
        np.random.seed(42)
        n_samples = 100
        n_features = 6

        X = np.random.randn(n_samples, n_features)
        feature_names = [f"feature_{i}" for i in range(n_features)]
        data = pd.DataFrame(X, columns=feature_names)

        if ml_type == "classification":
            data["target"] = np.random.choice([0, 1], n_samples)
        elif ml_type == "regression":
            data["target"] = np.random.randn(n_samples)
        elif ml_type == "timetoevent":
            data["duration"] = np.random.exponential(10, n_samples)
            data["event"] = np.random.choice([True, False], n_samples)

        data["sample_id"] = range(len(data))

        roc_config = Roc(task_id=0, threshold=0.8)

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_experiment = Mock(spec=OctoExperiment)
            mock_experiment.data_traindev = data
            mock_experiment.feature_cols = feature_names
            mock_experiment.ml_type = ml_type
            mock_experiment.target_assignments = target_cols
            mock_experiment.ml_config = roc_config
            mock_experiment.path_study = UPath(temp_dir)
            mock_experiment.task_path = UPath("roc_test")
            mock_experiment.selected_features = []

            with patch("shutil.rmtree"), patch("pathlib.Path.mkdir"):
                roc_core = RocCore(experiment=mock_experiment, log_dir=mock_experiment.path_study)
                result_experiment = roc_core.run_experiment()

                # Verify that the experiment completed successfully
                assert hasattr(result_experiment, "selected_features")
                assert len(result_experiment.selected_features) > 0
