"""Test metrics utility functions."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge

from octopus.metrics.utils import (
    get_performance_from_model,
    get_performance_from_predictions,
    get_score_from_model,
    get_score_from_prediction,
)


class TestGetPerformanceFromModel:
    """Test get_performance_from_model function."""

    def test_binary_classification(self):
        """Test binary classification."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.DataFrame(np.random.randint(0, 2, 100), columns=["target"])
        data = pd.concat([X, y], axis=1)

        model = LogisticRegression(random_state=42)
        model.fit(X, y.values.ravel())

        performance = get_performance_from_model(
            model=model,
            data=data,
            feature_cols=X.columns.tolist(),
            target_metric="AUCROC",
            target_assignments={"default": "target"},
            positive_class=1,
        )

        assert isinstance(performance, float)
        assert 0 <= performance <= 1

    def test_multiclass_classification(self):
        """Test multiclass classification."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(150, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.DataFrame(np.random.randint(0, 3, 150), columns=["target"])
        data = pd.concat([X, y], axis=1)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y.values.ravel())

        performance = get_performance_from_model(
            model=model,
            data=data,
            feature_cols=X.columns.tolist(),
            target_metric="ACCBAL_MC",
            target_assignments={"default": "target"},
        )

        assert isinstance(performance, float)
        assert 0 <= performance <= 1

    def test_regression(self):
        """Test regression."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.DataFrame(np.random.randn(100) * 10 + 50, columns=["target"])
        data = pd.concat([X, y], axis=1)

        model = Ridge(random_state=42)
        model.fit(X, y.values.ravel())

        performance = get_performance_from_model(
            model=model,
            data=data,
            feature_cols=X.columns.tolist(),
            target_metric="R2",
            target_assignments={"default": "target"},
        )

        assert isinstance(performance, float)


class TestGetPerformanceFromPredictions:
    """Test get_performance_from_predictions function.

    Tests use prediction format from training.py (lines 408-432):
    - Binary/multiclass: row_column, "prediction", target, probability columns (as int)
    - Regression: row_column, "prediction", target
    """

    def test_binary_classification(self):
        """Test binary classification predictions."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row": [0, 1, 2, 3, 4],
                        "prediction": [0, 1, 1, 0, 1],
                        "target": [0, 1, 1, 0, 0],
                        0: [0.8, 0.3, 0.2, 0.9, 0.4],
                        1: [0.2, 0.7, 0.8, 0.1, 0.6],
                    }
                )
            }
        }

        performance = get_performance_from_predictions(
            predictions=predictions, target_metric="AUCROC", target_assignments={"default": "target"}, positive_class=1
        )

        assert "training_0" in performance
        assert "dev" in performance["training_0"]
        assert isinstance(performance["training_0"]["dev"], float)

    def test_multiclass_standard_row_column(self):
        """Test multiclass with standard 'row' column."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row": [0, 1, 2, 3, 4],
                        "prediction": [0, 1, 2, 0, 1],
                        "target": [0, 1, 2, 0, 2],
                        0: [0.7, 0.2, 0.1, 0.8, 0.3],
                        1: [0.2, 0.6, 0.2, 0.1, 0.5],
                        2: [0.1, 0.2, 0.7, 0.1, 0.2],
                    }
                )
            }
        }

        performance = get_performance_from_predictions(
            predictions=predictions, target_metric="ACCBAL_MC", target_assignments={"default": "target"}
        )

        assert "training_0" in performance
        assert isinstance(performance["training_0"]["dev"], float)

    def test_multiclass_with_row_id_column(self):
        """Test multiclass with numeric 'row_id' column (should be excluded from probabilities)."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row_id": [10, 20, 30, 40, 50],  # numeric row identifier
                        "prediction": [0, 1, 2, 0, 1],
                        "target": [0, 1, 2, 0, 2],
                        0: [0.7, 0.2, 0.1, 0.8, 0.3],
                        1: [0.2, 0.6, 0.2, 0.1, 0.5],
                        2: [0.1, 0.2, 0.7, 0.1, 0.2],
                    }
                )
            }
        }

        performance = get_performance_from_predictions(
            predictions=predictions, target_metric="ACCBAL_MC", target_assignments={"default": "target"}
        )

        assert "training_0" in performance
        assert isinstance(performance["training_0"]["dev"], float)

    def test_multiclass_with_sample_id_col_column(self):
        """Test multiclass with numeric 'sample_id_col' column (should be excluded from probabilities)."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "sample_id_col": [100, 101, 102, 103, 104],  # another numeric identifier
                        "prediction": [0, 1, 2, 0, 1],
                        "target": [0, 1, 2, 0, 2],
                        0: [0.7, 0.2, 0.1, 0.8, 0.3],
                        1: [0.2, 0.6, 0.2, 0.1, 0.5],
                        2: [0.1, 0.2, 0.7, 0.1, 0.2],
                    }
                )
            }
        }

        performance = get_performance_from_predictions(
            predictions=predictions, target_metric="ACCBAL_MC", target_assignments={"default": "target"}
        )

        assert "training_0" in performance
        assert isinstance(performance["training_0"]["dev"], float)

    def test_multiclass_with_string_row_column(self):
        """Test multiclass with string row column (should be excluded from probabilities)."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "patient_id": ["P001", "P002", "P003", "P004", "P005"],  # string identifier
                        "prediction": [0, 1, 2, 0, 1],
                        "target": [0, 1, 2, 0, 2],
                        0: [0.7, 0.2, 0.1, 0.8, 0.3],
                        1: [0.2, 0.6, 0.2, 0.1, 0.5],
                        2: [0.1, 0.2, 0.7, 0.1, 0.2],
                    }
                )
            }
        }

        performance = get_performance_from_predictions(
            predictions=predictions, target_metric="ACCBAL_MC", target_assignments={"default": "target"}
        )

        assert "training_0" in performance
        assert isinstance(performance["training_0"]["dev"], float)

    def test_regression(self):
        """Test regression predictions."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row": [0, 1, 2, 3, 4],
                        "prediction": [50.2, 48.7, 52.1, 49.3, 51.8],
                        "target": [50.0, 49.0, 52.0, 49.5, 51.5],
                    }
                )
            }
        }

        performance = get_performance_from_predictions(
            predictions=predictions, target_metric="R2", target_assignments={"default": "target"}
        )

        assert "training_0" in performance
        assert isinstance(performance["training_0"]["dev"], float)


class TestGetScoreFromPrediction:
    """Test get_score_from_prediction function."""

    def test_maximize_metric(self):
        """Test score calculation for maximize metric (AUROC)."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row": [0, 1, 2, 3, 4],
                        "prediction": [0, 1, 1, 0, 1],
                        "target": [0, 1, 1, 0, 0],
                        0: [0.8, 0.3, 0.2, 0.9, 0.4],
                        1: [0.2, 0.7, 0.8, 0.1, 0.6],
                    }
                )
            }
        }

        scores = get_score_from_prediction(
            predictions=predictions, target_metric="AUCROC", target_assignments={"default": "target"}, positive_class=1
        )

        assert "training_0" in scores
        assert scores["training_0"]["dev"] > 0  # maximize: score = performance

    def test_minimize_metric(self):
        """Test score calculation for minimize metric (MSE)."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row": [0, 1, 2, 3, 4],
                        "prediction": [50.2, 48.7, 52.1, 49.3, 51.8],
                        "target": [50.0, 49.0, 52.0, 49.5, 51.5],
                    }
                )
            }
        }

        scores = get_score_from_prediction(
            predictions=predictions, target_metric="MSE", target_assignments={"default": "target"}
        )

        assert "training_0" in scores
        assert scores["training_0"]["dev"] < 0  # minimize: score = -performance


class TestGetScoreFromModel:
    """Test get_score_from_model function."""

    def test_maximize_metric(self):
        """Test score calculation for maximize metric."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.DataFrame(np.random.randint(0, 2, 100), columns=["target"])
        data = pd.concat([X, y], axis=1)

        model = LogisticRegression(random_state=42)
        model.fit(X, y.values.ravel())

        score = get_score_from_model(
            model=model,
            data=data,
            feature_cols=X.columns.tolist(),
            target_metric="AUCROC",
            target_assignments={"default": "target"},
            positive_class=1,
        )

        assert isinstance(score, float)
        assert score > 0  # maximize: score = performance

    def test_minimize_metric(self):
        """Test score calculation for minimize metric."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.DataFrame(np.random.randn(100) * 10 + 50, columns=["target"])
        data = pd.concat([X, y], axis=1)

        model = Ridge(random_state=42)
        model.fit(X, y.values.ravel())

        score = get_score_from_model(
            model=model,
            data=data,
            feature_cols=X.columns.tolist(),
            target_metric="MSE",
            target_assignments={"default": "target"},
        )

        assert isinstance(score, float)
        assert score < 0  # minimize: score = -performance
