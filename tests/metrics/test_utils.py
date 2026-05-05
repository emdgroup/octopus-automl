"""Test metrics utility functions."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge

from octopus.metrics import Metrics
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
    - Binary/multiclass: row_id_col, "prediction", target, probability columns (as int)
    - Regression: row_id_col, "prediction", target
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

    def test_multiclass_standard_row_id_col(self):
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

    def test_multiclass_with_row_id_col(self):
        """Test multiclass with numeric 'row_id' column (should be excluded from probabilities)."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row_id_col": [10, 20, 30, 40, 50],  # numeric row identifier
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

    def test_multiclass_with_string_row_id_col(self):
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


class TestMulticlassNonConsecutiveLabels:
    """Test multiclass with non-consecutive integer class labels."""

    def test_model_based_non_consecutive_labels(self):
        """Test get_performance_from_model with labels [1, 3, 5]."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(150, 5), columns=[f"f{i}" for i in range(5)])
        y = np.random.choice([1, 3, 5], 150)
        data = pd.concat([X, pd.DataFrame({"target": y})], axis=1)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        performance = get_performance_from_model(
            model=model,
            data=data,
            feature_cols=X.columns.tolist(),
            target_metric="ACCBAL_MC",
            target_assignments={"default": "target"},
        )

        assert isinstance(performance, float)
        assert 0 <= performance <= 1

    def test_model_based_non_zero_based_labels(self):
        """Test get_performance_from_model with labels [1, 2, 3]."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(150, 5), columns=[f"f{i}" for i in range(5)])
        y = np.random.choice([1, 2, 3], 150)
        data = pd.concat([X, pd.DataFrame({"target": y})], axis=1)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        performance = get_performance_from_model(
            model=model,
            data=data,
            feature_cols=X.columns.tolist(),
            target_metric="AUCROC_MACRO",
            target_assignments={"default": "target"},
        )

        assert isinstance(performance, float)
        assert 0 <= performance <= 1

    def test_predictions_based_non_consecutive_labels(self):
        """Test get_performance_from_predictions with labels [1, 3, 5]."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row": [0, 1, 2, 3, 4, 5],
                        "prediction": [1, 3, 5, 1, 3, 5],
                        "target": [1, 3, 5, 1, 5, 3],
                        1: [0.7, 0.1, 0.1, 0.8, 0.1, 0.2],
                        3: [0.2, 0.8, 0.1, 0.1, 0.2, 0.6],
                        5: [0.1, 0.1, 0.8, 0.1, 0.7, 0.2],
                    }
                )
            }
        }

        performance = get_performance_from_predictions(
            predictions=predictions,
            target_metric="ACCBAL_MC",
            target_assignments={"default": "target"},
        )

        assert isinstance(performance["training_0"]["dev"], float)
        assert 0 <= performance["training_0"]["dev"] <= 1

    def test_predictions_based_non_zero_based_labels(self):
        """Test get_performance_from_predictions with labels [1, 2, 3]."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row": [0, 1, 2, 3, 4, 5],
                        "prediction": [1, 2, 3, 1, 2, 3],
                        "target": [1, 2, 3, 1, 3, 2],
                        1: [0.7, 0.1, 0.1, 0.8, 0.1, 0.2],
                        2: [0.2, 0.8, 0.1, 0.1, 0.2, 0.6],
                        3: [0.1, 0.1, 0.8, 0.1, 0.7, 0.2],
                    }
                )
            }
        }

        performance = get_performance_from_predictions(
            predictions=predictions,
            target_metric="AUCROC_MACRO",
            target_assignments={"default": "target"},
        )

        assert isinstance(performance["training_0"]["dev"], float)
        assert 0 <= performance["training_0"]["dev"] <= 1

    def test_predictions_missing_probability_columns_raises(self):
        """Test ValueError when no integer-named probability columns exist."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row": [0, 1, 2],
                        "prediction": [1, 2, 3],
                        "target": [1, 2, 3],
                        "prob_1": [0.7, 0.1, 0.2],
                        "prob_2": [0.2, 0.8, 0.1],
                        "prob_3": [0.1, 0.1, 0.7],
                    }
                )
            }
        }

        with pytest.raises(ValueError, match="at least 2 integer-named probability columns"):
            get_performance_from_predictions(
                predictions=predictions,
                target_metric="AUCROC_MACRO",
                target_assignments={"default": "target"},
            )

    def test_predictions_missing_class_in_prob_columns_raises(self):
        """Test ValueError when target has a class not in probability columns."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row": [0, 1, 2],
                        "prediction": [1, 3, 5],
                        "target": [1, 3, 5],
                        1: [0.7, 0.2, 0.1],
                        3: [0.3, 0.8, 0.2],
                    }
                )
            }
        }

        with pytest.raises(ValueError, match="missing probability columns"):
            get_performance_from_predictions(
                predictions=predictions,
                target_metric="AUCROC_MACRO",
                target_assignments={"default": "target"},
            )


class TestBinaryBinarization:
    """Test binary classification with non-{0,1} labels."""

    def test_model_binary_labels_0_2_positive_class_2_aucroc(self):
        """Test AUCROC with binary labels {0, 2} and positive_class=2."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = np.random.choice([0, 2], 100)
        data = pd.concat([X, pd.DataFrame({"target": y})], axis=1)

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        performance = get_performance_from_model(
            model=model,
            data=data,
            feature_cols=X.columns.tolist(),
            target_metric="AUCROC",
            target_assignments={"default": "target"},
            positive_class=2,
        )

        assert isinstance(performance, float)
        assert 0 <= performance <= 1

    def test_model_binary_labels_0_2_positive_class_0_aucroc(self):
        """Test AUCROC with binary labels {0, 2} and positive_class=0 (was inverted before fix)."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = np.where(np.random.randn(100) > 0, 0, 2)
        data = pd.concat([X, pd.DataFrame({"target": y})], axis=1)

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        performance = get_performance_from_model(
            model=model,
            data=data,
            feature_cols=X.columns.tolist(),
            target_metric="AUCROC",
            target_assignments={"default": "target"},
            positive_class=0,
        )

        assert isinstance(performance, float)
        assert 0 <= performance <= 1
        assert performance > 0.4

    def test_model_binary_labels_0_2_thresholded_metrics(self):
        """Test ACCBAL and F1 with binary labels {0, 2}."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = np.random.choice([0, 2], 100)
        data = pd.concat([X, pd.DataFrame({"target": y})], axis=1)

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        for metric_name in ("ACCBAL", "F1"):
            performance = get_performance_from_model(
                model=model,
                data=data,
                feature_cols=X.columns.tolist(),
                target_metric=metric_name,
                target_assignments={"default": "target"},
                positive_class=2,
            )

            assert isinstance(performance, float)
            assert 0 <= performance <= 1

    def test_model_binary_labels_neg1_1(self):
        """Test all metrics with binary labels {-1, 1} and positive_class=1."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = np.random.choice([-1, 1], 100)
        data = pd.concat([X, pd.DataFrame({"target": y})], axis=1)

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        for metric_name in ("AUCROC", "ACCBAL", "F1"):
            performance = get_performance_from_model(
                model=model,
                data=data,
                feature_cols=X.columns.tolist(),
                target_metric=metric_name,
                target_assignments={"default": "target"},
                positive_class=1,
            )

            assert isinstance(performance, float)
            assert 0 <= performance <= 1

    def test_predictions_binary_labels_0_2(self):
        """Test get_performance_from_predictions with binary labels {0, 2}."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row": [0, 1, 2, 3, 4],
                        "prediction": [0, 2, 2, 0, 2],
                        "target": [0, 2, 2, 0, 0],
                        0: [0.8, 0.3, 0.2, 0.9, 0.4],
                        2: [0.2, 0.7, 0.8, 0.1, 0.6],
                    }
                )
            }
        }

        performance = get_performance_from_predictions(
            predictions=predictions,
            target_metric="AUCROC",
            target_assignments={"default": "target"},
            positive_class=2,
        )

        assert isinstance(performance["training_0"]["dev"], float)
        assert 0 <= performance["training_0"]["dev"] <= 1

    def test_predictions_binary_labels_0_2_aucpr(self):
        """Test AUCPR with binary labels {0, 2} (was raising pos_label error before fix)."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row": [0, 1, 2, 3, 4],
                        "prediction": [0, 2, 2, 0, 2],
                        "target": [0, 2, 2, 0, 0],
                        0: [0.8, 0.3, 0.2, 0.9, 0.4],
                        2: [0.2, 0.7, 0.8, 0.1, 0.6],
                    }
                )
            }
        }

        performance = get_performance_from_predictions(
            predictions=predictions,
            target_metric="AUCPR",
            target_assignments={"default": "target"},
            positive_class=2,
        )

        assert isinstance(performance["training_0"]["dev"], float)
        assert 0 <= performance["training_0"]["dev"] <= 1

    def test_predictions_binary_labels_0_2_logloss(self):
        """Test LOGLOSS with binary labels {0, 2} (was inverted before fix)."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row": [0, 1, 2, 3, 4],
                        "prediction": [0, 2, 2, 0, 2],
                        "target": [0, 2, 2, 0, 0],
                        0: [0.8, 0.3, 0.2, 0.9, 0.4],
                        2: [0.2, 0.7, 0.8, 0.1, 0.6],
                    }
                )
            }
        }

        for pos_cls in (0, 2):
            performance = get_performance_from_predictions(
                predictions=predictions,
                target_metric="LOGLOSS",
                target_assignments={"default": "target"},
                positive_class=pos_cls,
            )

            assert isinstance(performance["training_0"]["dev"], float)
            assert performance["training_0"]["dev"] > 0


class TestMetricCalculateKwargs:
    """Test that Metric.calculate() passes runtime kwargs to the metric function."""

    def test_kwargs_passed_to_metric_function(self):
        """Verify that runtime kwargs override metric_params."""
        metric = Metrics.get_instance("AUCROC_MACRO")

        np.random.seed(42)
        y_true = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2])
        y_score = np.random.dirichlet([1, 1, 1], size=12)

        result_default = metric.calculate(y_true, y_score)
        result_override = metric.calculate(y_true, y_score, average="weighted")
        assert result_default != result_override

    def test_preconfigured_metric_params_still_work(self):
        """Pre-configured metric_params work when no runtime kwargs passed."""
        metric = Metrics.get_instance("AUCROC_MACRO")
        assert "multi_class" in metric.metric_params

        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_score = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.7, 0.2, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
            ]
        )

        result = metric.calculate(y_true, y_score)
        assert isinstance(result, float)
        assert 0 <= result <= 1
