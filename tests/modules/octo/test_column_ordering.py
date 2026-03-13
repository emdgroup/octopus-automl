"""Test suite for ColumnTransformer column ordering fix.

Verifies that when mixed column types (numerical + categorical) are present,
the ColumnTransformer output is correctly relabeled and reordered to match
self.feature_cols, preventing silent column name mismatch.

Usage:
    pytest tests/modules/octo/test_column_ordering.py -v
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from octopus.models import Models
from octopus.models.hyperparameter import (
    CategoricalHyperparameter,
    FixedHyperparameter,
    FloatHyperparameter,
    IntHyperparameter,
)
from octopus.modules.octo.training import Training
from octopus.types import MLType


def _get_default_model_params(model_name: str) -> dict:
    """Get default parameters for a model from its hyperparameter configuration."""
    model_config = Models.get_config(model_name)
    params = {}

    for hp in model_config.hyperparameters:
        if isinstance(hp, FixedHyperparameter):
            params[hp.name] = hp.value
        elif isinstance(hp, CategoricalHyperparameter):
            params[hp.name] = hp.choices[0] if hp.choices else None
        elif isinstance(hp, IntHyperparameter):
            params[hp.name] = int((hp.low + hp.high) / 2)
        elif isinstance(hp, FloatHyperparameter):
            if hp.log:
                params[hp.name] = np.sqrt(hp.low * hp.high)
            else:
                params[hp.name] = (hp.low + hp.high) / 2
        else:
            raise AssertionError(f"Unsupported Hyperparameter type: {type(hp)}.")

    if model_config.n_jobs:
        params[model_config.n_jobs] = 1
    if model_config.model_seed:
        params[model_config.model_seed] = 42

    return params


def _create_mixed_type_data(n_samples: int = 60) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Create test data with interleaved categorical and numerical columns.

    Returns data where feature_cols order is ['cat1', 'num1', 'num2', 'cat2']
    to trigger the ColumnTransformer reordering bug.
    """
    np.random.seed(42)

    # Use integer-coded categoricals with pd.CategoricalDtype.
    # This triggers ColumnTransformer reordering (dtype is "category") while keeping
    # values numeric so sklearn models can process them.
    # Use values close together (0/1) so they don't dominate tree splits over num1.
    cat1_values = np.random.choice([0, 1, 2], n_samples)
    cat2_values = np.random.choice([0, 1], n_samples)

    data = pd.DataFrame(
        {
            "cat1": pd.Categorical(cat1_values),
            "num1": np.random.normal(10, 2, n_samples),
            "num2": np.random.normal(50, 10, n_samples),
            "cat2": pd.Categorical(cat2_values),
            "row_id": range(n_samples),
            "target_class": np.random.choice([0, 1], n_samples),
            "target_reg": np.random.normal(0, 1, n_samples),
        }
    )

    # Make target strongly correlated with num1 so we can verify FI labels
    data["target_reg"] = 10.0 * data["num1"] + 0.01 * data["num2"] + np.random.normal(0, 0.1, n_samples)

    feature_cols = ["cat1", "num1", "num2", "cat2"]
    feature_groups = {
        "numerical_group": ["num1", "num2"],
        "categorical_group": ["cat1", "cat2"],
    }
    return data, feature_cols, feature_groups


def _create_numerical_only_data(n_samples: int = 60) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Create test data with only numerical columns (no reordering expected)."""
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "num1": np.random.normal(10, 2, n_samples),
            "num2": np.random.normal(50, 10, n_samples),
            "num3": np.random.uniform(0, 100, n_samples),
            "row_id": range(n_samples),
            "target_class": np.random.choice([0, 1], n_samples),
            "target_reg": np.random.normal(0, 1, n_samples),
        }
    )
    data["target_reg"] = 3.0 * data["num1"] + 0.1 * data["num2"] + np.random.normal(0, 0.5, n_samples)

    feature_cols = ["num1", "num2", "num3"]
    feature_groups = {"numerical_group": ["num1", "num2"]}
    return data, feature_cols, feature_groups


def _split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, dev, test."""
    n = len(data)
    n_train = int(n * 0.5)
    n_dev = int(n * 0.25)

    indices = np.random.RandomState(42).permutation(n)
    train_idx = indices[:n_train]
    dev_idx = indices[n_train : n_train + n_dev]
    test_idx = indices[n_train + n_dev :]

    return (
        data.iloc[train_idx].reset_index(drop=True),
        data.iloc[dev_idx].reset_index(drop=True),
        data.iloc[test_idx].reset_index(drop=True),
    )


def _create_training(
    data_train: pd.DataFrame,
    data_dev: pd.DataFrame,
    data_test: pd.DataFrame,
    feature_cols: list[str],
    feature_groups: dict[str, list[str]],
    ml_type: MLType = MLType.REGRESSION,
    model_name: str = "ExtraTreesRegressor",
) -> Training:
    """Create a Training instance."""
    if ml_type == MLType.REGRESSION:
        target_assignments = {"target": "target_reg"}
        target_metric = "R2"
    else:
        target_assignments = {"target": "target_class"}
        target_metric = "AUCROC"

    ml_model_params = _get_default_model_params(model_name)
    training_config = {
        "ml_model_type": model_name,
        "ml_model_params": ml_model_params,
        "outl_reduction": 0,
    }
    if ml_type == MLType.BINARY:
        training_config["positive_class"] = 1

    return Training(
        training_id="test_0_0_0",
        ml_type=ml_type,
        target_assignments=target_assignments,
        feature_cols=feature_cols,
        row_id_col="row_id",
        data_train=data_train,
        data_dev=data_dev,
        data_test=data_test,
        target_metric=target_metric,
        max_features=0,
        feature_groups=feature_groups,
        config_training=training_config,
    )


class TestColumnOrdering:
    """Tests for ColumnTransformer column ordering with mixed types."""

    def test_x_train_processed_columns_match_feature_cols(self):
        """Verify x_train_processed has columns in feature_cols order after fit."""
        warnings.filterwarnings("ignore")
        data, feature_cols, feature_groups = _create_mixed_type_data()
        data_train, data_dev, data_test = _split_data(data)

        training = _create_training(data_train, data_dev, data_test, feature_cols, feature_groups)
        training.fit()

        assert list(training.x_train_processed.columns) == feature_cols, (
            f"x_train_processed columns {list(training.x_train_processed.columns)} "
            f"do not match feature_cols {feature_cols}"
        )

    def test_x_dev_processed_columns_match_feature_cols(self):
        """Verify x_dev_processed has columns in feature_cols order after fit."""
        warnings.filterwarnings("ignore")
        data, feature_cols, feature_groups = _create_mixed_type_data()
        data_train, data_dev, data_test = _split_data(data)

        training = _create_training(data_train, data_dev, data_test, feature_cols, feature_groups)
        training.fit()

        assert list(training.x_dev_processed.columns) == feature_cols

    def test_x_test_processed_columns_match_feature_cols(self):
        """Verify x_test_processed has columns in feature_cols order after fit."""
        warnings.filterwarnings("ignore")
        data, feature_cols, feature_groups = _create_mixed_type_data()
        data_train, data_dev, data_test = _split_data(data)

        training = _create_training(data_train, data_dev, data_test, feature_cols, feature_groups)
        training.fit()

        assert list(training.x_test_processed.columns) == feature_cols

    def test_numerical_data_in_numerical_column(self):
        """Verify that numerical columns in x_train_processed contain actual numerical data."""
        warnings.filterwarnings("ignore")
        data, feature_cols, feature_groups = _create_mixed_type_data()
        data_train, data_dev, data_test = _split_data(data)

        training = _create_training(data_train, data_dev, data_test, feature_cols, feature_groups)
        training.fit()

        # num1 should contain scaled numerical data (floats), not categorical codes
        num1_values = training.x_train_processed["num1"].values
        assert np.issubdtype(num1_values.dtype, np.floating), (
            f"num1 should be float dtype but got {num1_values.dtype}"
        )
        # The values should be scaled (StandardScaler) from the original ~N(10,2) distribution
        # They should NOT be categorical string values
        assert not any(isinstance(v, str) for v in num1_values), "num1 contains string values — column mislabeled!"

    def test_categorical_data_in_categorical_column(self):
        """Verify that categorical columns in x_train_processed contain actual categorical data."""
        warnings.filterwarnings("ignore")
        data, feature_cols, feature_groups = _create_mixed_type_data()
        data_train, data_dev, data_test = _split_data(data)

        training = _create_training(data_train, data_dev, data_test, feature_cols, feature_groups)
        training.fit()

        # cat1 should contain categorical integer values (0, 1, 2), not scaled numerics
        cat1_values = training.x_train_processed["cat1"].values.astype(float)
        unique_vals = set(cat1_values)
        # Original cat1 values are {0, 1, 2} — they should NOT be StandardScaler-transformed
        # (categorical columns only get imputation, not scaling)
        assert unique_vals.issubset({0.0, 1.0, 2.0}), (
            f"cat1 should contain only {{0, 1, 2}} but got {unique_vals}"
        )

    def test_internal_fi_labels_correct_with_mixed_types(self):
        """Verify feature importance labels are correct when mixed column types exist.

        Target is strongly correlated with num1, so num1 should have highest importance.
        """
        warnings.filterwarnings("ignore")
        data, feature_cols, feature_groups = _create_mixed_type_data(n_samples=500)
        data_train, data_dev, data_test = _split_data(data)

        training = _create_training(data_train, data_dev, data_test, feature_cols, feature_groups)
        training.fit()
        training.calculate_fi_internal()

        fi_df = training.feature_importances["internal"]
        assert not fi_df.empty, "Internal FI should not be empty"

        # num1 should be the most important feature (target = 10*num1 + 0.01*num2 + noise)
        top_feature = fi_df.loc[fi_df["importance"].idxmax(), "feature"]
        assert top_feature == "num1", (
            f"Expected num1 to be most important feature but got {top_feature}. "
            f"FI values:\n{fi_df.sort_values('importance', ascending=False)}"
        )

    def test_permutation_fi_labels_correct_with_mixed_types(self):
        """Verify permutation FI labels are correct when mixed column types exist."""
        warnings.filterwarnings("ignore")
        data, feature_cols, feature_groups = _create_mixed_type_data(n_samples=500)
        data_train, data_dev, data_test = _split_data(data)

        training = _create_training(data_train, data_dev, data_test, feature_cols, feature_groups)
        training.fit()
        training.calculate_fi_permutation(partition="dev", n_repeats=3)

        fi_df = training.feature_importances["permutation_dev"]
        assert not fi_df.empty, "Permutation FI should not be empty"

        # num1 should be the most important feature
        top_feature = fi_df.loc[fi_df["importance"].idxmax(), "feature"]
        assert top_feature == "num1", (
            f"Expected num1 to be most important feature but got {top_feature}. "
            f"FI values:\n{fi_df.sort_values('importance', ascending=False)}"
        )

    def test_all_numerical_columns_no_regression(self):
        """Verify all-numerical columns still work correctly (regression test)."""
        warnings.filterwarnings("ignore")
        data, feature_cols, feature_groups = _create_numerical_only_data()
        data_train, data_dev, data_test = _split_data(data)

        training = _create_training(data_train, data_dev, data_test, feature_cols, feature_groups)
        training.fit()

        assert list(training.x_train_processed.columns) == feature_cols
        assert list(training.x_dev_processed.columns) == feature_cols
        assert list(training.x_test_processed.columns) == feature_cols

        # Predictions should work
        preds = training.predict(data_dev[feature_cols])
        assert len(preds) == len(data_dev)

    def test_predict_works_with_mixed_types(self):
        """Verify predict() works correctly with mixed column types."""
        warnings.filterwarnings("ignore")
        data, feature_cols, feature_groups = _create_mixed_type_data()
        data_train, data_dev, data_test = _split_data(data)

        training = _create_training(data_train, data_dev, data_test, feature_cols, feature_groups)
        training.fit()

        preds = training.predict(data_dev[feature_cols])
        assert len(preds) == len(data_dev)
        assert np.all(np.isfinite(preds)), "Predictions should be finite"

    def test_predict_classification_with_mixed_types(self):
        """Verify predict_proba() works correctly with mixed column types for classification."""
        warnings.filterwarnings("ignore")
        data, feature_cols, feature_groups = _create_mixed_type_data()
        data_train, data_dev, data_test = _split_data(data)

        training = _create_training(
            data_train, data_dev, data_test, feature_cols, feature_groups,
            ml_type=MLType.BINARY, model_name="ExtraTreesClassifier",
        )
        training.fit()

        preds = training.predict(data_dev[feature_cols])
        assert len(preds) == len(data_dev)

        proba = training.predict_proba(data_dev[feature_cols])
        assert proba.shape == (len(data_dev), 2)
        assert np.allclose(proba.sum(axis=1), 1.0), "Probabilities should sum to 1"

    def test_relabel_fallback_when_get_feature_names_out_fails(self):
        """Verify fallback when get_feature_names_out() is not available."""
        warnings.filterwarnings("ignore")
        data, feature_cols, feature_groups = _create_numerical_only_data()
        data_train, data_dev, data_test = _split_data(data)

        training = _create_training(data_train, data_dev, data_test, feature_cols, feature_groups)
        training.fit()

        # Simulate a pipeline without get_feature_names_out
        original_pipeline = training.preprocessing_pipeline
        processed_data = original_pipeline.transform(data_dev[feature_cols])

        # Delete get_feature_names_out to trigger fallback
        class PipelineWithoutNames:
            """Mock pipeline without get_feature_names_out."""
            def __init__(self, pipeline):
                self._pipeline = pipeline
            def transform(self, data):
                return self._pipeline.transform(data)
            def fit_transform(self, data):
                return self._pipeline.fit_transform(data)

        training.preprocessing_pipeline = PipelineWithoutNames(original_pipeline)

        # _relabel_processed_output should fall back to feature_cols
        result = training._relabel_processed_output(processed_data)
        assert list(result.columns) == feature_cols

        # Restore pipeline
        training.preprocessing_pipeline = original_pipeline