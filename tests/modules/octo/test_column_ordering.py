"""Test suite for ColumnTransformer column ordering fix.

Verifies that when mixed column types (numerical + categorical) are present,
the ColumnTransformer output is correctly relabeled and reordered to match
self.feature_cols, preventing silent column name mismatch.

Usage:
    pytest tests/modules/octo/test_column_ordering.py -v
"""

import numpy as np
import pandas as pd
import pytest

from octopus.modules.octo.training import Training, TrainingConfig, fi_storage_key
from octopus.types import FIComputeMethod, MLType, ModelName
from tests.helpers import get_default_model_params


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
    model_name: ModelName = ModelName.ExtraTreesRegressor,
) -> Training:
    """Create a Training instance."""
    if ml_type == MLType.REGRESSION:
        target_assignments = {"default": "target_reg"}
        target_metric = "R2"
    else:
        target_assignments = {"default": "target_class"}
        target_metric = "AUCROC"

    ml_model_params = get_default_model_params(model_name)
    training_config: TrainingConfig = {
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


@pytest.mark.filterwarnings("ignore")
class TestColumnOrdering:
    """Tests for ColumnTransformer column ordering with mixed types."""

    def test_x_train_processed_columns_match_feature_cols(self):
        """Verify x_train_processed has columns in feature_cols order after fit."""
        data, feature_cols, feature_groups = _create_mixed_type_data()
        data_train, data_dev, data_test = _split_data(data)

        training = _create_training(data_train, data_dev, data_test, feature_cols, feature_groups)
        training.fit()

        assert training.x_train_processed is not None
        assert list(training.x_train_processed.columns) == feature_cols, (
            f"x_train_processed columns {list(training.x_train_processed.columns)} "
            f"do not match feature_cols {feature_cols}"
        )

    def test_x_dev_processed_columns_match_feature_cols(self):
        """Verify x_dev_processed has columns in feature_cols order after fit."""
        data, feature_cols, feature_groups = _create_mixed_type_data()
        data_train, data_dev, data_test = _split_data(data)

        training = _create_training(data_train, data_dev, data_test, feature_cols, feature_groups)
        training.fit()

        assert list(training.x_dev_processed.columns) == feature_cols

    def test_x_test_processed_columns_match_feature_cols(self):
        """Verify x_test_processed has columns in feature_cols order after fit."""
        data, feature_cols, feature_groups = _create_mixed_type_data()
        data_train, data_dev, data_test = _split_data(data)

        training = _create_training(data_train, data_dev, data_test, feature_cols, feature_groups)
        training.fit()

        assert list(training.x_test_processed.columns) == feature_cols

    def test_internal_fi_labels_correct_with_mixed_types(self):
        """Verify feature importance labels are correct when mixed column types exist.

        Target is strongly correlated with num1, so num1 should have highest importance.
        """
        data, feature_cols, feature_groups = _create_mixed_type_data(n_samples=500)
        data_train, data_dev, data_test = _split_data(data)

        training = _create_training(data_train, data_dev, data_test, feature_cols, feature_groups)
        training.fit()
        training.calculate_fi(FIComputeMethod.INTERNAL)

        fi_df = training.feature_importances[fi_storage_key(FIComputeMethod.INTERNAL)]
        assert not fi_df.empty, "Internal FI should not be empty"

        # num1 should be the most important feature (target = 10*num1 + 0.01*num2 + noise)
        top_feature = fi_df.loc[fi_df["importance"].idxmax(), "feature"]
        assert top_feature == "num1", (
            f"Expected num1 to be most important feature but got {top_feature}. "
            f"FI values:\n{fi_df.sort_values('importance', ascending=False)}"
        )

    def test_permutation_fi_labels_correct_with_mixed_types(self):
        """Verify permutation FI labels are correct when mixed column types exist."""
        data, feature_cols, feature_groups = _create_mixed_type_data(n_samples=500)
        data_train, data_dev, data_test = _split_data(data)

        training = _create_training(data_train, data_dev, data_test, feature_cols, feature_groups)
        training.fit()
        training.calculate_fi(FIComputeMethod.PERMUTATION, partition="dev", n_repeats=3)

        fi_df = training.feature_importances[fi_storage_key(FIComputeMethod.PERMUTATION, "dev")]
        assert not fi_df.empty, "Permutation FI should not be empty"

        # num1 (or its group numerical_group) should be the most important feature
        top_feature = fi_df.loc[fi_df["importance"].idxmax(), "feature"]
        assert top_feature in ("num1", "numerical_group"), (
            f"Expected num1 or numerical_group to be most important feature but got {top_feature}. "
            f"FI values:\n{fi_df.sort_values('importance', ascending=False)}"
        )

    def test_all_numerical_columns_no_regression(self):
        """Verify all-numerical columns still work correctly (regression test)."""
        data, feature_cols, feature_groups = _create_numerical_only_data()
        data_train, data_dev, data_test = _split_data(data)

        training = _create_training(data_train, data_dev, data_test, feature_cols, feature_groups)
        training.fit()

        assert training.x_train_processed is not None
        assert list(training.x_train_processed.columns) == feature_cols
        assert list(training.x_dev_processed.columns) == feature_cols
        assert list(training.x_test_processed.columns) == feature_cols

        # Predictions should work
        preds = training.predict(data_dev[feature_cols])
        assert len(preds) == len(data_dev)

    def test_relabel_fallback_when_get_feature_names_out_fails(self):
        """Verify fallback when get_feature_names_out() is not available."""
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

        training.preprocessing_pipeline = PipelineWithoutNames(original_pipeline)  # type: ignore[assignment]

        # _relabel_processed_output should fall back to feature_cols
        result = training._relabel_processed_output(processed_data)
        assert list(result.columns) == feature_cols

        # Restore pipeline
        training.preprocessing_pipeline = original_pipeline
