"""Comprehensive Model Fitted Validation Test Suite for Training Class.

This test suite validates the _validate_model_trained method across all available models:
- Tests all classification models
- Tests all regression models
- Tests time-to-event models (if available)
- Tests wrapper models
- Tests both fitted and unfitted model scenarios
- Uses sklearn's check_is_fitted utility for validation

Usage:
    pytest tests/modules/octo/test_model_fitted_validation.py -v
"""

import sys
import time
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

# ============================================================================
# CONFIGURATION AND FIXTURES
# ============================================================================

# Test data configuration
TEST_CONFIG = {
    "n_samples": 100,  # Smaller dataset for faster testing
    "test_split": 0.3,
    "dev_split": 0.2,
    "random_seed": 42,
}


class ModelCache:
    """Cache for available models to avoid repeated discovery."""

    def __init__(self):
        self._cached_models_by_type = None

    def get_available_models_by_type(self):
        """Get all available models dynamically from ModelInventory, grouped by ML type."""
        if self._cached_models_by_type is not None:
            return self._cached_models_by_type

        # Get all models from the registry
        all_models = Models._config_factories.keys()

        models_by_type = {"classification": [], "regression": [], "timetoevent": []}

        for model_name in all_models:
            try:
                model_config = Models.get_config(model_name)
                ml_type = model_config.ml_type
                if ml_type in models_by_type:
                    models_by_type[ml_type].append(model_name)
            except Exception as e:
                print(f"Warning: Could not get config for model {model_name}: {e}")
                continue

        self._cached_models_by_type = models_by_type
        return models_by_type


# Global instance for caching
_model_cache = ModelCache()


def get_available_models_by_type():
    """Get all available models dynamically from ModelInventory, grouped by ML type."""
    return _model_cache.get_available_models_by_type()


def get_model_configs():
    """Get model configurations with all available models."""
    available_models = get_available_models_by_type()

    return {
        "classification": {
            "models": available_models["classification"],
            "target_assignments": {"target": "target_class"},
            "target_metric": "accuracy",
        },
        "regression": {
            "models": available_models["regression"],
            "target_assignments": {"target": "target_reg"},
            "target_metric": "mse",
        },
        "timetoevent": {
            "models": available_models["timetoevent"],
            "target_assignments": {"duration": "duration", "event": "event"},
            "target_metric": "concordance_index",
        },
    }


def get_default_model_params(model_name: str) -> dict:
    """Get default parameters for a model from its hyperparameter configuration.

    This function should replicate the exact same parameter handling as the real octopus framework
    to ensure test results match real-world behavior.
    """
    # Models uses classmethods, no instantiation needed
    model_config = Models.get_config(model_name)

    params = {}

    # Extract fixed parameters and reasonable defaults for others
    # This should match exactly how the real framework handles parameters
    for hp in model_config.hyperparameters:
        if isinstance(hp, FixedHyperparameter):
            params[hp.name] = hp.value
        elif isinstance(hp, CategoricalHyperparameter):
            # Use first choice as default
            params[hp.name] = hp.choices[0] if hp.choices else None
        elif isinstance(hp, IntHyperparameter):
            # Use middle value as default
            params[hp.name] = int((hp.low + hp.high) / 2)
        elif isinstance(hp, FloatHyperparameter):
            # Use middle value as default (geometric mean for log scale)
            if hp.log:
                params[hp.name] = np.sqrt(hp.low * hp.high)
            else:
                params[hp.name] = (hp.low + hp.high) / 2
        else:
            raise AssertionError(f"Unsupported Hyperparameter type: {type(hp)}.")

    # Add standard parameters
    if model_config.n_jobs:
        params[model_config.n_jobs] = 1  # Single thread for testing
    if model_config.model_seed:
        params[model_config.model_seed] = 42  # Fixed seed for reproducibility

    # DO NOT modify parameters for any model - let the real configuration issues surface
    # This ensures the test detects the same issues as the real framework

    return params


@pytest.fixture(scope="session")
def test_data():
    """Create comprehensive test dataset with mixed data types."""
    np.random.seed(TEST_CONFIG["random_seed"])
    n_samples = TEST_CONFIG["n_samples"]

    # Generate numerical features
    data = pd.DataFrame(
        {
            "num_col1": np.random.normal(10, 2, n_samples),
            "num_col2": np.random.normal(50, 10, n_samples),
            "num_col3": np.random.uniform(0, 100, n_samples),
        }
    )

    # Add missing values to numerical columns
    data.loc[::15, "num_col1"] = np.nan
    data.loc[::20, "num_col2"] = np.nan
    data.loc[::25, "num_col3"] = np.nan

    # Generate categorical features
    nominal_col = np.random.choice([1, 2, 3, 4], n_samples).astype(float)
    nominal_col[::18] = np.nan
    data["nominal_col"] = nominal_col

    ordinal_col = np.random.choice([1, 2, 3], n_samples).astype(float)
    ordinal_col[::22] = np.nan
    data["ordinal_col"] = ordinal_col

    # Add row identifier
    data["row_id"] = range(n_samples)

    # Generate targets
    data["target_class"] = np.random.choice([0, 1], n_samples)
    data["target_reg"] = (
        0.5 * data["num_col1"].fillna(data["num_col1"].mean())
        + 0.3 * data["num_col2"].fillna(data["num_col2"].mean())
        + np.random.normal(0, 1, n_samples)
    )

    # Generate survival data
    data["duration"] = np.random.exponential(10, n_samples)
    data["event"] = np.random.choice([True, False], n_samples, p=[0.7, 0.3])

    # Split data
    n_train = int(n_samples * (1 - TEST_CONFIG["test_split"] - TEST_CONFIG["dev_split"]))
    n_dev = int(n_samples * TEST_CONFIG["dev_split"])

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    dev_idx = indices[n_train : n_train + n_dev]
    test_idx = indices[n_train + n_dev :]

    return (
        data.iloc[train_idx].reset_index(drop=True),
        data.iloc[dev_idx].reset_index(drop=True),
        data.iloc[test_idx].reset_index(drop=True),
    )


@pytest.fixture(scope="session")
def feature_config():
    """Feature configuration for tests."""
    feature_cols = ["num_col1", "num_col2", "num_col3", "nominal_col", "ordinal_col"]
    feature_groups = {
        "numerical_group": ["num_col1", "num_col2"],
        "categorical_group": ["nominal_col", "ordinal_col"],
    }
    return feature_cols, feature_groups


def create_training_instance(
    data_train: pd.DataFrame,
    data_dev: pd.DataFrame,
    data_test: pd.DataFrame,
    ml_type: str,
    model_name: str,
    feature_cols: list[str],
    feature_groups: dict[str, list[str]],
) -> Training:
    """Create a Training instance for testing."""
    model_configs = get_model_configs()
    config = model_configs[ml_type]

    # Get default parameters from model inventory
    ml_model_params = get_default_model_params(model_name)

    training_config = {
        "ml_model_type": model_name,
        "ml_model_params": ml_model_params,
        "outl_reduction": 0,
    }

    return Training(
        training_id=f"test_{ml_type}_{model_name}",
        ml_type=ml_type,
        target_assignments=config["target_assignments"],
        feature_cols=feature_cols,
        row_column="row_id",
        data_train=data_train,
        data_dev=data_dev,
        data_test=data_test,
        target_metric=config["target_metric"],
        max_features=0,  # Disable automatic feature selection
        feature_groups=feature_groups,
        config_training=training_config,
    )


# ============================================================================
# MODEL FITTED VALIDATION TESTING FUNCTIONS
# ============================================================================


def _test_model_fitted_validation(training: Training, model_name: str) -> dict:
    """Test the _validate_model_trained method for a specific model."""
    result = {
        "model": model_name,
        "fit_success": False,
        "fit_error": None,
        "fit_time": 0,
        "validation_success": False,
        "validation_error": None,
        "unfitted_detection_success": False,
        "unfitted_detection_error": None,
    }

    # Display model information dynamically (verbose only for standalone runs)
    _display_model_info(model_name, training.ml_model_params, verbose=_is_running_standalone())

    # Test 1: Try to validate an unfitted model (should fail)
    try:
        # Get the model instance but don't fit it
        # Models uses classmethods, no instantiation needed
        training.model = Models.get_instance(training.ml_model_type, training.ml_model_params)

        # This should raise RuntimeError
        training._validate_model_trained()
        result["unfitted_detection_error"] = "Unfitted model validation should have failed but didn't"

    except RuntimeError as e:
        if "model appears not to be fitted" in str(e):
            result["unfitted_detection_success"] = True
        else:
            result["unfitted_detection_error"] = f"Unexpected RuntimeError: {e}"
    except Exception as e:
        result["unfitted_detection_error"] = f"Unexpected exception: {type(e).__name__}: {e}"

    # Test 2: Fit the model and validate (should succeed)
    start_time = time.time()
    try:
        training.fit()  # This includes model fitting and validation
        result["fit_success"] = True
        result["validation_success"] = True
        result["fit_time"] = time.time() - start_time

    except RuntimeError as e:
        if "Model training failed" in str(e) or "model appears not to be fitted" in str(e):
            result["fit_error"] = str(e)
            result["validation_error"] = str(e)
        else:
            result["fit_error"] = f"Unexpected RuntimeError: {e}"
    except Exception as e:
        result["fit_error"] = f"Model fitting failed: {type(e).__name__}: {e}"
        result["fit_time"] = time.time() - start_time

    return result


def _display_model_info(model_name: str, model_params: dict, verbose: bool = False):
    """Display detailed model information and hyperparameters.

    Args:
        model_name: Name of the model
        model_params: Model parameters dictionary
        verbose: If True, display detailed information. If False, display minimal info.

    Returns:
        None: This function only prints information and does not return a value.

    Raises:
        AssertionError: If an unsupported Hyperparameter type is given
    """
    if not verbose:
        # Minimal output for pytest runs
        return

    try:
        # Models uses classmethods, no instantiation needed
        model_config = Models.get_config(model_name)

        print(f"\n  📊 {model_name}")
        print(f"     Model Class: {model_config.model_class.__name__}")
        print(f"     ML Type: {model_config.ml_type}")
        print(f"     Feature Method: {model_config.feature_method}")
        print(f"     Scaler: {model_config.scaler}")
        print(f"     Imputation Required: {model_config.imputation_required}")
        print(f"     Categorical Enabled: {model_config.categorical_enabled}")

        # Categorize hyperparameters
        fixed_params = []
        categorical_params = []
        numeric_params = []

        for hp in model_config.hyperparameters:
            if isinstance(hp, FixedHyperparameter):
                fixed_params.append(f"{hp.name}={hp.value}")
            elif isinstance(hp, CategoricalHyperparameter):
                categorical_params.append(f"{hp.name}={hp.choices[0]} (choices: {hp.choices})")
            elif isinstance(hp, IntHyperparameter | FloatHyperparameter):
                if isinstance(hp, IntHyperparameter):
                    default_val = int((hp.low + hp.high) / 2)
                else:
                    default_val = np.sqrt(hp.low * hp.high) if hp.log else (hp.low + hp.high) / 2
                numeric_params.append(f"{hp.name}={default_val:.4f} (range: {hp.low}-{hp.high}, log: {hp.log})")
            else:
                raise AssertionError(f"Unsupported Hyperparameter type: {type(hp)}.")

        # Add test overrides
        if model_config.n_jobs:
            fixed_params.append(f"{model_config.n_jobs}=1 (test override)")
        if model_config.model_seed:
            fixed_params.append(f"{model_config.model_seed}=42 (test override)")

        # Display parameters by category
        if fixed_params:
            print(f"     🔒 FIXED Parameters ({len(fixed_params)}):")
            for param in fixed_params:
                print(f"        • {param}")

        if categorical_params:
            print(f"     🎯 Categorical Parameters ({len(categorical_params)}):")
            for param in categorical_params:
                print(f"        • {param}")

        if numeric_params:
            print(f"     📈 Numeric Parameters ({len(numeric_params)}):")
            for param in numeric_params:
                print(f"        • {param}")

        print(f"     📋 Total Parameters Used: {len(model_params)}")

        # Show actual parameters being used
        print("     ⚙️  Actual Parameters:")
        for key, value in sorted(model_params.items()):
            print(f"        • {key}={value}")

    except Exception as e:
        print(f"     ❌ Error displaying model info: {e}")


def _is_running_standalone():
    """Check if the script is being run standalone (python script.py) vs pytest."""
    return __name__ == "__main__" or "pytest" not in sys.modules


# ============================================================================
# PYTEST TEST CLASSES
# ============================================================================


class TestModelFittedValidation:
    """Comprehensive test suite for model fitted validation."""

    def test_classification_models_fitted_validation(self, test_data, feature_config):
        """Test _validate_model_trained method for all classification models."""
        data_train, data_dev, data_test = test_data
        feature_cols, feature_groups = feature_config

        warnings.filterwarnings("ignore")

        model_configs = get_model_configs()
        classification_models = model_configs["classification"]["models"]

        failed_tests = []
        successful_tests = []

        for model_name in classification_models:
            try:
                training = create_training_instance(
                    data_train, data_dev, data_test, "classification", model_name, feature_cols, feature_groups
                )

                result = _test_model_fitted_validation(training, model_name)

                # Check if both unfitted detection and fitted validation work
                if not result["unfitted_detection_success"]:
                    failed_tests.append(
                        f"Classification/{model_name}/unfitted_detection: {result['unfitted_detection_error']}"
                    )
                else:
                    successful_tests.append(f"Classification/{model_name}/unfitted_detection")

                if not result["validation_success"]:
                    failed_tests.append(
                        f"Classification/{model_name}/fitted_validation: {result['validation_error'] or result['fit_error']}"
                    )
                else:
                    successful_tests.append(f"Classification/{model_name}/fitted_validation")

            except ImportError as e:
                # Skip models with missing dependencies
                print(f"Skipping {model_name} due to missing dependency: {e}")
                continue
            except Exception as e:
                failed_tests.append(f"Classification/{model_name}/setup: {e!s}")

        # Print summary
        print("\nClassification Models Summary:")
        print(f"  Successful tests: {len(successful_tests)}")
        print(f"  Failed tests: {len(failed_tests)}")

        if failed_tests:
            print("\nFailed tests:")
            for failure in failed_tests[:10]:  # Show first 10 failures
                print(f"  - {failure}")
            if len(failed_tests) > 10:
                print(f"  ... and {len(failed_tests) - 10} more")

        # Assert that ALL models should work - configuration issues should cause test failure
        assert len(failed_tests) == 0, "Classification model configuration/validation failures detected:\n" + "\n".join(
            f"  - {failure}" for failure in failed_tests
        )

    def test_regression_models_fitted_validation(self, test_data, feature_config):
        """Test _validate_model_trained method for all regression models."""
        data_train, data_dev, data_test = test_data
        feature_cols, feature_groups = feature_config

        warnings.filterwarnings("ignore")

        model_configs = get_model_configs()
        regression_models = model_configs["regression"]["models"]

        failed_tests = []
        successful_tests = []

        for model_name in regression_models:
            try:
                training = create_training_instance(
                    data_train, data_dev, data_test, "regression", model_name, feature_cols, feature_groups
                )

                result = _test_model_fitted_validation(training, model_name)

                # Check if both unfitted detection and fitted validation work
                if not result["unfitted_detection_success"]:
                    failed_tests.append(
                        f"Regression/{model_name}/unfitted_detection: {result['unfitted_detection_error']}"
                    )
                else:
                    successful_tests.append(f"Regression/{model_name}/unfitted_detection")

                if not result["validation_success"]:
                    failed_tests.append(
                        f"Regression/{model_name}/fitted_validation: {result['validation_error'] or result['fit_error']}"
                    )
                else:
                    successful_tests.append(f"Regression/{model_name}/fitted_validation")

            except ImportError as e:
                # Skip models with missing dependencies
                print(f"Skipping {model_name} due to missing dependency: {e}")
                continue
            except Exception as e:
                failed_tests.append(f"Regression/{model_name}/setup: {e!s}")

        # Print summary
        print("\nRegression Models Summary:")
        print(f"  Successful tests: {len(successful_tests)}")
        print(f"  Failed tests: {len(failed_tests)}")

        if failed_tests:
            print("\nFailed tests:")
            for failure in failed_tests[:10]:  # Show first 10 failures
                print(f"  - {failure}")
            if len(failed_tests) > 10:
                print(f"  ... and {len(failed_tests) - 10} more")

        # Assert that we have more successes than failures
        assert len(successful_tests) > len(failed_tests), (
            f"Too many regression model validation failures: {len(failed_tests)} failed vs {len(successful_tests)} successful"
        )

    def test_timetoevent_models_fitted_validation(self, test_data, feature_config):
        """Test _validate_model_trained method for all time-to-event models."""
        data_train, data_dev, data_test = test_data
        feature_cols, feature_groups = feature_config

        warnings.filterwarnings("ignore")

        model_configs = get_model_configs()
        timetoevent_models = model_configs["timetoevent"]["models"]

        if not timetoevent_models:
            pytest.skip("No time-to-event models available")

        failed_tests = []
        successful_tests = []

        for model_name in timetoevent_models:
            try:
                training = create_training_instance(
                    data_train, data_dev, data_test, "timetoevent", model_name, feature_cols, feature_groups
                )

                result = _test_model_fitted_validation(training, model_name)

                # Check if both unfitted detection and fitted validation work
                if not result["unfitted_detection_success"]:
                    failed_tests.append(
                        f"TimeToEvent/{model_name}/unfitted_detection: {result['unfitted_detection_error']}"
                    )
                else:
                    successful_tests.append(f"TimeToEvent/{model_name}/unfitted_detection")

                if not result["validation_success"]:
                    failed_tests.append(
                        f"TimeToEvent/{model_name}/fitted_validation: {result['validation_error'] or result['fit_error']}"
                    )
                else:
                    successful_tests.append(f"TimeToEvent/{model_name}/fitted_validation")

            except ImportError as e:
                # Skip models with missing dependencies
                print(f"Skipping {model_name} due to missing dependency: {e}")
                continue
            except Exception as e:
                failed_tests.append(f"TimeToEvent/{model_name}/setup: {e!s}")

        # Print summary
        print("\nTime-to-Event Models Summary:")
        print(f"  Successful tests: {len(successful_tests)}")
        print(f"  Failed tests: {len(failed_tests)}")

        if failed_tests:
            print("\nFailed tests:")
            for failure in failed_tests:
                print(f"  - {failure}")

        # For time-to-event models, we're more lenient due to optional dependencies
        if successful_tests:
            assert len(successful_tests) >= len(failed_tests), (
                f"Time-to-event model validation failures: {len(failed_tests)} failed vs {len(successful_tests)} successful"
            )

    def test_comprehensive_model_fitted_validation(self, test_data, feature_config):
        """Comprehensive test across all model types."""
        data_train, data_dev, data_test = test_data
        feature_cols, feature_groups = feature_config

        warnings.filterwarnings("ignore")

        model_configs = get_model_configs()

        total_tests = 0
        successful_tests = 0
        failed_tests = []

        for ml_type, config in model_configs.items():
            for model_name in config["models"]:
                try:
                    training = create_training_instance(
                        data_train, data_dev, data_test, ml_type, model_name, feature_cols, feature_groups
                    )

                    result = _test_model_fitted_validation(training, model_name)

                    # Count tests
                    total_tests += 2  # unfitted detection + fitted validation

                    if result["unfitted_detection_success"]:
                        successful_tests += 1
                    else:
                        failed_tests.append(
                            f"{ml_type}/{model_name}/unfitted_detection: {result['unfitted_detection_error']}"
                        )

                    if result["validation_success"]:
                        successful_tests += 1
                    else:
                        failed_tests.append(
                            f"{ml_type}/{model_name}/fitted_validation: {result['validation_error'] or result['fit_error']}"
                        )

                except ImportError as e:
                    # Skip models with missing dependencies
                    print(f"Skipping {model_name} due to missing dependency: {e}")
                    continue
                except Exception as e:
                    total_tests += 2
                    failed_tests.append(f"{ml_type}/{model_name}/setup: {e!s}")

        # Print comprehensive summary
        print("\nComprehensive Model Fitted Validation Summary:")
        print(f"  Total tests: {total_tests}")
        print(f"  Successful tests: {successful_tests}")
        print(f"  Failed tests: {len(failed_tests)}")
        print(f"  Success rate: {successful_tests / total_tests * 100:.1f}%")

        # Assert that ALL models should work - configuration issues should cause test failure
        assert len(failed_tests) == 0, "Model configuration/validation failures detected:\n" + "\n".join(
            f"  - {failure}" for failure in failed_tests
        )


# ============================================================================
# MAIN EXECUTION FOR STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    # Run standalone tests
    print("=" * 80)
    print("COMPREHENSIVE MODEL FITTED VALIDATION TEST SUITE")
    print("=" * 80)

    # Create test data manually for standalone execution
    np.random.seed(TEST_CONFIG["random_seed"])
    n_samples = TEST_CONFIG["n_samples"]

    data = pd.DataFrame(
        {
            "num_col1": np.random.normal(10, 2, n_samples),
            "num_col2": np.random.normal(50, 10, n_samples),
            "num_col3": np.random.uniform(0, 100, n_samples),
        }
    )

    # Add missing values
    data.loc[::15, "num_col1"] = np.nan
    data.loc[::20, "num_col2"] = np.nan
    data.loc[::25, "num_col3"] = np.nan

    # Generate categorical features
    nominal_col = np.random.choice([1, 2, 3, 4], n_samples).astype(float)
    nominal_col[::18] = np.nan
    data["nominal_col"] = nominal_col

    ordinal_col = np.random.choice([1, 2, 3], n_samples).astype(float)
    ordinal_col[::22] = np.nan
    data["ordinal_col"] = ordinal_col

    data["row_id"] = range(n_samples)
    data["target_class"] = np.random.choice([0, 1], n_samples)
    data["target_reg"] = (
        0.5 * data["num_col1"].fillna(data["num_col1"].mean())
        + 0.3 * data["num_col2"].fillna(data["num_col2"].mean())
        + np.random.normal(0, 1, n_samples)
    )
    data["duration"] = np.random.exponential(10, n_samples)
    data["event"] = np.random.choice([True, False], n_samples, p=[0.7, 0.3])

    # Split data
    n_train = int(n_samples * (1 - TEST_CONFIG["test_split"] - TEST_CONFIG["dev_split"]))
    n_dev = int(n_samples * TEST_CONFIG["dev_split"])

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    dev_idx = indices[n_train : n_train + n_dev]
    test_idx = indices[n_train + n_dev :]

    test_data = (
        data.iloc[train_idx].reset_index(drop=True),
        data.iloc[dev_idx].reset_index(drop=True),
        data.iloc[test_idx].reset_index(drop=True),
    )

    feature_config = (
        ["num_col1", "num_col2", "num_col3", "nominal_col", "ordinal_col"],
        {"numerical_group": ["num_col1", "num_col2"], "categorical_group": ["nominal_col", "ordinal_col"]},
    )

    # Run tests
    test_instance = TestModelFittedValidation()

    try:
        test_instance.test_classification_models_fitted_validation(test_data, feature_config)
        print("✅ Classification models test passed")
    except Exception as e:
        print(f"❌ Classification models test failed: {e}")

    try:
        test_instance.test_regression_models_fitted_validation(test_data, feature_config)
        print("✅ Regression models test passed")
    except Exception as e:
        print(f"❌ Regression models test failed: {e}")

    try:
        test_instance.test_timetoevent_models_fitted_validation(test_data, feature_config)
        print("✅ Time-to-event models test passed")
    except Exception as e:
        print(f"❌ Time-to-event models test failed: {e}")

    try:
        test_instance.test_comprehensive_model_fitted_validation(test_data, feature_config)
        print("✅ Comprehensive test passed")
    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")

    print("\n" + "=" * 80)
    print("Model fitted validation testing completed!")
