"""Comprehensive Feature Importance Test Suite for Training Class.

This test suite validates all feature importance methods across all available models:
- calculate_fi_internal
- calculate_fi_group_permutation
- calculate_fi_permutation
- calculate_fi_lofo
- calculate_fi_featuresused_shap
- calculate_fi_shap

Usage:
    python test_feature_importance_comprehensive.py
    pytest test_feature_importance_comprehensive.py -v
"""

import sys
import time
import traceback
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import pytest
except ImportError:
    pytest = None

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
    "n_samples": 30,  # Even smaller dataset for faster testing
    "test_split": 0.3,
    "dev_split": 0.2,
    "random_seed": 42,
}

# Feature importance methods to test
FI_METHODS = [
    "calculate_fi_internal",
    "calculate_fi_group_permutation",
    "calculate_fi_permutation",
    "calculate_fi_lofo",
    "calculate_fi_featuresused_shap",
    "calculate_fi_shap",
]


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

        models_by_type = {"classification": [], "regression": [], "timetoevent": [], "multiclass": []}

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
            "target_metric": "AUCROC",
        },
        "regression": {
            "models": available_models["regression"],
            "target_assignments": {"target": "target_reg"},
            "target_metric": "R2",
        },
        "timetoevent": {
            "models": available_models["timetoevent"],
            "target_assignments": {"duration": "duration", "event": "event"},
            "target_metric": "CI",
        },
        "multiclass": {
            "models": available_models["multiclass"],
            "target_assignments": {"target": "target_multiclass"},
            "target_metric": "ACCBAL_MC",
        },
    }


def get_default_model_params(model_name: str) -> dict:
    """Get default parameters for a model from its hyperparameter configuration."""
    # Models uses classmethods, no instantiation needed
    model_config = Models.get_config(model_name)

    params = {}

    # Extract fixed parameters and reasonable defaults for others
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

    # Generate fewer numerical features for speed
    data = pd.DataFrame(
        {
            "num_col1": np.random.normal(10, 2, n_samples),
            "num_col2": np.random.normal(50, 10, n_samples),
        }
    )

    # Minimal missing values for speed
    data.loc[::10, "num_col1"] = np.nan

    # Single categorical feature for speed
    nominal_col = np.random.choice([1, 2, 3], n_samples).astype(float)
    nominal_col[::15] = np.nan
    data["nominal_col"] = nominal_col

    # Add row identifier
    data["row_id"] = range(n_samples)

    # Generate targets
    data["target_class"] = np.random.choice([0, 1], n_samples)
    data["target_multiclass"] = np.random.choice([0, 1, 2], n_samples)
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

    return (
        data.iloc[train_idx].reset_index(drop=True),
        data.iloc[dev_idx].reset_index(drop=True),
        data.iloc[test_idx].reset_index(drop=True),
    )


@pytest.fixture(scope="session")
def feature_config():
    """Feature configuration for tests - optimized for speed."""
    feature_cols = ["num_col1", "num_col2", "nominal_col"]  # Reduced to 3 features
    feature_groups = {
        "numerical_group": ["num_col1", "num_col2"],
        "categorical_group": ["nominal_col"],
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

    # Add positive_class for classification tasks
    if ml_type == "classification":
        training_config["positive_class"] = 1

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
# FEATURE IMPORTANCE TESTING FUNCTIONS
# ============================================================================


def _test_fi_method(training: Training, method_name: str) -> dict:
    """Test a specific feature importance method."""
    result = {
        "method": method_name,
        "success": False,
        "error": None,
        "execution_time": 0,
        "fi_count": 0,
        "fi_keys": [],
    }

    start_time = time.time()

    try:
        if method_name == "calculate_fi_internal":
            training.calculate_fi_internal()
            fi_key = "internal"

        elif method_name == "calculate_fi_group_permutation":
            training.calculate_fi_group_permutation(partition="dev", n_repeats=1)  # Reduced repeats
            fi_key = "permutation_dev"

        elif method_name == "calculate_fi_permutation":
            training.calculate_fi_permutation(partition="dev", n_repeats=1)  # Reduced repeats
            fi_key = "permutation_dev"

        elif method_name == "calculate_fi_lofo":
            training.calculate_fi_lofo()
            fi_key = ["lofo_dev", "lofo_test"]

        elif method_name == "calculate_fi_featuresused_shap":
            training.calculate_fi_featuresused_shap(partition="dev")
            fi_key = "shap_dev"

        elif method_name == "calculate_fi_shap":
            training.calculate_fi_shap(partition="dev", shap_type="permutation")  # Faster SHAP method
            fi_key = "shap_dev"

        else:
            raise ValueError(f"Unknown method: {method_name}")

        # Check results
        if isinstance(fi_key, list):
            result["fi_keys"] = fi_key
            result["fi_count"] = sum(len(training.feature_importances.get(key, [])) for key in fi_key)
        else:
            result["fi_keys"] = [fi_key]
            fi_data = training.feature_importances.get(fi_key)
            result["fi_count"] = len(fi_data) if fi_data is not None else 0

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    result["execution_time"] = time.time() - start_time
    return result


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================


def run_comprehensive_feature_importance_tests():
    """Run comprehensive feature importance tests across all models and methods."""
    print("=" * 80)
    print("COMPREHENSIVE FEATURE IMPORTANCE TEST SUITE")
    print("=" * 80)

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Create test data
    np.random.seed(TEST_CONFIG["random_seed"])
    n_samples = TEST_CONFIG["n_samples"]

    # Generate test data (optimized for speed)
    data = pd.DataFrame(
        {
            "num_col1": np.random.normal(10, 2, n_samples),
            "num_col2": np.random.normal(50, 10, n_samples),
        }
    )

    # Minimal missing values
    data.loc[::10, "num_col1"] = np.nan

    # Single categorical feature
    nominal_col = np.random.choice([1, 2, 3], n_samples).astype(float)
    nominal_col[::15] = np.nan
    data["nominal_col"] = nominal_col

    data["row_id"] = range(n_samples)

    # Generate targets
    data["target_class"] = np.random.choice([0, 1], n_samples)
    data["target_multiclass"] = np.random.choice([0, 1, 2], n_samples)
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

    data_train = data.iloc[train_idx].reset_index(drop=True)
    data_dev = data.iloc[dev_idx].reset_index(drop=True)
    data_test = data.iloc[test_idx].reset_index(drop=True)

    feature_cols = ["num_col1", "num_col2", "nominal_col"]  # Reduced to 3 features
    feature_groups = {
        "numerical_group": ["num_col1", "num_col2"],
        "categorical_group": ["nominal_col"],
    }

    # Get all available models
    model_configs = get_model_configs()

    # Results storage
    results = defaultdict(lambda: defaultdict(dict))
    summary_stats = defaultdict(lambda: defaultdict(int))

    total_tests = 0
    successful_tests = 0

    print(f"Testing {len(FI_METHODS)} feature importance methods across all available models...")
    print()

    # Test each ML type
    for ml_type, config in model_configs.items():
        print(f"Testing {ml_type.upper()} models:")
        print("-" * 40)

        for model_name in config["models"]:
            print(f"  Model: {model_name}")

            try:
                # Create and fit training instance
                training = create_training_instance(
                    data_train, data_dev, data_test, ml_type, model_name, feature_cols, feature_groups
                )

                # Fit the model
                fit_start = time.time()
                training.fit()
                fit_time = time.time() - fit_start

                print(f"    Model fitted in {fit_time:.2f}s")

                # Test each feature importance method
                for method_name in FI_METHODS:
                    total_tests += 1
                    print(f"    Testing {method_name}...", end=" ")

                    result = _test_fi_method(training, method_name)
                    results[ml_type][model_name][method_name] = result

                    if result["success"]:
                        successful_tests += 1
                        summary_stats[ml_type][method_name] += 1
                        print(f"✓ ({result['execution_time']:.2f}s, {result['fi_count']} features)")
                    else:
                        print(f"✗ Error: {result['error']}")

            except Exception as e:
                print(f"    ✗ Model fitting failed: {e!s}")
                for method_name in FI_METHODS:
                    total_tests += 1
                    results[ml_type][model_name][method_name] = {
                        "method": method_name,
                        "success": False,
                        "error": f"Model fitting failed: {e!s}",
                        "execution_time": 0,
                        "fi_count": 0,
                        "fi_keys": [],
                    }

        print()

    # Print summary report
    print("=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    print(f"Total tests: {total_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Success rate: {successful_tests / total_tests * 100:.1f}%")
    print()

    # Method-wise summary
    print("Method Success Rates:")
    print("-" * 40)
    for method_name in FI_METHODS:
        method_success = sum(
            sum(1 for model_results in ml_results.values() if model_results.get(method_name, {}).get("success", False))
            for ml_results in results.values()
        )
        method_total = sum(len(ml_results) for ml_results in results.values())
        success_rate = method_success / method_total * 100 if method_total > 0 else 0
        print(f"  {method_name}: {method_success}/{method_total} ({success_rate:.1f}%)")

    print()

    # ML type summary
    print("ML Type Summary:")
    print("-" * 40)
    for ml_type in results:
        ml_success = sum(
            sum(1 for method_result in model_results.values() if method_result.get("success", False))
            for model_results in results[ml_type].values()
        )
        ml_total = sum(len(model_results) for model_results in results[ml_type].values())
        success_rate = ml_success / ml_total * 100 if ml_total > 0 else 0
        print(f"  {ml_type}: {ml_success}/{ml_total} ({success_rate:.1f}%)")

    print()

    # Detailed error analysis
    print("Error Analysis:")
    print("-" * 40)
    error_counts = defaultdict(int)
    for ml_results in results.values():
        for model_results in ml_results.values():
            for method_result in model_results.values():
                if not method_result["success"] and method_result["error"]:
                    # Simplify error message for counting
                    error_type = method_result["error"].split(":")[0]
                    error_counts[error_type] += 1

    for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {error_type}: {count} occurrences")

    print()

    # Detailed model-by-model summary
    print("Detailed Model Summary:")
    print("-" * 80)
    for ml_type, ml_results in results.items():
        print(f"\n{ml_type.upper()} Models:")
        print("-" * 40)

        for model_name, model_results in ml_results.items():
            # Count successes and failures for this model
            successes = sum(1 for result in model_results.values() if result.get("success", False))
            total = len(model_results)
            success_rate = successes / total * 100 if total > 0 else 0

            print(f"\n  {model_name}: {successes}/{total} ({success_rate:.1f}%)")

            # Show detailed results for each method
            for method_name, result in model_results.items():
                status = "✓" if result.get("success", False) else "✗"
                if result.get("success", False):
                    exec_time = result.get("execution_time", 0)
                    fi_count = result.get("fi_count", 0)
                    print(f"    {status} {method_name}: {exec_time:.2f}s, {fi_count} features")
                else:
                    error = result.get("error", "Unknown error")
                    # Truncate long error messages
                    if len(error) > 60:
                        error = error[:57] + "..."
                    print(f"    {status} {method_name}: {error}")

    print()
    print("=" * 80)

    # Final assessment summary
    failed_tests = []
    for ml_results in results.values():
        for model_name, model_results in ml_results.items():
            for method_name, method_result in model_results.items():
                if not method_result.get("success", False):
                    failed_tests.append(f"{model_name}/{method_name}: {method_result.get('error', 'Unknown error')}")

    print("FINAL ASSESSMENT:")
    print("=" * 80)
    if len(failed_tests) == 0:
        print("✅ ALL TESTS PASSED - No feature importance test failures detected")
        print(f"   Successfully tested {successful_tests}/{total_tests} feature importance methods")
    else:
        print("❌ TESTS FAILED - Feature importance test failures detected")
        print(f"   Failed: {len(failed_tests)} tests")
        print(f"   Passed: {successful_tests} tests")
        print(f"   Total: {total_tests} tests")
        print("\nFailed Tests:")
        for i, failure in enumerate(failed_tests, 1):
            # Truncate very long error messages for readability
            truncated_failure = failure[:97] + "..." if len(failure) > 100 else failure
            print(f"   {i:2d}. {truncated_failure}")

    print("=" * 80)

    return results, summary_stats, len(failed_tests) == 0


# ============================================================================
# PYTEST INTEGRATION
# ============================================================================


class TestFeatureImportanceComprehensive:
    """Comprehensive test suite for feature importance methods."""

    def test_all_feature_importance_methods(self, test_data, feature_config):
        """Test all feature importance methods across all models."""
        data_train, data_dev, data_test = test_data
        feature_cols, feature_groups = feature_config

        warnings.filterwarnings("ignore")

        model_configs = get_model_configs()
        failed_tests = []

        for ml_type, config in model_configs.items():
            for model_name in config["models"]:
                try:
                    # Create and fit training instance
                    training = create_training_instance(
                        data_train, data_dev, data_test, ml_type, model_name, feature_cols, feature_groups
                    )
                    training.fit()

                    # Test each feature importance method
                    for method_name in FI_METHODS:
                        result = _test_fi_method(training, method_name)
                        if not result["success"]:
                            failed_tests.append(f"{ml_type}/{model_name}/{method_name}: {result['error']}")

                except Exception as e:
                    for method_name in FI_METHODS:
                        failed_tests.append(f"{ml_type}/{model_name}/{method_name}: Model fitting failed: {e!s}")

        # Fail the test if there are any failures
        if failed_tests:
            print(f"\nFailed tests ({len(failed_tests)}):")
            for failure in failed_tests:  # Show all failures
                print(f"  - {failure}")

            # Fail the test on any failure
            assert len(failed_tests) == 0, f"Feature importance test failed: {len(failed_tests)} method(s) failed"


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        # Run with pytest
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "pytest", __file__, "-v"], check=False, capture_output=False, text=True
        )
        sys.exit(result.returncode)
    else:
        # Run standalone
        results, summary_stats, all_passed = run_comprehensive_feature_importance_tests()

        # Exit with appropriate code based on test results
        sys.exit(0 if all_passed else 1)
