"""Feature Importance Test Suite for Training Class.

Tests all feature importance methods across all available models.

SHAP-based tests (``calculate_fi_featuresused_shap``) run in isolated
subprocesses via ``subprocess.run()`` to prevent:

- CatBoost's C++ destructor segfault when Python's GC finalizes objects
- numba/llvmlite LLVM pass-manager crash from accumulated JIT compilations
- SHAP + CatBoost + NaN background data segfaults

All other FI methods (internal, permutation, LOFO) run directly in the
pytest process — they are safe, faster, and maintain normal pytest protocol
(fixture setup/teardown, SetupState tracking, etc.).

This replaces the previous ``@pytest.mark.forked`` approach, which bypassed
pytest's ``SetupState`` for every test and caused fixture teardown errors
(``AssertionError: previous item was not torn down properly``) when
transitioning to non-forked tests in CI.

See:
- datasets_local/specifications_refactorfi/02_ci_segfault_investigation.md
- datasets_local/specifications_refactorfi/07_ci_warnings_error_report_proposal.md §2.1.6

Usage:
    pytest test_training_feature_importances.py -v
"""

import gc
import signal
import subprocess
import sys
import warnings
from pathlib import Path

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

TEST_CONFIG = {
    "n_samples": 30,
    "test_split": 0.3,
    "dev_split": 0.2,
    "random_seed": 42,
}

FI_METHODS = [
    "calculate_fi_internal",
    "calculate_fi_permutation",
    "calculate_fi_lofo",
    "calculate_fi_featuresused_shap",
    # "calculate_fi_shap",  # Excluded: kernel SHAP is too slow/memory-heavy for CI
]

# Methods that require subprocess isolation due to segfault risk
_SUBPROCESS_FI_METHODS = frozenset(
    {
        "calculate_fi_featuresused_shap",
        "calculate_fi_shap",
    }
)

ML_TYPE_CONFIGS = {
    MLType.BINARY: {
        "target_assignments": {"default": "target_class"},
        "target_metric": "AUCROC",
    },
    MLType.REGRESSION: {
        "target_assignments": {"default": "target_reg"},
        "target_metric": "R2",
    },
    MLType.TIMETOEVENT: {
        "target_assignments": {"duration": "duration", "event": "event"},
        "target_metric": "CI",
    },
    MLType.MULTICLASS: {
        "target_assignments": {"default": "target_multiclass"},
        "target_metric": "ACCBAL_MC",
    },
}


def _get_available_models_by_type():
    """Get all available models dynamically from the registry, grouped by ML type.

    Gracefully skips models whose factory functions fail to import
    (e.g. TabularNN models that require ``torch``, which may not be
    available on all platforms such as Windows CI).
    """
    all_models = Models._config_factories.keys()
    models_by_type: dict[MLType, list[str]] = {ml_type: [] for ml_type in MLType}

    for model_name in all_models:
        try:
            model_config = Models.get_config(model_name)
            for ml_type in MLType:
                if model_config.supports_ml_type(ml_type):
                    models_by_type[ml_type].append(model_name)
        except Exception:
            # Skip models whose dependencies can't be loaded
            # (e.g. torch DLL failure on Windows)
            continue

    return models_by_type


def _generate_model_fi_params():
    """Generate (ml_type, model_name, fi_method) param combos for pytest.

    Each combo gets its own test so that segfaults or crashes clearly
    identify which (model, FI method) combination failed.
    """
    available_models = _get_available_models_by_type()
    params = []
    for ml_type, model_names in available_models.items():
        for model_name in model_names:
            for fi_method in FI_METHODS:
                params.append(
                    pytest.param(
                        ml_type,
                        model_name,
                        fi_method,
                        id=f"{ml_type.value}-{model_name}-{fi_method}",
                    )
                )
    return params


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


def _create_test_data():
    """Create test dataset with mixed data types."""
    np.random.seed(TEST_CONFIG["random_seed"])
    n_samples = TEST_CONFIG["n_samples"]

    data = pd.DataFrame(
        {
            "num_col1": np.random.normal(10, 2, n_samples),
            "num_col2": np.random.normal(50, 10, n_samples),
        }
    )

    # Compute regression target BEFORE injecting NaN into features,
    # so target_reg is always NaN-free.
    data["target_reg"] = 0.5 * data["num_col1"] + 0.3 * data["num_col2"] + np.random.normal(0, 1, n_samples)

    # Inject some NaN values to test robustness
    nan_mask = np.random.random(n_samples) < 0.1
    data.loc[nan_mask, "num_col1"] = np.nan

    nominal_col = np.random.choice([1, 2, 3], n_samples).astype(float)
    nominal_col[np.random.random(n_samples) < 0.05] = np.nan
    data["nominal_col"] = nominal_col

    data["row_id"] = range(n_samples)

    data["target_class"] = np.random.choice([0, 1], n_samples)
    data["target_multiclass"] = np.random.choice([0, 1, 2], n_samples)
    data["duration"] = np.random.exponential(10, n_samples)
    data["event"] = np.random.choice([True, False], n_samples, p=[0.7, 0.3])

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


def _create_training_instance(
    data_train: pd.DataFrame,
    data_dev: pd.DataFrame,
    data_test: pd.DataFrame,
    ml_type: MLType,
    model_name: str,
    feature_cols: list[str],
    feature_groups: dict[str, list[str]],
) -> Training:
    """Create a Training instance for testing."""
    config = ML_TYPE_CONFIGS[ml_type]
    ml_model_params = _get_default_model_params(model_name)

    training_config = {
        "ml_model_type": model_name,
        "ml_model_params": ml_model_params,
        "outl_reduction": 0,
    }

    if ml_type == MLType.BINARY:
        training_config["positive_class"] = 1

    return Training(
        training_id=f"test_{ml_type}_{model_name}",
        ml_type=ml_type,
        target_assignments=config["target_assignments"],
        feature_cols=feature_cols,
        row_id_col="row_id",
        data_train=data_train,
        data_dev=data_dev,
        data_test=data_test,
        target_metric=config["target_metric"],
        max_features=0,
        feature_groups=feature_groups,
        config_training=training_config,
    )


def _run_fi_method(training: Training, method_name: str) -> list[str]:
    """Run a feature importance method and return the expected result key(s)."""
    if method_name == "calculate_fi_internal":
        training.calculate_fi_internal()
        return ["internal"]
    elif method_name == "calculate_fi_permutation":
        training.calculate_fi_permutation(partition="dev", n_repeats=1)
        return ["permutation_dev"]
    elif method_name == "calculate_fi_lofo":
        training.calculate_fi_lofo()
        return ["lofo_dev", "lofo_test"]
    elif method_name == "calculate_fi_featuresused_shap":
        training.calculate_fi_featuresused_shap(partition="dev")
        return ["shap_dev"]
    elif method_name == "calculate_fi_shap":
        training.calculate_fi_shap(partition="dev", shap_type="permutation")
        return ["shap_dev"]
    else:
        raise ValueError(f"Unknown method: {method_name}")


def _run_test_body(ml_type: MLType, model_name: str, fi_method: str) -> None:
    """Core test logic — called in-process or from subprocess entry point.

    Args:
        ml_type: The machine learning task type.
        model_name: Name of the model to test.
        fi_method: Name of the feature importance method to run.
    """
    warnings.filterwarnings("ignore")

    data_train, data_dev, data_test = _create_test_data()
    feature_cols = ["num_col1", "num_col2", "nominal_col"]
    feature_groups = {
        "numerical_group": ["num_col1", "num_col2"],
        "categorical_group": ["nominal_col"],
    }

    training = _create_training_instance(
        data_train, data_dev, data_test, ml_type, model_name, feature_cols, feature_groups
    )
    training.fit()

    fi_keys = _run_fi_method(training, fi_method)

    for key in fi_keys:
        fi_data = training.feature_importances.get(key)
        assert fi_data is not None, f"Feature importance key '{key}' not found after {fi_method}"
        # calculate_fi_internal legitimately returns empty for models without
        # built-in feature importances (e.g. GaussianProcess, SVM with non-linear kernel)
        if fi_method != "calculate_fi_internal":
            assert len(fi_data) > 0, f"Feature importance '{key}' is empty after {fi_method}"


def _run_in_subprocess(ml_type: MLType, model_name: str, fi_method: str) -> None:
    """Run a single FI test in an isolated subprocess.

    Invokes this module's ``__main__`` block in a fresh Python interpreter
    via ``subprocess.run()``.  This provides complete process isolation
    without using ``os.fork()`` (which is deprecated in multi-threaded
    processes on Python 3.12+).

    Args:
        ml_type: The machine learning task type.
        model_name: Name of the model to test.
        fi_method: Name of the feature importance method to run.

    Raises:
        pytest.fail: If the subprocess exits with a non-zero code or crashes.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__)),
            "--ml-type",
            ml_type.value,
            "--model-name",
            model_name,
            "--fi-method",
            fi_method,
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    if result.returncode != 0:
        # Truncate output to avoid overwhelming the test report
        stdout_tail = result.stdout[-2000:] if result.stdout else "(empty)"
        stderr_tail = result.stderr[-2000:] if result.stderr else "(empty)"

        if result.returncode < 0:
            # Negative return code means the process was killed by a signal
            try:
                sig_name = signal.Signals(-result.returncode).name
            except (ValueError, AttributeError):
                sig_name = f"signal {-result.returncode}"
            header = f"Subprocess CRASHED ({sig_name})"
        else:
            header = "Subprocess test failed"

        pytest.fail(
            f"{header} for {ml_type.value}-{model_name}-{fi_method}\n"
            f"Exit code: {result.returncode}\n"
            f"stdout:\n{stdout_tail}\n"
            f"stderr:\n{stderr_tail}"
        )


@pytest.mark.parametrize("ml_type,model_name,fi_method", _generate_model_fi_params())
def test_feature_importance(ml_type, model_name, fi_method):
    """Test a single FI method for a single model.

    SHAP-based methods run in isolated subprocesses to prevent segfaults
    from CatBoost GC, numba/LLVM crashes, and memory accumulation.
    All other FI methods run directly in the pytest process for speed
    and proper pytest protocol compliance.

    Test IDs look like::

        test_feature_importance[binary-CatBoostClassifier-calculate_fi_featuresused_shap]
    """
    if fi_method in _SUBPROCESS_FI_METHODS:
        _run_in_subprocess(ml_type, model_name, fi_method)
    else:
        _run_test_body(ml_type, model_name, fi_method)
        gc.collect()


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------
# When this module is executed as a script (via _run_in_subprocess), it runs
# a single (ml_type, model_name, fi_method) combination and exits.  A non-zero
# exit code (including signal-based crashes) is detected by the parent pytest
# process and reported as a test failure with full diagnostics.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a single feature-importance test in isolation.",
    )
    parser.add_argument("--ml-type", required=True, help="MLType enum value")
    parser.add_argument("--model-name", required=True, help="Model registry name")
    parser.add_argument("--fi-method", required=True, help="FI method name")
    args = parser.parse_args()

    _run_test_body(MLType(args.ml_type), args.model_name, args.fi_method)
