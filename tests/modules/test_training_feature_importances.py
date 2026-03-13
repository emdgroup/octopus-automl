"""Feature Importance Test Suite for Training Class.

Tests all feature importance methods across all available models.
Each test is marked with ``@pytest.mark.forked`` so it runs in its own
subprocess — this provides complete isolation between tests, preventing:

- CatBoost's C++ destructor segfault when Python's GC finalizes objects
- numba/llvmlite LLVM pass-manager crash from accumulated JIT compilations
- Memory accumulation from session-scoped model caches

See datasets_local/specifications_refactorfi/02_ci_segfault_investigation.md
for details on why this structure was chosen.

Usage:
    pytest test_training_feature_importances.py -v
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

ML_TYPE_CONFIGS = {
    MLType.BINARY: {
        "target_assignments": {"target": "target_class"},
        "target_metric": "AUCROC",
    },
    MLType.REGRESSION: {
        "target_assignments": {"target": "target_reg"},
        "target_metric": "R2",
    },
    MLType.TIMETOEVENT: {
        "target_assignments": {"duration": "duration", "event": "event"},
        "target_metric": "CI",
    },
    MLType.MULTICLASS: {
        "target_assignments": {"target": "target_multiclass"},
        "target_metric": "ACCBAL_MC",
    },
}


def _get_available_models_by_type():
    """Get all available models dynamically from the registry, grouped by ML type."""
    all_models = Models._config_factories.keys()
    models_by_type = {ml_type: [] for ml_type in MLType}

    for model_name in all_models:
        try:
            model_config = Models.get_config(model_name)
            for ml_type in MLType:
                if model_config.supports_ml_type(ml_type):
                    models_by_type[ml_type].append(model_name)
        except Exception:
            continue

    return models_by_type


def _generate_model_params():
    """Generate (ml_type, model_name) param combos for pytest.

    Each combo gets one test that runs ALL FI methods sequentially.
    """
    available_models = _get_available_models_by_type()
    params = []
    for ml_type, model_names in available_models.items():
        for model_name in model_names:
            params.append(
                pytest.param(
                    ml_type,
                    model_name,
                    id=f"{ml_type.value}-{model_name}",
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

    # Inject some NaN values to test robustness
    nan_mask = np.random.random(n_samples) < 0.1
    data.loc[nan_mask, "num_col1"] = np.nan

    nominal_col = np.random.choice([1, 2, 3], n_samples).astype(float)
    nominal_col[np.random.random(n_samples) < 0.05] = np.nan
    data["nominal_col"] = nominal_col

    data["row_id"] = range(n_samples)

    data["target_class"] = np.random.choice([0, 1], n_samples)
    data["target_multiclass"] = np.random.choice([0, 1, 2], n_samples)
    data["target_reg"] = 0.5 * data["num_col1"] + 0.3 * data["num_col2"] + np.random.normal(0, 1, n_samples)
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


@pytest.mark.forked
@pytest.mark.parametrize("ml_type,model_name", _generate_model_params())
def test_feature_importance(ml_type, model_name):
    """Test all FI methods for a single model in an isolated subprocess.

    Each test runs in its own forked process (``@pytest.mark.forked``),
    providing complete isolation.  This prevents:

    - CatBoost C++ destructor segfaults during garbage collection
    - numba/llvmlite LLVM pass-manager crashes from accumulated JIT state
    - Memory accumulation across tests

    The model is fitted once, all FI methods run sequentially, and the
    entire process exits cleanly when the test completes.
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

    for fi_method in FI_METHODS:
        fi_keys = _run_fi_method(training, fi_method)

        for key in fi_keys:
            fi_data = training.feature_importances.get(key)
            assert fi_data is not None, f"Feature importance key '{key}' not found after {fi_method}"
            # calculate_fi_internal legitimately returns empty for models without
            # built-in feature importances (e.g. GaussianProcess, SVM with non-linear kernel)
            if fi_method != "calculate_fi_internal":
                assert len(fi_data) > 0, f"Feature importance '{key}' is empty after {fi_method}"
