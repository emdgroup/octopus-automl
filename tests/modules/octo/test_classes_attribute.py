"""Test that classification models have classes_ attribute after fitting."""

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


def get_classification_models():
    """Get all classification models dynamically from Models registry."""
    models = []

    # Models to exclude
    excluded_models = {}

    # Get all models from the registry
    for model_name in Models._config_factories:
        try:
            config = Models.get_config(model_name)
            if config.ml_type == "classification" and model_name not in excluded_models:
                models.append(model_name)
        except Exception:
            # Skip models that can't be loaded (e.g., missing dependencies)
            continue

    return sorted(models)  # Sort for consistent test order


def get_default_params(model_name):
    """Get default parameters for a model."""
    # Models uses classmethods, no instantiation needed
    config = Models.get_config(model_name)
    params = {}

    for hp in config.hyperparameters:
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

    if config.n_jobs:
        params[config.n_jobs] = 1
    if config.model_seed:
        params[config.model_seed] = 42

    return params


@pytest.fixture
def sample_data():
    """Create minimal test data."""
    np.random.seed(42)
    n = 50

    data = pd.DataFrame(
        {
            "x1": np.random.normal(0, 1, n),
            "x2": np.random.normal(0, 1, n),
            "target": np.random.choice([0, 1], n),
            "row_id_col": range(n),
        }
    )

    # Simple train/dev/test split
    train = data[:30].copy()
    dev = data[30:40].copy()
    test = data[40:].copy()

    return train, dev, test


@pytest.mark.parametrize("model_name", get_classification_models())
def test_classification_model_has_classes_attribute(sample_data, model_name):
    """Test that classification model has classes_ attribute after fitting."""
    train, dev, test = sample_data

    try:
        training = Training(
            training_id=f"test_{model_name}",
            ml_type="classification",
            target_assignments={"target": "target"},
            feature_cols=["x1", "x2"],
            row_column="row_id_col",
            data_train=train,
            data_dev=dev,
            data_test=test,
            target_metric="accuracy",
            max_features=0,
            feature_groups={},
            config_training={
                "ml_model_type": model_name,
                "ml_model_params": get_default_params(model_name),
                "outl_reduction": 0,
            },
        )

        # Fit the model
        training.fit()

        # Check that classes_ attribute exists
        assert hasattr(training.model, "classes_"), f"{model_name} missing classes_ attribute"

        # Check that classes_ contains expected values
        classes = training.model.classes_
        expected_classes = {0, 1}
        actual_classes = set(classes.tolist() if hasattr(classes, "tolist") else list(classes))
        assert actual_classes == expected_classes, (
            f"{model_name} classes_ = {actual_classes}, expected {expected_classes}"
        )

    except ImportError:
        pytest.skip(f"Skipping {model_name} due to missing dependencies")
