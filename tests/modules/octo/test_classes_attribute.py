"""Test that classification models have classes_ attribute after fitting."""

import numpy as np
import pandas as pd
import pytest

from octopus.models import Models
from octopus.models.model_name import ModelName
from octopus.modules.octo.training import Training
from octopus.types import MLType
from tests.helpers import get_default_model_params


def get_classification_models():
    """Get all classification models dynamically from Models registry."""
    models = []

    # Models to exclude
    excluded_models = {}

    # Get all models from the registry
    for model_name in Models._get_registered_models():
        try:
            config = Models.get_config(model_name)
            if config.supports_ml_type(MLType.BINARY) and model_name not in excluded_models:
                models.append(model_name)
        except Exception:
            # Skip models that can't be loaded (e.g., missing dependencies)
            continue

    return sorted(models)  # Sort for consistent test order


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
            "row_id": range(n),
        }
    )

    # Simple train/dev/test split
    train = data[:30].copy()
    dev = data[30:40].copy()
    test = data[40:].copy()

    return train, dev, test


@pytest.mark.parametrize("model_name", get_classification_models())
def test_classification_model_has_classes_attribute(sample_data, model_name: ModelName):
    """Test that classification model has classes_ attribute after fitting."""
    train, dev, test = sample_data

    try:
        training = Training(
            training_id=f"test_{model_name}",
            ml_type=MLType.BINARY,
            target_assignments={"default": "target"},
            feature_cols=["x1", "x2"],
            row_id_col="row_id",
            data_train=train,
            data_dev=dev,
            data_test=test,
            target_metric="accuracy",
            max_features=0,
            feature_groups={},
            config_training={
                "ml_model_type": model_name,
                "ml_model_params": get_default_model_params(model_name),
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
