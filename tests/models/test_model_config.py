"""Test hyperparameter."""

import pytest

from octopus.models.config import BaseModel, ModelConfig
from octopus.models.hyperparameter import CategoricalHyperparameter, FloatHyperparameter
from octopus.types import FIComputeMethod, MLType


class DummyModel(BaseModel):
    """Dummy model for testing."""

    pass


def test_model_config_initialization():
    """Test initialization of ModelConfig."""
    hyperparameters = [
        FloatHyperparameter(name="alpha", low=1e-5, high=1e5, log=True),
        CategoricalHyperparameter(name="fit_intercept", choices=[True, False]),
    ]

    model = ModelConfig(
        model_class=DummyModel,
        feature_method=FIComputeMethod.INTERNAL,
        ml_types=[MLType.REGRESSION],
        hyperparameters=hyperparameters,
        n_jobs="n_jobs",
        model_seed="random_seed",
    )

    # Name is not set during initialization - it's added by Models.get_config()
    assert not hasattr(model, "name")
    assert isinstance(model.model_class, type)
    assert model.feature_method == FIComputeMethod.INTERNAL
    assert model.supports_ml_type(MLType.REGRESSION)
    assert model.hyperparameters == hyperparameters
    assert model.n_jobs == "n_jobs"
    assert model.model_seed == "random_seed"
    assert model.n_repeats is None


def test_model_config_with_conflict():
    """Test ModelConfig initialization with hyperparameter name conflict."""
    hyperparameters = [
        FloatHyperparameter(name="n_jobs", low=1e-5, high=1e5, log=True),
        CategoricalHyperparameter(name="fit_intercept", choices=[True, False]),
    ]

    with pytest.raises(
        ValueError,
        match=r"Hyperparameter 'n_jobs' is not allowed in 'hyperparameters'\.",
    ):
        ModelConfig(
            model_class=DummyModel,
            feature_method=FIComputeMethod.INTERNAL,
            ml_types=[MLType.REGRESSION],
            hyperparameters=hyperparameters,
            n_jobs="n_jobs",
            model_seed="random_seed",
        )
