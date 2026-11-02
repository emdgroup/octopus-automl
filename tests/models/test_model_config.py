"""Test hyperparameter."""

import pytest
from sklearn.base import BaseEstimator

from octopus.models.config import ModelConfig
from octopus.models.hyperparameter import CategoricalHyperparameter, FloatHyperparameter
from octopus.types import FIComputeMethod, MLType


class DummyModel(BaseEstimator):
    """Dummy model for testing."""

    def fit(self, X, y):
        """Fit (no-op)."""
        return self

    def predict(self, X):
        """Predict (identity)."""
        return X


class DummyClassifier(DummyModel):
    """Dummy classifier for testing."""

    def predict_proba(self, X):
        """Predict probabilities (identity)."""
        return X


def test_model_config_initialization():
    """Test initialization of ModelConfig."""
    hyperparameters = [
        FloatHyperparameter(name="alpha", low=1e-5, high=1e5, log=True),
        CategoricalHyperparameter(name="fit_intercept", choices=[True, False]),
    ]

    model = ModelConfig(
        model_class=DummyModel,
        fi_method=FIComputeMethod.INTERNAL,
        ml_types=[MLType.REGRESSION],
        hyperparameters=hyperparameters,
        n_jobs="n_jobs",
        model_seed="random_seed",
    )

    # Name is not set during initialization - it's added by Models.get_config()
    assert not hasattr(model, "name")
    assert isinstance(model.model_class, type)
    assert model.fi_method == FIComputeMethod.INTERNAL
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
            fi_method=FIComputeMethod.INTERNAL,
            ml_types=[MLType.REGRESSION],
            hyperparameters=hyperparameters,
            n_jobs="n_jobs",
            model_seed="random_seed",
        )


@pytest.mark.parametrize("ml_type", [MLType.BINARY, MLType.MULTICLASS])
def test_model_config_classification_requires_predict_proba(ml_type):
    """Classification models must implement predict_proba."""
    with pytest.raises(TypeError, match=r"'DummyModel' is registered for .* but does not implement predict_proba"):
        ModelConfig(
            model_class=DummyModel,
            fi_method=FIComputeMethod.INTERNAL,
            ml_types=[ml_type],
            hyperparameters=[],
        )


@pytest.mark.parametrize("ml_types", [[MLType.BINARY], [MLType.MULTICLASS], [MLType.BINARY, MLType.MULTICLASS]])
def test_model_config_classification_with_predict_proba(ml_types):
    """Classification models implementing predict_proba are accepted."""
    model = ModelConfig(
        model_class=DummyClassifier,
        fi_method=FIComputeMethod.INTERNAL,
        ml_types=ml_types,
        hyperparameters=[],
    )
    assert model.model_class is DummyClassifier


@pytest.mark.parametrize("ml_type", [MLType.REGRESSION, MLType.TIMETOEVENT])
def test_model_config_non_classification_without_predict_proba(ml_type):
    """predict_proba is not required for regression and time-to-event models."""
    model = ModelConfig(
        model_class=DummyModel,
        fi_method=FIComputeMethod.INTERNAL,
        ml_types=[ml_type],
        hyperparameters=[],
    )
    assert model.supports_ml_type(ml_type)
