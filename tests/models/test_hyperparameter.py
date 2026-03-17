"""Test hyperparameter."""

from typing import cast
from unittest.mock import MagicMock

import optuna
import pytest

from octopus.models import Models
from octopus.models.config import BaseModel, ModelConfig
from octopus.models.hyperparameter import (
    CategoricalHyperparameter,
    FixedHyperparameter,
    FloatHyperparameter,
    IntHyperparameter,
)
from octopus.models.model_name import ModelName
from octopus.types import FIComputeMethod, MLType


@pytest.mark.parametrize(
    "hyperparameter_type, name, kwargs, expected_exception",
    [
        # Valid int hyperparameter
        (IntHyperparameter, "para1", {"low": 1, "high": 10, "step": None, "log": False}, None),
        # Invalid int hyperparameter: low > high
        (IntHyperparameter, "para1", {"low": 10, "high": 1, "step": None, "log": False}, ValueError),
        # Invalid int hyperparameter with step
        (IntHyperparameter, "para1", {"low": 1, "high": 10, "step": -1, "log": False}, ValueError),
        # Valid int hyperparameter step
        (IntHyperparameter, "para1", {"low": 1, "high": 10, "step": 1, "log": False}, None),
        # Valid int hyperparameter log
        (IntHyperparameter, "para1", {"low": 1, "high": 10, "step": None, "log": True}, None),
        # Invalid int hyperparameter step and log selected
        (IntHyperparameter, "para1", {"low": 1, "high": 10, "step": 1, "log": True}, ValueError),
        # Valid float hyperparameter
        (FloatHyperparameter, "para1", {"low": 0.1, "high": 1.0, "step": None, "log": False}, None),
        # Invalid float hyperparameter with high less than low
        (FloatHyperparameter, "param1", {"low": 1.0, "high": 0.1, "step": None, "log": False}, ValueError),
        # Valid float hyperparameter step
        (FloatHyperparameter, "para1", {"low": 1, "high": 10, "step": 1, "log": False}, None),
        # Valid float hyperparameter log
        (FloatHyperparameter, "para1", {"low": 1, "high": 10, "step": None, "log": True}, None),
        # Invalid float hyperparameter step and log selected
        (FloatHyperparameter, "para1", {"low": 1, "high": 10, "step": 1, "log": True}, ValueError),
        # Valid categorical hyperparameter
        (CategoricalHyperparameter, "para1", {"choices": ["a", "b"]}, None),
        # Invalid categorical hyperparameter without choices
        (CategoricalHyperparameter, "para1", {"choices": []}, ValueError),
        # Valid fixed hyperparameter
        (FixedHyperparameter, "para1", {"value": 5}, None),
        # Invalid fixed hyperparameter without value
        (FixedHyperparameter, "para1", {"value": None}, ValueError),
    ],
)
def test_validate_hyperparameters(hyperparameter_type, name, kwargs, expected_exception):
    """Test validate hyperparameters."""
    if expected_exception:
        with pytest.raises(expected_exception):
            hyperparameter_type(name=name, **kwargs)
    else:
        hyperparameter_type(name=name, **kwargs)


def create_mock_trial():
    """Create a mock optuna trial."""
    mock = MagicMock(spec=optuna.trial.Trial)
    mock.suggest_int.return_value = 5
    mock.suggest_float.return_value = 0.5
    mock.suggest_categorical.return_value = "a"
    return mock


def test_int_hyperparameter_suggest():
    """Test IntHyperparameter suggest method."""
    mock_trial = create_mock_trial()
    hp = IntHyperparameter(name="test", low=1, high=10, log=True)

    result = hp.suggest(mock_trial, "unique_name")

    mock_trial.suggest_int.assert_called_once_with(name="unique_name", low=1, high=10, log=True)
    assert result == 5


def test_float_hyperparameter_suggest():
    """Test FloatHyperparameter suggest method."""
    mock_trial = create_mock_trial()
    hp = FloatHyperparameter(name="test", low=0.1, high=1.0, log=True)

    result = hp.suggest(mock_trial, "unique_name")

    mock_trial.suggest_float.assert_called_once_with(name="unique_name", low=0.1, high=1.0, log=True)
    assert result == 0.5


def test_categorical_hyperparameter_suggest():
    """Test CategoricalHyperparameter suggest method."""
    mock_trial = create_mock_trial()
    hp = CategoricalHyperparameter(name="test", choices=["a", "b", "c"])

    result = hp.suggest(mock_trial, "unique_name")

    mock_trial.suggest_categorical.assert_called_once_with(name="unique_name", choices=["a", "b", "c"])
    assert result == "a"


def test_fixed_hyperparameter_suggest():
    """Test FixedHyperparameter suggest method."""
    mock_trial = create_mock_trial()
    hp = FixedHyperparameter(name="test", value=42)

    result = hp.suggest(mock_trial, "unique_name")

    assert result == 42
    mock_trial.suggest_int.assert_not_called()


def test_step_takes_priority_over_log():
    """Test step parameter takes priority over log."""
    mock_trial = create_mock_trial()
    hp = IntHyperparameter(name="test", low=1, high=10, step=2, log=False)

    hp.suggest(mock_trial, "unique_name")

    mock_trial.suggest_int.assert_called_once_with(name="unique_name", low=1, high=10, step=2)


def test_create_trial_parameters():
    """Test create_trial_parameters uses suggest methods."""
    mock_trial = create_mock_trial()
    # Models uses classmethods, no instantiation needed

    hyperparameters = [
        IntHyperparameter(name="int_param", low=1, high=10),
        FixedHyperparameter(name="fixed_param", value=42),
    ]

    class DummyModel(BaseModel):
        """Dummy model for testing."""

        pass

    # Register a test model for this test
    @Models.register("TestModel")
    def test_model() -> ModelConfig:
        """Test model config."""
        return ModelConfig(
            model_class=DummyModel,
            feature_method=FIComputeMethod.INTERNAL,
            ml_types=[MLType.BINARY, MLType.MULTICLASS],
            hyperparameters=hyperparameters,
            n_jobs="n_jobs",
            model_seed="random_state",
        )

    try:
        result = Models.create_trial_parameters(
            trial=mock_trial,
            model_name=cast("ModelName", "TestModel"),
            custom_hyperparameters=None,
            n_jobs=2,
            model_seed=123,
        )

        mock_trial.suggest_int.assert_called_once_with(name="int_param_TestModel", low=1, high=10, log=False)

        expected = {"int_param": 5, "fixed_param": 42, "n_jobs": 2, "random_state": 123}
        assert result == expected
    finally:
        # Clean up test model registration to avoid polluting other tests
        Models._config_factories.pop("TestModel", None)
        Models._model_configs.pop("TestModel", None)
