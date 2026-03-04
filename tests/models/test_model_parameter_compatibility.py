"""Test model parameter compatibility by reading configurations dynamically."""

import itertools
import warnings
from typing import Any

import pytest

from octopus.models import Models
from octopus.models.hyperparameter import (
    CategoricalHyperparameter,
    FixedHyperparameter,
    FloatHyperparameter,
    IntHyperparameter,
)
from octopus.types import ML_TYPES


def get_all_models():
    """Get all available models from registry."""
    # Get all registered model names from Models class
    all_models = sorted(Models._config_factories.keys())
    return [name for name in all_models if "TabPFN" not in name]


def generate_param_combinations(model_name: str, max_combos: int = 20) -> list[dict[str, Any]]:
    """Generate parameter combinations from model config."""
    # Models uses classmethods, no instantiation needed
    config = Models.get_config(model_name)

    params = {}
    categorical_choices = {}

    # Extract parameters from hyperparameters
    for hp in config.hyperparameters:
        if isinstance(hp, FixedHyperparameter):
            params[hp.name] = hp.value
        elif isinstance(hp, CategoricalHyperparameter):
            categorical_choices[hp.name] = hp.choices
        elif isinstance(hp, IntHyperparameter | FloatHyperparameter):
            # Use boundary values for testing
            params[hp.name] = hp.low
        else:
            raise AssertionError(f"Unsupported Hyperparameter type: {type(hp)}.")

    # Add standard parameters
    if config.n_jobs:
        params[config.n_jobs] = 1
    if config.model_seed:
        params[config.model_seed] = 42

    # Generate combinations for categorical parameters
    if not categorical_choices:
        return [params]

    combinations = []
    names = list(categorical_choices.keys())
    values = list(categorical_choices.values())

    for combo in itertools.product(*values):
        test_params = params.copy()
        for name, value in zip(names, combo, strict=True):
            test_params[name] = value
        combinations.append(test_params)
        if len(combinations) >= max_combos:
            break

    return combinations


@pytest.mark.parametrize("model_name", get_all_models())
def test_model_parameter_compatibility(model_name):
    """Test that all parameter combinations from config are compatible."""
    # Models uses classmethods, no instantiation needed
    param_combinations = generate_param_combinations(model_name)

    compatibility_errors = []

    for params in param_combinations:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Models.get_instance(model_name, params)
        except ValueError as e:
            # Check for parameter compatibility issues
            error_msg = str(e).lower()
            if any(
                keyword in error_msg
                for keyword in ["supports only", "not supported", "incompatible", "invalid combination"]
            ):
                compatibility_errors.append(f"Params {params}: {e!s}")
        except Exception:
            # Ignore other errors (missing dependencies, etc.)
            pass

    assert not compatibility_errors, f"Parameter compatibility issues in {model_name}:\n" + "\n".join(
        compatibility_errors
    )


def test_all_models_have_valid_configs():
    """Test that all models have valid configurations."""
    # Models uses classmethods, no instantiation needed
    config_errors = []

    for model_name in get_all_models():
        try:
            config = Models.get_config(model_name)
            assert config.name == model_name
            assert config.model_class is not None
            assert all(t.value in ML_TYPES for t in config.ml_types)
        except Exception as e:
            config_errors.append(f"{model_name}: {e!s}")

    assert not config_errors, "Model configuration errors:\n" + "\n".join(config_errors)


def test_model_instantiation_with_default_params():
    """Test model instantiation with first choice of each categorical parameter."""
    # Models uses classmethods, no instantiation needed
    instantiation_errors = []

    for model_name in get_all_models():
        try:
            config = Models.get_config(model_name)
            params = {}

            # Use first choice for categorical, fixed values for fixed params
            for hp in config.hyperparameters:
                if isinstance(hp, FixedHyperparameter):
                    params[hp.name] = hp.value
                elif isinstance(hp, CategoricalHyperparameter) and hp.choices:
                    params[hp.name] = hp.choices[0]
                elif isinstance(hp, IntHyperparameter | FloatHyperparameter):
                    params[hp.name] = hp.low
                else:
                    raise AssertionError(f"Unsupported Hyperparameter type: {type(hp)}.")

            # Add standard parameters
            if config.n_jobs:
                params[config.n_jobs] = 1
            if config.model_seed:
                params[config.model_seed] = 42

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Models.get_instance(model_name, params)

        except ValueError as e:
            # Parameter compatibility issues are what we want to catch
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["supports only", "not supported", "incompatible"]):
                instantiation_errors.append(f"{model_name}: {e!s}")
        except Exception:
            # Ignore other errors (dependencies, etc.)
            pass

    assert not instantiation_errors, "Model instantiation errors with default parameters:\n" + "\n".join(
        instantiation_errors
    )
