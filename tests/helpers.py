"""Shared test helper utilities."""

from __future__ import annotations

import numpy as np

from octopus.models import Models
from octopus.models.hyperparameter import (
    CategoricalHyperparameter,
    FixedHyperparameter,
    FloatHyperparameter,
    IntHyperparameter,
)
from octopus.types import ModelName


def get_default_model_params(model_name: ModelName) -> dict:
    """Get default parameters for a model from its hyperparameter configuration.

    Extracts a single representative value for each hyperparameter:
    - Fixed: the fixed value
    - Categorical: the first choice
    - Int/Float: the midpoint (geometric mean for log-scaled)

    Also pins n_jobs=1 and model_seed=42 when supported.

    Args:
        model_name: The model to get defaults for.

    Returns:
        Dict of parameter name to default value.
    """
    model_config = Models.get_config(model_name)
    params: dict = {}

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
