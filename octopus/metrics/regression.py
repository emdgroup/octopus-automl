"""Regression metrics."""

import math

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from octopus.types import MLType, PredictionType

from .config import Metric
from .core import Metrics


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE).

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        float: RMSE value
    """
    return math.sqrt(mean_squared_error(y_true, y_pred))


@Metrics.register("R2")
def r2_metric() -> Metric:
    """R2 metric configuration."""
    return Metric(
        name="R2",
        metric_function=r2_score,
        ml_types=[MLType.REGRESSION],
        higher_is_better=True,
        prediction_type=PredictionType.PREDICTIONS,
        scorer_string="r2",
    )


@Metrics.register("MAE")
def mae_metric() -> Metric:
    """MAE metric configuration."""
    return Metric(
        name="MAE",
        metric_function=mean_absolute_error,
        ml_types=[MLType.REGRESSION],
        higher_is_better=False,
        prediction_type=PredictionType.PREDICTIONS,
        scorer_string="neg_mean_absolute_error",
    )


@Metrics.register("MSE")
def mse_metric() -> Metric:
    """MSE metric configuration."""
    return Metric(
        name="MSE",
        metric_function=mean_squared_error,
        ml_types=[MLType.REGRESSION],
        higher_is_better=False,
        prediction_type=PredictionType.PREDICTIONS,
        scorer_string="neg_mean_squared_error",
    )


@Metrics.register("RMSE")
def rmse_metric() -> Metric:
    """RMSE metric configuration."""
    return Metric(
        name="RMSE",
        metric_function=root_mean_squared_error,
        ml_types=[MLType.REGRESSION],
        higher_is_better=False,
        prediction_type=PredictionType.PREDICTIONS,
        scorer_string="neg_root_mean_squared_error",
    )
