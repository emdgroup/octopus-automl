"""Metric Config."""

from collections.abc import Callable
from typing import Any

from attrs import define, field, validators

from octopus.models.config import PRED_TYPES, OctoArrayLike, PredType
from octopus.types import MLType, to_ml_types_frozenset, validate_ml_types

# Type alias for metric functions
# Metric functions should accept (y_true, y_pred, **kwargs) and return a numeric value.
# We use Callable[..., Any] because sklearn functions have varying signatures and
# return types (float, np.float64, floating[_16Bit], etc.).
# Parameter structure and return type conversion are handled by runtime validation
# and the compute() method which explicitly converts to float.
MetricFunction = Callable[..., Any]


@define
class Metric:
    """Metric instance.

    Represents a metric with its configuration and calculation methods.
    """

    name: str
    metric_function: MetricFunction = field(validator=validators.is_callable())
    ml_types: frozenset[MLType] = field(converter=to_ml_types_frozenset, validator=validate_ml_types)
    higher_is_better: bool = field(validator=validators.instance_of(bool))
    prediction_type: PredType = field(validator=validators.in_(PRED_TYPES))
    scorer_string: str = field(validator=validators.instance_of(str))  # needed for some sklearn functionalities
    metric_params: dict[str, Any] = field(factory=dict)

    def supports_ml_type(self, ml_type: MLType) -> bool:
        """Check if this metric supports the given ml_type."""
        return ml_type in self.ml_types

    @property
    def direction(self) -> str:
        """Optimization direction for Optuna ('maximize' or 'minimize')."""
        return "maximize" if self.higher_is_better else "minimize"

    def calculate(self, y_true: OctoArrayLike, y_pred: OctoArrayLike, **kwargs) -> float:
        """Calculate metric for classification/regression tasks.

        Args:
            y_true: True target values
            y_pred: Predicted values (predictions or probabilities depending on prediction_type)
            **kwargs: Additional keyword arguments passed to metric function

        Returns:
            Metric value as float

        Raises:
            ValueError: If called on a time-to-event metric
        """
        if self.supports_ml_type(MLType.TIMETOEVENT):
            raise ValueError(
                f"Metric '{self.name}' is a time-to-event metric. "
                "Use calculate_t2e(event_indicator, event_time, estimate) instead."
            )
        return float(self.metric_function(y_true, y_pred, **self.metric_params))

    def calculate_t2e(
        self, event_indicator: OctoArrayLike, event_time: OctoArrayLike, estimate: OctoArrayLike, **kwargs
    ) -> float:
        """Calculate metric for time-to-event tasks.

        Args:
            event_indicator: Boolean array indicating whether event occurred
            event_time: Array of event/censoring times
            estimate: Predicted risk/survival estimates from model
            **kwargs: Additional keyword arguments passed to metric function

        Returns:
            Metric value as float

        Raises:
            ValueError: If called on a non-time-to-event metric
        """
        if not self.supports_ml_type(MLType.TIMETOEVENT):
            raise ValueError(
                f"Metric '{self.name}' is not a time-to-event metric. Use calculate(y_true, y_pred) instead."
            )

        # Merge metric_params with any additional kwargs
        params = {**self.metric_params, **kwargs}
        result = self.metric_function(event_indicator, event_time, estimate, **params)

        # Handle tuple return (some T2E metrics return tuple)
        return float(result[0] if isinstance(result, tuple) else result)
