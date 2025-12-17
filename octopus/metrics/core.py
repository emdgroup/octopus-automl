"""Core metrics registry functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from octopus.exceptions import UnknownMetricError

if TYPE_CHECKING:
    from collections.abc import Callable

    from .config import Metric


class Metrics:
    """Central registry for metrics.

    Usage:
        # Get metric instance
        metric = Metrics.get_instance("AUCROC")
        metric.calculate(y_true, y_pred)

        # Get direction
        direction = Metrics.get_direction("AUCROC")
    """

    # Internal registry: metric name -> function returning Metric
    _config_factories: ClassVar[dict[str, Callable[[], Metric]]] = {}

    # Internal cache: metric name -> Metric
    _metric_configs: ClassVar[dict[str, Metric]] = {}

    @classmethod
    def get_all_metrics(cls) -> dict[str, Callable[[], Metric]]:
        """Get all registered metric factory functions.

        Returns:
            Dictionary mapping metric names to their factory functions.
        """
        return cls._config_factories

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[[], Metric]], Callable[[], Metric]]:
        """Register a metric factory function under a given name.

        Args:
            name: The name to register the metric under.

        Returns:
            Decorator function.
        """

        def decorator(factory: Callable[[], Metric]) -> Callable[[], Metric]:
            if name in cls._config_factories:
                raise ValueError(f"Metric '{name}' is already registered.")
            cls._config_factories[name] = factory
            return factory

        return decorator

    @classmethod
    def get_instance(cls, name: str) -> Metric:
        """Get metric instance by name.

        This is the primary method for getting a metric to use for calculation.
        Returns a Metric instance that has calculate() and calculate_t2e() methods.

        Args:
            name: The name of the metric to retrieve.

        Returns:
            Metric instance with calculate methods.

        Raises:
            UnknownMetricError: If no metric with the specified name is found.

        Usage:
            metric = Metrics.get_instance("AUCROC")
            value = metric.calculate(y_true, y_pred)
        """
        # Return cached config if available
        if name in cls._metric_configs:
            return cls._metric_configs[name]

        # Lookup factory
        factory = cls._config_factories.get(name)
        if factory is None:
            available = ", ".join(sorted(cls._config_factories.keys()))
            raise UnknownMetricError(
                f"Unknown metric '{name}'. Available metrics are: {available}. "
                "Please check the metric name and try again."
            )

        # Build config via factory and enforce name consistency
        config = factory()
        object.__setattr__(config, "name", name)
        cls._metric_configs[name] = config
        return config

    @classmethod
    def get_direction(cls, name: str) -> str:
        """Get the optuna direction by name.

        Args:
            name: The name of the metric.

        Returns:
            "maximize" if higher_is_better is True, else "minimize".
        """
        return "maximize" if cls.get_instance(name).higher_is_better else "minimize"

    @classmethod
    def get_by_type(cls, *ml_types: str) -> list[str]:
        """Get list of metric names for specified ML types.

        Args:
            *ml_types: One or more ML types (e.g., "regression", "classification", "multiclass", "timetoevent").

        Returns:
            List of metric names matching the specified ML types.

        Example:
            >>> Metrics.get_by_type("regression")
            ['RMSE', 'MAE', 'R2', ...]
            >>> Metrics.get_by_type("classification", "multiclass")
            ['AUCROC', 'Accuracy', ...]
        """
        matching_metrics = []
        for name, factory in cls._config_factories.items():
            metric = factory()
            if metric.ml_type in ml_types:
                matching_metrics.append(name)
        return sorted(matching_metrics)
