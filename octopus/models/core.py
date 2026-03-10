"""Core models registry and inventory functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from octopus.exceptions import UnknownModelError
from octopus.types import MLType, ModelName

if TYPE_CHECKING:
    from collections.abc import Callable

    import optuna

    from .config import ModelConfig
    from .hyperparameter import Hyperparameter


class Models:
    """Central registry and inventory for models."""

    # Internal registry: model name -> function returning ModelConfig
    _config_factories: ClassVar[dict[str, Callable[[], ModelConfig]]] = {}

    # Internal cache: model name -> ModelConfig
    _model_configs: ClassVar[dict[str, ModelConfig]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[[], ModelConfig]], Callable[[], ModelConfig]]:
        """Register a model configuration factory function under a given name.

        Args:
            name: The name to register the model under.

        Returns:
            Decorator function.
        """

        def decorator(factory: Callable[[], ModelConfig]) -> Callable[[], ModelConfig]:
            if name in cls._config_factories:
                raise ValueError(f"Model '{name}' is already registered.")
            cls._config_factories[name] = factory
            return factory

        return decorator

    @classmethod
    def get_config(cls, name: ModelName) -> ModelConfig:
        """Get model configuration by name.

        Args:
            name: The name of the model to retrieve.

        Returns:
            The ModelConfig instance for the specified model.

        Raises:
            UnknownModelError: If no model with the specified name is found.
        """
        # Return cached config if available
        if name in cls._model_configs:
            return cls._model_configs[name]

        # Lookup factory
        factory = cls._config_factories.get(name)
        if factory is None:
            available = ", ".join(sorted(cls._config_factories.keys()))
            raise UnknownModelError(
                f"Unknown model '{name}'. Available models are: {available}. Please check the model name and try again."
            )

        # Build config via factory and enforce name consistency
        config = factory()
        # Use object.__setattr__ to bypass attrs' attribute restrictions
        object.__setattr__(config, "name", name)
        cls._model_configs[name] = config
        return config

    @classmethod
    def get_instance(cls, name: ModelName, params: dict[str, Any]):
        """Get model class by name and initialize it with the provided parameters.

        Args:
            name: The name of the model to retrieve.
            params: The parameters for model initialization.

        Returns:
            The initialized model instance.
        """
        model_config = cls.get_config(name)
        return model_config.model_class(**params)

    @classmethod
    def create_trial_parameters(
        cls,
        trial: optuna.trial.Trial,
        model_name: ModelName,
        custom_hyperparameters: dict[ModelName, list[Hyperparameter]] | None,
        n_jobs: int,
        model_seed: int,
    ) -> dict[str, Any]:
        """Create Optuna parameters for a specific model.

        Args:
            trial: The Optuna trial object.
            model_name: The name of the model to create parameters for.
            custom_hyperparameters: Optional dict mapping model names to custom hyperparameter lists.
                                   If None or model not in dict, uses default hyperparameters from config.
            n_jobs: Number of jobs for parallel execution.
            model_seed: Random seed for the model.

        Returns:
            Dictionary of parameter names to values.
        """
        # Get model configuration
        model_item = cls.get_config(model_name)

        # Resolve hyperparameters: use custom if provided, otherwise use defaults
        if custom_hyperparameters is not None and model_name in custom_hyperparameters:
            hyperparameters = custom_hyperparameters[model_name]
        else:
            hyperparameters = model_item.hyperparameters

        # Create parameters
        params: dict[str, Any] = {}

        for hp in hyperparameters:
            # get_config() always sets name, safe to access
            unique_name = f"{hp.name}_{model_item.name}"  # type: ignore[attr-defined]
            params[hp.name] = hp.suggest(trial, unique_name)

        if model_item.n_jobs is not None:
            params[model_item.n_jobs] = n_jobs
        if model_item.model_seed is not None:
            params[model_item.model_seed] = model_seed

        return params

    @classmethod
    def get_models_for_type(cls, ml_type: MLType) -> list[ModelName]:
        """Get all registered model names compatible with the given ml_type.

        Args:
            ml_type: The MLType to filter by.

        Returns:
            List of model names that support the given ml_type.
        """
        return [mn for name in cls._config_factories if cls.get_config(mn := ModelName(name)).supports_ml_type(ml_type)]

    @classmethod
    def get_defaults(cls, ml_type: MLType) -> list[ModelName]:
        """Get default model names for a given ml_type.

        Args:
            ml_type: The MLType to filter by.

        Returns:
            List of default model names that support the given ml_type.

        Raises:
            ValueError: If no default models are defined for the given ml_type.
        """
        defaults = [
            mn
            for name in cls._config_factories
            if (config := cls.get_config(mn := ModelName(name))).supports_ml_type(ml_type) and config.default
        ]
        if not defaults:
            raise ValueError(f"No default models defined for ml_type '{ml_type.value}'. Specify models explicitly.")
        return defaults

    @classmethod
    def validate_model_compatibility(cls, model_name: ModelName, ml_type: MLType) -> None:
        """Validate that a model is compatible with the given ml_type.

        Args:
            model_name: Name of the registered model.
            ml_type: The MLType to check compatibility against.

        Raises:
            ValueError: If the model does not support the given ml_type.
        """
        config = cls.get_config(model_name)
        if not config.supports_ml_type(ml_type):
            raise ValueError(
                f"Model '{model_name}' does not support ml_type '{ml_type.value}'. "
                f"Supported types: {', '.join(t.value for t in config.ml_types)}"
            )
