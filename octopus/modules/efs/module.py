"""Efs module (Ensemble Feature Selection) with fit/predict interface."""

from __future__ import annotations

from attrs import define, field, validators

from ..base import ModuleExecution, Task


@define
class Efs(Task):
    """EFS module for ensemble feature selection.

    Creates multiple models on random feature subsets and uses ensemble
    optimization to select the best combination of models.

    Configuration:
        model: Model to use for EFS (defaults to CatBoost based on ml_type)
        subset_size: Number of features in each random subset
        n_subsets: Number of random subsets to create
        cv: Number of CV folds
        max_n_iterations: Maximum iterations for ensemble optimization
        max_n_models: Maximum number of models to consider
    """

    model: str = field(validator=[validators.instance_of(str)], default="")
    """Model used by EFS (empty string uses default for ml_type)."""

    subset_size: int = field(validator=[validators.instance_of(int)], default=30)
    """Number of features in the subset."""

    n_subsets: int = field(validator=[validators.instance_of(int)], default=100)
    """Number of subsets."""

    cv: int = field(validator=[validators.instance_of(int)], default=5)
    """Number of CV folds for EFS."""

    max_n_iterations: int = field(validator=[validators.instance_of(int)], default=50)
    """Number of iterations for ensemble optimization."""

    max_n_models: int = field(validator=[validators.instance_of(int)], default=30)
    """Maximum number of models used in optimization, pruning."""

    def create_module(self) -> ModuleExecution:
        """Create EfsModule execution instance."""
        # import only during execution to avoid heavy dependency at config stage
        from .core import EfsModule  # noqa: PLC0415

        return EfsModule(config=self)
