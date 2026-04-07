"""Boruta module with fit/predict interface."""

from __future__ import annotations

from attrs import define, field, validators

from octopus.types import ModelName

from ..base import ModuleExecution, Task


@define
class Boruta(Task):
    """Boruta module for feature selection.

    Uses the Boruta algorithm to identify all relevant features by comparing
    importance scores with shadow features.

    Configuration:
        model: Model to use for Boruta (defaults based on ml_type)
        n_inner_splits: Number of CV splits
        threshold: Percentile threshold for shadow feature comparison (0-100)
        alpha: Significance level for p-values (0-1)
    """

    model: ModelName | None = field(default=None, converter=lambda v: ModelName(v) if v is not None else None)
    """Model used by Boruta. If None, defaults are resolved at fit time based on ml_type."""

    n_inner_splits: int = field(validator=[validators.instance_of(int)], default=5)
    """Number of inner splits."""

    threshold: int = field(
        default=100,
        validator=[validators.instance_of(int), validators.ge(0), validators.le(100)],
    )
    """Percentile threshold for comparison between shadow and real features (0-100)."""

    alpha: float = field(
        default=0.05,
        validator=[validators.instance_of(float), validators.gt(0), validators.lt(1)],
    )
    """Significance level at which the corrected p-values will get rejected (0-1)."""

    def create_module(self) -> ModuleExecution:
        """Create BorutaModule execution instance."""
        # import only during execution to avoid heavy dependency at config stage
        from .core import BorutaModule  # noqa: PLC0415

        return BorutaModule(config=self)
