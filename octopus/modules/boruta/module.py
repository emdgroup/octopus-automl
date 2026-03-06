"""Boruta module with fit/predict interface."""

from __future__ import annotations

from attrs import define, field, validators

from ..base import ModuleExecution, Task


@define
class Boruta(Task):
    """Boruta module for feature selection.

    Uses the Boruta algorithm to identify all relevant features by comparing
    importance scores with shadow features.

    Configuration:
        model: Model to use for Boruta (defaults based on ml_type)
        cv: Number of CV folds
        perc: Percentile threshold for shadow feature comparison
        alpha: Significance level for p-values
    """

    model: str = field(validator=[validators.instance_of(str)], default="")
    """Model used by Boruta."""

    cv: int = field(validator=[validators.instance_of(int)], default=5)
    """Number of folds for CV."""

    perc: int = field(validator=[validators.instance_of(int)], default=100)
    """Percentile (threshold) for comparison between shadow and real features."""

    alpha: float = field(validator=[validators.instance_of(float)], default=0.05)
    """Level at which the corrected p-values will get rejected."""

    def create_module(self) -> ModuleExecution:
        """Create BorutaModule execution instance."""
        # import only during execution to avoid heavy dependency at config stage
        from .core import BorutaModule  # noqa: PLC0415

        return BorutaModule(config=self)
