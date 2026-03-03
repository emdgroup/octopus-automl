"""SFS module (sequential feature selection) with fit/predict interface."""

from __future__ import annotations

from attrs import define, field, validators

from octopus.modules.base import Task

from .core import SfsModule


@define
class Sfs(Task):
    """SFS module for sequential feature selection.

    Uses sequential feature selection (forward, backward, or floating variants)
    to find the optimal feature subset.

    Configuration:
        model: Model to use for SFS (defaults based on ml_type)
        cv: Number of CV folds
        sfs_type: Type of SFS (forward, backward, floating_forward, floating_backward)
    """

    model: str = field(validator=[validators.instance_of(str)], default="")
    """Model used by SFS."""

    cv: int = field(validator=[validators.instance_of(int)], default=5)
    """Number of CV folds for SFS."""

    sfs_type: str = field(
        validator=[validators.in_(["forward", "backward", "floating_forward", "floating_backward"])],
        default="backward",
    )
    """SFS type used."""

    def create_module(self) -> SfsModule:
        """Create SfsModule execution instance."""
        return SfsModule(config=self)
