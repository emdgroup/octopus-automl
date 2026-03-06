"""RFE module (Recursive Feature Elimination)."""

from __future__ import annotations

from attrs import define, field, validators

from ..base import ModuleExecution, Task


@define
class Rfe(Task):
    """RFE module for recursive feature elimination.

    Uses sklearn's RFECV with hyperparameter optimization to recursively
    eliminate features based on feature importances.

    Configuration:
        model: Model to use for RFE (defaults to CatBoost based on ml_type)
        step: Number of features to remove at each iteration
        min_features_to_select: Minimum number of features to keep
        cv: Number of CV folds for RFECV
        mode: "Mode1" (use optimized model) or "Mode2" (reoptimize at each step)
    """

    model: str = field(validator=[validators.instance_of(str)], default="")
    """Model used by RFE (empty string uses default for ml_type)."""

    step: int = field(validator=[validators.instance_of(int)], default=1)
    """Number of features to remove at each iteration."""

    min_features_to_select: int = field(validator=[validators.instance_of(int)], default=1)
    """Minimum number of features to be selected."""

    cv: int = field(validator=[validators.instance_of(int)], default=5)
    """Number of CV folds for RFE_CV."""

    mode: str = field(validator=[validators.in_(["Mode1", "Mode2"])], default="Mode1")
    """Mode used by RFE: Mode1=optimized model, Mode2=reoptimize each step."""

    def create_module(self) -> ModuleExecution:
        """Create RfeModule execution instance."""
        # import only during execution to avoid heavy dependency at config stage
        from .core import RfeModule  # noqa: PLC0415

        return RfeModule(config=self)
