"""AutoGluon module for automated machine learning."""

from __future__ import annotations

from typing import Literal

from attrs import define, field, validators

from ..base import ModuleExecution, Task


@define
class AutoGluon(Task):
    """AutoGluon TabularPredictor module for automated machine learning.

    Uses AutoGluon's TabularPredictor to automatically train and select
    the best model from an ensemble of models. Provides automatic feature
    engineering, hyperparameter optimization, and model selection.
    """

    time_limit: int | None = field(default=None, validator=validators.optional(validators.instance_of(int)))
    """Training time limit in seconds."""

    infer_limit: int | None = field(default=None, validator=validators.optional(validators.instance_of(int)))
    """Inference time limit per row in seconds."""

    memory_limit: float | Literal["auto"] = field(
        default="auto", validator=validators.or_(validators.instance_of(float), validators.in_(["auto"]))
    )
    """Memory limit in GB."""

    presets: list[str] = field(
        default=["medium_quality"],
        validator=validators.deep_iterable(
            member_validator=validators.and_(
                validators.instance_of(str),
                validators.in_(
                    [
                        "best_quality",
                        "high_quality",
                        "good_quality",
                        "medium_quality",
                        "experimental_quality",
                        "optimize_for_deployment",
                        "interpretable",
                        "ignore_text",
                    ]
                ),
            ),
            iterable_validator=validators.instance_of(list),
        ),
    )
    """AutoGluon quality presets."""

    n_bag_splits: int = field(default=5, validator=[validators.instance_of(int), validators.gt(1)])
    """Number of bagging/cross-validation splits (passed as num_bag_folds to AutoGluon)."""

    included_model_types: list[str] | None = field(
        default=None,
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.and_(
                    validators.instance_of(str),
                    validators.in_(
                        [
                            "GBM",
                            "CAT",
                            "XGB",
                            "RF",
                            "XT",
                            "KNN",
                            "LR",
                            "NN_TORCH",
                            "FASTAI",
                        ]
                    ),
                ),
                iterable_validator=validators.instance_of(list),
            )
        ),
    )
    """Specific model types to include (None = all)."""

    def create_module(self) -> ModuleExecution:
        """Create AutoGluonModule execution instance."""
        # import only during execution to avoid heavy dependency at config stage
        from .core import AutoGluonModule  # noqa: PLC0415

        return AutoGluonModule(config=self)
