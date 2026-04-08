"""Octo module with fit/predict interface."""

from __future__ import annotations

import os

from attrs import Factory, define, field, validators

from octopus.logger import get_logger
from octopus.models import Models
from octopus.modules import ModuleExecution, Task
from octopus.types import FIComputeMethod, ModelName, ScoringMethod

logger = get_logger()

_RUNNING_IN_TESTSUITE = "RUNNING_IN_TESTSUITE" in os.environ


def _convert_models(value):
    """Convert model names to ModelName enum values, deduplicating."""
    if value is None:
        return None
    return list({ModelName(m) for m in value})


@define
class Octo(Task):
    """Octo module for feature selection and model optimization.

    Uses Optuna for hyperparameter optimization with cross-validation, supporting:
    - Multiple ML models
    - MRMR feature selection
    - Ensemble selection
    - Bag-based model ensembling

    Configuration:
        models: List of model names to optimize
        n_inner_splits: Number of inner CV folds
        n_trials: Number of Optuna trials
        ensemble_selection: Whether to perform ensemble selection
        n_mrmr_features: Number-of-feature options for MRMR-based Optuna search
    """

    models: list[ModelName] | None = field(
        default=None,
        converter=_convert_models,
    )
    """Models for ML. If None, defaults are resolved at fit time based on ml_type."""

    n_inner_splits: int = field(validator=[validators.instance_of(int)], default=5)
    """Number of inner folds."""

    inner_split_seeds: list[int] = field(
        default=Factory(lambda: [0]),
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(int),
            iterable_validator=validators.instance_of(list),
        ),
    )
    """List of integers used as seeds for data splitting."""

    max_outliers: int = field(validator=[validators.instance_of(int)], default=3)
    """Maximum number of outliers, optimized by Optuna"""

    fi_methods: list[FIComputeMethod] = field(
        default=Factory(lambda: [FIComputeMethod.PERMUTATION]),
        converter=lambda vs: [FIComputeMethod(v) for v in vs],
        validator=validators.deep_iterable(
            member_validator=validators.in_(
                [FIComputeMethod.PERMUTATION, FIComputeMethod.SHAP, FIComputeMethod.CONSTANT]
            ),
            iterable_validator=validators.instance_of(list),
        ),
    )
    """Feature importance methods for best bag."""

    n_startup_trials: int = field(validator=[validators.instance_of(int)], default=15)
    """Number of Optuna startup trials (random sampler)"""

    ensemble_selection: bool = field(validator=[validators.in_([True, False])], default=False)
    """Whether to perform ensemble selection."""

    n_ensemble_candidates: int = field(validator=[validators.instance_of(int)], default=50)
    """Number of top-performing bags to keep as candidates for ensemble selection."""

    n_trials: int = field(validator=[validators.instance_of(int)], default=200 if not _RUNNING_IN_TESTSUITE else 3)
    """Number of Optuna trials."""

    hyperparameters: dict = field(validator=[validators.instance_of(dict)], default=Factory(dict))
    """Bring own hyperparameter space."""

    max_features: int = field(validator=[validators.instance_of(int)], default=0)
    """Maximum features to constrain hyperparameter optimization. Default is zero (off)."""

    penalty_factor: float = field(validator=[validators.instance_of(float)], default=1.0)
    """Penalty multiplier for the feature-count constraint in Optuna optimization.

    Scales the penalty applied when the number of selected features exceeds
    ``max_features``.  For regression tasks whose target metric is not in the
    0..1 range the default of 1.0 may be too small or too large; adjust this
    value to balance feature reduction against prediction quality.
    """

    n_mrmr_features: list[int] = field(validator=[validators.instance_of(list)], default=Factory(list))
    """Number-of-feature options for MRMR pre-selection during Optuna optimization.

    Each integer specifies a number of top features to pre-select via MRMR
    (Max-Relevance Min-Redundancy). The resulting subsets become an additional
    Optuna hyperparameter, so each trial may use a different subset size.
    The full feature set is always included as an option.

    Example: ``[10, 20, 50]`` pre-computes the top-10, top-20, and top-50
    MRMR features; Optuna then chooses among these three subsets plus all
    features.  An empty list (default) disables MRMR and uses all features
    in every trial.
    """

    scoring_method: ScoringMethod = field(
        default=ScoringMethod.COMBINED,
        converter=ScoringMethod,
        validator=validators.in_(list(ScoringMethod)),
    )
    """How to calculate the bag performance for the optuna optimization target."""

    def __attrs_post_init__(self):
        # Only enforce constrained-HPO compatibility when max_features > 0 and models are specified
        if self.max_features > 0 and self.models is not None:
            incompatible_models: list[ModelName] = []

            for m in self.models:
                config = Models.get_config(m)
                chpo_flag = config.chpo_compatible
                logger.info(f"Model '{m}': chpo_compatible={chpo_flag}")

                if not chpo_flag:
                    incompatible_models.append(m)

            if incompatible_models:
                msg = (
                    "Octo: The following models are not compatible with constrained HPO. "
                    "Please remove those model or turn constrained HPO off (max_features=0): "
                    + ", ".join(incompatible_models)
                )
                logger.error(msg)
                raise ValueError(msg)

    def create_module(self) -> ModuleExecution:
        """Create OctoModule execution instance."""
        # import only during execution to avoid heavy dependency at config stage
        from .core import OctoModuleTemplate  # noqa: PLC0415

        return OctoModuleTemplate(config=self)
