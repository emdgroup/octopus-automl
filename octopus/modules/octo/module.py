"""Octo module with fit/predict interface."""

from __future__ import annotations

import os

from attrs import Factory, define, field, validators

from octopus.logger import get_logger
from octopus.models import Models
from octopus.models.model_name import ModelName

from ..base import ModuleExecution, Task

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
        n_folds_inner: Number of inner CV folds
        n_trials: Number of Optuna trials
        ensemble_selection: Whether to perform ensemble selection
        mrmr_feature_numbers: Feature counts for MRMR feature selection
    """

    models: list[ModelName] | None = field(
        default=None,
        converter=_convert_models,
    )
    """Models for ML. If None, defaults are resolved at fit time based on ml_type."""

    n_folds_inner: int = field(validator=[validators.instance_of(int)], default=5)
    """Number of inner folds."""

    datasplit_seeds_inner: list[int] = field(
        default=Factory(lambda: [0]),
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(int),
            iterable_validator=validators.instance_of(list),
        ),
    )
    """List of integers used as seeds for data splitting."""

    model_seed: int = field(validator=[validators.instance_of(int)], default=0)
    """Model seed."""

    n_jobs: int = field(validator=[validators.instance_of(int)], default=1)
    """Number of CPUs used for every model training."""

    max_outl: int = field(validator=[validators.instance_of(int)], default=3)
    """Maximum number of outliers, optimized by Optuna"""

    fi_methods_bestbag: list[str] = field(
        default=Factory(lambda: ["permutation"]),
        validator=validators.deep_iterable(
            member_validator=validators.in_(["permutation", "shap", "constant"]),
            iterable_validator=validators.instance_of(list),
        ),
    )
    """Feature importance methods for best bag."""

    inner_parallelization: bool = field(validator=[validators.instance_of(bool)], default=True)
    """Enable inner parallelization. Defaults is True."""

    n_workers: int = field(default=None)
    """Number of workers."""

    optuna_seed: int = field(validator=[validators.instance_of(int)], default=0)
    """Seed for Optuna TPESampler, default=0"""

    n_optuna_startup_trials: int = field(validator=[validators.instance_of(int)], default=15)
    """Number of Optuna startup trials (random sampler)"""

    ensemble_selection: bool = field(validator=[validators.in_([True, False])], default=False)
    """Whether to perform ensemble selection."""

    ensel_n_save_trials: int = field(validator=[validators.instance_of(int)], default=50)
    """Number of top trials to be saved for ensemble selection (bags)."""

    n_trials: int = field(validator=[validators.instance_of(int)], default=200 if not _RUNNING_IN_TESTSUITE else 3)
    """Number of Optuna trials."""

    hyperparameters: dict = field(validator=[validators.instance_of(dict)], default=Factory(dict))
    """Bring own hyperparameter space."""

    max_features: int = field(validator=[validators.instance_of(int)], default=0)
    """Maximum features to constrain hyperparameter optimization. Default is zero (off)."""

    penalty_factor: float = field(validator=[validators.instance_of(float)], default=1.0)
    """Factor to penalize optuna target related to feature constraint."""

    mrmr_feature_numbers: list = field(validator=[validators.instance_of(list)], default=Factory(list))
    """List of feature numbers to be investigated by mrmr."""

    resume_optimization: bool = field(validator=[validators.instance_of(bool)], default=False)
    """Resume HPO, use existing optuna.db, don't delete optuna.db"""

    optuna_return: str = field(default="pool", validator=[validators.in_(["pool", "average"])])
    """How to calculate the bag performance for the optuna optimization target."""

    def __attrs_post_init__(self):
        # (1) set default of n_workers to n_folds_inner
        if self.n_workers is None:
            self.n_workers = self.n_folds_inner
        if self.n_workers != self.n_folds_inner:
            logger.warning(
                f"Octofull Warning: n_workers ({self.n_workers}) does not match n_folds_inner ({self.n_folds_inner})",
            )
        # (2) Only enforce constrained-HPO compatibility when max_features > 0 and models are specified
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
