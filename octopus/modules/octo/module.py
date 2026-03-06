"""Octo module with fit/predict interface."""

from __future__ import annotations

import os

from attrs import Factory, define, field, validators

from octopus.logger import get_logger
from octopus.models import Models
from octopus.modules.base import ModuleExecution, Task

logger = get_logger()

_RUNNING_IN_TESTSUITE = "RUNNING_IN_TESTSUITE" in os.environ


def _unique_unordered(seq):
    # assumes seq is an iterable of hashable items (strings here)
    return list(set(seq))


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

    models: list[str] = field(
        default=Factory(lambda: ["ExtraTreesClassifier"]),
        converter=_unique_unordered,
        validator=[
            validators.instance_of(list),
            validators.deep_iterable(
                member_validator=validators.instance_of(str),
                iterable_validator=validators.instance_of(list),
            ),
        ],
    )
    """Models for ML."""

    n_folds_inner: int = field(validator=[validators.instance_of(int)], default=Factory(lambda: 5))
    """Number of inner folds."""

    datasplit_seeds_inner: list[int] = field(
        default=Factory(lambda: [0]),
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(int),
            iterable_validator=validators.instance_of(list),
        ),
    )
    """List of integers used as seeds for data splitting."""

    model_seed: int = field(validator=[validators.instance_of(int)], default=Factory(lambda: 0))
    """Model seed."""

    n_jobs: int = field(validator=[validators.instance_of(int)], default=Factory(lambda: 1))
    """Number of CPUs used for every model training."""

    max_outl: int = field(validator=[validators.instance_of(int)], default=Factory(lambda: 3))
    """Maximum number of outliers, optimized by Optuna"""

    fi_methods_bestbag: list[str] = field(
        default=Factory(lambda: ["permutation"]),
        validator=validators.deep_iterable(
            member_validator=validators.in_(["permutation", "shap", "constant"]),
            iterable_validator=validators.instance_of(list),
        ),
    )
    """Feature importance methods for best bag."""

    inner_parallelization: bool = field(validator=[validators.instance_of(bool)], default=Factory(lambda: True))
    """Enable inner parallelization. Defaults is True."""

    n_workers: int = field(default=Factory(lambda: None))
    """Number of workers."""

    optuna_seed: int = field(validator=[validators.instance_of(int)], default=Factory(lambda: 0))
    """Seed for Optuna TPESampler, default=0"""

    n_optuna_startup_trials: int = field(validator=[validators.instance_of(int)], default=Factory(lambda: 10))
    """Number of Optuna startup trials (random sampler)"""

    ensemble_selection: bool = field(validator=[validators.in_([True, False])], default=Factory(lambda: False))
    """Whether to perform ensemble selection."""

    ensel_n_save_trials: int = field(validator=[validators.instance_of(int)], default=Factory(lambda: 50))
    """Number of top trials to be saved for ensemble selection (bags)."""

    n_trials: int = field(validator=[validators.instance_of(int)], default=100 if not _RUNNING_IN_TESTSUITE else 3)
    """Number of Optuna trials."""

    hyperparameters: dict = field(validator=[validators.instance_of(dict)], default=Factory(dict))
    """Bring own hyperparameter space."""

    max_features: int = field(validator=[validators.instance_of(int)], default=Factory(lambda: 0))
    """Maximum features to constrain hyperparameter optimization. Default is zero (off)."""

    penalty_factor: float = field(validator=[validators.instance_of(float)], default=Factory(lambda: 1.0))
    """Factor to penalize optuna target related to feature constraint."""

    mrmr_feature_numbers: list = field(validator=[validators.instance_of(list)], default=Factory(list))
    """List of feature numbers to be investigated by mrmr."""

    resume_optimization: bool = field(validator=[validators.instance_of(bool)], default=Factory(lambda: False))
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
        # (2) Only enforce constrained-HPO compatibility when max_features > 0
        if self.max_features > 0:
            incompatible_models: list[str] = []

            for m in self.models:
                try:
                    # Resolve model_config either by name (str) or by using get_model_config() on a class/object
                    if isinstance(m, str):
                        config = Models.get_config(m)
                    else:
                        get_cfg = getattr(m, "get_model_config", None)
                        if callable(get_cfg):
                            config = get_cfg()
                            if not getattr(config, "name", None):
                                config.name = getattr(m, "__name__", str(m))
                        else:
                            raise ValueError(
                                f"Model entry {m!r} is not a model name and does not provide get_model_config()"
                            )

                    chpo_flag = bool(getattr(config, "chpo_compatible", False))
                    # print/log chpo_compatible for each model
                    # Models.get_config() always sets name, safe to access
                    logger.info(f"Model '{config.name}': chpo_compatible={chpo_flag}")  # type: ignore[attr-defined]

                    if not chpo_flag:
                        incompatible_models.append(config.name)  # type: ignore[attr-defined]

                except Exception as exc:
                    logger.error(f"Could not retrieve model_config for model '{m}': {exc}")
                    # stop construction on resolution failures
                    raise ValueError(f"Could not retrieve model_config for model '{m}': {exc}") from exc

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
