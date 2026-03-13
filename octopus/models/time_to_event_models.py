"""Time to event models."""

from octopus.types import MLType

from .config import ModelConfig
from .core import Models
from .hyperparameter import FixedHyperparameter, FloatHyperparameter, IntHyperparameter
from .wrapper_models.CatBoostCoxSurvival import CatBoostCoxSurvival
from .wrapper_models.XGBoostCoxSurvival import XGBoostCoxSurvival


@Models.register("CatBoostCoxSurvival")
def catboost_cox_survival() -> ModelConfig:
    """CatBoost Cox survival model config."""
    return ModelConfig(
        model_class=CatBoostCoxSurvival,  # type: ignore[arg-type]
        ml_types=[MLType.TIMETOEVENT],
        feature_method="internal",
        chpo_compatible=True,
        scaler=None,
        imputation_required=False,
        categorical_enabled=True,
        default=True,
        hyperparameters=[
            FloatHyperparameter(name="learning_rate", low=1e-3, high=1e-1, log=True),
            IntHyperparameter(name="depth", low=3, high=10),
            FloatHyperparameter(name="l2_leaf_reg", low=2, high=10),
            FloatHyperparameter(name="random_strength", low=2, high=10),
            FloatHyperparameter(name="rsm", low=0.1, high=1),
            FixedHyperparameter(name="iterations", value=500),
            FixedHyperparameter(name="allow_writing_files", value=False),
            FixedHyperparameter(name="logging_level", value="Silent"),
            FixedHyperparameter(name="task_type", value="CPU"),
        ],
        n_jobs="thread_count",
        model_seed="random_state",
    )


@Models.register("XGBoostCoxSurvival")
def xgboost_cox_survival() -> ModelConfig:
    """XGBoost Cox survival model config."""
    return ModelConfig(
        model_class=XGBoostCoxSurvival,  # type: ignore[arg-type]
        ml_types=[MLType.TIMETOEVENT],
        feature_method="internal",
        chpo_compatible=True,
        scaler=None,
        imputation_required=False,
        categorical_enabled=False,
        default=True,
        hyperparameters=[
            FloatHyperparameter(name="learning_rate", low=1e-4, high=0.3, log=True),
            IntHyperparameter(name="min_child_weight", low=2, high=15),
            FloatHyperparameter(name="subsample", low=0.15, high=1.0),
            IntHyperparameter(name="n_estimators", low=30, high=500),
            IntHyperparameter(name="max_depth", low=3, high=9, step=1),
            FixedHyperparameter(name="validate_parameters", value=True),
        ],
        n_jobs="n_jobs",
        model_seed="random_state",
    )
