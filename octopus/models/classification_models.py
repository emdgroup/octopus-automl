"""Classification models."""

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from octopus.types import FIComputeMethod, MLType, ModelName

from .config import ModelConfig
from .core import Models
from .hyperparameter import CategoricalHyperparameter, FixedHyperparameter, FloatHyperparameter, IntHyperparameter
from .wrapper_models.GaussianProcessClassifier import GPClassifierWrapper


@Models.register(ModelName.ExtraTreesClassifier)
def extra_trees_classifier() -> ModelConfig:
    """ExtraTrees classification model config."""
    return ModelConfig(
        model_class=ExtraTreesClassifier,  # type: ignore[arg-type]
        ml_types=[MLType.BINARY, MLType.MULTICLASS],
        feature_method=FIComputeMethod.INTERNAL,
        chpo_compatible=True,
        scaler=None,
        imputation_required=True,
        categorical_enabled=False,
        default=True,
        hyperparameters=[
            IntHyperparameter(name="max_depth", low=2, high=32),
            IntHyperparameter(name="min_samples_split", low=2, high=100),
            IntHyperparameter(name="min_samples_leaf", low=1, high=50),
            FloatHyperparameter(name="max_features", low=0.1, high=1),
            IntHyperparameter(name="n_estimators", low=100, high=500, log=False),
            CategoricalHyperparameter(name="class_weight", choices=[None, "balanced"]),
            FixedHyperparameter(name="criterion", value="entropy"),
            FixedHyperparameter(name="bootstrap", value=True),
        ],
        n_jobs="n_jobs",
        model_seed="random_state",
    )


@Models.register(ModelName.HistGradientBoostingClassifier)
def hist_gradient_boosting_classifier() -> ModelConfig:
    """Histogram-based gradient boosting classification model config (scikit-learn 1.6.1)."""
    return ModelConfig(
        model_class=HistGradientBoostingClassifier,  # type: ignore[arg-type]
        ml_types=[MLType.BINARY, MLType.MULTICLASS],
        feature_method=FIComputeMethod.INTERNAL,
        chpo_compatible=True,
        scaler=None,
        imputation_required=False,
        categorical_enabled=True,
        default=True,
        hyperparameters=[
            FloatHyperparameter(name="learning_rate", low=0.01, high=0.3, log=True),
            IntHyperparameter(name="max_iter", low=50, high=1000),
            IntHyperparameter(name="max_leaf_nodes", low=7, high=256),
            IntHyperparameter(name="min_samples_leaf", low=1, high=200),
            IntHyperparameter(name="max_bins", low=16, high=255),
            FloatHyperparameter(name="l2_regularization", low=0.0, high=10.0, log=False),
            FixedHyperparameter(name="loss", value="log_loss"),
        ],
        # HistGradientBoostingClassifier uses `random_state` for seeding (map model_seed -> "random_state")
        n_jobs=None,
        model_seed="random_state",
    )


@Models.register(ModelName.GradientBoostingClassifier)
def gradient_boosting_classifier() -> ModelConfig:
    """Gradient boosting classification model config."""
    return ModelConfig(
        model_class=GradientBoostingClassifier,  # type: ignore[arg-type]
        ml_types=[MLType.BINARY, MLType.MULTICLASS],
        feature_method=FIComputeMethod.INTERNAL,
        chpo_compatible=True,
        scaler=None,
        imputation_required=True,
        categorical_enabled=False,
        hyperparameters=[
            FloatHyperparameter(name="learning_rate", low=0.01, high=1, log=True),
            IntHyperparameter(name="min_samples_leaf", low=1, high=200),
            IntHyperparameter(name="max_leaf_nodes", low=3, high=2047),
            IntHyperparameter(name="max_depth", low=3, high=9, step=2),
            IntHyperparameter(name="n_estimators", low=30, high=500),
            FloatHyperparameter(name="max_features", low=0.1, high=1),
            FixedHyperparameter(name="loss", value="log_loss"),
        ],
        n_jobs=None,
        model_seed="random_state",
    )


@Models.register(ModelName.RandomForestClassifier)
def random_forest_classifier() -> ModelConfig:
    """Random forest classification model config."""
    return ModelConfig(
        model_class=RandomForestClassifier,  # type: ignore[arg-type]
        ml_types=[MLType.BINARY, MLType.MULTICLASS],
        feature_method=FIComputeMethod.INTERNAL,
        chpo_compatible=True,
        scaler=None,
        imputation_required=True,  # maybe: False - check!
        categorical_enabled=False,
        default=True,
        hyperparameters=[
            IntHyperparameter(name="max_depth", low=2, high=32),
            IntHyperparameter(name="min_samples_split", low=2, high=100),
            IntHyperparameter(name="min_samples_leaf", low=1, high=50),
            FloatHyperparameter(name="max_features", low=0.1, high=1),
            IntHyperparameter(name="n_estimators", low=100, high=500, log=False),
            CategoricalHyperparameter(name="class_weight", choices=[None, "balanced"]),
        ],
        n_jobs="n_jobs",
        model_seed="random_state",
    )


@Models.register(ModelName.XGBClassifier)
def xgb_classifier() -> ModelConfig:
    """XGBoost classification model config."""
    return ModelConfig(
        model_class=XGBClassifier,
        ml_types=[MLType.BINARY, MLType.MULTICLASS],
        feature_method=FIComputeMethod.INTERNAL,
        chpo_compatible=True,
        scaler=None,
        imputation_required=False,
        categorical_enabled=False,  # Maybe True - check!
        default=True,
        hyperparameters=[
            FloatHyperparameter(name="learning_rate", low=1e-4, high=0.3, log=True),
            IntHyperparameter(name="min_child_weight", low=2, high=15),
            FloatHyperparameter(name="subsample", low=0.15, high=1.0),
            IntHyperparameter(name="n_estimators", low=30, high=200),
            IntHyperparameter(name="max_depth", low=3, high=9, step=2),
            FixedHyperparameter(name="validate_parameters", value=True),
        ],
        n_jobs="n_jobs",
        model_seed="random_state",
    )


@Models.register(ModelName.CatBoostClassifier)
def catboost_classifier() -> ModelConfig:
    """CatBoost classification model config."""
    return ModelConfig(
        model_class=CatBoostClassifier,
        # Multiclass excluded: SHAP explainers segfault on CatBoost multiclass models.
        ml_types=[MLType.BINARY],
        feature_method=FIComputeMethod.INTERNAL,
        chpo_compatible=True,
        scaler=None,
        imputation_required=False,
        categorical_enabled=True,
        default=True,
        hyperparameters=[
            FloatHyperparameter(name="learning_rate", low=1e-2, high=1e-1, log=True),
            IntHyperparameter(name="depth", low=3, high=10),
            FloatHyperparameter(name="l2_leaf_reg", low=2, high=10),
            FloatHyperparameter(name="random_strength", low=2, high=10),
            FloatHyperparameter(name="rsm", low=0.1, high=1),
            FixedHyperparameter(name="iterations", value=1000),
            CategoricalHyperparameter(name="auto_class_weights", choices=[None, "Balanced"]),
            FixedHyperparameter(name="allow_writing_files", value=False),
            FixedHyperparameter(name="logging_level", value="Silent"),
            FixedHyperparameter(name="thread_count", value=1),
            FixedHyperparameter(name="task_type", value="CPU"),
        ],
        n_jobs="thread_count",
        model_seed="random_state",
    )


@Models.register(ModelName.LogisticRegressionClassifier)
def logistic_regression_classifier() -> ModelConfig:
    """Logistic regression classification model config."""
    return ModelConfig(
        model_class=LogisticRegression,  # type: ignore[arg-type]
        ml_types=[MLType.BINARY, MLType.MULTICLASS],
        feature_method=FIComputeMethod.PERMUTATION,
        n_repeats=2,
        chpo_compatible=True,
        scaler="StandardScaler",
        imputation_required=True,
        categorical_enabled=False,
        default=True,
        hyperparameters=[
            IntHyperparameter(name="max_iter", low=100, high=500),
            FloatHyperparameter(name="C", low=1e-2, high=100, log=True),
            FloatHyperparameter(name="tol", low=1e-4, high=1e-2, log=True),
            CategoricalHyperparameter(name="penalty", choices=["l2", None]),
            CategoricalHyperparameter(name="fit_intercept", choices=[True, False]),
            CategoricalHyperparameter(name="class_weight", choices=[None, "balanced"]),
            FixedHyperparameter(name="solver", value="lbfgs"),
        ],
        n_jobs="n_jobs",
        model_seed="random_state",
    )


@Models.register(ModelName.GaussianProcessClassifier)
def gaussian_process_classifier() -> ModelConfig:
    """Gaussian process classification model config."""
    return ModelConfig(
        model_class=GPClassifierWrapper,  # type: ignore[arg-type]
        ml_types=[MLType.BINARY, MLType.MULTICLASS],
        feature_method=FIComputeMethod.PERMUTATION,
        n_repeats=2,
        chpo_compatible=False,
        scaler="StandardScaler",
        imputation_required=True,
        categorical_enabled=False,
        hyperparameters=[
            CategoricalHyperparameter(name="kernel", choices=["RBF", "Matern", "RationalQuadratic"]),
            CategoricalHyperparameter(name="optimizer", choices=["fmin_l_bfgs_b", None]),
            IntHyperparameter(name="n_restarts_optimizer", low=0, high=10, log=False),
            IntHyperparameter(name="max_iter_predict", low=50, high=200, log=False),
        ],
        n_jobs=None,
        model_seed="random_state",
    )
