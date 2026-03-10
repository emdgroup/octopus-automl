"""Regression models."""

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ARDRegression, ElasticNet, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor

from octopus.types import MLType

from .config import ModelConfig
from .core import Models
from .hyperparameter import CategoricalHyperparameter, FixedHyperparameter, FloatHyperparameter, IntHyperparameter
from .wrapper_models.GaussianProcessRegressor import GPRegressorWrapper


@Models.register("ARDRegressor")
def ard_regressor() -> ModelConfig:
    """ARD regression model class."""
    return ModelConfig(
        model_class=ARDRegression,  # type: ignore[arg-type]
        ml_types=[MLType.REGRESSION],
        feature_method="permutation",
        n_repeats=2,
        chpo_compatible=False,
        scaler="StandardScaler",
        imputation_required=True,
        categorical_enabled=False,
        hyperparameters=[
            FloatHyperparameter(name="alpha_1", low=1e-10, high=1e-3, log=True),
            FloatHyperparameter(name="alpha_2", low=1e-10, high=1e-3, log=True),
            FloatHyperparameter(name="lambda_1", low=1e-10, high=1e-3, log=True),
            FloatHyperparameter(name="lambda_2", low=1e-10, high=1e-3, log=True),
            FloatHyperparameter(name="threshold_lambda", low=1e3, high=1e5, log=True),
            FloatHyperparameter(name="tol", low=1e-5, high=1e-1, log=True),
            FixedHyperparameter(name="fit_intercept", value=True),
        ],
        n_jobs=None,
        model_seed=None,
    )


@Models.register("CatBoostRegressor")
def catboost_regressor() -> ModelConfig:
    """Cat boost regression model class."""
    return ModelConfig(
        model_class=CatBoostRegressor,
        ml_types=[MLType.REGRESSION],
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
            FixedHyperparameter(name="thread_count", value=1),
            FixedHyperparameter(name="task_type", value="CPU"),
        ],
        n_jobs="thread_count",
        model_seed="random_state",
    )


@Models.register("ElasticNetRegressor")
def elastic_net_regressor() -> ModelConfig:
    """ElasticNet regression model class."""
    return ModelConfig(
        model_class=ElasticNet,  # type: ignore[arg-type]
        ml_types=[MLType.REGRESSION],
        feature_method="shap",
        chpo_compatible=True,
        scaler="StandardScaler",
        imputation_required=True,
        categorical_enabled=False,
        default=True,
        hyperparameters=[
            FloatHyperparameter(name="alpha", low=1e-10, high=1e2, log=True),
            FloatHyperparameter(name="l1_ratio", low=0, high=1, log=False),
            CategoricalHyperparameter(name="fit_intercept", choices=[True, False]),
            FloatHyperparameter(name="tol", low=1e-5, high=1e-1, log=True),
            FixedHyperparameter(name="max_iter", value=4000),
            FixedHyperparameter(name="selection", value="random"),
        ],
        n_jobs=None,
        model_seed="random_state",
    )


@Models.register("ExtraTreesRegressor")
def extra_trees_regressor() -> ModelConfig:
    """ExtraTrees regression model class."""
    return ModelConfig(
        model_class=ExtraTreesRegressor,  # type: ignore[arg-type]
        ml_types=[MLType.REGRESSION],
        feature_method="internal",
        chpo_compatible=True,
        scaler=None,
        imputation_required=True,
        categorical_enabled=False,
        default=True,
        hyperparameters=[
            IntHyperparameter(name="max_depth", low=2, high=32),
            IntHyperparameter(name="min_samples_split", low=2, high=100),
            IntHyperparameter(name="min_samples_leaf", low=1, high=50),
            IntHyperparameter(name="n_estimators", low=100, high=500, log=False),
            FloatHyperparameter(name="max_features", low=0.1, high=1),
        ],
        n_jobs="n_jobs",
        model_seed="random_state",
    )


@Models.register("GaussianProcessRegressor")
def gaussian_process_regressor() -> ModelConfig:
    """Gaussian process regression model class."""
    return ModelConfig(
        model_class=GPRegressorWrapper,  # type: ignore[arg-type]
        ml_types=[MLType.REGRESSION],
        feature_method="permutation",
        n_repeats=2,
        chpo_compatible=False,
        scaler="StandardScaler",
        imputation_required=True,
        categorical_enabled=False,
        hyperparameters=[
            CategoricalHyperparameter(name="kernel", choices=["RBF", "Matern", "RationalQuadratic"]),
            FloatHyperparameter(name="alpha", low=1e-10, high=1e-1, log=True),
            FloatHyperparameter(name="alpha", low=1e-10, high=1e-1, log=True),
            CategoricalHyperparameter(name="normalize_y", choices=[True, False]),
            CategoricalHyperparameter(name="optimizer", choices=["fmin_l_bfgs_b", None]),
            IntHyperparameter(name="n_restarts_optimizer", low=0, high=10, log=False),
        ],
        n_jobs=None,
        model_seed="random_state",
    )


@Models.register("GradientBoostingRegressor")
def gradient_boosting_regressor() -> ModelConfig:
    """Gradient boost regression model class."""
    return ModelConfig(
        model_class=GradientBoostingRegressor,  # type: ignore[arg-type]
        ml_types=[MLType.REGRESSION],
        feature_method="internal",
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
            FixedHyperparameter(name="loss", value="squared_error"),
        ],
        n_jobs=None,
        model_seed="random_state",
    )


@Models.register("RandomForestRegressor")
def random_forest_regressor() -> ModelConfig:
    """Random forrest regression model class."""
    return ModelConfig(
        model_class=RandomForestRegressor,  # type: ignore[arg-type]
        ml_types=[MLType.REGRESSION],
        feature_method="internal",
        chpo_compatible=True,
        scaler=None,
        imputation_required=True,  # maybe: False -- check!
        categorical_enabled=False,
        default=True,
        hyperparameters=[
            IntHyperparameter(name="max_depth", low=2, high=32),
            IntHyperparameter(name="min_samples_split", low=2, high=100),
            IntHyperparameter(name="min_samples_leaf", low=1, high=50),
            IntHyperparameter(name="n_estimators", low=100, high=500),
            FloatHyperparameter(name="max_features", low=0.1, high=1),
        ],
        n_jobs="n_jobs",
        model_seed="random_state",
    )


@Models.register("RidgeRegressor")
def ridge_regressor() -> ModelConfig:
    """Ridge regression model class."""
    return ModelConfig(
        model_class=Ridge,  # type: ignore[arg-type]
        ml_types=[MLType.REGRESSION],
        feature_method="shap",
        chpo_compatible=False,
        scaler="StandardScaler",
        imputation_required=True,
        categorical_enabled=False,
        hyperparameters=[
            FloatHyperparameter(name="alpha", low=1e-5, high=1e5, log=True),
            CategoricalHyperparameter(name="fit_intercept", choices=[True, False]),
            FixedHyperparameter(name="solver", value="svd"),
        ],
        n_jobs=None,
        model_seed="random_state",
    )


@Models.register("SvrRegressor")
def svr_regressor() -> ModelConfig:
    """Svr regression model class."""
    return ModelConfig(
        model_class=SVR,  # type: ignore[arg-type]
        ml_types=[MLType.REGRESSION],
        feature_method="permutation",
        n_repeats=2,
        chpo_compatible=False,
        scaler="StandardScaler",
        imputation_required=True,
        categorical_enabled=False,
        hyperparameters=[
            FloatHyperparameter(name="C", low=0.03125, high=32768, log=True),
            FloatHyperparameter(name="epsilon", low=0.001, high=1, log=True),
            FloatHyperparameter(name="tol", low=1e-5, high=1e-1, log=True),
        ],
        n_jobs=None,
        model_seed=None,
    )


@Models.register("XGBRegressor")
def xgb_regressor() -> ModelConfig:
    """XGBoost regression model class."""
    return ModelConfig(
        model_class=XGBRegressor,
        ml_types=[MLType.REGRESSION],
        feature_method="internal",
        chpo_compatible=True,
        scaler=None,
        imputation_required=False,
        categorical_enabled=False,  # maybe:True -- check!
        default=True,
        hyperparameters=[
            FloatHyperparameter(name="learning_rate", low=1e-4, high=0.3, log=True),
            IntHyperparameter(name="min_child_weight", low=2, high=15),
            FloatHyperparameter(name="subsample", low=0.15, high=1.0),
            IntHyperparameter(name="n_estimators", low=30, high=500),
            IntHyperparameter(name="max_depth", low=3, high=9, step=2),
            FixedHyperparameter(name="validate_parameters", value=True),
            FloatHyperparameter(name="lambda", low=1e-8, high=1, log=True),
        ],
        n_jobs="n_jobs",
        model_seed="random_state",
    )


@Models.register("HistGradientBoostingRegressor")
def hist_gradient_boosting_regressor() -> ModelConfig:
    """Histogram-based gradient boosting regression model class (scikit-learn 1.6.1)."""
    return ModelConfig(
        model_class=HistGradientBoostingRegressor,  # type: ignore[arg-type]
        ml_types=[MLType.REGRESSION],
        feature_method="internal",
        chpo_compatible=True,
        scaler=None,
        imputation_required=False,
        categorical_enabled=True,
        default=True,
        hyperparameters=[
            FloatHyperparameter(name="learning_rate", low=0.01, high=0.3, log=True),
            IntHyperparameter(name="max_iter", low=50, high=1000),
            IntHyperparameter(name="max_leaf_nodes", low=7, high=256),
            FloatHyperparameter(name="l2_regularization", low=1e-6, high=10.0, log=True),
            IntHyperparameter(name="min_samples_leaf", low=1, high=200),
            IntHyperparameter(name="max_bins", low=16, high=255),
            FixedHyperparameter(name="loss", value="squared_error"),
        ],
        n_jobs=None,
        model_seed="random_state",
    )


@Models.register("TabularNNRegressor")
def tabular_nn_regressor() -> ModelConfig:
    """Tabular Neural Network regression model class with categorical embeddings."""
    from .wrapper_models.TabularNNRegressor import TabularNNRegressor  # noqa: PLC0415

    return ModelConfig(
        model_class=TabularNNRegressor,  # type: ignore[arg-type]
        ml_types=[MLType.REGRESSION],
        feature_method="permutation",
        n_repeats=2,
        chpo_compatible=False,
        scaler="StandardScaler",
        imputation_required=False,
        categorical_enabled=True,
        hyperparameters=[
            CategoricalHyperparameter(
                name="hidden_sizes",
                choices=[
                    [512, 256, 128],
                    [512, 256],
                    [512, 128],
                    [256, 256, 128],
                    [256, 128, 64],
                    [256, 128],
                    [256, 64],
                    [128, 128, 64],
                    [128, 64],
                    [128, 32],
                ],
            ),
            FloatHyperparameter(name="dropout", low=0.0, high=0.5),
            FloatHyperparameter(name="learning_rate", low=1e-5, high=1e-2, log=True),
            FixedHyperparameter(name="weight_decay", value=1e-5),
            FixedHyperparameter(name="activation", value="elu"),
            FixedHyperparameter(name="optimizer", value="adamw"),
            CategoricalHyperparameter(name="batch_size", choices=[32, 64, 128, 256]),
            FixedHyperparameter(name="epochs", value=200),
        ],
        n_jobs=None,
        model_seed="random_state",
    )
