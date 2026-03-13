"""Model name enum for user-friendly model selection with IDE autocomplete."""

from enum import StrEnum


class ModelName(StrEnum):
    """Available model names.

    Use this enum for IDE autocomplete when specifying models, e.g.::

        Octo(task_id=0, models=[ModelName.XGBClassifier, ModelName.CatBoostClassifier])

    Plain strings still work too::

        Octo(task_id=0, models=["XGBClassifier", "CatBoostClassifier"])
    """

    # Classification models
    ExtraTreesClassifier = "ExtraTreesClassifier"
    HistGradientBoostingClassifier = "HistGradientBoostingClassifier"
    GradientBoostingClassifier = "GradientBoostingClassifier"
    RandomForestClassifier = "RandomForestClassifier"
    XGBClassifier = "XGBClassifier"
    CatBoostClassifier = "CatBoostClassifier"
    LogisticRegressionClassifier = "LogisticRegressionClassifier"
    GaussianProcessClassifier = "GaussianProcessClassifier"

    # Regression models
    ARDRegressor = "ARDRegressor"
    CatBoostRegressor = "CatBoostRegressor"
    ElasticNetRegressor = "ElasticNetRegressor"
    ExtraTreesRegressor = "ExtraTreesRegressor"
    GaussianProcessRegressor = "GaussianProcessRegressor"
    GradientBoostingRegressor = "GradientBoostingRegressor"
    RandomForestRegressor = "RandomForestRegressor"
    RidgeRegressor = "RidgeRegressor"
    SvrRegressor = "SvrRegressor"
    XGBRegressor = "XGBRegressor"
    HistGradientBoostingRegressor = "HistGradientBoostingRegressor"
    TabularNNRegressor = "TabularNNRegressor"

    # Time-to-event (survival) models
    CatBoostCoxSurvival = "CatBoostCoxSurvival"
    XGBoostCoxSurvival = "XGBoostCoxSurvival"
