"""Shared types for the Octopus framework."""

from enum import Enum, StrEnum, auto
from typing import Any

# ============================================================================
# CORE MACHINE LEARNING TYPES
# ============================================================================


class ResultType(StrEnum):
    """Types of results produced by modules."""

    BEST = "best"
    ENSEMBLE_SELECTION = "ensemble_selection"


class MLType(StrEnum):
    """Machine learning task types."""

    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"
    TIMETOEVENT = "timetoevent"


# ============================================================================
# LOGGING AND INFRASTRUCTURE
# ============================================================================


class LogGroup(Enum):
    """Create log groups for contextual logging."""

    DEFAULT = auto()
    DATA_PREPARATION = auto()
    DATA_HEALTH_REPORT = auto()
    CREATING_DATASPLITS = auto()
    PREPARE_EXECUTION = auto()
    PROCESSING = auto()
    OPTUNA = auto()
    TRAINING = auto()
    SCORES = auto()
    RESULTS = auto()
    AUTOGLUON = auto()


# ============================================================================
# STUDY AND DATA PROCESSING
# ============================================================================


class ImputationMethod(str, Enum):
    """Imputation methods for handling missing data."""

    MEDIAN = "median"
    HALFMIN = "halfmin"
    MICE = "mice"


# ============================================================================
# MODEL SELECTION AND CONFIGURATION
# ============================================================================


class ModelName(StrEnum):
    """Available model names for user-friendly model selection with IDE autocomplete.

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


# ============================================================================
# MODULE EXECUTION AND RESULTS
# ============================================================================


class ResultType(StrEnum):
    """Types of results produced by modules."""

    BEST = "best"
    ENSEMBLE_SELECTION = "ensemble_selection"


class FIResultLabel(StrEnum):
    """Labels used in feature-importance result DataFrames.

    Every module writes a ``fi_method`` column into its result DataFrame.
    Use these members as the column values so downstream code can filter
    and aggregate results reliably.
    """

    INTERNAL = "internal"
    PERMUTATION = "permutation"
    SHAP = "shap"
    LOFO = "lofo"
    CONSTANT = "constant"
    COUNTS = "counts"
    COUNTS_RELATIVE = "counts_relative"


class FIComputeMethod(StrEnum):
    """Computation methods for feature importance calculation.

    Used in model configuration (``ModelConfig.feature_method``), module
    configuration (``Octo.fi_methods_bestbag``, ``Rfe2.fi_method_rfe``,
    ``Mrmr.feature_importance_method``), and internal dispatch in bag
    and training code.
    """

    INTERNAL = "internal"
    PERMUTATION = "permutation"
    SHAP = "shap"
    LOFO = "lofo"
    CONSTANT = "constant"


class FIDataset(StrEnum):
    """Dataset partitions for feature importance computation."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


ML_TYPES = [e.value for e in MLType]


def to_ml_types_frozenset(val: list | set | tuple | frozenset) -> frozenset[MLType]:
    """Convert a collection of MLType to frozenset[MLType]."""
    if isinstance(val, str):
        raise TypeError("ml_types must be a list, set, or tuple, not a bare string. Use e.g. [MLType.REGRESSION].")
    return frozenset(val)


def validate_ml_types(instance: Any, attribute: Any, value: frozenset[MLType]) -> None:
    """Attrs validator: ml_types is non-empty and all members are MLType."""
    if not value:
        raise ValueError("ml_types must not be empty.")
    for v in value:
        if not isinstance(v, MLType):
            raise ValueError(f"Invalid ml_type: {v!r}. Must be an MLType enum member.")
