"""Shared types for the Octopus framework."""

from enum import Enum, StrEnum, auto
from typing import Any

import numpy as np
import pandas as pd

type OctoArrayLike = np.typing.ArrayLike
type OctoMatrixLike = np.ndarray | pd.DataFrame


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


class ModelName(StrEnum):
    """Available model names.

    Use this enum for IDE autocomplete when specifying models, e.g.::

        Octo(task_id=0, models=[ModelName.XGBClassifier, ModelName.CatBoostClassifier])

    Plain strings still work too, but `ModelName` keeps call sites consistent.
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


class DataPartition(StrEnum):
    """Dataset partitions for feature importance computation."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class FIType(StrEnum):
    """Feature importance types for the prediction API.

    Used as the ``fi_type`` parameter in
    ``TaskPredictor.calculate_fi()`` and
    ``TaskPredictorTest.calculate_fi()``.
    """

    PERMUTATION = "permutation"
    GROUP_PERMUTATION = "group_permutation"
    SHAP = "shap"


class ShapType(StrEnum):
    """SHAP explainer implementations.

    Selects which SHAP explainer algorithm to use when computing
    SHAP-based feature importances via ``shap_type`` parameters
    in training and prediction code.
    """

    AUTO = "auto"
    KERNEL = "kernel"
    PERMUTATION = "permutation"
    EXACT = "exact"


class CorrelationType(StrEnum):
    """Correlation method used for measuring feature-feature or feature-target relationships.

    Used in:
    - ``Mrmr.correlation_type``: redundancy measure between features
    - ``Roc.correlation_type``: grouping correlated features for removal
    - ``HealthChecker._check_feature_feature_correlation``: data health report
    """

    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    RDC = "rdc"


class MRMRRelevance(StrEnum):
    """Method used to compute feature relevance to the target in MRMR.

    Used in ``Mrmr.relevance_type``:
    - ``PERMUTATION``: re-uses permutation importances from a prior workflow module
    - ``F_STATISTICS``: computes F-statistics (f_classif / f_regression) from scratch
    """

    PERMUTATION = "permutation"
    F_STATISTICS = "f-statistics"


class AutoGluonFitStrategy(StrEnum):
    """Controls whether AutoGluon trains models sequentially or in parallel.

    Used in ``AutoGluon.fit_strategy``, passed directly to ``TabularPredictor.fit()``.
    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class RFE2SelectionMethod(StrEnum):
    """Strategy for picking the final feature set from the RFE2 scan results.

    Used in ``Rfe2.selection_method``:
    - ``BEST``: selects the step with the highest mean cross-validation performance
    - ``PARSIMONIOUS``: selects the smallest feature set whose performance is within
      one standard error of the best (bias-variance trade-off)
    """

    BEST = "best"
    PARSIMONIOUS = "parsimonious"


class ROCFilterMethod(StrEnum):
    """Scoring method used to rank features within a correlated group in the ROC module.

    The highest-scoring feature in each group is kept; the rest are removed.
    Used in ``Roc.filter_type``.
    """

    MUTUAL_INFO = "mutual_info"
    F_STATISTICS = "f_statistics"


class SFSDirection(StrEnum):
    """Search direction for Sequential Feature Selection (mlxtend SequentialFeatureSelector).

    Used in ``Sfs.sfs_type``. Forward variants add features; backward variants remove them.
    Floating variants allow backtracking after each add/remove step.
    """

    FORWARD = "forward"
    BACKWARD = "backward"
    FLOATING_FORWARD = "floating_forward"
    FLOATING_BACKWARD = "floating_backward"


class RFEMode(StrEnum):
    """Execution mode for the RFE module.

    Used in ``Rfe.mode``:
    - ``FIXED``: runs RFE with the already-optimised model (faster)
    - ``REFIT``: re-optimises the model at each elimination step (slower, potentially better)
    """

    FIXED = "fixed"
    REFIT = "refit"


class OptunaReturnType(StrEnum):
    """Determines which bag performance statistic is used as the Optuna optimisation target.

    Used in ``Octo.optuna_return``:
    - ``POOL``: uses the pooled dev score across all inner folds (dev_pool)
    - ``AVERAGE``: uses the average of per-fold dev scores (dev_avg)
    """

    POOL = "pool"
    AVERAGE = "average"


class MetricDirection(StrEnum):
    """Optimisation direction passed to Optuna and used for sorting scan/ensemble results.

    Derived from ``Metric.higher_is_better`` via ``Metric.direction`` and ``Metrics.get_direction()``.
    Used in EFS scan/ensemble optimisation and diagnostics plots.
    """

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class PredictionType(StrEnum):
    """The format in which a metric expects its predictions.

    - ``PREDICTIONS``: the metric receives hard class labels (0/1 for binary,
      integer class indices for multiclass, continuous values for regression).
      For binary classification, these are derived by thresholding
      ``predict_proba`` output — ``predict`` is not called directly.
    - ``PROBABILITIES``: the metric receives raw probability scores or
      continuous outputs directly from ``predict_proba``.

    Used in:
    - ``Metric.prediction_type``: declared per metric in the registry
    - ``metrics/utils.py``: used to prepare the correct input before calling
      the metric function
    - ``EfsModule``: selects the correct column from the CV predictions table
    """

    PREDICTIONS = "predictions"
    PROBABILITIES = "probabilities"


class MRMRFIAggregation(StrEnum):
    """How per-training feature importances are aggregated before MRMR relevance scoring.

    Used in ``Mrmr.feature_importance_type``:
    - ``MEAN``: averages importance values across training runs
    - ``COUNT``: counts how often a feature has non-zero importance
    """

    MEAN = "mean"
    COUNT = "count"


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
