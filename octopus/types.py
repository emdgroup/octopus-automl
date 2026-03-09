"""Shared types for the Octopus framework."""

from enum import Enum
from typing import Any

# TODO: Group in a reasonable way.


class MLType(str, Enum):
    """Machine learning task types."""

    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"
    TIMETOEVENT = "timetoevent"


# TODO: Duplicate with octopus.modules.base.FIMethod, but this is used in module configuration while FIMethod is for results, so we may want to keep them separate for clarity and to avoid circular imports. Consider refactoring later to unify if it doesn't cause issues.
# TODO: Usedn in RFE2 Module.
# TODO: ued in octo module, but only subset.
class FeatureImportanceType(str, Enum):
    """Feature importance calculation methods."""

    PERMUTATION = "permutation"
    GROUP_PERMUTATION = "group_permutation"
    SHAP = "shap"
    GROUP_SHAP = "group_shap"
    INTERNAL = "internal"
    LOFO = "lofo"
    CONSTANT = "constant"
    PERMUTATION_DEV = "permutation_dev"


class ShapType(str, Enum):
    """SHAP explainer types."""

    KERNEL = "kernel"
    PERMUTATION = "permutation"
    EXACT = "exact"


class FeatureImportanceMethod(str, Enum):
    """Feature importance aggregation methods."""

    MEAN = "mean"
    COUNT = "count"


class CorrelationType(str, Enum):
    """Correlation calculation methods."""

    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    RDC = "rdc"


class MRMRRelevance(str, Enum):
    """Relevance calculation methods for MRMR feature selection."""

    PERMUTATION = "permutation"
    F_STATISTICS = "f-statistics"


class RocFilterMethod(str, Enum):
    """Filter methods for ROC feature selection."""

    MUTUAL_INFO = "mutual_info"
    F_STATISTICS = "f_statistics"


class SFSDirection(str, Enum):
    """Sequential Feature Selection directions."""

    FORWARD = "forward"
    BACKWARD = "backward"
    FLOATING_FORWARD = "floating_forward"
    FLOATING_BACKWARD = "floating_backward"


class AutoGluonFitStrategy(str, Enum):
    """AutoGluon fitting strategies."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class RFE2SelectionMethod(str, Enum):
    """Feature selection method for RFE2."""

    BEST = "best"
    PARSIMONIOUS = "parsimonious"


class OptunaReturnType(str, Enum):
    """Optuna return types."""

    POOL = "pool"
    AVERAGE = "average"

class RFEMode(str, Enum):
    """Recursive Feature Elimination modes."""
    MODE1 = "Mode1"
    MODE2 = "Mode2"


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
