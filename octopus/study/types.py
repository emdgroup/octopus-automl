"""Study types."""

from enum import Enum


class ImputationMethod(str, Enum):
    """Imputation methods."""

    MEDIAN = "median"
    HALFMIN = "halfmin"
    MICE = "mice"
