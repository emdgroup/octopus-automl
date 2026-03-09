"""Study types."""

from enum import Enum


class ImputationMethod(str, Enum):
    """Imputation methods."""

    MEDIAN = "median"
    HALFMIN = "halfmin"
    MICE = "mice"


class DatasplitType(str, Enum):
    """Datasplit types."""

    SAMPLE = "sample"
    GROUP_FEATURES = "group_features"
    GROUP_SAMPLE_AND_FEATURES = "group_sample_and_features"
