"""StudyContext - frozen runtime configuration for modules."""

from attrs import frozen
from upath import UPath

from octopus.types import MLType


@frozen
class StudyContext:
    """Immutable runtime context passed to modules during fit().

    Contains only the finalized/prepared values needed by modules.
    No OctoStudy dependency - only attrs + upath.
    """

    ml_type: MLType
    """MLType enum (e.g. MLType.BINARY, MLType.REGRESSION, MLType.TIMETOEVENT)."""

    target_metric: str
    """Primary metric for model evaluation."""

    target_assignments: dict[str, str]
    """Target column assignments (e.g. {'default': 'target'} or {'duration': ..., 'event': ...})."""

    positive_class: int | None
    """Positive class label for binary classification. None for regression/multiclass."""

    stratification_col: str | None
    """Column used for stratification during data splitting."""

    sample_id_col: str
    """Identifier for sample instances."""

    feature_cols: list[str]
    """Prepared feature columns (from PreparedData.feature_cols)."""

    row_id_col: str
    """Prepared row identifier (from PreparedData.row_id_col)."""

    output_path: UPath
    """Full output path for this study."""

    log_dir: UPath
    """Directory where logs are stored."""
