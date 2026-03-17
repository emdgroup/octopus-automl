"""ROC module (removal of correlated features)."""

from __future__ import annotations

from attrs import define, field, validators

from octopus.types import CorrelationType, ROCFilterMethod

from ..base import ModuleExecution, Task


@define
class Roc(Task):
    """ROC module for removing correlated features.

    This module identifies groups of correlated features and selects the most
    informative feature from each group, removing the rest. Uses correlation
    analysis (Spearman or RDC) combined with feature filtering (mutual information
    or F-statistics) to determine which features to keep.

    Configuration:
        threshold: Correlation threshold above which features are considered correlated
        correlation_type: Type of correlation measure (CorrelationType.SPEARMAN or CorrelationType.RDC)
        filter_type: Method to select best feature in group (ROCFilterMethod.MUTUAL_INFO or ROCFilterMethod.F_STATISTICS)
    """

    threshold: float = field(validator=[validators.instance_of(float)], default=0.8)
    """Threshold for feature removal (features with correlation > threshold are grouped)."""

    correlation_type: CorrelationType = field(
        converter=CorrelationType,
        validator=validators.in_([CorrelationType.SPEARMAN, CorrelationType.RDC]),
        default=CorrelationType.SPEARMAN,
    )
    """Selection of correlation type."""

    filter_type: ROCFilterMethod = field(
        converter=ROCFilterMethod,
        validator=validators.in_([ROCFilterMethod.MUTUAL_INFO, ROCFilterMethod.F_STATISTICS]),
        default=ROCFilterMethod.F_STATISTICS,
    )
    """Selection of filter type for correlated features."""

    def create_module(self) -> ModuleExecution:
        """Create RocModule execution instance."""
        # import only during execution to avoid heavy dependency at config stage
        from .core import RocModule  # noqa: PLC0415

        return RocModule(config=self)
