"""ROC module (removal of correlated features)."""

from __future__ import annotations

from attrs import define, field, validators

from octopus.types import CorrelationType, RelevanceMethod

from ..base import ModuleExecution, Task


@define
class Roc(Task):
    """ROC module for removing correlated features.

    This module identifies groups of correlated features and selects the most
    informative feature from each group, removing the rest. Uses correlation
    analysis (Spearman or RDC) combined with feature relevance scoring (mutual
    information or F-statistics) to determine which features to keep.

    Configuration:
        correlation_threshold: Correlation threshold above which features are considered correlated
        correlation_type: Type of correlation measure (CorrelationType.SPEARMAN or CorrelationType.RDC)
        relevance_method: Method to select best feature in group (RelevanceMethod.MUTUAL_INFO or RelevanceMethod.F_STATISTICS)
    """

    correlation_threshold: float = field(validator=[validators.instance_of(float)], default=0.8)
    """Correlation threshold for feature removal (features with correlation > threshold are grouped)."""

    correlation_type: CorrelationType = field(
        converter=CorrelationType,
        validator=validators.in_([CorrelationType.SPEARMAN, CorrelationType.RDC]),
        default=CorrelationType.SPEARMAN,
    )
    """Selection of correlation type."""

    relevance_method: RelevanceMethod = field(
        converter=RelevanceMethod,
        validator=validators.in_([RelevanceMethod.MUTUAL_INFO, RelevanceMethod.F_STATISTICS]),
        default=RelevanceMethod.F_STATISTICS,
    )
    """Method to score feature relevance within correlated groups."""

    def create_module(self) -> ModuleExecution:
        """Create RocModule execution instance."""
        # import only during execution to avoid heavy dependency at config stage
        from .core import RocModule  # noqa: PLC0415

        return RocModule(config=self)
