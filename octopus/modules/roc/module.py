# type: ignore

"""ROC module (removal of correlated features)."""

from __future__ import annotations

from attrs import define, field, validators

from octopus.modules.base import Task

from .core import RocModule


@define
class Roc(Task):
    """ROC module for removing correlated features.

    This module identifies groups of correlated features and selects the most
    informative feature from each group, removing the rest. Uses correlation
    analysis (Spearman or RDC) combined with feature filtering (mutual information
    or F-statistics) to determine which features to keep.

    Configuration:
        threshold: Correlation threshold above which features are considered correlated
        correlation_type: Type of correlation measure ("spearmanr" or "rdc")
        filter_type: Method to select best feature in group ("mutual_info" or "f_statistics")
    """

    threshold: float = field(validator=[validators.instance_of(float)], default=0.8)
    """Threshold for feature removal (features with correlation > threshold are grouped)."""

    correlation_type: str = field(validator=[validators.in_(["spearmanr", "rdc"])], default="spearmanr")
    """Selection of correlation type."""

    filter_type: str = field(
        validator=[validators.in_(["mutual_info", "f_statistics"])],
        default="f_statistics",
    )
    """Selection of filter type for correlated features."""

    def create_module(self) -> RocModule:
        """Create RocModule execution instance."""
        return RocModule(config=self)
