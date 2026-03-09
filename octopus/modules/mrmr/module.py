"""MRMR module (Minimum Redundancy Maximum Relevance feature selection)."""

from __future__ import annotations

from typing import Literal

from attrs import Factory, define, field, validators

from octopus.modules.base import ModuleExecution, Task
from octopus.types import FeatureImportanceType


@define
class Mrmr(Task):
    """MRMR module for feature selection based on mutual information and redundancy.

    Uses the maximum relevance minimum redundancy algorithm to select features
    that are maximally relevant to the target while minimizing redundancy among
    selected features.

    Configuration:
        n_features: Number of features to select
        correlation_type: Type of correlation to measure redundancy
        relevance_type: Method to calculate relevance ("permutation" or "f-statistics")
        results_module: Module name to filter prior results' feature importances (for permutation relevance)
        feature_importance_type: Type of FI aggregation ("mean" or "count")
        feature_importance_method: FI calculation method (FeatureImportanceType enum)
    """

    n_features: int = field(validator=[validators.instance_of(int)], default=Factory(lambda: 30))
    """Number of features selected by MRMR."""

    correlation_type: Literal["pearson", "rdc", "spearman"] = field(
        validator=validators.in_(["pearson", "rdc", "spearman"]), default="spearman"
    )
    """Selection of correlation type."""

    relevance_type: Literal["permutation", "f-statistics"] = field(
        validator=validators.in_(["permutation", "f-statistics"]), default="permutation"
    )
    """Selection of relevance measure."""

    results_module: str = field(
        validator=validators.instance_of(str),
        default="octo",
    )
    """Module name from which feature importances were created."""

    feature_importance_type: Literal["mean", "count"] = field(
        validator=validators.in_(["mean", "count"]), default="mean"
    )
    """Selection of feature importance type."""

    feature_importance_method: FeatureImportanceType = field(
        validator=validators.instance_of(FeatureImportanceType), default=FeatureImportanceType.PERMUTATION
    )
    """Selection of feature importance method."""

    def create_module(self) -> ModuleExecution:
        """Create MrmrModule execution instance."""
        # import only during execution to avoid heavy dependency at config stage
        from .core import MrmrModule  # noqa: PLC0415

        return MrmrModule(config=self)
