"""MRMR module (Minimum Redundancy Maximum Relevance feature selection)."""

from __future__ import annotations

from typing import Literal

from attrs import Factory, define, field, validators

from octopus.modules.base import Task

from .core import MrmrModule


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
        feature_importance_method: FI calculation method
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

    feature_importance_method: Literal["permutation", "shap", "internal", "lofo"] = field(
        validator=validators.in_(["permutation", "shap", "internal", "lofo"]), default="permutation"
    )
    """Selection of feature importance method."""

    def create_module(self) -> MrmrModule:
        """Create MrmrModule execution instance."""
        return MrmrModule(config=self)
