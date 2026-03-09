"""MRMR module (Minimum Redundancy Maximum Relevance feature selection)."""

from __future__ import annotations

from attrs import Factory, define, field, validators

from octopus.modules.base import ModuleExecution, Task
from octopus.types import CorrelationType, FeatureImportanceMethod, FeatureImportanceType, MRMRRelevance


@define
class Mrmr(Task):
    """MRMR module for feature selection based on mutual information and redundancy.

    Uses the maximum relevance minimum redundancy algorithm to select features
    that are maximally relevant to the target while minimizing redundancy among
    selected features.

    Configuration:
        n_features: Number of features to select
        correlation_type: Type of correlation to measure redundancy
        relevance_type: Method to calculate relevance (MRMRRelevance enum)
        results_module: Module name to filter prior results' feature importances (for permutation relevance)
        feature_importance_type: Type of FI aggregation ("mean" or "count")
        feature_importance_method: FI calculation method (FeatureImportanceType enum)
    """

    n_features: int = field(validator=[validators.instance_of(int)], default=Factory(lambda: 30))
    """Number of features selected by MRMR."""

    correlation_type: CorrelationType = field(
        validator=validators.instance_of(CorrelationType), default=CorrelationType.SPEARMAN
    )
    """Selection of correlation type."""

    relevance_type: MRMRRelevance = field(
        validator=validators.instance_of(MRMRRelevance), default=MRMRRelevance.PERMUTATION
    )
    """Selection of relevance measure."""

    results_module: str = field(
        validator=validators.instance_of(str),
        default="octo",
    )
    """Module name from which feature importances were created."""

    feature_importance_type: FeatureImportanceMethod = field(
        validator=validators.instance_of(FeatureImportanceMethod), default=FeatureImportanceMethod.MEAN
    )
    """Selection of feature importance aggregation method."""

    feature_importance_method: FeatureImportanceType = field(
        validator=validators.instance_of(FeatureImportanceType), default=FeatureImportanceType.PERMUTATION
    )
    """Selection of feature importance method."""

    def create_module(self) -> ModuleExecution:
        """Create MrmrModule execution instance."""
        # import only during execution to avoid heavy dependency at config stage
        from .core import MrmrModule  # noqa: PLC0415

        return MrmrModule(config=self)
