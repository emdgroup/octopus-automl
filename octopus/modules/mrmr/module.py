"""MRMR module (Minimum Redundancy Maximum Relevance feature selection)."""

from __future__ import annotations

from attrs import Factory, define, field, validators

from octopus.types import CorrelationType, FIComputeMethod, RelevanceMethod

from ..base import ModuleExecution, Task


@define
class Mrmr(Task):
    """MRMR module for feature selection based on mutual information and redundancy.

    Uses the maximum relevance minimum redundancy algorithm to select features
    that are maximally relevant to the target while minimizing redundancy among
    selected features.

    Configuration:
        n_features: Number of features to select
        correlation_type: Type of correlation to measure redundancy
        relevance_method: Method to calculate relevance (RelevanceMethod.PERMUTATION or RelevanceMethod.F_STATISTICS)
        feature_importance_method: FI calculation method, only used when relevance_method is PERMUTATION
    """

    n_features: int = field(validator=[validators.instance_of(int)], default=Factory(lambda: 30))
    """Number of features selected by MRMR."""

    correlation_type: CorrelationType = field(
        converter=CorrelationType,
        validator=validators.in_([CorrelationType.PEARSON, CorrelationType.SPEARMAN, CorrelationType.RDC]),
        default=CorrelationType.SPEARMAN,
    )
    """Selection of correlation type."""

    relevance_method: RelevanceMethod = field(
        converter=RelevanceMethod,
        validator=validators.in_([RelevanceMethod.F_STATISTICS, RelevanceMethod.PERMUTATION]),
        default=RelevanceMethod.PERMUTATION,
    )
    """Method to calculate relevance (permutation or f-statistics)."""

    feature_importance_method: FIComputeMethod = field(
        converter=FIComputeMethod,
        validator=validators.in_(
            [FIComputeMethod.PERMUTATION, FIComputeMethod.SHAP, FIComputeMethod.INTERNAL, FIComputeMethod.LOFO]
        ),
        default=FIComputeMethod.PERMUTATION,
    )
    """FI method to use from prior results. Only relevant when relevance_method is PERMUTATION."""

    def create_module(self) -> ModuleExecution:
        """Create MrmrModule execution instance."""
        # import only during execution to avoid heavy dependency at config stage
        from .core import MrmrModule  # noqa: PLC0415

        return MrmrModule(config=self)
