"""MRMR execution module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from attrs import define
from sklearn.feature_selection import f_classif, f_regression

from octopus.logger import get_logger
from octopus.modules import ModuleExecution, ModuleResult
from octopus.modules.utils import rdc_correlation_matrix
from octopus.types import CorrelationType, FIResultLabel, LogGroup, MLType, MRMRFIAggregation, MRMRRelevance, ResultType

if TYPE_CHECKING:
    from octopus.modules import StudyContext
    from octopus.modules.mrmr import Mrmr  # noqa: F401

logger = get_logger()

_EPS = 1e-8  # near-perfect correlation tolerance
_LARGE_RED = 1e6  # large finite penalty for near-perfect redundancy
_MIN_RED = 1e-6  # minimum redundancy to avoid division by zero


@define
class MrmrModule(ModuleExecution["Mrmr"]):
    """MRMR execution module. Created by Mrmr.create_module()."""

    def fit(
        self,
        *,
        data_traindev: pd.DataFrame,
        feature_cols: list[str],
        study_context: StudyContext,
        outer_split_id: int,
        dependency_results: dict[ResultType, ModuleResult],
        **kwargs,
    ) -> dict[ResultType, ModuleResult]:
        """Fit MRMR module by selecting features with maximum relevance and minimum redundancy."""
        logger.set_log_group(LogGroup.PROCESSING, "MRMR")
        logger.info(
            f"Outersplit {outer_split_id} | n_features_list={self.config.n_features} "
            f"correlation={self.config.correlation_type} relevance={self.config.relevance_type} "
            f"depends_on={self.config.depends_on}"
        )

        target_cols = list(study_context.target_assignments.values())
        x_traindev = data_traindev[feature_cols]
        y_traindev = data_traindev[target_cols]

        # Get relevance data
        if self.config.relevance_type == MRMRRelevance.FROM_DEPENDENCY:
            fi_method = FIResultLabel(self.config.feature_importance_method)
            aggregation = self.config.feature_importance_type
            relevance_df = _relevance_from_dependency(feature_cols, dependency_results, fi_method, aggregation)
        elif self.config.relevance_type == MRMRRelevance.F_STATISTICS:
            relevance_df = _relevance_fstats(x_traindev, y_traindev, feature_cols, study_context.ml_type)
        else:
            raise ValueError(f"Relevance type {self.config.relevance_type} not supported for MRMR.")

        # Calculate MRMR features
        mrmr_dict = _maxrminr(
            df_features=x_traindev,
            df_relevance=relevance_df,
            n_features_list=[self.config.n_features],
            correlation_type=self.config.correlation_type,
        )

        # Get selected features (first and only item from dict)
        selected_mrmr_features = list(mrmr_dict.values())[0]
        selected_features = sorted(selected_mrmr_features, key=lambda s: (len(s), s))

        logger.info(f"Selected features: {selected_features}")

        return {
            ResultType.BEST: ModuleResult(
                result_type=ResultType.BEST,
                module=self.config.module,
                selected_features=selected_features,
            )
        }


def _relevance_from_dependency(
    feature_cols: list[str],
    dependency_results: dict[ResultType, ModuleResult],
    fi_method: FIResultLabel,
    aggregation: MRMRFIAggregation = MRMRFIAggregation.MEAN,
) -> pd.DataFrame:
    """Derive MRMR relevance scores from the upstream task's feature importances.

    Uses pre-aggregated FI when available (Tako provides mean/count rows).
    For single-model dependencies (e.g. AutoGluon), uses FI values directly.
    """
    df_fi_all = dependency_results[ResultType.BEST].fi
    if df_fi_all is None or df_fi_all.empty:
        raise ValueError("Dependency task produced no feature importances.")

    df = df_fi_all[df_fi_all["fi_method"] == fi_method]

    df_agg = df[df["training_id"] == aggregation.value]
    if not df_agg.empty:
        df = df_agg

    return (
        df.loc[df["feature"].isin(feature_cols), ["feature", "importance"]]
        .groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )


def _relevance_fstats(
    features: pd.DataFrame,
    target: pd.DataFrame,
    feature_cols: list[str],
    ml_type: MLType,
) -> pd.DataFrame:
    """Calculate f-statistics based relevance."""
    features = features[feature_cols]
    target_array = target.to_numpy().ravel()

    if ml_type in (MLType.BINARY, MLType.MULTICLASS):
        values, _ = f_classif(features, target_array)
    elif ml_type == MLType.REGRESSION:
        values, _ = f_regression(features, target_array)
    else:
        raise ValueError(f"ML-type {ml_type} not supported.")

    return pd.DataFrame({"feature": feature_cols, "importance": values})


def _maxrminr(
    df_features: pd.DataFrame,
    df_relevance: pd.DataFrame,
    n_features_list: list[int],
    correlation_type: CorrelationType = CorrelationType.PEARSON,
) -> dict[int, list[str]]:
    """Perform mRMR feature selection.

    Selects features that maximize relevance to target while minimizing redundancy
    among selected features.

    Args:
        df_features: Dataset with columns as feature names
        df_relevance: DataFrame with "feature" and "importance" columns
        n_features_list: List of feature counts for partial snapshots
        correlation_type: Correlation method (CorrelationType.PEARSON, CorrelationType.SPEARMAN, or CorrelationType.RDC)

    Returns:
        Dictionary mapping feature counts to lists of selected features
    """
    # Drop features with NaN importance (e.g. zero-variance features from f_classif/f_regression)
    # Drop features with NaN or non-positive importance (smazzanti/mrmr convention)
    df_relevance = df_relevance.dropna(subset=["importance"])
    df_relevance = df_relevance[df_relevance["importance"] > 0]

    feature_names = list(df_relevance["feature"].unique())
    if not feature_names:
        raise ValueError("No features with positive relevance. Check upstream feature importances or input data.")

    max_features = len(feature_names)
    n_features_clamped = sorted({min(c, max_features) for c in n_features_list} | {max_features})

    # NaN values would break correlation calculations.
    df_features = df_features[feature_names].apply(pd.to_numeric, errors="coerce")
    if df_features.isna().any().any():
        bad_cols = df_features.columns[df_features.isna().any()].tolist()
        raise ValueError(f"Feature columns contain NaN values: {bad_cols}")

    # Calculate correlation matrix
    if correlation_type in (CorrelationType.PEARSON, CorrelationType.SPEARMAN):
        corr = df_features.corr(method=correlation_type.value).abs()  # type: ignore[arg-type]
    elif correlation_type == CorrelationType.RDC:
        corr_vals = rdc_correlation_matrix(df_features)
        corr = pd.DataFrame(corr_vals, index=df_features.columns, columns=df_features.columns).abs()
    else:
        raise ValueError(f"Correlation type {correlation_type} not supported for MRMR.")

    # Convert to numpy for fast iteration
    corr_matrix = corr.reindex(index=feature_names, columns=feature_names).to_numpy()
    importance = df_relevance.set_index("feature").loc[feature_names, "importance"].to_numpy()

    n = len(feature_names)
    is_selected = np.zeros(n, dtype=bool)
    selected_indices: list[int] = []
    results: dict[int, list[str]] = {}

    for i in range(1, n + 1):
        candidate_mask = ~is_selected

        if i == 1:
            scores = np.where(candidate_mask, importance, -np.inf)
        else:
            # Mean correlation with already-selected features
            redundancy = corr_matrix[np.ix_(candidate_mask, np.array(selected_indices))]

            # Handle near-perfect correlations
            redundancy = np.where(redundancy >= (1.0 - _EPS), np.nan, redundancy)
            with np.errstate(all="ignore"):
                mean_red = np.nanmean(redundancy, axis=1)
            mean_red = np.where(np.isnan(mean_red), _LARGE_RED, mean_red)
            mean_red = np.clip(mean_red, _MIN_RED, None)

            scores = np.full(n, -np.inf)
            scores[candidate_mask] = importance[candidate_mask] / mean_red

        best_idx = int(np.argmax(scores))
        is_selected[best_idx] = True
        selected_indices.append(best_idx)

        if i in n_features_clamped:
            results[i] = [feature_names[j] for j in selected_indices]

    return results
