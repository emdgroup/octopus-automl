# type: ignore

"""MRMR execution module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from attrs import define
from sklearn.feature_selection import f_classif, f_regression

from octopus.logger import LogGroup, get_logger
from octopus.modules.base import FeatureSelectionExecution, FIMethod
from octopus.modules.utils import rdc_correlation_matrix

if TYPE_CHECKING:
    from upath import UPath

    from octopus.modules.mrmr.module import Mrmr  # noqa: F401
    from octopus.study.context import StudyContext

logger = get_logger()


@define
class MrmrModule(FeatureSelectionExecution["Mrmr"]):
    """MRMR execution module. Created by Mrmr.create_module()."""

    def fit(
        self,
        data_traindev: pd.DataFrame,
        data_test: pd.DataFrame,
        feature_cols: list[str],
        study: StudyContext,
        outersplit_id: int,
        output_dir: UPath,
        num_assigned_cpus: int = 1,
        feature_groups: dict | None = None,
        prior_results: dict | None = None,
    ) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fit MRMR module by selecting features with maximum relevance and minimum redundancy."""
        prior_results = prior_results or {}

        logger.set_log_group(LogGroup.PROCESSING, "MRMR")
        self._validate_configuration(prior_results)
        self._log_outersplit_info(outersplit_id, prior_results)

        # Extract feature matrices (local variables, not stored)
        target_cols = list(study.target_assignments.values())
        x_traindev = data_traindev[feature_cols]
        y_traindev = data_traindev[target_cols]

        # Get relevance data
        relevance_df = self._get_relevance_data(
            x_traindev=x_traindev,
            y_traindev=y_traindev,
            feature_cols=feature_cols,
            ml_type=study.ml_type,
            prior_results=prior_results,
        )

        # Calculate MRMR features
        mrmr_dict = _maxrminr(
            features=x_traindev,
            relevance=relevance_df,
            requested_feature_counts=[self.config.n_features],
            correlation_type=self.config.correlation_type,
        )

        # Get selected features (first and only item from dict)
        selected_mrmr_features = list(mrmr_dict.values())[0]
        selected_features = sorted(selected_mrmr_features, key=lambda s: (len(s), s))

        logger.info(f"Selected features: {selected_features}")

        # Store fitted state
        self.selected_features_ = selected_features
        self.feature_importances_ = {}

        return (selected_features, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    def _get_fi_method(self) -> FIMethod:
        """Get FIMethod enum from configuration."""
        fi_method = self.config.feature_importance_method
        if fi_method == "internal":
            return FIMethod.INTERNAL
        elif fi_method == "permutation":
            return FIMethod.PERMUTATION
        elif fi_method == "shap":
            return FIMethod.SHAP
        elif fi_method == "lofo":
            return FIMethod.LOFO
        else:
            raise ValueError(f"Unknown FI method: {fi_method}")

    def _validate_configuration(self, prior_results: dict) -> None:
        """Validate MRMR configuration."""
        if self.config.relevance_type == "permutation":
            if self.config.task_id == 0:
                raise ValueError("MRMR module should not be the first workflow task.")
            fi_df = prior_results.get("feature_importances", pd.DataFrame())
            if fi_df.empty:
                raise ValueError("No feature importances available from prior results.")

            fi_method = self._get_fi_method()
            subset = fi_df[(fi_df["module"] == self.config.results_module) & (fi_df["fi_method"] == fi_method)]
            if subset.empty:
                available_types = fi_df[["module", "fi_method"]].drop_duplicates().to_dict("records")
                raise ValueError(
                    f"No feature importances for module={self.config.results_module}, fi_method={fi_method}. "
                    f"Available: {available_types}"
                )

    def _log_outersplit_info(self, outersplit_id: int, prior_results: dict) -> None:
        """Log basic MRMR info."""
        logger.info("MRMR-Module")
        logger.info(f"Outersplit: {outersplit_id}")
        logger.info(f"Workflow task: {self.config.task_id}")
        logger.info(f"Number of features selected by MRMR: {self.config.n_features}")
        logger.info(f"Correlation type used by MRMR: {self.config.correlation_type}")
        logger.info(f"Relevance type used by MRMR: {self.config.relevance_type}")
        logger.info(f"Specified results module: {self.config.results_module}")

    def _get_relevance_data(
        self,
        x_traindev: pd.DataFrame,
        y_traindev: pd.DataFrame,
        feature_cols: list[str],
        ml_type: str,
        prior_results: dict,
    ) -> pd.DataFrame:
        """Get relevance data based on relevance type."""
        if self.config.relevance_type == "permutation":
            return self._get_permutation_relevance(feature_cols, prior_results)
        elif self.config.relevance_type == "f-statistics":
            return self._get_fstats_relevance(x_traindev, y_traindev, feature_cols, ml_type)
        else:
            raise ValueError(f"Relevance type {self.config.relevance_type} not supported for MRMR.")

    def _get_permutation_relevance(self, feature_cols: list[str], prior_results: dict) -> pd.DataFrame:
        """Get permutation relevance from prior module results (flat DataFrame)."""
        fi_df = prior_results.get("feature_importances", pd.DataFrame())
        fi_method = self._get_fi_method()

        subset = fi_df[(fi_df["module"] == self.config.results_module) & (fi_df["fi_method"] == fi_method)]
        n = subset["training_id"].nunique()
        re_df = (subset.groupby("feature")["importance"].sum() / n).sort_values(ascending=False).reset_index()

        # Reduce to current feature_cols
        re_df = re_df[re_df["feature"].isin(feature_cols)]
        logger.info(f"Number of features in FI table (based on previous selected features): {len(re_df)}")

        # Keep only positive importance
        re_df = re_df[re_df["importance"] > 0].reset_index(drop=True)
        logger.info(f"Number features with positive importance: {len(re_df)}")

        return re_df

    def _get_fstats_relevance(
        self, x_traindev: pd.DataFrame, y_traindev: pd.DataFrame, feature_cols: list[str], ml_type: str
    ) -> pd.DataFrame:
        """Get f-statistics based relevance."""
        return _relevance_fstats(x_traindev, y_traindev, feature_cols, ml_type)


def _relevance_fstats(
    features: pd.DataFrame,
    target: pd.DataFrame,
    feature_cols: list[str],
    ml_type: str,
) -> pd.DataFrame:
    """Calculate f-statistics based relevance."""
    features = features[feature_cols]
    target_array = target.to_numpy().ravel()

    if ml_type == "classification":
        values, _ = f_classif(features, target_array)
    elif ml_type == "regression":
        values, _ = f_regression(features, target_array)
    else:
        raise ValueError(f"ML-type {ml_type} not supported.")

    return pd.DataFrame({"feature": feature_cols, "importance": values})


def _maxrminr(
    features: pd.DataFrame,
    relevance: pd.DataFrame,
    requested_feature_counts: list[int],
    correlation_type: Literal["pearson", "spearman", "rdc"] = "pearson",
    method: Literal["ratio", "difference"] = "ratio",
) -> dict[int, list[str]]:
    """Perform mRMR feature selection.

    Selects features that maximize relevance to target while minimizing redundancy
    among selected features.

    Args:
        features: Dataset with columns as feature names
        relevance: DataFrame with "feature" and "importance" columns
        requested_feature_counts: List of feature counts for partial snapshots
        correlation_type: Correlation method ("pearson", "spearman", or "rdc")
        method: Score method ("ratio" or "difference")

    Returns:
        Dictionary mapping feature counts to lists of selected features
    """
    # Constants for numeric stability
    EPS = 1e-8  # near-perfect correlation tolerance
    LARGE_RED = 1e6  # large finite penalty for near-perfect redundancy
    MIN_RED = 1e-6  # minimum redundancy to avoid division by zero

    # Validate arguments
    if correlation_type not in {"pearson", "spearman", "rdc"}:
        raise ValueError("correlation_type must be one of {'pearson','spearman','rdc'}")
    if method not in {"ratio", "difference"}:
        raise ValueError("method must be 'ratio' or 'difference'")
    if "feature" not in relevance.columns or "importance" not in relevance.columns:
        raise ValueError("relevance must contain 'feature' and 'importance' columns")

    # Coerce importance to numeric
    rel = relevance.copy(deep=True)
    rel["importance"] = pd.to_numeric(rel["importance"], errors="coerce")
    if rel["importance"].isna().any():
        bad = rel.loc[rel["importance"].isna(), "feature"].tolist()
        raise ValueError(f"relevance.importance contains non-numeric/NaN for: {bad}")

    relevant = list(rel["feature"].unique())
    if not relevant:
        return {}

    # Clean requested_feature_counts
    max_feats = len(relevant)
    cleaned_counts = sorted(
        {int(c) for c in requested_feature_counts if isinstance(c, int | np.integer) and 1 <= int(c) <= max_feats}
    )
    if max_feats not in cleaned_counts:
        cleaned_counts.append(max_feats)

    # Ensure features are numeric
    missing = set(relevant) - set(features.columns)
    if missing:
        raise ValueError(f"Missing features in `features` DataFrame: {sorted(missing)}")

    feats = features[relevant].apply(pd.to_numeric, errors="coerce")
    if feats.isna().any().any():
        bad_cols = feats.columns[feats.isna().any()].tolist()
        raise ValueError(f"Some relevant feature columns contain non-numeric/NaN values: {bad_cols}")

    # Calculate correlation matrix
    if correlation_type in {"pearson", "spearman"}:
        corr = feats.corr(method=correlation_type).abs()
    else:  # rdc
        corr_vals = rdc_correlation_matrix(feats)
        corr = pd.DataFrame(corr_vals, index=feats.columns, columns=feats.columns).abs()

    corr = corr.reindex(index=feats.columns, columns=feats.columns)

    # Iterative selection
    selected: list[str] = []
    not_selected = set(feats.columns)
    results: dict[int, list[str]] = {}

    for i in range(1, max_feats + 1):
        candidates = rel[rel["feature"].isin(not_selected)].copy()

        if i == 1:
            candidates["score"] = candidates["importance"]
        else:
            cand_feats = candidates["feature"].values
            candidate_corrs = corr.loc[cand_feats, selected]

            # Handle near-perfect correlations
            perfect_mask = candidate_corrs >= (1.0 - EPS)
            mean_red = candidate_corrs.mask(perfect_mask, np.nan).mean(axis=1)
            mean_red = mean_red.fillna(LARGE_RED).clip(lower=MIN_RED)

            candidates["redundancy"] = mean_red.values
            if method == "ratio":
                candidates["score"] = candidates["importance"] / candidates["redundancy"]
            else:
                candidates["score"] = candidates["importance"] - candidates["redundancy"]

        # Replace infinite scores with NaN
        candidates["score"] = candidates["score"].replace([np.inf, -np.inf], np.nan)
        if candidates["score"].dropna().empty:
            raise ValueError(f"No valid candidate scores at selection step {i}. Check inputs.")

        # Deterministic tie-handling
        candidates["score"] = candidates["score"].fillna(-np.finfo(float).max / 10)

        best = candidates.loc[candidates["score"].idxmax(), "feature"]
        selected.append(best)
        not_selected.remove(best)

        if i in cleaned_counts:
            results[i] = selected.copy()

    return results
