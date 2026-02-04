#  type: ignore

"""MRMR core function.

TOBEDONE:
(1) Is there a way to consider groups?
(2) relevance-type "permutation", importance_type="permutation" ?
(3) add mutual information to relevance methods
(4) saving results? any plots?
"""

from typing import Literal

import numpy as np
import pandas as pd
from attrs import define
from sklearn.feature_selection import f_classif, f_regression

from octopus.logger import LogGroup, get_logger
from octopus.modules.base import ModuleBaseCore
from octopus.modules.mrmr.module import Mrmr
from octopus.modules.utils import rdc_correlation_matrix

logger = get_logger()


@define
class MrmrCore(ModuleBaseCore[Mrmr]):
    """MRMR module for feature selection based on mutual information and redundancy.

    Inherits common properties from BaseCore including log_dir.
    """

    @property
    def correlation_type(self) -> Literal["pearson", "spearman", "rdc"]:
        """Correlation type."""
        return self.experiment.ml_config.correlation_type

    @property
    def relevance_type(self) -> str:
        """Relevance type."""
        return self.experiment.ml_config.relevance_type

    @property
    def results_key(self) -> str:
        """Results key."""
        return self.experiment.ml_config.results_key

    @property
    def n_features(self) -> int:
        """Number of features selected by MRMR."""
        return self.experiment.ml_config.n_features

    @property
    def feature_importances(self) -> dict:
        """Feature importances calculated by preceding module."""
        return self.experiment.prior_results[self.results_key].feature_importances

    @property
    def feature_importance_key(self) -> str:
        """Feature importance key."""
        fi_type = self.experiment.ml_config.feature_importance_type
        fi_method = self.experiment.ml_config.feature_importance_method
        return f"{'internal' if fi_method == 'internal' else fi_method + '_dev'}_{fi_type}"

    def __attrs_post_init__(self):
        """Initialize and validate MRMR configuration."""
        super().__attrs_post_init__()  # Create/clean results directory
        logger.set_log_group(LogGroup.PROCESSING, "MRMR")
        self._validate_configuration()

    def _validate_configuration(self):
        """Validate MRMR configuration.

        1. MRMR should not be the first workflow task
        2. Check if results_key exists
        3. Check if feature_importance key exists

        """
        if self.relevance_type == "permutation":
            if self.experiment.task_id == 0:
                raise ValueError("MRMR module should not be the first workflow task.")
            if self.results_key not in self.experiment.prior_results:
                raise ValueError(
                    f"Specified results key not found: {self.results_key}. Available results keys: {list(self.experiment.prior_feature_importances.keys())}"
                )
            if self.feature_importance_key not in self.feature_importances:
                raise ValueError(
                    f"No feature importances available for key {self.feature_importance_key}.Available keys: {self.feature_importances.keys()}"
                )

    def run_experiment(self):
        """Run mrmr module on experiment."""
        self._log_experiment_info()

        relevance_df = self._get_relevance_data()
        mrmr_dict = self._calculate_mrmr_features(relevance_df)

        # get value of first and only item
        selected_mrmr_features = list(mrmr_dict.values())[0]

        # save features selected by mrmr
        self.experiment.selected_features = sorted(selected_mrmr_features, key=lambda s: (len(s), s))
        logger.info(f"Selected features: {self.experiment.selected_features}")

        return self.experiment

    def _log_experiment_info(self):
        """Log basic MRMR Info."""
        logger.info("MRMR-Module")
        logger.info(f"Experiment: {self.experiment.experiment_id}")
        logger.info(f"Workflow task: {self.experiment.task_id}")
        logger.info(f"Number of features selected by MRMR: {self.n_features}")
        logger.info(f"Correlation type used by MRMR: {self.correlation_type}")
        logger.info(f"Relevance type used by MRMR: {self.relevance_type}")
        logger.info(f"Specified results key: {self.results_key}")
        logger.info(f"Available results keys: {list(self.experiment.prior_results.keys())}")

    def _get_relevance_data(self):
        if self.relevance_type == "permutation":
            return self._get_permutation_relevance()
        elif self.relevance_type == "f-statistics":
            return self._get_fstats_relevance()
        else:
            raise ValueError(f"Relevance type {self.relevance_type} not supported for MRMR.")

    def _get_permutation_relevance(self):
        """Get permutation relevance.

        Only use features with positive importance
        Reduce fi table to feature_cols (previous selected_features).
        Feature columns do not contain any groups.
        """
        re_df = self.feature_importances[self.feature_importance_key]
        re_df = re_df[re_df["feature"].isin(self.feature_cols)]
        logger.info(f"Number of features in fi table (based on previous selected features, no groups): {len(re_df)}")
        re_df = re_df[re_df["importance"] > 0].reset_index(drop=True)
        logger.info(f"Number features with positive importance: {len(re_df)}")
        return re_df

    def _get_fstats_relevance(self):
        """Get fstats relevance."""
        return relevance_fstats(self.x_traindev, self.y_traindev, self.feature_cols, self.ml_type)

    def _calculate_mrmr_features(self, relevance_df):
        """Calculate MRMR features."""
        return maxrminr(
            features=self.x_traindev,
            relevance=relevance_df,
            requested_feature_counts=[self.n_features],
            correlation_type=self.correlation_type,
        )


# shared functions
def relevance_fstats(
    features: pd.DataFrame,
    target: pd.DataFrame,
    feature_cols: list,
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


def maxrminr(
    features: pd.DataFrame,
    relevance: pd.DataFrame,
    requested_feature_counts: list[int],
    correlation_type: Literal["pearson", "spearman", "rdc"] = "pearson",
    method: Literal["ratio", "difference"] = "ratio",
) -> dict[int, list[str]]:
    """Perform mRMR feature selection.

    The followings steps are done:
      1. Determine the relevance of all predictor variables and select the feature
         with the highest relevance.
      2. Determine the mean redundancy between the remaining features and all features
         selected so far.
      3. Calculate an importance score as either ratio or difference between relevance
         and redundancy to select the next feature.
      4. Recalculate importance scores and select the next best feature.
      5. Repeat until the desired number of features (n_features_to_select) is reached.


    Further remarks:
      1. Qubim either uses F1, Smolonogov-stats or Shapley as input
         https://github.com/smazzanti/mrmr/blob/main/mrmr/pandas.py
         use this implementation?

      2. Feature importance from the development dataset are preferable as they show
         features that are relevant for model generalization.

      3. MRMR must not be done on test feature importances to avoid information leakage
         as the MRMR module may be preprocessing step for later model trainings.
         In this module the  features are taken from the traindev dataset.

      4. We ignore selected_features from the previous workflow task. The features
         used are extracted from the feature importance table

    Literature:
        https://github.com/ThomasBury/arfs?tab=readme-ov-file
        https://ar5iv.labs.arxiv.org/html/1908.05376
        https://github.com/smazzanti/mrmr

    Args:
        features: Dataset with columns as feature names.
        relevance: Must contain:
            - "feature": Name of the feature
            - "importance": Numeric measure of its relevance
        requested_feature_counts:
            A list of feature counts (e.g., [1, 3, 5]) for which
            partial selection snapshots will be returned.
        correlation_type:
            Correlation method, e.g., "pearson", "spearman", or "rdc"
            (if implemented). Default is "pearson".
        method: Score method, e.g., "ratio" or "difference".

    Returns:
        dict: A dictionary with the MRMR feature selection for given counts.

    Raises:
        ValueError: If correlation_type is not one of {'pearson', 'spearman', 'rdc'},
                    or if method is not either 'ratio' or 'difference'.
    Additional information:
      - Numeric coercion & validation:
          * relevance['importance'] is coerced to numeric and will raise ValueError if
            any NaNs or non-numeric values remain.
          * All relevant columns in `features` are coerced to numeric and will raise
            ValueError if NaNs are introduced (forcing you to clean/encode first).
      - Near-perfect correlation handling:
          * A small tolerance EPS = 1e-8 is used to detect near-perfect correlations.
          * If a candidate has near-perfect correlation (>= 1 - EPS) with a selected
            feature, that correlation is masked for mean redundancy calculation.
          * If all correlations for a candidate are near-perfect, a large finite redundancy
            (LARGE_RED = 1e6) is assigned to strongly penalize selection (avoids Inf).
          * Redundancy is clipped to MIN_RED = 1e-6 to avoid division-by-zero in ratio mode.
      - RDC:
          * The "rdc" option expects an external `rdc_correlation_matrix(features_df)`
            to be available; if not defined, calling with correlation_type="rdc" will
            raise a NameError/ImportError from that call.
      - Deterministic behavior & tie-breaking:
          * Any infinite scores are replaced with NaN and if no valid scores remain an
            informative ValueError is raised.
          * Remaining NaNs are replaced with a very low numeric value to allow idxmax()
            without arbitrary failures; ties are resolved by pandas' idxmax (first).
      - Output:
          * The function returns a dict mapping requested counts (and the max count if
            not requested) to the list of selected features at that step.
    """
    # Constants for numeric stability & policy
    EPS = 1e-8  # near-perfect correlation tolerance
    LARGE_RED = 1e6  # large finite penalty for near-perfect redundancy
    MIN_RED = 1e-6  # minimum redundancy to avoid division by zero

    # --- Basic argument validation ---
    if correlation_type not in {"pearson", "spearman", "rdc"}:
        raise ValueError("correlation_type must be one of {'pearson','spearman','rdc'}")
    if method not in {"ratio", "difference"}:
        raise ValueError("method must be 'ratio' or 'difference'")

    if "feature" not in relevance.columns or "importance" not in relevance.columns:
        raise ValueError("relevance must contain 'feature' and 'importance' columns")

    # Coerce importance to numeric and fail fast on NaNs
    rel = relevance.copy(deep=True)
    rel["importance"] = pd.to_numeric(rel["importance"], errors="coerce")
    if rel["importance"].isna().any():
        bad = rel.loc[rel["importance"].isna(), "feature"].tolist()
        raise ValueError(f"relevance.importance contains non-numeric/NaN for: {bad}")

    relevant = list(rel["feature"].unique())
    if not relevant:
        return {}

    # Clean and validate requested_feature_counts
    max_feats = len(relevant)
    cleaned_counts = sorted(
        {int(c) for c in requested_feature_counts if isinstance(c, (int | np.integer)) and 1 <= int(c) <= max_feats}
    )
    if max_feats not in cleaned_counts:
        cleaned_counts.append(max_feats)

    # Ensure features contains relevant columns and are numeric
    missing = set(relevant) - set(features.columns)
    if missing:
        raise ValueError(f"Missing features in `features` DataFrame: {sorted(missing)}")

    feats = features[relevant].apply(pd.to_numeric, errors="coerce")
    if feats.isna().any().any():
        bad_cols = feats.columns[feats.isna().any()].tolist()
        raise ValueError(f"Some relevant feature columns contain non-numeric/NaN values: {bad_cols}")

    # --- Correlation matrix (absolute) ---
    if correlation_type in {"pearson", "spearman"}:
        corr = feats.corr(method=correlation_type).abs()
    else:  # rdc
        corr_vals = rdc_correlation_matrix(feats)  # expects external implementation
        corr = pd.DataFrame(corr_vals, index=feats.columns, columns=feats.columns).abs()

    corr = corr.reindex(index=feats.columns, columns=feats.columns)

    # --- Iterative selection ---
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

            perfect_mask = candidate_corrs >= (1.0 - EPS)
            mean_red = candidate_corrs.mask(perfect_mask, np.nan).mean(axis=1)
            mean_red = mean_red.fillna(LARGE_RED).clip(lower=MIN_RED)

            candidates["redundancy"] = mean_red.values
            if method == "ratio":
                candidates["score"] = candidates["importance"] / candidates["redundancy"]
            else:
                candidates["score"] = candidates["importance"] - candidates["redundancy"]

        # Replace infinite scores with NaN and ensure at least one valid candidate
        candidates["score"] = candidates["score"].replace([np.inf, -np.inf], np.nan)
        if candidates["score"].dropna().empty:
            raise ValueError(f"No valid candidate scores at selection step {i}. Check inputs.")

        # Deterministic tie-handling: fill remaining NaNs with a very low value
        candidates["score"] = candidates["score"].fillna(-np.finfo(float).max / 10)

        best = candidates.loc[candidates["score"].idxmax(), "feature"]
        selected.append(best)
        not_selected.remove(best)

        if i in cleaned_counts:
            results[i] = selected.copy()

    return results
