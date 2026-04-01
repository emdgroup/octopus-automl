"""Feature importance algorithms for octopus.predict.

Architecture
------------
**Shared Computation Functions** (used by both training and predict):

Shared functions live in ``octopus.feature_importance`` and are re-exported
here for backward compatibility:

- ``compute_per_repeat_stats``  — t-distribution stats from repeat values
- ``compute_shap_single``       — SHAP explainer for a single model + dataset
- ``compute_permutation_single`` — custom draw-from-pool permutation for a single model

**Multi-split Orchestrators** (predict only):

- ``calculate_fi_permutation``  — per-split + ensemble permutation FI
- ``calculate_fi_shap``         — per-split + ensemble SHAP FI

Shared computation functions return DataFrames with column ``"importance"``
(not ``"importance_mean"``) so that Bag aggregation in the training pipeline
works unchanged.  Multi-split orchestrators rename to ``"importance_mean"``
for the predict schema.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Re-export shared computation functions for backward compatibility.
# New code should import directly from octopus.feature_importance.
from octopus.feature_importance import (
    compute_per_repeat_stats,
    compute_permutation_single,
    compute_shap_single,
)
from octopus.types import MLType

__all__ = [
    "calculate_fi_permutation",
    "calculate_fi_shap",
    "compute_per_repeat_stats",
    "compute_permutation_single",
    "compute_shap_single",
]


# ═══════════════════════════════════════════════════════════════
# Multi-split Orchestration (predict only)
# ═══════════════════════════════════════════════════════════════


def _aggregate_across_splits(
    per_split_dfs: dict[int, pd.DataFrame],
    extra_stat_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate per-split feature-importance DataFrames into a combined result.

    Builds per-split rows (``fi_source = split_id``) and ensemble rows
    (``fi_source = "ensemble"``).  The ``importance`` column from each
    per-split DataFrame is renamed to ``importance_mean`` in the output.

    Ensemble rows contain the mean and sample standard deviation of
    per-split importance values.  Extra stat columns (e.g. ``p_value``,
    ``ci_lower``, ``ci_upper``) are carried through from per-split rows
    but set to NaN on ensemble rows.

    Args:
        per_split_dfs: Dict mapping outer_split_id to a DataFrame with
            at least columns ``["feature", "importance"]``.  May also
            contain ``"importance_std"`` and any columns listed in
            *extra_stat_cols*.
        extra_stat_cols: Additional columns to carry through from per-split
            DataFrames (e.g. ``["p_value", "ci_lower", "ci_upper"]``).
            These columns are set to NaN on ensemble rows.

    Returns:
        DataFrame sorted by fi_source then importance_mean descending,
        with per-split rows followed by ensemble rows.
    """
    if extra_stat_cols is None:
        extra_stat_cols = []

    rows: list[dict[str, Any]] = []
    per_split_means: dict[str, list[float]] = {}

    for split_id in sorted(per_split_dfs.keys()):
        split_fi = per_split_dfs[split_id]
        for _, row in split_fi.iterrows():
            feat = row["feature"]
            imp_mean = row["importance"]
            row_dict: dict[str, Any] = {
                "fi_source": split_id,
                "feature": feat,
                "importance_mean": imp_mean,
                "importance_std": row["importance_std"] if "importance_std" in split_fi.columns else float("nan"),
            }
            for col in extra_stat_cols:
                row_dict[col] = row[col] if col in split_fi.columns else float("nan")
            rows.append(row_dict)
            per_split_means.setdefault(feat, []).append(imp_mean)

    for feature_name, split_means in per_split_means.items():
        arr = np.array(split_means)
        ensemble_mean = float(np.mean(arr))
        ensemble_std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        row_dict = {
            "fi_source": "ensemble",
            "feature": feature_name,
            "importance_mean": ensemble_mean,
            "importance_std": ensemble_std,
        }
        for col in extra_stat_cols:
            row_dict[col] = float("nan")
        rows.append(row_dict)

    result = pd.DataFrame(rows)
    if not result.empty:
        per_split = result[result["fi_source"] != "ensemble"].copy()
        ensemble = result[result["fi_source"] == "ensemble"].copy()

        per_split = per_split.sort_values(["fi_source", "importance_mean"], ascending=[True, False])
        ensemble = ensemble.sort_values("importance_mean", ascending=False)

        result = pd.concat([per_split, ensemble], ignore_index=True)
    return result


def calculate_fi_permutation(
    models: dict[int, Any],
    selected_features: dict[int, list[str]],
    test_data: dict[int, pd.DataFrame],
    train_data: dict[int, pd.DataFrame],
    target_assignments: dict[str, str],
    target_metric: str,
    positive_class: Any = None,
    n_repeats: int = 10,
    random_state: int = 42,
    feature_groups: dict[str, list[str]] | None = None,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate permutation feature importance across all outer splits.

    For each outer split, delegates to ``compute_permutation_single`` and
    then aggregates per-split results into ensemble rows via
    ``_aggregate_across_splits``.

    Features present in ``feature_cols`` but not selected in a given split
    receive zero importance for that split, ensuring the result covers the
    union of all input features.

    Args:
        models: Dict mapping outer_split_id to fitted model.
        selected_features: Dict mapping outer_split_id to feature list.
        test_data: Dict mapping outer_split_id to test DataFrame.
        train_data: Dict mapping outer_split_id to train DataFrame.
        target_assignments: Dict mapping semantic target roles to column
            names.  For single-target tasks: ``{"default": "y"}``.
            For time-to-event: ``{"duration": "time_col", "event": "event_col"}``.
        target_metric: Metric name for scoring.
        positive_class: Positive class label for classification.
        n_repeats: Number of permutation repeats per feature.
        random_state: Random seed for reproducibility.
        feature_groups: Optional dict mapping group names to feature lists.
        feature_cols: Union of all input feature columns across splits.
            Features not selected in a split get zero importance for that split.
            If None, only selected features are reported.

    Returns:
        DataFrame with columns: fi_source, feature, importance_mean,
        importance_std, p_value, ci_lower, ci_upper.  Per-split rows have
        full statistics; ensemble rows have NaN for p_value/ci_lower/ci_upper.
        Sorted by fi_source then importance_mean descending.
    """
    per_split_dfs: dict[int, pd.DataFrame] = {}

    for split_id, model in models.items():
        features = selected_features[split_id]
        test_df = test_data[split_id]
        train_df = train_data[split_id]

        split_fi = compute_permutation_single(
            model=model,
            X_test=test_df,
            X_train=train_df,
            feature_cols=features,
            target_metric=target_metric,
            target_assignments=target_assignments,
            positive_class=positive_class,
            n_repeats=n_repeats,
            random_state=random_state,
            feature_groups=feature_groups,
        )

        # Zero-pad features in feature_cols that were not in this split
        if feature_cols is not None:
            existing_features = set(split_fi["feature"].tolist())
            zero_rows = []
            for fc in feature_cols:
                if fc not in existing_features:
                    zero_rows.append(
                        {
                            "feature": fc,
                            "importance": 0.0,
                            "importance_std": 0.0,
                            "p_value": float("nan"),
                            "ci_lower": float("nan"),
                            "ci_upper": float("nan"),
                        }
                    )
            if zero_rows:
                split_fi = pd.concat([split_fi, pd.DataFrame(zero_rows)], ignore_index=True)

        per_split_dfs[split_id] = split_fi

    return _aggregate_across_splits(per_split_dfs, extra_stat_cols=["p_value", "ci_lower", "ci_upper"])


def calculate_fi_shap(
    models: dict[int, Any],
    selected_features: dict[int, list[str]],
    test_data: dict[int, pd.DataFrame],
    shap_type: str = "kernel",
    max_samples: int = 100,
    background_size: int = 200,
    *,
    ml_type: MLType,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate SHAP feature importance across all outer splits.

    For each outer split, delegates to ``compute_shap_single`` and then
    aggregates per-split results into ensemble rows via
    ``_aggregate_across_splits``.

    Args:
        models: Dict mapping outer_split_id to fitted model.
        selected_features: Dict mapping outer_split_id to feature list.
        test_data: Dict mapping outer_split_id to test DataFrame.
        shap_type: SHAP explainer type (``'kernel'``, ``'permutation'``,
            ``'exact'``).
        max_samples: Maximum number of evaluation samples per split.
        background_size: Maximum background dataset size for kernel explainer.
        ml_type: ML task type.  Passed through to ``compute_shap_single``
            to correctly choose ``predict`` vs ``predict_proba``.
        feature_cols: Union of all input feature columns across splits.
            Features not selected in a split get zero importance for that
            split.  If None, only selected features are reported.

    Returns:
        DataFrame with columns: fi_source, feature, importance_mean,
        importance_std.  Per-split rows have NaN for importance_std;
        ensemble rows have the std of per-split means.
    """
    per_split_dfs: dict[int, pd.DataFrame] = {}

    for split_id, model in models.items():
        features = selected_features[split_id]
        test_df = test_data[split_id]

        x_test = test_df[features]

        # Prepare background for kernel
        X_background = None
        if shap_type == "kernel":
            bg_size = min(background_size, len(x_test))
            rng = np.random.default_rng(42)
            bg_idx = rng.choice(len(x_test), size=bg_size, replace=False)
            X_background = x_test.to_numpy()[bg_idx]

        fi_df = compute_shap_single(
            model=model,
            X=x_test,
            feature_names=list(x_test.columns),
            shap_type=shap_type,
            X_background=X_background,
            max_samples=max_samples,
            threshold_ratio=None,  # Keep all for predict
            ml_type=ml_type,
        )

        # Zero-pad features in feature_cols that were not in this split
        if feature_cols is not None:
            existing_features = set(fi_df["feature"].tolist())
            zero_rows = []
            for fc in feature_cols:
                if fc not in existing_features:
                    zero_rows.append({"feature": fc, "importance": 0.0})
            if zero_rows:
                fi_df = pd.concat([fi_df, pd.DataFrame(zero_rows)], ignore_index=True)

        per_split_dfs[split_id] = fi_df

    return _aggregate_across_splits(per_split_dfs)
