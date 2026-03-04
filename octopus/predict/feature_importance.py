"""Feature importance algorithms for octopus.predict.

Provides permutation feature importance and SHAP-based feature importance
computed fresh from loaded models and data. Returns DataFrames with
per-outersplit and ensemble rows, distinguished by a ``fi_source`` column.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["calculate_fi_permutation", "calculate_fi_shap"]

import numpy as np
import pandas as pd

from octopus.metrics.utils import get_score_from_model

# Fixed confidence level for CI calculations (matches training.py)
_CONFIDENCE_LEVEL = 0.95


def _compute_per_split_stats(values: list[float]) -> dict[str, float]:
    """Compute statistics for a single outersplit's repeat scores.

    Uses t-distribution for both p-values and confidence intervals,
    consistent with the training pipeline (``training.py``).

    Args:
        values: List of importance values from permutation repeats.

    Returns:
        Dict with importance_mean, importance_std, p_value, ci_lower, ci_upper.
    """
    arr = np.array(values)
    mean_val = float(np.mean(arr))
    n = len(arr)
    std_val = float(np.std(arr, ddof=1)) if n > 1 else np.nan

    # One-sample t-test: H0 = importance <= 0
    p_value = np.nan
    if not np.isnan(std_val) and std_val > 0:
        from scipy import stats  # noqa: PLC0415

        t_stat = mean_val / (std_val / math.sqrt(n))
        p_value = float(stats.t.sf(t_stat, n - 1))
    elif std_val == 0:
        p_value = 0.5

    # Confidence interval using t-distribution (exact for small n)
    if any(np.isnan(val) for val in [std_val, mean_val]) or n <= 1:
        ci_lower = np.nan
        ci_upper = np.nan
    else:
        from scipy import stats  # noqa: PLC0415

        t_val = stats.t.ppf(1 - (1 - _CONFIDENCE_LEVEL) / 2, n - 1)
        ci_lower = mean_val - t_val * std_val / math.sqrt(n)
        ci_upper = mean_val + t_val * std_val / math.sqrt(n)

    return {
        "importance_mean": mean_val,
        "importance_std": std_val if not np.isnan(std_val) else 0.0,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def calculate_fi_permutation(
    models: dict[int, Any],
    selected_features: dict[int, list[str]],
    test_data: dict[int, pd.DataFrame],
    train_data: dict[int, pd.DataFrame],
    target_col: str,
    target_metric: str,
    positive_class: Any = None,
    n_repeats: int = 10,
    random_state: int = 42,
    feature_groups: dict[str, list[str]] | None = None,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate permutation feature importance across all outer splits.

    For each outer split, permutes each feature (or feature group) n_repeats
    times and measures the decrease in metric score.  Results include
    per-outersplit rows and an ensemble row, distinguished by ``fi_source``.

    When ``feature_groups`` is provided, computes importance for **both**
    individual features and feature groups in a single combined table —
    consistent with group permutation in the training pipeline.

    Features present in ``feature_cols`` but not selected in a given split
    receive zero importance for that split, ensuring the result covers the
    union of all input features.

    Args:
        models: Dict mapping outersplit_id to fitted model.
        selected_features: Dict mapping outersplit_id to feature list.
        test_data: Dict mapping outersplit_id to test DataFrame.
        train_data: Dict mapping outersplit_id to train DataFrame.  Used as the
            sampling pool for permuted feature values — drawing from training
            data provides a larger, more representative sample of each
            feature's marginal distribution than shuffling within the
            (typically smaller) test set alone.
        target_col: Target column name.
        target_metric: Metric name for scoring.
        positive_class: Positive class label for classification.
        n_repeats: Number of permutation repeats per feature.
        random_state: Random seed for reproducibility.
        feature_groups: Optional dict mapping group names to feature lists
            for group permutation importance.  When provided, the result
            table includes rows for both individual features and groups.
            If None, computes per-feature only.
        feature_cols: Union of all input feature columns across splits.
            Features not selected in a split get zero importance for that split.
            If None, only selected features are reported.

    Returns:
        DataFrame with columns: fi_source, feature, importance_mean,
        importance_std, p_value, ci_lower, ci_upper.  Per-split rows have
        full statistics; ensemble rows have NaN for p_value/ci_lower/ci_upper.
        Sorted by fi_source then importance_mean descending.
    """
    rng = np.random.RandomState(random_state)

    # Collect importance values per (split_id, feature)
    split_importances: dict[int, dict[str, list[float]]] = {}

    for split_id, model in models.items():
        features = selected_features[split_id]
        test_df = test_data[split_id]
        train_df = train_data[split_id]
        split_importances[split_id] = {}

        # Build target_assignments from target_col
        target_assignments = {target_col: target_col}

        # Baseline score (direction-adjusted via get_score_from_model)
        baseline = get_score_from_model(
            model, test_df, features, target_metric, target_assignments, positive_class=positive_class
        )

        # Determine what to permute.
        # Following the original pattern from calculate_fi_group_permutation:
        # when feature_groups is provided, permute BOTH individual features
        # AND groups, producing a combined result table.
        items_to_permute: list[tuple[str, list[str]]] = [(f, [f]) for f in features]
        if feature_groups is not None:
            for group_name, group_features in feature_groups.items():
                if any(f in features for f in group_features):
                    items_to_permute.append((group_name, group_features))

        for item_name, cols_to_permute in items_to_permute:
            # Filter to columns actually present
            active_cols = [c for c in cols_to_permute if c in features and c in test_df.columns]
            if not active_cols:
                continue

            repeat_scores = []
            for _ in range(n_repeats):
                # Create shuffled copy — draw replacement values from training
                # data to provide a larger, more representative sampling pool
                # for the feature's marginal distribution.
                test_shuffled = test_df.copy()
                for col in active_cols:
                    train_values = np.asarray(train_df[col].values)
                    test_shuffled[col] = rng.choice(train_values, size=len(test_shuffled), replace=True)

                perm_score = get_score_from_model(
                    model, test_shuffled, features, target_metric, target_assignments, positive_class=positive_class
                )
                # Importance = baseline_score - permuted_score
                # get_score_from_model already handles direction (minimize → negated),
                # so a positive difference always means the feature matters.
                importance = baseline - perm_score
                repeat_scores.append(importance)

            split_importances[split_id][item_name] = repeat_scores

        # Zero-pad features in feature_cols that were not selected in this split
        if feature_cols is not None:
            for fc in feature_cols:
                if fc not in split_importances[split_id]:
                    split_importances[split_id][fc] = [0.0] * n_repeats

    # Build per-split rows with full statistics
    rows: list[dict[str, Any]] = []
    # Track per-split means for ensemble aggregation
    per_split_means: dict[str, list[float]] = {}

    for split_id in sorted(split_importances.keys()):
        for feature_name, values in split_importances[split_id].items():
            stats = _compute_per_split_stats(values)
            rows.append(
                {
                    "fi_source": split_id,
                    "feature": feature_name,
                    **stats,
                }
            )
            if feature_name not in per_split_means:
                per_split_means[feature_name] = []
            per_split_means[feature_name].append(stats["importance_mean"])

    # Build ensemble rows (mean of per-split means, std of per-split means)
    for feature_name, split_means in per_split_means.items():
        arr = np.array(split_means)
        ensemble_mean = float(np.mean(arr))
        ensemble_std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        rows.append(
            {
                "fi_source": "ensemble",
                "feature": feature_name,
                "importance_mean": ensemble_mean,
                "importance_std": ensemble_std,
                "p_value": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        # Sort: per-split rows by split_id then importance desc,
        # ensemble rows last sorted by importance desc
        per_split = result[result["fi_source"] != "ensemble"].copy()
        ensemble = result[result["fi_source"] == "ensemble"].copy()

        per_split = per_split.sort_values(["fi_source", "importance_mean"], ascending=[True, False])
        ensemble = ensemble.sort_values("importance_mean", ascending=False)

        result = pd.concat([per_split, ensemble], ignore_index=True)
    return result


def calculate_fi_shap(
    models: dict[int, Any],
    selected_features: dict[int, list[str]],
    test_data: dict[int, pd.DataFrame],
    shap_type: str = "kernel",
    max_samples: int = 100,
    background_size: int = 200,
) -> pd.DataFrame:
    """Calculate SHAP feature importance across all outer splits.

    Returns per-outersplit rows and an ensemble row, distinguished by
    ``fi_source``.  SHAP does not provide p-values or confidence intervals,
    so those columns are omitted.

    Args:
        models: Dict mapping outersplit_id to fitted model.
        selected_features: Dict mapping outersplit_id to feature list.
        test_data: Dict mapping outersplit_id to test DataFrame.
        shap_type: SHAP explainer type.  One of:
            - ``'kernel'`` — Model-agnostic using KernelExplainer (default).
              Works with any model but slower.
            - ``'permutation'`` — Model-agnostic using PermutationExplainer.
              Uses permutation-based approach.
            - ``'exact'`` — Model-agnostic using ExactExplainer.
              Computes exact SHAP values (slowest, most accurate).
        max_samples: Maximum number of evaluation samples per split.
        background_size: Maximum background dataset size for kernel explainer.

    Returns:
        DataFrame with columns: fi_source, feature, importance_mean,
        importance_std.  Per-split rows have NaN for importance_std
        (single value per split); ensemble rows have the std of per-split
        means.  Sorted by fi_source then importance_mean descending.
    """
    try:
        import shap  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("SHAP is required for SHAP feature importance. Install with: pip install shap") from e

    # Collect per-split SHAP values: one mean|SHAP| per (split, feature)
    split_shap: dict[int, dict[str, float]] = {}

    for split_id, model in models.items():
        features = selected_features[split_id]
        test_df = test_data[split_id]
        split_shap[split_id] = {}

        x_test = test_df[features]
        if len(x_test) > max_samples:
            x_test = x_test.sample(n=max_samples, random_state=42)

        # Convert to numpy to avoid model attribute side-effects
        feature_names = list(x_test.columns)
        x_arr = x_test.to_numpy()
        n_features = x_arr.shape[1]

        # Build prediction function as a plain callable
        if hasattr(model, "predict_proba"):

            def predict_fn(x_in: np.ndarray, _m: Any = model) -> np.ndarray:
                return np.asarray(_m.predict_proba(np.asarray(x_in)))
        else:

            def predict_fn(x_in: np.ndarray, _m: Any = model) -> np.ndarray:
                return np.asarray(_m.predict(np.asarray(x_in)))

        # Create appropriate explainer
        if shap_type == "kernel":
            bg_size = min(background_size, x_arr.shape[0])
            rng = np.random.default_rng(42)
            bg_idx = rng.choice(x_arr.shape[0], size=bg_size, replace=False)
            bg = x_arr[bg_idx]
            explainer = shap.KernelExplainer(predict_fn, bg)
        elif shap_type == "permutation":
            explainer = shap.explainers.Permutation(predict_fn, x_arr)
        elif shap_type == "exact":
            explainer = shap.explainers.Exact(predict_fn, x_arr)
        else:
            raise ValueError(f"Unknown shap_type '{shap_type}'. Use 'kernel', 'permutation', or 'exact'.")

        sv = explainer(x_arr) if shap_type != "kernel" else None
        if shap_type == "kernel":
            shap_values = explainer.shap_values(x_arr)
        else:
            shap_values = sv.values  # type: ignore[union-attr]

        # Handle multi-output: (samples, features) or 3D with features on any non-sample axis
        vals = np.asarray(shap_values)
        if vals.ndim == 2 and vals.shape[1] == n_features:
            importance = np.abs(vals).mean(axis=0)
        elif vals.ndim == 3:
            feat_axes = [i for i in range(1, vals.ndim) if vals.shape[i] == n_features]
            if len(feat_axes) != 1:
                raise ValueError(f"Unexpected SHAP values shape {vals.shape} for {n_features} features")
            reduce_axes = tuple(i for i in range(vals.ndim) if i != feat_axes[0])
            importance = np.mean(np.abs(vals), axis=reduce_axes)
        else:
            raise ValueError(f"Unexpected SHAP values shape {vals.shape}")

        # Mean absolute SHAP per feature for this split
        for i, feat in enumerate(feature_names):
            split_shap[split_id][feat] = float(importance[i])

    # Build per-split rows (importance_std is NaN — single value per split)
    rows: list[dict[str, Any]] = []
    per_split_means: dict[str, list[float]] = {}

    for split_id in sorted(split_shap.keys()):
        for feature_name, importance_val in split_shap[split_id].items():
            rows.append(
                {
                    "fi_source": split_id,
                    "feature": feature_name,
                    "importance_mean": importance_val,
                    "importance_std": float("nan"),
                }
            )
            if feature_name not in per_split_means:
                per_split_means[feature_name] = []
            per_split_means[feature_name].append(importance_val)

    # Build ensemble rows
    for feature_name, split_means in per_split_means.items():
        arr = np.array(split_means)
        ensemble_mean = float(np.mean(arr))
        ensemble_std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        rows.append(
            {
                "fi_source": "ensemble",
                "feature": feature_name,
                "importance_mean": ensemble_mean,
                "importance_std": ensemble_std,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        per_split = result[result["fi_source"] != "ensemble"].copy()
        ensemble = result[result["fi_source"] == "ensemble"].copy()

        per_split = per_split.sort_values(["fi_source", "importance_mean"], ascending=[True, False])
        ensemble = ensemble.sort_values("importance_mean", ascending=False)

        result = pd.concat([per_split, ensemble], ignore_index=True)
    return result
