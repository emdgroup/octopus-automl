"""Feature importance algorithms for octopus.predict.

Provides permutation feature importance and SHAP-based feature importance
computed fresh from loaded models and data. Returns DataFrames with
statistical details (p-values, confidence intervals).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from octopus.metrics import Metrics
from octopus.metrics.utils import get_performance_from_model


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
    times and measures the decrease in metric performance. Results are
    aggregated across splits.

    Features present in ``feature_cols`` but not selected in a given split
    receive zero importance for that split, ensuring the result covers the
    union of all input features.

    Args:
        models: Dict mapping outersplit_id to fitted model.
        selected_features: Dict mapping outersplit_id to feature list.
        test_data: Dict mapping outersplit_id to test DataFrame.
        train_data: Dict mapping outersplit_id to train DataFrame.
        target_col: Target column name.
        target_metric: Metric name for scoring.
        positive_class: Positive class label for classification.
        n_repeats: Number of permutation repeats per feature.
        random_state: Random seed for reproducibility.
        feature_groups: Optional dict mapping group names to feature lists
            for group permutation importance. If None, computes per-feature.
        feature_cols: Union of all input feature columns across splits.
            Features not selected in a split get zero importance for that split.
            If None, only selected features are reported.

    Returns:
        DataFrame with columns: feature, importance_mean, importance_std,
        p_value, ci_lower, ci_upper, sorted by importance_mean descending.
    """
    rng = np.random.RandomState(random_state)

    # Collect all importance values per feature across splits
    all_importances: dict[str, list[float]] = {}

    for split_id, model in models.items():
        features = selected_features[split_id]
        test_df = test_data[split_id]

        # Build target_assignments from target_col
        target_assignments = {target_col: target_col}

        # Baseline score
        baseline = get_performance_from_model(
            model, test_df, features, target_metric, target_assignments, positive_class=positive_class
        )

        metric = Metrics.get_instance(target_metric)
        sign = 1.0 if metric.higher_is_better else -1.0

        # Determine what to permute.
        # Following the original pattern from get_fi_group_permutation:
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
                # Create shuffled copy
                test_shuffled = test_df.copy()
                for col in active_cols:
                    test_shuffled[col] = rng.permutation(test_shuffled[col].values)

                perm_score = get_performance_from_model(
                    model, test_shuffled, features, target_metric, target_assignments, positive_class=positive_class
                )
                # Importance = baseline - permuted (for higher_is_better)
                # For lower_is_better, importance = permuted - baseline
                importance = sign * (baseline - perm_score)
                repeat_scores.append(importance)

            if item_name not in all_importances:
                all_importances[item_name] = []
            all_importances[item_name].extend(repeat_scores)

    # Aggregate results
    rows = []
    for feature_name, values in all_importances.items():
        arr = np.array(values)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        n = len(arr)

        # One-sample t-test: H0 = importance <= 0
        if std_val > 0 and n > 1:
            from scipy import stats

            t_stat = mean_val / (std_val / np.sqrt(n))
            p_value = float(1.0 - stats.t.cdf(t_stat, df=n - 1))
        else:
            p_value = 0.0 if mean_val > 0 else 1.0

        # 95% confidence interval
        se = std_val / np.sqrt(n) if n > 0 else 0.0
        ci_lower = mean_val - 1.96 * se
        ci_upper = mean_val + 1.96 * se

        rows.append(
            {
                "feature": feature_name,
                "importance_mean": mean_val,
                "importance_std": std_val,
                "p_value": p_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return result


def calculate_fi_shap(
    models: dict[int, Any],
    selected_features: dict[int, list[str]],
    test_data: dict[int, pd.DataFrame],
    shap_type: str = "tree",
    max_samples: int = 100,
) -> pd.DataFrame:
    """Calculate SHAP feature importance across all outer splits.

    Args:
        models: Dict mapping outersplit_id to fitted model.
        selected_features: Dict mapping outersplit_id to feature list.
        test_data: Dict mapping outersplit_id to test DataFrame.
        shap_type: SHAP explainer type ('tree', 'kernel', 'linear').
        max_samples: Maximum number of samples for SHAP computation.

    Returns:
        DataFrame with columns: feature, importance_mean, importance_std,
        p_value, ci_lower, ci_upper, sorted by importance_mean descending.
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError("SHAP is required for SHAP feature importance. Install with: pip install shap") from e

    # Collect SHAP values per feature across splits
    all_shap: dict[str, list[float]] = {}

    for split_id, model in models.items():
        features = selected_features[split_id]
        test_df = test_data[split_id]

        x_test = test_df[features]
        if len(x_test) > max_samples:
            x_test = x_test.sample(n=max_samples, random_state=42)

        # Create appropriate explainer
        if shap_type == "tree":
            explainer = shap.TreeExplainer(model)
        elif shap_type == "kernel":
            # Use a small background sample
            bg = x_test.sample(n=min(50, len(x_test)), random_state=42)
            explainer = shap.KernelExplainer(model.predict, bg)
        elif shap_type == "linear":
            explainer = shap.LinearExplainer(model, x_test)
        else:
            raise ValueError(f"Unknown shap_type '{shap_type}'. Use 'tree', 'kernel', or 'linear'.")

        shap_values = explainer.shap_values(x_test)

        # Handle multi-output (classification): take absolute mean across classes
        if isinstance(shap_values, list):
            # Binary classification: use positive class
            shap_arr = np.abs(np.array(shap_values[-1]))
        else:
            shap_arr = np.abs(np.array(shap_values))

        # Mean absolute SHAP per feature for this split
        mean_abs_shap = np.mean(shap_arr, axis=0)

        for i, feat in enumerate(features):
            if feat not in all_shap:
                all_shap[feat] = []
            all_shap[feat].append(float(mean_abs_shap[i]))

    # Aggregate
    rows = []
    for feature_name, values in all_shap.items():
        arr = np.array(values)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        n = len(arr)

        p_value = 0.0 if mean_val > 0 else 1.0
        se = std_val / np.sqrt(n) if n > 0 else 0.0
        ci_lower = mean_val - 1.96 * se
        ci_upper = mean_val + 1.96 * se

        rows.append(
            {
                "feature": feature_name,
                "importance_mean": mean_val,
                "importance_std": std_val,
                "p_value": p_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return result
