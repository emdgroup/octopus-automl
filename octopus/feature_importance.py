"""Feature-importance computation: primitives and multi-split orchestration.

Primitives
----------
- ``compute_per_repeat_stats`` — t-distribution stats from repeat values
- ``compute_permutation_single`` — custom draw-from-pool permutation for a single model
- ``compute_shap_single`` — SHAP explainer for a single model + dataset

Multi-split orchestrators
-------------------------
- ``calculate_fi_permutation`` — per-split + ensemble permutation FI
- ``calculate_fi_shap`` — per-split + ensemble SHAP FI
- ``dispatch_fi`` — dispatch to the appropriate algorithm
- ``merge_feature_groups`` — union per-split feature groups
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from octopus.metrics.utils import get_score_from_model
from octopus.types import MLType, ShapType

__all__ = [
    "calculate_fi_permutation",
    "calculate_fi_shap",
    "compute_per_repeat_stats",
    "compute_permutation_single",
    "compute_shap_single",
    "dispatch_fi",
    "merge_feature_groups",
]

logger = logging.getLogger(__name__)

# Fixed confidence level for CI calculations
_CONFIDENCE_LEVEL = 0.95


# ═══════════════════════════════════════════════════════════════
# Shared Computation Primitives
# ═══════════════════════════════════════════════════════════════


def compute_per_repeat_stats(
    values: list[float],
    confidence_level: float = _CONFIDENCE_LEVEL,
) -> dict[str, float]:
    """Compute statistics for a single set of permutation-repeat scores.

    Uses t-distribution for both p-values and confidence intervals,
    consistent with the training pipeline.

    Args:
        values: List of importance values from permutation repeats.
        confidence_level: Confidence level for CI (default 0.95).

    Returns:
        Dict with keys: importance, importance_std, p_value, ci_lower, ci_upper.
        Note: the key is ``"importance"`` (not ``"importance_mean"``) for
        compatibility with the Bag aggregation schema.
    """
    from scipy import stats as sp_stats  # noqa: PLC0415 — lazy for import isolation (doc 16)

    arr = np.array(values)
    mean_val = float(np.mean(arr))
    n = len(arr)
    std_val = float(np.std(arr, ddof=1)) if n > 1 else np.nan

    # One-sample t-test: H0 = importance <= 0
    p_value: float = np.nan
    if not np.isnan(std_val) and std_val > 0:
        t_stat = mean_val / (std_val / math.sqrt(n))
        p_value = float(sp_stats.t.sf(t_stat, n - 1))
    elif std_val == 0:
        # Zero variance: importance is deterministic. Use sign of mean.
        if mean_val > 0:
            p_value = 0.0  # certainly positive importance
        elif mean_val < 0:
            p_value = 1.0  # certainly negative importance
        else:
            p_value = 0.5  # exactly zero — cannot reject null

    # Confidence interval using t-distribution
    if any(np.isnan(val) for val in [std_val, mean_val]) or n <= 1:
        ci_lower: float = np.nan
        ci_upper: float = np.nan
    else:
        t_val = sp_stats.t.ppf(1 - (1 - confidence_level) / 2, n - 1)
        ci_lower = mean_val - t_val * std_val / math.sqrt(n)
        ci_upper = mean_val + t_val * std_val / math.sqrt(n)

    return {
        "importance": mean_val,
        "importance_std": std_val if not np.isnan(std_val) else 0.0,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def compute_shap_single(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    feature_names: list[str],
    shap_type: ShapType | str = ShapType.AUTO,
    X_background: np.ndarray | pd.DataFrame | None = None,
    max_samples: int | None = None,
    threshold_ratio: float | None = None,
    *,
    ml_type: MLType,
) -> pd.DataFrame:
    """Compute SHAP feature importance for a single model and dataset.

    Args:
        model: A fitted model.
        X: Evaluation data (features only).
        feature_names: Feature column names matching columns/order of *X*.
        shap_type: SHAP explainer type.  One of:
            - ``ShapType.AUTO`` — let SHAP choose (fast for tree/linear models).
            - ``ShapType.KERNEL`` — ``KernelExplainer`` (model-agnostic, slower).
            - ``ShapType.PERMUTATION`` — ``PermutationExplainer``.
            - ``ShapType.EXACT`` — ``ExactExplainer``.
        X_background: Background dataset for auto/kernel explainers.
            If None for kernel, evaluation data is sampled as background.
        max_samples: If set, subsample *X* to at most this many rows.
        threshold_ratio: If set (e.g. ``1/1000``), drop features whose
            importance is below ``max_importance * threshold_ratio``.
            Use ``None`` to keep all features.
        ml_type: ML task type, used to choose predict vs predict_proba.

    Returns:
        DataFrame with columns ``["feature", "importance"]``.
    """
    try:
        import shap  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import error path
        raise ImportError(
            "The 'shap' package is required to compute SHAP feature "
            "importance. Please install it (e.g. with 'pip install shap') "
            "and try again."
        ) from exc

    # Convert to numpy to avoid model attribute side-effects
    if isinstance(X, pd.DataFrame):
        x_arr = X.to_numpy()
    else:
        x_arr = np.asarray(X)

    if max_samples is not None and x_arr.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(x_arr.shape[0], size=max_samples, replace=False)
        x_arr = x_arr[idx]

    n_features = x_arr.shape[1]

    # Normalize shap_type to ShapType enum for consistent comparisons
    shap_type = ShapType(shap_type)

    # Build prediction function
    # Classifiers: use predict_proba to give SHAP a continuous output.
    # model.predict returns class labels which are not meaningful for SHAP.
    # Regressors / time-to-event: use model.predict (scalar output).
    is_classifier = ml_type in (MLType.BINARY, MLType.MULTICLASS)

    if is_classifier:

        def predict_fn(x_in: np.ndarray, _m: Any = model, _cols: list[str] = feature_names) -> np.ndarray:
            return np.asarray(_m.predict_proba(pd.DataFrame(np.asarray(x_in), columns=_cols)))
    else:

        def predict_fn(x_in: np.ndarray, _m: Any = model, _cols: list[str] = feature_names) -> np.ndarray:
            return np.asarray(_m.predict(pd.DataFrame(np.asarray(x_in), columns=_cols)))

    # Prepare background data
    if X_background is not None:
        bg = X_background.to_numpy() if isinstance(X_background, pd.DataFrame) else np.asarray(X_background)
    else:
        bg = None

    # Build explainer based on shap_type
    if shap_type == ShapType.AUTO:
        bg_data = bg if bg is not None else x_arr
        bg_df = pd.DataFrame(bg_data, columns=feature_names)
        x_df = pd.DataFrame(x_arr, columns=feature_names)

        # Try to construct explainer with model; only catch construction failures
        evaluation_input: pd.DataFrame | np.ndarray
        try:
            # Try model directly — SHAP can auto-detect Tree/Linear explainers for speed
            explainer = shap.Explainer(model, bg_df)
            evaluation_input = x_df
        except Exception as e1:
            # Don't mask fatal errors
            if isinstance(e1, (MemoryError, KeyboardInterrupt)):
                raise
            logger.debug("SHAP auto explainer with model failed: %s. Falling back to callable wrapper.", e1)
            # Fall back to callable approach (predict_fn wraps numpy→DataFrame internally)
            if bg is None:
                bg = x_arr
            explainer = shap.Explainer(predict_fn, bg)
            evaluation_input = x_arr

        # Evaluate explainer (failures here propagate to caller)
        sv = explainer(evaluation_input)
        shap_values = np.asarray(sv.values)
    elif shap_type == ShapType.KERNEL:
        if bg is None:
            background_size = min(200, x_arr.shape[0])
            rng = np.random.default_rng(42)
            bg_idx = rng.choice(x_arr.shape[0], size=background_size, replace=False)
            bg = x_arr[bg_idx]
        explainer = shap.KernelExplainer(predict_fn, bg)
        shap_values = explainer.shap_values(x_arr)
        # KernelExplainer.shap_values may return a list of per-class arrays for multi-class models.
        # Stack such outputs into a numeric array so that downstream shape handling works correctly.
        if isinstance(shap_values, (list, tuple)):
            try:
                shap_values = np.stack(shap_values, axis=-1)
            except ValueError:
                shap_values = np.asarray(shap_values)
        else:
            shap_values = np.asarray(shap_values)
    elif shap_type in (ShapType.PERMUTATION, ShapType.EXACT):
        explainer_cls = shap.explainers.Permutation if shap_type == ShapType.PERMUTATION else shap.explainers.Exact
        explainer = explainer_cls(predict_fn, x_arr)
        sv = explainer(x_arr)
        shap_values = np.asarray(sv.values)
    else:
        raise ValueError(
            f"Unknown shap_type '{shap_type}'. Use ShapType.AUTO, ShapType.KERNEL, "
            f"ShapType.PERMUTATION, or ShapType.EXACT."
        )

    # Aggregate absolute SHAP to per-feature importances
    # 2D (n_samples, n_features): regression / time-to-event, or classifier fallback
    # 3D (n_samples, n_features, n_classes): classifier with predict_proba
    vals = shap_values
    if vals.ndim == 2 and vals.shape[1] == n_features:
        # Regression / single-output: mean |SHAP| over samples
        importance = np.abs(vals).mean(axis=0)
    elif vals.ndim == 3:
        # Classification: mean |SHAP| per class, then average across classes
        # vals shape: (n_samples, n_features, n_classes)
        mean_abs_per_class = np.abs(vals).mean(axis=0)  # (n_features, n_classes)
        importance = mean_abs_per_class.mean(axis=1)  # (n_features,)
    else:
        raise ValueError(f"Unexpected SHAP values shape {vals.shape}")

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importance})

    # Apply threshold filter if requested
    if threshold_ratio is not None and not fi_df["importance"].empty:
        fi_df = fi_df[fi_df["importance"] >= fi_df["importance"].max() * threshold_ratio]

    return fi_df


def compute_permutation_single(
    model: Any,
    X_test: pd.DataFrame,
    X_train: pd.DataFrame,
    feature_cols: list[str],
    target_metric: str,
    target_assignments: dict[str, str],
    positive_class: Any = None,
    n_repeats: int = 10,
    random_state: int = 42,
    feature_groups: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Compute permutation feature importance for a single model.

    Uses the custom draw-from-pool algorithm: replacement values are drawn
    from the combined pool of *X_train* and *X_test* feature values,
    providing a more representative approximation of the marginal
    distribution than either partition alone.

    When ``feature_groups`` is provided, computes importance for **both**
    individual features and feature groups.

    Args:
        model: A fitted model.
        X_test: Test data (must contain feature columns + target columns).
        X_train: Training data.  Feature values from both *X_train* and
            *X_test* are combined to form the sampling pool for replacement
            values.
        feature_cols: Feature column names used by the model.
        target_metric: Metric name for scoring.
        target_assignments: Dict mapping target roles to column names.
        positive_class: Positive class label for classification.
        n_repeats: Number of permutation repeats per feature.
        random_state: Random seed for reproducibility.
        feature_groups: Optional dict mapping group names to feature lists.

    Returns:
        DataFrame with columns:
        ``["feature", "importance", "importance_std", "p_value", "ci_lower", "ci_upper"]``.
        Column ``"importance"`` holds the mean across repeats (compatible
        with Bag aggregation).
    """
    rng = np.random.default_rng(random_state)

    # Baseline score
    baseline = get_score_from_model(
        model, X_test, feature_cols, target_metric, target_assignments, positive_class=positive_class
    )

    # O(1) membership checks for feature_cols
    feature_cols_set = set(feature_cols)

    # Build items to permute: individual features + groups
    items_to_permute: list[tuple[str, list[str]]] = [(f, [f]) for f in feature_cols]
    if feature_groups is not None:
        for group_name, group_features in feature_groups.items():
            if any(f in feature_cols_set for f in group_features):
                items_to_permute.append((group_name, group_features))

    results: list[dict[str, Any]] = []
    n_test = len(X_test)

    for item_name, cols_to_permute in items_to_permute:
        active_cols = [c for c in cols_to_permute if c in feature_cols_set and c in X_test.columns]
        if not active_cols:
            continue

        # Precompute combined pool values (X_train + X_test) per column once
        # to avoid repeated array conversions inside the permutation repeat loop.
        pool_values_per_col = {
            col: np.concatenate([np.asarray(X_train[col].values), np.asarray(X_test[col].values)])
            for col in active_cols
        }

        # Single DataFrame copy; permuted columns are restored after each repeat
        test_shuffled = X_test.copy()
        originals = {col: test_shuffled[col].values.copy() for col in active_cols}

        repeat_scores: list[float] = []
        for _ in range(n_repeats):
            for col in active_cols:
                test_shuffled[col] = rng.choice(pool_values_per_col[col], size=n_test, replace=True)

            perm_score = get_score_from_model(
                model, test_shuffled, feature_cols, target_metric, target_assignments, positive_class=positive_class
            )
            repeat_scores.append(baseline - perm_score)

            # Restore original column values for the next repeat
            for col in active_cols:
                test_shuffled[col] = originals[col]

        stats = compute_per_repeat_stats(repeat_scores)
        results.append({"feature": item_name, **stats})

    if not results:
        return pd.DataFrame(columns=["feature", "importance", "importance_std", "p_value", "ci_lower", "ci_upper"])

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# Multi-split Orchestration
# ═══════════════════════════════════════════════════════════════


def _aggregate_across_splits(
    per_split_dfs: dict[int, pd.DataFrame],
    extra_stat_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate per-split feature-importance DataFrames into a combined result.

    Builds per-split rows (``fi_source = split_id``) and ensemble rows
    (``fi_source = "ensemble"``).  The ``importance`` column from each
    per-split DataFrame is renamed to ``importance_mean`` in the output.

    Args:
        per_split_dfs: Dict mapping outersplit_id to a DataFrame with
            at least columns ``["feature", "importance"]``.
        extra_stat_cols: Additional columns to carry through from per-split
            DataFrames (e.g. ``["p_value", "ci_lower", "ci_upper"]``).

    Returns:
        DataFrame sorted by fi_source then importance_mean descending.
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

    Args:
        models: Dict mapping outersplit_id to fitted model.
        selected_features: Dict mapping outersplit_id to feature list.
        test_data: Dict mapping outersplit_id to test DataFrame.
        train_data: Dict mapping outersplit_id to train DataFrame.
        target_assignments: Target column assignments.
        target_metric: Metric name for scoring.
        positive_class: Positive class label for classification.
        n_repeats: Number of permutation repeats per feature.
        random_state: Random seed for reproducibility.
        feature_groups: Optional dict mapping group names to feature lists.
        feature_cols: Union of all input feature columns across splits.

    Returns:
        DataFrame with columns: fi_source, feature, importance_mean,
        importance_std, p_value, ci_lower, ci_upper.
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

    Args:
        models: Dict mapping outersplit_id to fitted model.
        selected_features: Dict mapping outersplit_id to feature list.
        test_data: Dict mapping outersplit_id to test DataFrame.
        shap_type: SHAP explainer type.
        max_samples: Maximum number of evaluation samples per split.
        background_size: Maximum background dataset size for kernel explainer.
        ml_type: ML task type.
        feature_cols: Union of all input feature columns across splits.

    Returns:
        DataFrame with columns: fi_source, feature, importance_mean, importance_std.
    """
    per_split_dfs: dict[int, pd.DataFrame] = {}

    for split_id, model in models.items():
        features = selected_features[split_id]
        test_df = test_data[split_id]

        x_test = test_df[features]

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
            threshold_ratio=None,
            ml_type=ml_type,
        )

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


def merge_feature_groups(
    feature_groups_per_split: dict[int, dict[str, list[str]]],
) -> dict[str, list[str]]:
    """Union per-split feature groups into one dict.

    Args:
        feature_groups_per_split: Dict mapping outersplit_id to a dict
            of group_name -> feature list.

    Returns:
        Dict mapping group names to sorted lists of feature names.
    """
    all_groups: dict[str, list[str]] = {}
    for split_groups in feature_groups_per_split.values():
        for group_name, group_features in split_groups.items():
            if group_name in all_groups:
                existing = set(all_groups[group_name])
                existing.update(group_features)
                all_groups[group_name] = sorted(existing)
            else:
                all_groups[group_name] = sorted(group_features)
    return all_groups


def dispatch_fi(
    models: dict[int, Any],
    selected_features: dict[int, list[str]],
    test_data: dict[int, pd.DataFrame],
    train_data: dict[int, pd.DataFrame],
    target_assignments: dict[str, str],
    target_metric: str,
    positive_class: Any,
    feature_cols: list[str],
    feature_groups_per_split: dict[int, dict[str, list[str]]],
    fi_type: str,
    *,
    n_repeats: int = 10,
    feature_groups: dict[str, list[str]] | None = None,
    random_state: int = 42,
    ml_type: MLType | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Dispatch FI calculation to the appropriate algorithm.

    Args:
        models: Dict mapping outersplit_id to fitted model.
        selected_features: Dict mapping outersplit_id to feature list.
        test_data: Dict mapping outersplit_id to test DataFrame.
        train_data: Dict mapping outersplit_id to train DataFrame.
        target_assignments: Target column assignments.
        target_metric: Metric name for scoring.
        positive_class: Positive class label for classification.
        feature_cols: Union of feature columns across splits.
        feature_groups_per_split: Per-split feature groups.
        fi_type: Feature importance type (a ``FIType`` value).
        n_repeats: Number of permutation repeats.
        feature_groups: Explicit feature groups for group permutation.
        random_state: Random seed.
        ml_type: ML task type (required for SHAP).
        **kwargs: Additional kwargs forwarded to the FI function.

    Returns:
        DataFrame with FI results including a ``fi_type`` column.

    Raises:
        ValueError: If ``fi_type`` is not recognised.
    """
    from octopus.types import FIType  # noqa: PLC0415

    fi_type_enum = FIType(fi_type)

    if fi_type_enum in (FIType.PERMUTATION, FIType.GROUP_PERMUTATION):
        resolved_groups = None
        if fi_type_enum == FIType.GROUP_PERMUTATION:
            resolved_groups = feature_groups if feature_groups is not None else merge_feature_groups(feature_groups_per_split)

        result = calculate_fi_permutation(
            models=models,
            selected_features=selected_features,
            test_data=test_data,
            train_data=train_data,
            target_assignments=target_assignments,
            target_metric=target_metric,
            positive_class=positive_class,
            n_repeats=n_repeats,
            random_state=random_state,
            feature_groups=resolved_groups,
            feature_cols=feature_cols,
        )
    elif fi_type_enum == FIType.SHAP:
        result = calculate_fi_shap(
            models=models,
            selected_features=selected_features,
            test_data=test_data,
            ml_type=ml_type,
            feature_cols=feature_cols,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown fi_type '{fi_type}'. Use FIType.PERMUTATION, FIType.GROUP_PERMUTATION, or FIType.SHAP."
        )

    result.insert(0, "fi_type", fi_type_enum.value)
    return result
