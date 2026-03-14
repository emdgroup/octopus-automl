"""Feature importance algorithms for octopus.predict.

Architecture
------------
**Layer 1 — Computation Primitives** (shared by training *and* predict):

- ``compute_per_repeat_stats``  — t-distribution stats from repeat values
- ``compute_internal_fi``       — tree / linear model attribute extraction
- ``compute_shap_single``       — SHAP explainer for a single model + dataset
- ``compute_permutation_single`` — custom draw-from-pool permutation for a single model

**Layer 2 — Multi-split Orchestration** (predict only):

- ``calculate_fi_permutation``  — per-split + ensemble permutation FI
- ``calculate_fi_shap``         — per-split + ensemble SHAP FI

Layer 1 primitives return DataFrames with column ``"importance"`` (not
``"importance_mean"``) so that Bag aggregation in the training pipeline
works unchanged.  Layer 2 renames to ``"importance_mean"`` for the predict
schema.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from octopus.metrics.utils import get_score_from_model
from octopus.types import MLType

__all__ = [
    "calculate_fi_permutation",
    "calculate_fi_shap",
    "compute_internal_fi",
    "compute_per_repeat_stats",
    "compute_permutation_single",
    "compute_shap_single",
]

logger = logging.getLogger(__name__)

# Fixed confidence level for CI calculations
_CONFIDENCE_LEVEL = 0.95

# ═══════════════════════════════════════════════════════════════
# Layer 1 — Computation Primitives
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
    arr = np.array(values)
    mean_val = float(np.mean(arr))
    n = len(arr)
    std_val = float(np.std(arr, ddof=1)) if n > 1 else np.nan

    # One-sample t-test: H0 = importance <= 0
    p_value: float = np.nan
    if not np.isnan(std_val) and std_val > 0:
        from scipy import stats as sp_stats  # noqa: PLC0415

        t_stat = mean_val / (std_val / math.sqrt(n))
        p_value = float(sp_stats.t.sf(t_stat, n - 1))
    elif std_val == 0:
        p_value = 0.5

    # Confidence interval using t-distribution
    if any(np.isnan(val) for val in [std_val, mean_val]) or n <= 1:
        ci_lower: float = np.nan
        ci_upper: float = np.nan
    else:
        from scipy import stats as sp_stats  # noqa: PLC0415

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


def compute_internal_fi(
    model: Any,
    feature_names: list[str],
    ml_type: MLType | None = None,
) -> pd.DataFrame:
    """Extract internal feature importance from a fitted model.

    Handles tree-based models (``feature_importances_``), linear models
    (``abs(coef_)``), and returns an empty DataFrame for unsupported models
    or time-to-event models.

    Args:
        model: A fitted scikit-learn-compatible model.
        feature_names: List of feature column names matching the model.
        ml_type: The ML task type.  If ``MLType.TIMETOEVENT``, returns
            an empty DataFrame immediately.

    Returns:
        DataFrame with columns ``["feature", "importance"]``.
    """
    empty = pd.DataFrame(columns=["feature", "importance"])

    # Time-to-event: not supported
    if ml_type == MLType.TIMETOEVENT:
        logger.warning("Internal feature importances not available for timetoevent.")
        return empty

    # Tree-based models
    if hasattr(model, "feature_importances_"):
        fi = np.asarray(model.feature_importances_)
        return pd.DataFrame({"feature": feature_names, "importance": fi})

    # Linear models
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 1:
            importance = np.abs(coef)
        else:
            importance = np.mean(np.abs(coef), axis=0)

        if len(importance) != len(feature_names):
            logger.warning(
                "Length mismatch between coefficients (%d) and feature columns (%d). Skipping internal importances.",
                len(importance),
                len(feature_names),
            )
            return empty

        return pd.DataFrame({"feature": feature_names, "importance": importance})

    # Fallback
    logger.warning("Internal feature importances not available for this estimator.")
    return empty


def compute_shap_single(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    feature_names: list[str],
    shap_type: str = "auto",
    X_background: np.ndarray | pd.DataFrame | None = None,
    max_samples: int | None = None,
    threshold_ratio: float | None = None,
    ml_type: MLType | None = None,
) -> pd.DataFrame:
    """Compute SHAP feature importance for a single model and dataset.

    Args:
        model: A fitted model.
        X: Evaluation data (features only).
        feature_names: Feature column names matching columns/order of *X*.
        shap_type: SHAP explainer type.  One of:
            - ``"auto"`` — let SHAP choose (fast for tree/linear models).
            - ``"kernel"`` — ``KernelExplainer`` (model-agnostic, slower).
            - ``"permutation"`` — ``PermutationExplainer``.
            - ``"exact"`` — ``ExactExplainer``.
        X_background: Background dataset for auto/kernel explainers.
            If None for kernel, evaluation data is sampled as background.
        max_samples: If set, subsample *X* to at most this many rows.
        threshold_ratio: If set (e.g. ``1/1000``), drop features whose
            importance is below ``max_importance * threshold_ratio``.
            Use ``None`` to keep all features.
        ml_type: ML type, used to choose predict vs predict_proba.

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

    # Build prediction function
    # Classifiers: use predict_proba to give SHAP a continuous output.
    # model.predict returns class labels which are not meaningful for SHAP.
    # Regressors / time-to-event: use model.predict (scalar output).
    # Note: we use ml_type (not hasattr(model, "classes_")) because some
    # regressors (e.g. CatBoostRegressor) expose a classes_ attribute.
    is_classifier = (
        ml_type in (MLType.BINARY, MLType.MULTICLASS) if ml_type is not None else hasattr(model, "predict_proba")
    )

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
    if shap_type == "auto":
        bg_data = bg if bg is not None else x_arr
        bg_df = pd.DataFrame(bg_data, columns=feature_names)
        x_df = pd.DataFrame(x_arr, columns=feature_names)
        try:
            # Try model directly — SHAP can auto-detect Tree/Linear explainers for speed
            explainer = shap.Explainer(model, bg_df)
            sv = explainer(x_df)
            shap_values = np.asarray(sv.values)
        except Exception as e1:
            logger.debug("SHAP auto explainer with model failed: %s. Falling back to callable wrapper.", e1)
            # Fall back to callable approach (predict_fn wraps numpy→DataFrame internally)
            if bg is None:
                bg = x_arr
            explainer = shap.Explainer(predict_fn, bg)
            sv = explainer(x_arr)
            shap_values = np.asarray(sv.values)
    elif shap_type == "kernel":
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
    elif shap_type == "permutation":
        explainer = shap.explainers.Permutation(predict_fn, x_arr)
        sv = explainer(x_arr)
        shap_values = np.asarray(sv.values)
    elif shap_type == "exact":
        explainer = shap.explainers.Exact(predict_fn, x_arr)
        sv = explainer(x_arr)
        shap_values = np.asarray(sv.values)
    else:
        raise ValueError(f"Unknown shap_type '{shap_type}'. Use 'auto', 'kernel', 'permutation', or 'exact'.")

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
# Layer 2 — Multi-split Orchestration (predict only)
# ═══════════════════════════════════════════════════════════════


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
    then aggregates per-split results into ensemble rows.

    Features present in ``feature_cols`` but not selected in a given split
    receive zero importance for that split, ensuring the result covers the
    union of all input features.

    Args:
        models: Dict mapping outersplit_id to fitted model.
        selected_features: Dict mapping outersplit_id to feature list.
        test_data: Dict mapping outersplit_id to test DataFrame.
        train_data: Dict mapping outersplit_id to train DataFrame.
        target_assignments: Dict mapping semantic target roles to column
            names.  For single-target tasks: ``{"default": "target_col"}``.
            For time-to-event: ``{"duration": "...", "event": "..."}``.
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
    # Per-split: call compute_permutation_single, collect results
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

    # Build per-split rows (rename importance -> importance_mean for predict schema)
    rows: list[dict[str, Any]] = []
    per_split_means: dict[str, list[float]] = {}

    for split_id in sorted(per_split_dfs.keys()):
        split_fi = per_split_dfs[split_id]
        for _, row in split_fi.iterrows():
            feat = row["feature"]
            imp_mean = row["importance"]
            rows.append(
                {
                    "fi_source": split_id,
                    "feature": feat,
                    "importance_mean": imp_mean,
                    "importance_std": row["importance_std"],
                    "p_value": row["p_value"],
                    "ci_lower": row["ci_lower"],
                    "ci_upper": row["ci_upper"],
                }
            )
            if feat not in per_split_means:
                per_split_means[feat] = []
            per_split_means[feat].append(imp_mean)

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

    For each outer split, delegates to ``compute_shap_single`` and then
    aggregates per-split results into ensemble rows.

    Args:
        models: Dict mapping outersplit_id to fitted model.
        selected_features: Dict mapping outersplit_id to feature list.
        test_data: Dict mapping outersplit_id to test DataFrame.
        shap_type: SHAP explainer type (``'kernel'``, ``'permutation'``,
            ``'exact'``).
        max_samples: Maximum number of evaluation samples per split.
        background_size: Maximum background dataset size for kernel explainer.

    Returns:
        DataFrame with columns: fi_source, feature, importance_mean,
        importance_std.  Per-split rows have NaN for importance_std;
        ensemble rows have the std of per-split means.
    """
    # Collect per-split SHAP values
    split_shap: dict[int, dict[str, float]] = {}

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
        )

        for _, row in fi_df.iterrows():
            split_shap.setdefault(split_id, {})[row["feature"]] = float(row["importance"])

    # Build per-split rows
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
