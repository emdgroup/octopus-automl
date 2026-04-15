"""Shared feature-importance computation primitives.

This module contains the shared computation functions used by both the
poststudy package (``octopus.poststudy.feature_importance``) and the tako
training pipeline (``octopus.modules.tako.training``).

Architecture
------------
Only two feature-importance algorithms are truly general-purpose and
shared between poststudy and tako:

- **Permutation FI** — ``compute_permutation_single``
- **SHAP FI** — ``compute_shap_single``

A helper function ``compute_per_repeat_stats`` is a transitive dependency
of ``compute_permutation_single`` and lives here as well.

Tako-only primitives (``compute_internal_fi``, LOFO, constant) remain in
``octopus.modules.tako.training`` because they depend on training-time
context and are never called from the poststudy layer.

Multi-split orchestrators (``calculate_fi_permutation``, ``calculate_fi_shap``)
remain in ``octopus.poststudy.feature_importance`` — they coordinate
multi-split FI across outer splits and are poststudy-only.
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
    "compute_per_repeat_stats",
    "compute_permutation_single",
    "compute_shap_single",
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
