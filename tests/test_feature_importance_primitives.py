"""Unit tests for shared feature-importance computation primitives.

Tests ``compute_per_repeat_stats``, ``compute_permutation_single``,
and ``compute_shap_single`` from ``octopus.feature_importance``.
"""

import math

import numpy as np
import pandas as pd
import pytest

from octopus.feature_importance import compute_per_repeat_stats, compute_permutation_single, compute_shap_single
from octopus.modules.octo.training import fi_storage_key, parse_fi_storage_key
from octopus.types import DataPartition, FIComputeMethod, MLType, ShapType

# ═══════════════════════════════════════════════════════════════
# compute_per_repeat_stats
# ═══════════════════════════════════════════════════════════════


class TestComputePerRepeatStats:
    """Tests for compute_per_repeat_stats."""

    def test_returns_expected_keys(self) -> None:
        """Result dict contains all required keys."""
        result = compute_per_repeat_stats([0.1, 0.2, 0.3])
        assert set(result.keys()) == {"importance", "importance_std", "p_value", "ci_lower", "ci_upper"}

    def test_mean_is_correct(self) -> None:
        """Mean importance is computed correctly."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_per_repeat_stats(values)
        assert result["importance"] == pytest.approx(3.0)

    def test_std_is_sample_std(self) -> None:
        """Standard deviation uses ddof=1 (sample std)."""
        values = [2.0, 4.0, 6.0]
        result = compute_per_repeat_stats(values)
        expected_std = float(np.std(values, ddof=1))
        assert result["importance_std"] == pytest.approx(expected_std)

    def test_single_value_returns_nan_std(self) -> None:
        """Single value produces zero importance_std (nan mapped to 0)."""
        result = compute_per_repeat_stats([5.0])
        assert result["importance"] == pytest.approx(5.0)
        assert result["importance_std"] == pytest.approx(0.0)

    def test_single_value_ci_is_nan(self) -> None:
        """Single value produces NaN confidence intervals (n <= 1)."""
        result = compute_per_repeat_stats([5.0])
        assert math.isnan(result["ci_lower"])
        assert math.isnan(result["ci_upper"])

    def test_single_value_p_value_is_nan(self) -> None:
        """Single value produces NaN p_value (std is nan)."""
        result = compute_per_repeat_stats([5.0])
        assert math.isnan(result["p_value"])

    def test_constant_positive_values_p_value_is_zero(self) -> None:
        """Identical positive values (std=0, mean>0) produce p_value=0.0."""
        result = compute_per_repeat_stats([3.0, 3.0, 3.0])
        assert result["p_value"] == pytest.approx(0.0)
        assert result["importance_std"] == pytest.approx(0.0)

    def test_constant_negative_values_p_value_is_one(self) -> None:
        """Identical negative values (std=0, mean<0) produce p_value=1.0."""
        result = compute_per_repeat_stats([-2.0, -2.0])
        assert result["p_value"] == pytest.approx(1.0)
        assert result["importance_std"] == pytest.approx(0.0)

    def test_constant_zero_values_p_value_is_half(self) -> None:
        """Identical zero values (std=0, mean=0) produce p_value=0.5."""
        result = compute_per_repeat_stats([0.0, 0.0, 0.0])
        assert result["p_value"] == pytest.approx(0.5)
        assert result["importance_std"] == pytest.approx(0.0)

    def test_positive_values_low_p_value(self) -> None:
        """Strongly positive values should yield a low p-value."""
        values = [10.0, 11.0, 12.0, 10.5, 11.5]
        result = compute_per_repeat_stats(values)
        assert result["p_value"] < 0.01

    def test_symmetric_values_high_p_value(self) -> None:
        """Values centered around zero should yield a high p-value."""
        values = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        result = compute_per_repeat_stats(values)
        assert result["p_value"] > 0.4

    def test_ci_contains_mean(self) -> None:
        """Confidence interval contains the mean."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_per_repeat_stats(values)
        assert result["ci_lower"] < result["importance"] < result["ci_upper"]

    def test_ci_widens_with_lower_confidence(self) -> None:
        """Lower confidence level produces narrower CI."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result_95 = compute_per_repeat_stats(values, confidence_level=0.95)
        result_80 = compute_per_repeat_stats(values, confidence_level=0.80)
        ci_width_95 = result_95["ci_upper"] - result_95["ci_lower"]
        ci_width_80 = result_80["ci_upper"] - result_80["ci_lower"]
        assert ci_width_95 > ci_width_80

    def test_two_values(self) -> None:
        """Two values produces valid statistics."""
        result = compute_per_repeat_stats([1.0, 3.0])
        assert result["importance"] == pytest.approx(2.0)
        assert not math.isnan(result["p_value"])
        assert not math.isnan(result["ci_lower"])
        assert not math.isnan(result["ci_upper"])


# ═══════════════════════════════════════════════════════════════
# compute_permutation_single
# ═══════════════════════════════════════════════════════════════


def _make_simple_model():
    """Create a simple fitted classifier for testing permutation FI."""
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415

    rng = np.random.default_rng(42)
    n = 100
    X = pd.DataFrame(
        {
            "informative": rng.standard_normal(n),
            "noise": rng.standard_normal(n),
        }
    )
    y = (X["informative"] > 0).astype(int)

    model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1)
    model.fit(X[["informative", "noise"]], y)
    return model, X, y


class TestComputePermutationSingle:
    """Tests for compute_permutation_single."""

    @pytest.fixture()
    def model_and_data(self):
        """Provide a fitted model with informative + noise features."""
        model, X, y = _make_simple_model()
        # Split into train/test
        X_train = X.iloc[:70].copy()
        X_test = X.iloc[70:].copy()
        y_test = y.iloc[70:]

        target_col = "target"
        X_train[target_col] = y.iloc[:70].values
        X_test[target_col] = y_test.values

        return {
            "model": model,
            "X_train": X_train,
            "X_test": X_test,
            "feature_cols": ["informative", "noise"],
            "target_assignments": {"default": target_col},
            "positive_class": 1,
        }

    def test_returns_dataframe_with_expected_columns(self, model_and_data: dict) -> None:
        """Result DataFrame has the expected column schema."""
        d = model_and_data
        result = compute_permutation_single(
            model=d["model"],
            X_test=d["X_test"],
            X_train=d["X_train"],
            feature_cols=d["feature_cols"],
            target_metric="AUCROC",
            target_assignments=d["target_assignments"],
            positive_class=d["positive_class"],
            n_repeats=5,
        )
        expected_cols = {"feature", "importance", "importance_std", "p_value", "ci_lower", "ci_upper"}
        assert set(result.columns) == expected_cols

    def test_returns_one_row_per_feature(self, model_and_data: dict) -> None:
        """One row per feature when no groups are provided."""
        d = model_and_data
        result = compute_permutation_single(
            model=d["model"],
            X_test=d["X_test"],
            X_train=d["X_train"],
            feature_cols=d["feature_cols"],
            target_metric="AUCROC",
            target_assignments=d["target_assignments"],
            positive_class=d["positive_class"],
            n_repeats=3,
        )
        assert len(result) == len(d["feature_cols"])
        assert set(result["feature"]) == set(d["feature_cols"])

    def test_informative_feature_has_higher_importance(self, model_and_data: dict) -> None:
        """Informative feature should have higher importance than noise."""
        d = model_and_data
        result = compute_permutation_single(
            model=d["model"],
            X_test=d["X_test"],
            X_train=d["X_train"],
            feature_cols=d["feature_cols"],
            target_metric="AUCROC",
            target_assignments=d["target_assignments"],
            positive_class=d["positive_class"],
            n_repeats=10,
        )
        imp_informative = result.loc[result["feature"] == "informative", "importance"].iloc[0]
        imp_noise = result.loc[result["feature"] == "noise", "importance"].iloc[0]
        assert imp_informative > imp_noise

    def test_feature_groups_adds_group_rows(self, model_and_data: dict) -> None:
        """Feature groups produce additional rows in the result."""
        d = model_and_data
        groups = {"all_features": ["informative", "noise"]}
        result = compute_permutation_single(
            model=d["model"],
            X_test=d["X_test"],
            X_train=d["X_train"],
            feature_cols=d["feature_cols"],
            target_metric="AUCROC",
            target_assignments=d["target_assignments"],
            positive_class=d["positive_class"],
            n_repeats=3,
            feature_groups=groups,
        )
        # 2 individual features + 1 group
        assert len(result) == 3
        assert "all_features" in result["feature"].values

    def test_reproducible_with_same_seed(self, model_and_data: dict) -> None:
        """Same random_state produces identical results."""
        d = model_and_data
        kwargs = {
            "model": d["model"],
            "X_test": d["X_test"],
            "X_train": d["X_train"],
            "feature_cols": d["feature_cols"],
            "target_metric": "AUCROC",
            "target_assignments": d["target_assignments"],
            "positive_class": d["positive_class"],
            "n_repeats": 5,
            "random_state": 123,
        }
        result1 = compute_permutation_single(**kwargs)
        result2 = compute_permutation_single(**kwargs)
        pd.testing.assert_frame_equal(result1, result2)

    def test_different_seed_produces_different_results(self, model_and_data: dict) -> None:
        """Different random_state produces different results."""
        d = model_and_data
        common = {
            "model": d["model"],
            "X_test": d["X_test"],
            "X_train": d["X_train"],
            "feature_cols": d["feature_cols"],
            "target_metric": "AUCROC",
            "target_assignments": d["target_assignments"],
            "positive_class": d["positive_class"],
            "n_repeats": 5,
        }
        result1 = compute_permutation_single(**common, random_state=1)
        result2 = compute_permutation_single(**common, random_state=99)
        # Results should differ (at least importance values)
        vals1 = result1.set_index("feature")["importance"]
        vals2 = result2.set_index("feature")["importance"]
        assert not np.allclose(np.asarray(vals1.values), np.asarray(vals2.values))

    def test_group_with_partial_overlap(self, model_and_data: dict) -> None:
        """Feature groups with features not in feature_cols are ignored."""
        d = model_and_data
        groups = {"mixed_group": ["informative", "nonexistent_feature"]}
        result = compute_permutation_single(
            model=d["model"],
            X_test=d["X_test"],
            X_train=d["X_train"],
            feature_cols=d["feature_cols"],
            target_metric="AUCROC",
            target_assignments=d["target_assignments"],
            positive_class=d["positive_class"],
            n_repeats=3,
            feature_groups=groups,
        )
        # 2 individual features + 1 group (mixed_group has 1 valid feature)
        assert len(result) == 3
        assert "mixed_group" in result["feature"].values

    def test_importance_values_are_finite(self, model_and_data: dict) -> None:
        """All importance and importance_std values are finite numbers."""
        d = model_and_data
        result = compute_permutation_single(
            model=d["model"],
            X_test=d["X_test"],
            X_train=d["X_train"],
            feature_cols=d["feature_cols"],
            target_metric="AUCROC",
            target_assignments=d["target_assignments"],
            positive_class=d["positive_class"],
            n_repeats=5,
        )
        assert np.all(np.isfinite(result["importance"].values))
        assert np.all(np.isfinite(result["importance_std"].values))


# ── fi_storage_key / parse_fi_storage_key tests ─────────────────


class TestFiStorageKey:
    """Tests for fi_storage_key and parse_fi_storage_key."""

    def test_partition_less_key(self) -> None:
        """Partition-less methods produce a single-segment key."""
        assert fi_storage_key(FIComputeMethod.INTERNAL) == "internal"
        assert fi_storage_key(FIComputeMethod.CONSTANT) == "constant"

    def test_two_part_key(self) -> None:
        """Partition-aware methods produce method_partition keys."""
        assert fi_storage_key(FIComputeMethod.PERMUTATION, DataPartition.DEV) == "permutation_dev"
        assert fi_storage_key(FIComputeMethod.SHAP, "test") == "shap_test"

    def test_three_part_key(self) -> None:
        """fi_storage_key with stat produces method_partition_stat."""
        assert fi_storage_key(FIComputeMethod.PERMUTATION, DataPartition.DEV, "mean") == "permutation_dev_mean"
        assert fi_storage_key(FIComputeMethod.SHAP, "test", "count") == "shap_test_count"

    def test_data_partition_enum_and_str_produce_same_key(self) -> None:
        """DataPartition enum and equivalent string produce identical keys."""
        assert fi_storage_key(FIComputeMethod.PERMUTATION, DataPartition.DEV) == fi_storage_key(
            FIComputeMethod.PERMUTATION, "dev"
        )
        assert fi_storage_key(FIComputeMethod.SHAP, DataPartition.TEST, "mean") == fi_storage_key(
            FIComputeMethod.SHAP, "test", "mean"
        )


class TestParseFiStorageKey:
    """Tests for parse_fi_storage_key, including three-part keys."""

    def test_partition_less_key(self) -> None:
        """Partition-less keys return the method and 'train'."""
        method, dataset = parse_fi_storage_key("internal")
        assert method == "internal"
        assert dataset == "train"

    def test_two_part_key(self) -> None:
        """Two-part keys return method and partition."""
        method, dataset = parse_fi_storage_key("permutation_dev")
        assert method == "permutation"
        assert dataset == "dev"

    def test_three_part_key_mean(self) -> None:
        """Three-part keys strip the stat suffix, returning only the partition."""
        method, dataset = parse_fi_storage_key("permutation_dev_mean")
        assert method == "permutation"
        assert dataset == "dev"

    def test_three_part_key_count(self) -> None:
        """Three-part keys with 'count' stat suffix are handled correctly."""
        method, dataset = parse_fi_storage_key("shap_test_count")
        assert method == "shap"
        assert dataset == "test"

    def test_lofo_key(self) -> None:
        """LOFO keys are parsed correctly."""
        method, dataset = parse_fi_storage_key("lofo_dev")
        assert method == "lofo"
        assert dataset == "dev"

    def test_lofo_three_part_key(self) -> None:
        """LOFO three-part keys strip stat suffix."""
        method, dataset = parse_fi_storage_key("lofo_dev_mean")
        assert method == "lofo"
        assert dataset == "dev"


# ═══════════════════════════════════════════════════════════════
# compute_shap_single
# ═══════════════════════════════════════════════════════════════


def _make_binary_classifier():
    """Create a simple binary classifier for SHAP testing."""
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415

    rng = np.random.default_rng(42)
    n = 150
    X = pd.DataFrame(
        {
            "informative": rng.standard_normal(n),
            "noise": rng.standard_normal(n),
            "constant": np.ones(n),
        }
    )
    y = (X["informative"] > 0).astype(int)

    model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
    model.fit(X, y)
    return model, X, y


def _make_multiclass_classifier():
    """Create a multiclass classifier for testing 3D SHAP aggregation."""
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415

    rng = np.random.default_rng(123)
    n = 200
    X = pd.DataFrame(
        {
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
            "f3": rng.standard_normal(n),
        }
    )
    # Create 3 classes based on feature combinations
    y = np.zeros(n, dtype=int)
    y[X["f1"] > 0.5] = 1
    y[X["f2"] < -0.5] = 2

    model = RandomForestClassifier(n_estimators=10, random_state=123, n_jobs=1)
    model.fit(X, y)
    return model, X, y


def _make_regressor():
    """Create a simple regressor for SHAP testing."""
    from sklearn.ensemble import RandomForestRegressor  # noqa: PLC0415

    rng = np.random.default_rng(456)
    n = 150
    X = pd.DataFrame(
        {
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
        }
    )
    y = X["f1"] * 2 + X["f2"] + rng.standard_normal(n) * 0.1

    model = RandomForestRegressor(n_estimators=10, random_state=456, n_jobs=1)
    model.fit(X, y)
    return model, X, y


@pytest.mark.slow
class TestComputeShapSingle:
    """Tests for compute_shap_single.

    Marked as slow because SHAP computations are inherently expensive.
    These tests cover the key untested code paths identified in doc 23 F9:
    - 3D SHAP shape handling for multiclass
    - ShapType.AUTO mode
    - threshold_ratio feature filter
    - Basic functionality across model types
    """

    def test_returns_dataframe_with_expected_columns(self) -> None:
        """Result DataFrame has the expected column schema."""
        model, X, _y = _make_binary_classifier()
        result = compute_shap_single(
            model=model,
            X=X.iloc[:30],
            feature_names=list(X.columns),
            shap_type=ShapType.AUTO,
            X_background=X.iloc[:20],
            max_samples=None,
            threshold_ratio=None,
            ml_type=MLType.BINARY,
        )
        assert set(result.columns) == {"feature", "importance"}
        assert len(result) == 3  # One row per feature

    def test_multiclass_3d_aggregation(self) -> None:
        """Multiclass models produce 3D SHAP values that are correctly aggregated.

        This is the highest-value test from doc 23 F9.
        SHAP returns shape (n_samples, n_features, n_classes) for multiclass.
        The aggregation should be: abs → mean over classes → mean over samples.
        """
        model, X, _y = _make_multiclass_classifier()

        result = compute_shap_single(
            model=model,
            X=X.iloc[:40],
            feature_names=list(X.columns),
            shap_type=ShapType.AUTO,
            X_background=X.iloc[:20],
            max_samples=None,
            threshold_ratio=None,
            ml_type=MLType.MULTICLASS,
        )

        # Should aggregate to one importance per feature
        assert len(result) == 3
        assert set(result["feature"]) == {"f1", "f2", "f3"}

        # All importance values should be non-negative (because we take abs)
        assert all(result["importance"] >= 0)

        # Importance values should be finite
        assert all(np.isfinite(result["importance"]))

    def test_threshold_ratio_filters_low_importance_features(self) -> None:
        """threshold_ratio parameter filters features below the threshold."""
        model, X, _y = _make_binary_classifier()

        # Without threshold - all features returned
        result_no_threshold = compute_shap_single(
            model=model,
            X=X.iloc[:30],
            feature_names=list(X.columns),
            shap_type=ShapType.AUTO,
            X_background=X.iloc[:20],
            max_samples=None,
            threshold_ratio=None,
            ml_type=MLType.BINARY,
        )
        assert len(result_no_threshold) == 3

        # With aggressive threshold - only high-importance features
        # threshold_ratio = 0.5 means only features with importance > 50% of max
        result_with_threshold = compute_shap_single(
            model=model,
            X=X.iloc[:30],
            feature_names=list(X.columns),
            shap_type=ShapType.AUTO,
            X_background=X.iloc[:20],
            max_samples=None,
            threshold_ratio=0.5,
            ml_type=MLType.BINARY,
        )

        # Should have fewer features (or equal if all pass threshold)
        assert len(result_with_threshold) <= len(result_no_threshold)

        # All returned features should have importance > threshold * max_importance
        if len(result_with_threshold) > 0:
            max_imp = result_with_threshold["importance"].max()
            threshold = 0.5 * max_imp
            assert all(result_with_threshold["importance"] >= threshold)

    def test_shap_auto_mode_works(self) -> None:
        """ShapType.AUTO successfully creates an explainer and computes values."""
        model, X, _y = _make_binary_classifier()

        result = compute_shap_single(
            model=model,
            X=X.iloc[:25],
            feature_names=list(X.columns),
            shap_type=ShapType.AUTO,
            X_background=X.iloc[:15],
            max_samples=None,
            threshold_ratio=None,
            ml_type=MLType.BINARY,
        )

        assert len(result) == 3
        assert all(np.isfinite(result["importance"]))

    def test_shap_kernel_mode_works(self) -> None:
        """ShapType.KERNEL mode works correctly."""
        model, X, _y = _make_binary_classifier()

        result = compute_shap_single(
            model=model,
            X=X.iloc[:20],
            feature_names=list(X.columns),
            shap_type=ShapType.KERNEL,
            X_background=X.iloc[:10],
            max_samples=None,
            threshold_ratio=None,
            ml_type=MLType.BINARY,
        )

        assert len(result) == 3
        assert all(np.isfinite(result["importance"]))

    def test_regression_model(self) -> None:
        """SHAP works correctly for regression models."""
        model, X, _y = _make_regressor()

        result = compute_shap_single(
            model=model,
            X=X.iloc[:30],
            feature_names=list(X.columns),
            shap_type=ShapType.AUTO,
            X_background=X.iloc[:20],
            max_samples=None,
            threshold_ratio=None,
            ml_type=MLType.REGRESSION,
        )

        assert len(result) == 2  # Two features
        assert set(result["feature"]) == {"f1", "f2"}
        assert all(np.isfinite(result["importance"]))

        # f1 should have higher importance than f2 (coefficient is 2 vs 1)
        imp_f1 = result.loc[result["feature"] == "f1", "importance"].iloc[0]
        imp_f2 = result.loc[result["feature"] == "f2", "importance"].iloc[0]
        assert imp_f1 > imp_f2

    def test_importance_values_are_sorted_descending(self) -> None:
        """Result is sorted by importance (descending)."""
        model, X, _y = _make_binary_classifier()

        result = compute_shap_single(
            model=model,
            X=X.iloc[:30],
            feature_names=list(X.columns),
            shap_type=ShapType.AUTO,
            X_background=X.iloc[:20],
            max_samples=None,
            threshold_ratio=None,
            ml_type=MLType.BINARY,
        )

        importances = result["importance"].values
        # Check that each value is >= the next one
        for i in range(len(importances) - 1):
            assert importances[i] >= importances[i + 1]

    def test_max_samples_limits_computation(self) -> None:
        """max_samples parameter limits the number of samples used."""
        model, X, _y = _make_binary_classifier()

        # Test with small max_samples - should still work
        result = compute_shap_single(
            model=model,
            X=X.iloc[:50],
            feature_names=list(X.columns),
            shap_type=ShapType.KERNEL,
            X_background=X.iloc[:20],
            max_samples=10,  # Only compute SHAP for 10 samples
            threshold_ratio=None,
            ml_type=MLType.BINARY,
        )

        assert len(result) == 3
        assert all(np.isfinite(result["importance"]))

    def test_background_data_affects_results(self) -> None:
        """Different background data produces different SHAP values."""
        model, X, _y = _make_binary_classifier()

        result1 = compute_shap_single(
            model=model,
            X=X.iloc[50:70],
            feature_names=list(X.columns),
            shap_type=ShapType.KERNEL,
            X_background=X.iloc[:20],
            max_samples=None,
            threshold_ratio=None,
            ml_type=MLType.BINARY,
        )

        result2 = compute_shap_single(
            model=model,
            X=X.iloc[50:70],
            feature_names=list(X.columns),
            shap_type=ShapType.KERNEL,
            X_background=X.iloc[100:120],  # Different background
            max_samples=None,
            threshold_ratio=None,
            ml_type=MLType.BINARY,
        )

        # Results should differ due to different background
        imp1 = result1.set_index("feature")["importance"]
        imp2 = result2.set_index("feature")["importance"]
        assert not np.allclose(np.asarray(imp1.values), np.asarray(imp2.values))
