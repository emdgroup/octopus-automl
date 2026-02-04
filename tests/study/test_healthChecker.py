"""Test OctoData healthChecker."""

import numpy as np
import pandas as pd
import pytest

from octopus.study.healthChecker import OctoDataHealthChecker


@pytest.fixture
def sample_data():
    """Create sample data."""
    return pd.DataFrame(
        {
            "id": range(1, 101),
            "feature1": np.random.rand(100),
            "feature2": np.random.randint(0, 5, 100),
            "feature3": ["A", "B", "C"] * 33 + ["A"],
            "target": np.random.randint(0, 2, 100),
            "sample_id": [f"S{i}" for i in range(1, 101)],
        }
    )


@pytest.fixture
def health_checker(sample_data):
    """Create health checker instance."""
    return OctoDataHealthChecker(
        data=sample_data,
        feature_cols=["feature1", "feature2", "feature3"],
        target_cols=["target"],
        row_id="id",
        sample_id="sample_id",
        stratification_column=None,
    )


def test_initialization(health_checker):
    """Test initialization."""
    assert isinstance(health_checker, OctoDataHealthChecker)
    assert len(health_checker.issues) == 0


def test_add_issue(health_checker):
    """Test add issues."""
    health_checker.add_issue(
        category="test",
        issue_type="test_issue",
        affected_items=["item1", "item2"],
        severity="Info",
        description="Test description",
        action="Test action",
    )
    assert len(health_checker.issues) == 1
    assert health_checker.issues[0]["Category"] == "test"
    assert health_checker.issues[0]["Issue Type"] == "test_issue"
    assert health_checker.issues[0]["Affected Items"] == "item1, item2"


def test_generate_report(health_checker):
    """Test report creation."""
    report = health_checker.generate_report()
    assert isinstance(report, pd.DataFrame)
    assert len(report) > 0


def test_check_critical_column_missing_values(health_checker):
    """Test critical column missing values check."""
    health_checker.data.loc[0, "target"] = np.nan
    health_checker._check_critical_column_missing_values()
    assert any(issue["Issue Type"] == "critical_missing_values" for issue in health_checker.issues)


def test_check_feature_cols_missing_values(health_checker):
    """Test feature column missing values check."""
    health_checker.data.loc[:25, "feature1"] = np.nan
    health_checker._check_feature_cols_missing_values()
    assert any(issue["Issue Type"] == "high_missing_values" for issue in health_checker.issues)


def test_check_row_missing_values(health_checker):
    """Test row missing values check."""
    health_checker.data.iloc[0, 1:4] = np.nan
    health_checker._check_row_missing_values()
    assert any(issue["Issue Type"] == "low_missing_values" for issue in health_checker.issues)


def test_check_int_col_with_few_uniques(health_checker):
    """Test int with few uniques check."""
    health_checker._check_int_col_with_few_uniques()
    assert any(issue["Issue Type"] == "int_columns_with_few_uniques" for issue in health_checker.issues)


def test_check_duplicated_features(health_checker):
    """Test duplicated features check."""
    health_checker.data = pd.concat([health_checker.data] * 2).reset_index(drop=True)
    health_checker._check_duplicated_features()
    assert any(issue["Issue Type"] == "duplicated_features" for issue in health_checker.issues)


def test_check_feature_feature_correlation(health_checker):
    """Test check feature feature correlation check."""
    health_checker.data["feature4"] = health_checker.data["feature1"] * 2
    health_checker.feature_cols.append("feature4")
    health_checker._check_feature_feature_correlation()
    assert any(issue["Issue Type"] == "high_correlation" for issue in health_checker.issues)


def test_check_identical_features(health_checker):
    """Test identical features check."""
    health_checker.data["feature5"] = health_checker.data["feature1"]
    health_checker.feature_cols.append("feature5")
    health_checker._check_identical_features()
    assert any(issue["Issue Type"] == "identical_features" for issue in health_checker.issues)


def test_check_duplicated_rows(health_checker):
    """Test duplicated rows check."""
    health_checker.data = pd.concat([health_checker.data.iloc[:5]] * 2).reset_index(drop=True)
    health_checker._check_duplicated_rows()
    assert any(issue["Issue Type"] == "duplicated_rows" for issue in health_checker.issues)


def test_check_infinity_values(health_checker):
    """Test infinity values check."""
    health_checker.data.loc[0, "feature1"] = np.inf
    health_checker._check_infinity_values()
    assert any(issue["Issue Type"] == "infinity_values" for issue in health_checker.issues)


def test_check_string_mismatch(health_checker):
    """Test string mismatch check."""
    health_checker.data["feature6"] = ["apple", "aple", "appl", "banana"] * 25
    health_checker.feature_cols.append("feature6")
    health_checker._check_string_mismatch()
    assert any(issue["Issue Type"] == "string_mismatch" for issue in health_checker.issues)


def test_check_string_out_of_bounds(health_checker):
    """Test string out of bounds check."""
    health_checker.data["feature7"] = [
        "short",
        "medium",
        "very_long_string_that_exceeds_threshold",
    ] * 33 + ["short"]
    health_checker.feature_cols.append("feature7")
    health_checker._check_string_out_of_bounds()
    assert any(issue["Issue Type"] == "string_out_of_bounds" for issue in health_checker.issues)


def test_check_class_imbalance(health_checker):
    """Test class imbalance check."""
    health_checker.data["target"] = [0] * 90 + [1] * 10
    health_checker._check_class_imbalance()
    assert any(issue["Issue Type"] == "class_imbalance" for issue in health_checker.issues)


def test_check_high_cardinality(health_checker):
    """Test high cardinality check."""
    health_checker.data["feature8"] = [f"cat_{i}" for i in range(80)] + ["cat_0"] * 20
    health_checker.feature_cols.append("feature8")
    health_checker._check_high_cardinality()
    assert any(issue["Issue Type"] == "high_cardinality" for issue in health_checker.issues)


def test_check_target_leakage(health_checker):
    """Test target leakage check."""
    health_checker.data["target_numeric"] = health_checker.data["target"].astype(float)
    health_checker.target_cols = ["target_numeric"]
    health_checker.data["feature9"] = health_checker.data["target_numeric"] * 0.99 + np.random.randn(100) * 0.01
    health_checker.feature_cols.append("feature9")
    health_checker._check_target_leakage()
    assert any(issue["Issue Type"] == "target_leakage" for issue in health_checker.issues)


def test_check_target_distribution(health_checker):
    """Test target distribution check."""
    health_checker.data["target_regression"] = np.exp(np.random.randn(100) * 2)  # Right-skewed
    health_checker.target_cols = ["target_regression"]
    health_checker._check_target_distribution()
    assert any(issue["Issue Type"] == "problematic_distribution" for issue in health_checker.issues)


def test_check_minimum_samples(sample_data):
    """Test minimum samples check."""
    health_checker_sufficient = OctoDataHealthChecker(
        data=sample_data,
        feature_cols=["feature1"],
        target_cols=["target"],
        row_id="id",
        sample_id="sample_id",
        stratification_column=None,
    )
    health_checker_sufficient._check_minimum_samples()
    assert not any(issue["Issue Type"] == "insufficient_samples" for issue in health_checker_sufficient.issues)

    small_data = sample_data.head(10)
    health_checker_insufficient = OctoDataHealthChecker(
        data=small_data,
        feature_cols=["feature1"],
        target_cols=["target"],
        row_id="id",
        sample_id="sample_id",
        stratification_column=None,
    )
    health_checker_insufficient._check_minimum_samples()
    assert any(issue["Issue Type"] == "insufficient_samples" for issue in health_checker_insufficient.issues)
    assert any("10" in issue["Description"] for issue in health_checker_insufficient.issues)


def test_check_row_id_unique(sample_data):
    """Test row_id unique check."""
    health_checker_unique = OctoDataHealthChecker(
        data=sample_data,
        feature_cols=["feature1"],
        target_cols=["target"],
        row_id="id",
        sample_id="sample_id",
        stratification_column=None,
    )
    health_checker_unique._check_row_id_unique()
    assert not any(issue["Issue Type"] == "duplicate_row_ids" for issue in health_checker_unique.issues)

    duplicate_data = sample_data.copy()
    duplicate_data.loc[5, "id"] = duplicate_data.loc[0, "id"]
    health_checker_duplicate = OctoDataHealthChecker(
        data=duplicate_data,
        feature_cols=["feature1"],
        target_cols=["target"],
        row_id="id",
        sample_id="sample_id",
        stratification_column=None,
    )
    health_checker_duplicate._check_row_id_unique()
    assert any(issue["Issue Type"] == "duplicate_row_ids" for issue in health_checker_duplicate.issues)
    assert any("duplicate" in issue["Description"].lower() for issue in health_checker_duplicate.issues)


def test_check_features_not_all_null(sample_data):
    """Test features not all null check."""
    health_checker_valid = OctoDataHealthChecker(
        data=sample_data,
        feature_cols=["feature1", "feature2"],
        target_cols=["target"],
        row_id="id",
        sample_id="sample_id",
        stratification_column=None,
    )
    health_checker_valid._check_features_not_all_null()
    assert not any(issue["Issue Type"] == "all_null_features" for issue in health_checker_valid.issues)

    null_data = sample_data.copy()
    null_data["feature1"] = np.nan
    health_checker_null = OctoDataHealthChecker(
        data=null_data,
        feature_cols=["feature1", "feature2"],
        target_cols=["target"],
        row_id="id",
        sample_id="sample_id",
        stratification_column=None,
    )
    health_checker_null._check_features_not_all_null()
    assert any(issue["Issue Type"] == "all_null_features" for issue in health_checker_null.issues)
    assert any("feature1" in issue["Affected Items"] for issue in health_checker_null.issues)
