"""Test OctoData validator."""

import numpy as np
import pandas as pd
import pytest

from octopus.study.data_validation import (
    _validate_column_dtypes,
    _validate_columns_exist,
    _validate_duplicated_columns,
    _validate_feature_target_overlap,
    _validate_nonempty_dataframe,
    _validate_reserved_column_conflicts,
    _validate_stratification_col,
    validate_data,
)


@pytest.fixture
def sample_data():
    """Create sample data."""
    return pd.DataFrame(
        {
            "id": range(1, 101),
            "sample_id_col": [f"S{i}" for i in range(1, 101)],
            "feature1": np.random.rand(100),
            "feature2": np.random.randint(0, 5, 100),
            "feature3": ["A", "B", "C"] * 33 + ["A"],
            "target": np.random.randint(0, 2, 100),
            "time": np.random.rand(100) * 10,
            "strat": ["X", "Y"] * 50,
        }
    ).astype({"strat": "category", "feature3": "category"})


def test_validate_columns_exist_valid(sample_data):
    """Test column exists validation with valid columns."""
    result = _validate_columns_exist(
        sample_data,
        ["feature1", "feature2", "feature3"],
        None,
        None,
        "target",
        "sample_id_col",
        "id",
        "strat",
    )
    assert result is None


def test_validate_columns_exist_missing(sample_data):
    """Test column exists validation with missing column."""
    result = _validate_columns_exist(
        sample_data,
        ["feature1", "non_existent_column"],
        None,
        None,
        "target",
        "sample_id_col",
        "id",
        "strat",
    )
    assert "non_existent_column" in result


@pytest.mark.parametrize(
    "duplicate_setup,should_fail",
    [
        (None, False),
        ("feature_to_target", True),
    ],
)
def test_validate_duplicated_columns(duplicate_setup, should_fail):
    """Test duplicated columns validation."""
    feature_cols = ["feature1", "feature2", "feature3"]

    if duplicate_setup == "feature_to_target":
        feature_cols = [*feature_cols, "target"]

    result = _validate_duplicated_columns(feature_cols, None, None, "target", "sample_id_col", "id")
    if should_fail:
        assert result is not None
    else:
        assert result is None


@pytest.mark.parametrize(
    "stratification_col,should_fail",
    [
        ("strat", False),
        ("sample_id_col", True),
        ("id", True),
    ],
)
def test_validate_stratification_col(stratification_col, should_fail):
    """Test stratification column validation."""
    result = _validate_stratification_col(stratification_col, "sample_id_col", "id")
    if should_fail:
        assert result is not None
    else:
        assert result is None


def test_validate_column_dtypes(sample_data):
    """Test column dtype validation."""
    assert _validate_column_dtypes(sample_data, ["feature1", "feature2", "feature3"], None, None, "target") is None

    data_with_invalid_dtype = sample_data.copy()
    data_with_invalid_dtype["feature1"] = data_with_invalid_dtype["feature1"].astype("object")
    result = _validate_column_dtypes(
        data_with_invalid_dtype, ["feature1", "feature2", "feature3"], None, None, "target"
    )
    assert "feature1" in result


def test_validate_nonempty_dataframe(sample_data):
    """Test nonempty dataframe validation."""
    assert _validate_nonempty_dataframe(sample_data) is None
    assert "empty" in _validate_nonempty_dataframe(pd.DataFrame()).lower()


def test_validate_feature_target_overlap():
    """Test feature target overlap validation."""
    assert _validate_feature_target_overlap(["feature1", "feature2"], "target", None, None) is None

    result = _validate_feature_target_overlap(["feature1", "target"], "target", None, None)
    assert "target" in result


def test_validate_reserved_column_conflicts(sample_data):
    """Test reserved column conflicts validation."""
    assert _validate_reserved_column_conflicts(sample_data, "id") is None

    data_with_reserved = sample_data.copy()
    data_with_reserved["group_features"] = 0
    result = _validate_reserved_column_conflicts(data_with_reserved, "id")
    assert "group_features" in result


def test_validate_error_accumulation(sample_data):
    """Test that validate_data() accumulates multiple errors."""
    data_with_issues = sample_data.copy()
    data_with_issues["group_features"] = 0
    data_with_issues.loc[0, "feature1"] = "invalid_string"  # This will fail dtype validation

    with pytest.raises(ValueError) as exc_info:
        validate_data(
            data=data_with_issues,
            feature_cols=["feature1", "feature2"],
            ml_type="classification",
            target_col="target",
            sample_id_col="sample_id_col",
            row_id_col="id",
            stratification_col="sample_id_col",  # This will fail as sample_id_col can't be stratification column
            positive_class=1,
        )

    error_message = str(exc_info.value)
    assert "Multiple validation errors found" in error_message
    assert "Reserved column names found in data" in error_message
    assert "Stratification column cannot be the same as sample_id_col" in error_message
