"""Test OctoData validator."""

import numpy as np
import pandas as pd
import pytest

from octopus.study.data_validator import OctoDataValidator
from octopus.types import MLType


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
            "strat": [0, 1] * 50,
        }
    ).astype({"feature3": "category"})


@pytest.fixture
def validator_factory(sample_data):
    """Create validators with custom parameters."""

    def _create_validator(
        data=None,
        feature_cols=None,
        target_col=None,
        sample_id_col="sample_id_col",
        row_id_col="id",
        stratification_col="strat",
        ml_type=MLType.BINARY,
        positive_class=1,
    ):
        if data is None:
            data = sample_data
        if feature_cols is None:
            feature_cols = ["feature1", "feature2", "feature3"]
        if target_col is None:
            target_col = "target"

        return OctoDataValidator(
            data=data,
            feature_cols=feature_cols,
            target_col=target_col,
            sample_id_col=sample_id_col,
            row_id_col=row_id_col,
            stratification_col=stratification_col,
            ml_type=ml_type,
            positive_class=positive_class,
        )

    return _create_validator


@pytest.fixture
def valid_validator(validator_factory):
    """Create valid validator."""
    return validator_factory()


def test_initialization(valid_validator):
    """Test initialization."""
    assert isinstance(valid_validator, OctoDataValidator)


def test_validate_columns_exist_valid(validator_factory):
    """Test column exists validation with valid columns."""
    validator = validator_factory()
    validator._validate_columns_exist()  # Should not raise


def test_validate_columns_exist_missing(validator_factory):
    """Test column exists validation with missing column."""
    validator = validator_factory(feature_cols=["feature1", "non_existent_column"])
    with pytest.raises(ValueError):
        validator._validate_columns_exist()


@pytest.mark.parametrize(
    "duplicate_setup,should_fail",
    [
        (None, False),
        ("feature_to_target", True),
    ],
)
def test_validate_duplicated_columns(validator_factory, duplicate_setup, should_fail):
    """Test duplicated columns validation."""
    validator = validator_factory()

    if duplicate_setup == "feature_to_target":
        validator.feature_cols.append("target")

    if should_fail:
        with pytest.raises(ValueError):
            validator._validate_duplicated_columns()
    else:
        validator._validate_duplicated_columns()


@pytest.mark.parametrize(
    "stratification_col,should_fail",
    [
        ("strat", False),
        ("sample_id_col", True),
        ("id", True),
    ],
)
def test_validate_stratification_col(validator_factory, stratification_col, should_fail):
    """Test stratification column validation."""
    validator = validator_factory(stratification_col=stratification_col)

    if should_fail:
        with pytest.raises(ValueError):
            validator._validate_stratification_col()
    else:
        validator._validate_stratification_col()


@pytest.mark.parametrize(
    "strat_dtype,should_fail",
    [
        ("int64", False),
        ("bool", False),
        ("float64", True),
        ("category", True),
        ("object", True),
    ],
)
def test_validate_stratification_col_dtype(validator_factory, sample_data, strat_dtype, should_fail):
    """Test stratification column dtype validation."""
    data = sample_data.copy()
    if strat_dtype == "bool":
        data["strat"] = [True, False] * 50
    elif strat_dtype == "int64":
        data["strat"] = list(range(100))
    elif strat_dtype == "float64":
        data["strat"] = np.random.rand(100)
    data["strat"] = data["strat"].astype(strat_dtype)

    validator = validator_factory(data=data, stratification_col="strat")

    if should_fail:
        with pytest.raises(ValueError, match="unsupported dtype"):
            validator._validate_stratification_col()
    else:
        validator._validate_stratification_col()


def test_validate_column_dtypes(validator_factory, sample_data):
    """Test column dtype validation."""
    validator_factory()._validate_column_dtypes()

    data_with_invalid_dtype = sample_data.copy()
    data_with_invalid_dtype["feature1"] = data_with_invalid_dtype["feature1"].astype("object")
    with pytest.raises(ValueError):
        validator_factory(data=data_with_invalid_dtype)._validate_column_dtypes()


def test_validate_nonempty_dataframe(validator_factory):
    """Test nonempty dataframe validation."""
    validator_factory(feature_cols=["feature1"])._validate_nonempty_dataframe()

    empty_validator = validator_factory(data=pd.DataFrame(), feature_cols=["feature1"])
    with pytest.raises(ValueError, match="DataFrame is empty"):
        empty_validator._validate_nonempty_dataframe()


def test_validate_feature_target_overlap(validator_factory):
    """Test feature target overlap validation."""
    validator_factory(feature_cols=["feature1", "feature2"])._validate_feature_target_overlap()

    with pytest.raises(ValueError, match="Columns cannot be both features and targets"):
        validator_factory(feature_cols=["feature1", "target"])._validate_feature_target_overlap()


def test_validate_reserved_column_conflicts(validator_factory, sample_data):
    """Test reserved column conflicts validation."""
    validator_factory(feature_cols=["feature1"])._validate_reserved_column_conflicts()

    data_with_reserved = sample_data.copy()
    data_with_reserved["datasplit_group"] = 0
    with pytest.raises(ValueError, match="Reserved column names found in data"):
        validator_factory(data=data_with_reserved, feature_cols=["feature1"])._validate_reserved_column_conflicts()


def test_validate_error_accumulation(validator_factory, sample_data):
    """Test that validate() accumulates multiple errors."""
    data_with_issues = sample_data.copy()
    data_with_issues["datasplit_group"] = 0
    data_with_issues["feature1"] = data_with_issues["feature1"].astype(object)
    data_with_issues.loc[0, "feature1"] = "invalid_string"  # This will fail dtype validation

    validator = validator_factory(
        data=data_with_issues,
        feature_cols=["feature1", "feature2"],
        stratification_col="sample_id_col",  # This will fail as sample_id_col can't be stratification column
    )

    with pytest.raises(ValueError) as exc_info:
        validator.validate()

    error_message = str(exc_info.value)
    assert "Multiple validation errors found" in error_message
    assert "Reserved column names found in data" in error_message
    assert "Stratification column cannot be the same as sample_id_col" in error_message
