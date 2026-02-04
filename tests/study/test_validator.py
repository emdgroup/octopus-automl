"""Test OctoData validator."""

import numpy as np
import pandas as pd
import pytest

from octopus.study.data_validator import OctoDataValidator


@pytest.fixture
def sample_data():
    """Create sample data."""
    return pd.DataFrame(
        {
            "id": range(1, 101),
            "sample_id": [f"S{i}" for i in range(1, 101)],
            "feature1": np.random.rand(100),
            "feature2": np.random.randint(0, 5, 100),
            "feature3": ["A", "B", "C"] * 33 + ["A"],
            "target": np.random.randint(0, 2, 100),
            "time": np.random.rand(100) * 10,
            "strat": ["X", "Y"] * 50,
        }
    ).astype({"strat": "category", "feature3": "category"})


@pytest.fixture
def validator_factory(sample_data):
    """Create validators with custom parameters."""

    def _create_validator(
        data=None,
        feature_cols=None,
        target_cols=None,
        sample_id="sample_id",
        row_id="id",
        stratification_column="strat",
        target_assignments=None,
        ml_type="classification",
        positive_class=1,
    ):
        if data is None:
            data = sample_data
        if feature_cols is None:
            feature_cols = ["feature1", "feature2", "feature3"]
        if target_cols is None:
            target_cols = ["target"]
        if target_assignments is None:
            target_assignments = {}

        return OctoDataValidator(
            data=data,
            feature_cols=feature_cols,
            target_cols=target_cols,
            sample_id=sample_id,
            row_id=row_id,
            stratification_column=stratification_column,
            target_assignments=target_assignments,
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


def test_validate(valid_validator):
    """Test validate function."""
    valid_validator.validate()


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
    "stratification_column,should_fail",
    [
        ("strat", False),
        ("sample_id", True),
        ("id", True),
    ],
)
def test_validate_stratification_column(validator_factory, stratification_column, should_fail):
    """Test stratification column validation."""
    validator = validator_factory(stratification_column=stratification_column)

    if should_fail:
        with pytest.raises(ValueError):
            validator._validate_stratification_column()
    else:
        validator._validate_stratification_column()


@pytest.mark.parametrize(
    "target_cols,target_assignments,should_fail",
    [
        (["target"], {}, False),
        (["target"], {"event": "target"}, True),
        (["target", "time"], {}, True),
        (["target", "time"], {"event": "target"}, True),
        (["target", "time"], {"event": "target", "duration": "non_existent"}, True),
        (["target", "time"], {"event": "target", "duration": "target"}, True),
        (["target", "time"], {"event": "target", "duration": "time"}, False),
    ],
)
def test_validate_target_assignments(validator_factory, target_cols, target_assignments, should_fail):
    """Test target assignment validation."""
    validator = validator_factory(target_cols=target_cols, target_assignments=target_assignments)

    if should_fail:
        with pytest.raises(ValueError):
            validator._validate_target_assignments()
    else:
        validator._validate_target_assignments()


def test_validate_number_of_targets(valid_validator):
    """Test number of targets validation."""
    valid_validator._validate_number_of_targets()

    # Test with too many targets
    invalid_validator = valid_validator
    invalid_validator.target_cols = ["target1", "target2", "target3"]
    with pytest.raises(ValueError):
        invalid_validator._validate_number_of_targets()


def test_validate_column_dtypes(validator_factory, sample_data):
    """Test column dtype validation."""
    validator_factory()._validate_column_dtypes()

    data_with_invalid_dtype = sample_data.copy()
    data_with_invalid_dtype["feature1"] = data_with_invalid_dtype["feature1"].astype("object")
    with pytest.raises(ValueError):
        validator_factory(data=data_with_invalid_dtype)._validate_column_dtypes()


def test_validate_with_two_targets(validator_factory):
    """Test two targets."""
    validator = validator_factory(
        target_cols=["target", "time"],
        target_assignments={"event": "target", "time": "time"},
    )
    validator.validate()

    invalid_validator = validator_factory(target_cols=["target", "time"], target_assignments={})
    with pytest.raises(ValueError):
        invalid_validator._validate_number_of_targets()


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
    data_with_reserved["group_features"] = 0
    with pytest.raises(ValueError, match="Reserved column names found in data"):
        validator_factory(data=data_with_reserved, feature_cols=["feature1"])._validate_reserved_column_conflicts()


def test_validate_error_accumulation(validator_factory, sample_data):
    """Test that validate() accumulates multiple errors."""
    data_with_issues = sample_data.copy()
    data_with_issues["group_features"] = 0
    data_with_issues.loc[0, "feature1"] = "invalid_string"  # This will fail dtype validation

    validator = validator_factory(
        data=data_with_issues,
        feature_cols=["feature1", "feature2"],
        stratification_column="sample_id",  # This will fail as sample_id can't be stratification column
    )

    with pytest.raises(ValueError) as exc_info:
        validator.validate()

    error_message = str(exc_info.value)
    assert "Multiple validation errors found" in error_message
    assert "Reserved column names found in data" in error_message
    assert "Stratification column cannot be the same as sample_id" in error_message
