"""Tests for the DataChecker class."""

import numpy as np
import pandas as pd
import pytest

from octopus.study.data_checker import DataChecker, DataCheckReport, check_data
from octopus.study.types import MLType


@pytest.fixture
def valid_classification_data():
    """Create a valid classification dataset."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(100)],
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
            "target": np.random.choice([0, 1], 100),
        }
    )


@pytest.fixture
def valid_multiclass_data():
    """Create a valid multiclass dataset."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(100)],
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
            "target": np.random.choice([0, 1, 2], 100),
        }
    )


@pytest.fixture
def valid_regression_data():
    """Create a valid regression dataset."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(100)],
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
            "target": np.random.randn(100),
        }
    )


@pytest.fixture
def valid_timetoevent_data():
    """Create a valid time-to-event dataset."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(100)],
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
            "duration": np.random.uniform(1, 100, 100),
            "event": np.random.choice([0, 1], 100),
        }
    )


def test_data_checker_initialization(valid_classification_data):
    """Test DataChecker can be initialized."""
    checker = DataChecker(
        data=valid_classification_data,
        feature_cols=["feature1", "feature2", "feature3"],
        ml_type=MLType.CLASSIFICATION,
        sample_id_col="sample_id",
        target_col="target",
        positive_class=1,
    )
    assert checker is not None
    assert checker.ml_type == MLType.CLASSIFICATION


def test_valid_classification_data(valid_classification_data):
    """Test that valid classification data passes all checks."""
    checker = DataChecker(
        data=valid_classification_data,
        feature_cols=["feature1", "feature2", "feature3"],
        ml_type=MLType.CLASSIFICATION,
        sample_id_col="sample_id",
        target_col="target",
        positive_class=1,
    )
    report = checker.check()

    assert isinstance(report, DataCheckReport)
    assert report.is_valid
    assert len(report.errors) == 0


def test_valid_multiclass_data(valid_multiclass_data):
    """Test that valid multiclass data passes all checks."""
    checker = DataChecker(
        data=valid_multiclass_data,
        feature_cols=["feature1", "feature2", "feature3"],
        ml_type=MLType.MULTICLASS,
        sample_id_col="sample_id",
        target_col="target",
    )
    report = checker.check()

    assert isinstance(report, DataCheckReport)
    assert report.is_valid
    assert len(report.errors) == 0
    assert report.statistics["n_classes"] == 3


def test_valid_regression_data(valid_regression_data):
    """Test that valid regression data passes all checks."""
    checker = DataChecker(
        data=valid_regression_data,
        feature_cols=["feature1", "feature2", "feature3"],
        ml_type=MLType.REGRESSION,
        sample_id_col="sample_id",
        target_col="target",
    )
    report = checker.check()

    assert isinstance(report, DataCheckReport)
    assert report.is_valid
    assert len(report.errors) == 0
    assert "target_mean" in report.statistics


def test_valid_timetoevent_data(valid_timetoevent_data):
    """Test that valid time-to-event data passes all checks."""
    checker = DataChecker(
        data=valid_timetoevent_data,
        feature_cols=["feature1", "feature2", "feature3"],
        ml_type=MLType.TIMETOEVENT,
        sample_id_col="sample_id",
        duration_col="duration",
        event_col="event",
    )
    report = checker.check()

    assert isinstance(report, DataCheckReport)
    assert report.is_valid
    assert len(report.errors) == 0
    assert "event_distribution" in report.statistics


def test_empty_dataframe():
    """Test that empty dataframe is detected."""
    df = pd.DataFrame()
    checker = DataChecker(
        data=df,
        feature_cols=["feature1"],
        ml_type=MLType.CLASSIFICATION,
        sample_id_col="sample_id",
        target_col="target",
    )
    report = checker.check()

    assert not report.is_valid
    assert any("empty" in error.lower() for error in report.errors)


def test_missing_target_column(valid_classification_data):
    """Test that missing target column is detected."""
    checker = DataChecker(
        data=valid_classification_data,
        feature_cols=["feature1", "feature2", "feature3"],
        ml_type=MLType.CLASSIFICATION,
        sample_id_col="sample_id",
        target_col="nonexistent_target",
        positive_class=1,
    )
    report = checker.check()

    assert not report.is_valid
    assert any("nonexistent_target" in error for error in report.errors)


def test_missing_duration_col_for_timetoevent(valid_timetoevent_data):
    """Test that missing duration_col for time-to-event is detected."""
    checker = DataChecker(
        data=valid_timetoevent_data,
        feature_cols=["feature1", "feature2", "feature3"],
        ml_type=MLType.TIMETOEVENT,
        sample_id_col="sample_id",
        event_col="event",
        # Missing duration_col
    )
    report = checker.check()

    assert not report.is_valid
    assert any("duration_col is required" in error for error in report.errors)


def test_missing_event_col_for_timetoevent(valid_timetoevent_data):
    """Test that missing event_col for time-to-event is detected."""
    checker = DataChecker(
        data=valid_timetoevent_data,
        feature_cols=["feature1", "feature2", "feature3"],
        ml_type=MLType.TIMETOEVENT,
        sample_id_col="sample_id",
        duration_col="duration",
        # Missing event_col
    )
    report = checker.check()

    assert not report.is_valid
    assert any("event_col is required" in error for error in report.errors)


def test_no_feature_columns():
    """Test that missing feature columns is detected."""
    df = pd.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(50)],
            "target": [0, 1] * 25,
        }
    )
    checker = DataChecker(
        data=df,
        feature_cols=[],
        ml_type=MLType.CLASSIFICATION,
        sample_id_col="sample_id",
        target_col="target",
    )
    report = checker.check()

    assert not report.is_valid
    assert any("feature" in error.lower() for error in report.errors)


def test_too_few_samples():
    """Test that insufficient sample size is detected."""
    df = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3"],
            "feature1": [1, 2, 3],
            "target": [0, 1, 0],
        }
    )
    checker = DataChecker(
        data=df,
        feature_cols=["feature1"],
        ml_type=MLType.CLASSIFICATION,
        sample_id_col="sample_id",
        target_col="target",
    )
    report = checker.check()

    assert not report.is_valid
    assert any("minimum" in error.lower() or "rows" in error.lower() for error in report.errors)


def test_statistics_classification(valid_classification_data):
    """Test that classification statistics are collected."""
    checker = DataChecker(
        data=valid_classification_data,
        feature_cols=["feature1", "feature2", "feature3"],
        ml_type=MLType.CLASSIFICATION,
        sample_id_col="sample_id",
        target_col="target",
        positive_class=1,
    )
    report = checker.check()

    assert "n_samples" in report.statistics
    assert "n_features" in report.statistics
    assert "n_classes" in report.statistics
    assert "class_distribution" in report.statistics
    assert report.statistics["n_samples"] == 100
    assert report.statistics["n_features"] == 3
    assert report.statistics["n_classes"] == 2


def test_statistics_regression(valid_regression_data):
    """Test that regression statistics are collected."""
    checker = DataChecker(
        data=valid_regression_data,
        feature_cols=["feature1", "feature2", "feature3"],
        ml_type=MLType.REGRESSION,
        sample_id_col="sample_id",
        target_col="target",
    )
    report = checker.check()

    assert "n_samples" in report.statistics
    assert "n_features" in report.statistics
    assert "target_mean" in report.statistics
    assert "target_std" in report.statistics
    assert "target_min" in report.statistics
    assert "target_max" in report.statistics


def test_check_data_convenience_function(valid_classification_data):
    """Test the convenience function check_data."""
    report = check_data(
        data=valid_classification_data,
        feature_cols=["feature1", "feature2", "feature3"],
        ml_type=MLType.CLASSIFICATION,
        sample_id_col="sample_id",
        target_col="target",
        positive_class=1,
        print_summary=False,
    )

    assert isinstance(report, DataCheckReport)
    assert report.is_valid


def test_data_check_report_to_dict(valid_classification_data):
    """Test that DataCheckReport can be converted to dict."""
    checker = DataChecker(
        data=valid_classification_data,
        feature_cols=["feature1", "feature2", "feature3"],
        ml_type=MLType.CLASSIFICATION,
        sample_id_col="sample_id",
        target_col="target",
        positive_class=1,
    )
    report = checker.check()
    report_dict = report.to_dict()

    assert isinstance(report_dict, dict)
    assert "is_valid" in report_dict
    assert "errors" in report_dict
    assert "warnings" in report_dict
    assert "statistics" in report_dict


def test_data_check_report_print_summary(valid_classification_data, capsys):
    """Test that DataCheckReport can print a summary."""
    checker = DataChecker(
        data=valid_classification_data,
        feature_cols=["feature1", "feature2", "feature3"],
        ml_type=MLType.CLASSIFICATION,
        sample_id_col="sample_id",
        target_col="target",
        positive_class=1,
    )
    report = checker.check()
    report.print_summary()

    captured = capsys.readouterr()
    assert "DATA CHECK SUMMARY" in captured.out
    assert "Overall Status" in captured.out


def test_checker_does_not_modify_original_df(valid_classification_data):
    """Test that DataChecker doesn't modify the original dataframe."""
    original_df = valid_classification_data.copy()

    checker = DataChecker(
        data=valid_classification_data,
        feature_cols=["feature1", "feature2", "feature3"],
        ml_type=MLType.CLASSIFICATION,
        sample_id_col="sample_id",
        target_col="target",
        positive_class=1,
    )
    checker.check()

    # Original should be unchanged
    pd.testing.assert_frame_equal(valid_classification_data, original_df)


def test_ml_type_string_conversion(valid_classification_data):
    """Test that ml_type can be passed as string."""
    checker = DataChecker(
        data=valid_classification_data,
        feature_cols=["feature1", "feature2", "feature3"],
        ml_type="classification",  # String instead of MLType
        sample_id_col="sample_id",
        target_col="target",
        positive_class=1,
    )
    report = checker.check()

    assert report.is_valid
    assert checker.ml_type == MLType.CLASSIFICATION


def test_health_issues_in_report(valid_classification_data):
    """Test that health issues are included in report."""
    # Create data with health issues
    df = valid_classification_data.copy()
    df.loc[0:30, "feature1"] = np.nan  # High missing values

    checker = DataChecker(
        data=df,
        feature_cols=["feature1", "feature2", "feature3"],
        ml_type=MLType.CLASSIFICATION,
        sample_id_col="sample_id",
        target_col="target",
        positive_class=1,
    )
    report = checker.check()

    # Should still be valid (warnings don't block)
    assert report.is_valid
    assert not report.health_issues.empty
    assert len(report.warnings) > 0


def test_multiple_validation_errors(valid_classification_data):
    """Test that multiple validation errors are captured."""
    df = valid_classification_data.copy()
    # Add reserved column
    df["group_features"] = 0

    checker = DataChecker(
        data=df,
        feature_cols=["feature1", "feature2", "feature3", "group_features"],
        ml_type=MLType.CLASSIFICATION,
        sample_id_col="sample_id",
        target_col="target",
        positive_class=1,
    )
    report = checker.check()

    assert not report.is_valid
    assert len(report.errors) > 0
    assert any("reserved" in error.lower() for error in report.errors)


def test_numeric_and_categorical_feature_counts(valid_classification_data):
    """Test that numeric and categorical features are counted correctly."""
    df = valid_classification_data.copy()
    df["cat_feature"] = np.random.choice(["A", "B", "C"], 100)

    checker = DataChecker(
        data=df,
        feature_cols=["feature1", "feature2", "feature3", "cat_feature"],
        ml_type=MLType.CLASSIFICATION,
        sample_id_col="sample_id",
        target_col="target",
        positive_class=1,
    )
    report = checker.check()

    assert report.statistics["n_numeric_features"] == 3
    assert report.statistics["n_categorical_features"] == 1
