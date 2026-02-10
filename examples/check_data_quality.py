"""Example: Check data quality before running a study.

This example demonstrates how to use the DataChecker class to validate
your data before creating a Study. This can save time by identifying
potential issues early without creating a study directory.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

from octopus.study import DataChecker, check_data
from octopus.study.types import MLType


def example_classification():
    """Example with classification data."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Binary Classification Data Check")
    print("=" * 80)

    # Load a well-known dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    df["sample_id"] = [f"S{i}" for i in range(len(df))]

    # Check the data before creating a study
    report = check_data(
        data=df,
        feature_cols=list(data.feature_names),
        ml_type=MLType.CLASSIFICATION,
        sample_id_col="sample_id",
        target_col="target",
        positive_class=1,
        print_summary=True,
    )

    if report.is_valid:
        print("\n✓ Data is valid! You can proceed with study creation.")
    else:
        print("\n✗ Data has errors. Please fix them before creating a study.")

    # Access machine-readable report
    print("\nMachine-readable report available:")
    print(f"  - Errors: {len(report.errors)}")
    print(f"  - Warnings: {len(report.warnings)}")
    print(f"  - Statistics: {len(report.statistics)} metrics")


def example_multiclass():
    """Example with multiclass classification data."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Multiclass Classification Data Check")
    print("=" * 80)

    # Create multiclass dataset
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(200)],
            "feature1": np.random.randn(200),
            "feature2": np.random.randn(200),
            "feature3": np.random.randn(200),
            "feature4": np.random.randn(200),
            "target": np.random.choice([0, 1, 2, 3], 200),
        }
    )

    # Check the data
    checker = DataChecker(
        data=df,
        feature_cols=["feature1", "feature2", "feature3", "feature4"],
        ml_type=MLType.MULTICLASS,
        sample_id_col="sample_id",
        target_col="target",
    )

    report = checker.check()
    report.print_summary()

    # Export to dict for further processing
    report_dict = report.to_dict()
    print(f"\nClass distribution: {report_dict['statistics']['class_distribution']}")


def example_regression():
    """Example with regression data."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Regression Data Check")
    print("=" * 80)

    # Load regression dataset
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    df["sample_id"] = [f"S{i}" for i in range(len(df))]

    # Check the data
    report = check_data(
        data=df,
        feature_cols=list(data.feature_names),
        ml_type=MLType.REGRESSION,
        sample_id_col="sample_id",
        target_col="target",
        print_summary=True,
    )

    print("\nTarget statistics:")
    print(f"  - Mean: {report.statistics['target_mean']:.2f}")
    print(f"  - Std: {report.statistics['target_std']:.2f}")
    print(f"  - Range: [{report.statistics['target_min']:.2f}, {report.statistics['target_max']:.2f}]")


def example_timetoevent():
    """Example with time-to-event data."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Time-to-Event Data Check")
    print("=" * 80)

    # Create time-to-event dataset
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(150)],
            "feature1": np.random.randn(150),
            "feature2": np.random.randn(150),
            "feature3": np.random.randn(150),
            "duration": np.random.uniform(1, 100, 150),
            "event": np.random.choice([0, 1], 150),
        }
    )

    # Check the data
    checker = DataChecker(
        data=df,
        feature_cols=["feature1", "feature2", "feature3"],
        ml_type=MLType.TIMETOEVENT,
        sample_id_col="sample_id",
        duration_col="duration",
        event_col="event",
    )

    report = checker.check()
    report.print_summary()

    if "event_distribution" in report.statistics:
        print(f"\nEvent distribution: {report.statistics['event_distribution']}")
    if "event_rate" in report.statistics:
        print(f"Event rate: {report.statistics['event_rate']}")


def example_with_issues():
    """Example with data that has various issues."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Data with Quality Issues")
    print("=" * 80)

    # Create problematic dataset
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(100)],
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
            "target": [0] * 90 + [1] * 10,  # Highly imbalanced
        }
    )

    # Introduce issues
    # 1. Missing values
    df.loc[0:40, "feature1"] = np.nan

    # 2. Add a constant feature
    df["feature4"] = 42

    # 3. Add infinite values
    df.loc[0:5, "feature2"] = np.inf

    # Check the data
    checker = DataChecker(
        data=df,
        feature_cols=["feature1", "feature2", "feature3", "feature4"],
        ml_type=MLType.CLASSIFICATION,
        sample_id_col="sample_id",
        target_col="target",
        positive_class=1,
    )

    report = checker.check()
    report.print_summary()

    print("\nAnalysis of issues:")
    print(f"- Critical errors: {len(report.errors)}")
    print(f"- Warnings: {len(report.warnings)}")
    print(f"- Informational: {len(report.info)}")

    if not report.health_issues.empty:
        print("\nHealth issues by severity:")
        severity_counts = report.health_issues["Severity"].value_counts()
        for severity, count in severity_counts.items():
            print(f"  - {severity}: {count}")


def example_programmatic_usage():
    """Example showing programmatic usage for automation."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Programmatic Usage for Automation")
    print("=" * 80)

    # Create dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    df["sample_id"] = [f"S{i}" for i in range(len(df))]

    # Check data without printing
    report = check_data(
        data=df,
        feature_cols=list(data.feature_names),
        ml_type=MLType.CLASSIFICATION,
        sample_id_col="sample_id",
        target_col="target",
        positive_class=1,
        print_summary=False,
    )

    # Use report in automation pipeline
    if report.is_valid:
        print("✓ Data validation passed")
        print(f"  Dataset: {report.statistics['n_samples']} samples, {report.statistics['n_features']} features")
        print(f"  Classes: {report.statistics['n_classes']} ({report.statistics['imbalance_ratio']})")

        # Export report for logging
        report_dict = report.to_dict()
        print(f"\n  Exported {len(report_dict)} report fields")

        # Could now create study automatically
        # study = OctoClassification(...)
    else:
        print("✗ Data validation failed")
        print(f"  Errors: {len(report.errors)}")
        for i, error in enumerate(report.errors, 1):
            print(f"    {i}. {error}")

        # Stop automation or alert user
        raise ValueError("Data validation failed - cannot proceed with study")


if __name__ == "__main__":
    # Run all examples
    try:
        example_classification()
        example_multiclass()
        example_regression()
        example_timetoevent()
        example_with_issues()
        example_programmatic_usage()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
    except Exception as e:
        print(f"\nExample failed with error: {e}")
