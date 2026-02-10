"""Data checker for validating datasets before running studies.

This module provides a standalone class for checking data quality and compatibility
before running a study. It combines validation from DataValidator and health
checking from HealthChecker to provide comprehensive feedback without creating
a study directory.
"""

import pandas as pd
from attrs import define, field

from octopus.study.data_validator import OctoDataValidator
from octopus.study.healthChecker import HealthCheckConfig, OctoDataHealthChecker
from octopus.study.types import MLType


def _convert_ml_type(value: MLType | str) -> MLType:
    """Convert ml_type string to MLType enum."""
    if isinstance(value, str):
        return MLType(value.lower())
    return value


@define
class DataCheckReport:
    """Machine-readable report of data checking operations.

    Attributes:
        is_valid: Whether the data passed all validation checks.
        errors: List of error messages about critical issues.
        warnings: List of warning messages about potential issues.
        info: List of informational messages.
        statistics: Dictionary of data statistics and metrics.
        health_issues: DataFrame containing detailed health check results.
    """

    is_valid: bool
    errors: list[str] = field(factory=list)
    warnings: list[str] = field(factory=list)
    info: list[str] = field(factory=list)
    statistics: dict = field(factory=dict)
    health_issues: pd.DataFrame = field(factory=pd.DataFrame)

    def to_dict(self) -> dict:
        """Convert report to dictionary format.

        Returns:
            Dictionary representation of the report.
        """
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "statistics": self.statistics,
            "health_issues": self.health_issues.to_dict(orient="records") if not self.health_issues.empty else [],
        }

    def print_summary(self) -> None:
        """Print a formatted summary of the data check results."""
        print("\n" + "=" * 80)
        print("DATA CHECK SUMMARY")
        print("=" * 80)

        print(f"\nOverall Status: {'✓ PASS' if self.is_valid else '✗ FAIL'}")

        if self.errors:
            print(f"\n{'ERRORS (' + str(len(self.errors)) + ')':-^80}")
            for i, error in enumerate(self.errors, 1):
                print(f"{i}. {error}")

        if self.warnings:
            print(f"\n{'WARNINGS (' + str(len(self.warnings)) + ')':-^80}")
            for i, warning in enumerate(self.warnings, 1):
                print(f"{i}. {warning}")

        if self.info:
            print(f"\n{'INFO (' + str(len(self.info)) + ')':-^80}")
            for i, info_msg in enumerate(self.info, 1):
                print(f"{i}. {info_msg}")

        if self.statistics:
            print(f"\n{'STATISTICS':-^80}")
            for key, value in self.statistics.items():
                print(f"{key}: {value}")

        if not self.health_issues.empty:
            print(f"\n{'HEALTH ISSUES (' + str(len(self.health_issues)) + ')':-^80}")
            print(self.health_issues.to_string(index=False))

        print("\n" + "=" * 80 + "\n")


@define
class DataChecker:
    """Standalone data checker for validating datasets before running studies.

    This class provides comprehensive data quality checks that can be run
    independently before creating a Study object. It validates data types,
    checks for missing values, identifies health issues, and provides
    a machine-readable report.

    Example:
        >>> import pandas as pd
        >>> from octopus.study.data_checker import DataChecker
        >>> from octopus.study.types import MLType
        >>>
        >>> df = pd.read_csv("data.csv")
        >>> checker = DataChecker(
        ...     data=df,
        ...     feature_cols=["feature1", "feature2", "feature3"],
        ...     target_col="target",
        ...     ml_type=MLType.CLASSIFICATION,
        ...     sample_id_col="sample_id"
        ... )
        >>> report = checker.check()
        >>> report.print_summary()
        >>>
        >>> if report.is_valid:
        ...     # Proceed with study
        ...     study = OctoClassification(...)
    """

    data: pd.DataFrame
    feature_cols: list[str]
    ml_type: MLType = field(converter=_convert_ml_type)
    sample_id_col: str
    target_col: str | None = None
    duration_col: str | None = None
    event_col: str | None = None
    row_id_col: str | None = None
    stratification_col: str | None = None
    positive_class: int | None = None
    health_check_config: HealthCheckConfig | None = field(default=None)

    errors: list[str] = field(init=False, factory=list)
    warnings: list[str] = field(init=False, factory=list)
    info: list[str] = field(init=False, factory=list)
    statistics: dict = field(init=False, factory=dict)
    health_issues: pd.DataFrame = field(init=False, factory=pd.DataFrame)

    def __attrs_post_init__(self) -> None:
        """Post-initialization processing."""
        # Make a copy of the dataframe
        object.__setattr__(self, "data", self.data.copy())

        # Set default health check config if not provided
        if self.health_check_config is None:
            object.__setattr__(self, "health_check_config", HealthCheckConfig())

    def check(self) -> DataCheckReport:
        """Run all data checks and return comprehensive report.

        Returns:
            DataCheckReport: Object containing validation results, statistics, and health issues.
        """
        self.errors = []
        self.warnings = []
        self.info = []
        self.statistics = {}
        self.health_issues = pd.DataFrame()

        # Basic structure checks
        self._check_basic_structure()

        # Only proceed with further checks if basic structure is valid
        if not self.errors:
            # Data validation checks
            self._run_data_validator()

            # Health checks
            self._run_health_checker()

            # Statistical summary
            self._gather_statistics()

        is_valid = len(self.errors) == 0

        return DataCheckReport(
            is_valid=is_valid,
            errors=self.errors,
            warnings=self.warnings,
            info=self.info,
            statistics=self.statistics,
            health_issues=self.health_issues,
        )

    def _check_basic_structure(self) -> None:
        """Check basic dataframe structure and ML type requirements."""
        # Check if dataframe is empty
        if self.data.empty:
            self.errors.append("Dataframe is empty")
            return

        # Check if feature columns exist
        if not self.feature_cols:
            self.errors.append("No feature columns specified")
            return

        # Check ML type specific requirements
        if self.ml_type == MLType.TIMETOEVENT:
            if not self.duration_col:
                self.errors.append("duration_col is required for TIME_TO_EVENT tasks")
            if not self.event_col:
                self.errors.append("event_col is required for TIME_TO_EVENT tasks")
        elif not self.target_col:
            self.errors.append(f"target_col is required for {self.ml_type.value} tasks")

        # Check for duplicate column names
        if self.data.columns.duplicated().any():
            duplicates = self.data.columns[self.data.columns.duplicated()].tolist()
            self.errors.append(f"Duplicate column names found: {duplicates}")

        # Check minimum sample size
        min_samples = 10
        if len(self.data) < min_samples:
            self.errors.append(f"Dataframe has only {len(self.data)} rows. Minimum {min_samples} required.")

    def _run_data_validator(self) -> None:
        """Run OctoDataValidator checks."""
        try:
            validator = OctoDataValidator(
                data=self.data,
                feature_cols=self.feature_cols,
                target_col=self.target_col,
                duration_col=self.duration_col,
                event_col=self.event_col,
                sample_id_col=self.sample_id_col,
                row_id_col=self.row_id_col,
                stratification_col=self.stratification_col,
                ml_type=self.ml_type.value,
                positive_class=self.positive_class,
            )

            # Run validation (this will raise exceptions if critical issues found)
            validator.validate()

        except (ValueError, KeyError) as e:
            error_msg = str(e)
            # If it's a multi-error message, split it
            if "Multiple validation errors found:" in error_msg:
                # Extract individual errors
                lines = error_msg.split("\n")
                for line in lines[1:]:  # Skip the first line
                    if line.startswith("- "):
                        self.errors.append(line[2:])  # Remove "- " prefix
            else:
                self.errors.append(f"Data validation error: {error_msg}")

    def _run_health_checker(self) -> None:
        """Run OctoDataHealthChecker checks."""
        try:
            # health_check_config is guaranteed to be HealthCheckConfig after __attrs_post_init__
            assert self.health_check_config is not None
            health_checker = OctoDataHealthChecker(
                data=self.data,
                feature_cols=self.feature_cols,
                target_col=self.target_col,
                duration_col=self.duration_col,
                event_col=self.event_col,
                row_id_col=self.row_id_col,
                sample_id_col=self.sample_id_col,
                stratification_col=self.stratification_col,
                config=self.health_check_config,
            )

            # Generate health report
            self.health_issues = health_checker.generate_report()

            # Categorize health issues
            if not self.health_issues.empty and "Severity" in self.health_issues.columns:
                critical_issues = self.health_issues[self.health_issues["Severity"] == "Critical"]
                warning_issues = self.health_issues[self.health_issues["Severity"] == "Warning"]
                info_issues = self.health_issues[self.health_issues["Severity"] == "Info"]

                # Add critical health issues to warnings (not errors)
                # Health issues indicate data quality concerns but don't prevent study creation
                for _, issue in critical_issues.iterrows():
                    self.warnings.append(f"[CRITICAL] {issue['Issue Type']}: {issue['Description']}")

                # Add warnings
                for _, issue in warning_issues.iterrows():
                    self.warnings.append(f"{issue['Issue Type']}: {issue['Description']}")

                # Add info
                for _, issue in info_issues.iterrows():
                    self.info.append(f"{issue['Issue Type']}: {issue['Description']}")

        except Exception as e:
            self.warnings.append(f"Health check error: {e!s}")

    def _gather_statistics(self) -> None:
        """Gather statistical summary of the data."""
        if self.data.empty:
            return

        self.statistics["n_samples"] = len(self.data)
        self.statistics["n_features"] = len(self.feature_cols)

        # Count numeric and categorical features
        numeric_cols = self.data[self.feature_cols].select_dtypes(include=["number"]).columns
        self.statistics["n_numeric_features"] = len(numeric_cols)
        self.statistics["n_categorical_features"] = len(self.feature_cols) - len(numeric_cols)

        # ML type specific statistics
        if self.ml_type in [MLType.CLASSIFICATION, MLType.MULTICLASS]:
            if self.target_col and self.target_col in self.data.columns:
                value_counts = self.data[self.target_col].value_counts()
                self.statistics["n_classes"] = len(value_counts)
                self.statistics["class_distribution"] = value_counts.to_dict()

                if len(value_counts) == 2:
                    imbalance_ratio = value_counts.max() / value_counts.min()
                    self.statistics["imbalance_ratio"] = f"{imbalance_ratio:.2f}:1"

        elif self.ml_type == MLType.REGRESSION:
            if self.target_col and self.target_col in self.data.columns:
                self.statistics["target_mean"] = float(self.data[self.target_col].mean())
                self.statistics["target_std"] = float(self.data[self.target_col].std())
                self.statistics["target_min"] = float(self.data[self.target_col].min())
                self.statistics["target_max"] = float(self.data[self.target_col].max())

        elif self.ml_type == MLType.TIMETOEVENT:
            if self.event_col and self.event_col in self.data.columns:
                event_counts = self.data[self.event_col].value_counts()
                self.statistics["event_distribution"] = event_counts.to_dict()
                if len(event_counts) == 2:
                    event_rate = event_counts.get(1, 0) / len(self.data)
                    self.statistics["event_rate"] = f"{event_rate:.2%}"

        # Memory usage
        memory_mb = self.data.memory_usage(deep=True).sum() / 1024**2
        self.statistics["memory_usage_mb"] = f"{memory_mb:.2f}"


def check_data(
    data: pd.DataFrame,
    feature_cols: list[str],
    ml_type: MLType | str,
    sample_id_col: str,
    target_col: str | None = None,
    duration_col: str | None = None,
    event_col: str | None = None,
    row_id_col: str | None = None,
    stratification_col: str | None = None,
    positive_class: int | None = None,
    health_check_config: HealthCheckConfig | None = None,
    print_summary: bool = True,
) -> DataCheckReport:
    """Quick function to check data quality.

    Args:
        data: Input dataframe to check.
        feature_cols: List of feature column names.
        ml_type: Type of ML task (CLASSIFICATION, MULTICLASS, REGRESSION, TIMETOEVENT).
        sample_id_col: Identifier for sample instances.
        target_col: Target column for classification/regression. None for time-to-event.
        duration_col: Duration column for time-to-event. None for other tasks.
        event_col: Event column for time-to-event. None for other tasks.
        row_id_col: Unique row identifier. Optional.
        stratification_col: Column used for stratification. Optional.
        positive_class: Positive class label for binary classification. Optional.
        health_check_config: Configuration for health check thresholds. Optional.
        print_summary: Whether to print the summary to console.

    Returns:
        DataCheckReport: Object containing validation results and recommendations.

    Example:
        >>> from octopus.study.data_checker import check_data
        >>> from octopus.study.types import MLType
        >>>
        >>> report = check_data(
        ...     data=df,
        ...     feature_cols=["f1", "f2"],
        ...     ml_type=MLType.CLASSIFICATION,
        ...     sample_id_col="id",
        ...     target_col="target"
        ... )
        >>> if report.is_valid:
        ...     print("Data is ready for study!")
    """
    checker = DataChecker(
        data=data,
        feature_cols=feature_cols,
        ml_type=ml_type,
        sample_id_col=sample_id_col,
        target_col=target_col,
        duration_col=duration_col,
        event_col=event_col,
        row_id_col=row_id_col,
        stratification_col=stratification_col,
        positive_class=positive_class,
        health_check_config=health_check_config,
    )
    report = checker.check()

    if print_summary:
        report.print_summary()

    return report
