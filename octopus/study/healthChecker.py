"""OctoData Health Checker."""

import logging
import re
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd
from attrs import define, field, validators
from rapidfuzz import fuzz


@define
class HealthCheckConfig:
    """Configuration for health check thresholds.

    This class defines configurable thresholds for various data quality checks
    performed by the OctoDataHealthChecker. All thresholds have sensible defaults
    but can be customized based on specific data quality requirements.

    Attributes:
        missing_value_column_threshold: Maximum acceptable proportion of missing values
            in feature columns (range: 0.0-1.0). Columns exceeding this threshold are
            flagged as critical. Default: 0.25.
        missing_value_row_threshold: Maximum acceptable proportion of missing values
            in rows (range: 0.0-1.0). Rows exceeding this threshold are flagged as
            critical. Default: 0.5.
        int_few_uniques_threshold: Maximum number of unique values for integer columns
            to be flagged for potential categorical conversion (must be > 0).
            Default: 5.
        feature_correlation_threshold: Minimum correlation coefficient (range: 0.0-1.0)
            for features to be flagged as highly correlated. Default: 0.8.
        string_similarity_threshold_short: Similarity threshold (range: 0-100) for
            detecting similar strings of 7 characters or fewer. Default: 80.
        string_similarity_threshold_medium: Similarity threshold (range: 0-100) for
            detecting similar strings of 8-12 characters. Default: 85.
        string_similarity_threshold_long: Similarity threshold (range: 0-100) for
            detecting similar strings longer than 12 characters. Default: 90.
        string_length_threshold_factor: Multiplier for average string length to detect
            unusually long strings (must be > 0.0). Default: 2.0.
        class_imbalance_threshold: Maximum acceptable proportion of majority class
            (range: 0.0-1.0). Values exceeding this indicate class imbalance. Default: 0.8.
        high_cardinality_threshold: Maximum acceptable ratio of unique values to total
            rows (range: 0.0-1.0). Features exceeding this may be IDs. Default: 0.5.
        target_leakage_threshold: Minimum correlation coefficient (range: 0.0-1.0) with
            target that indicates potential data leakage. Default: 0.95.
        target_skewness_threshold: Maximum acceptable absolute skewness (must be > 0.0).
            Values exceeding this indicate highly skewed distributions. Default: 1.0.
        target_kurtosis_threshold: Maximum acceptable excess kurtosis (must be > 0.0).
            Values exceeding this indicate heavy-tailed distributions. Default: 3.0.
        minimum_samples_threshold: Minimum number of samples required in the dataset
            (must be > 0). Default: 20.
    """

    missing_value_column_threshold: float = field(
        default=0.25,
        validator=[validators.instance_of(float), validators.ge(0.0), validators.le(1.0)],
    )
    """Threshold for high missing values in columns (default: 0.25 or 25%)."""

    missing_value_row_threshold: float = field(
        default=0.5,
        validator=[validators.instance_of(float), validators.ge(0.0), validators.le(1.0)],
    )
    """Threshold for high missing values in rows (default: 0.5 or 50%)."""

    int_few_uniques_threshold: int = field(
        default=5,
        validator=[validators.instance_of(int), validators.gt(0)],
    )
    """Threshold for integer columns with few unique values (default: 5)."""

    feature_correlation_threshold: float = field(
        default=0.8,
        validator=[validators.instance_of(float), validators.ge(0.0), validators.le(1.0)],
    )
    """Threshold for high feature correlation (default: 0.8)."""

    string_similarity_threshold_short: int = field(
        default=80,
        validator=[validators.instance_of(int), validators.ge(0), validators.le(100)],
    )
    """Similarity threshold for short strings (<=7 chars) (default: 80)."""

    string_similarity_threshold_medium: int = field(
        default=85,
        validator=[validators.instance_of(int), validators.ge(0), validators.le(100)],
    )
    """Similarity threshold for medium strings (7-12 chars) (default: 85)."""

    string_similarity_threshold_long: int = field(
        default=90,
        validator=[validators.instance_of(int), validators.ge(0), validators.le(100)],
    )
    """Similarity threshold for long strings (>12 chars) (default: 90)."""

    string_length_threshold_factor: float = field(
        default=2.0,
        validator=[validators.instance_of(float), validators.gt(0.0)],
    )
    """Factor for detecting unusually long strings (default: 2.0)."""

    class_imbalance_threshold: float = field(
        default=0.8,
        validator=[validators.instance_of(float), validators.ge(0.0), validators.le(1.0)],
    )
    """Threshold for class imbalance detection (default: 0.8 or 80%)."""

    high_cardinality_threshold: float = field(
        default=0.5,
        validator=[validators.instance_of(float), validators.ge(0.0), validators.le(1.0)],
    )
    """Threshold for high cardinality detection (default: 0.5 or 50%)."""

    target_leakage_threshold: float = field(
        default=0.95,
        validator=[validators.instance_of(float), validators.ge(0.0), validators.le(1.0)],
    )
    """Threshold for target leakage detection (default: 0.95)."""

    target_skewness_threshold: float = field(
        default=1.0,
        validator=[validators.instance_of(float), validators.gt(0.0)],
    )
    """Threshold for target skewness detection (default: 1.0)."""

    target_kurtosis_threshold: float = field(
        default=3.0,
        validator=[validators.instance_of(float), validators.gt(0.0)],
    )
    """Threshold for target kurtosis detection (default: 3.0)."""

    minimum_samples_threshold: int = field(
        default=20,
        validator=[validators.instance_of(int), validators.gt(0)],
    )
    """Minimum number of samples required in dataset (default: 20)."""


@define
class OctoDataHealthChecker:
    """Performs comprehensive data quality checks on OctoData datasets.

    This class analyzes datasets for various data quality issues including missing values,
    duplicates, feature correlations, data type inconsistencies, and string anomalies.
    Issues are categorized by severity (Critical, Warning, Info) and stored in a report
    format for easy review and action.

    Attributes:
        data: The pandas DataFrame containing the dataset to be checked.
        feature_cols: List of column names designated as features. Can be empty.
        target_columns: List of column names designated as targets. Can be empty.
        row_id: Name of the column containing unique row identifiers. Can be None.
        sample_id: Name of the column containing sample identifiers. Can be None.
        stratification_column: Name of the column used for stratification. Can be None.
        config: Configuration object containing customizable thresholds for health checks.
            Uses default HealthCheckConfig if not provided.
        issues: List of dictionaries storing detected data quality issues. Each issue
            contains category, type, affected items, severity, description, and
            recommended action.
    """

    data: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    """DataFrame containing the dataset to check."""

    feature_cols: list[str] = field(factory=list, validator=validators.instance_of(list))
    """List of feature column names."""

    target_columns: list[str] = field(factory=list, validator=validators.instance_of(list))
    """List of target column names."""

    row_id: str | None = field(default=None, validator=validators.optional(validators.instance_of(str)))
    """Name of the row ID column."""

    sample_id: str | None = field(default=None, validator=validators.optional(validators.instance_of(str)))
    """Name of the sample ID column."""

    stratification_column: str | None = field(default=None, validator=validators.optional(validators.instance_of(str)))
    """Name of the stratification column."""

    config: HealthCheckConfig = field(
        factory=HealthCheckConfig,
        validator=[validators.instance_of(HealthCheckConfig)],
    )
    """Configuration for health check thresholds."""

    issues: list[dict] = field(factory=list)
    """List to store detected health issues."""

    def add_issue(
        self,
        category: str,
        issue_type: str,
        affected_items: list[str],
        severity: str,
        description: str,
        action: str,
    ):
        """Add a health issue to the report.

        Args:
            category: Category of the issue (e.g., 'columns', 'rows', 'features').
            issue_type: Specific type of issue (e.g., 'high_missing_values',
                'duplicated_rows').
            affected_items: List of column names, row indices, or feature names
                affected by this issue.
            severity: Severity level of the issue ('Critical', 'Warning', or 'Info').
            description: Detailed description of the issue.
            action: Recommended action to address the issue.
        """
        issue = {
            "Category": category,
            "Issue Type": issue_type,
            "Affected Items": ", ".join(affected_items),
            "Severity": severity,
            "Description": description,
            "Recommended Action": action,
        }
        self.issues.append(issue)

    def generate_report(self):
        """Generate the full health report.

        Executes all configured health checks on the dataset and compiles
        identified issues into a comprehensive report.

        Returns:
            pd.DataFrame: A DataFrame containing all detected issues with columns:
                - Category: Type of data element affected (columns, rows, features)
                - Issue Type: Specific issue identifier
                - Affected Items: Comma-separated list of affected items
                - Severity: Issue severity level (Critical, Warning, Info)
                - Description: Detailed description of the issue
                - Recommended Action: Suggested steps to address the issue
        """
        self._check_minimum_samples()
        self._check_row_id_unique()
        self._check_critical_column_missing_values()
        self._check_features_not_all_null()
        self._check_feature_cols_missing_values()
        self._check_row_missing_values()
        self._check_int_col_with_few_uniques()
        self._check_duplicated_features()
        self._check_feature_feature_correlation()
        self._check_identical_features()
        self._check_duplicated_rows()
        self._check_infinity_values()
        self._check_string_mismatch()
        # self._check_string_out_of_bounds() # see issue#60

        return pd.DataFrame(self.issues)

    def _check_critical_column_missing_values(self):
        """Check for missing values in critical columns.

        Examines target columns, sample_id, row_id, and stratification_column for any
        missing values. These columns are considered critical for model training and
        data integrity, so any missing values are flagged as Critical severity.

        Note:
            Missing values in critical columns can cause failures in downstream
            modeling processes.
        """
        missing_value_share_col = self.data.isnull().mean()

        critical_columns = [
            *self.target_columns,
            self.sample_id,
            self.row_id,
            self.stratification_column,
        ]
        critical_columns_not_none = [col for col in critical_columns if col is not None]

        critical_missing = [col for col in critical_columns_not_none if missing_value_share_col.get(col, 0) > 0]
        if critical_missing:
            self.add_issue(
                category="columns",
                issue_type="critical_missing_values",
                affected_items=critical_missing,
                severity="Critical",
                description=("These critical columns (target, sample_id, or row_id) have missing values."),
                action=(
                    "Investigate and resolve missing values in these columns immediately. These are crucial for model training and data integrity."
                ),
            )

    def _check_feature_cols_missing_values(self):
        """Check for missing values in feature columns.

        Analyzes each feature column for missing values and categorizes them based
        on the configured threshold. Columns with high proportions of missing values
        are flagged as Critical, while those with lower proportions are flagged as Info.

        Uses:
            config.missing_value_column_threshold to distinguish between high and
            low missing value proportions.
        """
        missing_value_share_col = self.data.isnull().mean(axis=0)

        threshold = self.config.missing_value_column_threshold
        high_missing_cols = [col for col in self.feature_cols if missing_value_share_col.get(col, 0) > threshold]
        low_missing_cols = [col for col in self.feature_cols if 0 < missing_value_share_col.get(col, 0) <= threshold]

        if high_missing_cols:
            self.add_issue(
                category="columns",
                issue_type="high_missing_values",
                affected_items=high_missing_cols,
                severity="Critical",
                description=f"These feature columns have more than {threshold * 100:.0f}% missing values.",
                action=("Consider removing these columns or using advanced imputation techniques."),
            )

        if low_missing_cols:
            self.add_issue(
                category="columns",
                issue_type="low_missing_values",
                affected_items=low_missing_cols,
                severity="Info",
                description=f"These feature columns have some missing values (<={threshold * 100:.0f}%).",
                action="Consider appropriate imputation methods for these columns.",
            )

    def _check_row_missing_values(self):
        """Check for missing values in rows.

        Analyzes each row for missing values and categorizes them based on the
        configured threshold. Rows with high proportions of missing values are
        flagged as Critical, while those with lower proportions are flagged as Info.

        Uses:
            config.missing_value_row_threshold to distinguish between high and
            low missing value proportions.
        """
        missing_value_share_row = self.data.isnull().mean(axis=1)

        threshold = self.config.missing_value_row_threshold
        high_missing_rows = missing_value_share_row[missing_value_share_row > threshold]
        low_missing_rows = missing_value_share_row[
            (missing_value_share_row > 0) & (missing_value_share_row <= threshold)
        ]

        if not high_missing_rows.empty:
            self.add_issue(
                category="rows",
                issue_type="high_missing_values",
                affected_items=[str(idx) for idx in high_missing_rows.index],
                severity="Critical",
                description=(f"{len(high_missing_rows)} rows have more than {threshold * 100:.0f}% missing values."),
                action=("Consider removing these rows or using advanced imputation techniques."),
            )

        if not low_missing_rows.empty:
            self.add_issue(
                category="rows",
                issue_type="low_missing_values",
                affected_items=[str(idx) for idx in low_missing_rows.index],
                severity="Info",
                description=(f"{len(low_missing_rows)} rows have some missing values (<={threshold * 100:.0f}%)."),
                action="Review these rows and consider appropriate imputation methods.",
            )

    def _check_int_col_with_few_uniques(self):
        """Check for integer columns with few unique values.

        Identifies integer-type columns that have a small number of unique values,
        which may indicate they should be treated as categorical variables rather
        than numeric features.

        Uses:
            config.int_few_uniques_threshold to determine what constitutes "few"
            unique values.

        Note:
            Columns with only 1-2 unique values are not flagged as they are typically
            already handled elsewhere.
        """
        threshold = self.config.int_few_uniques_threshold
        int_cols_with_few_uniques = {
            col: self.data[col].nunique()
            for col in self.feature_cols
            if pd.api.types.is_integer_dtype(self.data[col]) and 2 < self.data[col].nunique() <= threshold
        }

        if int_cols_with_few_uniques:
            affected_items = list(int_cols_with_few_uniques.keys())
            self.add_issue(
                category="columns",
                issue_type="int_columns_with_few_uniques",
                affected_items=affected_items,
                severity="Warning",
                description=(
                    f"These integer columns have between 3 and {threshold} unique values: {', '.join(affected_items)}"
                ),
                action=(
                    "Consider converting these columns to categorical variables or investigate if they should be ordinal features."
                ),
            )

    def _check_duplicated_features(self):
        """Check for duplicate rows based on feature values.

        Identifies rows that have identical values across all feature columns.
        If sample_id is provided, also checks for duplicates when considering
        both features and sample_id together.

        Note:
            Duplicates in features AND sample_id are flagged as Critical, as they
            may indicate serious data integrity issues. Duplicates in features only
            are flagged as Warning.
        """
        duplicated_features = self.data[self.feature_cols].duplicated().any()

        if self.sample_id is not None:
            duplicated_features_and_sample = self.data[[*self.feature_cols, self.sample_id]].duplicated().any()
        else:
            duplicated_features_and_sample = None

        if duplicated_features:
            self.add_issue(
                category="rows",
                issue_type="duplicated_features",
                affected_items=["all_features"],
                severity="Warning",
                description=("There are duplicated rows when considering all feature columns."),
                action=("Investigate the cause of these duplicates and consider removing or consolidating them."),
            )

        if duplicated_features_and_sample:
            self.add_issue(
                category="rows",
                issue_type="duplicated_features_and_sample",
                affected_items=["all_features_and_sample_id"],
                severity="Critical",
                description=("There are duplicated rows when considering all feature columns and the sample ID."),
                action=(
                    "This is a critical issue. Investigate and resolve these duplicates immediately as they may indicate data integrity problems."
                ),
            )

    def _check_feature_feature_correlation(self, method: Literal["pearson", "kendall", "spearman"] = "spearman"):
        """Detect highly correlated feature pairs.

        Calculates pairwise correlations between all numeric features and identifies
        groups of features that exceed the correlation threshold. Highly correlated
        features can cause multicollinearity issues in modeling.

        Args:
            method: Correlation method to use ('pearson', 'kendall', or 'spearman').
                Default: 'spearman'.

        Uses:
            config.feature_correlation_threshold to determine what constitutes
            "high" correlation.

        Note:
            Only numeric (float and int) features are analyzed. String and categorical
            features are excluded from this check.
        """
        threshold = self.config.feature_correlation_threshold
        numeric_features = self.data[self.feature_cols].select_dtypes(include=[float, int]).columns
        corr_matrix = self.data[numeric_features].corr(method=method)

        highly_correlated: dict[str, set[str]] = {}
        for col in corr_matrix.columns:
            for row in corr_matrix.index:
                if col != row and abs(corr_matrix.loc[row, col]) > threshold:
                    if col not in highly_correlated:
                        highly_correlated[col] = set()
                    if row not in highly_correlated:
                        highly_correlated[row] = set()
                    highly_correlated[col].add(row)
                    highly_correlated[row].add(col)

        merged_groups: list[set[str]] = []
        for feature, correlated_features in highly_correlated.items():
            new_group = set(correlated_features) | {feature}
            merged = False
            for group in merged_groups:
                if not new_group.isdisjoint(group):
                    group.update(new_group)
                    merged = True
                    break
            if not merged:
                merged_groups.append(new_group)

        for group in merged_groups:
            correlation_details = []
            for feat1, feat2 in combinations(sorted(group), 2):
                corr_value: float = corr_matrix.loc[feat1, feat2]  # type: ignore
                if abs(corr_value) > threshold:
                    correlation_details.append(f"{feat1} - {feat2} ({corr_value:.2f})")

            correlation_description = ", ".join(correlation_details)

            self.add_issue(
                category="features",
                issue_type="high_correlation",
                affected_items=sorted(group),
                severity="Info",
                description=(f"The following features are highly correlated (>{threshold}): {correlation_description}"),
                action=("Consider removing or combining these highly correlated features to reduce multicollinearity."),
            )

    def _check_identical_features(self):
        """Identify features with identical values.

        Finds feature columns that contain exactly the same values despite having
        different column names. These redundant features should typically be removed
        to simplify the dataset.

        Note:
            This check is more strict than correlation checking - it identifies
            features that are 100% identical, not just highly correlated.
        """
        identical_features: dict[str, list[str]] = {col: [] for col in self.feature_cols}

        for col in self.feature_cols:
            for other_col in self.feature_cols:
                if col != other_col and self.data[col].equals(self.data[other_col]):
                    identical_features[col].append(other_col)

        identical_features = {k: v for k, v in identical_features.items() if v}

        for feature, identical_list in identical_features.items():
            self.add_issue(
                category="features",
                issue_type="identical_features",
                affected_items=[feature, *identical_list],
                severity="Warning",
                description=(
                    f"The feature '{feature}' is identical to the following feature(s): {', '.join(identical_list)}"
                ),
                action=("Consider removing redundant features to simplify the dataset and improve model performance."),
            )

    def _check_duplicated_rows(self):
        """Check for completely duplicated rows.

        Identifies rows that are exact duplicates across all columns (not just
        features). This is a comprehensive duplicate check that considers the
        entire dataset.

        Note:
            This differs from _check_duplicated_features which only considers
            feature columns.
        """
        duplicated_mask = self.data.duplicated()
        duplicated_rows = self.data[duplicated_mask]

        if not duplicated_rows.empty:
            num_duplicates = len(duplicated_rows)
            self.add_issue(
                category="rows",
                issue_type="duplicated_rows",
                affected_items=[str(idx) for idx in duplicated_rows.index],
                severity="Warning",
                description=f"Found {num_duplicates} duplicated row(s) in the dataset.",
                action=(
                    "Investigate these duplicates and consider removing or consolidating them based on your data requirements."
                ),
            )

    def _check_infinity_values(self):
        """Check for infinity values in feature columns.

        Scans all feature columns for positive and negative infinity values.
        Infinity values can arise from mathematical operations like division by zero
        and can cause issues in modeling algorithms.

        Note:
            Non-numeric columns are coerced to numeric before checking, with errors
            being ignored. This ensures robust checking across mixed data types.
        """
        numeric_df = self.data[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        infinity_mask = numeric_df.map(np.isinf)
        infinity_value_share = infinity_mask.mean()
        infinity_value_dict = {col: share for col, share in infinity_value_share.items() if share > 0}

        if infinity_value_dict:
            affected_items = list(infinity_value_dict.keys())
            description = "The following columns contain infinity values:\n"
            for col, share in infinity_value_dict.items():
                percentage = share * 100
                description += f"- {col}: {percentage:.2f}% of values\n"

            self.add_issue(
                category="columns",
                issue_type="infinity_values",
                affected_items=affected_items,
                severity="Info",
                description=description.strip(),
                action=(
                    "Investigate these columns and decide on an appropriate "
                    "strategy to handle infinity values (e.g., imputation, "
                    "removal, or capping)."
                ),
            )

    def _check_string_mismatch(self):
        """Find similar strings that may indicate data entry errors.

        Analyzes string and categorical columns to identify groups of similar values
        that might be misspellings, inconsistent formatting, or data entry errors.
        Numeric suffixes are ignored to focus on the base string content.

        Uses:
            - config.string_similarity_threshold_short for strings ≤7 characters
            - config.string_similarity_threshold_medium for strings 8-12 characters
            - config.string_similarity_threshold_long for strings >12 characters

        Note:
            Columns containing only integer strings are skipped. Uses fuzzy string
            matching (Levenshtein distance) to detect similarities.
        """
        string_mismatch = {}

        def remove_numbers(entry):
            """Remove numbers from the end of a string."""
            return re.sub(r"\d+$", "", str(entry))

        def determine_threshold(length):
            """Determine the similarity threshold based on the length of the string."""
            if length < 7:
                return self.config.string_similarity_threshold_short
            elif length <= 12:
                return self.config.string_similarity_threshold_medium
            else:
                return self.config.string_similarity_threshold_long

        def is_all_integers(series):
            """Check if all non-null values in a series are integers."""
            return series.dropna().apply(lambda x: str(x).isdigit()).all()

        for column in self.feature_cols:
            if self.data[column].dtype == object or self.data[column].dtype.name == "category":
                if is_all_integers(self.data[column]):
                    continue

                try:
                    column_values = self.data[column].dropna().apply(remove_numbers).unique()
                    if len(column_values) > 2:
                        processed = set()
                        similar_groups = []

                        for value in column_values:
                            if value not in processed:
                                threshold = determine_threshold(len(value))
                                similar = {
                                    other
                                    for other in column_values
                                    if value != other and fuzz.ratio(value, other) >= threshold
                                }
                                if similar:
                                    similar.add(value)
                                    similar_groups.append(list(similar))
                                    processed.update(similar)
                        if similar_groups:
                            string_mismatch[column] = similar_groups
                except Exception as e:
                    print(f"An error occurred while processing column {column}: {e}")

        if string_mismatch:
            for column, similar_groups in string_mismatch.items():
                description = (
                    f"Column '{column}' contains similar strings that might be misspellings or inconsistencies:\n"
                )
                for group in similar_groups:
                    description += f"- {', '.join(group)}\n"

                self.add_issue(
                    category="columns",
                    issue_type="string_mismatch",
                    affected_items=[column],
                    severity="Warning",
                    description=description.strip(),
                    action=(
                        "Review these similar strings and consider standardizing them to improve data consistency."
                    ),
                )

    def _check_string_out_of_bounds(self):
        """Detect unusually long strings in string and categorical columns.

        Identifies string values that are significantly longer than the average
        string length in their respective columns. Such outliers may indicate
        data quality issues, incorrect data entry, or the need for text truncation.

        Uses:
            config.string_length_threshold_factor as a multiplier of the average
            length to determine what constitutes "unusually long".

        Note:
            Only the first 5 unusually long strings are shown in the report for
            each column. Strings are truncated to 50 characters in the display.
        """
        length_threshold_factor = self.config.string_length_threshold_factor
        long_string = {}
        for column in self.feature_cols:
            if self.data[column].dtype == object or self.data[column].dtype.name == "category":
                try:
                    column_values = self.data[column].dropna().tolist()
                    avg_length = sum(len(str(value)) for value in column_values) / len(column_values)
                    long_strings = [
                        s for value in column_values if len(s := str(value)) > length_threshold_factor * avg_length
                    ]
                    if long_strings:
                        long_string[column] = long_strings
                except Exception as e:
                    logging.warning(f"Failed to process column '{column}' in _check_string_out_of_bounds: {e}")

        if long_string:
            for column, long_strings in long_string.items():
                description = (
                    f"Column '{column}' contains strings that are significantly longer than the average length:\n"
                )
                for value in long_strings[:5]:
                    description += f"- {value[:50]}{'...' if len(value) > 50 else ''}\n"
                if len(long_strings) > 5:
                    description += f"(and {len(long_strings) - 5} more...)\n"

                self.add_issue(
                    category="columns",
                    issue_type="string_out_of_bounds",
                    affected_items=[column],
                    severity="Warning",
                    description=description.strip(),
                    action=(
                        "Review these unusually long strings and consider if they are valid or if they need cleaning or truncation."
                    ),
                )

    def _check_class_imbalance(self):
        """Check for class imbalance in classification target columns.

        Analyzes target columns to detect severe class imbalance, which can negatively
        impact model performance. For each target column, calculates the proportion
        of the majority class and flags if it exceeds the threshold.

        Uses:
            config.class_imbalance_threshold to determine what constitutes severe
            class imbalance.

        Note:
            Only analyzes target columns with discrete values (object, category, or
            integer types with reasonable number of unique values). Regression targets
            are skipped.
        """
        threshold = self.config.class_imbalance_threshold

        for target_col in self.target_columns:
            if target_col not in self.data.columns:
                continue

            if self.data[target_col].dtype == float:
                continue

            value_counts = self.data[target_col].value_counts(dropna=True)

            if len(value_counts) <= 1:
                continue

            total_count = value_counts.sum()
            majority_class = value_counts.index[0]
            majority_proportion = value_counts.iloc[0] / total_count

            if majority_proportion > threshold:
                class_distribution = ", ".join(
                    [
                        f"{cls}: {count} ({count / total_count * 100:.1f}%)"
                        for cls, count in value_counts.head(5).items()
                    ]
                )

                self.add_issue(
                    category="target",
                    issue_type="class_imbalance",
                    affected_items=[target_col],
                    severity="Warning",
                    description=(
                        f"Target column '{target_col}' has severe class imbalance. "
                        f"Majority class '{majority_class}' represents {majority_proportion * 100:.1f}% of the data. "
                        f"Class distribution: {class_distribution}"
                    ),
                    action=(
                        "Consider using stratified sampling, class weights, or resampling techniques (SMOTE, undersampling). "
                        "Choose appropriate evaluation metrics (F1-score, PR-AUC) instead of accuracy. "
                        "Review available metrics in your modeling framework."
                    ),
                )

    def _check_high_cardinality(self):
        """Check for high cardinality categorical features.

        Identifies categorical or object-type features with an excessive number of
        unique values relative to the total number of rows. Such features are often
        ID-like columns that should not be used as features in modeling.

        Uses:
            config.high_cardinality_threshold to determine what constitutes "high"
            cardinality (as a proportion of total rows).

        Note:
            Only checks object and category dtype columns. Numeric columns are excluded.
        """
        threshold = self.config.high_cardinality_threshold
        total_rows = len(self.data)
        high_cardinality_features = {}

        for column in self.feature_cols:
            if self.data[column].dtype not in [object, "category"] and self.data[column].dtype.name != "category":
                continue

            unique_count = self.data[column].nunique()
            cardinality_ratio = unique_count / total_rows

            if cardinality_ratio > threshold:
                high_cardinality_features[column] = {"unique_count": unique_count, "ratio": cardinality_ratio}

        if high_cardinality_features:
            for column, stats in high_cardinality_features.items():
                self.add_issue(
                    category="columns",
                    issue_type="high_cardinality",
                    affected_items=[column],
                    severity="Warning",
                    description=(
                        f"Feature '{column}' has high cardinality with {stats['unique_count']} unique values "
                        f"({stats['ratio'] * 100:.1f}% of total rows). This may indicate an ID column or "
                        f"inappropriate feature for modeling."
                    ),
                    action=(
                        "Verify if this column is an identifier that should be excluded from features. "
                        "If it's a legitimate feature, consider encoding strategies like target encoding, "
                        "frequency encoding, or grouping rare categories."
                    ),
                )

    def _check_target_leakage(self):
        """Check for potential target leakage in features.

        Detects features that are suspiciously highly correlated with the target
        variable, which may indicate data leakage. Leakage occurs when features
        contain information about the target that would not be available at
        prediction time.

        Uses:
            config.target_leakage_threshold to determine what constitutes suspicious
            correlation with the target.

        Note:
            Only checks correlation for numeric features and numeric targets.
            Categorical features and targets are skipped as correlation calculation
            requires numeric data.
        """
        threshold = self.config.target_leakage_threshold

        numeric_targets = [
            col
            for col in self.target_columns
            if col in self.data.columns and pd.api.types.is_numeric_dtype(self.data[col])
        ]

        if not numeric_targets:
            return

        numeric_features = self.data[self.feature_cols].select_dtypes(include=[float, int]).columns

        if len(numeric_features) == 0:
            return

        for target_col in numeric_targets:
            suspicious_features = {}

            for feature in numeric_features:
                try:
                    correlation: float = self.data[[feature, target_col]].corr().loc[feature, target_col]  # type: ignore

                    if pd.notna(correlation) and abs(correlation) > threshold:
                        suspicious_features[feature] = correlation
                except Exception:
                    continue

            if suspicious_features:
                sorted_features = sorted(suspicious_features.items(), key=lambda x: abs(x[1]), reverse=True)

                feature_details = ", ".join([f"{feat} ({corr:.3f})" for feat, corr in sorted_features[:5]])

                self.add_issue(
                    category="features",
                    issue_type="target_leakage",
                    affected_items=[feat for feat, _ in sorted_features],
                    severity="Warning",
                    description=(
                        f"The following features have suspiciously high correlation (>{threshold}) "
                        f"with target '{target_col}': {feature_details}. "
                        f"This may indicate data leakage."
                    ),
                    action=(
                        "Investigate these features carefully. Verify they would be available at prediction time. "
                        "Check if they are derived from the target or contain future information. "
                        "Consider removing features that represent data leakage to avoid overfitting."
                    ),
                )

    def _check_target_distribution(self):
        """Check for problematic distributions in regression target columns.

        Analyzes numeric target columns for skewness and heavy tails (kurtosis),
        which can negatively impact regression model performance. Highly skewed
        or heavy-tailed distributions may require transformation or robust modeling
        approaches.

        Uses:
            - config.target_skewness_threshold to detect highly skewed distributions
            - config.target_kurtosis_threshold to detect heavy-tailed distributions

        Note:
            Only analyzes numeric (float or int) target columns. Categorical targets
            are skipped as they're handled by class imbalance checks.
        """
        skewness_threshold = self.config.target_skewness_threshold
        kurtosis_threshold = self.config.target_kurtosis_threshold

        for target_col in self.target_columns:
            if target_col not in self.data.columns:
                continue

            if not pd.api.types.is_numeric_dtype(self.data[target_col]):
                continue

            if pd.api.types.is_integer_dtype(self.data[target_col]):
                unique_count = self.data[target_col].nunique()
                if unique_count < 20:
                    continue

            try:
                target_data: pd.Series[float | int] = self.data[target_col].dropna()

                if len(target_data) < 3:
                    continue

                skewness: float = target_data.skew()  # type: ignore
                kurtosis: float = target_data.kurtosis()  # type: ignore

                stats = {
                    "mean": target_data.mean(),
                    "std": target_data.std(),
                    "min": target_data.min(),
                    "q25": target_data.quantile(0.25),
                    "median": target_data.median(),
                    "q75": target_data.quantile(0.75),
                    "max": target_data.max(),
                }

                issues_found = []

                if abs(skewness) > skewness_threshold:
                    skew_direction = "right" if skewness > 0 else "left"
                    issues_found.append(f"High skewness: {skewness:.3f} ({skew_direction}-skewed)")

                if kurtosis > kurtosis_threshold:
                    issues_found.append(f"Heavy tails: excess kurtosis {kurtosis:.3f}")

                if issues_found:
                    description = (
                        f"Target '{target_col}' has a problematic distribution:\n"
                        f"- {', '.join(issues_found)}\n"
                        f"- Statistics: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                        f"median={stats['median']:.2f}\n"
                        f"- Range: [{stats['min']:.2f}, {stats['max']:.2f}], "
                        f"IQR=[{stats['q25']:.2f}, {stats['q75']:.2f}]"
                    )

                    actions = []
                    if skewness > skewness_threshold:
                        actions.append("log transformation or Box-Cox transformation to reduce right skewness")
                    elif skewness < -skewness_threshold:
                        actions.append("square or exponential transformation to reduce left skewness")

                    if kurtosis > kurtosis_threshold:
                        actions.append("consider robust loss functions (Huber loss, quantile regression)")
                        actions.append("investigate and handle outliers")

                    action_text = "Consider applying " + "; or ".join(actions) + "."

                    self.add_issue(
                        category="target",
                        issue_type="problematic_distribution",
                        affected_items=[target_col],
                        severity="Warning",
                        description=description,
                        action=action_text,
                    )

            except Exception:
                continue

    def _check_minimum_samples(self):
        """Check if dataset has minimum number of samples.

        Validates that the dataset contains at least the minimum required number
        of samples for meaningful analysis and modeling.

        Uses:
            config.minimum_samples_threshold to determine the minimum required
            number of samples.
        """
        threshold = self.config.minimum_samples_threshold
        actual_samples = len(self.data)

        if actual_samples < threshold:
            self.add_issue(
                category="rows",
                issue_type="insufficient_samples",
                affected_items=["dataset"],
                severity="Critical",
                description=f"Dataset has only {actual_samples} samples, which is below the minimum threshold of {threshold}.",
                action=(
                    f"Collect more data to reach at least {threshold} samples, or adjust the minimum_samples_threshold "
                    "if your use case allows for smaller datasets."
                ),
            )

    def _check_row_id_unique(self):
        """Check if row_id column contains unique values.

        If a row_id column is specified, ensures that all values in that column
        are unique, as row IDs are used to uniquely identify each data row.
        """
        if self.row_id and self.row_id in self.data.columns and not self.data[self.row_id].is_unique:
            duplicate_count = self.data[self.row_id].duplicated().sum()
            self.add_issue(
                category="columns",
                issue_type="duplicate_row_ids",
                affected_items=[self.row_id],
                severity="Critical",
                description=f"Row ID column '{self.row_id}' contains {duplicate_count} duplicate values. Each row ID must be unique.",
                action="Investigate and resolve duplicate row IDs. Each row must have a unique identifier.",
            )

    def _check_features_not_all_null(self):
        """Check if feature columns are entirely null.

        Checks that each feature column contains at least one non-null value.
        Features with all null values provide no information for modeling.
        """
        if not self.feature_cols:
            return

        all_null_features = [
            col for col in self.feature_cols if col in self.data.columns and self.data[col].isnull().all()
        ]

        if all_null_features:
            self.add_issue(
                category="columns",
                issue_type="all_null_features",
                affected_items=all_null_features,
                severity="Critical",
                description=f"Feature columns are entirely null: {', '.join(all_null_features)}",
                action="Remove these columns or investigate why they contain no data. Features with all null values cannot be used for modeling.",
            )
