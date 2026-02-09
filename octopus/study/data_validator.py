"""OctoData Validator."""

from collections import Counter

import pandas as pd
from attrs import define, field, validators


@define
class OctoDataValidator:
    """Validator for OctoData."""

    data: pd.DataFrame
    """DataFrame containing the dataset."""

    feature_cols: list[str]
    """List of all feature columns in the dataset."""

    ml_type: str
    """Machine learning type (classification, regression, etc.)."""

    duration_col: str | None = field(default=None, validator=validators.optional(validators.instance_of(str)))
    """Duration column for time-to-event tasks. None for non-time-to-event tasks."""

    event_col: str | None = field(default=None, validator=validators.optional(validators.instance_of(str)))
    """Event column for time-to-event tasks. None for non-time-to-event tasks."""

    target_col: str | None = field(default=None, validator=validators.optional(validators.instance_of(str)))
    """Target column in the dataset. For regression and classification,
    only one target is allowed. For time-to-event, two targets need to be provided.
    """

    sample_id_col: str | None = field(default=None, validator=validators.optional(validators.instance_of(str)))
    """Identifier for sample instances."""

    row_id_col: str | None = field(default=None, validator=validators.optional(validators.instance_of(str)))
    """Unique row identifier."""

    stratification_col: str | None = field(default=None, validator=validators.optional(validators.instance_of(str)))
    """List of columns used for stratification."""

    positive_class: int | None = field(default=None, validator=validators.optional(validators.instance_of(int)))
    """The positive class label for binary classification. None for non-classification tasks."""

    def validate(self):
        """Run all validation checks on the OctoData configuration.

        Performs comprehensive validation checks in a specific order to ensure
        data quality and structural integrity before processing. Validates basic
        structure, column names, relationships, data types, and data quality.

        Collects all validation errors and raises a single exception with all
        error messages if any validation fails.

        Raises:
            ValueError: If any validation check fails. Includes all validation
                errors in a single exception message.
        """
        validators = [
            self._validate_nonempty_dataframe,
            self._validate_reserved_column_conflicts,
            self._validate_columns_exist,
            self._validate_duplicated_columns,
            self._validate_feature_target_overlap,
            self._validate_stratification_col,
            self._validate_column_dtypes,
            self._validate_positive_class,
        ]

        errors = []
        for validator in validators:
            try:
                validator()
            except ValueError as e:
                errors.append(f"- {e!s}")

        if errors:
            error_message = "Multiple validation errors found:\n" + "\n".join(errors)
            raise ValueError(error_message)

    def _validate_columns_exist(self):
        """Validate that all relevant columns exist in the DataFrame.

        Checks that all columns specified in feature_cols, target_cols,
        sample_id_col, row_id_col, and stratification_col are present in the DataFrame.

        Raises:
            ValueError: If any relevant columns are missing from the DataFrame.
        """
        relevant_columns = self.feature_cols + [
            c
            for c in (
                self.duration_col,
                self.event_col,
                self.target_col,
                self.sample_id_col,
                self.row_id_col,
                self.stratification_col,
            )
            if c is not None
        ]
        missing_columns = [col for col in relevant_columns if col not in self.data.columns]
        if missing_columns:
            missing_str = ", ".join(missing_columns)
            raise ValueError(f"Columns not found in the DataFrame: {missing_str}")

    def _validate_duplicated_columns(self):
        """Validate that no duplicate column names exist in the configuration.

        Validates that no column appears multiple times across feature_cols,
        target_cols, sample_id_col, and row_id_col. This prevents ambiguous column
        references.

        Raises:
            ValueError: If any column name appears more than once in the
                configuration.
        """
        columns_to_check = self.feature_cols + [
            c
            for c in (
                self.duration_col,
                self.event_col,
                self.target_col,
                self.sample_id_col,
                self.row_id_col,
            )
            if c is not None
        ]
        duplicates = [col for col, count in Counter(columns_to_check).items() if count > 1]

        if duplicates:
            duplicates_str = ", ".join(duplicates)
            raise ValueError(f"Duplicate columns found: {duplicates_str}")

    def _validate_stratification_col(self):
        """Validate that stratification_col is not a reserved identifier.

        Ensures that the stratification column (if specified) is not the same as
        sample_id_col or row_id_col, which are reserved for data identification.

        Raises:
            ValueError: If stratification_col is the same as sample_id_col or row_id_col.
        """
        if self.stratification_col and self.stratification_col in [
            self.sample_id_col,
            self.row_id_col,
        ]:
            raise ValueError("Stratification column cannot be the same as sample_id_col or row_id_col")

    def _validate_column_dtypes(self):
        """Validate that feature and target columns have supported data types.

        Checks that all feature columns, target columns, and stratification column
        (if present) have dtypes that are compatible with machine learning models.
        Supported dtypes are: integer, float, boolean, and categorical.

        Raises:
            ValueError: If any column has an unsupported dtype. Lists all columns
                with invalid dtypes along with their actual dtype.
        """
        non_matching_columns = []

        columns_to_check = self.feature_cols + [
            c
            for c in (
                self.duration_col,
                self.event_col,
                self.target_col,
            )
            if c is not None
        ]

        for column in columns_to_check:
            dtype = self.data[column].dtype
            if not (
                pd.api.types.is_integer_dtype(dtype)
                or pd.api.types.is_float_dtype(dtype)
                or pd.api.types.is_bool_dtype(dtype)
                or isinstance(dtype, pd.CategoricalDtype)
            ):
                non_matching_columns.append(f"{column} ({dtype})")

        if non_matching_columns:
            non_matching_str = ", ".join(non_matching_columns)
            raise ValueError(f"Columns with wrong dtypes: {non_matching_str}")

    def _validate_nonempty_dataframe(self):
        """Validate that the DataFrame is not empty.

        Ensures the DataFrame contains at least one row of data.

        Raises:
            ValueError: If the DataFrame has zero rows.
        """
        if len(self.data) == 0:
            raise ValueError("DataFrame is empty. Cannot proceed with empty dataset.")

    def _validate_feature_target_overlap(self):
        """Validate that no column is both a feature and a target.

        Ensures that feature_cols and target_cols do not share any column
        names, preventing logical conflicts in model training.

        Raises:
            ValueError: If any columns appear in both feature_cols and
                target_cols.
        """
        present = [c for c in (self.target_col, self.duration_col, self.event_col) if c is not None]
        overlap = set(self.feature_cols) & set(present)
        if overlap:
            raise ValueError(f"Columns cannot be both features and targets: {', '.join(sorted(overlap))}")

    def _validate_reserved_column_conflicts(self):
        """Validate that reserved column names are not present in the DataFrame.

        Checks for conflicts with columns that will be created during data
        preparation: 'group_features', 'group_sample_and_features', and 'row_id_col'
        (if not provided by user).

        Raises:
            ValueError: If any reserved column names are found in the DataFrame.
        """
        reserved = ["group_features", "group_sample_and_features"]
        if not self.row_id_col:
            reserved.append("row_id")

        conflicts = [col for col in reserved if col in self.data.columns]
        if conflicts:
            raise ValueError(
                f"Reserved column names found in data: {', '.join(conflicts)}. "
                "These column names are used internally by Octopus."
            )

    def _validate_positive_class(self):
        """Validate positive class for binary classification.

        For classification tasks, validates that:
        - Target column is integer type
        - Target has exactly 2 unique values (binary)
        - positive_class value exists in target column

        Returns:
            None: Returns early for non-classification ml_types or if positive_class is None.

        Raises:
            ValueError: If any validation fails for binary classification.
        """
        if self.ml_type != "classification":
            return

        target_data = self.data[self.target_col]

        if not pd.api.types.is_integer_dtype(target_data):
            raise ValueError(f"Target column must be integer type for binary classification, got {target_data.dtype}")

        unique_values = target_data.dropna().unique()
        if len(unique_values) != 2:
            raise ValueError(
                f"Binary classification requires exactly 2 unique values, found {len(unique_values)}: {unique_values}"
            )

        if self.positive_class not in unique_values:
            raise ValueError(f"positive_class {self.positive_class} not found in target. Available: {unique_values}")
