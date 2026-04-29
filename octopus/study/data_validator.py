"""OctoData Validator."""

from collections import Counter

import pandas as pd
from attrs import define, field, validators

from octopus.types import MLType

from ..datasplit import DATASPLIT_COL


@define
class OctoDataValidator:
    """Validator for OctoData."""

    data: pd.DataFrame
    """DataFrame containing the dataset."""

    feature_cols: list[str]
    """List of all feature columns in the dataset."""

    ml_type: MLType = field(validator=validators.instance_of(MLType))
    """Machine learning type (MLType.BINARY, MLType.REGRESSION, etc.)."""

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
            self._validate_classification_target,
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
        """Validate stratification column constraints.

        Ensures that the stratification column (if specified):
        - Is not the same as sample_id_col or row_id_col
        - Has a supported dtype (integer or boolean)

        Returns:
            None: Returns early if stratification_col is not set.

        Raises:
            ValueError: If stratification_col is the same as sample_id_col or row_id_col,
                or has an unsupported dtype.
        """
        if not self.stratification_col:
            return

        if self.stratification_col in [self.sample_id_col, self.row_id_col]:
            raise ValueError("Stratification column cannot be the same as sample_id_col or row_id_col")

        dtype = self.data[self.stratification_col].dtype
        if not (pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)):
            raise ValueError(
                f"Stratification column '{self.stratification_col}' has unsupported dtype '{dtype}'. "
                "Only int and bool are supported."
            )

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
        preparation: 'datasplit_group' and 'row_id' (if not provided by user).

        Raises:
            ValueError: If any reserved column names are found in the DataFrame.
        """
        reserved = [DATASPLIT_COL]
        if not self.row_id_col:
            reserved.append("row_id")

        conflicts = [col for col in reserved if col in self.data.columns]
        if conflicts:
            raise ValueError(
                f"Reserved column names found in data: {', '.join(conflicts)}. "
                "These column names are used internally by Octopus."
            )

    def _validate_classification_target(self):
        """Validate target column for classification tasks (binary and multiclass).

        Enforces a unified contract:
        - Target must be integer or boolean dtype (rejects float, object, string-categorical)
        - Target must not contain missing values
        - For binary: exactly 2 unique values, positive_class set and present
        - For multiclass: at least 3 unique values

        Returns:
            None: Returns early for non-classification ml_types.

        Raises:
            ValueError: If any validation fails.
        """
        if self.ml_type not in (MLType.BINARY, MLType.MULTICLASS):
            return

        if self.target_col is None:
            return

        target_data = self.data[self.target_col]

        if not (pd.api.types.is_integer_dtype(target_data) or pd.api.types.is_bool_dtype(target_data)):
            if isinstance(target_data.dtype, pd.CategoricalDtype):
                raise ValueError(
                    f"Categorical target columns are not supported for classification. "
                    f"Convert target to integer labels. Got categories: {list(target_data.cat.categories)}"
                )
            raise ValueError(
                f"Classification target must be integer or boolean dtype, got {target_data.dtype}. "
                f"Convert target to integer labels."
            )

        if target_data.isna().any():
            raise ValueError(
                "Classification target contains missing values (NaN). "
                "Remove or impute missing target values before fitting."
            )

        unique_values = target_data.dropna().unique()

        if self.ml_type == MLType.BINARY:
            if len(unique_values) != 2:
                raise ValueError(
                    f"Binary classification requires exactly 2 unique target values, "
                    f"found {len(unique_values)}: {sorted(unique_values)}"
                )
            if isinstance(self.positive_class, bool):
                raise ValueError(
                    "positive_class must be an integer, not a boolean. "
                    "Use positive_class=1 for True or positive_class=0 for False."
                )
            if self.positive_class is None:
                raise ValueError("positive_class must be specified for binary classification.")
            if self.positive_class not in unique_values:
                raise ValueError(
                    f"positive_class {self.positive_class} not found in target. "
                    f"Available values: {sorted(unique_values)}"
                )

        elif self.ml_type == MLType.MULTICLASS:
            if len(unique_values) < 3:
                raise ValueError(
                    f"Multiclass classification requires at least 3 unique target values, "
                    f"found {len(unique_values)}: {sorted(unique_values)}. "
                    f"Use ml_type=MLType.BINARY for 2-class problems."
                )
