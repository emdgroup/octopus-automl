"""OctoData Validator."""

from collections import Counter

import pandas as pd


def validate_data(
    data: pd.DataFrame,
    feature_cols: list[str],
    ml_type: str,
    *,
    duration_col: str | None = None,
    event_col: str | None = None,
    target_col: str | None = None,
    sample_id_col: str | None = None,
    row_id_col: str | None = None,
    stratification_col: str | None = None,
    positive_class: int | None = None,
) -> None:
    """Run all validation checks on the OctoData configuration.

    Collects all validation errors and raises a single ValueError with
    all error messages if any check fails.

    Args:
        data: DataFrame containing the dataset.
        feature_cols: List of feature column names.
        ml_type: Machine learning type (e.g., "classification", "regression").
        duration_col: Time-to-event duration column name.
        event_col: Time-to-event event indicator column name.
        target_col: Target column name.
        sample_id_col: Sample identifier column name.
        row_id_col: Row identifier column name.
        stratification_col: Stratification column name.
        positive_class: Positive class label for binary classification.

    Raises:
        ValueError: If any validation check fails.
    """
    errors = [
        _validate_nonempty_dataframe(data),
        _validate_reserved_column_conflicts(data, row_id_col),
        _validate_columns_exist(
            data,
            feature_cols,
            duration_col,
            event_col,
            target_col,
            sample_id_col,
            row_id_col,
            stratification_col,
        ),
        _validate_duplicated_columns(feature_cols, duration_col, event_col, target_col, sample_id_col, row_id_col),
        _validate_feature_target_overlap(feature_cols, target_col, duration_col, event_col),
        _validate_stratification_col(stratification_col, sample_id_col, row_id_col),
        _validate_column_dtypes(data, feature_cols, duration_col, event_col, target_col),
        _validate_positive_class(data, ml_type, target_col, positive_class),
    ]
    errors = [e for e in errors if e is not None]

    if errors:
        raise ValueError("Multiple validation errors found:\n" + "\n".join(f"- {e}" for e in errors))


def _validate_columns_exist(
    data: pd.DataFrame,
    feature_cols: list[str],
    duration_col: str | None,
    event_col: str | None,
    target_col: str | None,
    sample_id_col: str | None,
    row_id_col: str | None,
    stratification_col: str | None,
) -> str | None:
    """Validate that all relevant columns exist in the DataFrame."""
    relevant_columns = feature_cols + [
        c for c in (duration_col, event_col, target_col, sample_id_col, row_id_col, stratification_col) if c is not None
    ]
    missing_columns = [col for col in relevant_columns if col not in data.columns]
    if missing_columns:
        return f"Columns not found in the DataFrame: {', '.join(missing_columns)}"
    return None


def _validate_duplicated_columns(
    feature_cols: list[str],
    duration_col: str | None,
    event_col: str | None,
    target_col: str | None,
    sample_id_col: str | None,
    row_id_col: str | None,
) -> str | None:
    """Validate that no duplicate column names exist in the configuration."""
    columns_to_check = feature_cols + [
        c for c in (duration_col, event_col, target_col, sample_id_col, row_id_col) if c is not None
    ]
    duplicates = [col for col, count in Counter(columns_to_check).items() if count > 1]
    if duplicates:
        return f"Duplicate columns found: {', '.join(duplicates)}"
    return None


def _validate_stratification_col(
    stratification_col: str | None,
    sample_id_col: str | None,
    row_id_col: str | None,
) -> str | None:
    """Validate that stratification_col is not a reserved identifier."""
    if stratification_col and stratification_col in [sample_id_col, row_id_col]:
        return "Stratification column cannot be the same as sample_id_col or row_id_col"
    return None


def _validate_column_dtypes(
    data: pd.DataFrame,
    feature_cols: list[str],
    duration_col: str | None,
    event_col: str | None,
    target_col: str | None,
) -> str | None:
    """Validate that feature and target columns have supported data types."""
    non_matching_columns = []

    columns_to_check = feature_cols + [c for c in (duration_col, event_col, target_col) if c is not None]

    for column in columns_to_check:
        dtype = data[column].dtype
        if not (
            pd.api.types.is_integer_dtype(dtype)
            or pd.api.types.is_float_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            non_matching_columns.append(f"{column} ({dtype})")

    if non_matching_columns:
        return f"Columns with wrong dtypes: {', '.join(non_matching_columns)}"
    return None


def _validate_nonempty_dataframe(data: pd.DataFrame) -> str | None:
    """Validate that the DataFrame is not empty."""
    if len(data) == 0:
        return "DataFrame is empty. Cannot proceed with empty dataset."
    return None


def _validate_feature_target_overlap(
    feature_cols: list[str],
    target_col: str | None,
    duration_col: str | None,
    event_col: str | None,
) -> str | None:
    """Validate that no column is both a feature and a target."""
    present = [c for c in (target_col, duration_col, event_col) if c is not None]
    overlap = set(feature_cols) & set(present)
    if overlap:
        return f"Columns cannot be both features and targets: {', '.join(sorted(overlap))}"
    return None


def _validate_reserved_column_conflicts(
    data: pd.DataFrame,
    row_id_col: str | None,
) -> str | None:
    """Validate that reserved column names are not present in the DataFrame."""
    reserved = ["group_features", "group_sample_and_features"]
    if not row_id_col:
        reserved.append("row_id")

    conflicts = [col for col in reserved if col in data.columns]
    if conflicts:
        return (
            f"Reserved column names found in data: {', '.join(conflicts)}. "
            "These column names are used internally by Octopus."
        )
    return None


def _validate_positive_class(
    data: pd.DataFrame,
    ml_type: str,
    target_col: str | None,
    positive_class: int | None,
) -> str | None:
    """Validate positive class for binary classification."""
    if ml_type != "classification":
        return None

    if positive_class is None:
        return "For binary classification, `positive_class` must be specified."

    target_data = data[target_col]

    if not pd.api.types.is_integer_dtype(target_data):
        return f"Target column must be integer type for binary classification, got {target_data.dtype}"

    unique_values = target_data.dropna().unique()
    if len(unique_values) != 2:
        return f"Binary classification requires exactly 2 unique values, found {len(unique_values)}: {unique_values}"

    if positive_class not in unique_values:
        return f"positive_class {positive_class} not found in target. Available: {unique_values}"

    return None
