"""OctoData Preparator."""

import numpy as np
import pandas as pd

from ..logger import get_logger
from .prepared_data import PreparedData

logger = get_logger()

DEFAULT_NULL_VALUES = {"none", "null", "nan", "na", "", "\x00", "\x00\x00", "n/a"}
DEFAULT_INF_VALUES = {"inf", "infinity", "∞"}


def prepare_data(
    data: pd.DataFrame,
    feature_cols: list[str],
    sample_id_col: str,
    row_id_col: str | None,
) -> PreparedData:
    """Run all data preparation steps and return PreparedData instance.

    Args:
        data: DataFrame containing the dataset.
        feature_cols: List of all feature columns in the dataset.
        sample_id_col: Identifier for sample instances.
        row_id_col: Unique row identifier, or None to auto-generate.

    Returns:
        PreparedData: The transformed data with effective feature columns, row_id_col, etc.
    """
    feature_cols = _sort_features(feature_cols)
    data = _standardize_null_values(data)
    data = _standardize_inf_values(data)
    feature_cols = _remove_singlevalue_features(data, feature_cols)
    data = _transform_bool_to_int(data)
    data, row_id_col = _create_row_id_col(data, row_id_col)
    data = _add_group_features(data, feature_cols, sample_id_col)  # needs to be done at the end

    return PreparedData(
        data=data,
        feature_cols=feature_cols,
        row_id_col=row_id_col,
    )


def _sort_features(feature_cols: list[str]) -> list[str]:
    """Sort feature columns deterministically by length and lexicographically."""
    return sorted(feature_cols, key=lambda col: (len(s := str(col)), s))


def _remove_singlevalue_features(data: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Remove features that contain only a single unique value."""
    removed_features = [feature for feature in feature_cols if data[feature].nunique() <= 1]
    if removed_features:
        logger.info(f"Removing {len(removed_features)} feature(s) with single unique value: {removed_features}")
    return [feature for feature in feature_cols if data[feature].nunique() > 1]


def _transform_bool_to_int(data: pd.DataFrame) -> pd.DataFrame:
    """Convert all boolean columns to integer."""
    bool_cols = data.select_dtypes(include="bool").columns
    data = data.copy()
    data[bool_cols] = data[bool_cols].astype(int)
    return data


def _create_row_id_col(data: pd.DataFrame, row_id_col: str | None) -> tuple[pd.DataFrame, str]:
    """Create a unique row identifier if not provided."""
    if not row_id_col:
        data = data.copy()
        data["row_id"] = list(range(len(data)))
        row_id_col = "row_id"
    return data, row_id_col


def _add_group_features(data: pd.DataFrame, feature_cols: list[str], sample_id_col: str) -> pd.DataFrame:
    """Add group feature columns for data splitting and tracking.

    Creates two grouping columns used for data splitting strategies:
    - group_features: Groups rows with identical feature values
    - group_sample_and_features: Groups rows by sample_id_col OR identical features
      using transitive closure (if row A and B share sample_id_col, and B and C
      share features, then A, B, and C are all in the same group)

    The DataFrame index is reset after adding these columns.

    Note:
        This must be called at the end of preparation, after all feature
        transformations are complete, to ensure accurate grouping.
    """
    # Step 1: Create group_features column (groups by identical feature values)
    data = data.assign(group_features=lambda df_: df_.groupby(feature_cols, dropna=False, observed=True).ngroup())

    # Step 2: Initialize Union-Find data structure
    # Each row starts as its own parent (independent set)
    # parent[i] represents the parent of row i in the forest of sets
    parent = list(range(len(data)))

    def find(x):
        """Find the root (representative) of the set containing x.

        Uses path compression: makes all nodes on the path point directly to the root.
        This flattens the tree structure for faster future lookups.

        Example: If we have 0->1->2->3 and call find(0), it becomes 0->3, 1->3, 2->3
        """
        if parent[x] != x:
            # Recursively find root and compress path
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        """Merge the sets containing x and y.

        Finds the roots of both sets and makes one point to the other.
        After this operation, x and y are in the same connected component.

        Example: union(0, 5) connects all elements in set of 0 with all elements in set of 5
        """
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            # Make root_x point to root_y (could be reversed, doesn't matter)
            parent[root_x] = root_y

    # Step 3: Union rows with the same sample_id_col
    # All rows with same sample_id_col must be in the same group
    sample_groups = data.groupby(sample_id_col, dropna=False).indices
    for indices in sample_groups.values():
        indices_list = list(indices)
        # Connect all rows in this sample_id_col group by linking them sequentially
        # e.g., if rows [2, 5, 7] have same sample_id_col, do: union(2,5), union(5,7)
        for i in range(1, len(indices_list)):
            union(indices_list[0], indices_list[i])

    # Step 4: Union rows with the same features (group_features)
    # All rows with identical feature values must be in the same group
    feature_groups = data.groupby("group_features").indices
    for indices in feature_groups.values():
        indices_list = list(indices)
        # Connect all rows in this feature group by linking them sequentially
        for i in range(1, len(indices_list)):
            union(indices_list[0], indices_list[i])

    # Step 5: Assign each row to its root component
    # find(i) returns the representative (root) of the set containing row i
    # All rows with the same root are in the same connected component
    group_sample_and_features = [find(i) for i in range(len(data))]

    # Step 6: Renumber groups to be sequential starting from 0
    # Roots might be arbitrary numbers (e.g., [5, 5, 12, 12, 23])
    # Convert to sequential (e.g., [0, 0, 1, 1, 2]) for cleaner output
    unique_groups = {}
    next_group_id = 0
    normalized_groups = []
    for group_root in group_sample_and_features:
        if group_root not in unique_groups:
            unique_groups[group_root] = next_group_id
            next_group_id += 1
        normalized_groups.append(unique_groups[group_root])

    # Step 7: Add the column and reset index
    data = data.assign(group_sample_and_features=normalized_groups).reset_index(drop=True)

    return data


def _standardize_null_values(data: pd.DataFrame) -> pd.DataFrame:
    """Standardize null values to np.nan.

    Converts common string representations of null values (case-insensitive)
    to np.nan for consistent handling. Recognized values include: 'none',
    'null', 'nan', 'na', '', and similar variants.
    """
    null_values_case_insensitive = {val.lower() for val in DEFAULT_NULL_VALUES}

    def replace_null(x):
        if isinstance(x, str) and x.strip().lower() in null_values_case_insensitive:
            return np.nan
        return x

    return data.map(replace_null)


def _standardize_inf_values(data: pd.DataFrame) -> pd.DataFrame:
    """Standardize infinity values to np.inf.

    Converts common string representations of infinity (case-insensitive)
    to np.inf or -np.inf for consistent numerical handling. Recognized values
    include: 'inf', 'infinity', '∞', and their negative variants.
    """
    infinity_values_case_insensitive = {val.lower() for val in DEFAULT_INF_VALUES}

    def replace_infinity(x):
        if isinstance(x, str):
            stripped = x.strip().lower()
            if stripped in infinity_values_case_insensitive:
                return np.inf
            if stripped in {f"-{val}" for val in infinity_values_case_insensitive}:
                return -np.inf
        return x

    return data.map(replace_infinity)
