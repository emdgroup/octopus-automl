"""OctoData Preparator."""

import numpy as np
import pandas as pd
from attrs import define

from ..datasplit import DATASPLIT_COL
from ..logger import get_logger
from .prepared_data import PreparedData

logger = get_logger()

DEFAULT_NULL_VALUES = {"none", "null", "nan", "na", "", "\x00", "\x00\x00", "n/a"}
DEFAULT_INF_VALUES = {"inf", "infinity", "∞"}


@define
class OctoDataPreparator:
    """Validator for OctoData."""

    data: pd.DataFrame
    """DataFrame containing the dataset."""

    feature_cols: list[str]
    """List of all feature columns in the dataset."""

    sample_id_col: str
    """Identifier for sample instances."""

    row_id_col: str | None
    """Unique row identifier."""

    target_col: str | None = None
    """Target column name."""

    stratification_col: str | None = None
    """Stratification column name."""

    duration_col: str | None = None
    """Duration column for time-to-event tasks."""

    event_col: str | None = None
    """Event column for time-to-event tasks."""

    @property
    def _columns_to_standardize(self) -> list[str]:
        """Columns that should be standardized (null/inf conversion).

        Includes feature columns and all pipeline columns except sample_id_col,
        which must not be standardized to avoid merging unrelated samples
        during datasplit grouping.
        """
        return list(
            dict.fromkeys(
                self.feature_cols
                + [
                    c
                    for c in (
                        self.target_col,
                        self.row_id_col,
                        self.stratification_col,
                        self.duration_col,
                        self.event_col,
                    )
                    if c is not None
                ]
            )
        )

    def prepare(self) -> PreparedData:
        """Run all data preparation steps and return PreparedData instance.

        Returns:
            PreparedData: The transformed data with effective feature columns, row_id_col, etc.
        """
        self._sort_features()
        self._standardize_null_values()
        self._standardize_inf_values()
        self._remove_singlevalue_features()
        self._transform_bool_to_int()
        self._create_row_id_col()
        self._add_datasplit_group()  # needs to be done at the end

        return PreparedData(
            data=self.data,
            feature_cols=self.feature_cols,
            row_id_col=self.row_id_col,  # type: ignore[arg-type]  # row_id_col is always set after _create_row_id_col
        )

    def _sort_features(self):
        """Sort feature columns deterministically by length and lexicographically."""
        self.feature_cols = sorted(self.feature_cols, key=lambda col: (len(s := str(col)), s))

    def _remove_singlevalue_features(self):
        """Remove features that contain only a single unique value."""
        removed_features = [feature for feature in self.feature_cols if self.data[feature].nunique() <= 1]
        if removed_features:
            logger.info(f"Removing {len(removed_features)} feature(s) with single unique value: {removed_features}")
        self.feature_cols = [feature for feature in self.feature_cols if self.data[feature].nunique() > 1]

    def _transform_bool_to_int(self):
        """Convert all boolean columns to integer."""
        bool_cols = self.data.select_dtypes(include="bool").columns
        self.data[bool_cols] = self.data[bool_cols].astype(int)

    def _create_row_id_col(self):
        """Create a unique row identifier if not provided."""
        if not self.row_id_col:
            self.data["row_id"] = list(range(len(self.data)))
            self.row_id_col = "row_id"

    def _add_datasplit_group(self):
        """Add datasplit_group column for data splitting.

        Groups rows by sample_id_col OR identical features using transitive closure
        (if row A and B share sample_id_col, and B and C share features, then A, B,
        and C are all in the same group).

        Note:
            This must be called at the end of preparation, after all feature
            transformations are complete, to ensure accurate grouping.
        """
        # Union-Find with path compression
        parent = list(range(len(self.data)))

        def find(x):
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        def union(x, y):
            root_x, root_y = find(x), find(y)
            if root_x != root_y:
                parent[root_x] = root_y

        # Union rows sharing the same sample_id_col
        for indices in self.data.groupby(self.sample_id_col, dropna=False).indices.values():
            indices_list = list(indices)
            for i in range(1, len(indices_list)):
                union(indices_list[0], indices_list[i])

        # Union rows sharing identical feature values
        for indices in self.data.groupby(self.feature_cols, dropna=False, observed=True).indices.values():
            indices_list = list(indices)
            for i in range(1, len(indices_list)):
                union(indices_list[0], indices_list[i])

        # Assign sequential group IDs (0, 1, 2, ...) from arbitrary root values
        roots = np.array([find(i) for i in range(len(self.data))])
        self.data = self.data.assign(**{DATASPLIT_COL: pd.factorize(roots)[0]}).reset_index(drop=True)

    def _standardize_null_values(self):
        """Standardize null values to np.nan in pipeline columns.

        Converts common string representations of null values (case-insensitive)
        to np.nan for consistent handling. Recognized values include: 'none',
        'null', 'nan', 'na', '', and similar variants.

        Only applies to explicitly used pipeline columns (features, target, row_id,
        stratification, duration, event). Excludes sample_id_col to prevent
        merging unrelated samples during datasplit grouping, and excludes any
        columns not used by the pipeline.
        """
        null_values_case_insensitive = {val.lower() for val in DEFAULT_NULL_VALUES}

        def replace_null(x):
            if isinstance(x, str) and x.strip().lower() in null_values_case_insensitive:
                return np.nan
            return x

        cols = [c for c in self._columns_to_standardize if c in self.data.columns]
        self.data[cols] = self.data[cols].map(replace_null)

    def _standardize_inf_values(self):
        """Standardize infinity values to np.inf in pipeline columns.

        Converts common string representations of infinity (case-insensitive)
        to np.inf or -np.inf for consistent numerical handling. Recognized values
        include: 'inf', 'infinity', '∞', and their negative variants.

        Only applies to explicitly used pipeline columns (features, target, row_id,
        stratification, duration, event). Excludes sample_id_col and any columns
        not used by the pipeline.
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

        cols = [c for c in self._columns_to_standardize if c in self.data.columns]
        self.data[cols] = self.data[cols].map(replace_infinity)
