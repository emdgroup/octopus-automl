"""Container for prepared study data and metadata."""

import pandas as pd
from attrs import define


@define
class PreparedData:
    """Container for prepared study data and metadata after data preparation."""

    data: pd.DataFrame
    """Prepared DataFrame after all transformations."""

    feature_cols: list[str]
    """Feature columns actually used after preparation (sorted, single-value features removed)."""

    row_id_col: str
    """Row ID column used (auto-generated if not provided by user)."""

    target_assignments: dict[str, str]
    """Target assignments with defaults applied."""

    @property
    def num_features(self) -> list[str]:
        """Get numerical feature columns from effective features."""
        return [
            col
            for col in self.feature_cols
            if col in self.data.columns
            and pd.api.types.is_numeric_dtype(self.data[col])
            and not isinstance(self.data[col].dtype, pd.CategoricalDtype)
        ]

    @property
    def cat_nominal_features(self) -> list[str]:
        """Get categorical nominal feature columns from effective features."""
        cat_nominal_features = []
        for col in self.feature_cols:
            if col in self.data.columns:
                dtype = self.data[col].dtype
                if isinstance(dtype, pd.CategoricalDtype) and not dtype.ordered:
                    cat_nominal_features.append(col)
        return cat_nominal_features

    @property
    def cat_ordinal_features(self) -> list[str]:
        """Get categorical ordinal feature columns from effective features."""
        cat_ordinal_features = []
        for col in self.feature_cols:
            if col in self.data.columns:
                dtype = self.data[col].dtype
                if isinstance(dtype, pd.CategoricalDtype) and dtype.ordered:
                    cat_ordinal_features.append(col)
        return cat_ordinal_features
