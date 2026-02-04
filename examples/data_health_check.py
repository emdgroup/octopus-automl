"""Data health check example."""

import os
import random
from decimal import Decimal

import numpy as np
import pandas as pd
from attrs import define, field
from sklearn.datasets import make_classification

from octopus import OctoClassification
from octopus.modules import Octo


@define
class DataFrameGenerator:
    """A class to generate an example DataFrame."""

    n_samples: int = 1000
    n_features: int = 20
    n_informative: int = 10
    n_redundant: int = 10
    n_classes: int = 3
    random_state: int = None

    df: pd.DataFrame = field(init=False)

    def __attrs_post_init__(self):
        self._generate_data()

    def _generate_data(self):
        """Generate the classification dataset and initialize the DataFrame."""
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_redundant=self.n_redundant,
            n_classes=self.n_classes,
            n_clusters_per_class=2,
            class_sep=5.0,
            random_state=self.random_state,
        )

        # Create DataFrame from features
        feature_cols = [f"feature_{i + 1}" for i in range(self.n_features)]
        self.df = pd.DataFrame(X, columns=feature_cols)

        # Add the target column
        self.df["target"] = y

    def add_nan_to_features(self, min_frac=0.02, max_frac=0.8):
        """Add a random proportion of NaNs to the first half of the feature columns."""
        half_features = self.df.columns[: self.n_features // 2]
        rng = np.random.default_rng(self.random_state)
        num_rows = len(self.df)

        for feature in half_features:
            # Determine the number of NaNs to introduce based on the random fraction
            nan_fraction = rng.uniform(min_frac, max_frac)
            num_nan = int(nan_fraction * num_rows)

            # Select random indices to set as NaN
            nan_indices = rng.choice(self.df.index, size=num_nan, replace=False)
            self.df.loc[nan_indices, feature] = np.nan

    def add_nan_to_target(self, num_nan=10):
        """Add NaN values to the target column."""
        rng = np.random.default_rng(self.random_state)
        nan_indices = rng.choice(self.df.index, size=num_nan, replace=False)
        self.df.loc[nan_indices, "target"] = np.nan

    def add_id_column(
        self,
        column_name="id",
        prefix="ID_",
        unique=True,
        duplicate_factor=2,
        include_nans=False,
        nan_ratio=0.1,
    ):
        """Add an ID column with unique or non-unique identifiers."""
        if prefix is None:
            # Use integers for IDs
            ids = np.arange(len(self.df), dtype="uint" if unique else "int")
            if not unique:
                ids = np.repeat(ids, duplicate_factor)[: len(self.df)]
        elif unique:
            # Create unique IDs with prefix
            ids = [prefix + str(i) for i in self.df.index]
        else:
            # Create non-unique IDs with prefix
            ids = [prefix + str(i) for i in range(len(self.df) // duplicate_factor)]
            non_unique_ids = ids * duplicate_factor
            ids = non_unique_ids[: len(self.df)]

        if include_nans:
            # Determine number of NaNs to include
            num_nans = int(len(self.df) * nan_ratio)
            nan_indices = np.random.choice(len(self.df), num_nans, replace=False)
            ids = np.array(ids, dtype=object)  # Convert to a mutable array
            ids[nan_indices] = np.nan

        self.df[column_name] = ids

    def add_constant_column(self, column_name="one", value=1):
        """Add a constant column to the DataFrame."""
        self.df[column_name] = value

    def add_decimal_columns(self, column_names: list[str] | None = None, precision=8):
        """Add columns with Decimal data type."""
        if column_names is None:
            column_names = ["decimal_1", "decimal_2"]

        rng = np.random.default_rng(self.random_state)
        for col_name in column_names:
            random_numbers = rng.random(size=len(self.df))
            formatted_numbers = [Decimal(f"{num:.{precision}f}") for num in random_numbers]
            self.df[col_name] = formatted_numbers

    def add_inf_columns(self, column_names: list[str] | None = None, num_inf=10):
        """Add columns with infinite values."""
        if column_names is None:
            column_names = ["inf_col"]

        rng = np.random.default_rng(self.random_state)
        for col_name in column_names:
            # Initialize the column with random float values
            self.df[col_name] = rng.standard_normal(size=len(self.df))
            # Introduce inf values
            inf_indices = rng.choice(self.df.index, size=num_inf, replace=False)
            self.df.loc[inf_indices, col_name] = np.inf

    def add_fixed_unique_values_column(self, column_name="few_unique", num_unique=3):
        """Add a column with a specified number of unique values."""
        # Create a list of unique values
        unique_values = list(range(num_unique))
        # Repeat these values to fill the column
        repeated_values = unique_values * (len(self.df) // num_unique + 1)
        # Assign to the DataFrame, trimming to the correct length
        self.df[column_name] = repeated_values[: len(self.df)]

    def add_string_mismatch_column(self, column_name="mismatch_col", base_string="sample", error_rate=0.1):
        """Add a column with strings that contain random typos or mismatches, and convert it to categorical."""

        def introduce_typo(s):
            if random.random() < error_rate:
                # Introduce a typo by swapping two adjacent characters
                idx = random.randint(0, len(s) - 2)
                return s[:idx] + s[idx + 1] + s[idx] + s[idx + 2 :]
            return s

        # Generate the column with potential typos
        self.df[column_name] = [introduce_typo(base_string) for _ in range(len(self.df))]

        # Convert the column to categorical type
        self.df[column_name] = self.df[column_name].astype("category")

    def get_dataframe(self):
        """Return the generated DataFrame.

        Returns:
        - pd.DataFrame: The generated DataFrame.
        """
        return self.df.copy()


# Error example
generator_errors = DataFrameGenerator(random_state=42)
generator_errors.add_nan_to_features()
generator_errors.add_nan_to_target(num_nan=10)
generator_errors.add_id_column(unique=True, include_nans=True)
generator_errors.add_id_column(column_name="sample_id", prefix="Sample", unique=True, include_nans=True)
generator_errors.add_id_column(
    column_name="stratification",
    prefix="Strat_",
    unique=True,
    include_nans=False,
)
generator_errors.add_constant_column()
# generator_errors.add_decimal_columns()
generator_errors.add_inf_columns()

df_error = generator_errors.get_dataframe()

# warning example
generator_warnings = DataFrameGenerator(random_state=42, n_classes=2)
generator_warnings.add_fixed_unique_values_column()
generator_warnings.add_id_column(unique=True, include_nans=False)
generator_warnings.add_id_column(column_name="sample_id", prefix="Sample", unique=True, include_nans=False)
generator_warnings.add_id_column(
    column_name="stratification",
    prefix=None,
    unique=True,
    include_nans=False,
)
generator_warnings.add_string_mismatch_column()

df_warnings = generator_warnings.get_dataframe()

print(df_warnings)

### Create and run OctoClassification with health check

study = OctoClassification(
    name="health_check",
    path=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="AUCROC",
    feature_cols=df_warnings.columns.drop("target").drop("id").drop("sample_id").drop("stratification").tolist(),
    target="target",
    sample_id="sample_id",
    datasplit_type="group_sample_and_features",
    stratification_column="target",
    ignore_data_health_warning=False,  # Will stop if health check finds issues
    outer_parallelization=True,
    workflow=[
        Octo(
            task_id=0,
            depends_on_task=-1,
            description="step_1_octo",
            models=["RandomForestClassifier"],
            n_trials=3,
        )
    ],
)

study.fit(data=df_warnings)
