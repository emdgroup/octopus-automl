"""Basic example for using Octopus Regression.

This example demonstrates how to use Octopus to create a machine learning
regression model with a two-step workflow: prediction followed by feature
selection and a second prediction on the reduced feature set.

We use the diabetes dataset. Please ensure your dataset is clean, with
no missing values (NaN), and that all features are numeric.

Run this script before using the analysis_regression.ipynb notebook::

    python examples/basic_regression.py
"""

import os

from octopus.example_data import load_diabetes_data
from octopus.modules import Mrmr, Octo
from octopus.study import OctoRegression
from octopus.types import ModelName

### Load the diabetes dataset
df, features, targets = load_diabetes_data()

print("Dataset info:")
print(f"  Features: {len(features)} - {features}")
print(f"  Samples: {df.shape[0]}")

### Create and run OctoRegression
study = OctoRegression(
    study_name="basic_regression",
    studies_directory=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="MAE",
    feature_cols=features,
    target_col="target",
    sample_id_col="index",
    workflow=[
        Octo(
            description="step1_octo_full",
            task_id=0,
            depends_on=None,
            models=[ModelName.ExtraTreesRegressor],
            n_trials=100,
            n_inner_splits=5,
            ensemble_selection=True,
        ),
        Mrmr(
            description="step2_mrmr",
            task_id=1,
            depends_on=0,
            n_features=5,
        ),
        Octo(
            description="step3_octo_reduced",
            task_id=2,
            depends_on=1,
            models=[ModelName.ExtraTreesRegressor],
            n_trials=100,
            n_inner_splits=5,
            ensemble_selection=True,
        ),
    ],
)

study.fit(data=df)

print("Done! Study saved to:", study.output_path)
