"""Example for using custom hyperparameters in Octopus regression."""

# This example demonstrates how to use custom hyperparameters with Octopus.
# The key difference from the basic example is the use of the `hyperparameters` parameter
# in the Octo configuration, where you can define custom hyperparameter ranges
# for each model using the Hyperparameter class.

# Instead of letting Optuna automatically search the hyperparameter space,
# you can define your own hyperparameter ranges for the models.
# We will use the diabetes dataset for this purpose.

### Necessary imports for this example
import os

from octopus.example_data import load_diabetes_data
from octopus.models.hyperparameter import IntHyperparameter
from octopus.modules import Octo
from octopus.study import OctoRegression

### Load the diabetes dataset
df, features, targets = load_diabetes_data()

print("Dataset info:")
print(f"  Features: {len(features)} - {features}")
print(f"  Samples: {df.shape[0]}")
print(f"  Classes: {len(targets)} - {targets}")
print(f"  Target distribution: {df['target'].value_counts().sort_index().to_dict()}")

### Create and run OctoRegression with custom hyperparameters
study = OctoRegression(
    name="use_own_hyperparameters_example",
    path=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="MAE",
    feature_cols=features,
    target_col="target",
    sample_id_col="index",
    ignore_data_health_warning=True,
    outer_parallelization=False,
    run_single_outersplit_num=0,
    workflow=[
        Octo(
            task_id=0,
            models=["RandomForestRegressor"],
            n_trials=3,
            hyperparameters={
                "RandomForestRegressor": [
                    IntHyperparameter(name="max_depth", low=2, high=32),
                    IntHyperparameter(name="min_samples_split", low=2, high=100),
                ]
            },
        ),
    ],
)

study.fit(data=df)

print("Workflow completed")
