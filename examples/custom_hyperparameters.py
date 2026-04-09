"""Example for using custom hyperparameters in Octopus regression."""

# # Custom Hyperparameters
#
# By default, Octopus uses Optuna to search over a built-in hyperparameter space for
# each model. This works well in most cases, but sometimes you want more control —
# for example, to narrow the search range based on domain knowledge, or to fix a
# parameter to a known good value.
#
# **What this example covers:**
#
# - Defining custom hyperparameter ranges using `IntHyperparameter`
# - Passing them to the `Octo` module via the `hyperparameters` dictionary
# - Running a regression study with a constrained search space
#
# **Available hyperparameter types:**
#
# - `IntHyperparameter(name, low, high)` — integer range (e.g., tree depth)
# - `FloatHyperparameter(name, low, high)` — float range (e.g., learning rate)
# - `CategoricalHyperparameter(name, choices)` — discrete options (e.g., split criterion)
# - `FixedHyperparameter(name, value)` — lock a parameter to a constant
#
# The `name` must match an actual hyperparameter of the underlying scikit-learn (or
# XGBoost/CatBoost) model. Octopus passes these directly to Optuna's `suggest_*` methods.

### Imports
import os

from octopus.example_data import load_diabetes_data
from octopus.models.hyperparameter import IntHyperparameter
from octopus.modules import Octo
from octopus.study import OctoRegression
from octopus.types import ModelName

### Load the Diabetes Dataset
df, features, targets = load_diabetes_data()

print("Dataset info:")
print(f"  Features: {len(features)} - {features}")
print(f"  Samples: {df.shape[0]}")
print(f"  Classes: {len(targets)} - {targets}")
print(f"  Target distribution: {df['target'].value_counts().sort_index().to_dict()}")

### Create and Run the Study
#
# Here we restrict `RandomForestRegressor` to only explore `max_depth` between 2 and 32
# and `min_samples_split` between 2 and 100. All other hyperparameters keep their
# Optuna defaults.
#
# Note: `single_outer_split=0` runs only the first outer fold — useful for quick
# iteration while tuning your configuration.
study = OctoRegression(
    study_name="use_own_hyperparameters_example",
    studies_directory=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="MAE",
    feature_cols=features,
    target_col="target",
    sample_id_col="index",
    single_outer_split=0,
    workflow=[
        Octo(
            task_id=0,
            models=[ModelName.RandomForestRegressor],
            n_trials=3,
            hyperparameters={
                ModelName.RandomForestRegressor: [
                    IntHyperparameter(name="max_depth", low=2, high=32),
                    IntHyperparameter(name="min_samples_split", low=2, high=100),
                ]
            },
        ),
    ],
)

study.fit(data=df)

print("Workflow completed")
