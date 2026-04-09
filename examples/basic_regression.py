"""Basic example for using Octopus regression."""

# # Basic Regression
#
# This example shows how to use Octopus for a regression task. Instead of predicting
# a class label, we predict a continuous numeric value — in this case, a measure of
# disease progression from the well-known diabetes dataset.
#
# **What this example covers:**
#
# - Setting up an `OctoRegression` study
# - Choosing an appropriate regression metric (`MAE`)
# - Running Octopus with default settings for a quick start
#
# **How regression differs from classification in Octopus:**
#
# - Use `OctoRegression` instead of `OctoClassification`.
# - There is no `stratification_col` because the target is continuous, not categorical.
# - Metrics like `MAE` (Mean Absolute Error), `R2`, `RMSE`, or `MSE` replace
#   classification metrics like `AUCROC` or `ACCBAL`.
# - Octopus automatically selects regression-appropriate models when none are specified.

### Imports
import os

from octopus.example_data import load_diabetes_data
from octopus.study import OctoRegression

### Load the Diabetes Dataset
df, features, targets = load_diabetes_data()

print("Dataset info:")
print(f"  Features: {len(features)} - {features}")
print(f"  Samples: {df.shape[0]}")
print(f"  Classes: {len(targets)} - {targets}")
print(f"  Target distribution: {df['target'].value_counts().sort_index().to_dict()}")

### Create and Run the Study
#
# This is the simplest possible Octopus setup: we only specify the dataset columns,
# a target metric, and let Octopus handle everything else — model selection, hyperparameter
# optimization, and cross-validation are all done automatically with sensible defaults.
study = OctoRegression(
    study_name="basic_regression",
    studies_directory=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="MAE",
    feature_cols=features,
    target_col="target",
    sample_id_col="index",
)

study.fit(data=df)

print("Workflow completed")
