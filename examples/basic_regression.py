"""Basic example for using Octopus regression."""

# This example demonstrates how to use Octopus to create a machine learning regression model.
# We will use the famous diabetes dataset for this purpose.
# Please ensure your dataset is clean, with no missing values (`NaN`),
# and that all features are numeric.

### Necessary imports for this example
import os

from sklearn.datasets import load_diabetes

from octopus import OctoRegression

### Load the diabetes dataset
diabetes = load_diabetes(as_frame=True)

### Create and run OctoRegression
study = OctoRegression(
    name="basic_regression",
    path=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="MAE",
    feature_cols=diabetes["feature_names"],
    target="target",
    sample_id_col="index",
)

study.fit(data=diabetes["frame"].reset_index())

print("Workflow completed")
