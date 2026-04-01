"""Basic example for using Octopus regression."""

# This example demonstrates how to use Octopus to create a machine learning regression model.
# We will use the famous diabetes dataset for this purpose.
# Please ensure your dataset is clean, with no missing values (`NaN`),
# and that all features are numeric.

### Necessary imports for this example
import os

from octopus.example_data import load_diabetes_data
from octopus.study import OctoRegression

### Load the diabetes dataset
df, features, targets = load_diabetes_data()

print("Dataset info:")
print(f"  Features: {len(features)} - {features}")
print(f"  Samples: {df.shape[0]}")
print(f"  Classes: {len(targets)} - {targets}")
print(f"  Target distribution: {df['target'].value_counts().sort_index().to_dict()}")

### Create and run OctoRegression
study = OctoRegression(
    study_name="basic_regression",
    study_path=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="MAE",
    feature_cols=features,
    target_col="target",
    sample_id_col="index",
)

study.fit(data=df)

print("Workflow completed")
