"""Basic example for using Octopus Classification."""

# This example demonstrates how to use Octopus to create a machine learning classification model.
# We will use the breast cancer dataset for this purpose.
# Please ensure your dataset is clean, with no missing values (`NaN`),
# and that all features are numeric.

### Necessary imports for this example
import os

from octopus.example_data import load_breast_cancer_data
from octopus.modules import Tako
from octopus.study import OctoClassification
from octopus.types import ModelName

### Load and Preprocess Data
df, features, targets = load_breast_cancer_data()

print("Dataset info:")
print(f"  Features: {len(features)} - {features}")
print(f"  Samples: {df.shape[0]}")
print(f"  Classes: {len(targets)} - {targets}")
print(f"  Target distribution: {df['target'].value_counts().sort_index().to_dict()}")

### Create and run OctoClassification
study = OctoClassification(
    study_name="basic_classification",
    studies_directory=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="AUCROC",
    feature_cols=features,
    target_col="target",
    sample_id_col="index",
    stratification_col="target",
    workflow=[
        Tako(
            description="step1_tako_full",
            task_id=0,
            depends_on=None,  # First task, depends on input
            models=[ModelName.ExtraTreesClassifier],
            n_trials=100,  # 100 trials for hyperparameter optimization
            n_inner_splits=5,  # 5 inner splits
            max_features=30,  # Use all 30 features
            ensemble_selection=True,  # Enable ensemble selection
        ),
    ],
)

study.fit(data=df)

print("Workflow completed")
