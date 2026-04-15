"""Example for using a multi workflow."""

# This example demonstrates how to create a workflow
# using Octopus with the diabetes dataset.

### Necessary imports for this example
import os

from octopus.example_data import load_diabetes_data
from octopus.modules import Mrmr, Tako
from octopus.study import OctoRegression
from octopus.types import CorrelationType, ModelName

### Load the diabetes dataset
df, features, targets = load_diabetes_data()

print("Dataset info:")
print(f"  Features: {len(features)} - {features}")
print(f"  Samples: {df.shape[0]}")
print(f"  Classes: {len(targets)} - {targets}")
print(f"  Target distribution: {df['target'].value_counts().sort_index().to_dict()}")

### Create and run OctoRegression with multi-step workflow
study = OctoRegression(
    study_name="wf_multi_step_regression",
    studies_directory=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="R2",
    feature_cols=features,
    target_col="target",
    sample_id_col="index",
    single_outer_split=1,
    workflow=[
        Tako(
            description="step1_takofull",
            task_id=0,
            depends_on=None,
            models=[ModelName.ExtraTreesRegressor, ModelName.RandomForestRegressor],
            n_trials=2,
            max_features=70,
        ),
        Mrmr(
            description="step2_mrmr",
            task_id=1,
            depends_on=0,
            n_features=6,
            correlation_type=CorrelationType.RDC,
        ),
        Tako(
            description="step3_tako_reduced",
            task_id=2,
            depends_on=1,
            models=[ModelName.ExtraTreesRegressor, ModelName.RandomForestRegressor],
            n_trials=1,
            max_features=70,
        ),
    ],
)

study.fit(data=df)

print("Multi-workflow completed")
