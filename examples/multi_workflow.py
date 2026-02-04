"""Example for using a multi workflow."""

# This example demonstrates how to create a workflow
# using Octopus with the diabetes dataset.

### Necessary imports for this example
import os

from sklearn.datasets import load_diabetes

from octopus import OctoRegression
from octopus.modules import Mrmr, Octo

### Load the diabetes dataset
diabetes = load_diabetes(as_frame=True)

### Create and run OctoRegression with multi-step workflow
study = OctoRegression(
    name="example_multiworkflow",
    path=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="R2",
    feature_cols=diabetes["feature_names"],
    target="target",
    sample_id="index",
    ignore_data_health_warning=True,
    outer_parallelization=False,
    run_single_experiment_num=1,
    workflow=[
        Octo(
            description="step1_octofull",
            task_id=0,
            depends_on_task=-1,
            models=["ExtraTreesRegressor", "RandomForestRegressor"],
            n_trials=2,
            max_features=70,
        ),
        Mrmr(
            description="step2_mrmr",
            task_id=1,
            depends_on_task=0,
            n_features=6,
            correlation_type="rdc",
        ),
        Octo(
            description="step3_octo_reduced",
            task_id=2,
            depends_on_task=1,
            models=["ExtraTreesRegressor", "RandomForestRegressor"],
            n_trials=1,
            max_features=70,
        ),
    ],
)

study.fit(data=diabetes["frame"].reset_index())

print("Multi-workflow completed")
