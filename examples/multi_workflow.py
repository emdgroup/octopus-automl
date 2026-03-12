"""Example for using a multi workflow."""

# This example demonstrates how to create a workflow
# using Octopus with the diabetes dataset.

### Necessary imports for this example
import os

from sklearn.datasets import load_diabetes
from sklearn.utils import Bunch

from octopus.modules import Mrmr, Octo
from octopus.study import OctoRegression

### Load the diabetes dataset
diabetes: Bunch = load_diabetes(as_frame=True)  # type: ignore[assignment]

### Create and run OctoRegression with multi-step workflow
study = OctoRegression(
    name="example_multiworkflow",
    path=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="R2",
    feature_cols=diabetes["feature_names"],
    target_col="target",
    sample_id_col="index",
    ignore_data_health_warning=True,
    outer_parallelization=False,
    run_single_outersplit_num=1,
    workflow=[
        Octo(
            description="step1_octofull",
            task_id=0,
            depends_on=None,
            models=["ExtraTreesRegressor", "RandomForestRegressor"],
            n_trials=2,
            max_features=70,
        ),
        Mrmr(
            description="step2_mrmr",
            task_id=1,
            depends_on=0,
            n_features=6,
            correlation_type="rdc",
        ),
        Octo(
            description="step3_octo_reduced",
            task_id=2,
            depends_on=1,
            models=["ExtraTreesRegressor", "RandomForestRegressor"],
            n_trials=1,
            max_features=70,
        ),
    ],
)

study.fit(data=diabetes["frame"].reset_index())

print("Multi-workflow completed")
