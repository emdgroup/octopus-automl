"""Example for using a multi workflow."""

# # Sequential Regression Workflow
#
# Feature selection often works best when done in stages. A common pattern is:
#
# 1. Start with a broad model to get initial feature importances.
# 2. Use a dedicated feature selection method to narrow the set further.
# 3. Retrain a final model on the reduced features.
#
# This example builds exactly that pipeline for a **regression** task using the
# diabetes dataset.
#
# **What this example covers:**
#
# - Chaining three tasks with `depends_on`
# - Using MRMR (Max-Relevance Min-Redundancy) for feature selection
# - Combining feature selection and model training in one workflow
#
# **How the workflow connects:**
#
# ```
# Task 0 (Octo)   → all 10 features → trains models, selects features
#       ↓
# Task 1 (MRMR)   → receives Task 0's features → selects top 6
#       ↓
# Task 2 (Octo)   → receives Task 1's 6 features → trains final models
# ```
#
# Each task sets `depends_on` to the `task_id` of the task it should receive
# features from. Task 0 has `depends_on=None` because it's the entry point.

### Imports
import os

from octopus.example_data import load_diabetes_data
from octopus.modules import Mrmr, Octo
from octopus.study import OctoRegression
from octopus.types import CorrelationType, ModelName

### Load the Diabetes Dataset
df, features, targets = load_diabetes_data()

print("Dataset info:")
print(f"  Features: {len(features)} - {features}")
print(f"  Samples: {df.shape[0]}")
print(f"  Classes: {len(targets)} - {targets}")
print(f"  Target distribution: {df['target'].value_counts().sort_index().to_dict()}")

### Create and Run the Study
#
# MRMR selects features by balancing two goals: each feature should be highly
# relevant to the target (max-relevance) and minimally correlated with the
# features already selected (min-redundancy). Here we ask it to keep the top 6
# features using RDC (Randomized Dependence Coefficient) as the correlation measure.
study = OctoRegression(
    study_name="example_multiworkflow",
    studies_directory=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="R2",
    feature_cols=features,
    target_col="target",
    sample_id_col="index",
    single_outer_split=1,
    workflow=[
        Octo(
            description="step1_octofull",
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
        Octo(
            description="step3_octo_reduced",
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

# ## Learn more
#
# - [Regression User Guide](../userguide/regression.md) — regression setup, models, and metrics.
# - [MRMR Module](../concepts/workflow/mrmr.md) — Maximum Relevance Minimum Redundancy feature selection.
# - [Workflow & Modules](../concepts/workflow/index.md) — how multi-step pipelines work.
