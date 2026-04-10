"""Create a small study with 5 tasks (3 predict + 2 feature selection) for testing analysis functions.

Workflow:
    [task 0] Octo  "step1_octo_full"
    └── [task 1] Roc   "step2_roc"
        └── [task 2] Octo  "step3_octo_reduced"
            └── [task 3] Mrmr  "step4_mrmr"
                └── [task 4] Octo  "step5_octo_final"
"""

import os

from octopus.example_data import load_diabetes_data
from octopus.modules import Mrmr, Octo, Roc
from octopus.study import OctoRegression
from octopus.types import ModelName

df, features, targets = load_diabetes_data()

print("Dataset info:")
print(f"  Features: {len(features)} - {features}")
print(f"  Samples: {df.shape[0]}")

study = OctoRegression(
    study_name="analysis_example",
    studies_directory=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="MAE",
    feature_cols=features,
    target_col="target",
    sample_id_col="index",
    n_outer_splits=3,
    workflow=[
        Octo(
            description="step1_octo_full",
            task_id=0,
            depends_on=None,
            models=[ModelName.ExtraTreesRegressor],
            n_trials=3,
            n_inner_splits=3,
        ),
        Roc(
            description="step2_roc",
            task_id=1,
            depends_on=0,
        ),
        Octo(
            description="step3_octo_reduced",
            task_id=2,
            depends_on=1,
            models=[ModelName.ExtraTreesRegressor],
            n_trials=3,
            n_inner_splits=3,
        ),
        Mrmr(
            description="step4_mrmr",
            task_id=3,
            depends_on=0,
            n_features=4,
        ),
        Octo(
            description="step5_octo_final",
            task_id=4,
            depends_on=3,
            models=[ModelName.ExtraTreesRegressor],
            n_trials=3,
            n_inner_splits=3,
        ),
    ],
)

study.fit(data=df)

print("Done! Study saved to:", study.output_path)
