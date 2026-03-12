"""Multiclass Workflow script for Octopus using Wine dataset."""

# Multiclass classification example using Octopus

# This example demonstrates how to use Octopus to create a multiclass classification model.
# We will use the Wine dataset from sklearn for this purpose.
# The Wine dataset contains 3 classes (wine types) with 13 features.
# Please ensure your dataset is clean, with no missing values (`NaN`),
# and that all features are numeric.

import os

from octopus.example_data import load_wine_data
from octopus.modules import Octo
from octopus.study import OctoClassification

### Load and Preprocess Data
df, features, targets = load_wine_data()

print("Dataset info:")
print(f"  Features: {len(features)} - {features}")
print(f"  Samples: {df.shape[0]}")
print(f"  Classes: {len(targets)} - {targets}")
print(f"  Target distribution: {df['target'].value_counts().sort_index().to_dict()}")

### Create and run OctoClassification for multiclass classification
# OctoClassification automatically detects multiclass (>2 classes) from the data
study = OctoClassification(
    name="multiclass_wine",
    path=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="AUCROC_MACRO",
    feature_cols=features,
    target_col="target",
    sample_id_col="index",
    stratification_col="target",
    metrics=["AUCROC_MACRO", "AUCROC_WEIGHTED", "ACCBAL_MC"],
    datasplit_seed_outer=1234,
    ignore_data_health_warning=True,
    outer_parallelization=True,
    run_single_outersplit_num=0,  # only process first outersplit, for quick testing
    workflow=[
        Octo(
            task_id=0,
            depends_on=None,
            description="step_1_octo_multiclass",
            n_folds_inner=5,
            models=[
                "ExtraTreesClassifier",
                "RandomForestClassifier",
                "XGBClassifier",
                "CatBoostClassifier",
            ],
            model_seed=0,
            n_jobs=1,
            max_outl=0,
            fi_methods_bestbag=["permutation"],
            inner_parallelization=True,
            n_workers=5,
            n_trials=20,
        ),
    ],
)

study.fit(data=df)

print("Workflow completed successfully!")
print(f"Results saved to: {study.output_path}")
