"""Multiclass Workflow script for Octopus using Wine dataset."""

# Multiclass classification example using Octopus

# This example demonstrates how to use Octopus to create a multiclass classification model.
# We will use the Wine dataset from sklearn for this purpose.
# The Wine dataset contains 3 classes (wine types) with 13 features.
# Please ensure your dataset is clean, with no missing values (`NaN`),
# and that all features are numeric.

import os

from sklearn.datasets import load_wine

from octopus import OctoClassification
from octopus.modules import Octo

### Load and Preprocess Data
wine = load_wine(as_frame=True)

df = wine["frame"].reset_index()
df.columns = df.columns.str.replace(" ", "_")
features = list(wine["feature_names"])
features = [feature.replace(" ", "_") for feature in features]

print("Dataset info:")
print(f"  Features: {len(features)}")
print(f"  Samples: {df.shape[0]}")
print(f"  Classes: {len(wine.target_names)} - {wine.target_names}")
print(f"  Target distribution: {df['target'].value_counts().sort_index().to_dict()}")

### Create and run OctoClassification for multiclass classification
# OctoClassification automatically detects multiclass (>2 classes) from the data
study = OctoClassification(
    name="multiclass_wine",
    path=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="AUCROC_MACRO",
    feature_columns=features,
    target="target",
    sample_id="index",
    stratification_column="target",
    metrics=["AUCROC_MACRO", "AUCROC_WEIGHTED", "ACCBAL_MC"],
    datasplit_seed_outer=1234,
    ignore_data_health_warning=True,
    outer_parallelization=True,
    run_single_experiment_num=0,  # only process first outer loop experiment, for quick testing
    workflow=[
        Octo(
            task_id=0,
            depends_on_task=-1,
            description="step_1_octo_multiclass",
            load_task=False,
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
