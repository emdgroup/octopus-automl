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
from octopus.types import FIComputeMethod, ModelName

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
    datasplit_seed_outer=1234,
    ignore_data_health_warning=True,
    run_single_outersplit_num=0,  # only process first outersplit, for quick testing
    workflow=[
        Octo(
            task_id=0,
            depends_on=None,
            description="step_1_octo_multiclass",
            n_folds_inner=5,
            models=[
                ModelName.ExtraTreesClassifier,
                ModelName.RandomForestClassifier,
                ModelName.XGBClassifier,
                ModelName.CatBoostClassifier,
            ],
            model_seed=0,
            max_outl=0,
            fi_methods_bestbag=[FIComputeMethod.PERMUTATION],
            n_trials=20,
        ),
    ],
)

study.fit(data=df)

print("Workflow completed successfully!")
print(f"Results saved to: {study.output_path}")
