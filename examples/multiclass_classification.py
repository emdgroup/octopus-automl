"""Multiclass Workflow script for Octopus using Wine dataset."""

# # Multiclass Classification
#
# Most classification examples deal with two classes (binary). But many real-world
# problems have three or more categories — for example, predicting a disease subtype,
# a product category, or in this case, the type of wine.
#
# **What this example covers:**
#
# - Running a multiclass classification study with `OctoClassification`
# - How Octopus automatically detects multiclass problems
# - Choosing an appropriate multiclass metric
# - Using multiple model types in a single study
#
# **How multiclass differs from binary classification in Octopus:**
#
# - You still use `OctoClassification` — Octopus detects that there are more than two
#   classes in the target column and switches to multiclass mode automatically.
# - Use multiclass-aware metrics like `AUCROC_MACRO` (averages AUCROC across all classes)
#   instead of binary metrics like `AUCROC`.
# - Internally, models and evaluation adapt to the multiclass setting (e.g., one-vs-rest
#   for ROC curves, macro averaging for balanced metrics).
# - The Wine dataset used here has 3 classes and only 13 features — a small but
#   well-separated problem.

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

### Create and Run the Study
#
# We try four different tree-based models in a single Octo task. Optuna will explore
# hyperparameters for each of them and select the best-performing configuration.
#
# Using `single_outer_split=0` runs only the first outer fold for faster iteration.
study = OctoClassification(
    study_name="multiclass_wine",
    studies_directory=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="AUCROC_MACRO",
    feature_cols=features,
    target_col="target",
    sample_id_col="index",
    stratification_col="target",
    outer_split_seed=1234,
    single_outer_split=0,  # only process first outer split, for quick testing
    workflow=[
        Octo(
            task_id=0,
            depends_on=None,
            description="step_1_octo_multiclass",
            n_inner_splits=5,
            models=[
                ModelName.ExtraTreesClassifier,
                ModelName.RandomForestClassifier,
                ModelName.XGBClassifier,
                ModelName.CatBoostClassifier,
            ],
            max_outliers=0,
            fi_methods=[FIComputeMethod.PERMUTATION],
            n_trials=20,
        ),
    ],
)

study.fit(data=df)

print("Workflow completed successfully!")
print(f"Results saved to: {study.output_path}")

# ## Learn more
#
# - [Classification User Guide](../userguide/classification.md#multiclass-classification) — multiclass-specific metrics and model support.
# - [Nested Cross-Validation](../concepts/nested_cv.md) — how Octopus evaluates models on small datasets.
