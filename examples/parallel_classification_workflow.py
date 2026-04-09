"""Example: Parallel Octo and AutoGluon workflow for binary classification."""

# # Parallel Classification Workflow
#
# So far, all workflow examples have been *sequential* — each task feeds into the next.
# But Octopus also supports **parallel** execution: multiple tasks that independently
# process the same input data. This is useful when you want to compare different
# approaches side by side, or combine specialized modules on the same feature set.
#
# In this example, we run **Octo** and **AutoGluon** in parallel on a synthetic
# high-dimensional dataset (1000 features, only 60 informative). Both modules
# receive the full feature set and work independently.
#
# **What this example covers:**
#
# - Running two modules in parallel using `depends_on=None` on both
# - Using AutoGluon alongside Octo in the same study
# - Working with high-dimensional synthetic data
#
# **How parallel execution works:**
#
# When two or more tasks set `depends_on=None`, they all receive the original
# feature set and run independently. Compare this to sequential workflows where
# `depends_on=<task_id>` chains tasks together. You can also mix both patterns —
# for example, two parallel branches that each feed into a downstream task.

import os

import pandas as pd
from sklearn.datasets import make_classification

from octopus.modules import AutoGluon, Octo
from octopus.study import OctoClassification
from octopus.types import FIComputeMethod, ModelName

### Generate Synthetic Dataset
#
# We create a challenging binary classification problem with 1000 features but only
# 30 informative and 30 redundant ones. The remaining 940 features are pure noise.
# This simulates a real-world scenario where most measured variables are irrelevant.
n_informative = 30
n_redundant = 30
n_repeated = 0

X, y = make_classification(
    n_samples=300,
    n_features=1000,
    n_informative=n_informative,
    n_redundant=n_redundant,  # generated as random linear combinations of the informative features
    n_repeated=n_repeated,  # drawn randomly from the informative and the redundant features.
    n_classes=2,
    class_sep=1.0,  # Controls class separability (higher = easier)
    weights=[0.5, 0.5],  # 60% class 0, 40% class 1
    flip_y=0.01,  # Add 1% label noise for realism
    random_state=42,
    shuffle=False,  # ensure order of features
)

# Create a pandas DataFrame with proper structure
# Without shuffling, features are ordered: informative, redundant, repeated, then noise
feature_names = []
# Informative features (first n_informative)
feature_names.extend([f"informative_{i}" for i in range(n_informative)])
# Redundant features (next n_redundant)
feature_names.extend([f"redundant_{i}" for i in range(n_redundant)])
# Repeated features (next n_repeated)
if n_repeated > 0:
    feature_names.extend([f"repeated_{i}" for i in range(n_repeated)])
# Remaining features are noise
n_noise = X.shape[1] - n_informative - n_redundant - n_repeated
feature_names.extend([f"noise_{i}" for i in range(n_noise)])

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
df = df.reset_index()

# Display dataset information
print("=== Synthetic Dataset Information ===")
print(f"Dataset shape: {df.shape}")
print(f"Features: {len(feature_names)}")
print(f"Class distribution:\n{df['target'].value_counts()}")
print(f"Class balance: {df['target'].value_counts(normalize=True).to_dict()}")
print("=====================================\n")

### Create and Run the Study
#
# Both tasks have `depends_on=None`, so they run in parallel on the full feature set.
# After the study completes, you can compare their results to see which approach
# worked better for this particular dataset.

study = OctoClassification(
    study_name="wf_octo_autogluon_parallel",
    studies_directory=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="AUCROC",  # Area Under ROC Curve for binary classification
    feature_cols=feature_names,
    target_col="target",
    sample_id_col="index",
    stratification_col="target",  # Ensure balanced splits
    n_outer_splits=5,  # 5-split outer cross-validation
    workflow=[
        # Task 0: Octo — Optuna-based hyperparameter optimization
        Octo(
            description="step_0_octo",
            task_id=0,
            depends_on=None,  # parallel: receives all features
            n_inner_splits=5,
            models=[
                ModelName.ExtraTreesClassifier,
            ],
            fi_methods=[FIComputeMethod.PERMUTATION],
            n_trials=100,
        ),
        # Task 1: AutoGluon — AutoML with its own model selection and ensembling
        AutoGluon(
            description="step_1_autogluon",
            task_id=1,
            depends_on=None,  # parallel: also receives all features
            time_limit=600,
            presets=["medium_quality"],
            n_bag_splits=5,
            included_model_types=[
                "XT",
            ],
        ),
    ],
)

# Run the study on the synthetic data
print("Starting Octo + AutoGluon workflow...")
study.fit(data=df)

print("Workflow completed successfully!")
print(f"Results saved to: {study.output_path}")
