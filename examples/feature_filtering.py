"""Example: Use of ROC and Octo modules for breast cancer classification."""

# # Feature Filtering Before Model Training
#
# Real-world datasets often contain redundant features — columns that are highly
# correlated with each other. Keeping all of them wastes compute, can hurt model
# performance, and makes results harder to interpret. A common strategy is to
# filter out redundant features *before* training a model.
#
# This example shows how to do exactly that using a two-step workflow:
#
# 1. **ROC (Remove Outliers and Correlations)** — a fast, filter-based module that
#    removes features which are too correlated with each other. It computes pairwise
#    correlations, groups features above a threshold, and keeps only the most
#    relevant feature from each group (scored by F-statistics against the target).
#
# 2. **Octo** — the main ML module that trains models and optimizes hyperparameters
#    on the reduced feature set from step 1.
#
# **What this example covers:**
#
# - Building a two-step workflow with `depends_on`
# - Using ROC for correlation-based feature filtering
# - Passing the filtered features to a downstream Octo task
#
# **How `depends_on` works:**
#
# Each task in a workflow has a `task_id` (starting at 0). When a task sets
# `depends_on=None`, it receives all features from the original dataset. When it
# sets `depends_on=0` (or any other task ID), it receives only the features selected
# by that upstream task. This is how Octopus chains modules into pipelines.

import os

from octopus.example_data import load_breast_cancer_data
from octopus.modules import Octo, Roc
from octopus.study import OctoClassification
from octopus.types import CorrelationType, FIComputeMethod, ModelName, RelevanceMethod

### Load and Preprocess Data

# The breast cancer dataset has 30 numeric features and two classes (malignant vs. benign).

df, features, targets = load_breast_cancer_data()

print("Dataset info:")
print(f"  Features: {len(features)} - {features}")
print(f"  Samples: {df.shape[0]}")
print(f"  Classes: {len(targets)} - {targets}")
print(f"  Target distribution: {df['target'].value_counts().sort_index().to_dict()}")

### Create and Run the Study
#
# The workflow has two tasks:
#
# - **Task 0 (ROC):** Filters out features with Spearman correlation above 0.85.
#   `depends_on=None` means it receives all 30 features.
# - **Task 1 (Octo):** Trains an ExtraTrees model on whatever features ROC kept.
#   `depends_on=0` means it only sees the features selected by Task 0.
#
# We also use `single_outer_split=0` to run only the first outer fold — this is
# a convenient way to test your workflow quickly before committing to a full run.

study = OctoClassification(
    study_name="example_roc_octo",
    studies_directory=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="ACCBAL",  # Balanced accuracy for binary classification
    feature_cols=features,
    target_col="target",
    sample_id_col="index",
    stratification_col="target",
    outer_split_seed=1234,
    single_outer_split=0,  # Process only first outer split for quick testing
    workflow=[
        # Task 0: ROC — filter correlated features
        Roc(
            description="step_0_roc",
            task_id=0,
            depends_on=None,  # receives all original features
            correlation_threshold=0.85,
            correlation_type=CorrelationType.SPEARMAN,
            relevance_method=RelevanceMethod.F_STATISTICS,
        ),
        # Task 1: Octo — train models on the reduced feature set from Task 0
        Octo(
            description="step_1_octo",
            task_id=1,
            depends_on=0,  # receives only features that passed ROC filtering
            n_inner_splits=5,
            models=[
                ModelName.ExtraTreesClassifier,
            ],
            max_outliers=0,
            fi_methods=[FIComputeMethod.PERMUTATION],
            n_startup_trials=10,
            n_trials=12,
            max_features=12,
        ),
    ],
)

study.fit(data=df)

print("Workflow completed successfully!")
print(f"Results saved to: {study.output_path}")
