"""Sequential workflow example with artificial classification dataset.

This example demonstrates a multi-step workflow using:
- Artificial dataset with 30 features
- Binary classification problem (not too easy)
- 5 outer splits, 5 inner splits
- 100 trials for hyperparameter optimization
- ExtraTreesClassifier model
- Sequential tasks: Octo -> Mrmr -> Octo (with reduced features)
"""

# # Sequential Classification Workflow
#
# This example demonstrates a **sequential three-step workflow** — the most common
# pattern in Octopus. Each task passes its selected features to the next, progressively
# narrowing the feature set:
#
# ```
# Task 0 (Octo)   → 30 features → trains models, selects features
#       ↓
# Task 1 (MRMR)   → receives Task 0's features → selects top 15
#       ↓
# Task 2 (Octo)   → receives 15 features → trains final ensemble
# ```
#
# **What this example covers:**
#
# - Chaining three tasks sequentially using `depends_on`
# - Using MRMR feature selection between two Octo stages
# - Enabling ensemble selection in the final Octo step
# - Working with a synthetic dataset designed to be realistically difficult
#
# **Why use multiple Octo steps?**
#
# The first Octo step explores a broad feature space and produces initial feature
# importances. MRMR then uses those importances to select a compact, low-redundancy
# subset. The second Octo step can focus its hyperparameter search on this smaller
# space, often finding better models with fewer features. The `ensemble_selection=True`
# flag on the final step combines the top-performing models into a robust ensemble.

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from octopus.modules import Mrmr, Octo
from octopus.study import OctoClassification
from octopus.types import CorrelationType, ModelName

### Generate Synthetic Dataset
#
# We create a binary classification problem that's intentionally challenging:
# only half the features are informative, there's 10% label noise, and the
# classes overlap moderately. This mimics real-world conditions where not
# every measured feature carries a signal.

np.random.seed(42)

X, y = make_classification(
    n_samples=500,
    n_features=30,
    n_informative=15,
    n_redundant=10,
    n_repeated=5,
    n_classes=2,
    n_clusters_per_class=3,
    weights=[0.6, 0.4],
    flip_y=0.1,
    class_sep=0.5,
    random_state=42,
)

feature_names = [f"feature_{i:02d}" for i in range(30)]
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
df = df.reset_index()

print("Dataset created:")
print(f"  Samples: {len(df)}")
print(f"  Features: {len(feature_names)}")
print(f"  Class distribution: {df['target'].value_counts().to_dict()}")
print()

### Create and Run the Study
#
# Notice how `depends_on` chains the tasks: Task 1 depends on Task 0, Task 2
# depends on Task 1. Each downstream task only sees the features selected by
# its upstream dependency.
study = OctoClassification(
    study_name="wf_octo_mrmr_octo",
    target_metric="ACCBAL",
    feature_cols=feature_names,
    target_col="target",
    sample_id_col="index",
    stratification_col="target",
    n_outer_splits=5,  # 5 outer splits
    workflow=[
        # Task 0: Initial Octo — explore the full feature space
        Octo(
            description="step1_octo_full",
            task_id=0,
            depends_on=None,  # entry point: receives all 30 features
            models=[ModelName.ExtraTreesClassifier],
            n_trials=100,
            n_inner_splits=5,
            max_features=30,
        ),
        # Task 1: MRMR — select top 15 features from Task 0's output
        Mrmr(
            description="step2_mrmr",
            task_id=1,
            depends_on=0,  # receives features selected by Task 0
            n_features=15,
            correlation_type=CorrelationType.SPEARMAN,
        ),
        # Task 2: Final Octo — retrain on reduced features with ensemble selection
        Octo(
            description="step3_octo_reduced",
            task_id=2,
            depends_on=1,  # receives the 15 features selected by MRMR
            models=[ModelName.ExtraTreesClassifier],
            n_trials=100,
            n_inner_splits=5,
            ensemble_selection=True,
        ),
    ],
)

print("Starting workflow execution...")

study.fit(data=df)

print("Workflow completed successfully!")
print(f"Results saved to: {study.output_path}")

# ## Learn more
#
# - [Workflow & Modules](../concepts/workflow/index.md) — how multi-step pipelines work.
# - [MRMR Module](../concepts/workflow/mrmr.md) — Maximum Relevance Minimum Redundancy feature selection.
# - [Octo Module](../concepts/workflow/octo.md) — Optuna-based HPO with ensembling.
