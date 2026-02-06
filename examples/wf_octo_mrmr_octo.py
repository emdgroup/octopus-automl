"""Sequential workflow example with artificial classification dataset.

This example demonstrates a multi-step workflow using:
- Artificial dataset with 30 features
- Binary classification problem (not too easy)
- 5 outer folds, 5 inner folds
- 100 trials for hyperparameter optimization
- ExtraTreesClassifier model
- Sequential tasks: Octo -> Mrmr -> Octo (with reduced features)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from octopus import OctoClassification
from octopus.modules import Mrmr, Octo

# Set random seed for reproducibility
np.random.seed(42)

# Create artificial classification dataset
# Parameters chosen to make the problem "not too easy":
# - 30 features total
# - Only 15 informative features (50%)
# - 10 redundant features (correlated with informative)
# - 5 repeated features (duplicates)
# - class_sep=0.5 for moderate difficulty (lower = harder)
# - flip_y=0.1 to add 10% label noise
X, y = make_classification(
    n_samples=500,
    n_features=30,
    n_informative=15,
    n_redundant=10,
    n_repeated=5,
    n_classes=2,
    n_clusters_per_class=3,
    weights=[0.6, 0.4],  # Imbalanced classes
    flip_y=0.1,  # 10% label noise
    class_sep=0.5,  # Moderate class separation (not too easy)
    random_state=42,
)

# Create DataFrame with feature names
feature_names = [f"feature_{i:02d}" for i in range(30)]
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
df = df.reset_index()

print("Dataset created:")
print(f"  Samples: {len(df)}")
print(f"  Features: {len(feature_names)}")
print(f"  Class distribution: {df['target'].value_counts().to_dict()}")
print()

# Create and run OctoClassification with sequential multi-step workflow
study = OctoClassification(
    name="wf_octo_mrmr_octo",
    target_metric="ACCBAL",
    feature_cols=feature_names,
    target="target",
    sample_id_col="index",
    stratification_col="target",
    n_folds_outer=5,  # 5 outer folds
    ignore_data_health_warning=True,
    outer_parallelization=True,  # Run all outer folds in parallel
    workflow=[
        # Task 0: Initial Octo with all features
        Octo(
            description="step1_octo_full",
            task_id=0,
            depends_on_task=-1,  # First task, depends on input
            models=["ExtraTreesClassifier"],
            n_trials=100,  # 100 trials for hyperparameter optimization
            n_folds_inner=5,  # 5 inner folds
            max_features=30,  # Use all 30 features
        ),
        # Task 1: Feature selection using Mrmr
        Mrmr(
            description="step2_mrmr",
            task_id=1,
            depends_on_task=0,
            n_features=15,  # Select top 15 features
            correlation_type="spearman",
        ),
        # Task 2: Octo with reduced features
        Octo(
            description="step3_octo_reduced",
            task_id=2,
            depends_on_task=1,
            models=["ExtraTreesClassifier"],
            n_trials=100,
            n_folds_inner=5,
            ensemble_selection=True,
        ),
    ],
)

print("Starting workflow execution...")

study.fit(data=df)

print("Workflow completed successfully!")
print(f"Results saved to: {study.output_path}")
