"""Example: Parallel Octo and AutoGluon workflow for binary classification."""

# This example demonstrates how to use Octopus with both Octo and AutoGluon modules
# in a PARALLEL workflow for binary classification. In this case, both modules are run on the same input data.
#
# The workflow includes:
# 1. Octo module
# 2. AutoGluon module
# Both modules operate on the same base input data.

import os

import pandas as pd
from sklearn.datasets import make_classification

from octopus import OctoClassification
from octopus.modules import AutoGluon, Octo

### Generate Synthetic Binary Classification Dataset
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

### Create and run OctoClassification with PARALLEL Octo + AutoGluon workflow

study = OctoClassification(
    name="wf_octo_autogluon_parallel",
    path=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="AUCROC",  # Area Under ROC Curve for binary classification
    feature_cols=feature_names,
    target="target",
    sample_id_col="index",
    stratification_col="target",  # Ensure balanced splits
    metrics=["AUCROC", "ACCBAL", "ACC", "LOGLOSS", "F1"],
    n_folds_outer=5,  # 5-fold outer cross-validation
    ignore_data_health_warning=True,
    outer_parallelization=True,
    run_single_experiment_num=-1,  # process all outersplits
    workflow=[
        # Step 0: octo
        Octo(
            description="step_0_octo",
            task_id=0,
            depends_on_task=-1,  # -1 = base input (parallel with AutoGluon)
            # Cross-validation settings
            n_folds_inner=5,
            # Model selection - using tree-based models for feature importance
            models=[
                "ExtraTreesClassifier",
            ],
            fi_methods_bestbag=["permutation"],  # Feature importance method
            # Parallelization settings
            inner_parallelization=True,
            n_workers=5,
            n_trials=100,  # Number of hyperparameter optimization trials
            # Constrained hyperparameter optimization
            # max_features=60,  # Maximum number of features to select
            # penalty_factor=1.0,  # Complexity penalty for feature selection
        ),
        # Step 1: AutoGluon
        AutoGluon(
            description="step_1_autogluon",
            task_id=1,
            depends_on_task=-1,  # -1 = base input (parallel with Octo)
            verbosity=3,  # Standard logging
            time_limit=600,
            presets=["medium_quality"],  # Balance between speed and accuracy
            num_bag_folds=5,  # 5-fold bagging for ensemble models
            included_model_types=[
                "XT",  # ExtraTrees
            ],
        ),
    ],
)

# Run the study on the synthetic data
print("Starting Octo + AutoGluon workflow...")
study.fit(data=df)

print("Workflow completed successfully!")
print(f"Results saved to: {study.output_path}")
