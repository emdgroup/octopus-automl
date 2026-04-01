"""Example: Use of ROC and Octo modules for breast cancer classification."""

# This example demonstrates how to use Octopus with ROC (Remove Outliers and Correlations)
# and Octo modules for binary classification on the breast cancer dataset.
# The workflow includes:
# 1. ROC module for feature correlation analysis and filtering
# 2. Octo module for model training and hyperparameter optimization

import os

from octopus.example_data import load_breast_cancer_data
from octopus.modules import Octo, Roc
from octopus.study import OctoClassification
from octopus.types import CorrelationType, FIComputeMethod, ModelName, ROCFilterMethod

### Load and Preprocess Data

# Load the breast cancer dataset from sklearn
# This is a binary classification dataset with 30 features
# Target: 0 = malignant, 1 = benign

df, features, targets = load_breast_cancer_data()

print("Dataset info:")
print(f"  Features: {len(features)} - {features}")
print(f"  Samples: {df.shape[0]}")
print(f"  Classes: {len(targets)} - {targets}")
print(f"  Target distribution: {df['target'].value_counts().sort_index().to_dict()}")

### Create and run OctoClassification with ROC + Octo workflow

study = OctoClassification(
    study_name="example_roc_octo",
    study_path=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="ACCBAL",  # Balanced accuracy for binary classification
    feature_cols=features,
    target_col="target",
    sample_id_col="index",
    stratification_col="target",
    outer_split_seed=1234,
    single_outer_split=0,  # Process only first outersplit for quick testing
    workflow=[
        # Step 0: ROC - Remove highly correlated features and apply statistical filtering
        Roc(
            description="step_0_roc",
            task_id=0,
            depends_on=None,  # First step, no input dependency
            threshold=0.85,  # Remove features with correlation > 0.85
            correlation_type=CorrelationType.SPEARMAN,  # Use Spearman correlation
            filter_type=ROCFilterMethod.F_STATISTICS,  # Apply F-statistics filtering
        ),
        # Step 1: Octo - Train models on filtered features from ROC step
        Octo(
            description="step_1_octo",
            task_id=1,
            depends_on=0,  # Use output from ROC step
            # Cross-validation settings
            n_inner_splits=5,
            # Model selection
            models=[
                ModelName.ExtraTreesClassifier,
                # ModelName.RandomForestClassifier,
            ],
            max_outliers=0,  # No outlier removal
            fi_methods=[FIComputeMethod.PERMUTATION],  # Feature importance method
            # Hyperparameter optimization with Optuna
            n_startup_trials=10,
            n_trials=12,  # Number of hyperparameter optimization trials
            max_features=12,  # Maximum number of features to select
        ),
    ],
)

study.fit(data=df)

print("Workflow completed successfully!")
print(f"Results saved to: {study.output_path}")
