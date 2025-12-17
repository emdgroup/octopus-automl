"""Example: Use of ROC and Octo modules for breast cancer classification."""

# This example demonstrates how to use Octopus with ROC (Remove Outliers and Correlations)
# and Octo modules for binary classification on the breast cancer dataset.
# The workflow includes:
# 1. ROC module for feature correlation analysis and filtering
# 2. Octo module for model training and hyperparameter optimization

import os

from sklearn.datasets import load_breast_cancer

from octopus import OctoClassification
from octopus.modules import Octo, Roc

### Load and Preprocess Data

# Load the breast cancer dataset from sklearn
# This is a binary classification dataset with 30 features
# Target: 0 = malignant, 1 = benign

breast_cancer = load_breast_cancer(as_frame=True)

df = breast_cancer["frame"].reset_index()
df.columns = df.columns.str.replace(" ", "_")
features = list(breast_cancer["feature_names"])
features = [feature.replace(" ", "_") for feature in features]

### Create and run OctoClassification with ROC + Octo workflow

study = OctoClassification(
    name="example_roc_octo",
    path=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="ACCBAL",  # Balanced accuracy for binary classification
    feature_columns=features,
    target="target",
    sample_id="index",
    stratification_column="target",
    metrics=["AUCROC", "ACCBAL", "ACC", "LOGLOSS"],
    datasplit_seed_outer=1234,
    ignore_data_health_warning=True,
    outer_parallelization=True,
    run_single_experiment_num=0,  # Process only first outer loop experiment for quick testing
    workflow=[
        # Step 0: ROC - Remove highly correlated features and apply statistical filtering
        Roc(
            description="step_0_roc",
            task_id=0,
            depends_on_task=-1,  # First step, no input dependency
            load_task=False,
            threshold=0.85,  # Remove features with correlation > 0.85
            correlation_type="spearmanr",  # Use Spearman correlation
            filter_type="f_statistics",  # Apply F-statistics filtering
        ),
        # Step 1: Octo - Train models on filtered features from ROC step
        Octo(
            description="step_1_octo",
            task_id=1,
            depends_on_task=0,  # Use output from ROC step
            load_task=False,
            # Cross-validation settings
            n_folds_inner=5,
            # Model selection
            models=[
                "ExtraTreesClassifier",
                # "RandomForestClassifier",
            ],
            model_seed=0,
            n_jobs=1,
            max_outl=0,  # No outlier removal
            fi_methods_bestbag=["permutation"],  # Feature importance method
            # Parallelization settings
            inner_parallelization=True,
            n_workers=5,
            # Hyperparameter optimization with Optuna
            optuna_seed=0,
            n_optuna_startup_trials=10,
            resume_optimization=False,
            n_trials=12,  # Number of hyperparameter optimization trials
            max_features=12,  # Maximum number of features to select
            penalty_factor=1.0,
        ),
    ],
)

study.fit(data=df)

print("Workflow completed successfully!")
print(f"Results saved to: {study.output_path}")
