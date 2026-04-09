"""Basic example for using Octopus Classification."""

# # Basic Classification
#
# This is the simplest way to get started with Octopus. We build a binary classification
# model using the breast cancer dataset, which has 30 numeric features and two classes
# (malignant vs. benign).
#
# **What this example covers:**
#
# - Loading a dataset and passing it to Octopus
# - Setting up an `OctoClassification` study with a single-step workflow
# - Running the study with `study.fit()`
#
# **A few things to keep in mind:**
#
# - Octopus expects your data as a pandas DataFrame with no missing values and all numeric features.
# - Every study needs a `sample_id_col` — a column that uniquely identifies each row.
# - For classification, setting `stratification_col` ensures each cross-validation fold
#   preserves the original class distribution. This matters especially with imbalanced classes.

### Imports
import os

from octopus.example_data import load_breast_cancer_data
from octopus.modules import Octo
from octopus.study import OctoClassification
from octopus.types import ModelName

### Load and Preprocess Data
df, features, targets = load_breast_cancer_data()

print("Dataset info:")
print(f"  Features: {len(features)} - {features}")
print(f"  Samples: {df.shape[0]}")
print(f"  Classes: {len(targets)} - {targets}")
print(f"  Target distribution: {df['target'].value_counts().sort_index().to_dict()}")

### Create and Run the Study
#
# The workflow below has a single task: an `Octo` module that handles both model training
# and hyperparameter optimization using Optuna.
#
# Key parameters:
#
# - `target_metric="AUCROC"` — the metric Octopus optimizes for. AUCROC is a good default
#   for binary classification because it is insensitive to the classification threshold.
# - `depends_on=None` — this is the first (and only) task, so it receives all features directly.
# - `n_trials=100` — how many hyperparameter configurations Optuna will try.
# - `ensemble_selection=True` — after optimization, Octopus combines the best-performing
#   models into an ensemble for more robust predictions.

study = OctoClassification(
    study_name="basic_classification",
    studies_directory=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="AUCROC",
    feature_cols=features,
    target_col="target",
    sample_id_col="index",
    stratification_col="target",
    workflow=[
        Octo(
            description="step1_octo_full",
            task_id=0,
            depends_on=None,
            models=[ModelName.ExtraTreesClassifier],
            n_trials=100,
            n_inner_splits=5,
            max_features=30,
            ensemble_selection=True,
        ),
    ],
)

study.fit(data=df)

print("Workflow completed")

# ## Learn more
#
# - [Classification User Guide](../userguide/classification.md) — all options for binary and multiclass classification, available models, and metrics.
# - [Nested Cross-Validation](../concepts/nested_cv.md) — why nested CV matters for small datasets.
# - [Understanding the Output](../userguide/output_structure.md) — what files Octopus creates and how to load the results.
