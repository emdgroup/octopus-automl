"""Regression on the Tecator dataset — a high-dimensional, small-sample benchmark.

The Tecator dataset contains near-infrared absorbance spectra of 240 meat samples
with 124 features, used to predict fat content. With more features than half the
sample count, this dataset demonstrates why feature selection and regularization
matter for small datasets.

Workflow:
  1. ROC — remove highly correlated spectral features (typical in spectroscopy data)
  2. Octo — train regression models on the filtered feature set
"""

import os

from octopus.example_data import load_tecator_data
from octopus.modules import Octo, Roc
from octopus.study import OctoRegression

### Load Data
df, features, targets = load_tecator_data()

print("Dataset info:")
print(f"  Features: {len(features)}")
print(f"  Samples: {df.shape[0]}")
print("  Target: fat content (continuous)")
print(f"  Feature-to-sample ratio: {len(features) / df.shape[0]:.2f}")
print()

### Create and run OctoRegression
study = OctoRegression(
    name="tecator_regression",
    path=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="R2",
    feature_cols=features,
    target_col="target",
    sample_id_col="index",
    n_folds_outer=5,
    ignore_data_health_warning=True,
    workflow=[
        # Step 0: Remove correlated spectral features
        Roc(
            description="step0_roc_filter",
            task_id=0,
            depends_on=None,
            threshold=0.95,
            correlation_type="spearmanr",
            filter_type="mutual_info",
        ),
        # Step 1: Train regression models on filtered features
        Octo(
            description="step1_octo_regression",
            task_id=1,
            depends_on=0,
            n_trials=100,
            n_folds_inner=5,
            max_features=30,
            ensemble_selection=True,
        ),
    ],
)

study.fit(data=df)

print("Workflow completed")
