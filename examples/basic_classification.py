"""Basic example for using Octopus Classification.

This example demonstrates how to use Octopus to create a machine learning
classification model with a two-step workflow: prediction followed by feature
selection and a second prediction on the reduced feature set.

We use the breast cancer dataset. Please ensure your dataset is clean, with
no missing values (NaN), and that all features are numeric.

Run this script before using the analysis_classification.ipynb notebook::

    python examples/basic_classification.py
"""

import os

from octopus.example_data import load_breast_cancer_data
from octopus.modules import Mrmr, Octo
from octopus.study import OctoClassification
from octopus.types import ModelName

### Load and Preprocess Data
df, features, targets = load_breast_cancer_data()

print("Dataset info:")
print(f"  Features: {len(features)} - {features}")
print(f"  Samples: {df.shape[0]}")
print(f"  Classes: {len(targets)} - {targets}")
print(f"  Target distribution: {df['target'].value_counts().sort_index().to_dict()}")

### Create and run OctoClassification
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
        Mrmr(
            description="step2_mrmr",
            task_id=1,
            depends_on=0,
            n_features=10,
        ),
        Octo(
            description="step3_octo_reduced",
            task_id=2,
            depends_on=1,
            models=[ModelName.ExtraTreesClassifier],
            n_trials=100,
            n_inner_splits=5,
            ensemble_selection=True,
        ),
    ],
)

study.fit(data=df)

print("Done! Study saved to:", study.output_path)
