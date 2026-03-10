"""Classification on the SPECTF Heart dataset — handling class imbalance.

The SPECTF dataset contains 349 cardiac SPECT image samples with 44 features.
Patients are classified as normal (0) or abnormal (1), with an imbalanced
class distribution (254 abnormal vs. 95 normal).

This example uses MRMR feature selection followed by model training to
select the most relevant features for this imbalanced classification task.
"""

import os

from octopus.example_data import load_spectf_data
from octopus.modules import Mrmr, Octo
from octopus.study import OctoClassification

### Load Data
df, features, targets = load_spectf_data()

print("Dataset info:")
print(f"  Features: {len(features)}")
print(f"  Samples: {df.shape[0]}")
print(f"  Classes: {len(targets)} - {targets}")
print(f"  Target distribution: {df['target'].value_counts().sort_index().to_dict()}")
print()

### Create and run OctoClassification with Octo -> Mrmr -> Octo workflow
study = OctoClassification(
    name="spectf_classification",
    path=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="ACCBAL",  # Balanced accuracy for imbalanced classes
    feature_cols=features,
    target_col="target",
    sample_id_col="index",
    stratification_col="target",
    n_folds_outer=5,
    ignore_data_health_warning=True,
    workflow=[
        # Step 0: Initial model with all features
        Octo(
            description="step0_octo_full",
            task_id=0,
            depends_on=None,
            models=["ExtraTreesClassifier"],
            n_trials=50,
            n_folds_inner=5,
            max_features=44,
        ),
        # Step 1: Select top features using MRMR
        Mrmr(
            description="step1_mrmr_select",
            task_id=1,
            depends_on=0,
            n_features=15,
            correlation_type="spearman",
        ),
        # Step 2: Retrain with selected features
        Octo(
            description="step2_octo_reduced",
            task_id=2,
            depends_on=1,
            models=["ExtraTreesClassifier"],
            n_trials=100,
            n_folds_inner=5,
            ensemble_selection=True,
        ),
    ],
)

study.fit(data=df)

print("Workflow completed")
