"""Classification on the Promoters dataset — extreme high-dimensional, small-sample scenario.

The Molecular Biology Promoters dataset has only 106 DNA sequences classified as
promoter (+) or non-promoter (-). After one-hot encoding the 57 nucleotide
positions (a/c/g/t), the dataset has 228 features — more than twice the number
of samples.

This is an extreme case where feature selection is essential. The workflow uses
ROC to remove correlated features, then MRMR to select the most informative
subset, and finally trains a model on the reduced feature set.
"""

import os

from octopus.example_data import load_promoters_data
from octopus.modules import Mrmr, Octo, Roc
from octopus.study import OctoClassification

### Load Data
df, features, targets = load_promoters_data()

print("Dataset info:")
print(f"  Features: {len(features)} (after one-hot encoding)")
print(f"  Samples: {df.shape[0]}")
print(f"  Classes: {len(targets)} - {targets}")
print(f"  Target distribution: {df['target'].value_counts().sort_index().to_dict()}")
print(f"  Feature-to-sample ratio: {len(features) / df.shape[0]:.2f}")
print()

### Create and run OctoClassification
study = OctoClassification(
    name="promoters_classification",
    path=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="ACCBAL",
    feature_cols=features,
    target_col="target",
    sample_id_col="index",
    stratification_col="target",
    n_folds_outer=5,
    ignore_data_health_warning=True,
    workflow=[
        # Step 0: Remove correlated one-hot features
        Roc(
            description="step0_roc_filter",
            task_id=0,
            depends_on=None,
            threshold=0.90,
            correlation_type="spearmanr",
            filter_type="f_statistics",
        ),
        # Step 1: Select top features using MRMR
        Mrmr(
            description="step1_mrmr_select",
            task_id=1,
            depends_on=0,
            n_features=20,
            correlation_type="spearman",
        ),
        # Step 2: Train on the reduced feature set
        Octo(
            description="step2_octo_final",
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
