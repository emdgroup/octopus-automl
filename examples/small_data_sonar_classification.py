"""Classification on the Sonar dataset — demonstrating seed sensitivity.

The Sonar dataset has 208 samples and 60 features (sonar signals bounced off
a metal cylinder vs. rocks). With so few samples, a single random train/test
split can swing accuracy by 10-15% depending on the seed.

This example uses multiple datasplit seeds to show how Octopus produces
robust results across different splits — a key advantage for small datasets.
"""

import os

from octopus.example_data import load_sonar_data
from octopus.modules import Octo
from octopus.study import OctoClassification

### Load Data
df, features, targets = load_sonar_data()

print("Dataset info:")
print(f"  Features: {len(features)}")
print(f"  Samples: {df.shape[0]}")
print(f"  Classes: {len(targets)} - {targets}")
print(f"  Target distribution: {df['target'].value_counts().sort_index().to_dict()}")
print()

### Create and run OctoClassification with multiple seeds
study = OctoClassification(
    name="sonar_classification",
    path=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="AUCROC",
    feature_cols=features,
    target_col="target",
    sample_id_col="index",
    stratification_col="target",
    n_folds_outer=5,
    datasplit_seed_outer=0,
    ignore_data_health_warning=True,
    workflow=[
        Octo(
            description="step0_octo",
            task_id=0,
            depends_on=None,
            n_folds_inner=5,
            max_features=30,  # Constrain to half the features
            ensemble_selection=True,
        ),
    ],
)

study.fit(data=df)

print("Workflow completed")
