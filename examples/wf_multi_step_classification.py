"""Sequential workflow example with artificial classification dataset.

This example demonstrates a multi-step workflow using:
- Artificial dataset with 30 features
- Binary classification problem (not too easy)
- 5 outer splits, 5 inner splits
- 100 trials for hyperparameter optimization
- ExtraTreesClassifier model
- Sequential tasks: Octo -> Mrmr -> Octo (with reduced features)

Run this script before using the analysis_classification.ipynb notebook::

    python examples/wf_multi_step_classification.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from octopus.modules import Mrmr, Octo
from octopus.study import OctoClassification
from octopus.types import CorrelationType, ModelName

np.random.seed(42)

X, y = make_classification(
    n_samples=500,
    n_features=30,
    n_informative=15,
    n_redundant=10,
    n_repeated=5,
    n_classes=2,
    n_clusters_per_class=3,
    weights=[0.6, 0.4],
    flip_y=0.1,
    class_sep=0.5,
    random_state=42,
)

feature_names = [f"feature_{i:02d}" for i in range(30)]
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
df = df.reset_index()

print("Dataset created:")
print(f"  Samples: {len(df)}")
print(f"  Features: {len(feature_names)}")
print(f"  Class distribution: {df['target'].value_counts().to_dict()}")
print()

study = OctoClassification(
    study_name="wf_multi_step_classification",
    target_metric="ACCBAL",
    feature_cols=feature_names,
    target_col="target",
    sample_id_col="index",
    stratification_col="target",
    n_outer_splits=5,
    workflow=[
        Octo(
            description="step1_octo_full",
            task_id=0,
            depends_on=None,
            models=[ModelName.ExtraTreesClassifier],
            n_trials=100,
            n_inner_splits=5,
            max_features=30,
        ),
        Mrmr(
            description="step2_mrmr",
            task_id=1,
            depends_on=0,
            n_features=15,
            correlation_type=CorrelationType.SPEARMAN,
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

print("Starting workflow execution...")

study.fit(data=df)

print("Workflow completed successfully!")
print(f"Results saved to: {study.output_path}")
