"""Basic example for using Octopus Time-to-Event (Survival Analysis).

This example demonstrates how to use Octopus to create a time-to-event
model using Cox proportional hazards gradient boosting.

We use a synthetic survival dataset. Please ensure your dataset is clean, with
no missing values (NaN), and that all features are numeric.

Run this script before using the analysis_timetoevent.ipynb notebook::

    python examples/basic_timetoevent.py
"""

import os

from octopus.example_data import load_survival_data
from octopus.modules import Octo
from octopus.study import OctoTimeToEvent
from octopus.types import ModelName

### Load synthetic survival dataset
df, features = load_survival_data()

print("Dataset info:")
print(f"  Features: {len(features)} - {features}")
print(f"  Samples: {df.shape[0]}")
print(f"  Events: {df['event'].sum()} ({df['event'].mean():.0%})")
print(f"  Censored: {(df['event'] == 0).sum()} ({(df['event'] == 0).mean():.0%})")

### Create and run OctoTimeToEvent
study = OctoTimeToEvent(
    study_name="basic_timetoevent",
    studies_directory=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="CI",
    feature_cols=features,
    duration_col="duration",
    event_col="event",
    sample_id_col="index",
    workflow=[
        Octo(
            description="step1_octo",
            task_id=0,
            depends_on=None,
            models=[ModelName.CatBoostCoxSurvival],
            n_trials=100,
            n_inner_splits=5,
            ensemble_selection=True,
        ),
    ],
)

study.fit(data=df)

print("Done! Study saved to:", study.output_path)
