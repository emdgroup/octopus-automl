"""Basic example for using Octopus Time-to-Event (Survival Analysis).

This example demonstrates how to use Octopus to create a time-to-event
model. We use a synthetic survival dataset with Cox proportional hazards.

Run this script before using the analysis_timetoevent.ipynb notebook::

    python examples/basic_timetoevent.py
"""

import os

from octopus.example_data import load_survival_data
from octopus.modules import Tako
from octopus.study import OctoTimeToEvent
from octopus.types import ModelName

df, features = load_survival_data()

print("Dataset info:")
print(f"  Features: {len(features)} - {features}")
print(f"  Samples: {df.shape[0]}")
print(f"  Events: {df['event'].sum()} ({df['event'].mean():.0%})")
print(f"  Censored: {(df['event'] == 0).sum()} ({(df['event'] == 0).mean():.0%})")

study = OctoTimeToEvent(
    study_name="basic_timetoevent",
    studies_directory=os.environ.get("STUDIES_PATH", "./studies"),
    target_metric="CI",
    feature_cols=features,
    duration_col="duration",
    event_col="event",
    sample_id_col="index",
    workflow=[
        Tako(
            description="step1_tako",
            task_id=0,
            models=[ModelName.XGBoostCoxSurvival],
            n_trials=25,
        ),
    ],
)

study.fit(data=df)

print("Workflow completed")
