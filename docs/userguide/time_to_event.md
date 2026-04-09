# Time to Event (Survival Analysis)

Octopus supports **time-to-event** (survival analysis) modeling as a first-class task type. This guide covers how to set up and run a survival analysis study.

> **Installation note**
> Time-to-event modeling depends on optional survival dependencies (including `lifelines`).
> Install them with:
>
> ```bash
> pip install "octopus-automl[survival]"
> ```

## Overview

Time-to-event analysis models the time until an event of interest occurs (e.g., disease progression, equipment failure, customer churn), while accounting for **censored** observations — subjects where the event has not yet been observed.

Octopus provides two gradient boosting models with Cox proportional hazards objectives:

| Model | Description |
|-------|-------------|
| `CatBoostCoxSurvival` | CatBoost with Cox loss function. Supports native categoricals. |
| `XGBoostCoxSurvival` | XGBoost with Cox survival objective. |

Both models output **risk scores**, where higher values indicate higher risk (shorter expected survival). The exact scale of these scores (e.g., log-hazard ratio vs hazard ratio) may differ by implementation but is monotonic in risk.

## Data Format

Your dataset must contain:

- **Feature columns**: Numeric or categorical predictors
- **Duration column**: Non-negative numeric, the time to event or censoring
- **Event column**: Binary (0/1 or True/False), where 1 = event observed, 0 = censored
- **Sample ID column**: Sample identifier column

**Requirements and constraints:**

| Column | Type | Missing values allowed | Notes |
|--------|------|----------------------|-------|
| Feature columns | int, float, bool, categorical | Yes (imputed automatically) | Single-value features are removed automatically. Bool is converted to int. |
| Duration column | int or float | No | Must be non-negative (>= 0) |
| Event column | int or bool | No | Binary indicator. At least one event (1) must be present. |
| Sample ID column | any | No | Used for group-aware splitting. Rows with the same ID are kept together. |

Octopus also auto-converts null-like strings (`"None"`, `"null"`, `"nan"`, `"NA"`, `""`)
to `NaN` in feature columns. The reserved column names `datasplit_group` and
`row_id` cannot appear in your data.

```python
import pandas as pd

df = pd.DataFrame({
    "patient_id": [1, 2, 3, 4, 5],
    "age": [55, 62, 48, 71, 59],
    "biomarker": [1.2, 0.8, 1.5, 0.3, 1.1],
    "duration": [12.5, 8.3, 24.0, 5.1, 18.7],
    "event": [1, 1, 0, 1, 0],  # 1=event, 0=censored
})
```

## Basic Usage

```python
from octopus.study import OctoTimeToEvent
from octopus.modules import Octo
from octopus.types import ModelName

study = OctoTimeToEvent(
    study_name="my_survival_study",
    target_metric="CI",
    feature_cols=["age", "biomarker"],
    duration_col="duration",
    event_col="event",
    sample_id_col="patient_id",
    metrics=["CI"],
    studies_directory="./results",
    workflow=[
        Octo(
            task_id=0,
            depends_on=None,
            description="survival_step",
            models=[ModelName.CatBoostCoxSurvival],
            n_trials=20,
            max_features=5,
            ensemble_selection=True,
        )
    ],
)

study.fit(data=df)
```

## Available Models

### CatBoostCoxSurvival

CatBoost gradient boosting with Cox proportional hazards loss. Handles categorical features natively.

**Tunable hyperparameters:**

| Parameter | Range | Scale |
|-----------|-------|-------|
| `learning_rate` | [0.001, 0.1] | log |
| `depth` | [3, 10] | linear |
| `l2_leaf_reg` | [2, 10] | linear |
| `random_strength` | [2, 10] | linear |
| `rsm` | [0.1, 1] | linear |

**Fixed:** `iterations=500`, `logging_level="Silent"`, `task_type="CPU"`

### XGBoostCoxSurvival

XGBoost gradient boosting with Cox partial likelihood objective.

**Tunable hyperparameters:**

| Parameter | Range | Scale |
|-----------|-------|-------|
| `learning_rate` | [0.0001, 0.3] | log |
| `min_child_weight` | [2, 15] | linear |
| `subsample` | [0.15, 1.0] | linear |
| `n_estimators` | [30, 500] | linear |
| `max_depth` | [3, 9] | linear |

## Available Metrics

| Metric Key | Description | Direction |
|------------|-------------|-----------|
| `CI` | Harrell's concordance index | maximize |
| `CI_UNO` | Uno's concordance index (IPCW-corrected) | maximize |

**Harrell's C-index (`CI`)** measures discrimination — how well the model ranks subjects by risk. A value of 1.0 means perfect ranking, 0.5 means random.

**Uno's C-index (`CI_UNO`)** applies Inverse Probability of Censoring Weighting to correct for bias under heavy or informative censoring.

## Using Multiple Models

```python
Octo(
    task_id=0,
    depends_on=None,
    description="compare_models",
    models=[ModelName.CatBoostCoxSurvival, ModelName.XGBoostCoxSurvival],
    n_trials=20,
    max_features=5,
    ensemble_selection=True,
)
```

## Feature Importance

The following feature importance methods are supported for T2E models via `fi_methods`:

- **`permutation`** — Permutation importance using concordance index as scoring
- **`shap`** — SHAP-based feature importance
- **`constant`** — Constant (baseline) feature importance

Additionally, tree-based internal feature importances are always computed automatically by the underlying models.

```python
from octopus.types import FIComputeMethod

Octo(
    ...,
    fi_methods=[FIComputeMethod.PERMUTATION],
)
```

## See also

- [Nested Cross-Validation](../concepts/nested_cv.md) — how Octopus evaluates models and prevents information leakage.
- [Terminology](../concepts/terminology.md) — definitions of Bag, Training, Outer/Inner Split, and other key terms.
- [Understanding the Output](output_structure.md) — what files Octopus creates after a study completes.
