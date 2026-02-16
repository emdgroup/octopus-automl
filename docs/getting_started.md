# Getting Started

This page walks you through installing Octopus and running your first study.

## Installation

**Requirements:** Python 3.12 or later.

```bash
# Install with recommended dependencies (includes optional packages such as AutoGluon)
pip install "octopus-automl[recommended]"
```

You can also pick only the extras you need:

```bash
pip install "octopus-automl[autogluon]"     # AutoGluon
pip install "octopus-automl[boruta]"        # Boruta feature selection
pip install "octopus-automl[survival]"      # Support time-to-event / survival analysis
pip install "octopus-automl[examples]"      # Dependencies for running examples

# Combine multiple extras
pip install "octopus-automl[autogluon,examples]"
```

!!! tip "Hardware"
    For maximum speed, run Octopus on a machine with n × m CPUs for an
    n × m nested cross-validation. Development is
    typically done on an AWS c5.9xlarge EC2 instance (36 vCPUs).

## Run your first study

```python
from octopus.example_data import load_breast_cancer_data
from octopus.modules import Octo
from octopus.study import OctoClassification
from octopus.types import ModelName

# 1. Load a built-in example dataset (breast cancer, 569 samples, 30 features)
df, features, targets = load_breast_cancer_data()

# 2. Create a classification study
study = OctoClassification(
    study_name="my_first_study",
    target_metric="AUCROC",
    feature_cols=features,
    target_col="target",
    sample_id_col="index",
    stratification_col="target",
    workflow=[
        Octo(
            task_id=0,
            depends_on=None,
            models=[ModelName.ExtraTreesClassifier],
            n_trials=50,
            n_inner_splits=5,
            ensemble_selection=True,
        ),
    ],
)

# 3. Fit — this runs the full nested cross-validation pipeline
study.fit(data=df)

print(f"Results saved to: {study.output_path}")
```

## What just happened?

Octopus performed the following steps automatically:

1. **Data health check** — validated your dataset for missing values, class
   imbalance, duplicate rows, and potential leakage.
2. **Outer cross-validation** — split the data into 5 outer folds. Each fold
   holds out 20 % as a test set that is never used for training or tuning.
3. **Inner cross-validation + HPO** — within each outer fold, further split
   into 5 inner folds and ran 50 Optuna trials to find the best
   hyperparameters.
4. **Ensemble selection** — combined the top-performing trial models into a
   robust ensemble.
5. **Evaluation** — scored the ensemble on the held-out outer test set to
   produce an unbiased performance estimate.

For details on why this matters, see
[Nested Cross-Validation](concepts/nested_cv.md).

## Find the results

After `fit()` completes, results are saved to the `studies/` directory (or
wherever you set `studies_directory`). The output folder contains:

- **`study_config.json`** — the full study configuration for reproducibility.
- **`health_check_report.csv`** — data quality findings.
- **`outersplit0/` … `outersplit4/`** — one folder per outer CV split, each
  containing task results: predictions, feature importances, scores, and
  trained models.

For a complete walkthrough of the output directory, see
[Understanding the Output](userguide/output_structure.md).

## Load and inspect results

```python
from octopus.diagnostics import StudyDiagnostics

diag = StudyDiagnostics(study.output_path)

# Print summary
print(f"ML type:    {diag.ml_type}")
print(f"Predictions: {len(diag.predictions)} rows")
print(f"FI entries:  {len(diag.fi)} rows")
```

## Next steps

- **[Classification](userguide/classification.md)** — all options for binary
  and multiclass classification, including available models and metrics.
- **[Regression](userguide/regression.md)** — continuous-target prediction with
  `OctoRegression`.
- **[Time to Event](userguide/time_to_event.md)** — survival analysis with
  censored data.
- **[Workflow & Modules](concepts/workflow/index.md)** — chain feature selection
  and ML modules into multi-step pipelines.
- **[Examples](examples/index.md)** — runnable end-to-end workflows from basic to
  advanced.
