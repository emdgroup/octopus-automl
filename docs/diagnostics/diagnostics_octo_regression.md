# Diagnostics Regression

This notebook provides tools for diagnosing regression model results for Octo module tasks.



```
import os
from pathlib import Path

from octopus.diagnostics import StudyDiagnostics
```

## Select Study Directory

Update the `study_path` variable below to point to your study directory:



```
# Update this path to your study directory
studies_root = os.environ.get("STUDIES_PATH", "../studies")
study_path = os.path.join(studies_root, "wf_octo_mrmr_octo")  # Change this to your study path

study_path_abs = Path(study_path).resolve()
print(f"Using study path: {study_path_abs}")

if not study_path_abs.exists():
    raise ValueError(f"Path does not exist: {study_path_abs}. Please update the study_path variable above.")
```

## Load Study Diagnostics



```
diag = StudyDiagnostics(study_path_abs)
print(f"ML type: {diag.ml_type}")
print(f"Predictions: {len(diag.predictions)} rows")
print(f"Feature importances: {len(diag.fi)} rows")
print(f"Optuna trials: {len(diag.optuna_trials)} rows")
```

## Prediction vs Ground Truth

Interactive scatter plot with diagonal reference line, colored by partition (train/test).



```
diag.plot_predictions_vs_truth()
```

## Feature Importance

Interactive bar chart filtered by outer split, task, training ID, and FI method.



```
diag.plot_fi()
```

## Optuna Insights

### Number of Unique Trials by Model Type



```
diag.plot_optuna_trial_counts()
```

### Optuna Trials: Objective Value and Best Value



```
diag.plot_optuna_trials()
```

### Optuna Hyperparameters



```
diag.plot_optuna_hyperparameters()
```
