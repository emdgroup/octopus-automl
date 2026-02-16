# Understanding the Output

After `study.fit()` completes, Octopus writes all results to a timestamped
directory. This page explains the directory layout, what each file contains,
and how to load results programmatically.

## Output directory

The root output directory is created inside `studies_directory` (default:
`./studies/`) with the pattern:

```
{studies_directory}/{study_name}-{YYYYMMDD_HHMMSS}/
```

You can access the path after fitting via `study.output_path`.

## Directory tree

A typical study with 5 outer splits and a two-task workflow (e.g., ROC → Octo)
produces the following structure:

```
my_study-20260409_143000/
├── study_config.json
├── study_meta.json
├── data_raw.parquet
├── data_prepared.parquet
├── health_check_report.csv
├── study.log
│
├── outersplit0/
│   ├── split_row_ids.json
│   ├── task0/
│   │   ├── config/
│   │   │   ├── task_config.json
│   │   │   ├── feature_cols.json
│   │   │   └── feature_groups.json
│   │   └── results/
│   │       └── best/
│   │           ├── selected_features.json
│   │           ├── scores.parquet
│   │           ├── predictions.parquet
│   │           ├── feature_importances.parquet
│   │           └── model/
│   │               ├── model.joblib
│   │               └── predictor.json
│   └── task1/
│       ├── config/
│       │   └── ...
│       └── results/
│           ├── best/
│           │   └── ...
│           ├── ensemble_selection/
│           │   └── ...
│           └── optuna_results.parquet
├── outersplit1/
│   └── ...
└── outersplit4/
    └── ...
```

## Study-level files

| File | Description |
|------|-------------|
| `study_config.json` | Complete study configuration (parameters, workflow definition, feature columns, target assignments). Useful for reproducibility. |
| `study_meta.json` | Metadata: Octopus version, Python version, and creation timestamp. |
| `data_raw.parquet` | The original input DataFrame as passed to `fit()`. |
| `data_prepared.parquet` | Cleaned DataFrame after deduplication and internal column additions (`row_id`). |
| `health_check_report.csv` | Data quality report with one row per issue found. See [Data Health Check](health_check.md). |
| `study.log` | Full execution log. |

## Outer split level

Each `outersplitN/` directory corresponds to one iteration of the
[outer cross-validation loop](../concepts/nested_cv.md#outer-loop-performance-estimation).

**`split_row_ids.json`** contains the train+dev and test row IDs for this split:

```json
{
  "row_id_col": "row_id",
  "traindev_row_ids": [0, 1, 3, 5, ...],
  "test_row_ids": [2, 4, 7, ...]
}
```

## Task level

Each `taskN/` directory holds the configuration and results for one
[workflow task](../concepts/workflow/index.md).

### config/

| File | Description |
|------|-------------|
| `task_config.json` | Task configuration: module type, `task_id`, `depends_on`, description. |
| `feature_cols.json` | Input feature columns received by this task. |
| `feature_groups.json` | Correlation-based feature groups (if applicable). |

### results/

Results are organized by result type:

- **`best/`** — the best single-trial result (always present for
  [Octo](../concepts/workflow/octo.md) tasks).
- **`ensemble_selection/`** — the ensemble result when
  `ensemble_selection=True` is set on the Octo task.

Each result directory contains:

| File | Description |
|------|-------------|
| `selected_features.json` | Features selected by this task (JSON list). |
| `scores.parquet` | Performance metrics across inner CV folds (columns include `outer_split_id`, `inner_split_id`, `task_id`, and one column per metric). |
| `predictions.parquet` | Predictions on train and test partitions. For classification: `pred_proba_0`, `pred_proba_1`, `pred_class`. For regression: `prediction`. |
| `feature_importances.parquet` | Feature importance scores (columns: `feature`, `importance_score`, plus metadata). |
| `model/model.joblib` | Serialized fitted model (a [Bag](../concepts/terminology.md#bag) of inner-split models). |
| `model/predictor.json` | Predictor metadata (selected features). |

**`optuna_results.parquet`** (Octo tasks only) sits directly under `results/`
and contains all Optuna trial results: trial number, objective value, model
type, and hyperparameter values.

## Loading results programmatically

### StudyDiagnostics

The simplest way to access results across all outer splits:

```python
from octopus.diagnostics import StudyDiagnostics

diag = StudyDiagnostics("studies/my_study-20260409_143000")

# Aggregated DataFrames across all outer splits and tasks
diag.predictions       # all predictions
diag.fi                # all feature importances
diag.optuna_trials     # all Optuna trial results
diag.scores            # all performance scores
```


### TaskPredictorTest

To load predictions and models for a specific task:

```python
from octopus.predict import TaskPredictorTest

predictor = TaskPredictorTest(
    study_path="studies/my_study-20260409_143000",
    task_id=1,
)
```

See the [API Reference](../reference/predict.md) for full details.
