# Classification

Octopus supports both **binary** and **multiclass** classification out of the box.
You use the same `OctoClassification` class for both — Octopus automatically detects
which mode to use based on the number of unique values in your target column.

## Overview

Classification predicts a discrete class label for each sample. Octopus wraps the
entire pipeline — data splitting, hyperparameter optimization, model training, and
evaluation — in a single `OctoClassification` object. Under the hood, it runs
nested cross-validation to produce reliable performance estimates even on small
datasets.

**Binary vs. multiclass:**

- If the target column has **2 unique values**, Octopus uses binary classification.
- If it has **3 or more**, it switches to multiclass mode automatically.
- You can also set `ml_type` explicitly if needed.

## Data Format

Your dataset should be a pandas DataFrame with:

- **Feature columns** — numeric, boolean, or categorical. No text, datetime, or object columns.
- **Target column** — integer or boolean class labels (e.g., 0/1 for binary, 0/1/2 for multiclass).
- **Sample ID column** — a column that uniquely identifies each row.

**Requirements and constraints:**

| Column | Type | Missing values allowed | Notes |
|--------|------|----------------------|-------|
| Feature columns | int, float, bool, categorical | Yes (imputed automatically) | Single-value features are removed automatically. Bool is converted to int. |
| Target column | int or bool | No | Exactly 2 unique values for binary, 3+ for multiclass |
| Sample ID column | any | No | Used for group-aware splitting. Rows with the same ID are kept together. |
| Stratification column | int or bool | No | Optional. Cannot be the same as `sample_id_col`. |

Octopus also auto-converts null-like strings (`"None"`, `"null"`, `"nan"`, `"NA"`, `""`)
to `NaN` in feature and target columns. The reserved column names `datasplit_group` and
`row_id` cannot appear in your data.

For binary classification you must specify `positive_class` (the integer value
representing the positive class).

```python
import pandas as pd

# Binary classification example
df = pd.DataFrame({
    "sample_id": [1, 2, 3, 4, 5],
    "age": [55, 62, 48, 71, 59],
    "biomarker_a": [1.2, 0.8, 1.5, 0.3, 1.1],
    "biomarker_b": [0.5, 1.3, 0.9, 1.7, 0.4],
    "target": [1, 0, 1, 0, 1],  # binary: 0 or 1
})
```

## Basic Usage

```python
from octopus.study import OctoClassification
from octopus.modules import Octo
from octopus.types import ModelName

study = OctoClassification(
    study_name="my_classification",
    target_metric="AUCROC",
    feature_cols=["age", "biomarker_a", "biomarker_b"],
    target_col="target",
    sample_id_col="sample_id",
    stratification_col="target",
    workflow=[
        Octo(
            task_id=0,
            depends_on=None,
            description="classification_step",
            models=[ModelName.ExtraTreesClassifier],
            n_trials=100,
            n_inner_splits=5,
            ensemble_selection=True,
        )
    ],
)

study.fit(data=df)
```

**Key parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `target_metric` | Metric to optimize | `"AUCROC"` |
| `sample_id_col` | Column identifying unique subjects (prevents correlated observation leakage) | `None` |
| `stratification_col` | Column used to keep class ratios balanced across CV splits | `None` |
| `n_outer_splits` | Number of outer cross-validation splits | `5` |
| `single_outer_split` | Run only one split for quick testing (e.g., `0`) | `None` |
| `n_cpus` | Number of CPUs (`0` = all, `-1` = all but one) | `0` |

!!! tip
    Always set `stratification_col` for classification tasks. Without it, some splits
    may end up with very few samples of the minority class, especially on small datasets.

!!! tip
    If your dataset contains multiple rows per subject (e.g. longitudinal measurements,
    repeated experiments), set `sample_id_col` to the column identifying subjects.
    Octopus will ensure all rows from the same subject stay in the same split,
    preventing [information leakage](../concepts/nested_cv.md#what-is-information-leakage).

## Multiclass Classification

For multiclass problems, use `OctoClassification` exactly the same way — just make sure
your target column has more than two unique values and choose a multiclass-compatible
metric:

```python
study = OctoClassification(
    study_name="my_multiclass",
    target_metric="AUCROC_MACRO",  # macro-averaged AUCROC across all classes
    feature_cols=features,
    target_col="target",
    sample_id_col="sample_id",
    stratification_col="target",
    workflow=[
        Octo(
            task_id=0,
            depends_on=None,
            models=[
                ModelName.ExtraTreesClassifier,
                ModelName.RandomForestClassifier,
                ModelName.XGBClassifier,
            ],
            n_trials=50,
        )
    ],
)
```

!!! note
    `CatBoostClassifier` is currently only available for binary classification.
    All other classification models work in both binary and multiclass mode.

## Available Models

| Model | Type | Default | Notes |
|-------|------|---------|-------|
| `ExtraTreesClassifier` | Tree ensemble | Yes | Fast, good baseline |
| `RandomForestClassifier` | Tree ensemble | Yes | Robust general-purpose model |
| `XGBClassifier` | Gradient boosting | Yes | Strong performance on tabular data |
| `CatBoostClassifier` | Gradient boosting | Yes | Native categorical support; binary only |
| `HistGradientBoostingClassifier` | Gradient boosting | Yes | Native categorical support, handles missing values |
| `LogisticRegressionClassifier` | Linear | Yes | Simple, interpretable |
| `GradientBoostingClassifier` | Gradient boosting | No | Sklearn implementation |
| `GaussianProcessClassifier` | Kernel | No | Best for very small datasets |

Models marked as "Default" are included automatically when you don't specify a `models` list
in the `Octo` configuration.

## Available Metrics

### Binary Classification

| Metric | Description | Direction |
|--------|-------------|-----------|
| `AUCROC` | Area Under ROC Curve | Maximize |
| `ACCBAL` | Balanced Accuracy | Maximize |
| `ACC` | Accuracy | Maximize |
| `F1` | F1 Score | Maximize |
| `AUCPR` | Area Under Precision-Recall Curve | Maximize |
| `PRECISION` | Precision | Maximize |
| `RECALL` | Recall | Maximize |
| `MCC` | Matthews Correlation Coefficient | Maximize |
| `NEGBRIERSCORE` | Negative Brier Score (calibration) | Maximize |
| `LOGLOSS` | Log Loss (cross-entropy) | Maximize |

### Multiclass Classification

| Metric | Description | Direction |
|--------|-------------|-----------|
| `AUCROC_MACRO` | Macro-averaged AUCROC (one-vs-rest) | Maximize |
| `AUCROC_WEIGHTED` | Weighted AUCROC (by class frequency) | Maximize |
| `ACCBAL` | Balanced Accuracy | Maximize |
| `ACC` | Accuracy | Maximize |
| `MCC` | Matthews Correlation Coefficient | Maximize |
| `LOGLOSS` | Log Loss | Maximize |

!!! tip
    For binary classification, `AUCROC` is a solid default — it evaluates the model's
    ability to rank positive samples higher than negative ones, independent of the
    classification threshold. For multiclass, `AUCROC_MACRO` is the recommended
    starting point.

## Feature Importance

Octopus can compute feature importances using several methods via the `fi_methods`
parameter in the `Octo` configuration:

- **`permutation`** — Permutation importance: measures how much performance drops when
  a feature's values are shuffled. Model-agnostic and reliable.
- **`shap`** — SHAP values: game-theoretic feature attributions. More detailed but slower.
- **`constant`** — Baseline constant importance (for reference).

Tree-based models also provide built-in (internal) feature importances automatically.

```python
from octopus.types import FIComputeMethod

Octo(
    ...,
    fi_methods=[FIComputeMethod.PERMUTATION, FIComputeMethod.SHAP],
)
```

## Example Workflows

For runnable end-to-end examples, see:

- [Basic Classification](../examples/basic_classification.md): The simplest way to run a binary classification study with a single Octo step.
- [Multiclass Classification](../examples/multiclass_classification.md): Shows how Octopus handles three or more classes using the Wine dataset.
- [Feature Filtering](../examples/feature_filtering.md): Demonstrates removing correlated features with ROC before training a model.
- [Sequential Classification Workflow](../examples/sequential_classification_workflow.md): Chains Octo, MRMR, and Octo into a three-step pipeline that progressively narrows the feature set.
- [Parallel Classification Workflow](../examples/parallel_classification_workflow.md): Runs Octo and AutoGluon independently on the same data to compare approaches.
