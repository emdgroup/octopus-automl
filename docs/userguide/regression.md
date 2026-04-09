# Regression

Octopus supports regression tasks for predicting continuous numeric outcomes. The setup
is very similar to classification — you use `OctoRegression` instead of
`OctoClassification`, pick a regression metric, and the rest of the pipeline
(nested CV, hyperparameter optimization, ensembling) works the same way.

## Overview

Regression predicts a continuous value (e.g., price, temperature, disease severity
score) rather than a class label. Octopus handles the full pipeline: data splitting,
model training with Optuna-based hyperparameter optimization, and evaluation — all
wrapped in nested cross-validation.

**Key differences from classification:**

- Use `OctoRegression` instead of `OctoClassification`.
- Metrics measure prediction error (MAE, RMSE) or explained variance (R²) rather than
  class discrimination (AUCROC, F1).
- There is no `positive_class` or `ml_type` parameter.
- Stratification is optional — it can still be useful if you have a categorical column
  that should be balanced across splits (e.g., site or batch).

## Data Format

Your dataset should be a pandas DataFrame with:

- **Feature columns** — numeric, boolean, or categorical. No text, datetime, or object columns.
- **Target column** — a continuous numeric value.
- **Sample ID column** — a column that uniquely identifies each row.

**Requirements and constraints:**

| Column | Type | Missing values allowed | Notes |
|--------|------|----------------------|-------|
| Feature columns | int, float, bool, categorical | Yes (imputed automatically) | Single-value features are removed automatically. Bool is converted to int. |
| Target column | int or float | No | Continuous numeric value |
| Sample ID column | any | No | Used for group-aware splitting. Rows with the same ID are kept together. |
| Stratification column | int or bool | No | Optional. Cannot be the same as `sample_id_col`. |

Octopus also auto-converts null-like strings (`"None"`, `"null"`, `"nan"`, `"NA"`, `""`)
to `NaN` in feature and target columns. The reserved column names `datasplit_group` and
`row_id` cannot appear in your data.

```python
import pandas as pd

df = pd.DataFrame({
    "sample_id": [1, 2, 3, 4, 5],
    "temperature": [22.1, 18.5, 25.3, 19.8, 23.7],
    "humidity": [0.65, 0.80, 0.55, 0.70, 0.60],
    "pressure": [1013, 1008, 1020, 1015, 1011],
    "yield": [85.2, 72.1, 91.4, 78.3, 88.6],  # continuous target
})
```

## Basic Usage

```python
from octopus.study import OctoRegression
from octopus.modules import Octo
from octopus.types import ModelName

study = OctoRegression(
    study_name="my_regression",
    target_metric="MAE",
    feature_cols=["temperature", "humidity", "pressure"],
    target_col="yield",
    sample_id_col="sample_id",
    workflow=[
        Octo(
            task_id=0,
            depends_on=None,
            description="regression_step",
            models=[ModelName.ExtraTreesRegressor],
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
| `target_metric` | Metric to optimize | `"RMSE"` |
| `sample_id_col` | Column identifying unique subjects (prevents correlated observation leakage) | `None` |
| `n_outer_splits` | Number of outer cross-validation splits | `5` |
| `single_outer_split` | Run only one split for quick testing (e.g., `0`) | `None` |
| `n_cpus` | Number of CPUs (`0` = all, `-1` = all but one) | `0` |

!!! tip
    If your dataset contains multiple rows per subject (e.g. longitudinal measurements,
    repeated experiments), set `sample_id_col` to the column identifying subjects.
    Octopus will ensure all rows from the same subject stay in the same split,
    preventing [information leakage](../concepts/nested_cv.md#what-is-information-leakage).

## Choosing a Metric

| Metric | Description | Direction | When to use |
|--------|-------------|-----------|-------------|
| `MAE` | Mean Absolute Error | Minimize | Default choice; easy to interpret in the target's units |
| `RMSE` | Root Mean Squared Error | Minimize | Penalizes large errors more than MAE |
| `MSE` | Mean Squared Error | Minimize | Same as RMSE but without the square root |
| `R2` | R² (coefficient of determination) | Maximize | Measures proportion of variance explained; 1.0 = perfect |

!!! tip
    `MAE` is robust and interpretable — an MAE of 5.0 means the model is off by 5 units
    on average. `R2` is useful when you want a normalized score between 0 and 1, but
    can be misleading on small datasets or when the target has low variance.

## Available Models

Octopus offers a broad range of regression models, from simple linear methods to
gradient boosting and neural networks:

| Model | Type | Default | Notes |
|-------|------|---------|-------|
| `ExtraTreesRegressor` | Tree ensemble | Yes | Fast, good baseline |
| `RandomForestRegressor` | Tree ensemble | Yes | Robust general-purpose model |
| `XGBRegressor` | Gradient boosting | Yes | Strong on tabular data |
| `CatBoostRegressor` | Gradient boosting | Yes | Native categorical support |
| `HistGradientBoostingRegressor` | Gradient boosting | Yes | Native categoricals, handles missing values |
| `ElasticNetRegressor` | Linear (regularized) | Yes | Combines L1 and L2 regularization |
| `GradientBoostingRegressor` | Gradient boosting | No | Sklearn implementation |
| `RidgeRegressor` | Linear (L2) | No | Simple regularized linear model |
| `ARDRegressor` | Bayesian linear | No | Automatic relevance determination |
| `SvrRegressor` | Support Vector | No | Kernel-based; best for small datasets |
| `GaussianProcessRegressor` | Kernel | No | Probabilistic predictions; small datasets only |
| `TabularNNRegressor` | Neural network | No | Embedding-based NN for mixed feature types |

Models marked as "Default" are included automatically when you don't specify a `models`
list in the `Octo` configuration.

!!! note
    Linear models (`ElasticNetRegressor`, `RidgeRegressor`, `ARDRegressor`, `SvrRegressor`,
    `LogisticRegressionClassifier`) apply a `StandardScaler` to features automatically.
    Tree-based models do not require scaling.

## Feature Importance

Feature importance methods work the same way as in classification:

- **`permutation`** — Permutation importance: shuffles each feature and measures the
  performance drop. Works with any model.
- **`shap`** — SHAP values: game-theoretic attributions. More detailed but slower.
- **`constant`** — Baseline constant importance (for reference).

```python
from octopus.types import FIComputeMethod

Octo(
    ...,
    fi_methods=[FIComputeMethod.PERMUTATION],
)
```

## Custom Hyperparameters

You can override the default hyperparameter search space for any model using the
`hyperparameters` parameter. This is useful when you have domain knowledge about
reasonable parameter ranges:

```python
from octopus.models.hyperparameter import IntHyperparameter, FloatHyperparameter

Octo(
    ...,
    models=[ModelName.RandomForestRegressor],
    hyperparameters={
        ModelName.RandomForestRegressor: [
            IntHyperparameter(name="max_depth", low=2, high=32),
            FloatHyperparameter(name="min_samples_split", low=0.01, high=0.5),
        ]
    },
)
```

See [Custom Hyperparameters](../examples/custom_hyperparameters.md) for a full example.

## Example Workflows

For runnable end-to-end examples, see:

- [Basic Regression](../examples/basic_regression.md): The simplest way to run a regression study using the diabetes dataset.
- [Custom Hyperparameters](../examples/custom_hyperparameters.md): Shows how to define your own hyperparameter search ranges instead of using the defaults.
- [Sequential Regression Workflow](../examples/sequential_regression_workflow.md): Chains Octo and MRMR into a multi-step pipeline that selects features before final model training.
