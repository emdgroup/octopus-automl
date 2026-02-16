# AutoGluon

*Based on: [AutoGluon](https://github.com/autogluon/autogluon)*

AutoGluon wraps the [AutoGluon TabularPredictor](https://auto.gluon.ai/) to
provide fully automated model selection, hyperparameter tuning, and
stacking/ensembling within an Octopus workflow. Unlike Octo, which exposes
fine-grained control over optimization, AutoGluon aims for a hands-off
experience: you configure a quality preset and a time budget, and AutoGluon
handles the rest.

## How it works

1. **Initialize the TabularPredictor.** A `TabularPredictor` is created with the
   target column, evaluation metric (mapped from Octopus metric names to
   AutoGluon scorers), and verbosity level.

2. **Fit on training data.** AutoGluon's `fit()` method is called with the
   combined feature + target DataFrame. Internally, AutoGluon:
    - Performs automatic feature engineering (type inference, missing value
      handling, encoding).
    - Trains a portfolio of model types (controlled by `included_model_types` or
      the full default set).
    - Tunes hyperparameters using the strategy defined by the `presets`.
    - Builds multi-layer stacking ensembles when using higher-quality presets
      (`"good_quality"` and above).
    - Uses `n_bag_splits` for bagging/cross-validation within each model.

3. **Evaluate performance.** After training, the module evaluates on train, dev
   (out-of-split), and test partitions. Scores are computed using both
   AutoGluon's built-in metrics and Octopus's metric implementations for
   cross-comparison.

4. **Feature importance.** Permutation feature importance is computed on the test
   set using AutoGluon's `feature_importance()` method with confidence bands
   (15 shuffle sets, 95% confidence). If feature groups are defined, group-level
   importances are also calculated.

5. **Sklearn-compatible model.** The fitted AutoGluon predictor is wrapped in a
   sklearn-compatible class (`SklearnClassifier` or `SklearnRegressor`) so that
   downstream Octopus code (e.g., feature importance methods) can use it
   seamlessly.

6. **No feature selection.** AutoGluon does not perform feature selection -- it
   returns all input features. To select features, place AutoGluon after a
   feature-selection module in the workflow.

## Supported model types

When `included_model_types` is not set, AutoGluon considers all available
model families:

| Code | Model |
|------|-------|
| `GBM` | LightGBM |
| `CAT` | CatBoost |
| `XGB` | XGBoost |
| `RF` | Random Forest |
| `XT` | Extra Trees |
| `KNN` | K-Nearest Neighbors |
| `LR` | Linear/Logistic Regression |
| `NN_TORCH` | PyTorch Neural Network |
| `FASTAI` | FastAI Neural Network |

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `presets` | `["medium_quality"]` | Quality presets: `"best_quality"`, `"high_quality"`, `"good_quality"`, `"medium_quality"` |
| `time_limit` | `None` | Total training time in seconds |
| `infer_limit` | `None` | Per-row inference time limit in seconds |
| `n_bag_splits` | `5` | Bagging splits |
| `included_model_types` | `None` | Restrict to specific model types (see table above) |
| `memory_limit` | `"auto"` | Memory limit in GB |

## When to use

AutoGluon is ideal when:

- You want a **fully automated baseline** with minimal configuration effort.
- You want to **compare** Octo's manually-configured pipeline against an
  AutoML approach.
- You need access to model types not available in Octo (e.g., neural networks,
  KNN, linear models, LightGBM).
- Time-constrained scenarios where setting a `time_limit` and a `presets` level
  is sufficient.

## Limitations

- AutoGluon **does not perform feature selection**. All input features are passed
  through. Combine it with upstream feature-selection modules if needed.
- Requires the `autogluon` optional dependency (`pip install octopus[autogluon]`).
- Higher-quality presets (`"best_quality"`, `"high_quality"`) use multi-layer
  stacking which is memory-intensive and can be slow.
- The module integrates with Ray for resource management, which can conflict with
  Octo's own Ray usage if not configured carefully.

## Examples

- [Octo & AutoGluon](../../examples/wf_octo_autogluon.md) — runs Octo and AutoGluon side by side on the same dataset.
