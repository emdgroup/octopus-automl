# Understanding the Results

This page explains how to interpret the predictions and metrics that Octopus
produces. For a detailed description of the output files and directory layout,
see [Understanding the Output](../userguide/output_structure.md).

## Performance estimates are averaged across outer splits

Every performance number Octopus reports is the **mean of the outer test-set
scores** across all outer splits. Because each outer test set was never seen
during training or hyperparameter tuning, this average is an unbiased estimate
of generalization performance.

For a 5 × 5 nested CV study the pipeline fits models on five different 80/20
splits. Each split produces an independent test-set score. The final reported
metric is the average of these five scores, and you can also inspect the
per-split values to assess variance.

See [Nested Cross-Validation](nested_cv.md) for the full explanation of why
this matters.

## Every sample gets a test prediction

Across all outer splits, every sample in the dataset serves as a test sample
exactly once. This means Octopus produces a **test prediction for every row**
in your data — not just a subset. These predictions are collected in the
`predictions.parquet` files inside each outer split and are aggregated by
[`StudyDiagnostics`](../userguide/output_structure.md#studydiagnostics).

For classification, each prediction includes:

- **Predicted probabilities** (`pred_proba_0`, `pred_proba_1`, …) — the
  ensemble's probability estimate for each class.
- **Predicted class** (`pred_class`) — the class with the highest probability.

For regression, each prediction includes a single `prediction` column.

## Bags and ensembles

A [Bag](terminology.md#bag) is an ensemble of models trained on different
inner splits with the same hyperparameters. When making a prediction, the Bag
averages the outputs of all its inner models
([Trainings](terminology.md#training)). This averaging smooths out noise from
any single train/validation split.

When `ensemble_selection=True` is set, multiple Bags from different Optuna
trials are further combined into a meta-ensemble. The ensemble selection
procedure greedily picks the combination of Bags that maximizes dev-set
performance.

Both the `best` (single-trial) and `ensemble_selection` results are saved
separately in the output directory. The ensemble result, when present, is
typically the stronger model.

## Feature importances

Each [Octo](workflow/octo.md) task computes feature importances using the
methods specified in `fi_methods` (default: permutation). Importances are
computed per outer split and per inner split, then aggregated.

Use the feature importance values to understand which features drive
predictions. Features with consistently high importance across outer splits are
the most reliable. See [Feature Importance](feature_importance.md) for details
on the available methods.

## Multi-task workflows

In a multi-step [workflow](workflow/index.md), each task produces its own
results: predictions, scores, and feature importances. The downstream task
operates on a reduced feature set. When inspecting results, pay attention to
which task you are looking at — the final task's metrics reflect the full
pipeline, while earlier tasks show intermediate results.

## What to look at first

1. **Target metric** — check the mean outer test score for your chosen metric
   (e.g., AUCROC, MAE). This is the single best summary of your model's
   expected real-world performance.
2. **Per-split variance** — large differences across outer splits suggest
   instability, often caused by very small datasets or class imbalance.
3. **Feature importances** — which features matter most? Are they scientifically
   plausible?
