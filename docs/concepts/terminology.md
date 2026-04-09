# Terminology

This page defines key terms used throughout the Octopus documentation.

## Outer Split

One iteration of the outer cross-validation loop. The dataset is divided into
a **train+dev set** (used for feature selection and hyperparameter tuning) and
a **test set** (used only for final, unbiased evaluation). With the default
`n_outer_splits=5`, five outer splits are created, each holding out a
different 20 % of the data as the test set.

See [Nested Cross-Validation](nested_cv.md) for full details.

## Inner Split

One iteration of the inner cross-validation loop, operating _within_ a single
outer split's train+dev set. The train+dev data is further divided into a
**training set** and a **dev set** (validation). Inner splits are used
exclusively for hyperparameter tuning; the outer test set is never seen.

## Bag

A Bag is a collection of [Trainings](#training), one per inner split. When
making predictions the Bag averages the outputs of all its Trainings, forming
a within-split ensemble.

Each Optuna trial produces one Bag. After optimization the best trial's
hyperparameters are used to build the **best Bag**, which becomes the primary
output model for that outer split.

```
Outer Split 0
  └─ Best Bag
       ├─ Training 0  (fit on inner split 0 train, evaluated on split 0 dev)
       ├─ Training 1  (fit on inner split 1 train, evaluated on split 1 dev)
       └─ Training 2  (fit on inner split 2 train, evaluated on split 2 dev)
```

When `ensemble_selection=True` is set on the [Octo](workflow/octo.md) task,
multiple Bags from different Optuna trials are combined into a larger
meta-ensemble (see [Ensemble Selection](#ensemble-selection) below).

## Training

A Training is a single fitted model for one inner split. It holds the trained
scikit-learn pipeline, the hyperparameters used, and predictions on the train,
dev, and test partitions. Trainings are never used in isolation; they always
live inside a [Bag](#bag) and contribute to its ensemble predictions.

## Ensemble Selection

An optional post-processing step where the top-performing Bags from Optuna
trials are greedily combined into a meta-ensemble. The selection procedure
(hill-climbing with replacement) picks the subset of Bags that maximizes dev
performance. The resulting ensemble Bag replaces the best single-trial Bag as
the primary model for that outer split.

Enabled by setting `ensemble_selection=True` on the Octo task.
