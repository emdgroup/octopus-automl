# Workflow & Modules

## Overview

Real-world datasets often contain many columns, but only a subset of them actually helps
a machine-learning model make accurate predictions. Finding that subset -- **feature selection** --
is a core goal of Octopus.

A **workflow** is an ordered list of **tasks** that are executed one after another.
Each task wraps a module, and each module either selects features, trains models, or both.
By chaining tasks together you build a pipeline that progressively narrows the feature set:
start with cheap, fast filters to discard obvious noise, then hand the reduced set to more
expensive methods for further refinement.

### Module types

Octopus ships two kinds of modules:

| Type | Purpose | Examples |
|------|---------|----------|
| **Feature Selection** | Reduce the number of features | [ROC](roc.md), [MRMR](mrmr.md), [RFE](rfe.md), [RFE2](rfe2.md), [SFS](sfs.md), [Boruta](boruta.md), [EFS](efs.md) |
| **Machine Learning** | Train models, optimize hyperparameters, and optionally select features | [Octo](octo.md), [AutoGluon](autogluon.md) |

Both types return a list of **selected features** that the next task in the workflow can consume.

### How tasks are connected

Every task has a `task_id` (starting at 0) and an optional `depends_on` parameter pointing to
the `task_id` of a prior task.

- The **first task** (`depends_on=None`) receives all columns listed in `feature_cols`.
- A **dependent task** (`depends_on=N`) receives only the features selected by task *N*,
  plus any scores, predictions, and feature-importance tables that task *N* produced.

### Example workflow

A typical three-step pipeline looks like this:

```
Task 0 (Octo)          all 30 features
        |
        v               selected_features (e.g. 20)
Task 1 (MRMR)          receives 20 features from Task 0
        |
        v               selected_features (e.g. 15)
Task 2 (Octo)          receives 15 features from Task 1
```

In Python this translates to:

```python
from octopus import OctoClassification
from octopus.modules import Mrmr, Octo

study = OctoClassification(
    ...,
    workflow=[
        Octo(
            task_id=0,
            depends_on=None,
            description="step1_octo_full",
            models=["ExtraTreesClassifier"],
            n_trials=100,
            n_folds_inner=5,
            max_features=30,
        ),
        Mrmr(
            task_id=1,
            depends_on=0,
            description="step2_mrmr",
            n_features=15,
        ),
        Octo(
            task_id=2,
            depends_on=1,
            description="step3_octo_reduced",
            models=["ExtraTreesClassifier"],
            n_trials=100,
            n_folds_inner=5,
            ensemble_selection=True,
        ),
    ],
)

study.fit(data=df)
```

!!! tip
    Ordering matters: tasks with `depends_on=None` must appear before tasks that reference
    them, and `task_id` values must form a contiguous sequence starting at 0.

---

## Feature Selection Modules

The table below lists all feature-selection modules roughly ordered from cheapest to most
expensive:

| Module | Wraps | Description |
|--------|-------|-------------|
| **[ROC](roc.md)** | scipy, networkx (custom) | Removes correlated features using graph-based grouping |
| **[MRMR](mrmr.md)** | Custom implementation | Maximum Relevance Minimum Redundancy filter |
| **[RFE](rfe.md)** | sklearn `RFECV` | Recursive Feature Elimination with cross-validation |
| **[RFE2](rfe2.md)** | Extends Octo (custom) | RFE using Octo's Optuna-based models |
| **[SFS](sfs.md)** | mlxtend / sklearn | Sequential forward/backward selection |
| **[Boruta](boruta.md)** | Custom (based on BorutaPy) | Shadow-feature statistical test |
| **[EFS](efs.md)** | Custom implementation | Ensemble of models on random feature subsets |

---

## Machine Learning Modules

| Module | Description |
|--------|-------------|
| **[Octo](octo.md)** | Core ML module with HPO, ensembling, and feature importance |
| **[AutoGluon](autogluon.md)** | AutoGluon TabularPredictor wrapper |
