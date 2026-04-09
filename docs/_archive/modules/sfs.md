# SFS -- Sequential Feature Selection


SFS is a wrapper-based feature selection method that evaluates subsets of
features by adding or removing one feature at a time. At each step it measures
the cross-validated performance of every candidate change and picks the one that
improves the score the most. This exhaustive, stepwise search produces
high-quality subsets but is computationally expensive when many features are
involved.

## How it works

1. **Hyperparameter optimization.** A `GridSearchCV` tunes the chosen model
   (CatBoost, XGBoost, RandomForest, or ExtraTrees) on the full feature set to
   obtain well-calibrated hyperparameters.

2. **Sequential selection.** The module uses mlxtend's `SequentialFeatureSelector`
   (SFS) with the tuned model. Depending on `sfs_type`, the behaviour differs:

    - **`"forward"`**: Start with an empty set. At each step, add the feature
      that gives the best CV score.
    - **`"backward"`**: Start with all features. At each step, remove the
      feature whose removal hurts the least (or improves performance).
    - **`"floating_forward"`**: Like forward, but after each addition, try
      removing previously added features if that improves the score (SFFS
      algorithm).
    - **`"floating_backward"`**: Like backward, but after each removal, try
      re-adding previously removed features (SBFS algorithm).

    The selector uses `k_features="best"`, meaning it evaluates all possible
    feature counts and returns the set with the highest CV score.

3. **Post-selection evaluation.** The selected features are evaluated in two
   ways:
    - **Refit**: The best model is retrained on the selected features and
      evaluated on the test set.
    - **Grid search + refit**: A fresh grid search is run on the selected
      features, then the best model is retrained and evaluated on the test set.

4. **Return results.** Selected feature names, dev/test scores, and internal
   feature importances from the final model are returned.

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `""` (auto) | Model name -- `CatBoost`, `XGB`, `RandomForest`, or `ExtraTrees` |
| `cv` | `5` | Cross-validation folds |
| `sfs_type` | `"backward"` | `"forward"`, `"backward"`, `"floating_forward"`, or `"floating_backward"` |

## When to use

SFS is best suited for situations where:

- The feature count is moderate (tens of features, ideally after a prior
  reduction step like ROC or MRMR).
- You want a thorough, exhaustive search over feature subsets rather than a
  ranking-based heuristic.
- The floating variants (SFFS/SBFS) are useful when features interact and a
  purely monotonic forward or backward path would miss the optimal combination.

## Limitations

- **Computational cost** scales as O(*n* x *k*) per step, where *n* is the
  number of candidate features and *k* is the number of CV folds. This makes
  SFS impractical for hundreds of features without prior reduction.
- The greedy / floating strategy does not guarantee a globally optimal subset.
- Does not support time-to-event targets.
