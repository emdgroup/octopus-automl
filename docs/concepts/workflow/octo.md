# Octo


Octo is the core ML module of Octopus. It combines Optuna-based hyperparameter
optimization, cross-validated model training, optional ensemble selection, and
multiple feature-importance methods into a single task. Most workflows begin and
end with an Octo task -- the first to establish a baseline and produce feature
importances, the last to train a final model on the refined feature set.

## How it works

### Hyperparameter optimization

1. **Inner cross-validation setup.** The train+dev data is divided into inner
   splits (controlled by `n_inner_splits` and `inner_split_seeds`). Each
   Optuna trial trains a [Bag](../terminology.md#bag) of models, one per
   inner split, and evaluates them on the held-out dev splits. See
   [Nested Cross-Validation](../nested_cv.md) for the full picture.

2. **Optuna optimization.** A TPE (Tree-structured Parzen Estimator) sampler
   explores the hyperparameter space over `n_trials` trials. The first
   `n_startup_trials` use random sampling; the rest use multivariate TPE
   with grouping and constant-liar parallelism. The optimization target is
   either the *combined* or *averaged* dev-set performance across inner splits
   (controlled by `scoring_method`).

3. **MRMR feature subsets (optional).** When `n_mrmr_features` is set, Octo
   pre-computes MRMR feature subsets of various sizes. Optuna can then sample
   from these subsets during optimization, effectively searching over both
   hyperparameters and feature counts simultaneously.

4. **Constrained HPO (optional).** When `max_features > 0`, the optimization
   penalizes trials that use more features than the constraint. The
   `penalty_factor` controls how aggressively excess features are penalized. Only
   models flagged as `chpo_compatible` support this mode.

### Best bag construction

5. **Build the best bag.** After optimization, the best trial's hyperparameters
   are used to train a fresh bag of models (one per inner split) on the full
   train+dev data. This "best bag" is the primary output model.

6. **Feature importance calculation.** Feature importances are computed on the
   best bag using the methods specified in `fi_methods`:
    - **`"permutation"`**: Permutation importance on the dev partition.
    - **`"shap"`**: SHAP values on the dev partition.
    - **`"constant"`**: A baseline method that returns equal importance for all
      features.

7. **Feature selection.** Features are selected based on the computed
   importances, typically those with positive permutation importance.

### Ensemble selection (optional)

8. **Ensemble selection.** When `ensemble_selection=True`, the top
   `n_ensemble_candidates` trial bags are used as candidates. An ensemble
   optimization procedure (hill-climbing with replacement) finds the combination
   of trial bags that maximizes dev-set performance. The resulting ensemble bag
   replaces the best bag as the primary output.

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `models` | `["ExtraTreesClassifier"]` | Models to train (e.g., `ExtraTrees`, `RandomForest`, `XGB`, `CatBoost`) |
| `n_trials` | `200` | Number of Optuna hyperparameter optimization trials |
| `n_inner_splits` | `5` | Inner cross-validation splits |
| `inner_split_seeds` | `[0]` | Seeds for inner splits; more seeds = more robust |
| `max_features` | `0` | Constrain maximum features during HPO (0 = no constraint) |
| `penalty_factor` | `1.0` | Penalty for exceeding `max_features` |
| `ensemble_selection` | `False` | Enable ensemble selection over top trials |
| `n_ensemble_candidates` | `50` | Number of top trials saved for ensemble selection |
| `fi_methods` | `["permutation"]` | Feature importance methods: `"permutation"`, `"shap"`, `"constant"` |
| `n_startup_trials` | `15` | Random trials before TPE sampler kicks in |
| `max_outliers` | `3` | Maximum outlier samples to optimize/remove |
| `n_mrmr_features` | `[]` | Feature counts for integrated MRMR feature selection |
| `scoring_method` | `"combined"` | Bag performance mode: `"combined"` or `"average"` |

## When to use

Octo is the workhorse of Octopus and should be used:

- As the **first task** to get a baseline and feature importances that downstream
  modules (e.g., MRMR) can consume.
- As the **last task** to train a final model on a refined feature set.
- When you need **ensemble selection** over multiple Optuna trials for maximum
  performance.
- When you want **constrained HPO** to limit the number of features used during
  optimization.

## Limitations

- Computationally expensive: `n_trials` x `n_inner_splits` model fits, plus
  feature importance computation.
- The constrained HPO mode requires models with `chpo_compatible=True` in their
  model configuration.
