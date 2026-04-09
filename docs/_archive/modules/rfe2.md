# RFE2 -- RFE with Octo Optimization


RFE2 combines the iterative elimination strategy of RFE with Octo's
Optuna-powered hyperparameter optimization. Instead of relying on a simple grid
search, RFE2 first runs a full Octo optimization to produce a high-quality model
ensemble, then progressively removes the least important features while
retraining the ensemble at each step. This yields more reliable importance
estimates than standard RFE, at the cost of significantly more compute.

## How it works

RFE2 extends the `Octo` class and inherits all of its configuration and
optimization machinery.

1. **Run a full Octo optimization.** The module starts by executing the standard
   Octo pipeline: Optuna-based hyperparameter search, inner cross-validation,
   and construction of the best model bag. Feature importances (permutation or
   SHAP) are computed on the resulting ensemble.

2. **Record baseline performance.** The dev-set mean performance and standard
   error of the best bag are stored as step 0 of the RFE process.

3. **Iterative feature elimination.** At each subsequent step:
    - Feature importances are extracted from the current bag.
    - The feature with the lowest importance (absolute or signed, depending on
      `abs_on_fi`) is identified.
    - If the feature is a group feature, all constituent features of the group
      are removed together.
    - The bag is retrained on the reduced feature set and new importances are
      computed.
    - Performance (mean and SEM) is recorded.
    - The loop continues until fewer than `min_features_to_select` features
      would remain.

4. **Select the final solution.** Two strategies are available:
    - **`"best"`** (default): Pick the step with the highest mean dev performance.
    - **`"parsimonious"`**: Pick the step with the fewest features whose
      performance is still within one SEM of the best. This favours simpler
      models when the performance difference is not statistically significant.

5. **Return results.** The selected features, the corresponding model bag,
   scores, predictions, and feature importances are returned.

## Key parameters

All Octo parameters are inherited. The following are specific to RFE2:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_features_to_select` | `1` | Minimum features to retain |
| `fi_method_rfe` | `"permutation"` | `"permutation"` or `"shap"` -- importance method used for elimination |
| `selection_method` | `"best"` | `"best"` (highest score) or `"parsimonious"` (fewest features within one SEM of best) |
| `abs_on_fi` | `False` | Take absolute value of importance scores before ranking |

## When to use

RFE2 is appropriate when you want the most accurate feature-elimination process
possible and can afford the compute budget. It is particularly valuable when:

- Feature importances from a simpler model (like those used by standard RFE) are
  unreliable.
- You want to leverage Octo's Optuna-based optimization for the underlying model
  at every elimination step.
- The `"parsimonious"` selection mode is desirable to find the most compact
  feature set without significant performance loss.

## Limitations

- Very expensive: every RFE step retrains the full Octo bag ensemble.
- The `"parsimonious"` selection depends on the SEM estimate, which may be noisy
  with few inner folds.
- Inherits all Octo limitations (e.g., supported model types, parallelization
  constraints).
