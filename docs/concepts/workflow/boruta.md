# Boruta -- Shadow-Feature Statistical Test


Boruta is a statistically principled, "all-relevant" feature selection method.
Unlike most other modules that select a fixed-size subset, Boruta asks a
different question: *which features are genuinely more important than random
noise?* It answers this by creating "shadow" copies of every feature, training a
model on both real and shadow features, and using a statistical test to decide
which real features carry true signal.

## How it works

1. **Hyperparameter optimization.** A `GridSearchCV` tunes the tree-based model
   (RandomForest, ExtraTrees, or XGBoost) on the full feature set. Only
   tree-based models are supported because Boruta relies on
   `feature_importances_` from the trained model.

2. **Shadow feature generation.** For every real feature, a "shadow" copy is
   created by randomly permuting its values across samples. This destroys any
   relationship with the target while preserving the marginal distribution.

3. **Iterative importance comparison.** Over multiple rounds:
    - A model is trained on the combined real + shadow feature set.
    - The maximum importance among all shadow features in this round is recorded
      (the "shadow max").
    - Each real feature's importance is compared to the shadow max.
    - A hit counter tracks how often each real feature exceeds the shadow max.

4. **Statistical testing.** After all rounds, a binomial test (with Bonferroni
   correction for multiple testing) is applied to each real feature's hit count:
    - **Confirmed**: The feature is significantly more important than random
      noise at the `alpha` significance level.
    - **Tentative**: The evidence is inconclusive.
    - **Rejected**: The feature is not significantly better than noise.

    Only *Confirmed* features are returned.

5. **Post-selection evaluation.** The selected features are evaluated on dev
   (cross-validated) and test sets using both a refit and a grid-search + refit
   strategy, matching the pattern used by RFE and SFS.

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `""` (auto) | Tree-based model only (`RandomForest`, `ExtraTrees`, or `XGB`) |
| `cv` | `5` | Cross-validation splits for hyperparameter tuning |
| `perc` | `100` | Percentile threshold for shadow-feature comparison (100 = max shadow importance) |
| `alpha` | `0.05` | Significance level for the statistical test |

## When to use

Boruta is particularly well-suited when:

- You want to find **all relevant features** rather than a fixed-size subset.
  This is valuable for interpretability or when downstream models benefit from
  having every informative feature available.
- The dataset has many noise features and you want a principled way to separate
  signal from noise.
- You are uncertain about how many features to keep and prefer letting a
  statistical test decide.

## Limitations

- Only supports tree-based models (RandomForest, ExtraTrees, XGBoost). CatBoost
  is not supported because the BorutaPy implementation requires sklearn-style
  `feature_importances_`.
- Runtime grows with the number of features (shadow features double the feature
  space) and the number of Boruta iterations.
- The `perc` parameter (percentile of shadow importances) can affect sensitivity:
  lowering it below 100 makes the test more conservative.
- Does not support time-to-event targets.
