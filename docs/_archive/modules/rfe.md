# RFE -- Recursive Feature Elimination


RFE is a wrapper-based feature selection method that wraps sklearn's `RFECV`. It
repeatedly trains a model, ranks features by their internal importance scores,
removes the least important ones, and uses cross-validation to identify the
feature set that gives the best performance.

## How it works

1. **Hyperparameter optimization.** A `GridSearchCV` is run on the full feature
   set to find the best hyperparameters for the chosen model (CatBoost, XGBoost,
   RandomForest, or ExtraTrees). This ensures the model used for feature ranking
   is well-tuned.

2. **Recursive elimination with cross-validation.** sklearn's `RFECV` is
   executed using the tuned model:
    - At each step, the model is trained and features are ranked by their
      `feature_importances_`.
    - The `step` least-important features are removed.
    - Cross-validated performance is recorded at each feature count.
    - The process continues until `min_features_to_select` features remain.

3. **Select the optimal feature set.** RFECV picks the number of features that
   maximized the CV score. The module additionally refits the selected features
   with a fresh grid search to report both a refit test score and a
   grid-search + refit test score.

4. **Return results.** The selected feature names, performance scores (dev CV,
   test refit, test grid-search refit), and internal feature importances from
   the final model are returned.

The module supports two modes via the `mode` parameter:

- **Mode1** (default): Uses the already-optimized best model for RFECV. Faster,
  since hyperparameters are fixed throughout elimination.
- **Mode2**: Passes the full `GridSearchCV` object to RFECV, re-optimizing
  hyperparameters at every elimination step. More thorough but significantly
  slower.

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `""` (auto) | Model name -- `CatBoost`, `XGB`, `RandomForest`, or `ExtraTrees` |
| `step` | `1` | Features removed per iteration |
| `min_features_to_select` | `1` | Minimum features to keep |
| `cv` | `5` | Cross-validation folds |
| `mode` | `"Mode1"` | `"Mode1"` (fixed HP) or `"Mode2"` (re-optimize HP each step) |

## When to use

RFE is a good choice when you want a straightforward, model-driven wrapper
method and the feature count is moderate (up to a few hundred). It is
particularly effective when feature importances from tree-based models are
reliable indicators of true predictive value.

For large feature sets, consider running ROC or MRMR first to reduce the
dimensionality, then applying RFE on the reduced set.

## Limitations

- Computational cost grows with feature count and `step=1` is expensive for
  high-dimensional datasets. Increase `step` to speed things up at the cost of
  granularity.
- Relies on the model's internal `feature_importances_`, which can be biased for
  certain model types (e.g., favouring high-cardinality features in tree-based
  models).
- Does not support time-to-event targets.
