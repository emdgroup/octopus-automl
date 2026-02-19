# TaskPredictor Concept — Feature Importance Specification

**Parent document:** [01_overview.md](01_overview.md)

---

## 5. `octopus/predict/feature_importance.py` — FI Calculations

The feature importance code is **adapted from `octopus/modules/utils.py` on the main branch**, which provides:
- **Permutation FI with p-values** and confidence intervals (not sklearn's `permutation_importance`)
- **Group permutation FI** (permute groups of features together)
- **SHAP FI** (via the `shap` library)
- **Group SHAP FI**

### 5.1 What Changes from the Main Branch Version

The main branch FI code is tightly coupled to `ExperimentInfo`. The decoupled version:

| Main branch (modules/utils.py) | Proposed (predict/feature_importance.py) |
|--------------------------------|------------------------------------------|
| Takes `ExperimentInfo` as input | Takes `model`, `data`, `feature_cols`, `target_metric`, `target_assignments`, `positive_class` directly |
| Imports `ExperimentInfo`, `BaseModel` | No imports from modules/models |
| Uses `experiment.model`, `experiment.feature_cols`, etc. | Uses function parameters directly |
| Uses `experiment.data_traindev` for shuffling pool | Takes `data_pool` parameter (or uses `data` itself) |
| Uses `get_score_from_model()` from `octopus.metrics.utils` | Uses private copy `_get_score_from_model()` — copied into this file to avoid `octopus/__init__.py` import chain (see `06_implementation.md` §10.0, §10.2) |

### 5.2 Decoupled FI Functions

**`calculate_permutation_fi(model, data, data_pool, feature_cols, target_metric, target_assignments, positive_class, n_repeat)`**

Adapted from main branch's `get_fi_permutation()`. Key algorithm:
1. Calculate baseline score using `_get_score_from_model()` (private copy)
2. For each feature, repeat `n_repeat` times: shuffle that feature's values (sampling from `data_pool`), re-score
3. Importance = baseline_score − shuffled_score
4. Calculate **mean, stddev, p-value** (one-sided t-test), **95% confidence intervals**
5. Returns DataFrame with columns: `feature`, `importance`, `stddev`, `p-value`, `n`, `ci_low_95`, `ci_high_95`

**`data_pool` parameter:** The pool of values used for shuffling. In the main branch this is `pd.concat([data_traindev, data_test])`. In `TaskPredictor`, this is built from `data_train.parquet` combined with the evaluation data (see Q6 in `06_implementation.md`). Note: `data_train.parquet` on this branch is equivalent to `data_traindev` on the main branch — it contains all non-test data for the outersplit (train + dev combined, before inner split).

**`calculate_group_permutation_fi(model, data, data_pool, feature_cols, target_metric, target_assignments, positive_class, feature_groups, n_repeat)`**

Adapted from main branch's `get_fi_group_permutation()`. Same algorithm but permutes groups of features together. `feature_groups` is a `dict[str, list[str]]` mapping group names to feature lists.

**`calculate_shap_fi(model, data, feature_cols, ml_type, shap_type)`**

Adapted from main branch's `get_fi_shap()`. Uses the `shap` library. No `data_pool` needed — SHAP computes values directly on `data`.

**`calculate_group_shap_fi(model, data, feature_cols, ml_type, feature_groups, shap_type)`**

Adapted from main branch's `get_fi_group_shap()`. Computes SHAP values and aggregates by feature group.

### 5.3 Key Difference from sklearn's `permutation_importance`

The Octopus permutation FI implementation is **superior to sklearn's** for our use case because:
- Uses `_get_score_from_model()` (private copy from `octopus.metrics.utils`) which supports all Octopus metrics (AUCROC, ACCBAL, R2, etc.)
- Calculates **p-values** via one-sided t-test — tells you if a feature is statistically significant
- Calculates **95% confidence intervals** — gives uncertainty bounds on importance
- Shuffling pool is `data_all` (train+test) to avoid small-sample issues
- Returns a richer DataFrame with statistical information
