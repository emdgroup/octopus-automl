# TaskPredictor Concept — Implementation Plan, Comparison & Open Questions

**Parent document:** [01_overview.md](01_overview.md)

---

## 8. Comparison: Main Branch vs. Proposed

| Aspect | Main branch `OctoPredict` | Proposed `TaskPredictor` |
|--------|--------------------------|--------------------------|
| **Data loading** | `OctoExperiment.from_pickle()` — monolithic pickle | `study_io` functions — JSON/Parquet/joblib |
| **Model access** | Via `ExperimentInfo` (depends on `BaseModel`) | Direct `joblib.load()` — no intermediate class |
| **FI calculation** | `get_fi_permutation()` etc. from `modules/utils.py` | Same algorithms, decoupled in `predict/feature_importance.py` |
| **FI p-values** | ✓ (one-sided t-test) | ✓ (same algorithm) |
| **FI confidence intervals** | ✓ (95% CI) | ✓ (same algorithm) |
| **Group permutation FI** | ✓ | ✓ |
| **SHAP FI** | ✓ | ✓ |
| **Ensemble FI** | ✓ (passes `self` as model) | ✓ (same pattern) |
| **Dependencies** | `OctoExperiment`, `ExperimentInfo`, `BaseModel`, `modules/utils` | `pandas`, `numpy`, `joblib`, `json`, `upath`, `scipy`, `shap` (zero `octopus.*` imports) |
| **Version stable?** | No — breaks when execution code changes | Yes — reads only stable file formats |

---

## 9. Usage Examples

### 9.1 Predict on New Data

```python
from octopus.predict import TaskPredictor
import pandas as pd

predictor = TaskPredictor("./studies/my_study/", task_id=0)

new_data = pd.read_csv("new_samples.csv")
predictions = predictor.predict(new_data)              # np.ndarray, shape (n_samples,)
predictions_df = predictor.predict(new_data, df=True)  # pd.DataFrame with sample index

probabilities = predictor.predict_proba(new_data)              # np.ndarray, shape (n_samples, n_classes)
probabilities_df = predictor.predict_proba(new_data, df=True)  # pd.DataFrame with class columns
```

### 9.2 Predict on Test Data

```python
predictor = TaskPredictor("./studies/my_study/", task_id=0)

test_preds = predictor.predict_test()           # np.ndarray (default)
test_preds_df = predictor.predict_test(df=True) # DataFrame: row_id, prediction, prediction_std, n

test_proba = predictor.predict_proba_test()           # np.ndarray (default)
test_proba_df = predictor.predict_proba_test(df=True) # DataFrame: row_id, probability, probability_std, n
```

### 9.3 Feature Importance on New Data

```python
predictor = TaskPredictor("./studies/my_study/", task_id=0)

# Permutation FI with p-values
fi_results = predictor.calculate_fi(new_data, fi_type="permutation", n_repeat=10)

# Per-outersplit results
fi_exp0 = fi_results["fi_table_permutation_exp0"]
# Columns: feature, importance, stddev, p-value, n, ci_low_95, ci_high_95

# Ensemble result (using TaskPredictor as model)
fi_ensemble = fi_results["fi_table_permutation_ensemble"]
```

### 9.4 Feature Importance on Test Data

```python
predictor = TaskPredictor("./studies/my_study/", task_id=0)

# Uses saved test data automatically
fi_results = predictor.calculate_fi(fi_type="permutation", n_repeat=10)
```

### 9.5 SHAP Feature Importance

```python
predictor = TaskPredictor("./studies/my_study/", task_id=0)

fi_results = predictor.calculate_fi(new_data, fi_type="shap", shap_type="kernel")
```

### 9.6 Analysis Notebook (via convenience functions)

```python
from octopus.analysis import plot_aucroc, show_confusionmatrix, testset_performance_overview

# These functions internally create a TaskPredictor
testset_performance_overview("./studies/my_study/", task_id=0, metrics=["AUCROC", "ACCBAL"])
plot_aucroc("./studies/my_study/", task_id=0)
show_confusionmatrix("./studies/my_study/", task_id=0, threshold=0.5)
```

---

## 10. Version Stability Analysis

### 10.0 `octopus/__init__.py` Import Chain

**Problem discovered:** Importing _any_ `octopus.*` submodule triggers `octopus/__init__.py`, which contains:

```python
from octopus.study import OctoClassification, OctoRegression, OctoTimeToEvent
```

This transitively pulls in the **entire octopus codebase**: `octopus.study`, `octopus.manager`, `octopus.modules`, `octopus.models`, `octopus.datasplit`, and all their dependencies (Optuna, Ray, attrs, etc.).

**Impact:** `from octopus.metrics.utils import get_score_from_model` loads **60+ octopus modules** including the full execution stack. Any change to any module could break `octopus.predict` at import time.

**Chosen approach: Private copy of scoring functions.**

`get_score_from_model()` and `get_performance_from_model()` are copied into `predict/feature_importance.py` as private functions (`_get_score_from_model`, `_get_performance_from_model`). This gives `octopus/predict/` **zero imports from any `octopus.*` module** — complete isolation regardless of `__init__.py` behavior.

The copied functions depend on `Metrics.get_instance()` from `octopus.metrics.core`, which in turn uses the `Metric` dataclass from `octopus.metrics.config`. When copying, these dependencies are inlined: the relevant metric lookup logic (metric function, direction, ml_type, prediction_type) is included directly in the copied functions, using only `sklearn` metric functions and standard library code.

#### 10.0.1 Future Alternative: Lazy `__init__.py`

If `octopus/__init__.py` were fixed with lazy imports (`__getattr__`), `predict/` could import directly from `octopus.metrics.utils` instead of using private copies. This was empirically verified:

| Scenario | Modules loaded | What loads |
|----------|---------------|------------|
| **Current (eager `__init__.py`)** | **60+** | Entire codebase: study, manager, modules, models, datasplit, Optuna, Ray, etc. |
| **With lazy `__init__.py`** | **20** | Only: `metrics/*` (7), `models/*` (8), `exceptions` (1), helpers (4) |

The 20 modules loaded with lazy init are all **stable definition code**:
- `octopus.metrics.*` — metric registry + definitions (AUCROC, R2, etc.). Only added, never removed.
- `octopus.models.*` — model configs + type aliases (`MLType`, `PredType`). Stable types.
- `octopus.exceptions` — simple exception classes.

**NOT loaded with lazy init:** `octopus.study`, `octopus.manager`, `octopus.modules`, `octopus.datasplit`, `octopus.logger`, `octopus.utils`, `octopus._optional` — the entire execution stack is excluded.

**To switch to direct import in the future:**
1. Fix `octopus/__init__.py`: replace eager `from octopus.study import ...` with `__getattr__` lazy loading
2. Optionally fix `octopus/models/__init__.py`: lazy-load model definitions (reduces 20 → ~12 modules)
3. Replace private copies in `predict/feature_importance.py` with `from octopus.metrics.utils import get_score_from_model, get_performance_from_model`
4. Remove the inlined metric lookup logic

This is the cleaner long-term approach (single source of truth, no duplication), but requires the lazy init prerequisite. The private copy approach works today without any changes to existing code.

### 10.1 Dependencies of `octopus/predict/`

| Import | Purpose | Stability risk |
|--------|---------|----------------|
| `pandas` | DataFrames, Parquet I/O | Very stable |
| `numpy` | Arrays | Very stable |
| `joblib` | Load `model.joblib` | Very stable |
| `json` | Load `config.json`, `predictor.json` | Stdlib — always stable |
| `upath` | File paths | Stable |
| `scipy.stats` | t-test for p-values in FI | Very stable |
| `shap` | SHAP explainers | Stable public API |
| `sklearn.metrics` | Metric functions (roc_auc_score, etc.) used by private scoring copies | Very stable |

**NOT imported by `octopus/predict/`:** Any `octopus.*` module. Zero internal dependencies. Complete isolation.

### 10.2 Private Scoring Function Copies

`octopus/predict/feature_importance.py` contains private copies of two functions from `octopus/metrics/utils.py`:

| Original (metrics/utils.py) | Private copy (predict/feature_importance.py) | Used by |
|------------------------------|----------------------------------------------|---------|
| `get_score_from_model()` | `_get_score_from_model()` | Permutation FI |
| `get_performance_from_model()` | `_get_performance_from_model()` | FI analysis |

**Why copies instead of imports:** The `octopus/__init__.py` import chain problem (§10.0) means importing from `octopus.metrics` loads 60+ modules. Private copies give `predict/` zero internal dependencies.

**Keeping copies in sync:** If `get_score_from_model` or `get_performance_from_model` are modified in `octopus/metrics/utils.py`, the private copies must be updated to match. These functions are stable (last changed rarely), so the maintenance burden is low.

**Signature contract (must match between original and copy):**

| Function | Signature |
|----------|-----------|
| `get_score_from_model()` | `(model, data, feature_cols, target_metric, target_assignments, positive_class) → float` |
| `get_performance_from_model()` | `(model, data, feature_cols, target_metric, target_assignments, threshold, positive_class) → float` |

**Future: can be replaced with direct imports** if `octopus/__init__.py` is fixed with lazy loading (see §10.0.1).

### 10.3 Model Pickle Compatibility

Models are loaded via `joblib.load()`. The deserialized object (e.g., `BagClassifier`) must be importable from the same module path at load time. If `octopus.modules.octo.bag.BagClassifier` is renamed or moved, old models won't load.

**Mitigations:**
- Keep stable import paths for model classes used in `joblib.dump()`
- Or: register module aliases (`sys.modules`) for backward compatibility
- Or: use custom unpickler that maps old paths to new paths

### 10.4 File Format Stability

| File | Format | Rule |
|------|--------|------|
| `config.json` | JSON | Add new keys freely; never remove/rename existing keys |
| `predictor.json` | JSON | Simple: `{"selected_features": [...]}` |
| `feature_cols.json` | JSON | `["feat_a", "feat_b", ...]` — input features to the task. Must be saved during execution |
| `feature_groups.json` | JSON | `{"group0": ["feat_a", "feat_b"], ...}` — may not exist in old studies |
| `model.joblib` | joblib/pickle | Stable as long as model class path is stable |
| `data_test.parquet` | Parquet | Add columns freely; never remove `row_id`, target columns |

---

## 11. Implementation Plan

| Phase | Action | Effort |
|-------|--------|--------|
| 1 | Create `octopus/predict/study_io.py` with all I/O functions (including `load_model()`, `load_selected_features()`, `load_feature_cols()`) | Small |
| 2 | Replace model saving in `octopus/modules/base.py`: instead of `Predictor.save()`, use direct `joblib.dump()` + JSON writes. No shared class between modules/ and predict/ | Small |
| 2b | Save `feature_cols.json` alongside `selected_features.json` in `workflow_runner._save_task_results()` — persists the input feature columns available to the task | Small |
| 3 | Remove `octopus/modules/predictor.py` entirely | Small |
| 4 | Create `octopus/predict/feature_importance.py` — adapt FI functions from main branch `modules/utils.py`, decouple from `ExperimentInfo` | Medium |
| 5 | Create `octopus/predict/task_predictor.py` — constructor loads models/features directly via `study_io`, inlines feature subsetting. Properties, `predict()`, `predict_proba()` | Medium |
| 6 | Add `predict_test()`, `predict_proba_test()` | Small |
| 7 | Add `calculate_fi()` — per-experiment + ensemble FI | Medium |
| 8 | Create `octopus/predict/__init__.py` — exports `TaskPredictor` only | Small |
| 9 | Update `octopus/analysis/notebook_utils.py` to use `TaskPredictor` internally | Medium |
| 10 | Remove imports from `octopus.modules.utils` in `analysis/__init__.py` | Small |
| 11 | Deprecate/remove `analysis/loaders.py` and `analysis/module_loader.py` | Small |
| 12 | Update tests (remove `test_predictor.py`, add `test_task_predictor.py`) | Medium |
| 13 | Update example notebooks | Medium |

---

## 12. Open Questions

### Resolved

1. **~~Should `TaskPredictor` also support non-ML tasks (feature selection only)?~~**  
   **Decision: No.** `TaskPredictor` only handles tasks that produce a model. Feature-selection-only tasks (whose only output is `selected_features.json`) are out of scope.

2. **~~Should analysis visualization methods live on `TaskPredictor` or stay as standalone functions?~~**  
   **Decision: Option B** — `plot_aucroc(predictor)`. Analysis functions take a `TaskPredictor` as input, keeping the class lean and focused on prediction/FI. This also allows analysis functions to be added independently without modifying `TaskPredictor`.

4. **~~Single class or two classes?~~**  
   **Decision: Single class preferred.** A single `TaskPredictor` with both `predict(new_data)` and `predict_test()` methods.

   *Pros of single class:*
   - Simpler API — one class to learn and import
   - Shared state (loaded predictors, config) is reused across both use cases
   - No need for inheritance hierarchy or abstract base class
   - `calculate_fi()` already supports both modes via `data=None` (test data) vs `data=df` (new data)

   *Cons of single class:*
   - Slightly larger class surface area
   - A user who only needs new-data prediction still sees test-data methods

   The pros clearly outweigh the cons — the test-data methods are a natural extension of the same loaded state, and splitting them into two classes would duplicate the constructor and config-loading logic.

### Still Open

3. **Model class path stability strategy:**  
   How do we ensure `octopus.modules.octo.bag.BagClassifier` remains loadable across versions? Register aliases? Freeze the path? Custom unpickler?

### Resolved (continued)

5. **~~Feature groups:~~**  
   **Decision: Persist during study execution (option b)** — save `feature_groups.json` alongside `predictor.json`.

   **Background:** Feature groups are currently **not persisted to disk**. They are computed at runtime by `calculate_feature_groups()` in `octopus/utils.py` (Spearman correlation-based grouping at thresholds 0.7, 0.8, 0.9) and passed through the execution chain in memory only: `workflow_runner.run_outersplit()` → `module.fit()` → `training`. They are never saved.

   **Save plan — changes required:**
   - In `ModuleExecution.fit()` (`octopus/modules/base.py`): the `feature_groups` parameter is already received. Store it on the instance (e.g., `self._feature_groups = feature_groups or {}`).
   - In `ModuleExecution.save()` (`octopus/modules/base.py`): add saving of `feature_groups.json`:
     ```python
     # In save() method, after saving predictor.json:
     feature_groups = getattr(self, "_feature_groups", {}) or {}
     with (path / "feature_groups.json").open("w") as f:
         json.dump(feature_groups, f, indent=2)
     ```
   - In `study_io.py`: add `load_feature_groups(study_path, outersplit_id, task_id)` → reads `feature_groups.json`
   - In `TaskPredictor`: load feature groups during construction, use them as default for group FI methods

   **New data scenario:** When calculating group FI on new data, the saved training-time groups are used by default. Feature groups represent correlation structure learned during training — they are a property of the feature space, not the evaluation data. Users can optionally override via `calculate_fi(..., feature_groups=custom_groups)`.

   **Backward compatibility:** Old studies without `feature_groups.json` will have no saved groups. `TaskPredictor` handles this gracefully: group FI methods raise a clear error if no groups are available and none are provided.

6. **~~`data_pool` for permutation FI shuffling:~~**  
   **Decision: Use train + evaluation data combined** (same as main branch).

   The main branch's `get_fi_permutation()` builds the shuffling pool as:
   ```python
   data_all = pd.concat([data_traindev, data], axis=0)
   ```
   Then for each feature, shuffled values are sampled from `data_all[feature]`. This is the correct approach.

   **Consequences of each option:**

   | Option | Pool | Pros | Cons |
   |--------|------|------|------|
   | **Test data only** | `data` (evaluation set) | Simplest, no extra data needed | Small test sets → limited value diversity → shuffled columns still correlated with originals → **underestimates importance** |
   | **Train data only** | `data_train` | Large pool, good diversity | Values may come from distribution the model was trained on → feature may appear less "broken" → **underestimates importance for overfit models** |
   | **Train + evaluation** ✓ | `pd.concat([data_train, data])` | Largest pool, maximum value diversity, matches main branch behavior | Slightly more memory usage |

   The key insight: with a small test set (e.g., 50 samples), sampling only from those 50 values means the shuffled column still partially preserves the original distribution. Combining with training data (typically much larger) provides a richer pool of replacement values, making the permutation more disruptive and the importance estimates more reliable.

   **For `TaskPredictor`:** when computing permutation FI, load `data_train.parquet` for the outersplit and combine with the evaluation data as the shuffling pool. The exact pool composition depends on the context:

   | Context | Shuffling pool | Rationale |
   |---------|---------------|-----------|
   | `calculate_fi(fi_type="permutation")` (test data) | `train + test` | Same as main branch — test data is the evaluation set, train provides additional diversity |
   | `calculate_fi(new_data, fi_type="permutation")` (new data) | `train + test + new_data` | Maximum diversity — includes all available data from the study plus the new evaluation data |

   This maximizes the pool size in both cases, ensuring the most disruptive permutations and the most reliable importance estimates.
