# Branch Issues — Known Problems on `Refactor/pr308_anlysis_predict`

**Date:** 2025-02-18
**Scope:** Issues discovered during specification work that exist in current branch code

---

## B1: Ensemble Selection Model Not Saved as `model.joblib` ⚠️ CRITICAL

**Location:** `octopus/modules/octo/module.py`

**Problem:** When ensemble selection is enabled (`ensemble_selection=True`), the ensemble bag is saved as `ensel_bag.pkl` (pickle format), but `self.model_` is always set to `best_bag`:

```python
# In _run_globalhp_optimization():
self.model_ = best_bag    # ← always the best single bag, never the ensel bag
```

```python
# In _create_ensemble_bag():
ensel_bag.to_pickle(self.path_results / "ensel_bag.pkl")   # ← only saved as pkl
```

The `base.py` save method writes `model.joblib` from `self.model_`:
```python
# In ModuleExecution.save():
if hasattr(self, "model_") and self.model_ is not None:
    with (path / "model.joblib").open("wb") as f:
        joblib.dump(self.model_, f)
```

**Consequence:**
- `model.joblib` always contains the **best single bag**, never the ensemble selection model
- The ensemble selection model is only available via `ensel_bag.pkl` (monolithic pickle)
- `TaskPredictor` cannot access the ensemble selection model through stable file formats
- The `result_type` column in parquet files has `"ensemble_selection"` entries for scores/predictions/FI, but there is no corresponding model that can be loaded for new data prediction

**Impact on TaskPredictor:** Even if a user ran ensemble selection and it performed better, `TaskPredictor.predict()` will use the best single bag, not the ensemble model.

**Suggested fix:** After ensemble selection completes, if ensel performance is better, set `self.model_ = ensel_bag` so it gets saved as `model.joblib`. Or save `ensel_bag` separately via `joblib.dump()` alongside `model.joblib`.

---

## B2: `feature_cols` Not Saved to Disk ⚠️ IMPORTANT

**Location:** `octopus/manager/workflow_runner.py`, `octopus/modules/base.py`

**Problem:** The input feature columns (`feature_cols`) are passed through the execution chain entirely in memory:

```
study.prepared.feature_cols
  → workflow_runner.run_outersplit()
    → module.fit(feature_cols=...)
      → passed to Training, ObjectiveOptuna, etc.
```

**No `feature_cols.json` is saved anywhere.** The only features saved to disk are:
- `selected_features` (in `predictor.json`) — these are the **output** of feature selection, a subset of `feature_cols`
- `feature_cols` in individual `Training` objects — but only in pickle format (inside bag pkl or trial pkl)

**Consequence:**
- `TaskPredictor` cannot determine the original input features for a task
- Cannot validate that new prediction data has the correct columns
- Cannot compute ensemble-level FI (which needs the union of all input features)
- The spec requires `feature_cols.json` (see `02_architecture.md` §3.1b) but this file is not yet saved

**Suggested fix:** In `workflow_runner._save_task_results()` or `ModuleExecution.save()`, add:
```python
import json
with (path / "feature_cols.json").open("w") as f:
    json.dump(feature_cols, f)
```

---

## B3: Ensemble Selection Performance — Very Slow ⚠️ PERFORMANCE

**Location:** `octopus/modules/octo/enssel.py`

**Problem:** The `EnSel` class has severe performance issues in its `__attrs_post_init__()` which runs three expensive phases sequentially:

### Phase 1: `_collect_trials()` — Load ALL trial bags from pickle

```python
for file in pkl_files:
    bag = Bag.from_pickle(file)        # ← loads full bag with all trainings + data
    self.bags[file] = {
        "performance": bag.get_performance(),    # ← computes predictions + metrics
        "predictions": bag.get_predictions(),    # ← generates prediction DataFrames
    }
```

With `ensel_n_save_trials=50` (default), this loads **50 full bag objects** from pickle. Each bag contains multiple `Training` objects, each containing full train/dev/test DataFrames.

### Phase 2: `_ensemble_scan()` — O(N) ensemble evaluations

```python
for i in range(len(self.model_table)):           # ← N iterations
    bag_keys = self.model_table[: i + 1]["path"].tolist()
    scores = self._ensemble_models(bag_keys)     # ← each call does from_pickle + metric computation
```

Each `_ensemble_models()` call:
1. Concatenates predictions from all bags
2. Groups by `row_id_col` and averages
3. **Reloads `Bag.from_pickle(first_bag_key)` every time** — just to get the target column dtype
4. Calls `get_performance_from_predictions()` to compute metrics

### Phase 3: `_ensemble_optimization()` — O(iterations × N_models) evaluations

```python
for i in range(self.max_n_iterations):
    for model in self.model_table["path"].tolist():       # ← ALL models
        bags_lst = copy.deepcopy(bags_ensemble)
        bags_lst.append(model)
        perf = self._ensemble_models(bags_lst)            # ← from_pickle + metrics again
```

**Total `Bag.from_pickle()` calls inside `_ensemble_models`:**
- Phase 2: N calls (one per scan step)
- Phase 3: N_models × max_n_iterations calls

**With defaults** (50 saved trials, say 10 optimization iterations): ~50 + 50×10 = **~550 pickle loads** of the first bag, just for dtype.

### Root causes of slowness:

| Cause | Impact |
|-------|--------|
| Loading all 50 trial bags from pickle upfront | Memory + I/O heavy — each bag has full data |
| `_ensemble_models()` calls `Bag.from_pickle()` every time | Unnecessary repeated I/O — just to get target dtype |
| No caching of predictions — `get_predictions()` recomputed for each bag during collect | CPU heavy |
| `_ensemble_optimization` tries ALL models for each iteration | O(N²) complexity with metric computation each time |

### Suggested optimizations:

1. **Cache the first bag's target dtype** — load once in `_collect_trials()`, reuse in `_ensemble_models()`
2. **Store predictions during collection** — already done (stored in `self.bags[file]["predictions"]`), but `_ensemble_models` doesn't read them efficiently
3. **Pre-compute pooled predictions** — instead of re-concatenating and re-grouping every time a model is added, incrementally update the ensemble predictions
4. **Consider storing trial predictions as parquet** instead of full bag pickles — much faster I/O for the read-only data needed by ensemble selection

---

## B4: Trial Bags Saved as Monolithic Pickles

**Location:** `octopus/modules/octo/objective_optuna.py` → `_save_topn_trials()`

**How trial bags are saved:**
```python
path_save = self.path_study / self.task_path / "trials" / f"study{study_name}trial{n_trial}_bag.pkl"
bag.to_pickle(path_save)
```

Each trial bag is saved as a **full pickle** containing:
- All `Training` objects (typically 5 for 5-fold inner CV)
- Each `Training` contains: `data_train`, `data_dev`, `data_test` DataFrames, fitted model, preprocessing pipeline, feature importances, etc.
- The same `data_test` DataFrame is duplicated in every training within every bag

**Top-N management:** Uses a min-heap (`heapq`) to keep only the top `ensel_n_save_trials` bags. When a new bag is better than the worst in the heap, the worst is deleted:
```python
heapq.heappush(self.top_trials, (-1 * target_value, path_save))
bag.to_pickle(path_save)
if len(self.top_trials) > max_n_trials:
    _, path_delete = heapq.heappop(self.top_trials)
    path_delete.unlink()
```

**Parameters:**
- `ensel_n_save_trials` (default: 50) — number of top trial bags kept on disk
- `ensemble_selection` (default: False) — whether to run ensemble selection at all

**Suggested improvements:**
- Save only what ensemble selection needs (predictions, performance, metadata) — not full training objects with data
- Use a lighter serialization format (parquet for predictions, JSON for metadata)
- This would also fix the B3 performance issue

---

## B5: `result_type` Not Properly Supported ⚠️ UNCLEAR

**Context:** The `ResultType` enum in `octopus/modules/base.py` has two values:
```python
class ResultType(StrEnum):
    BEST = "best"
    ENSEMBLE_SELECTION = "ensemble_selection"
```

These are written as `result_type` columns in the parquet output files (`scores.parquet`, `predictions.parquet`, `feature_importances.parquet`). When ensemble selection is enabled, both `"best"` and `"ensemble_selection"` rows exist in the same parquet file.

**Problems:**

1. **No model for `result_type="ensemble_selection"`** — the ensemble model is only in `ensel_bag.pkl` (see B1), not in `model.joblib`. So the `result_type` column promises results from an ensemble model, but there's no loadable model to reproduce those results on new data.

2. **No filtering by `result_type`** — when loading scores/predictions from parquet, there is no mechanism to select which `result_type` to use. Both rows are concatenated into the same file without clear separation.

3. **`TaskPredictor` ignores `result_type`** — it loads `model.joblib` (always the best bag) regardless of whether ensemble selection produced better results. The `result_type` column in parquet files is effectively invisible to post-hoc analysis.

**Questions to resolve:**
- Should `TaskPredictor` be aware of `result_type` at all?
- If ensemble selection produces better results, should `model.joblib` contain the ensemble model instead? (Related to B1)
- Should parquet files be filtered by `result_type` when loading predictions/scores for analysis?

---

## B6: Duplicated Saving Responsibilities Between `WorkflowTaskRunner` and `ModuleExecution.save()` ⚠️ DESIGN

**Location:** `octopus/manager/workflow_runner.py` → `_run_task()`, `octopus/modules/base.py` → `ModuleExecution.save()`

**Problem:** After `module.fit()` completes, saving is split across two separate owners with overlapping concerns:

### What `WorkflowTaskRunner._save_task_results()` saves:
| File | Content |
|------|---------|
| `selected_features.json` | List of selected features |
| `scores.parquet` | Model performance metrics |
| `predictions.parquet` | Model predictions |
| `feature_importances.parquet` | Feature importance values |
| `task_config.json` | Task configuration (via `_save_task_config()`) |

### What `ModuleExecution.save()` saves (into `module/` subdirectory):
| File | Content |
|------|---------|
| `model.joblib` | Fitted model object (if `self.model_` exists) |
| `module_state.json` | `selected_features` + `feature_importances` (as JSON) |
| `predictor.json` | `selected_features` (again, duplicated) |

### Overlap and redundancy:

1. **`selected_features` is saved 3 times:**
   - `selected_features.json` by `WorkflowTaskRunner`
   - `module_state.json` → `"selected_features"` by `ModuleExecution.save()`
   - `predictor.json` → `"selected_features"` by `ModuleExecution.save()`

2. **`feature_importances` is saved in 2 formats:**
   - `feature_importances.parquet` by `WorkflowTaskRunner` (structured DataFrame)
   - `module_state.json` → `"feature_importances"` by `ModuleExecution.save()` (JSON dict via `default=str`)

3. **The `results` dict returned by `_run_task()` contains `scores`, `predictions`, `feature_importances` — but NOT `model` or `selected_features` directly.** The return signature is `(selected_features, results_dict)` where `results_dict = {"scores": df, "predictions": df, "feature_importances": df}`. The model is only persisted via `module.save()` and is not part of the results flow.

### Current execution order in `_run_task()`:
```python
# 1. Fit the module
selected_features, scores, predictions, feature_importances = module.fit(...)

# 2. Stamp module column
for df in [scores, predictions, feature_importances]:
    df["module"] = module_name

# 3. Module saves its own state (model + duplicated metadata)
module.save(output_dir / "module")

# 4. WorkflowTaskRunner saves config + results
self._save_task_config(task, output_dir)
self._save_task_results(selected_features, scores, predictions, feature_importances, output_dir)
```

### Questions to resolve:

1. **Should `ModuleExecution.save()` exist at all?** Its only unique contribution is saving `model.joblib`. Everything else it saves is redundant with what `WorkflowTaskRunner` already saves in cleaner formats (parquet vs JSON-with-`default=str`).

2. **Cleaner design options:**
   - **Option A — Module only saves model:** `ModuleExecution.save()` should only save `model.joblib` (and possibly `predictor.json` for `TaskPredictor`). All results and metadata saving stays in `WorkflowTaskRunner`.
   - **Option B — Module saves everything:** Move all saving into `ModuleExecution.save()` and have `_run_task()` just call `module.save()`. This gives modules full control over persistence.
   - **Option C — Separate concerns cleanly:** `WorkflowTaskRunner` saves *results* (scores, predictions, feature_importances, selected_features) for analysis/aggregation. `ModuleExecution.save()` saves only what `TaskPredictor` needs for new-data prediction (model + selected_features + any preprocessing state). No overlap.

3. **`module_state.json` purpose is unclear** — it stores `selected_features` and `feature_importances` as JSON, but both are already saved in better formats elsewhere. Is anything reading `module_state.json`?

4. **`predictor.json` vs `selected_features.json`** — both contain the same list. Which one should `TaskPredictor` use? Currently `predictor.json` is nested inside `module/` while `selected_features.json` is at the task level.

**Recommended approach (Option C):**
- `WorkflowTaskRunner` owns: `selected_features.json`, `scores.parquet`, `predictions.parquet`, `feature_importances.parquet`, `task_config.json`
- `ModuleExecution.save()` owns: `model.joblib`, `predictor.json` (with only what `TaskPredictor` needs)
- Remove `module_state.json` entirely — it duplicates data in worse formats
- This gives a clean separation: runner saves analysis artifacts, module saves prediction artifacts

---

## Summary

| Issue | Severity | Fix Effort | Blocks TaskPredictor? |
|-------|----------|------------|----------------------|
| B1: Ensemble model not in model.joblib | Critical | Small | Yes — can't predict with ensel model |
| B2: feature_cols not saved | Important | Small | Yes — needed for validation + ensemble FI |
| B3: Ensemble selection very slow | Performance | Medium | No — but affects study execution time |
| B4: Trial bags as monolithic pickles | Design | Medium | No — but causes B3 |
| B5: result_type not properly supported | Unclear | TBD | No — but affects analysis correctness |
| B6: Duplicated saving responsibilities | Design | Small | No — but causes confusion + redundant files |
| B7: ML info not given during module init | Design | Medium | No — but fragile module authoring |
| B8: Ensemble selection results not saved | Critical | Medium | Yes — only "best" results persisted |
| B9: Inconsistent results pattern | Design | Medium | No — but confusing and incomplete |

---

## B7: ML Information Not Given During Module Initialization ⚠️ DESIGN

**Location:** `octopus/modules/base.py`, `octopus/manager/workflow_runner.py`

**Problem:** ML information (task type, target column, metric configuration, etc.) is not provided during module initialization via `create_module()`. Instead, these parameters are submitted to `fit()`, which is called later. This means we rely on the module author to manually connect the `fit()` parameters to module attributes.

If we gave those parameters to `.create_module()`, we would have much more control over module construction and would not need to update attributes after initialization.

**Current flow:**
```python
# 1. Module is created — no ML context
module = create_module(module_config)

# 2. ML info is passed later via fit()
selected_features, scores, predictions, fi = module.fit(
    task_type=...,
    target_col=...,
    feature_cols=...,
    ...
)
```

**Consequences:**
- Module authors must manually wire `fit()` parameters into internal state — error-prone and boilerplate-heavy
- No compile-time or construction-time validation of ML parameters
- Module objects exist in an incomplete state between `create_module()` and `fit()`
- Cannot enforce invariants (e.g., "a module must always know its task type") at construction time

**Questions to resolve:**
- What is the motivation for deferring ML parameters to `fit()` instead of providing them at construction?
- Is there a use case where the same module instance is `fit()` with different ML parameters?
- Could `create_module()` accept a context/config object containing all ML information?

**Suggested fix:** Pass ML information (task type, target, metrics, etc.) to `create_module()` so that module instances are fully configured at construction time. This would allow validation at init, reduce boilerplate in module implementations, and prevent partially-initialized module states.

---

## B8: Ensemble Selection Results Not Saved Separately ⚠️ CRITICAL

**Location:** `octopus/modules/octo/module.py`, `octopus/manager/workflow_runner.py`

**Problem:** The octo module produces two types of results:

1. **`best`** — results from the best single model/bag
2. **`ensemble_selection`** — results from the ensemble selection model

Each result type consists of:
- `feature_importances`
- `scores`
- `selected_features`
- `model`
- `predictions`

**Currently, only the `best` results are fully saved.** The ensemble selection results are partially written into parquet files (with `result_type="ensemble_selection"` rows), but:
- The ensemble selection **model** is only saved as `ensel_bag.pkl` (not as a proper `model.joblib`) — see B1
- The ensemble selection **selected_features** are not saved separately
- There is no clear, findable location in the file structure for the complete ensemble selection result set

**Expected behavior:** Both result sets should be saved in separate, well-defined locations on disk, e.g.:

```
task_output/
├── results_best/
│   ├── model.joblib
│   ├── scores.parquet
│   ├── predictions.parquet
│   ├── feature_importances.parquet
│   └── selected_features.json
├── results_ensemble_selection/
│   ├── model.joblib
│   ├── scores.parquet
│   ├── predictions.parquet
│   ├── feature_importances.parquet
│   └── selected_features.json
```

Or alternatively, a structured results object:

```python
results = {
    "best": results_best,
    "ensemble_selection": results_ensemble_sel,
}
```

**Impact:** Users who enable ensemble selection cannot reliably load, inspect, or use the ensemble model and its associated results after a study completes. This makes ensemble selection effectively a "fire and forget" feature with no post-hoc usability.

---

## B9: Inconsistent Results Pattern ⚠️ DESIGN

**Location:** `octopus/modules/base.py`, `octopus/manager/workflow_runner.py`

**Problem:** There is a very inconsistent pattern in how module results are produced, returned, and saved.

A module produces the following results after `fit()`:
- `scores`
- `predictions`
- `feature_importances`
- `model` (via `self.model_`)

**Inconsistencies:**

1. **The results dict is incomplete** — `fit()` returns `(selected_features, scores, predictions, feature_importances)` but does NOT include the model. The model is an implicit side-effect stored as `self.model_`.

2. **No results object with a `.save()` method** — results are scattered across loose variables and module attributes. There is no unified `Results` object that encapsulates all outputs and knows how to persist itself.

3. **Models are not returned from `fit()`** — the model is accessed as a module attribute (`self.model_`), not as part of the results. This breaks the principle that `fit()` should return everything it produced.

4. **Models are saved by the module; everything else is saved by the runner** — `ModuleExecution.save()` persists the model (as `model.joblib`), while `WorkflowTaskRunner._save_task_results()` persists scores, predictions, and feature importances. This split ownership creates confusion about who is responsible for what (see also B6).

**Suggested fix:** Introduce a `ModuleResults` dataclass or similar structure:

```python
@dataclass
class ModuleResults:
    scores: pd.DataFrame
    predictions: pd.DataFrame
    feature_importances: pd.DataFrame
    selected_features: list[str]
    model: Any  # the fitted model/bag

    def save(self, path: Path) -> None:
        """Save all results to the given directory."""
        ...
```

This would:
- Make results complete and self-contained (including the model)
- Provide a single `.save()` method for all result persistence
- Eliminate the split saving responsibility between module and runner
- Make the return type of `fit()` explicit and inspectable
- Enable clean separation of `best` vs `ensemble_selection` results (see B8)
