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

## Summary

| Issue | Severity | Fix Effort | Blocks TaskPredictor? |
|-------|----------|------------|----------------------|
| B1: Ensemble model not in model.joblib | Critical | Small | Yes — can't predict with ensel model |
| B2: feature_cols not saved | Important | Small | Yes — needed for validation + ensemble FI |
| B3: Ensemble selection very slow | Performance | Medium | No — but affects study execution time |
| B4: Trial bags as monolithic pickles | Design | Medium | No — but causes B3 |
| B5: result_type not properly supported | Unclear | TBD | No — but affects analysis correctness |
