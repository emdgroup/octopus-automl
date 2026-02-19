# Specification Review — Inconsistencies & Architectural Weaknesses

**Date:** 2025-02-18
**Scope:** All specification documents (01–06)
**Status:** Most issues resolved — see resolution notes below

---

## Critical Issues

### C1: `predict()` / `predict_proba()` return type breaks sklearn compatibility ✅ RESOLVED

**Where:** 05_task_predictor_api.md §7.2, §7.3, §7.7

**Problem:** The spec originally said predict methods return DataFrames, but `get_score_from_model()` expects array-like returns for sklearn compatibility.

**Resolution:** All four methods (`predict`, `predict_proba`, `predict_test`, `predict_proba_test`) now default to returning `np.ndarray`. A `df=True` parameter provides a formatted DataFrame with sample index, prediction_std, etc. Updated in: `05_task_predictor_api.md` §6.2, §6.3, §7.2–7.5, `06_implementation.md` §9.1–9.2.

### C2: Missing `classes_` attribute on TaskPredictor ✅ RESOLVED

**Where:** 05_task_predictor_api.md §7.7, §7.8

**Resolution:** Added `classes_` property to §7.7 and §7.8. Delegates to first predictor's model. Raises `AttributeError` for regression/t2e.

### C3: Feature selection variability across outersplits ✅ RESOLVED (REVISED)

**Where:** 05_task_predictor_api.md §7.2, §7.8

**Original problem:** `selected_features` property was ambiguous across outersplits.

**Revised resolution (per user feedback):** `selected_features` are an **outcome of model training** and should NOT be exposed on `TaskPredictor`. Instead, `TaskPredictor` uses `feature_cols` — the **input features** available to the task. These are saved to `feature_cols.json` during study execution, loaded by `TaskPredictor`, and unioned across outersplits. Each `Predictor` internally handles its own `selected_features_` subsetting.

Updated in: `05_task_predictor_api.md` §6.5, §7.8; `02_architecture.md` §3.1b; `03_study_io.md` (new function + directory structure); `06_implementation.md` §11 phase 3b.

### C4: `result_type` not mapped to file system ✅ RESOLVED

**Where:** 05_task_predictor_api.md §6.1, §7.1

**Investigation findings:**
- `model.joblib` always contains the **best bag** model (`self.model_ = best_bag` in octo module)
- `result_type` column exists in `predictions.parquet`, `scores.parquet`, `feature_importances.parquet` as a discriminator
- No separate model for "ensemble_selection" — the ensel bag is only saved as a pickle (`ensel_bag.pkl`) in the old format
- For new data prediction, only the best model is available

**Resolution:** Removed `result_type` from `TaskPredictor` constructor. The saved model is always the best bag. The `result_type` column in parquet files is a study execution artifact. Added explanatory note in §6.1.

---

## Inconsistencies

### I1: `predict()` signature mismatch ✅ RESOLVED

**Where:** 05_task_predictor_api.md §6.2 vs §7.2

**Original problem:** `method="mean"` parameter appeared in §7.2 but not §6.2.

**Origin:** `method="mean"` comes from `ensemble_predict()` in `analysis/module_loader.py`, which supports "mean", "median", and "vote" aggregation. Since `predict()` must be sklearn-compatible (no extra params beyond `data`), the aggregation is always mean. The `method` parameter was removed from the predict signature — aggregation is fixed at mean.

**Resolution:** Aligned to `predict(data, df=False)` in both §6.2 and §7.2. No `method` parameter.

### I2: `feature_groups` missing from API surface ✅ RESOLVED

**Resolution:** Added `feature_groups` to §6.5 properties and `feature_groups=None` override parameter to `calculate_fi()` in §7.6.

### I2b: `calculate_fi` on Predictor ✅ RESOLVED (NEW)

**Problem (per user feedback):** The existing `Predictor` class has `get_feature_importances()` with internal/permutation/coefficients methods using sklearn's basic `permutation_importance`. These are inferior to Octopus's custom FI (no p-values, no CIs, no group support).

**Resolution:** All FI methods are **removed from Predictor** when moving it to `predict/predictor.py`. FI belongs on `TaskPredictor` via the decoupled functions in `predict/feature_importance.py`. Updated in: `02_architecture.md` §3.1; `06_implementation.md` §11 phase 2.

### I3: FI function signatures reference `data_traindev` vs `data_train` ✅ RESOLVED

**Resolution:** Clarified in `04_feature_importance.md` §5.2 that `data_train.parquet` on this branch is equivalent to `data_traindev` on the main branch — it contains all non-test data for the outersplit (train + dev combined, before inner split). Added explicit `data_pool` parameter documentation.

### I4: Directory structure naming — No action needed

The code uses `task{task_id}` (e.g., `task0/`), matching the spec. Only the `task{id}` naming convention is supported — no backward compatibility with legacy naming conventions is needed.

---

## Architectural Weaknesses

### A1: `octopus/metrics/` stability contract undefined ✅ RESOLVED

**Resolution:** The two scoring functions (`get_score_from_model`, `get_performance_from_model`) are **private copies** in `predict/feature_importance.py`, not imports. This eliminates the stability concern entirely — `predict/` has zero `octopus.*` imports.

The function signatures and model protocol are documented in `01_overview.md` §2.3 and `06_implementation.md` §10.2 as a sync contract between the originals and copies.

### A2: `Predictor.predict()` feature subsetting implicit ✅ RESOLVED (SUPERSEDED)

**Resolution:** The `Predictor` class has been **eliminated entirely** (per user decision). `TaskPredictor` now loads models and selected features directly via `study_io` functions and handles feature subsetting per outersplit internally. Documented in `05_task_predictor_api.md` §7.2, §7.9.

### A5: `Predictor` class eliminated ✅ NEW DECISION

**Decision (per user feedback):** The `Predictor` class (`octopus/modules/predictor.py`) was an unnecessary abstraction layer. It was a thin wrapper around `model.joblib` + `predictor.json` that added:
- Feature subsetting (`data[self.selected_features_]`)
- `predict()` / `predict_proba()` delegation to the model
- `save()` / `load()` serialization

All of this is now inlined into `TaskPredictor`:
- Models are loaded directly via `study_io.load_model()` → stored in `self._models: dict[int, Any]`
- Selected features loaded via `study_io.load_selected_features()` → stored in `self._selected_features: dict[int, list[str]]`
- Feature subsetting done inline in `predict()` / `predict_proba()`
- On the save side: module execution code uses plain `joblib.dump()` + JSON writes (no shared class)

**Benefits:**
- No shared class between `predict/` and `modules/` — complete separation
- Simpler mental model — one class to understand (`TaskPredictor`)
- No `Predictor` import path stability concerns

Updated in: all spec files (01–06).

### A6: `octopus/__init__.py` Import Chain ✅ RESOLVED (CRITICAL)

**Problem discovered during audit:** `octopus/__init__.py` contains:
```python
from octopus.study import OctoClassification, OctoRegression, OctoTimeToEvent
```

This means importing **any** `octopus.*` submodule triggers the entire import chain. Verified empirically: `from octopus.metrics.utils import get_score_from_model` loads **60+ octopus modules** including `octopus.modules`, `octopus.manager`, `octopus.study`, `octopus.models`, `octopus.datasplit`, and all their dependencies.

**Resolution (current):** `octopus/predict/` uses **private copies** of the two scoring functions. This gives zero `octopus.*` imports — complete isolation regardless of `__init__.py` behavior.

**Future alternative (documented in `06_implementation.md` §10.0.1):** Fix `octopus/__init__.py` with lazy imports (`__getattr__`). Empirically verified: with lazy init, importing `octopus.metrics` loads only **20 modules** — all stable definition code (metrics registry, model configs, type aliases). No execution stack loaded. If this fix is applied, the private copies can be replaced with direct imports from `octopus.metrics.utils` (single source of truth, no duplication). Step-by-step migration path documented in §10.0.1.

Updated in: `01_overview.md` §2.2, §2.3; `06_implementation.md` §10.0, §10.0.1, §10.1, §10.2.

### A3: Time-to-event support under-specified — STILL OPEN

Needs clarification on:
- What `predict()` returns for t2e (risk scores?)
- What `predict_proba()` does for t2e (raise NotImplementedError?)
- FI for t2e requires `event_time` and `event_indicator` columns — how does the data flow work?

### A4: No rollback / error handling strategy — STILL OPEN

Needs decision on partial failure behavior (e.g., one outersplit fails to load).

---

## Minor Issues

### M1: 03_study_io.md missing utility functions — LOW PRIORITY

Optional additions: `load_predictions()`, `load_task_config()`. Not needed for initial implementation.

### M2: `predict_test()` vs `predict()` aggregation semantics ✅ RESOLVED

Now explicitly documented in §7.2 (row index aggregation) and §7.4 (row_id_col aggregation).
