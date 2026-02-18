# TaskPredictor Concept — Overview & Design Principles

**Status:** DRAFT — For Review  
**Date:** 2025-02-18  
**Branch:** `Refactor/pr308_anlysis_predict`  
**Related:** PR#308 (task predictor concept lost), version stability requirements

---

## 1. Problem Statement

### 1.1 Lost TaskPredictor Concept

When Octopus runs a study, jobs are broken down to the **ML experiment level**, where each experiment is a combination of `(outer_split, task)`. At prediction time, all ML experiments belonging to a single task must be brought back together to produce a unified prediction (e.g., averaging across outer splits). This aggregation responsibility was previously handled by a "task predictor" concept that was lost in PR#308.

### 1.2 Main Branch — `OctoPredict` and Its Issues

The main branch has `OctoPredict` in `octopus/predict/core.py`. It provides the right API shape:

```python
predictor = OctoPredict(study_path, task_id, results_key)
predictor.predict(new_data)
predictor.predict_proba(new_data)
predictor.predict_test()
predictor.calculate_fi(data, fi_type="permutation")
predictor.calculate_fi_test(fi_type="group_permutation")
```

**Problems with `OctoPredict`:**

1. **Data loading via `OctoExperiment.from_pickle()`** — loads monolithic `.pkl` files that contain the entire experiment state. These pickle files are tightly coupled to the execution code and break across versions.
2. **Depends on `ExperimentInfo`** from `octopus/modules/utils.py` — which itself imports `BaseModel` from `octopus/models/config.py`, pulling in study-execution dependencies.
3. **Feature importance uses** `get_fi_permutation`, `get_fi_shap` etc. from `octopus/modules/utils.py` — these functions have the right algorithm (p-values, confidence intervals, group permutation, SHAP) but are coupled to `ExperimentInfo` and execution code.
4. **Not version-stable** — any change to `OctoExperiment`, `ExperimentInfo`, `BaseModel`, or the module infrastructure breaks old saved studies.

### 1.3 This Branch — Good Building Blocks, Missing the Unifying Class

This branch has already done significant work to decouple data loading from execution:

| Component | Location | What it does |
|-----------|----------|--------------|
| `StudyLoader` | `octopus/analysis/loaders.py` | Loads study config, discovers outersplits |
| `OuterSplitLoader` | `octopus/analysis/loaders.py` | Loads test/train data, predictions, scores, FI per outersplit+task |
| `Predictor` | `octopus/modules/predictor.py` | Lightweight wrapper: model + selected features → `predict()`, `predict_proba()`. **Will be eliminated** — functionality absorbed into `TaskPredictor` |
| `load_task_modules()` | `octopus/analysis/module_loader.py` | Loads all Predictors for a task across outersplits |
| `ensemble_predict/proba()` | `octopus/analysis/module_loader.py` | Aggregates predictions across outersplits |
| `notebook_utils` | `octopus/analysis/notebook_utils.py` | High-level functions: ROC, confusion matrix, performance tables, FI plots |

**What's missing:**

- A single `TaskPredictor` class that ties these together
- The `Predictor` class is an unnecessary abstraction — its model loading and feature-subsetting logic should be inlined into `TaskPredictor`
- The data loading code is spread across `analysis/loaders.py` and `analysis/module_loader.py` — it should be streamlined and self-contained within `octopus/predict/`
- `analysis/__init__.py` imports from `octopus.modules.utils` (e.g., `get_fi_permutation`) — this creates a dependency on module/execution code
- The feature importance code from the main branch (with p-values, CIs, group permutation, SHAP) needs to be brought in but decoupled from `ExperimentInfo`

### 1.4 Goal

Create a **`TaskPredictor`** class that:

1. Is the **single entry point** for all prediction and analysis operations on a completed study task
2. Has a **simple constructor**: `TaskPredictor(study_path, task_id, result_type)`
3. Lives in `octopus/predict/` along with **all** its dependencies (data loading, feature importance)
4. Has **zero imports** from `octopus/modules/`, `octopus/study/`, `octopus/manager/`, or any other study-execution code
5. Is **long-term version stable** by design

---

## 2. Key Design Principle: Complete Separation

### 2.1 The Three Phases of Octopus Usage

| Phase | Code | Stability |
|-------|------|-----------|
| **(a) Running a study** (`fit()`) | `OctoStudy`, `Manager`, `Module`, etc. | Can change freely between versions |
| **(b) Analyzing a study** | `TaskPredictor` + analysis utilities | **Must be version-stable** |
| **(c) Predicting on new data** | `TaskPredictor` | **Must be version-stable** |

Phases (b) and (c) only need:
- Study directory path
- Saved study config (`config.json`)
- Saved models (`model.joblib`)
- Saved metadata (`predictor.json` or `module_state.json`)
- Saved test/train data per outer split (`data_test.parquet`, `data_train.parquet`)

### 2.2 Dependency Rule

```
octopus/predict/     →  ONLY: pandas, numpy, joblib, json, upath, sklearn, scipy, shap
                     →  NEVER: octopus/metrics/, octopus/modules/, octopus/study/, octopus/manager/, octopus/models/
                     →  Scoring functions are PRIVATE COPIES (see §2.3)

octopus/modules/     →  NEVER imports from: octopus/predict/
                     →  Saves models via plain joblib.dump() + JSON writes (no shared class)

octopus/analysis/    →  CAN import from: octopus/predict/
                     →  NEVER: octopus/modules/, octopus/study/, octopus/manager/
```

> **⚠️ Why no `octopus.*` imports at all:** `octopus/__init__.py` eagerly imports `OctoClassification, OctoRegression, OctoTimeToEvent` from `octopus.study`. This triggers the entire codebase — 60+ modules loaded including the full execution stack. Private copies of the two needed scoring functions give `predict/` complete isolation. See `06_implementation.md` §10.0 for the full analysis and §10.0.1 for a future lazy-loading alternative.

### 2.3 Private Scoring Function Copies

`octopus/predict/feature_importance.py` contains private copies of two functions originally from `octopus/metrics/utils.py`:

| Original | Private copy | Used by |
|----------|--------------|---------|
| `get_score_from_model()` | `_get_score_from_model()` | Permutation FI |
| `get_performance_from_model()` | `_get_performance_from_model()` | FI analysis |

These are copied (not imported) to avoid the `octopus/__init__.py` import chain. See `06_implementation.md` §10.0 and §10.2 for details.

> **Future alternative:** If `octopus/__init__.py` is fixed with lazy imports (`__getattr__`), these copies can be replaced with direct imports from `octopus.metrics.utils`. Empirically verified: with lazy init, importing `octopus.metrics` loads only 20 stable definition modules (metric registry + model type aliases). See `06_implementation.md` §10.0.1 for the step-by-step migration path.

**Requirements on the model protocol** (what these functions expect):
- `model.predict(data[feature_cols])` → array-like
- `model.predict_proba(data[feature_cols])` → 2D array-like (classification)
- `model.classes_` → array-like of class labels (classification)

`TaskPredictor` satisfies this protocol (see §7.7 in `05_task_predictor_api.md`).

This means:
- **`Predictor` is eliminated** — `octopus/modules/predictor.py` is removed entirely. Model loading and feature subsetting are inlined into `TaskPredictor`
- **Data loading code moves into `octopus/predict/study_io.py`** — streamlined, self-contained
- **Feature importance code is brought from main branch** into `octopus/predict/` — decoupled from `ExperimentInfo`, using direct model access and data
- **`octopus/analysis/`** becomes a thin layer of notebook convenience functions that delegates to `TaskPredictor`
