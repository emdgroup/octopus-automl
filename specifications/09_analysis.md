# TaskPredictor Concept — Analysis Layer & Notebook Specification

**Parent document:** [01_overview.md](01_overview.md)
**Date:** 2025-02-18

---

## 1. Goal

All analysis functionality (notebook utilities, study loaders, scoring) must live in `octopus/predict/` and be **long-term version stable**. The example notebook `examples/analyse_study_classification.ipynb` and all supporting code must work with any study directory produced by any version of Octopus, reading only stable file formats (JSON, Parquet, joblib). The current `octopus/analysis/` package is consolidated into `predict/`.

---

## 2. Current State — Problems

### 2.1 `analysis/__init__.py` Imports Execution Code

```python
from octopus.modules.utils import get_fi_permutation   # ← execution code dependency
```

This import pulls in `octopus.modules.utils`, which imports `ExperimentInfo`, `BaseModel`, and other execution-layer classes. Any change to the module infrastructure can break `import octopus.analysis`.

### 2.2 `analysis/module_loader.py` Depends on `Predictor`

```python
from octopus.modules.predictor import Predictor   # ← execution code dependency
loaded_module = Predictor.load(loader.module_dir)
```

`load_task_modules()` uses the `Predictor` class from `octopus/modules/predictor.py` — a class that is being eliminated (see `02_architecture.md` §3.1). This function is the central data-loading function used by almost every notebook utility.

### 2.3 `analysis/notebook_utils.py` Depends on `metrics.utils`

```python
from octopus.metrics.utils import get_performance_from_model   # ← triggers __init__.py chain
```

`testset_performance_overview()` and `show_confusionmatrix()` call `get_performance_from_model()` from `octopus.metrics.utils`. Due to the `octopus/__init__.py` import chain (see `06_implementation.md` §10.0), this loads 60+ modules.

### 2.4 `analysis/loaders.py` — Good Foundation, Needs Minor Cleanup

`StudyLoader` and `OuterSplitLoader` are well-designed and read only stable file formats (JSON, Parquet). They have **no execution code dependencies** — these are the right abstractions. Minor issues:
- Some methods load from `module_state.json` (legacy format)
- `get_outersplit_loader()` takes `module` and `result_type` params that aren't well-defined

### 2.5 Notebook Uses `notebook_utils` Directly (Correct)

The notebook imports from `octopus.analysis.notebook_utils` — the right pattern. But transitively, this triggers all the problematic dependencies listed above.

---

## 3. Metrics Dependency Analysis

### 3.1 Current Metrics Dependency Chain

`notebook_utils.py` imports `get_performance_from_model()` from `octopus.metrics.utils`. This function:

1. Calls `Metrics.get_instance(metric_name)` → returns a `Metric` dataclass with:
   - `ml_type` (classification/regression/multiclass/timetoevent)
   - `prediction_type` (predict/predict_proba)
   - `direction` (maximize/minimize)
   - `calculate(target, predictions) → float` (wraps sklearn metric function)
2. Gets predictions from the model (`predict()` or `predict_proba()` depending on metric)
3. Calls `metric.calculate()` to compute the score

**Where this is used in notebook_utils:**
- `testset_performance_overview()` — calls `get_performance_from_model()` for each outersplit × metric
- `show_confusionmatrix()` — calls `get_performance_from_model()` for each outersplit × metric

### 3.2 How to Eliminate the Dependency

The `octopus.metrics.Metrics` class is a **registry** — calling `Metrics.get_instance("AUCROC")` returns a dataclass containing a sklearn function (`sklearn.metrics.roc_auc_score`), the ml_type, prediction_type, and direction. The actual calculation is just `metric.calculate(target, predictions)` which calls the sklearn function.

The `predict/` package avoids importing `octopus.metrics` by **inlining a private copy** of this logic in `predict/feature_importance.py` (see `06_implementation.md` §10.2). Specifically:

1. **Private metric registry** — a dictionary mapping metric names to their sklearn functions, prediction types, and directions:
   ```python
   # Inside predict/feature_importance.py (private, no octopus.metrics import)
   _METRIC_REGISTRY = {
       "AUCROC": {"fn": sklearn.metrics.roc_auc_score, "prediction_type": "predict_proba", "direction": "maximize"},
       "ACCBAL": {"fn": sklearn.metrics.balanced_accuracy_score, "prediction_type": "predict", "direction": "maximize"},
       "R2":     {"fn": sklearn.metrics.r2_score, "prediction_type": "predict", "direction": "maximize"},
       # ... all supported metrics
   }
   ```

2. **Private scoring function** — `_get_performance_from_model(model, data, features, metric_name, ...)` looks up the metric in `_METRIC_REGISTRY`, gets predictions from the model, and calls the sklearn function directly. No `Metrics.get_instance()` call.

3. **`TaskPredictor.performance_test(metrics)`** uses `_get_performance_from_model()` internally — so `analysis/notebook_utils.py` calls `predictor.performance_test(["AUCROC", "ACC"])` and gets scores back without any `octopus.metrics` import.

**Result:** After cleanup, `predict/` is fully self-contained:
- Zero imports from `octopus.metrics` — scoring is encapsulated inside `TaskPredictor` via `_METRIC_REGISTRY`
- Zero imports from `octopus.modules` — model loading is encapsulated inside `TaskPredictor`
- `notebook_utils.py`, `loaders.py`, and `TaskPredictor` all live in `predict/` with no external `octopus.*` dependencies

---

## 4. Target Architecture — Based on Main Branch Pattern

### 4.1 Reference: Main Branch Notebook Structure

The main branch notebook (`examples/analyse_study_classification.ipynb`) defines the **target pattern**. The current branch notebook should be **ignored** — it will be replaced to match the main branch structure.

**Main branch pattern:**

```python
# 1. Import from octopus.predict — NOT octopus.analysis
from octopus.predict import OctoPredict                    # = TaskPredictor
from octopus.predict.notebook_utils import (               # notebook_utils lives in predict/
    show_study_details, show_target_metric_performance,
    show_selected_features, testset_performance_overview,
    plot_aucroc, show_confusionmatrix,
    show_overall_fi_table, show_overall_fi_plot,
)

# 2. Create TaskPredictor ONCE, pass to all functions
task_predictor = OctoPredict(study_path=study_info["path"], task_id=0, results_key="best")

# 3. Functions take the predictor object — NOT study_path + task_id
testset_performance = testset_performance_overview(predictor=task_predictor, metrics=metrics)
plot_aucroc(task_predictor, show_individual=True)
show_confusionmatrix(task_predictor, threshold=0.5, metrics=metrics)

# 4. FI computed FRESH by TaskPredictor — never loaded from disk
task_predictor.calculate_fi_test(fi_type="group_permutation", n_repeat=3)
fi_table = show_overall_fi_table(task_predictor, fi_type="group_permutation")
show_overall_fi_plot(task_predictor, fi_type="group_permutation")
```

### 4.2 Key Design Decisions from Main Branch

| Decision | Description |
|----------|-------------|
| **notebook_utils lives in `predict/`** | NOT in `analysis/` — utilities are part of the prediction package |
| **Functions take a `predictor` object** | NOT `study_path` + `task_id`. The predictor is created once and reused |
| **FI is always computed fresh** | `task_predictor.calculate_fi_test()` runs permutation/SHAP on the loaded models + test data. Nothing is loaded from parquet |
| **`show_overall_fi_table(predictor, fi_type)`** | Reads FI from the predictor's in-memory results after `calculate_fi_test()` — not from disk |
| **No `module` or `result_type` params on functions** | These are set when creating the predictor, not on each function call |

### 4.3 Implications for This Branch

1. **Move all analysis functionality into `predict/`** — matching main branch structure
2. **Move `notebook_utils.py` from `analysis/` to `predict/`**
3. **Move `loaders.py` from `analysis/` to `predict/`** — `StudyLoader` is needed by `notebook_utils` for study-level data
4. **Delete `analysis/module_loader.py`** — all model loading is in `TaskPredictor`
5. **Refactor all notebook_utils functions** to take a `predictor` argument instead of `study_path`
6. **Remove all FI-loading-from-disk code** — FI is always computed fresh by `TaskPredictor.calculate_fi_test()`

### 4.4 Function Signatures After Cleanup

All functions that need per-outersplit model access take a `predictor: TaskPredictor` argument:

```python
# Functions that take a predictor (need models/data)
def testset_performance_overview(predictor: TaskPredictor, metrics: list[str] | None = None) -> pd.DataFrame
def plot_aucroc(predictor: TaskPredictor, show_individual: bool = False) -> None
def show_confusionmatrix(predictor: TaskPredictor, threshold: float = 0.5, metrics: list[str] | None = None) -> None
def show_overall_fi_table(predictor: TaskPredictor, fi_type: str = "group_permutation") -> pd.DataFrame
def show_overall_fi_plot(predictor: TaskPredictor, fi_type: str = "group_permutation", top_n: int | None = None) -> None

# Functions that only need study-level info (no predictor needed)
def show_study_details(study_directory: str | Path) -> dict
def show_target_metric_performance(study_info: dict, details: bool = False) -> list[pd.DataFrame]
def show_selected_features(study_info: dict, sort_task: int | None = None, sort_key: str | None = None) -> tuple
```

### 4.5 Feature Importance — Always Computed Fresh

**Critical:** Feature importances are **never loaded from saved parquet files** in the analysis notebook. The workflow is:

1. **Create TaskPredictor** → loads models + test data from `model.joblib` + `data_test.parquet`
2. **Call `task_predictor.calculate_fi_test(fi_type=...)`** → computes FI using the loaded models on their test data
3. **Call `show_overall_fi_table(task_predictor, fi_type=...)`** → reads FI from `predictor.fi_results` (in-memory), aggregates across outersplits
4. **Call `show_overall_fi_plot(task_predictor, fi_type=...)`** → visualizes the aggregated FI

The saved `feature_importances.parquet` files (from training) are **not used** by notebook_utils. They exist for archival/reproducibility but are not the source of truth for analysis.

### 4.6 What Gets Deleted

| Code | Reason |
|------|--------|
| `analysis/module_loader.py` (entire file) | Replaced by `TaskPredictor` |
| `load_task_modules()` (250 lines) | `TaskPredictor.__init__()` does the same loading |
| `ensemble_predict()` (60 lines) | `TaskPredictor.predict()` |
| `ensemble_predict_proba()` (60 lines) | `TaskPredictor.predict_proba()` |
| `get_performance_from_model` import in notebook_utils | Replaced by `TaskPredictor.performance_test()` |
| `get_fi_permutation` import in `__init__.py` | FI computed fresh by `TaskPredictor.calculate_fi_test()` |
| All FI-loading-from-disk code in `show_overall_fi_table()` | FI read from `predictor.fi_results` in memory |
| `module` and `result_type` params on functions | Set on predictor creation, not per-function |
| `study_path` + `task_id` params on functions | Replaced by `predictor` argument |

---

## 5. Cleanup Plan — File-by-File

### 5.1 `octopus/analysis/` — Consolidate into `predict/`

All analysis functionality moves into `octopus/predict/`. The `analysis/` package is reduced to a minimal shell or removed entirely.

**Files to move to `predict/`:**
- `analysis/notebook_utils.py` → `predict/notebook_utils.py`
- `analysis/loaders.py` → `predict/study_loader.py` (or keep as `predict/loaders.py`)

**Files to delete:**
- `analysis/module_loader.py` — replaced by `TaskPredictor`
- `analysis/__init__.py` — reduced to re-exports or removed

**`analysis/loaders.py` (`StudyLoader`, `OuterSplitLoader`):**
Well-designed and version-stable (zero `octopus.*` imports). Move to `predict/` since all analysis functionality lives there. Changes:
- Remove `module` and `result_type` parameters from `get_outersplit_loader()` if not needed
- Handle missing files gracefully for older studies (e.g., `feature_cols.json`, `feature_groups.json`)

### 5.2 `predict/notebook_utils.py` — Refactored from `analysis/`

Per the main branch pattern, `notebook_utils.py` lives in `octopus/predict/notebook_utils.py`.

Zero `octopus.*` imports outside of `predict/`. Uses `TaskPredictor` and the moved `StudyLoader` (now in `predict/`). All model-related functions take a `predictor: TaskPredictor` argument.

Functions that need refactoring:

| Function | Current | After |
|----------|---------|-------|
| `testset_performance_overview()` | `load_task_modules()` + `get_performance_from_model()` | Takes `predictor` arg, calls `predictor.performance_test(metrics)` + display |
| `plot_aucroc()` | `load_task_modules()` + `module.predict_proba()` | Takes `predictor` arg, calls `predictor.predict_proba_test(df=True)` + ROC + plotly |
| `show_confusionmatrix()` | `load_task_modules()` + `get_performance_from_model()` | Takes `predictor` arg, calls `predictor.predict_proba_test(df=True)` + `performance_test()` + plotly |
| `show_overall_fi_table()` | `load_task_modules()` + loads FI from parquet | Takes `predictor` arg, reads `predictor.fi_results` (in-memory, after `calculate_fi_test()`) |
| `show_overall_fi_plot()` | Calls `show_overall_fi_table()` | Takes `predictor` arg, passes to `show_overall_fi_table()` |

Functions already clean (no model access needed): `show_study_details`, `_build_performance_dataframe`, `show_target_metric_performance`, `show_selected_features`, `display_table`.

---

## 6. Notebook Specification

### 6.1 Notebook Must Match Main Branch Structure

The notebook on the **main branch** (`examples/analyse_study_classification.ipynb`) defines the target structure. The current branch notebook should be **discarded and replaced** to match the main branch pattern.

**Main branch notebook structure:**

1. **Imports** — from `octopus.predict` only (TaskPredictor + notebook_utils)
2. **Input** — study directory
3. **Study Details** — `show_study_details(study_directory)`
4. **Target Metric Performance** — `show_target_metric_performance(study_info)`
5. **Selected Features Summary** — `show_selected_features(study_info)`
6. **Create TaskPredictor** — `task_predictor = TaskPredictor(study_path, task_id=0)`
7. **Test Performance** — `testset_performance_overview(predictor=task_predictor, metrics=...)`
8. **AUCROC Plots** — `plot_aucroc(task_predictor, show_individual=True)`
9. **Confusion Matrix** — `show_confusionmatrix(task_predictor, threshold=0.5, metrics=...)`
10. **Compute FI** — `task_predictor.calculate_fi_test(fi_type="group_permutation", n_repeat=3)`
11. **FI Table + Plot** — `show_overall_fi_table(task_predictor, fi_type=...)`, `show_overall_fi_plot(...)`
12. **Compute SHAP FI** — `task_predictor.calculate_fi_test(fi_type="shap", shap_type="kernel")`
13. **SHAP FI Table + Plot** — `show_overall_fi_table(task_predictor, fi_type="shap")`, `show_overall_fi_plot(...)`

### 6.2 Notebook Import Pattern

```python
# All imports from octopus.predict — matching main branch
from octopus.predict import TaskPredictor
from octopus.predict.notebook_utils import (
    show_study_details,
    show_target_metric_performance,
    show_selected_features,
    testset_performance_overview,
    plot_aucroc,
    show_confusionmatrix,
    show_overall_fi_table,
    show_overall_fi_plot,
)
```

### 6.3 Notebook Parameters

```python
# INPUT: Select study
study_directory = "../studies/wf_octo_mrmr_octo/"
```

The `task_id` is set when creating the predictor. The `module` and `result_type` parameters are **removed** — `TaskPredictor` loads whatever model was saved as `model.joblib`.

### 6.4 Sections Still Open (from notebook TODO)

The current notebook has these unfinished items:
- Individual test feature importances (table + plot)
- Summary confusion matrix
- Beeswarm plot (individual + merged)
- Tests for notebook utils

These should be planned but can be implemented after the core cleanup.

---

## 7. Long-Term Version Stability

> **See [`11_version_stability.md`](11_version_stability.md) for the comprehensive version stability analysis, including model deserialization risks, ONNX migration strategy, reference study test suite, and the distinction between version stability and development isolation.**

### 7.1 Stability Guarantee

The `octopus/predict/` package (including notebook_utils and study_loader) must satisfy:

> **Any study produced by any version of Octopus can be analyzed by any future version of `octopus/predict/`.**

Analysis is prediction + visualization — it belongs in `predict/`.

### 7.2 Stability Rules

**Development isolation** (import separation) is a valuable guideline that prevents accidental coupling, but the **primary version stability mechanisms** are backward-compatible file reading, model deserialization safety, and reference study tests. See `11_version_stability.md` §2 for details.

| Rule | Category | Description |
|------|----------|-------------|
| **No execution code imports** | Development isolation | `predict/` should not import from `octopus.modules/`, `octopus.study/`, `octopus.manager/`, `octopus.models/`, or `octopus.metrics/` |
| **Read-only file access** | Version stability | Analysis functions only read from study directories, never write |
| **Stable file formats only** | Version stability | JSON, Parquet, joblib — no pickle files, no monolithic serialization |
| **Backward-compatible file reading** | Version stability | Handle missing files/keys gracefully with defaults (see `11_version_stability.md` §3.2) |
| **Model class path stability** | Version stability | Never move classes saved in `model.joblib` without compatibility aliases (see `11_version_stability.md` §6) |
| **Reference study tests** | Version stability | CI tests verify all historical study directories load correctly (see `11_version_stability.md` §5) |
| **External dependencies are stable** | Version stability | `pandas`, `numpy`, `plotly`, `sklearn.metrics` — all have stable public APIs; pin compatible ranges |

### 7.3 Dependency Layers (After Cleanup)

```
Layer 3: examples/analyse_study_classification.ipynb
  │       (user-facing notebook — imports only from octopus.predict)
  ▼
Layer 2: octopus/predict/notebook_utils.py
  │       (visualization + formatting — uses TaskPredictor + StudyLoader)
  ▼
Layer 1: octopus/predict/task_predictor.py    octopus/predict/study_loader.py
  │       (prediction + FI + scoring)          (study-level: scores, selected features)
  ▼                                            ▼
Layer 0: Study directory on disk
          config.json, model.joblib, predictor.json, feature_cols.json,
          data_test.parquet, data_train.parquet, scores.parquet,
          predictions.parquet
```

**Every layer depends only on the layer below it.** No upward or lateral dependencies to execution code.

### 7.4 What Can Break Version Stability

| Risk | Mitigation |
|------|------------|
| Adding new `octopus.*` imports to `predict/` | Code review + CI check for forbidden imports |
| Changing study directory structure | Backward-compatible reading in `study_loader.py` (handle both old and new naming) |
| Changing parquet column names | Only add columns, never rename/remove |
| Changing `config.json` schema | Only add keys, never rename/remove |
| Changing `model.joblib` contents | Keep model class import paths stable (see `06_implementation.md` §10.3) |

### 7.5 CI Enforcement (Recommended)

Add a CI check that verifies `octopus/predict/` has no forbidden imports:

```python
# In tests or CI:
import ast, pathlib

FORBIDDEN_MODULES = {"octopus.modules", "octopus.study", "octopus.manager", "octopus.models", "octopus.metrics"}

for py_file in pathlib.Path("octopus/predict").glob("*.py"):
    tree = ast.parse(py_file.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = node.module if isinstance(node, ast.ImportFrom) else None
            if module and any(module.startswith(forbidden) for forbidden in FORBIDDEN_MODULES):
                raise AssertionError(f"{py_file}: forbidden import '{module}'")
```

---

## 8. Summary of Changes

| File | Action | Effort | Status |
|------|--------|--------|--------|
| `analysis/module_loader.py` | **Deleted** — replaced by TaskPredictor | Small | ✅ DONE |
| `analysis/notebook_utils.py` | **Deleted** — replaced by `predict/notebook_utils.py` | Medium | ✅ DONE |
| `analysis/loaders.py` | **Deleted** — replaced by `predict/study_io.py` | Small | ✅ DONE |
| `analysis/__init__.py` | **Deleted** — entire `analysis/` directory removed | Small | ✅ DONE |
| `predict/notebook_utils.py` | Created — all model functions take `predictor` arg | Medium | ✅ DONE |
| `predict/notebook_utils.py` | FI computed fresh by `predictor.calculate_fi()` — no loading from disk | Small | ✅ DONE |
| `predict/_metrics.py` | Created — private metric registry, no `octopus.metrics` import | Small | ✅ DONE |
| `examples/analyse_study_classification.ipynb` | **Updated** to use `octopus.predict.TaskPredictor` pattern | Medium | ✅ DONE |
| CI/tests | Add forbidden-import check for `predict/` | Small | |
