# Evaluation Notebooks & Predict Package — Refactoring & Integration Strategy

**Date:** 2025-02-19  
**Scope:** (A) Refactoring of `octopus/predict/` only, (B) Refactoring of `predict/` + evaluation notebooks  
**Related:** specs 01–12, `examples/evaluation_classification.ipynb`, `examples/evaluation_regression.ipynb`  
**Reference code:** `maincode/` (old version of predict code, kept as documentation)  
**Design decision:** `octopus.metrics` imports are intentional and will remain — lazy loading of `octopus.metrics` will be implemented later to address import weight. No private `_metrics.py` copy.

---

## Part 1: Side-by-Side Analysis — Old (`maincode/`) vs New (`predict/`)

The `maincode/` directory contains the old version of the prediction code. `maincode/core.py` has `OctoPredict` (attrs class using pickled `OctoExperiment` + `ExperimentInfo`). `maincode/notebook_utils.py` has the original notebook utility functions.

### 1.1 What Was Properly Adapted ✅

| Component | Old (`maincode/`) | New (`predict/`) | Assessment |
|-----------|-------------------|------------------|------------|
| **Model loading** | `OctoExperiment.from_pickle()` → `ExperimentInfo` | `StudyLoader` + `OuterSplitLoader` (parquet/joblib) | ✅ Correctly migrated from pickle to parquet |
| **Predictor class** | `OctoPredict` (attrs class, `experiments` dict) | `TaskPredictor` (plain class, per-split accessors) | ✅ Cleaner API, same semantics |
| **Directory naming** | `workflowtask{id}` | `task{id}` | ✅ Updated to new naming convention |
| **`show_study_details()`** | Globs `workflowtask*`, validates structure | Globs `task*`, same validation logic | ✅ Adapted to new dirs |
| **`_build_performance_dataframe()`** | Loads `exp.pkl`, accesses `exp.results[key].scores` | Uses `StudyLoader.load_scores()` from parquet | ✅ Adapted from pickle to parquet, handles `result_type`/`module`/`aggregation` columns |
| **`show_target_metric_performance()`** | Expands `Scores_dict` column | Expands `Performance_dict` column | ✅ Same output format |
| **`show_selected_features()`** | From `exp.results[key].selected_features` | From `selected_features.json` via loader | ✅ Same pivot + frequency logic |
| **`testset_performance_overview()`** | Calls `get_performance_from_model()` per experiment | Calls `predictor.performance_test()` (delegates to TaskPredictor) | ✅ **Properly refactored** — scoring delegated |
| **`plot_aucroc()`** | `_get_predictions_df(experiment)` | `_get_predictions_from_predictor(predictor, split_id)` | ✅ Same ROC logic, adapted accessors |
| **`show_confusionmatrix()`** | CM plot via plotly subplots, per-experiment | Same CM layout, iterates `predictor.outersplits` | ✅ Plot logic correctly adapted |
| **FI computation** | `OctoPredict.calculate_fi()` → stores in `self.results` dict | `TaskPredictor.calculate_fi()` → stores in `self._fi_results` dict | ✅ Same caching pattern |
| **`show_overall_fi_table()`** | Aggregates across `predictor.results` keys by regex | Uses `predictor.fi_results[fi_type]` | ✅ Simplified — no regex key parsing |
| **`show_overall_fi_plot()`** | Simple plotly bar chart | Same + optional CI error bars | ✅ Enhanced with error bars |

### 1.2 What Was NOT Fully Adapted ⚠️

#### 1.2.1 Import Isolation Violation in `show_confusionmatrix()`

`show_confusionmatrix()` still directly calls `get_performance_from_model()` from `octopus.metrics.utils`:

```python
# predict/notebook_utils.py line ~430
from octopus.metrics.utils import get_performance_from_model
...
performance = get_performance_from_model(
    model=model, data=data_test, feature_cols=features,
    target_metric=metric, target_assignments=target_assignments,
    positive_class=predictor.positive_class,
)
```

This was copied from the old code (`maincode/notebook_utils.py`) where the same call exists. However, `testset_performance_overview()` was correctly refactored to use `predictor.performance_test()`, showing the intended pattern. `show_confusionmatrix()` was NOT given the same treatment.

**Note:** The old code passed `threshold=threshold` to `get_performance_from_model()`, but the new code drops this parameter. This means metrics in `show_confusionmatrix()` are computed at the model's default threshold, NOT the user-specified threshold. This may be a deliberate simplification or a bug depending on whether the original `get_performance_from_model()` actually used the threshold parameter.

**Fix:** The metrics scoring in `show_confusionmatrix()` should use `predictor.performance_test()`, consistent with how `testset_performance_overview()` already works. This would make the scoring pattern consistent within `notebook_utils.py`.

#### 1.2.2 `octopus.metrics` Imports — Intentional (Not a Violation)

The `octopus.metrics` imports exist in multiple predict/ files:

| File | Import | Purpose |
|------|--------|---------|
| `task_predictor.py` | `from octopus.metrics.utils import get_performance_from_model` | `performance_test()` metric computation |
| `notebook_utils.py` | `from octopus.metrics.utils import get_performance_from_model` | `show_confusionmatrix()` scoring (should be refactored to use predictor.performance_test) |
| `feature_importance.py` | `from octopus.metrics import Metrics` | Metric lookup for permutation FI |
| `feature_importance.py` | `from octopus.metrics.utils import get_performance_from_model` | Baseline scoring for permutation FI |

**Design decision:** These imports are **intentional and will remain**. Spec 12 §7 proposed a private `_metrics.py` copy, but this approach is rejected in favor of keeping `octopus.metrics` as the single source of truth. Lazy loading of `octopus.metrics` will be implemented later to address any import weight concerns. The only change needed is making `notebook_utils.py` consistent by routing CM scoring through `predictor.performance_test()` instead of calling `get_performance_from_model()` directly (see §1.2.1).

#### 1.2.3 `_build_performance_dataframe()` O(N²) Concat

Both old and new versions use the same anti-pattern:

```python
df = pd.concat([df, new_row], ignore_index=True)  # in a loop
```

This was carried forward from the old code. While functional, it's O(N²) for large studies.

#### 1.2.4 Old Code Feature Missing: `threshold` in CM Scoring

Old `show_confusionmatrix()`:
```python
performance = get_performance_from_model(..., threshold=threshold)
```

New `show_confusionmatrix()`:
```python
performance = get_performance_from_model(...)  # no threshold
```

If `get_performance_from_model()` uses the threshold for ACC/F1 metrics, the new code silently uses the default threshold (0.5) for metrics even when the user passes a different threshold for the CM. Needs verification.

#### 1.2.5 Old Code Feature Missing: Combined FI (ensemble prediction)

The old `OctoPredict.calculate_fi()` computed FI in two ways:
1. **Per-experiment**: FI for each individual model on its test data
2. **Combined (ensemble)**: Creates a combined `ExperimentInfo` with `model=self` (OctoPredict itself as the model) and union of all feature cols, then computes FI using the ensemble prediction

The new `TaskPredictor.calculate_fi()` only computes per-outersplit FI and averages. The combined/ensemble FI (where the ensemble predictor itself is treated as the model) is not implemented.

#### 1.2.6 Old Code Feature Missing: PDF Plot Saving for FI

The old code saved FI plots as PDF files in the study results directory:
- `_plot_permutation_fi()` → saves bar charts as PDF in `results/`
- `_plot_shap_fi()` → saves bar + beeswarm plots as PDF

The new code only displays plots in notebooks. No file saving.

### 1.3 Summary of Adaptation Quality

The migration from `maincode/` to `predict/` was done **methodically and correctly** for the core functionality:
- Pickle → parquet/joblib loading ✅
- OctoExperiment → StudyLoader/OuterSplitLoader ✅  
- OctoPredict → TaskPredictor ✅
- ExperimentInfo properties → TaskPredictor accessors ✅
- workflowtask → task directory naming ✅
- notebook_utils functions → adapted for TaskPredictor ✅

The remaining gaps are:
1. **O(N²) concat** — carried from old code
2. **`show_confusionmatrix()` scoring inconsistency** — uses direct `get_performance_from_model()` instead of `predictor.performance_test()` like `testset_performance_overview()` does
3. **Threshold parameter dropped** in CM scoring
4. **Combined/ensemble FI** not ported
5. **PDF saving** not ported

---

## Part 2: Scenario A — Refactoring of `predict/` Only

This scenario covers improvements to the existing `octopus/predict/` package without introducing evaluation notebook functionality. The `octopus.metrics` imports are **intentional and will remain** — lazy loading of `octopus.metrics` will be implemented later to address import weight. No private `_metrics.py` copy will be created.

### A1: Fix `show_confusionmatrix()` Scoring Consistency

**Severity:** Medium | **Effort:** Small

`testset_performance_overview()` was correctly refactored to delegate scoring to `predictor.performance_test()`. However, `show_confusionmatrix()` still calls `get_performance_from_model()` directly from `octopus.metrics.utils`. This creates an inconsistency within `notebook_utils.py` — two functions doing the same thing (scoring per outersplit) use different patterns.

**Fix:** Refactor `show_confusionmatrix()` to use `predictor.performance_test(metrics)` for the per-outersplit metric computation, consistent with `testset_performance_overview()`. The confusion matrix visualization itself (which uses the threshold for binary predictions) stays as-is.

**Also investigate:** The old code passed `threshold=threshold` to `get_performance_from_model()`, but the new code drops this. Verify whether this is a deliberate simplification or a regression.

### A2: Fix O(N²) Concat in `_build_performance_dataframe()`

**Severity:** Small | **Effort:** Small

Both old and new versions use the concat-in-loop anti-pattern. Replace with list-of-dicts:

```python
# Instead of:
df = pd.concat([df, new_row], ignore_index=True)

# Use:
rows = []
for ...:
    rows.append({...})
df = pd.DataFrame(rows)
```

### A3: Port Combined/Ensemble FI

**Severity:** Optional | **Effort:** Medium

The old `OctoPredict.calculate_fi()` computed FI in two ways:
1. **Per-experiment**: FI for each individual model on its test data
2. **Combined (ensemble)**: Creates a combined `ExperimentInfo` with `model=self` (OctoPredict itself as the model) and union of all feature cols, then computes FI using the ensemble prediction

The new `TaskPredictor.calculate_fi()` only computes per-outersplit FI and averages. Consider porting the combined/ensemble FI capability with a `combined=True` flag, where `TaskPredictor` itself is treated as the model (using its `predict()`/`predict_proba()` methods across all splits).

### A4: Restore `threshold` Support in CM Metrics

**Severity:** Small | **Effort:** Small

The old `show_confusionmatrix()` passed `threshold` to `get_performance_from_model()`. The new code drops this parameter. If the original function used the threshold for computing ACC, F1, etc., this is a regression. Verify and restore if needed.

### A5: Summary — Scenario A

| Issue | Severity | Effort | Description |
|-------|----------|--------|-------------|
| A1 | Medium | Small | Make CM scoring consistent with testset_performance_overview (use predictor.performance_test) |
| A2 | Small | Small | Fix O(N²) concat → list-of-dicts |
| A3 | Optional | Medium | Port combined/ensemble FI from old OctoPredict |
| A4 | Small | Small | Restore threshold support in CM metrics |

**What does NOT change in Scenario A:**
- `octopus.metrics` imports remain (lazy loading to be added later)
- No `_metrics.py` private copy
- `TaskPredictor` public API unchanged
- Study-level functions in `notebook_utils.py` unchanged
- `study_io.py` and `feature_importance.py` unchanged

---

## Part 3: Scenario B — `octopus/diagnostics/` Package for Evaluation Notebooks

**Decision: Option C (revised)** — Create a new top-level `octopus/diagnostics/` package.

This scenario extends Scenario A with the addition of evaluation notebook functionality as a dedicated library module. All Scenario A items apply (and have been completed), plus the following.

### 3.1 Key Design Decisions

1. **New `octopus/diagnostics/` directory** — evaluation code lives in its own top-level package, NOT inside `predict/`. This keeps `predict/` focused on model-based analysis and avoids bloating it with diagnostic visualization code.
2. **Plotly replaces Altair** — the existing codebase uses Plotly exclusively. Adding Altair would introduce a second visualization library for no benefit. All evaluation charts are simple (bar, scatter, heatmap) and trivially expressible in Plotly.
3. **pandas replaces DuckDB + Polars** — analysis of the evaluation notebooks shows that DuckDB is used ONLY as a glob-parquet reader (`read_parquet('*/*/*.parquet', hive_partitioning=true)`) and Polars is used only for basic filtering/grouping. Both are fully replaceable with `pd.read_parquet()` + directory iteration + `pd.concat()`. This eliminates two heavy dependencies.
4. **ipywidgets stays as optional** — the `@interact` decorator with `Dropdown` is lightweight and provides genuine interactive value in Jupyter. It will be an optional dependency guarded with try/except.
5. **Only add to `predict/` if no additional imports and it truly makes sense** — nothing from evaluation goes into `predict/`.

### 3.2 Dependency Analysis: Why DuckDB and Polars Are NOT Needed

**DuckDB usage in evaluation notebooks:**
```python
con = duckdb.connect()
df = con.execute(
    f"SELECT * FROM read_parquet('{path}/*/*/optuna*.parquet', hive_partitioning=true)"
).pl()
```

This is equivalent to:
```python
import pandas as pd
from pathlib import Path

dfs = []
for parquet_file in Path(study_path).glob("*/*/optuna*.parquet"):
    parts = parquet_file.relative_to(study_path).parts
    df = pd.read_parquet(parquet_file)
    df["experiment_id"] = int(parts[0].replace("outersplit", ""))
    df["task_id"] = int(parts[1].replace("task", ""))
    dfs.append(df)
result = pd.concat(dfs, ignore_index=True)
```

**Polars usage in evaluation notebooks:**
- `.filter()` → `df[mask]`
- `.with_columns(pl.col("x").cast(pl.Int64))` → `df["x"] = df["x"].astype(int)`
- `.group_by().agg()` → `df.groupby().agg()`
- `.select().unique()` → `df[cols].drop_duplicates()`
- `.to_pandas()` → already pandas

**Conclusion:** Both DuckDB and Polars are used only for trivial operations that pandas handles natively. Removing them eliminates ~200MB of optional dependencies with zero functionality loss.

### 3.3 What the Evaluation Notebooks Provide

The two evaluation notebooks provide a **complementary** analysis layer:

| Aspect | `analyse_study_*.ipynb` (predict/) | `evaluation_*.ipynb` (diagnostics/) |
|--------|-------------------------------------|-------------------------------------|
| **Focus** | Post-hoc model analysis (fresh computation) | Experiment-level diagnostics (saved artifacts) |
| **Technology** | Plotly, pandas, sklearn | **Plotly, pandas, ipywidgets** |
| **Interactivity** | Static (execute all) | **Interactive dropdowns** (`@interact`) |
| **Data source** | Models loaded from joblib (fresh scoring) | **Saved parquet files** only |
| **Granularity** | Task-level (aggregated across outersplits) | **Experiment × task × training** level |
| **Dependencies** | octopus.predict (TaskPredictor) | **Zero model loading** |

### 3.4 Evaluation Notebook Features to Port

#### Classification-specific:
1. **Feature Importance** — interactive bar chart filtered by experiment_id, task_id, training_id, fi_type
2. **Confusion Matrix** — per experiment × task × training, Plotly heatmap + text labels
3. **Optuna Trial Counts** — bar chart: unique trials per model_type × experiment × task
4. **Optuna Trial Values** — scatter plot with cumulative best line (object value vs trial number)
5. **Optuna Hyperparameters** — per model_type: param_value vs target_metric scatter plots

#### Regression-specific:
1. **Prediction vs Ground Truth** — scatter plot with diagonal reference, colored by split
2. **Feature Importance** — same as classification
3. **Optuna insights** — same as classification

### 3.5 Schema Compatibility Issue ⚠️

The evaluation notebooks reference columns from the **main branch** parquet schema:

| Evaluation Notebook Column | Notes |
|---------------------------|-------|
| `experiment_id` | Maps to `outersplit` in current branch |
| `task_id` | Same |
| `training_id` | Inner fold / bag training ID |
| `fi_type` | Feature importance type column |
| `model_type`, `trial`, `hyper_param`, `param_value` | Optuna parquet columns |
| `split` (test/train/dev) | May be `partition` in current branch |

The evaluation notebooks use `feature-importance*.parquet` (with hyphen), while the current branch saves `feature_importances*.parquet` (with underscore). This glob pattern mismatch means the notebooks **will not work** on current branch studies without adaptation.

---

## Part 4: `octopus/diagnostics/` Package Design

### 4.1 Directory Structure

```
octopus/diagnostics/
  __init__.py                → exports StudyEvaluator
  _data_loader.py            → parquet glob-loading with pandas (replaces DuckDB)
  _plots.py                  → Plotly chart functions (replaces Altair)
  evaluator.py               → StudyEvaluator class (main entry point)
```

### 4.2 API Design

```python
from octopus.diagnostics import StudyEvaluator

evaluator = StudyEvaluator("./studies/my_study/")

# Data access (pandas DataFrames)
evaluator.predictions          # loaded from predictions*.parquet across all splits
evaluator.feature_importances  # loaded from feature_importances*.parquet
evaluator.optuna_trials        # loaded from optuna*.parquet

# Interactive plots (require ipywidgets in Jupyter)
evaluator.plot_feature_importance()          # interactive dropdown
evaluator.plot_confusion_matrix()            # classification only
evaluator.plot_predictions_vs_truth()        # regression only
evaluator.plot_optuna_trial_counts()
evaluator.plot_optuna_trials()
evaluator.plot_optuna_hyperparameters()
```

### 4.3 Dependency Strategy

**Required dependencies** (already in octopus):
- `pandas` — data loading, filtering, grouping
- `plotly` — all visualization (already used by predict/)
- `numpy` — numerical ops
- `sklearn` — confusion_matrix (already used by predict/)

**Optional dependencies** (guarded with try/except):
- `ipywidgets` — interactive dropdowns in Jupyter. Without it, plots still work but use default parameters instead of interactive selection.

```toml
# pyproject.toml — NO new dependency group needed
# ipywidgets is already commonly available in Jupyter environments
# If not available, functions degrade gracefully to non-interactive mode
```

**What is NOT needed:**
- ~~DuckDB~~ — replaced by `pd.read_parquet()` + `Path.glob()`
- ~~Polars~~ — replaced by native pandas operations
- ~~Altair~~ — replaced by Plotly (consistent with rest of codebase)

### 4.4 Data Loading Pattern (`_data_loader.py`)

```python
def load_parquet_glob(study_path: Path, pattern: str) -> pd.DataFrame:
    """Load and concatenate parquet files matching a glob pattern.
    
    Extracts outersplit_id and task_id from the directory structure
    (equivalent to DuckDB's hive_partitioning=true).
    """
    dfs = []
    for parquet_file in study_path.glob(pattern):
        parts = parquet_file.relative_to(study_path).parts
        df = pd.read_parquet(parquet_file)
        # Extract hive partition keys from directory names
        for part in parts[:-1]:  # exclude filename
            if part.startswith("outersplit"):
                df["experiment_id"] = int(part.replace("outersplit", ""))
            elif part.startswith("task"):
                df["task_id"] = int(part.replace("task", ""))
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)
```

### 4.5 Graceful Degradation Without ipywidgets

```python
def plot_feature_importance(self, experiment_id=None, task_id=None, ...):
    """Plot feature importance. Interactive if ipywidgets is available."""
    try:
        from ipywidgets import Dropdown, interact
        _HAS_WIDGETS = True
    except ImportError:
        _HAS_WIDGETS = False
    
    if _HAS_WIDGETS and experiment_id is None:
        # Interactive mode: show dropdowns
        @interact(experiment_id=Dropdown(options=...), ...)
        def _plot(**kwargs):
            self._render_fi_chart(**kwargs)
    else:
        # Non-interactive: use provided values or defaults
        self._render_fi_chart(experiment_id=experiment_id or 0, ...)
```

---

## Part 5: Architecture Diagram

```
┌───────────────────────────────────────────────────────────────┐
│  User's Jupyter Notebook                                       │
│                                                                │
│  ┌────────────────────────────┐  ┌────────────────────────────┐│
│  │  analyse_study_*.ipynb     │  │  evaluation_*.ipynb         ││
│  │                            │  │                             ││
│  │  Uses: TaskPredictor       │  │  Uses: StudyEvaluator       ││
│  │  Focus: Model analysis     │  │  Focus: Experiment diags    ││
│  │  Tech: Plotly              │  │  Tech: Plotly + ipywidgets  ││
│  │  Data: Models (fresh)      │  │  Data: Saved parquets only  ││
│  └──────────┬─────────────────┘  └──────────┬─────────────────┘│
│             │                                │                  │
│  ┌──────────▼─────────────────┐  ┌──────────▼─────────────────┐│
│  │  predict/notebook_utils    │  │  diagnostics/evaluator.py   ││
│  │  predict/task_predictor    │  │  diagnostics/_data_loader   ││
│  │  predict/study_io          │  │  diagnostics/_plots         ││
│  │  predict/feature_import    │  │                             ││
│  │  (uses octopus.metrics)    │  │  pandas + plotly only       ││
│  └────────────────────────────┘  └─────────────────────────────┘│
│                                                                │
│                      Study Directory (Disk)                     │
│         config.json, model.joblib, *.parquet, optuna_*         │
└───────────────────────────────────────────────────────────────┘
```

---

## Part 6: Implementation Plan

### Scenario A — predict/ only (COMPLETED ✅)

| Step | Description | Status |
|------|-------------|--------|
| A-1 | Fix `show_confusionmatrix()` to use `predictor.performance_test()` | ✅ Done |
| A-2 | Restore `threshold` parameter in `performance_test()` and CM metrics | ✅ Done |
| A-3 | Fix `_build_performance_dataframe()` and `show_confusionmatrix()` O(N²) concat → list-of-dicts | ✅ Done |
| A-4 | (Optional) Port combined/ensemble FI from old OctoPredict | Deferred |

### Scenario B — `octopus/diagnostics/` package (extends A)

| Step | Description | Effort |
|------|-------------|--------|
| B-1 | Create `octopus/diagnostics/__init__.py` | Small |
| B-2 | Create `octopus/diagnostics/_data_loader.py` — parquet glob-loading with pandas | Small |
| B-3 | Create `octopus/diagnostics/_plots.py` — Plotly chart functions (port from Altair) | Medium |
| B-4 | Create `octopus/diagnostics/evaluator.py` — `StudyEvaluator` class | Medium |
| B-5 | Adapt to current branch parquet schema (column names, glob patterns, file naming) | Small |
| B-6 | Update evaluation notebooks to use `StudyEvaluator` | Small |
| B-7 | Add tests for `StudyEvaluator` | Medium |

**No new required dependencies.** Only `ipywidgets` as an optional graceful-degradation dependency. pandas, plotly, numpy, sklearn are all already in the dependency tree.

---

## Part 7: Summary

### What was done well in the migration:
- Pickle-based loading → parquet/joblib loading ✅
- `OctoExperiment`/`ExperimentInfo` → `StudyLoader`/`OuterSplitLoader`/`TaskPredictor` ✅
- `workflowtask*` → `task*` directory naming ✅
- `testset_performance_overview()` properly delegates to `predictor.performance_test()` ✅
- All plotly visualization code preserved faithfully ✅
- FI caching pattern adapted correctly ✅
- `octopus.metrics` imports are intentional and correct ✅

### Scenario A — Completed ✅
1. ~~Fix `show_confusionmatrix()` scoring consistency~~ — now uses `predictor.performance_test()` ✅
2. ~~Restore `threshold` support~~ — `performance_test(threshold=)` parameter added ✅
3. ~~O(N²) concat~~ — replaced with list-of-dicts pattern in both functions ✅
4. Combined/ensemble FI — deferred (optional)

### Scenario B — Plan for `octopus/diagnostics/`:
1. **New `octopus/diagnostics/` package** — separate from `predict/`
2. **Plotly only** — replaces Altair, consistent with rest of codebase
3. **pandas only** — replaces DuckDB + Polars, zero new heavy dependencies
4. **ipywidgets optional** — graceful degradation to non-interactive mode
5. **Schema adaptation** — evaluation notebooks use old column names and glob patterns
6. **`StudyEvaluator` class** — single entry point for all diagnostic plots

### What does NOT change:
- `octopus.metrics` imports remain — lazy loading to be implemented later
- `TaskPredictor` public API unchanged — no evaluation/diagnostic code goes here
- `predict/notebook_utils.py` — analysis functions stay in predict/, diagnostics go to diagnostics/
- `predict/` gets NO additional imports from this work
- `analyse_study_classification.ipynb` — continues to use TaskPredictor + notebook_utils
