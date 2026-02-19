# Metrics Strategy — Options for `octopus/predict/`

**Parent document:** [01_overview.md](01_overview.md)  
**Date:** 2025-02-19

---

## 1. Problem Statement

The `octopus/predict/` package follows **import isolation** as a development guideline (no imports from other `octopus.*` sub-packages) and must be **long-term version stable** (see [`11_version_stability.md`](11_version_stability.md) for the full version stability analysis). However, multiple components need metrics calculation:

| Component | Needs Metrics? | What For |
|-----------|---------------|----------|
| `TaskPredictor.performance_test()` | **Yes** | Score test predictions against ground truth |
| `predict/feature_importance.py` | **Yes** | `_get_score_from_model()` — baseline + permuted scores for permutation FI |
| `predict/notebook_utils.py` | **Yes** | `testset_performance_overview()`, `show_confusionmatrix()` — metric calculation per outersplit |
| `Predictor` (modules/predictor.py) | **No** | Only does `predict()` / `predict_proba()` — no scoring |

### 1.1 Why Predictor Does NOT Need Metrics

The `Predictor` class in `octopus/modules/predictor.py` is a **pure prediction wrapper**. It:
- Holds a fitted model + selected features
- Provides `predict(data)` and `predict_proba(data)`
- Provides basic feature importance (`internal`, `permutation` via sklearn, `coefficients`)
- Does **not** score predictions against targets using named metrics

Its `_get_permutation_importance()` uses `sklearn.inspection.permutation_importance` directly — which uses sklearn's internal scoring, not the Octopus `Metrics` registry. This is correct: `Predictor` is an execution-layer class that doesn't need application-level metric names.

### 1.2 Why Feature Importance DOES Need Metrics

The Octopus permutation FI implementation (from `modules/utils.py`) is **superior to sklearn's** because it:
- Supports all Octopus metric names (AUCROC, ACCBAL, R2, etc.)
- Calculates **p-values** via one-sided t-test
- Calculates **95% confidence intervals**
- Uses a combined train+test shuffling pool for more reliable estimates

This implementation calls `get_score_from_model()` internally, which:
1. Looks up the metric name → gets the sklearn function + prediction_type + direction
2. Gets predictions from the model (predict or predict_proba depending on metric)
3. Calls the sklearn function to compute the score
4. Applies direction (negate for minimize metrics)

**Without metric lookup, permutation FI cannot work.**

### 1.3 Why notebook_utils DOES Need Metrics

- `testset_performance_overview()` calls `get_performance_from_model()` for each outersplit × metric
- `show_confusionmatrix()` calls `get_performance_from_model()` for each outersplit × metric

Both need the same metric lookup: name → sklearn function + prediction_type + any metric_params.

### 1.4 The Import Chain Problem

```
from octopus.metrics.utils import get_performance_from_model
  → import octopus.metrics
    → import octopus.metrics.config
      → from octopus.models.config import ML_TYPES, PRED_TYPES, MLType, OctoArrayLike, PredType
        → imports attrs, sklearn.base, numpy, pandas (acceptable)
    → import octopus.metrics.classification, .regression, .multiclass, .timetoevent
      → these import sklearn.metrics functions (acceptable)
    → import octopus.metrics.core
      → from octopus.exceptions import UnknownMetricError (acceptable)
```

BUT: **any** `from octopus.X import Y` triggers `octopus/__init__.py`:
```python
from octopus.study import OctoClassification, OctoRegression, OctoTimeToEvent
```
This loads **60+ modules** — the entire execution stack (study, manager, modules, models, datasplit, Optuna, Ray, attrs, etc.).

---

## 2. What the Metrics Class Actually Provides

The `Metrics` class is a **registry** mapping metric names to `Metric` dataclass instances. Each `Metric` has:

| Field | Type | Example (AUCROC) | Example (R2) | Example (MAE) |
|-------|------|-------------------|--------------|----------------|
| `name` | str | `"AUCROC"` | `"R2"` | `"MAE"` |
| `metric_function` | Callable | `sklearn.metrics.roc_auc_score` | `sklearn.metrics.r2_score` | `sklearn.metrics.mean_absolute_error` |
| `ml_type` | str | `"classification"` | `"regression"` | `"regression"` |
| `higher_is_better` | bool | `True` | `True` | `False` |
| `prediction_type` | str | `"predict_proba"` | `"predict"` | `"predict"` |
| `scorer_string` | str | `"roc_auc"` | `"r2"` | `"neg_mean_absolute_error"` |
| `metric_params` | dict | `{}` | `{}` | `{}` |

**Total registered metrics (currently):**

| Category | Metrics | Count |
|----------|---------|-------|
| Classification | AUCROC, ACC, ACCBAL, LOGLOSS, F1, NEGBRIERSCORE, AUCPR, MCC, PRECISION, RECALL | 10 |
| Multiclass | ACCBAL_MC, AUCROC_MACRO, AUCROC_WEIGHTED | 3 |
| Regression | R2, MAE, MSE, RMSE | 4 |
| Time-to-event | *(currently disabled)* | 0 |
| **Total** | | **17** |

---

## 3. All Options for Dealing with the Metrics Class

### Option A: Private Metric Registry (Inline Copy) — Current Spec Decision

**Approach:** Create a private `_METRIC_REGISTRY` dictionary inside `predict/feature_importance.py` that maps metric names to their sklearn functions, prediction_type, higher_is_better, and metric_params. Copy `_get_score_from_model()` and `_get_performance_from_model()` as private functions.

```python
# predict/feature_importance.py (or predict/_metrics.py)
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, r2_score, ...

_METRIC_REGISTRY = {
    "AUCROC": {"fn": roc_auc_score, "prediction_type": "predict_proba", "higher_is_better": True, "ml_type": "classification", "metric_params": {}},
    "ACC": {"fn": accuracy_score, "prediction_type": "predict", "higher_is_better": True, "ml_type": "classification", "metric_params": {}},
    "R2": {"fn": r2_score, "prediction_type": "predict", "higher_is_better": True, "ml_type": "regression", "metric_params": {}},
    # ... all 17 metrics
}

def _get_score_from_model(model, data, feature_cols, target_metric, target_assignments, positive_class=None):
    """Private copy — no octopus.metrics import."""
    entry = _METRIC_REGISTRY[target_metric]
    # ... same logic as octopus/metrics/utils.py but using entry["fn"] instead of Metrics.get_instance()
```

| Pros | Cons |
|------|------|
| **Zero `octopus.*` imports** — complete isolation | **Duplication** — 17 metric definitions exist in two places |
| Works today, no changes to existing code needed | Must be kept in sync when metrics are added/changed |
| Fully version-stable | More code to maintain in `predict/` |
| Simple to understand and test | No single source of truth for metric registry |

**Maintenance burden:** Low in practice — metrics are rarely added (last change: months ago). When a new metric is added to `octopus/metrics/`, someone must also add it to `_METRIC_REGISTRY`. A test can enforce this.

---

### Option B: Fix `octopus/__init__.py` with Lazy Imports, Then Import Metrics Directly

**Approach:** Change `octopus/__init__.py` from eager imports to lazy `__getattr__`:

```python
# octopus/__init__.py — AFTER fix
import sys

def __getattr__(name):
    if name == "OctoClassification":
        from octopus.study import OctoClassification
        return OctoClassification
    # ... same for OctoRegression, OctoTimeToEvent
    raise AttributeError(f"module 'octopus' has no attribute '{name}'")
```

Then `predict/` can safely import from `octopus.metrics`:
```python
from octopus.metrics.utils import get_score_from_model, get_performance_from_model
```

**Empirically verified** (from spec 06_implementation.md §10.0.1): with lazy init, this loads only **20 stable definition modules**:
- `octopus.metrics.*` (7 files) — metric registry + definitions
- `octopus.models.*` (8 files) — model configs + type aliases (MLType, PredType)
- `octopus.exceptions` (1 file) — simple exception classes
- helpers (4 files) — standard library

**NOT loaded:** `octopus.study`, `octopus.manager`, `octopus.modules`, `octopus.datasplit`, `octopus.logger` — the entire execution stack excluded.

| Pros | Cons |
|------|------|
| **Single source of truth** — one metric registry | Requires changing `octopus/__init__.py` (affects all users) |
| No duplication | `predict/` technically imports `octopus.metrics` and transitively `octopus.models.config` (type aliases) |
| Easier maintenance — add metric once | Violates the strict "zero `octopus.*` imports" rule |
| Cleaner code in `predict/` | 20 modules still loaded (vs. 0 with Option A) |
| | Lazy `__init__.py` could be reverted accidentally |

**Risk mitigation:** Add a CI test that imports `octopus.predict` in isolation and verifies no execution-layer modules are loaded.

---

### Option C: Extract a Standalone `octopus-metrics` Micro-Package

**Approach:** Move the metrics registry (`Metric` dataclass, `Metrics` class, all metric definitions) into a separate lightweight package (`octopus-metrics` or `octopus.metrics_core`). This package has zero dependencies on any `octopus.*` execution code. Both `octopus/predict/` and `octopus/metrics/` import from it.

```
octopus-metrics/          ← standalone package (or octopus/metrics_core/)
  metric.py               ← Metric dataclass (no octopus.models.config dependency)
  registry.py             ← Metrics class
  classification.py       ← metric definitions
  regression.py
  multiclass.py

octopus/predict/          ← imports from octopus-metrics
octopus/metrics/          ← re-exports from octopus-metrics (backward compat)
```

The key change: the `Metric` dataclass currently imports `MLType`, `PredType`, `OctoArrayLike` from `octopus.models.config`. These are just type aliases:
```python
type MLType = Literal["classification", "multiclass", "regression", "timetoevent"]
type PredType = Literal["predict", "predict_proba"]
type OctoArrayLike = np.typing.ArrayLike
```
These would be moved into (or duplicated in) the metrics package, breaking the dependency on `octopus.models`.

| Pros | Cons |
|------|------|
| **Clean separation** — metrics are a true standalone concern | **Significant refactoring** — extract package, update all imports |
| Single source of truth | Adds project complexity (separate package or sub-package) |
| `predict/` imports are clean and explicit | Requires careful backward compatibility |
| Version-stable by design | Over-engineered for 17 metrics that rarely change |

---

### Option D: Hybrid — Private Registry + Sync Test Against `Metrics` Class

**Approach:** Use Option A (private registry) but add an automated test that imports both registries and verifies they match:

```python
# tests/test_metric_registry_sync.py
def test_predict_metrics_match_octopus_metrics():
    """Ensure predict/_metrics.py stays in sync with octopus.metrics."""
    from octopus.metrics import Metrics
    from octopus.predict._metrics import _METRIC_REGISTRY
    
    for name in _METRIC_REGISTRY:
        metric = Metrics.get_instance(name)
        entry = _METRIC_REGISTRY[name]
        assert entry["fn"] == metric.metric_function
        assert entry["prediction_type"] == metric.prediction_type
        assert entry["higher_is_better"] == metric.higher_is_better
        assert entry["ml_type"] == metric.ml_type
        assert entry["metric_params"] == metric.metric_params
    
    # Also check nothing is missing
    all_metrics = set(Metrics.get_all_metrics().keys())
    private_metrics = set(_METRIC_REGISTRY.keys())
    assert all_metrics == private_metrics, f"Missing: {all_metrics - private_metrics}, Extra: {private_metrics - all_metrics}"
```

| Pros | Cons |
|------|------|
| Zero `octopus.*` imports in `predict/` at runtime | Still duplicated code |
| Automated sync enforcement | Test has the cross-dependency (acceptable — tests aren't shipped) |
| Catches drift immediately in CI | Slightly more test infrastructure |
| Best of Option A with safety net | |

---

### Option E: Lazy Import Within `predict/` Only at Metric Resolution Time

**Approach:** `predict/` defines its own metrics functions but lazily falls back to `octopus.metrics` at function call time (not import time):

```python
# predict/_metrics.py
_METRIC_REGISTRY = None

def _get_metric(name):
    global _METRIC_REGISTRY
    if _METRIC_REGISTRY is None:
        try:
            from octopus.metrics import Metrics  # lazy import
            _METRIC_REGISTRY = {
                n: {
                    "fn": Metrics.get_instance(n).metric_function,
                    "prediction_type": Metrics.get_instance(n).prediction_type,
                    "higher_is_better": Metrics.get_instance(n).higher_is_better,
                    "ml_type": Metrics.get_instance(n).ml_type,
                    "metric_params": Metrics.get_instance(n).metric_params,
                }
                for n in Metrics.get_all_metrics()
            }
        except ImportError:
            # Fallback: standalone registry for when octopus isn't fully installed
            _METRIC_REGISTRY = _BUILTIN_REGISTRY  # hardcoded fallback
    return _METRIC_REGISTRY[name]
```

| Pros | Cons |
|------|------|
| Single source of truth when full octopus is installed | Lazy import still triggers `octopus/__init__.py` (60+ modules) on first use |
| Fallback for standalone `predict/` usage | Complex logic (try/except, fallback) |
| No duplication in the common case | Not truly isolated — the heavy import happens, just deferred |
| | Import penalty paid on first `performance_test()` or `calculate_fi()` call |
| | Harder to test and reason about |

**This option does NOT solve the core problem** — the 60+ module import still happens. It just defers it from import time to first use. Unless combined with Option B (lazy `__init__.py`), this is strictly worse than Options A/D.

---

## 4. Recommendation

> **Decision (2025-02-19):** We chose **Option B — Import directly from `octopus.metrics.utils`**.
>
> The lazy `__init__.py` fix is deferred (accepted as a known issue for now), but
> we import directly from `octopus.metrics.utils` and `octopus.metrics` anyway.
> This gives a single source of truth with zero code duplication. The `predict/_metrics.py`
> private registry has been **deleted**.
>
> **Rationale:** The import-isolation guideline was aspirational. In practice:
> - The heavy `octopus/__init__.py` import only matters at startup, not at scoring time
> - Duplicating 17 metric definitions creates maintenance burden with no real benefit
> - The `octopus.metrics` module is stable infrastructure unlikely to break
> - A single source of truth is always preferable to enforced-by-test duplication

### Implemented: **Option B — Import from `octopus.metrics.utils` directly**

Files changed:
1. `predict/feature_importance.py` — imports `get_performance_from_model` from `octopus.metrics.utils` and `Metrics` from `octopus.metrics`
2. `predict/task_predictor.py` — imports `get_performance_from_model` from `octopus.metrics.utils`
3. `predict/notebook_utils.py` — imports `get_performance_from_model` from `octopus.metrics.utils`
4. `predict/_metrics.py` — **deleted** (no longer needed)

### Future improvement: **Fix `octopus/__init__.py` with lazy imports**

When prioritized, fix the eager import in `octopus/__init__.py`:
1. Replace eager imports with `__getattr__` lazy loading
2. This eliminates the 60+ module load when importing `octopus.predict`
3. The direct imports from `octopus.metrics.utils` will then be truly lightweight

### Decision Matrix

| Criterion | A (Copy) | B (Lazy init) | C (Extract) | D (Copy+Test) | E (Lazy import) |
|-----------|----------|---------------|-------------|----------------|-----------------|
| Zero `octopus.*` imports at runtime | ✅ | ❌ (20 modules) | ✅ | ✅ | ❌ (60+ deferred) |
| Single source of truth | ❌ | ✅ | ✅ | ❌ (enforced by test) | ✅ |
| No changes to existing code | ✅ | ❌ | ❌ | ✅ | ✅ |
| Implementation effort | Small | Small | Large | Small | Medium |
| Long-term maintenance | Medium | Low | Low | Low (test catches drift) | High |
| Version stability guarantee | ✅ | ⚠️ (could regress) | ✅ | ✅ | ❌ |

---

## 5. Implementation Plan for Option D

### 5.1 Create `predict/_metrics.py`

Private module containing:
- `_METRIC_REGISTRY` dictionary with all 17 metrics
- `_get_score_from_model()` — private copy from `octopus/metrics/utils.py`
- `_get_performance_from_model()` — private copy from `octopus/metrics/utils.py`
- Helper function `_get_metric(name)` — looks up metric in registry, raises clear error if unknown

### 5.2 Usage in `predict/`

```python
# predict/feature_importance.py
from octopus.predict._metrics import _get_score_from_model

# predict/task_predictor.py  
from octopus.predict._metrics import _get_performance_from_model

# predict/notebook_utils.py
# Uses TaskPredictor.performance_test() which internally uses _get_performance_from_model
```

### 5.3 Sync Test

```python
# tests/test_metric_registry_sync.py
def test_predict_metrics_match_octopus_metrics():
    from octopus.metrics import Metrics
    from octopus.predict._metrics import _METRIC_REGISTRY
    
    all_metrics = set(Metrics.get_all_metrics().keys())
    private_metrics = set(_METRIC_REGISTRY.keys())
    assert all_metrics == private_metrics
    
    for name in all_metrics:
        metric = Metrics.get_instance(name)
        entry = _METRIC_REGISTRY[name]
        assert entry["fn"] == metric.metric_function
        assert entry["prediction_type"] == metric.prediction_type
        assert entry["higher_is_better"] == metric.higher_is_better
        assert entry["ml_type"] == metric.ml_type
        assert entry["metric_params"] == metric.metric_params
```

### 5.4 CI Forbidden Import Check

```python
# tests/test_predict_isolation.py
def test_predict_has_no_octopus_imports():
    """Verify octopus/predict/ has zero octopus.* imports."""
    import ast, pathlib
    
    FORBIDDEN = {"octopus.modules", "octopus.study", "octopus.manager", 
                 "octopus.models", "octopus.metrics", "octopus.datasplit"}
    
    for py_file in pathlib.Path("octopus/predict").glob("**/*.py"):
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for forbidden in FORBIDDEN:
                    assert not node.module.startswith(forbidden), \
                        f"{py_file}: forbidden import 'from {node.module}'"
```

---

## 6. Summary

| Question | Answer |
|----------|--------|
| Does `Predictor` need the metrics class? | **No** — it only does predict/predict_proba, no scoring |
| Does feature importance need the metrics class? | **Yes** — permutation FI needs `_get_score_from_model()` which uses metric lookup |
| Does notebook_utils need metrics calculation? | **Yes** — `testset_performance_overview()` and `show_confusionmatrix()` compute metrics |
| Best approach for `predict/`? | **Option B: Import directly from `octopus.metrics.utils`** (implemented). Lazy `__init__.py` fix deferred as future improvement. |
| Goal: long-term version stability? | Achieved primarily through backward-compatible file reading, model class path stability, reference study tests, and dependency pinning (see `11_version_stability.md`). Import isolation + private metrics copy provide additional development stability. |
