# Version Stability — What Actually Matters

**Parent document:** [01_overview.md](01_overview.md)  
**Date:** 2025-02-19

---

## 1. The Core Promise

> **A study produced with Octopus v1.1 must be analyzable and usable for prediction with any future version of Octopus (v2.0, v3.0, …)**

This means: `TaskPredictor` and `notebook_utils` must be able to read, load, predict, score, and visualize results from **any** previously saved study — regardless of what Octopus version created it.

---

## 2. Distinguishing Real Risks from Perceived Risks

### 2.1 What "Version Stability" Actually Means

There are two distinct concerns that the previous specs conflated:

| Concern | What it means | Example |
|---------|---------------|---------|
| **Version stability** (user-facing) | Old study files work with new code | Study from v1.1 still loads in v3.0 |
| **Development isolation** (developer-facing) | Refactoring one package doesn't break another | Changing `octopus/manager/` doesn't break `octopus/predict/` |

Import isolation (zero `octopus.*` imports) primarily serves **development isolation**. It's valuable, but it is NOT the same as version stability. A perfectly isolated `predict/` package can still break old studies if:
- Model pickle format changes
- Parquet column names change
- Metric definitions change meaning
- Feature importance algorithms change behavior

Conversely, a `predict/` package that imports from `octopus.metrics` can be perfectly version-stable if the metrics API never breaks.

**The real question is: what are the actual contracts between the saved study files and the code that reads them?**

---

## 3. The Five Layers of Version Stability (Ranked by Risk)

### Layer 1: Model Deserialization — ⚠️ HIGHEST RISK

**What:** `model.joblib` contains a pickled Python object (e.g., `BagClassifier`, `RandomForestClassifier`).

**Why it's the highest risk:** Python pickle embeds the **full module path** of the class. When you `joblib.load()`, Python must be able to import the class from that exact path. If the class moves, renames, or changes its internal structure, deserialization fails.

**What can break:**
- Renaming `octopus.modules.octo.bag.BagClassifier` → old models won't load
- Moving model wrapper classes to a different module path
- Changing class attributes that pickle expects (e.g., adding required `__init__` params)
- Major sklearn version upgrades (sklearn models are also pickled inside `BagClassifier`)
- Python major version changes (pickle protocol compatibility)

**Mitigations:**
1. **Never move or rename model classes** that appear in saved joblib files
2. If classes must move, register `sys.modules` aliases:
   ```python
   # In octopus/__init__.py or predict/compat.py
   import sys
   sys.modules["octopus.modules.octo.bag"] = octopus.modules.octo.bag  # keep old path working
   ```
3. Custom unpickler that maps old module paths to new paths
4. **Test:** Maintain a set of reference `model.joblib` files from each major version; CI loads them all
5. Pin sklearn compatibility ranges in `pyproject.toml`
6. Document the model class paths that are "frozen" and must never change

**Current frozen paths (must remain stable):**
```
octopus.modules.octo.bag.BagClassifier
octopus.modules.octo.bag.BagRegressor
octopus.modules.octo.bag.Bag
# ... any other classes that end up in model.joblib
```

### Layer 2: File Schema Evolution — MEDIUM RISK

**What:** The structure and schema of saved files (JSON, Parquet).

**Files and their schema:**

| File | Format | Schema | Risk |
|------|--------|--------|------|
| `config.json` | JSON | `ml_type`, `n_folds_outer`, `workflow`, `target_metric`, `positive_class`, `prepared.target_assignments`, `prepared.row_id_col` | Medium — fields may be added |
| `predictor.json` | JSON | `{"selected_features": [...]}` | Low — very simple |
| `feature_cols.json` | JSON | `["feat_a", "feat_b", ...]` | Low — very simple (not yet saved, see B2) |
| `feature_groups.json` | JSON | `{"group": ["feat_a", "feat_b"]}` | Low — may not exist in old studies |
| `data_test.parquet` | Parquet | Must have `row_id`, target columns, feature columns | Low — Parquet is self-describing |
| `data_train.parquet` | Parquet | Same as data_test | Low |
| `scores.parquet` | Parquet | `result_type`, `module`, `partition`, `aggregation`, `value` | Medium — columns may evolve |
| `predictions.parquet` | Parquet | Varies by ml_type | Medium |
| `feature_importances.parquet` | Parquet | `feature`, `importance`, `fi_method`, `fi_dataset` | Medium |
| `model.joblib` | joblib/pickle | Depends on model class (see Layer 1) | HIGH |
| `module_state.json` | JSON | Legacy format, predecessor of `predictor.json` | Must keep fallback reading |

**Rules for maintaining schema stability:**
1. **Only add** new keys/columns — never remove or rename existing ones
2. **Code must handle missing keys gracefully** — old studies won't have new fields
3. **Default values** for new fields must produce correct behavior for old studies
4. **Test:** Maintain reference study directories from each major version; CI verifies they load correctly

**Example of correct evolution:**
```python
# BAD: breaks old studies that don't have "feature_groups" key
feature_groups = config["feature_groups"]

# GOOD: graceful fallback for old studies
feature_groups = config.get("feature_groups", {})
```

### Layer 3: Directory Structure Evolution — LOW RISK

**What:** The naming and layout of directories within a study.

**Current structure (only supported format):**
```
study/
  config.json
  outersplit0/
    task0/
      model.joblib
      predictor.json
      data_test.parquet
      data_train.parquet
      ...
    task1/
  outersplit1/
    ...
```

The directory naming convention is `task{id}` (e.g., `task0/`, `task1/`). There is no need to support legacy naming conventions — only the current `task{id}` format is used.

**What can break:** Adding new required directories or files without graceful fallback.

**Mitigation:** When adding new files to the study structure, always handle their absence gracefully for older studies.

### Layer 4: Metric Reproducibility — LOW-MEDIUM RISK

**What:** Given the same predictions and targets, the same metric name must produce the same numeric value.

**This is actually very stable because:**
- Metric functions are from **sklearn**, which has extremely stable APIs
- `roc_auc_score(y_true, y_pred)` will return the same value in sklearn 1.0 and sklearn 2.0
- The metric registry is just a name→function mapping

**What could break (rare):**
- sklearn changes a metric's default parameters (extremely unlikely for established metrics)
- A custom metric (like RMSE wrapper) changes behavior
- Multiclass metrics change `average` parameter defaults

**Mitigation:**
- Pin major sklearn version compatibility
- Store `metric_params` explicitly (already done in `Metric` dataclass)
- The private registry vs. importing metrics is **irrelevant** to this risk — same sklearn functions either way

### Layer 5: Algorithm Reproducibility — LOW RISK

**What:** Feature importance algorithms, ensemble prediction aggregation, etc. produce consistent results.

**For prediction:** `model.predict(data)` is deterministic given the loaded model. No risk here.

**For permutation FI:** Results are stochastic by nature (random permutations). Minor algorithm changes are acceptable. What matters:
- Same metric is used for scoring (covered by Layer 4)
- Same data pool strategy (train + test combined)
- Same statistical calculations (p-values, CIs)

**For SHAP:** Depends on `shap` library version. Pin compatibility range.

---

## 4. Rethinking the Import Isolation Requirement

### 4.1 Original Claim (from previous specs)

> `predict/` must have ZERO imports from any `octopus.*` sub-package

### 4.2 Why This Was Overclaimed

The import isolation serves **development stability** (refactoring protection), not version stability. The actual risks are:

1. If `octopus/__init__.py` changes to break imports → `predict/` breaks **at import time** (development stability issue)
2. If `octopus.metrics.Metric` adds a required field → `predict/` breaks **if it imports Metric** (development stability issue)
3. If `model.joblib` can't deserialize → `predict/` breaks **at load time** (version stability issue)

Items 1-2 are about code coupling, not study compatibility. Item 3 is the real version stability concern, and import isolation doesn't help with it at all.

### 4.3 What Actually Protects Version Stability

| Protection | What it does | Import isolation helps? |
|------------|-------------|----------------------|
| Stable file formats (JSON, Parquet, joblib) | Old files remain readable | No |
| Backward-compatible file reading | Handle missing fields/keys gracefully | No |
| Model class path stability | Old model.joblib files deserialize correctly | No |
| Reference study tests in CI | Verify old studies still load and produce correct results | No |
| Metric function stability | Same metric name → same calculation | Marginally (prevents accidental changes via import chain) |

### 4.4 Revised Recommendation

**Import isolation is still valuable** (for development stability and preventing accidental coupling), but it should not be treated as the primary mechanism for version stability. The actual version stability mechanisms are:

1. **Reference study test suite** (most important)
2. **Model class path freezing + compatibility aliases**
3. **Backward-compatible file reading patterns**
4. **Dependency version pinning** (sklearn, shap, pandas)

---

## 5. Reference Study Test Suite — The Real Safety Net

### 5.1 What It Is

Maintain a set of **real study directories** produced by different versions of Octopus. CI runs `TaskPredictor` and `notebook_utils` against each one, verifying:

1. Study loads without errors
2. Models deserialize correctly
3. Predictions are produced
4. Metrics match expected values (within floating-point tolerance)
5. Feature importance can be computed
6. All notebook_utils functions run without errors

### 5.2 Structure

```
tests/reference_studies/
  v1.0_classification/           ← study produced by Octopus v1.0
    config.json
    outersplit0/
      task0/
        model.joblib
        module_state.json        ← old metadata format (before predictor.json)
        data_test.parquet
        ...
    expected_results.json        ← expected metric values
  
  v1.1_classification/           ← study produced by Octopus v1.1
    config.json
    outersplit0/
      task0/
        model.joblib
        predictor.json           ← new metadata format
        feature_cols.json        ← new file
        data_test.parquet
        ...
    expected_results.json
  
  v1.1_regression/
    ...
```

### 5.3 Test Implementation

```python
# tests/test_version_stability.py
import pytest
from pathlib import Path

REFERENCE_STUDIES = list(Path("tests/reference_studies").glob("v*"))

@pytest.fixture(params=REFERENCE_STUDIES, ids=lambda p: p.name)
def reference_study(request):
    return request.param

def test_study_loads(reference_study):
    """TaskPredictor can load any reference study."""
    from octopus.predict import TaskPredictor
    predictor = TaskPredictor(reference_study, task_id=0)
    assert predictor is not None

def test_predictions_work(reference_study):
    """TaskPredictor can make predictions on test data."""
    from octopus.predict import TaskPredictor
    predictor = TaskPredictor(reference_study, task_id=0)
    preds = predictor.predict_test()
    assert len(preds) > 0

def test_metrics_match(reference_study):
    """Metric values match expected results."""
    import json
    from octopus.predict import TaskPredictor
    
    expected_path = reference_study / "expected_results.json"
    if not expected_path.exists():
        pytest.skip("No expected results for this study")
    
    with open(expected_path) as f:
        expected = json.load(f)
    
    predictor = TaskPredictor(reference_study, task_id=0)
    for metric_name, expected_value in expected.get("metrics", {}).items():
        scores = predictor.performance_test(metrics=[metric_name])
        assert abs(scores[metric_name] - expected_value) < 1e-6

def test_notebook_utils_work(reference_study):
    """All notebook_utils functions run without errors."""
    from octopus.predict import TaskPredictor
    from octopus.predict.notebook_utils import (
        show_study_details,
        testset_performance_overview,
    )
    
    study_info = show_study_details(reference_study, verbose=False)
    assert "config" in study_info
    
    predictor = TaskPredictor(reference_study, task_id=0)
    # Test that performance overview works  
    df = testset_performance_overview(predictor=predictor)
    assert not df.empty
```

### 5.4 When to Add New Reference Studies

- **Every major version release** — save a representative study (classification + regression)
- **When file formats change** — save a study with the new format
- **When model classes change** — save a study with the new model

---

## 6. Model Class Path Strategy

### 6.1 The Problem

When `joblib.dump(bag_classifier, "model.joblib")` is called, Python pickle records:
```
octopus.modules.octo.bag.BagClassifier
```

If this class is moved to `octopus.predict.models.BagClassifier`, old joblib files fail to load:
```
ModuleNotFoundError: No module named 'octopus.modules.octo.bag'
```

### 6.2 The Solution: Compatibility Registry

```python
# octopus/predict/_compat.py
"""Backward-compatible model loading."""

import sys

# Register old module paths so pickle can find classes that were moved
_COMPAT_ALIASES = {
    # "old.module.path": actual_module
    # Add entries here when model classes are moved
}

def register_compatibility_aliases():
    """Register module aliases for backward-compatible model loading."""
    for old_path, new_module in _COMPAT_ALIASES.items():
        if old_path not in sys.modules:
            sys.modules[old_path] = new_module
```

### 6.3 Rules

1. **Never move model classes** that appear in `model.joblib` without adding a compatibility alias
2. **Document all frozen class paths** in this specification
3. **Test deserialization** of reference study models in CI (see §5)

---

## 7. Dependency Version Strategy

### 7.1 Critical Dependencies for Version Stability

| Dependency | Role | Stability risk | Strategy |
|------------|------|----------------|----------|
| **scikit-learn** | Model deserialization, metric functions | Medium (internal model pickle format can change between major versions) | Pin compatible range (e.g., `>=1.3,<2.0`) |
| **pandas** | Parquet I/O, DataFrame operations | Low (Parquet format is stable, API is stable) | Pin compatible range |
| **numpy** | Array operations | Low | Pin compatible range |
| **joblib** | Model file I/O | Low (uses pickle internally) | Pin compatible range |
| **shap** | SHAP FI calculation | Medium (API can change) | Pin compatible range |
| **scipy** | t-test for FI p-values | Low | Pin compatible range |
| **plotly** | Visualization in notebook_utils | Low (output only, not data) | Flexible |

### 7.2 sklearn Model Pickle Compatibility

sklearn models use Python pickle internally. Between major sklearn versions (e.g., 1.x → 2.x), the internal representation of models can change. This means:
- A `RandomForestClassifier` pickled with sklearn 1.3 may not load with sklearn 2.0
- sklearn provides **no backward compatibility guarantee** for pickled models across major versions

**Mitigations:**
1. Pin sklearn to a compatible range (e.g., `>=1.3,<2.0`)
2. When upgrading sklearn major version, re-run reference study tests
3. Document that studies should be re-run (re-fitted) when upgrading sklearn major versions
4. Consider ONNX export as a future alternative to pickle (see §7.3)

### 7.3 ONNX as a Future Alternative to Pickle

**ONNX (Open Neural Network Exchange)** is an open standard for representing machine learning models. It provides a **version-stable, language-agnostic serialization format** that solves many of the pickle/joblib fragility issues.

#### 7.3.1 Why ONNX Is Superior to Pickle for Version Stability

| Aspect | Pickle/Joblib | ONNX |
|--------|---------------|------|
| **Format** | Python-specific binary serialization | Open standard, language-agnostic protobuf |
| **Class path dependency** | Embeds full Python module path (`octopus.modules.octo.bag.BagClassifier`) — class must be importable at that exact path | No Python class dependency — model is a self-contained graph of operations |
| **Python version sensitivity** | Pickle protocol changes between Python versions | No Python dependency — ONNX files are independent of Python version |
| **sklearn version sensitivity** | Internal model attributes change between sklearn major versions → deserialization fails | Converted once to ONNX → stable representation regardless of sklearn version |
| **Cross-language support** | Python only | C++, Java, JavaScript, C#, Rust (via ONNX Runtime) |
| **Inference performance** | Python model objects, no optimization | ONNX Runtime provides optimized inference (often 2-10x faster) |
| **File size** | Often larger (includes Python object overhead) | Often smaller (optimized graph representation) |
| **Forward compatibility** | No guarantee — new Python/sklearn versions may break old pickles | ONNX spec is versioned and backward-compatible by design |

#### 7.3.2 How It Would Work for Octopus

**Conversion at study save time:**
```python
# During ModuleExecution.save() — alongside or instead of model.joblib
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Convert fitted model to ONNX
initial_type = [("X", FloatTensorType([None, n_features]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save as model.onnx
with open(path / "model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

**Loading at prediction time:**
```python
# In TaskPredictor — load ONNX model
import onnxruntime as rt

session = rt.InferenceSession(str(path / "model.onnx"))
input_name = session.get_inputs()[0].name

# Predict
predictions = session.run(None, {input_name: data[feature_cols].values.astype(np.float32)})[0]
```

#### 7.3.3 What Octopus Models Can Be Converted to ONNX

| Model Type | ONNX Support | Notes |
|-----------|-------------|-------|
| `RandomForestClassifier/Regressor` | ✅ Full | Via `skl2onnx` |
| `GradientBoostingClassifier/Regressor` | ✅ Full | Via `skl2onnx` |
| `LogisticRegression` | ✅ Full | Via `skl2onnx` |
| `SVM` variants | ✅ Full | Via `skl2onnx` |
| `ElasticNet`, `Lasso`, `Ridge` | ✅ Full | Via `skl2onnx` |
| `GaussianProcessClassifier/Regressor` | ⚠️ Partial | Custom kernel conversion may be needed |
| `TabularNNClassifier/Regressor` | ⚠️ Depends | If PyTorch-based, use `torch.onnx.export` |
| `BagClassifier/BagRegressor` (ensemble) | ⚠️ Custom | Requires custom converter or save individual models + ensemble logic separately |
| AutoGluon models | ❌ Complex | AutoGluon has its own serialization; ONNX conversion would be model-specific |

#### 7.3.4 Migration Strategy

A realistic migration path would be:

1. **Phase 1 (current): Keep joblib as primary format.** Model is saved as `model.joblib`. Apply all mitigations from §6 (class path freezing, compatibility aliases, reference tests).

2. **Phase 2 (future): Dual-save joblib + ONNX.** During study execution, save both `model.joblib` (for backward compatibility) and `model.onnx` (for forward stability). `TaskPredictor` prefers ONNX when available, falls back to joblib.

3. **Phase 3 (long-term): ONNX as primary, joblib as legacy.** New studies save ONNX by default. `TaskPredictor` still loads old joblib files via fallback.

**Key advantage of dual-save:** No breaking change. Old code ignores `model.onnx`. New code prefers it. Both formats coexist.

#### 7.3.5 Challenges

- **BagClassifier** is a custom ensemble → needs custom ONNX conversion or decomposition (save individual models as ONNX + ensemble averaging logic in Python)
- **predict_proba** output handling differs between ONNX Runtime and sklearn models
- **Preprocessing pipelines** (scalers, imputers) must also be converted to ONNX
- **Additional dependency:** `skl2onnx` + `onnxruntime` would become required (or optional) dependencies

#### 7.3.6 Recommendation

ONNX is the **correct long-term direction** for model serialization stability, but it requires non-trivial work (especially for `BagClassifier` and preprocessing pipelines). For now:
- Implement the pickle-based mitigations (§6) — they work today
- Plan ONNX dual-save as a future milestone
- Track `skl2onnx` support for the specific model types used in Octopus

---

## 8. Complete Version Stability Checklist

### For Every Release

- [ ] Reference study tests pass (all historical studies load and produce correct results)
- [ ] No model class paths were moved without compatibility aliases
- [ ] No file schema fields were removed or renamed
- [ ] New file fields have default values for backward compatibility
- [ ] Dependency versions are compatible with existing saved studies

### For Every Code Change in `predict/`

- [ ] Can read both `predictor.json` and `module_state.json` (legacy)
- [ ] Handles missing files gracefully (e.g., `feature_cols.json`, `feature_groups.json` may not exist)
- [ ] Handles missing JSON keys gracefully (use `.get()` with defaults)
- [ ] Does not write to study directories (read-only access)

### For Major Version Upgrades

- [ ] Save new reference studies with current format
- [ ] Test all historical reference studies still work
- [ ] If sklearn major version changes, test model deserialization
- [ ] Update dependency pins
- [ ] Document any breaking changes and migration path

---

## 9. Revised Architecture Recommendation

Given this analysis, the architecture for `predict/` should focus on:

### 9.1 What Matters Most (in priority order)

1. **Backward-compatible file reading** — handle all historical file formats, naming conventions, missing fields
2. **Model deserialization safety** — compatibility aliases, reference study tests, dependency pinning
3. **Reference study test suite** — the PRIMARY mechanism for version stability assurance
4. **Import isolation** — valuable for development stability, but secondary to the above

### 9.2 Metrics Strategy Revisited

Given this framing, the metrics question becomes less critical:

- **Option B (lazy init) is actually fine** for version stability — the concern was development isolation, not version stability
- **Option D (private copy + sync test) is fine** too — more isolated but more maintenance
- **Either way, the metrics themselves are stable** (they're sklearn functions)
- **The choice should be made based on developer experience**, not version stability

The real version stability protection for metrics is: **same metric name always computes the same value**, which is guaranteed by sklearn's stability, not by import isolation.

### 9.3 What We Should Actually Invest Time In

| Investment | Impact on version stability | Priority |
|-----------|---------------------------|----------|
| Reference study test suite | **Critical** — catches ALL regressions | **P0** |
| Model class path freezing + docs | **High** — prevents the most common breakage | **P0** |
| Backward-compatible file reading | **High** — handles schema evolution | **P0** |
| Import isolation for `predict/` | Medium — prevents development coupling | P1 |
| Private metrics registry | Low — convenience, not stability | P2 |
| Dependency version pins | **High** — prevents sklearn pickle breaks | **P0** |

---

## 10. Summary

**The most important aspects for long-term version stability are:**

1. **🔴 Model deserialization** — Never move model class paths. Add compatibility aliases. Test with reference studies. Pin sklearn version ranges.

2. **🔴 File schema evolution** — Only add fields, never remove/rename. Always use `.get()` with defaults. Handle missing files/keys gracefully.

3. **🔴 Reference study test suite** — Maintain real study directories from each version. CI tests verify they all load, predict, and score correctly. This is the **primary safety net**.

4. **🟡 Dependency management** — Pin compatible ranges. Document that sklearn major upgrades may require re-fitting studies.

5. **🟢 Import isolation** — Good practice for development stability, but does not directly protect version stability. Should not drive architectural decisions at the expense of simplicity.

**The import chain problem (`octopus/__init__.py` loading 60+ modules) is a performance/coupling issue, not a version stability issue.** It should be fixed (with lazy imports), but it's separate from the version stability guarantee.