# TaskPredictor Concept — Architecture & File Structure

**Parent document:** [01_overview.md](01_overview.md)

---

## 3. File Structure

```
octopus/
├── predict/                              # Version-stable predict & analyze domain
│   ├── __init__.py                       # Exports: TaskPredictor
│   ├── task_predictor.py                 # TaskPredictor class (includes model loading & feature subsetting)
│   ├── study_io.py                       # Self-contained study data loading (config, data, models)
│   └── feature_importance.py             # FI calculations (adapted from main branch modules/utils.py)
│
├── analysis/                             # DELETED — all functionality moved to predict/
│
├── modules/                              # Study execution code
│   ├── (predictor.py)                    # REMOVE — not replaced; saving logic moves to base.py/workflow_runner
│   ├── base.py                           # Module base classes (study execution only)
│   ├── utils.py                          # Module utilities (study execution only)
│   └── octo/, rfe/, mrmr/, ...           # Module implementations
│
├── metrics/                              # Metrics (used by modules/ — NOT imported by predict/)
│   ├── utils.py                          # get_score_from_model, get_performance_from_model
│   └── ...                               # predict/ uses PRIVATE COPIES of scoring functions (see §10.0, §10.2 in 06_implementation.md)
│
├── study/                                # Study execution code (unchanged)
├── manager/                              # Manager execution code (unchanged)
└── models/                               # Model definitions (unchanged)
```

### 3.1 What Gets Removed from `/modules`

- `octopus/modules/predictor.py` — **REMOVED entirely**. The `Predictor` class is eliminated.
  - Its model loading logic (`model.joblib` + `predictor.json`) is absorbed into `study_io.py` as simple functions: `load_model()`, `load_selected_features_from_predictor()`.
  - Its `predict(data)` feature-subsetting logic is absorbed into `TaskPredictor` directly.
  - Its `save()` logic is replaced by direct `joblib.dump()` + JSON writes in the module execution code (`base.py` or `workflow_runner.py`).
  - Its FI methods (`get_feature_importances`, etc.) are dropped — FI belongs on `TaskPredictor` via `predict/feature_importance.py`.
- **No cross-dependency** between `predict/` and `modules/` — both sides use plain file I/O (joblib + JSON) with no shared class.

### 3.1b What Gets Added to Study Execution

- **`feature_cols.json`** — saved alongside `selected_features.json` in each task directory. Contains the input feature columns available to the task (passed to `fit()`). This is needed by `TaskPredictor` for ensemble FI and data validation. `feature_cols` are different from `selected_features`: `feature_cols` are the inputs, `selected_features` are the outputs of model training.

### 3.2 What Gets Removed/Deprecated from `/analysis`

- `octopus/analysis/loaders.py` — **REPLACED** by `octopus/predict/study_io.py`
- `octopus/analysis/module_loader.py` — **REPLACED** by `TaskPredictor`
- `octopus/analysis/__init__.py` — **UPDATED** to no longer import from `octopus.modules.utils`
