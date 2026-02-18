# TaskPredictor Concept — study_io Specification

**Parent document:** [01_overview.md](01_overview.md)

---

## 4. `octopus/predict/study_io.py` — Streamlined Data Loading

All file-system data loading is consolidated into one module with pure functions. This replaces `analysis/loaders.py` and the loading logic in `analysis/module_loader.py`.

**Design:**
- **Pure functions** instead of classes — simpler, easier to maintain, no `attrs` dependency
- **Self-contained** — no imports from any other octopus module
- **Consistent interface** — every function takes `study_path` + identifiers, returns data

**Functions:**

| Function | Returns | Description |
|----------|---------|-------------|
| `load_study_config(study_path)` | `dict` | Load `config.json` |
| `load_test_data(study_path, outersplit_id)` | `DataFrame` | Load `data_test.parquet` |
| `load_train_data(study_path, outersplit_id)` | `DataFrame` | Load `data_train.parquet` |
| `get_module_dir(study_path, outersplit_id, task_id)` | `UPath` | Get module directory path |
| `get_task_dir(study_path, outersplit_id, task_id)` | `UPath` | Get task directory path |
| `load_model(study_path, outersplit_id, task_id)` | `Any` | Load `module/model.joblib` via `joblib.load()` — returns the fitted sklearn model |
| `load_selected_features(study_path, outersplit_id, task_id)` | `list[str]` | Load from `module/predictor.json` — the features the model was trained on |
| `load_feature_cols(study_path, outersplit_id, task_id)` | `list[str]` | Load `feature_cols.json` — the input feature columns available to the task (must be added to save logic) |
| `load_task_config(study_path, outersplit_id, task_id)` | `dict` | Load `task_config.json` |
| `load_scores(study_path, outersplit_id, task_id)` | `DataFrame` | Load `scores.parquet` |
| `load_predictions(study_path, outersplit_id, task_id)` | `DataFrame` | Load `predictions.parquet` |
| `load_feature_importances(study_path, outersplit_id, task_id)` | `DataFrame` | Load `feature_importances.parquet` |
| `load_feature_groups(study_path, outersplit_id, task_id)` | `dict[str, list[str]]` | Load `module/feature_groups.json` (returns `{}` if not found) |
| `discover_outersplits(study_path)` | `list[int]` | Scan for available `outersplit{N}/` directories |
| `discover_tasks(study_path, outersplit_id)` | `list[int]` | Scan for available `task{N}/` directories |

---

## 4.1 Actual Study Directory Structure

Derived from real output of `wf_octo_mrmr_octo.py`. This is the **only** structure supported — no fallback formats.

```
study_path/
├── config.json                           # Study configuration
├── data.parquet                          # Original input data
├── data_prepared.parquet                 # Prepared data (after validation/preparation)
├── health_check_report.csv              # Data health check results
│
├── outersplit0/
│   ├── data_test.parquet                # Test data for this outersplit
│   ├── data_train.parquet               # Train data for this outersplit (= traindev in main branch)
│   │
│   ├── task0/                           # ML task (e.g., octo)
│   │   ├── task_config.json             # Task configuration (task_id, depends_on, module, etc.)
│   │   ├── selected_features.json       # Features selected by the model (output of training)
│   │   ├── feature_cols.json            # Input features available to the task [TO BE ADDED]
│   │   ├── scores.parquet               # Model scores per metric/result_type
│   │   ├── predictions.parquet          # Model predictions on test data
│   │   ├── feature_importances.parquet  # Training-time feature importances
│   │   ├── module/
│   │   │   ├── model.joblib             # Fitted sklearn model (best bag)
│   │   │   ├── predictor.json           # {"selected_features": [...]}
│   │   │   └── module_state.json        # {"selected_features": [...], "feature_importances": {...}}
│   │   └── results/
│   │       ├── best_bag.pkl             # Best bag object (execution artifact)
│   │       ├── best_bag_performance.json # Best bag performance metrics
│   │       ├── ensel_bag.pkl            # Ensemble selection bag (if applicable)
│   │       └── ensel_scores_scores.json # Ensemble selection scores (if applicable)
│   │
│   ├── task1/                           # Feature selection task (e.g., mrmr)
│   │   ├── task_config.json
│   │   ├── selected_features.json       # Features selected by mrmr
│   │   └── module/
│   │       └── module_state.json        # Module state (selected_features, etc.)
│   │
│   └── task2/                           # Another ML task (e.g., second octo)
│       ├── (same structure as task0)
│       └── ...
│
├── outersplit1/
│   └── (same structure as outersplit0)
└── ...
```

### 4.2 Files Used by TaskPredictor

`TaskPredictor` only reads from a subset of the files above. This is the **complete list** of files it needs:

| File | Required? | Purpose |
|------|-----------|---------|
| `config.json` | **Yes** | ML type, target metric, target assignments, positive class, row_id_col, n_folds_outer, workflow |
| `outersplit{N}/data_test.parquet` | For `predict_test()` and FI on test data | Test data per outersplit |
| `outersplit{N}/data_train.parquet` | For FI (permutation pool) | Train data per outersplit |
| `outersplit{N}/task{M}/module/model.joblib` | **Yes** | The fitted model |
| `outersplit{N}/task{M}/module/predictor.json` | **Yes** | `selected_features` — which features the model uses |
| `outersplit{N}/task{M}/feature_cols.json` | **Yes** | Input feature columns (TO BE ADDED to save logic) |
| `outersplit{N}/task{M}/module/feature_groups.json` | Optional | Feature groups for group FI (TO BE ADDED to save logic) |

**Not used by TaskPredictor** (study execution artifacts only):
- `data.parquet`, `data_prepared.parquet`, `health_check_report.csv`
- `task_config.json`, `module_state.json`
- `scores.parquet`, `predictions.parquet`, `feature_importances.parquet`
- `results/` directory (best_bag.pkl, ensel_bag.pkl, etc.)
- `trials/` directory (optuna trial bags)
- `optuna_*.log`, `optuna_*.parquet`

### 4.3 Notes on File Structure

1. **`predictor.json` vs `module_state.json`:** Both contain `selected_features`. `predictor.json` is the clean, minimal file created by `Predictor.save()`. `module_state.json` is a more detailed module state dump. `TaskPredictor` reads `predictor.json` only.

2. **`feature_cols.json`** does not exist yet — it must be added to the save logic in `workflow_runner.py`.

3. **`feature_groups.json`** does not exist yet — it must be added to the save logic in `ModuleExecution.save()` (see `06_implementation.md` Q5).

4. **ML tasks vs feature selection tasks:** ML tasks (octo, autogluon) have `module/model.joblib` and `results/`. Feature selection tasks (mrmr, rfe, roc) only have `selected_features.json` and `module/module_state.json` — no model. `TaskPredictor` only loads ML tasks.