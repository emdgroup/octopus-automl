ase# Clean Design — TaskPredictor, notebook_utils, and Analysis Notebook

**Date:** 2025-02-19  
**Status:** FINAL — Supersedes architectural decisions in specs 01–11 where they conflict  
**Principle:** Simplicity over isolation. Clean code over over-engineering.  
**Based on:** Main branch `octopus/analysis/notebook_utils.py` and `examples/analyse_study_classification.ipynb`

---

## 1. Design Philosophy

1. **TaskPredictor replaces `load_task_modules()` + `Predictor`** — single object that holds all loaded data for a task
2. **notebook_utils preserves all main branch outputs** — same rich visualizations, same tables, same flow
3. **Study-level functions remain unchanged** — `show_study_details()`, `show_target_metric_performance()`, `show_selected_features()` work at the study level and don't need TaskPredictor
4. **Task-level functions take a TaskPredictor** — `testset_performance_overview()`, `plot_aucroc()`, `show_confusionmatrix()` receive a predictor instead of calling `load_task_modules()` internally
5. **FI is computed fresh** — `TaskPredictor.calculate_fi()` replaces loading from saved parquet files. This gives p-values, CIs, and group support that the saved files don't have.

---

## 2. What Changes vs. Main Branch

### 2.1 What Stays the Same

| Aspect | Main Branch | New Design |
|--------|-------------|------------|
| `show_study_details(study_directory)` | Returns rich dict with validation | **Same** — unchanged |
| `show_target_metric_performance(study_info)` | Shows scores from saved parquet | **Same** — uses `StudyLoader` |
| `show_selected_features(study_info)` | Feature count + frequency tables | **Same** — uses `StudyLoader` |
| ROC visualization | Merged + averaged with confidence bands + individual | **Same** output |
| Confusion matrix | Absolute + relative side-by-side per split + metrics | **Same** output |
| Performance table | Per-split + Mean row, `display_table()` via IPython | **Same** output |
| FI bar charts | Plotly bar chart, optional top_n | **Same** output |

### 2.2 What Changes

| Aspect | Main Branch | New Design | Why |
|--------|-------------|------------|-----|
| Model loading | `load_task_modules()` → dict of Predictors | `TaskPredictor(study_path, task_id)` | Single object replaces scattered dicts |
| Function signatures | `fn(study_path, task_id, module, result_type)` | `fn(predictor)` | Predictor created once, reused everywhere |
| `Predictor` class | In `modules/predictor.py` | **Eliminated** — inlined into TaskPredictor | No shared class between predict/ and modules/ |
| FI source | Loaded from saved `feature_importances.parquet` | Computed fresh by `TaskPredictor.calculate_fi()` | p-values, CIs, group support, SHAP |
| Metrics | `from octopus.metrics.utils import get_performance_from_model` | Private `_metrics.py` in predict/ | Import isolation |
| Package location | `octopus/analysis/` | `octopus/predict/` | Version stability boundary |

---

## 3. Actual Disk Structure (from Real Study)

Understanding the actual file layout is critical. From `studies/wf_octo_mrmr_octo/`:

```
studies/wf_octo_mrmr_octo/
  config.json                          ← study config (ml_type, workflow, metrics, etc.)
  data.parquet                         ← original input data
  data_prepared.parquet                ← prepared/imputed data
  health_check_report.csv              ← data health check report
  octo_manager.log                     ← execution log
  outersplit0/
    data_test.parquet                  ← test data (at outersplit level, NOT task level)
    data_train.parquet                 ← train data (at outersplit level)
    task0/                             ← first workflow task (e.g., "step1_octo_full")
      selected_features.json           ← features selected by this task
      scores.parquet                   ← performance scores
      predictions.parquet              ← model predictions
      feature_importances.parquet      ← saved FI (permutation, shap, etc.)
      task_config.json                 ← task-specific config
      optuna_0_0_optuna.log            ← Optuna optimization log
      optuna_0_0_optuna_results.parquet ← Optuna trial results
      module/                          ← subdirectory for module artifacts
        model.joblib                   ← fitted model
        module_state.json              ← module state (selected features, results info)
        predictor.json                 ← predictor metadata
      results/                         ← result artifacts
        best_bag.pkl                   ← best bag of models (pickle)
        best_bag_performance.json      ← best bag performance summary
    task1/                             ← second task (e.g., "step2_mrmr") — feature selection only
      selected_features.json           ← (no model — feature selection only)
      task_config.json
      module/
        module_state.json              ← module state (no model.joblib for FS tasks)
    task2/                             ← third task (e.g., "step3_octo_reduced")
      selected_features.json
      scores.parquet
      predictions.parquet
      feature_importances.parquet
      task_config.json
      optuna_0_2_optuna.log
      optuna_0_2_optuna_results.parquet
      module/
        model.joblib
        module_state.json
        predictor.json
      results/
        best_bag.pkl
        best_bag_performance.json
        ensel_bag.pkl                  ← ensemble selection bag (when enabled)
        ensel_scores_scores.json       ← ensemble selection scores
      trials/                          ← saved Optuna trial bags (for ensemble selection)
        studyoptuna_0_2trial0_bag.pkl
        studyoptuna_0_2trial1_bag.pkl
        ...
  outersplit1/
    data_test.parquet
    data_train.parquet
    task0/
      ...  (same structure)
  outersplit2/ ...
  outersplit3/ ...
  outersplit4/ ...
```

**Key observations:**
- `data_test.parquet` and `data_train.parquet` are at the **outersplit level** (not task level)
- `model.joblib` is inside a `module/` subdirectory within the task
- `selected_features.json` is at the task level
- Not all tasks have models (e.g., task1/mrmr has no `model.joblib` — feature selection only)
- Tasks with `ensemble_selection: true` (task2) have a `trials/` directory and `ensel_*` result files
- Study-level files: `data.parquet` (original), `data_prepared.parquet` (imputed), `health_check_report.csv`, `octo_manager.log`
- `config.json` has `positive_class`, `target_col`, `ml_type`, `target_metric`, `feature_cols`, `workflow` (task list), and `prepared` with `row_id`, `target_assignments`
- 5 outersplits (outersplit0–4), each with 3 tasks (task0: octo full, task1: mrmr, task2: octo reduced)

---

## 4. Architecture

```
octopus/predict/
  __init__.py                → exports TaskPredictor
  task_predictor.py          → TaskPredictor class (replaces load_task_modules + Predictor)
  study_io.py                → file I/O functions (load config, models, data)
  feature_importance.py      → FI algorithms (permutation, group permutation, SHAP)
  _metrics.py                → private metric registry + scoring helpers
  notebook_utils.py          → display/visualization (preserves all main branch outputs)
```

### 4.1 Dependency Graph

```
notebook_utils.py
  ├── study_io.py (StudyLoader — for study-level functions)
  └── task_predictor.py (for task-level functions)
        ├── study_io.py (loads files from disk)
        ├── _metrics.py (metric lookup + scoring)
        └── feature_importance.py (FI algorithms)
              └── _metrics.py (scoring for permutation FI)
```

### 4.2 Import Isolation (Development Guideline)

`octopus/predict/` should not import from `octopus.modules/`, `octopus.study/`, `octopus.manager/`, `octopus.models/`, or `octopus.metrics/`. See `11_version_stability.md` for the full version stability analysis.

---

## 5. `study_io.py` — File I/O Layer

Preserves the main branch's `StudyLoader` and `OuterSplitLoader` pattern (from `analysis/loaders.py`) but simplified. These are moved to `predict/study_io.py`.

**Key design:** Keep the `StudyLoader` and `OuterSplitLoader` classes from main branch because:
- `show_study_details()`, `show_target_metric_performance()`, `show_selected_features()` all use them
- They handle real study structure correctly (outersplit-level data, task-level artifacts, module subdirectory)
- They're already well-designed with no execution code imports

```python
"""File I/O for reading study directories.

Moved from octopus/analysis/loaders.py. Classes are preserved because study-level
notebook functions (show_study_details, show_target_metric_performance, show_selected_features)
depend on them.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
from upath import UPath

class OuterSplitLoader:
    """Loads data for a single outersplit from disk.

    Matches actual disk structure:
    - data_test.parquet, data_train.parquet → at outersplit level
    - model.joblib, predictor.json → inside task/module/ subdirectory
    - selected_features.json, scores.parquet → at task level
    """

    def __init__(
        self,
        study_path: str | UPath,
        outersplit_id: int,
        task_id: int,
        module: str = "octo",
        result_type: str = "best",
    ) -> None:
        self.study_path = UPath(study_path)
        self.outersplit_id = outersplit_id
        self.task_id = task_id
        self.module = module
        self.result_type = result_type

    @property
    def fold_dir(self) -> UPath:
        return self.study_path / f"outersplit{self.outersplit_id}"

    @property
    def task_dir(self) -> UPath:
        return self.fold_dir / f"task{self.task_id}"

    @property
    def module_dir(self) -> UPath:
        return self.task_dir / "module"

    def load_test_data(self) -> pd.DataFrame:
        """Load test data (at outersplit level)."""
        return pd.read_parquet(self.fold_dir / "data_test.parquet")

    def load_train_data(self) -> pd.DataFrame:
        """Load train data (at outersplit level)."""
        return pd.read_parquet(self.fold_dir / "data_train.parquet")

    def load_model(self) -> Any:
        """Load fitted model from module/model.joblib."""
        import joblib
        return joblib.load(self.module_dir / "model.joblib")

    def has_model(self) -> bool:
        """Check if this task has a fitted model."""
        return (self.module_dir / "model.joblib").exists()

    def load_selected_features(self) -> list[str]:
        """Load selected_features.json from task directory."""
        path = self.task_dir / "selected_features.json"
        if not path.exists():
            return []
        with path.open() as f:
            return json.load(f)

    def load_scores(self) -> pd.DataFrame:
        """Load scores.parquet. Returns empty DataFrame if not found."""
        path = self.task_dir / "scores.parquet"
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    def load_predictions(self) -> pd.DataFrame:
        """Load predictions.parquet with optional filtering."""
        path = self.task_dir / "predictions.parquet"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        mask = pd.Series(True, index=df.index)
        if "module" in df.columns and self.module:
            mask &= df["module"] == self.module
        if "result_type" in df.columns and self.result_type:
            mask &= df["result_type"] == self.result_type
        filtered = df[mask]
        return filtered if not filtered.empty else df

    def load_feature_importance(self) -> pd.DataFrame:
        """Load feature_importances.parquet with optional filtering."""
        path = self.task_dir / "feature_importances.parquet"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        mask = pd.Series(True, index=df.index)
        if "module" in df.columns and self.module:
            mask &= df["module"] == self.module
        if "result_type" in df.columns and self.result_type:
            mask &= df["result_type"] == self.result_type
        filtered = df[mask]
        return filtered if not filtered.empty else df


class StudyLoader:
    """Study-level data access. Used by study-level notebook functions."""

    def __init__(self, study_path: str | UPath) -> None:
        self.study_path = UPath(study_path)

    def load_config(self) -> dict[str, Any]:
        with (self.study_path / "config.json").open() as f:
            return json.load(f)

    def get_outersplit_loader(
        self, outersplit_id: int, task_id: int, module: str = "octo", result_type: str = "best"
    ) -> OuterSplitLoader:
        return OuterSplitLoader(self.study_path, outersplit_id, task_id, module, result_type)

    def get_available_outersplits(self) -> list[int]:
        dirs = sorted(
            [d for d in self.study_path.glob("outersplit*") if d.is_dir()],
            key=lambda x: int(x.name.replace("outersplit", "")),
        )
        return [int(d.name.replace("outersplit", "")) for d in dirs]

    def get_task_directories(self, outersplit_id: int) -> list[tuple[int, UPath]]:
        fold_dir = self.study_path / f"outersplit{outersplit_id}"
        if not fold_dir.exists():
            return []
        task_dirs = []
        for task_dir in fold_dir.glob("task*"):
            if task_dir.is_dir():
                task_dirs.append((int(task_dir.name.replace("task", "")), task_dir))
        return sorted(task_dirs)
```

---

## 6. `task_predictor.py` — Replaces `load_task_modules()` + `Predictor`

The main branch's `load_task_modules()` returns:
```python
{outersplit_id: {
    "module": Predictor,       # model + selected_features + predict/predict_proba
    "data_test": DataFrame,
    "data_train": DataFrame,
    "ml_type": str,
    "target_metric": str,
    "target_assignments": dict,
    "positive_class": Any,
    "row_id_col": str,
    "is_ml_module": bool,
}}
```

`TaskPredictor` holds exactly this data but as a single clean object:

```python
"""TaskPredictor — replaces load_task_modules() + Predictor."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from upath import UPath

from octopus.predict.study_io import OuterSplitLoader, StudyLoader
from octopus.predict._metrics import score_model


class TaskPredictor:
    """Unified predictor for a single task across all outer splits.

    Replaces the pattern:
        modules = load_task_modules(study_path, task_id, module, result_type)
        for outersplit_id, module_info in modules.items():
            module_info["module"].predict(data)

    With:
        tp = TaskPredictor(study_path, task_id)
        tp.predict(data)

    Example:
        >>> tp = TaskPredictor("studies/my_study", task_id=2)
        >>> scores = tp.performance_test(metrics=["AUCROC", "ACC"])
        >>> fi = tp.calculate_fi("permutation", n_repeats=10)
    """

    def __init__(
        self,
        study_path: str | UPath,
        task_id: int = -1,
        module: str = "octo",
        result_type: str = "best",
    ) -> None:
        self._study_path = UPath(study_path)
        self._module_name = module
        self._result_type = result_type

        # Load config
        loader = StudyLoader(self._study_path)
        self._config = loader.load_config()

        # Resolve task_id (-1 → last task)
        if task_id < 0:
            task_id = len(self._config["workflow"]) - 1
        self._task_id = task_id

        # Extract config values (same keys as main branch module_info dict)
        self._ml_type = self._config.get("ml_type", "")
        self._target_metric = self._config.get("target_metric", "")
        self._target_col = self._config.get("target_col", "")
        self._target_assignments = self._config.get("prepared", {}).get("target_assignments", {})
        self._positive_class = self._config.get("positive_class")
        self._row_id_col = self._config.get("prepared", {}).get("row_id_col")
        # Fallback: row_id_col from top-level config
        if not self._row_id_col:
            self._row_id_col = self._config.get("row_id_col") or "row_id"

        # Load per-outersplit data
        self._outersplits: list[int] = []
        self._models: dict[int, Any] = {}
        self._selected_features: dict[int, list[str]] = {}
        self._test_data: dict[int, pd.DataFrame] = {}
        self._train_data: dict[int, pd.DataFrame] = {}

        n_outersplits = self._config.get("n_folds_outer", 0)
        for split_id in range(n_outersplits):
            try:
                split_loader = OuterSplitLoader(
                    self._study_path, split_id, self._task_id, module, result_type
                )
                if not split_loader.has_model():
                    continue

                self._outersplits.append(split_id)
                self._models[split_id] = split_loader.load_model()
                self._selected_features[split_id] = split_loader.load_selected_features()
                self._test_data[split_id] = split_loader.load_test_data()
                self._train_data[split_id] = split_loader.load_train_data()
            except (FileNotFoundError, Exception):
                continue

        if not self._outersplits:
            raise ValueError(
                f"No models found for task {task_id}. Check that the study has been run."
            )

        # Feature cols from config (input features available to the task)
        self._feature_cols = self._config.get("feature_cols", [])

        # FI results cache
        self._fi_results: dict[str, pd.DataFrame] = {}

    # ── Properties (match main branch module_info keys) ─────────

    @property
    def ml_type(self) -> str:
        return self._ml_type

    @property
    def target_metric(self) -> str:
        return self._target_metric

    @property
    def target_col(self) -> str:
        return self._target_col

    @property
    def target_assignments(self) -> dict:
        """Target column assignments (as in main branch module_info)."""
        return self._target_assignments

    @property
    def positive_class(self) -> Any:
        return self._positive_class

    @property
    def row_id_col(self) -> str:
        return self._row_id_col

    @property
    def feature_cols(self) -> list[str]:
        return self._feature_cols

    @property
    def n_outersplits(self) -> int:
        return len(self._outersplits)

    @property
    def outersplits(self) -> list[int]:
        return list(self._outersplits)

    @property
    def config(self) -> dict:
        return self._config

    @property
    def classes_(self) -> np.ndarray:
        """Class labels (classification only)."""
        model = self._models[self._outersplits[0]]
        if not hasattr(model, "classes_"):
            raise AttributeError(f"Not a classification model: {type(model).__name__}")
        return model.classes_

    @property
    def fi_results(self) -> dict[str, pd.DataFrame]:
        return self._fi_results

    # ── Per-outersplit access (for notebook_utils) ──────────────

    def get_model(self, outersplit_id: int) -> Any:
        """Get the fitted model for an outersplit."""
        return self._models[outersplit_id]

    def get_selected_features(self, outersplit_id: int) -> list[str]:
        """Get selected features for an outersplit."""
        return self._selected_features[outersplit_id]

    def get_test_data(self, outersplit_id: int) -> pd.DataFrame:
        """Get test data for an outersplit."""
        return self._test_data[outersplit_id]

    def get_train_data(self, outersplit_id: int) -> pd.DataFrame:
        """Get train data for an outersplit."""
        return self._train_data[outersplit_id]

    # ── Prediction ──────────────────────────────────────────────

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict on new data (mean across outer splits)."""
        preds = []
        for split_id in self._outersplits:
            features = self._selected_features[split_id]
            preds.append(self._models[split_id].predict(data[features]))
        return np.mean(preds, axis=0)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Predict probabilities on new data (mean across outer splits)."""
        probas = []
        for split_id in self._outersplits:
            features = self._selected_features[split_id]
            probas.append(self._models[split_id].predict_proba(data[features]))
        return np.mean(probas, axis=0)

    # ── Scoring ─────────────────────────────────────────────────

    def performance_test(self, metrics: list[str] | None = None) -> pd.DataFrame:
        """Score test predictions per outer split.

        Replaces:
            for outersplit_id, module_info in modules.items():
                get_performance_from_model(module_info["module"].model_, ...)

        Returns:
            DataFrame with columns: outersplit, metric, score.
        """
        if metrics is None:
            metrics = [self._target_metric]

        target_col = list(self._target_assignments.values())[0] if self._target_assignments else self._target_col

        rows = []
        for split_id in self._outersplits:
            model = self._models[split_id]
            features = self._selected_features[split_id]
            test = self._test_data[split_id]

            for metric_name in metrics:
                score = score_model(
                    model=model,
                    data=test,
                    feature_cols=features,
                    target_col=target_col,
                    metric_name=metric_name,
                    positive_class=self._positive_class,
                )
                rows.append({"outersplit": split_id, "metric": metric_name, "score": score})

        return pd.DataFrame(rows)

    # ── Feature Importance ──────────────────────────────────────

    def calculate_fi(
        self,
        fi_type: str = "permutation",
        *,
        n_repeats: int = 10,
        feature_groups: dict[str, list[str]] | None = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Calculate feature importance across all outer splits.

        Replaces loading from saved feature_importances.parquet.
        Provides p-values, CIs, and group permutation support.
        """
        from octopus.predict.feature_importance import (
            calculate_fi_permutation,
            calculate_fi_shap,
        )

        target_col = list(self._target_assignments.values())[0] if self._target_assignments else self._target_col

        if fi_type in ("permutation", "group_permutation"):
            result = calculate_fi_permutation(
                models=self._models,
                selected_features=self._selected_features,
                test_data=self._test_data,
                train_data=self._train_data,
                target_col=target_col,
                target_metric=self._target_metric,
                positive_class=self._positive_class,
                n_repeats=n_repeats,
                random_state=random_state,
                feature_groups=feature_groups if fi_type == "group_permutation" else None,
            )
        elif fi_type == "shap":
            result = calculate_fi_shap(
                models=self._models,
                selected_features=self._selected_features,
                test_data=self._test_data,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown fi_type '{fi_type}'. Use 'permutation', 'group_permutation', or 'shap'.")

        self._fi_results[fi_type] = result
        return result
```

---

## 7. `_metrics.py` — Private Metric Registry

Same as previous design. Self-contained metric lookup with `score_model()` function. Maps metric names → sklearn functions + metadata. See `10_metrics_strategy.md` for the full analysis (Option D: private registry + sync test).

Key functions:
- `get_metric(name) → dict` — lookup metric by name
- `compute_score(y_true, y_pred, metric_name) → float` — compute score from arrays
- `score_model(model, data, feature_cols, target_col, metric_name, positive_class) → float` — score a model end-to-end

---

## 8. `feature_importance.py` — FI Algorithms

Same as previous design. Stateless functions:
- `calculate_fi_permutation(models, selected_features, test_data, train_data, ...) → DataFrame`
- `calculate_fi_shap(models, selected_features, test_data, ...) → DataFrame`

Returns DataFrames with columns: `feature`, `importance_mean`, `importance_std`, `p_value`, `ci_lower`, `ci_upper`.

---

## 9. `notebook_utils.py` — Preserving All Main Branch Outputs

This is the most important file. It must produce the **exact same outputs** as the main branch notebook. The only change is: task-level functions receive a `TaskPredictor` instead of calling `load_task_modules()` internally.

### 9.1 Study-Level Functions (UNCHANGED from main branch)

These functions work at the study level and don't need models. They use `StudyLoader` directly:

```python
def show_study_details(study_directory: str | Path, verbose: bool = True) -> dict:
    """Unchanged from main branch.
    Validates study structure, prints info, returns rich dict.
    """
    # Exact same implementation as main branch

def show_target_metric_performance(study_info: dict, details: bool = False) -> list[pd.DataFrame]:
    """Unchanged from main branch.
    Loads scores from saved parquet, displays per task/key.
    """
    # Exact same implementation, using _build_performance_dataframe()

def show_selected_features(study_info: dict, sort_task=None, sort_key=None) -> tuple:
    """Unchanged from main branch.
    Feature count + frequency tables.
    """
    # Exact same implementation

def _build_performance_dataframe(study_info: dict) -> pd.DataFrame:
    """Unchanged from main branch.
    Internal helper for study-level performance data.
    """
    # Exact same implementation using StudyLoader
```

### 9.2 Task-Level Functions (Take TaskPredictor)

These functions previously called `load_task_modules()` internally. Now they take a `TaskPredictor`:

```python
def testset_performance_overview(
    predictor: TaskPredictor,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Display test performance metrics across all outersplits.

    Replaces main branch version that took (study_path, task_id, module, result_type, metrics).

    Output is identical: per-outersplit rows + Mean row, metrics as columns.
    """
    if metrics is None:
        if predictor.ml_type == "classification":
            metrics = ["AUCROC", "ACCBAL", "ACC"]
        elif predictor.ml_type == "regression":
            metrics = ["R2", "MAE", "RMSE"]
        else:
            metrics = []

    print("Performance on test dataset (pooling)")

    # performance_test returns long format; pivot to wide
    scores_long = predictor.performance_test(metrics=metrics)
    df = scores_long.pivot(index="outersplit", columns="metric", values="score")

    # Reorder columns to match metrics order
    df = df[metrics]

    # Add Mean row
    df.loc["Mean"] = df.mean()

    display_table(df)
    return df


def plot_aucroc(
    predictor: TaskPredictor,
    figsize: tuple[int, int] = (8, 8),
    dpi: int = 100,
    show_individual: bool = False,
) -> None:
    """Plot ROC curves: merged, averaged with CI bands, and optionally individual.

    Produces identical output to main branch:
    1. Merged ROC curve (all predictions pooled)
    2. Averaged ROC curve (mean ± 1 std dev)
    3. Individual ROC curves per outersplit (if show_individual=True)
    """
    if predictor.ml_type != "classification":
        raise ValueError("AUCROC plots are only available for classification tasks")

    width_px, height_px = int(figsize[0] * 80), int(figsize[1] * 80)

    # Collect predictions per outersplit (same pattern as main branch _get_predictions_df)
    predictions_list = []
    roc_data = []
    mean_fpr = np.linspace(0, 1, 100)

    for split_id in predictor.outersplits:
        model = predictor.get_model(split_id)
        features = predictor.get_selected_features(split_id)
        data_test = predictor.get_test_data(split_id)
        target_col = list(predictor.target_assignments.values())[0]

        probas = model.predict_proba(data_test[features])
        pos_idx = list(model.classes_).index(predictor.positive_class)
        probabilities = probas[:, pos_idx]

        df_pred = pd.DataFrame({
            "row_id": data_test[predictor.row_id_col] if predictor.row_id_col in data_test.columns else range(len(data_test)),
            "prediction": model.predict(data_test[features]),
            "probabilities": probabilities,
            "target": data_test[target_col],
        })
        predictions_list.append(df_pred)

        fpr, tpr, _ = roc_curve(df_pred["target"], df_pred["probabilities"], drop_intermediate=True)
        auc_score = float(auc(fpr, tpr))
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        roc_data.append((split_id, fpr, tpr, auc_score, interp_tpr))

    # === PLOT 1: MERGED === (same as main branch)
    # === PLOT 2: AVERAGED WITH CI === (same as main branch)
    # === PLOT 3: INDIVIDUAL === (same as main branch, if show_individual)
    # ... exact same plotly code as main branch ...


def show_confusionmatrix(
    predictor: TaskPredictor,
    threshold: float = 0.5,
    metrics: list[str] | None = None,
) -> None:
    """Display confusion matrices and performance metrics per outersplit.

    Produces identical output to main branch:
    - Absolute + relative confusion matrix side-by-side (plotly subplots)
    - Performance metrics per outersplit
    - Overall mean performance summary
    """
    if metrics is None:
        metrics = ["AUCROC", "ACCBAL", "ACC", "F1", "AUCPR", "NEGBRIERSCORE"]

    if predictor.ml_type != "classification":
        raise ValueError("show_confusionmatrix() is only applicable for classification tasks")

    # Same implementation as main branch, but using predictor.get_model/get_test_data/etc.
    # instead of module_info["module"] / module_info["data_test"]
    # ... exact same plotly subplot code ...
```

### 9.3 FI Functions (New — Computed Fresh)

```python
def show_overall_fi_table(
    predictor: TaskPredictor,
    fi_type: str = "permutation",
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Display feature importance table.

    Replaces main branch version that loaded from saved parquet.
    Now computes FI fresh using TaskPredictor.calculate_fi().

    Returns:
        DataFrame with columns: feature, importance_mean, importance_std,
        p_value, ci_lower, ci_upper (sorted by importance, descending).
    """
    # Use cached result if available
    if fi_type in predictor.fi_results:
        fi_df = predictor.fi_results[fi_type]
    else:
        fi_df = predictor.calculate_fi(fi_type, n_repeats=n_repeats)

    display_table(fi_df)
    return fi_df


def show_overall_fi_plot(
    predictor: TaskPredictor,
    fi_type: str = "permutation",
    n_repeats: int = 10,
    top_n: int | None = None,
) -> None:
    """Display bar chart of feature importance.

    Same plotly bar chart as main branch but with CI error bars (new).
    """
    fi_df = show_overall_fi_table(predictor, fi_type=fi_type, n_repeats=n_repeats)

    if top_n is not None:
        fi_df = fi_df.head(top_n)
        title = f"Top {top_n} Feature Importances ({fi_type})"
    else:
        title = f"Feature Importances ({fi_type})"

    fig = go.Figure(
        data=go.Bar(
            x=fi_df["feature"],
            y=fi_df["importance_mean"],
            marker={"color": "royalblue"},
            # Add error bars if available
            error_y=dict(
                type="data",
                symmetric=False,
                array=(fi_df["ci_upper"] - fi_df["importance_mean"]).values,
                arrayminus=(fi_df["importance_mean"] - fi_df["ci_lower"]).values,
            ) if "ci_lower" in fi_df.columns else None,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="Importance Value",
        xaxis={"tickangle": -45, "tickfont": {"size": 10}},
        height=600,
        width=max(800, len(fi_df) * 20),
        showlegend=False,
    )
    fig.show()
```

---

## 10. The Notebook — `analyse_study_classification.ipynb`

Preserves the same flow and sections as main branch. Only difference: TaskPredictor is created once and passed to task-level functions.

```python
# Cell 1 (markdown): Analyze Study (Binary Classification)

# Cell 2 (markdown): ## Imports

# Cell 3 (code):
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

# Cell 4 (markdown): ## Input

# Cell 5 (code):
# INPUT: Select study
study_directory = "../studies/wf_octo_mrmr_octo/"

# INPUT: Select task for detailed analysis
task_id = -1  # -1 = last task (same as main branch default)

# Cell 6 (markdown): ## Study Details

# Cell 7 (code):
study_info = show_study_details(study_directory)

# Cell 8 (markdown): ## Target Metric Performance for all Tasks

# Cell 9 (code):
performance_tables = show_target_metric_performance(study_info, details=False)

# Cell 10 (markdown): ## Selected Features Summary

# Cell 11 (code):
feature_table, feature_frequency_table, raw_feature_table = show_selected_features(
    study_info, sort_task=None, sort_key=None
)

# Cell 12 (markdown): ## Model Performance on Test Dataset

# Cell 13 (code):
# Create TaskPredictor for the selected task
tp = TaskPredictor(study_directory, task_id=task_id)

# Cell 14 (code):
metrics = ["AUCROC", "ACCBAL", "ACC", "F1", "AUCPR", "NEGBRIERSCORE"]
testset_performance = testset_performance_overview(tp, metrics=metrics)

# Cell 15 (markdown): ### AUCROC Plots

# Cell 16 (code):
plot_aucroc(tp, show_individual=True)

# Cell 17 (markdown): ### Confusion Matrix

# Cell 18 (code):
show_confusionmatrix(tp, threshold=0.5, metrics=metrics)

# Cell 19 (markdown): ### Test Feature Importances

# Cell 20 (markdown): #### Permutation Feature Importances

# Cell 21 (code):
# Compute fresh permutation FI (with p-values and CIs — improvement over main branch)
fi_perm = tp.calculate_fi("permutation", n_repeats=10)
fi_perm.head(10)

# Cell 22 (code):
show_overall_fi_plot(tp, fi_type="permutation")

# Cell 23 (code):
show_overall_fi_plot(tp, fi_type="permutation", top_n=20)

# Cell 24 (markdown): #### SHAP Feature Importances

# Cell 25 (code):
fi_shap = tp.calculate_fi("shap", shap_type="kernel")
fi_shap.head(10)

# Cell 26 (code):
show_overall_fi_plot(tp, fi_type="shap")
```

### Key Differences from Main Branch Notebook

| Cell | Main Branch | New Design |
|------|-------------|------------|
| Imports | `from octopus.analysis.notebook_utils import ...` | `from octopus.predict.notebook_utils import ...` + `TaskPredictor` |
| Input | `task_id, module, result_type` as separate vars | `task_id` only (module/result_type set on TaskPredictor) |
| Cells 1–11 | Study-level functions with `study_info` | **Identical** |
| Cell 13 | N/A | **New:** `tp = TaskPredictor(study_directory, task_id=task_id)` |
| Cell 14 | `testset_performance_overview(study_path, task_id, module, result_type, metrics)` | `testset_performance_overview(tp, metrics=metrics)` |
| Cell 16 | `plot_aucroc(study_directory, task_id, module, result_type, ...)` | `plot_aucroc(tp, ...)` |
| Cell 18 | `show_confusionmatrix(study_directory, task_id, module, result_type, ...)` | `show_confusionmatrix(tp, ...)` |
| Cells 21–26 | `show_overall_fi_table(study_directory, ..., fi_method, fi_dataset)` (from disk) | `tp.calculate_fi("permutation")` (fresh) + `show_overall_fi_plot(tp, ...)` |

---

## 11. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Keep study-level functions unchanged** | `show_study_details`, `show_target_metric_performance`, `show_selected_features` don't need models — no reason to force TaskPredictor on them |
| **TaskPredictor takes `module` and `result_type`** | Main branch passes these through to `load_task_modules`. TaskPredictor absorbs them as constructor params |
| **`task_id=-1` defaults to last task** | Matches main branch `load_task_modules()` behavior |
| **Per-outersplit accessors** (`get_model`, `get_test_data`, etc.) | `plot_aucroc` and `show_confusionmatrix` need per-split model access for detailed visualizations (ROC per split, CM per split). Public accessors are cleaner than accessing `_` attributes |
| **FI computed fresh, not from disk** | Provides p-values, CIs, group permutation — features the saved parquet files don't have |
| **FI caching on TaskPredictor** | `calculate_fi("permutation")` stores result in `predictor.fi_results["permutation"]`. `show_overall_fi_plot` reuses cached result. Avoids recomputing for table + plot |
| **Preserve main branch plotly code** | The ROC averaged + CI bands, the absolute+relative CM side-by-side — these are polished visualizations. Preserve them exactly. |
| **StudyLoader/OuterSplitLoader preserved as classes** | Study-level functions use them. They're well-designed. Moving to predict/ but keeping the same API. |
| **`display_table()` helper preserved** | Uses IPython.display when available, print otherwise. Same as main branch. |

---

## 12. Migration from Main Branch Code

### 12.1 File Moves

| Main Branch | New Location | Changes |
|------------|-------------|---------|
| `analysis/loaders.py` | `predict/study_io.py` | Remove `attrs` dependency, simplify slightly |
| `analysis/notebook_utils.py` | `predict/notebook_utils.py` | Task-level functions take `TaskPredictor` arg |
| `analysis/module_loader.py` | **Deleted** | Replaced by `TaskPredictor.__init__()` |
| `analysis/__init__.py` | Deprecation stub or removed | Re-exports from `predict/` for backward compat |
| `modules/predictor.py` | **Not moved to predict/** | Stays in modules/ but predict/ doesn't use it |
| N/A (new) | `predict/task_predictor.py` | New file |
| N/A (new) | `predict/_metrics.py` | New file (private metric registry) |
| N/A (new) | `predict/feature_importance.py` | New file (FI algorithms) |

### 12.2 Import Changes in Notebook

```python
# Main branch:
from octopus.analysis.notebook_utils import show_study_details, testset_performance_overview, ...

# New:
from octopus.predict import TaskPredictor
from octopus.predict.notebook_utils import show_study_details, testset_performance_overview, ...
```

### 12.3 Backward Compatibility

`octopus/analysis/__init__.py` can re-export from `predict/` for backward compatibility:
```python
# octopus/analysis/__init__.py — deprecation stub
from octopus.predict.notebook_utils import *  # noqa
from octopus.predict.study_io import StudyLoader, OuterSplitLoader  # noqa
```

---

## 13. Testing Strategy

```python
# tests/test_task_predictor.py
class TestTaskPredictor:
    def test_loads_study(self, study_path):
        tp = TaskPredictor(study_path, task_id=0)
        assert tp.n_outersplits > 0

    def test_last_task_default(self, study_path):
        tp = TaskPredictor(study_path, task_id=-1)
        assert tp._task_id == len(tp.config["workflow"]) - 1

    def test_predict_returns_array(self, study_path):
        tp = TaskPredictor(study_path, task_id=0)
        test_data = tp.get_test_data(tp.outersplits[0])
        preds = tp.predict(test_data)
        assert isinstance(preds, np.ndarray)

    def test_performance_test(self, study_path):
        tp = TaskPredictor(study_path, task_id=0)
        scores = tp.performance_test(metrics=["ACCBAL"])
        assert "score" in scores.columns
        assert len(scores) == tp.n_outersplits

    def test_calculate_fi(self, study_path):
        tp = TaskPredictor(study_path, task_id=0)
        fi = tp.calculate_fi("permutation", n_repeats=3)
        assert "feature" in fi.columns
        assert "importance_mean" in fi.columns
        assert "p_value" in fi.columns

# tests/test_metrics_sync.py
def test_predict_metrics_match_octopus_metrics():
    """Verify private registry matches octopus.metrics (Option D sync test)."""
    from octopus.metrics import Metrics
    from octopus.predict._metrics import _REGISTRY
    # ... verify all entries match ...

# tests/test_notebook_utils.py
def test_show_study_details(study_path):
    info = show_study_details(study_path, verbose=False)
    assert "config" in info
    assert "ml_type" in info

def test_testset_performance_overview(study_path):
    tp = TaskPredictor(study_path, task_id=0)
    df = testset_performance_overview(tp, metrics=["ACCBAL"])
    assert "ACCBAL" in df.columns
    assert "Mean" in df.index

# tests/test_version_stability.py — see 11_version_stability.md §5
```

---

## 14. File Summary

| File | Lines (est.) | Purpose |
|------|-------------|---------|
| `predict/__init__.py` | 5 | Export TaskPredictor |
| `predict/task_predictor.py` | ~200 | Core class (replaces load_task_modules + Predictor) |
| `predict/study_io.py` | ~200 | StudyLoader + OuterSplitLoader (moved from analysis/loaders.py) |
| `predict/_metrics.py` | ~100 | Private metric registry + scoring |
| `predict/feature_importance.py` | ~200 | FI algorithms (permutation, SHAP) |
| `predict/notebook_utils.py` | ~600 | All visualization functions (preserving main branch outputs) |
| **Total** | **~1300** | Complete predict + analysis package |

The increase from ~785 to ~1300 lines reflects preserving the rich main branch notebook outputs (averaged ROC with CI bands, absolute+relative confusion matrices, study validation, performance tables, feature frequency tables).