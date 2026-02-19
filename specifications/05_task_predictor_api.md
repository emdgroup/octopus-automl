# TaskPredictor Concept — API & Detailed Specification

**Parent document:** [01_overview.md](01_overview.md)

---

## 6. Proposed API

### 6.1 Constructor

```python
from octopus.predict import TaskPredictor

predictor = TaskPredictor(
    study_path="./studies/my_study/",   # Path to completed study
    task_id=0,                          # Task ID (-1 = last task)
    result_type="best",                 # Results key for filtering (default: "best")
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `study_path` | `str \| Path` | — | Path to the completed study directory |
| `task_id` | `int` | `-1` | Task ID within the workflow. `-1` means last task |
| `module` | `str` | `"octo"` | Module name for filtering results |
| `result_type` | `str` | `"best"` | Results key for filtering saved artifacts. Corresponds to `results_key` in the maincode's `OctoPredict` class. Used when loading predictions, scores, and feature importance parquet files that contain a `result_type` column. Common values: `"best"` (best single model) or `"ensemble_selection"` |

> **Note on `result_type` / `results_key` mapping:**
> - In the maincode's `OctoPredict` class, this parameter is called `results_key`
> - In `TaskPredictor`, it is called `result_type` for consistency with the column name in parquet files
> - The `result_type` is passed through to `OuterSplitLoader` for filtering `predictions.parquet`, `scores.parquet`, and `feature_importances.parquet`
> - The saved `model.joblib` always contains the best bag model regardless of `result_type`

### 6.2 Predict on New Data

```python
predictions = predictor.predict(new_data)            # → np.ndarray (default)
predictions = predictor.predict(new_data, df=True)   # → pd.DataFrame with sample index

probabilities = predictor.predict_proba(new_data)            # → np.ndarray (default)
probabilities = predictor.predict_proba(new_data, df=True)   # → pd.DataFrame with sample index
```

### 6.3 Predict on Test Data (from saved study)

```python
predictions = predictor.predict_test()               # → np.ndarray (default)
predictions = predictor.predict_test(df=True)        # → pd.DataFrame with row_id, prediction, prediction_std, n

probabilities = predictor.predict_proba_test()              # → np.ndarray (default)
probabilities = predictor.predict_proba_test(df=True)       # → pd.DataFrame with row_id, probability, probability_std, n
```

### 6.4 Feature Importances

```python
# On new data — per-experiment and ensemble
fi = predictor.calculate_fi(new_data, fi_type="permutation", n_repeat=10)

# On saved test data (no data argument) — per-experiment and ensemble
fi = predictor.calculate_fi(fi_type="permutation", n_repeat=10)

# Group permutation
fi = predictor.calculate_fi(new_data, fi_type="group_permutation", n_repeat=10)

# SHAP
fi = predictor.calculate_fi(new_data, fi_type="shap", shap_type="kernel")
```

**Return value:** Dictionary containing:
- `fi_table_{fi_type}_exp{id}` — DataFrame per experiment (feature, importance, stddev, p-value, n, ci_low_95, ci_high_95)
- `fi_table_{fi_type}_ensemble` — DataFrame for the ensemble (using `TaskPredictor` itself as the model, same as main branch)

This matches the main branch's `OctoPredict.calculate_fi()` behavior where FI is computed:
1. For each individual outersplit model
2. For the ensemble (using the `TaskPredictor` as the model, which averages predictions across outersplits)

### 6.5 Properties

```python
predictor.study_config       # dict — from config.json
predictor.ml_type            # str — "classification", "regression", or "timetoevent"
predictor.target_metric      # str — e.g., "AUCROC"
predictor.target_assignments # dict — e.g., {"default": "target"}
predictor.positive_class     # int | str | None
predictor.n_outersplits      # int
predictor.outersplit_ids     # list[int]
predictor.feature_cols       # list[str] — union of feature_cols across all outersplits
predictor.feature_groups     # dict[str, list[str]] — from saved feature_groups.json
predictor.classes_           # np.ndarray — class labels (classification only, from first model)
```

---

## 7. TaskPredictor — Detailed Specification

### 7.1 Constructor

- Accepts `study_path`, `task_id`
- Loads study config via `study_io.load_study_config()`
- Resolves `task_id` (if `-1`, uses last task in workflow)
- For each available outersplit:
  - Loads the fitted model via `study_io.load_model()` → stores in `self._models: dict[int, Any]`
  - Loads selected features via `study_io.load_selected_features()` → stores in `self._selected_features: dict[int, list[str]]`
  - Loads feature_cols via `study_io.load_feature_cols()` → stores in `self._feature_cols_per_outersplit: dict[int, list[str]]`
- Loads `feature_groups` from `study_io.load_feature_groups()` (first outersplit)
- Raises `ValueError` if no models found

### 7.2 `predict(data, df=False)`

- For each outersplit: subsets `data[selected_features]` and calls `model.predict()` → each returns `np.ndarray`
- Different outersplits can use different feature sets — `TaskPredictor` handles the subsetting per outersplit
- Aggregates predictions by row index (mean across outersplits)
- **Default (`df=False`):** Returns `np.ndarray` — mean predictions, shape `(n_samples,)`. This is **sklearn-compatible**, allowing `TaskPredictor` to be used as a model in FI functions.
- **`df=True`:** Returns `pd.DataFrame` with columns: sample index, `prediction`, `prediction_std`

### 7.3 `predict_proba(data, df=False)`

- For each outersplit: subsets `data[selected_features]` and calls `model.predict_proba()` → each returns 2D `np.ndarray`
- Averages probability arrays across outersplits
- **Default (`df=False`):** Returns `np.ndarray` — mean probabilities, shape `(n_samples, n_classes)`. This is **sklearn-compatible**.
- **`df=True`:** Returns `pd.DataFrame` with one column per class + `prediction` (argmax), indexed by sample

### 7.4 `predict_test(df=False)`

- For each outersplit: loads test data from disk, subsets features, calls `model.predict()`
- Aggregates by `row_id_col` (mean, std, count)
- **Default (`df=False`):** Returns `np.ndarray` — mean predictions per unique row_id
- **`df=True`:** Returns `pd.DataFrame` with `row_id`, `prediction`, `prediction_std`, `n`

### 7.5 `predict_proba_test(df=False)`

- For each outersplit: loads test data from disk, subsets features, calls `model.predict_proba()`
- Extracts positive class probability
- Aggregates by `row_id_col` (mean, std, count)
- **Default (`df=False`):** Returns `np.ndarray` — mean probabilities per unique row_id
- **`df=True`:** Returns `pd.DataFrame` with `row_id`, `probability`, `probability_std`, `n`

### 7.6 `calculate_fi(data=None, fi_type="permutation", n_repeat=10, shap_type="kernel", feature_groups=None)`

Follows the main branch pattern:

**Step 1 — Per-experiment FI:**
For each outersplit model:
- If `data` is None, use the outersplit's saved test data
- Call the appropriate function from `predict/feature_importance.py`:
  - `"permutation"` → `calculate_permutation_fi(model, data, ...)`
  - `"group_permutation"` → `calculate_group_permutation_fi(model, data, ...)`
  - `"shap"` → `calculate_shap_fi(model, data, ...)`
  - `"group_shap"` → `calculate_group_shap_fi(model, data, ...)`
- Store result as `fi_table_{fi_type}_exp{outersplit_id}`

**Step 2 — Ensemble FI:**
- Use the `TaskPredictor` itself as the model (it returns `np.ndarray` from `predict()` and `predict_proba()`, and exposes `classes_` — making it sklearn-compatible)
- Use `self.feature_cols` (union of feature_cols across all outersplits) as `feature_cols`
- Call the same FI function with `model=self`
- Store result as `fi_table_{fi_type}_ensemble`

**Feature groups:** If `fi_type` is `"group_permutation"` or `"group_shap"`:
- Uses `feature_groups` parameter if provided (override)
- Otherwise uses `self.feature_groups` (loaded from `feature_groups.json`)
- Raises `ValueError` if no groups available from either source

**Returns:** `dict[str, pd.DataFrame]` — all FI tables keyed by name

### 7.7 `TaskPredictor` as sklearn-compatible Model

`TaskPredictor` is sklearn-compatible because:
- `predict(data)` returns `np.ndarray` (not DataFrame)
- `predict_proba(data)` returns `np.ndarray` (not DataFrame)
- `classes_` property exposes class labels (needed by `get_score_from_model()` for classification)
- Feature subsetting is handled internally per outersplit — `TaskPredictor` can accept data with any superset of features

This allows `TaskPredictor` to be passed directly to the FI functions as the `model` parameter for ensemble FI. Same pattern as main branch's `OctoPredict`.

**Note:** With `df=True`, the predict methods return DataFrames with statistics — these are NOT used as sklearn protocol.

### 7.8 Properties

| Property | Source | Description |
|----------|--------|-------------|
| `study_config` | `config.json` | Full study configuration dict |
| `ml_type` | `config["ml_type"]` | `"classification"`, `"regression"`, or `"timetoevent"` |
| `target_metric` | `config["target_metric"]` | e.g., `"AUCROC"` |
| `target_assignments` | `config["prepared"]["target_assignments"]` | e.g., `{"default": "target"}` |
| `positive_class` | `config["positive_class"]` | Positive class for binary classification |
| `row_id_col` | `config["prepared"]["row_id_col"]` | Row identifier column name |
| `n_outersplits` | Count of loaded predictors | Number of successfully loaded outersplits |
| `outersplit_ids` | Keys of predictors dict | Sorted list of outersplit IDs |
| `feature_cols` | Union across all outersplits | All input feature columns (union of `feature_cols.json` across outersplits) |
| `feature_cols_per_outersplit` | Dict per outersplit | `dict[int, list[str]]` — input feature columns per outersplit |
| `feature_groups` | `feature_groups.json` | `dict[str, list[str]]` — loaded from first outersplit (empty dict if not found) |
| `classes_` | First predictor's model | `np.ndarray` — class labels (classification only). Raises `AttributeError` for regression/t2e |

**Important distinction:** `feature_cols` are the **input features** available to the task (what was passed to `fit()`). These are different from `selected_features` (stored in `predictor.json`), which are the **output** of model training — the subset of features the model actually uses. `TaskPredictor` handles feature subsetting internally per outersplit.

### 7.9 Internal State

`TaskPredictor` stores per-outersplit data directly (no intermediate `Predictor` class):

| Internal attribute | Type | Description |
|---|---|---|
| `_models` | `dict[int, Any]` | Fitted sklearn models, keyed by outersplit_id |
| `_selected_features` | `dict[int, list[str]]` | Features each model was trained on, keyed by outersplit_id |
| `_feature_cols_per_outersplit` | `dict[int, list[str]]` | Input feature columns per outersplit |
| `_feature_groups` | `dict[str, list[str]]` | Feature groups (from first outersplit) |

### 7.10 Data Access Methods

| Method | Description |
|--------|-------------|
| `load_test_data(outersplit_id)` | Load test DataFrame for a specific outersplit |
| `load_train_data(outersplit_id)` | Load train DataFrame for a specific outersplit |
| `get_model(outersplit_id)` | Get the fitted model for a specific outersplit |
