# Feature Importance in Octopus Modules

All modules in Octopus inherit from the `Module` base class, which provides standardized methods for extracting feature importances from fitted models. This guide explains how to use the feature importance functionality across different module types.

## Overview

The `Module` base class provides a unified interface for feature importance extraction through the `get_feature_importances()` method. This method supports three different calculation strategies:

1. **Internal** - Uses built-in feature importances from tree-based models
2. **Permutation** - Calculates permutation importance (works with any model)
3. **Coefficients** - Uses coefficient magnitudes from linear models

## Basic Usage

After fitting a module, you can extract feature importances using:

```python
# Fit a module
module.fit(
    data_traindev=train_data,
    data_test=test_data,
    feature_cols=feature_cols,
    study=study,
    outersplit_id=0,
    output_dir=output_dir,
)

# Get feature importances (default: internal method)
importance_df = module.get_feature_importances()
print(importance_df)
```

**Output format:**
```
       feature  importance
0    feature_1    0.450123
1    feature_0    0.320456
2    feature_3    0.120789
3    feature_2    0.108632
```

The returned DataFrame contains two columns:
- `feature`: Feature name
- `importance`: Importance score (higher = more important)

Features are sorted by importance in descending order.

## Methods

### 1. Internal Importance (Tree-based Models)

Best for: Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost

```python
importance_df = module.get_feature_importances(method="internal")
```

This method extracts the built-in `feature_importances_` attribute from tree-based models. It's fast and directly reflects how the model uses features during training.

**Advantages:**
- Very fast (no additional computation)
- Directly reflects model's internal feature usage
- Most interpretable for tree-based models

**Limitations:**
- Only works with models that have `feature_importances_` attribute
- Not available for linear models or other model types

**Example:**
```python
from octopus.modules import Octo
from octopus.types import ModelName

# Configure Octo with RandomForest
octo = Octo(
    task_id=0,
    models=[ModelName.RandomForestClassifier],
    n_trials=50,
)

# Fit the module
octo.fit(...)

# Get internal feature importances
fi_df = octo.get_feature_importances(method="internal")
```

### 2. Permutation Importance (Any Model)

Best for: Any fitted model, especially when internal importances aren't available

```python
importance_df = module.get_feature_importances(
    method="permutation",
    data=validation_data,
    target=validation_target,
)
```

Permutation importance measures how much the model's performance decreases when a feature's values are randomly shuffled. It provides model-agnostic feature importance scores.

**Parameters:**
- `data`: DataFrame with feature columns (required)
- `target`: Target values as Series (required)

**Advantages:**
- Works with any model type
- Model-agnostic (comparable across different models)
- Reflects actual predictive importance

**Limitations:**
- Slower (requires multiple predictions)
- Results can vary slightly between runs
- May be correlated for highly correlated features

**Example:**
```python
from octopus.modules import Mrmr

# Configure and fit MRMR module
mrmr = Mrmr(task_id=1, depends_on=0, n_features=50)
mrmr.fit(...)

# Get permutation importance on test set
fi_df = mrmr.get_feature_importances(
    method="permutation",
    data=test_data,
    target=test_data["target"],
)
```

### 3. Coefficient Importance (Linear Models)

Best for: Logistic Regression, Linear Regression, Ridge, Lasso, ElasticNet

```python
importance_df = module.get_feature_importances(method="coefficients")
```

This method extracts and ranks features by the absolute magnitude of their coefficients in linear models.

**Advantages:**
- Fast (uses existing coefficients)
- Directly interpretable for linear models
- Handles multi-class models (averages across classes)

**Limitations:**
- Only works with models that have `coef_` attribute
- Assumes features are on comparable scales
- Not suitable for non-linear models

**Example:**
```python
from octopus.modules import Octo
from octopus.types import ModelName

# Configure Octo with LogisticRegression
octo = Octo(
    task_id=0,
    models=[ModelName.LogisticRegressionClassifier],
    n_trials=30,
)

# Fit the module
octo.fit(...)

# Get coefficient-based importances
fi_df = octo.get_feature_importances(method="coefficients")
```

## Error Handling

The feature importance methods include comprehensive error checking:

### Unfitted Model Error
```python
module = Octo(task_id=0)
# Forgot to call fit()!

try:
    importance = module.get_feature_importances()
except ValueError as e:
    print(e)  # "Octo must be fitted before getting feature importances"
```

### Incompatible Method Error
```python
# Using LogisticRegression (no feature_importances_ attribute)
octo.fit(...)

try:
    importance = octo.get_feature_importances(method="internal")
except ValueError as e:
    print(e)  # "Model LogisticRegression does not have feature_importances_..."
```

### Missing Parameters Error
```python
try:
    importance = module.get_feature_importances(method="permutation")
    # Forgot to provide data and target!
except ValueError as e:
    print(e)  # "Permutation importance requires data and target parameters"
```

## Module-Specific Examples

### Octo (Optimization Module)

```python
from octopus.modules import Octo
from octopus.types import ModelName

octo = Octo(
    task_id=0,
    models=[ModelName.RandomForestClassifier, ModelName.XGBClassifier],
    n_trials=100,
)

octo.fit(
    data_traindev=train_data,
    data_test=test_data,
    feature_cols=feature_cols,
    study=study,
    outersplit_id=0,
    output_dir=output_dir,
)

# Get internal importances (works if best model is tree-based)
fi_internal = octo.get_feature_importances(method="internal")

# Get permutation importances (works for any best model)
fi_permutation = octo.get_feature_importances(
    method="permutation",
    data=test_data,
    target=test_data["target"],
)
```

### ROC (Feature Selection Module)

```python
from octopus.modules import Roc

roc = Roc(
    task_id=0,
    threshold=0.8,
    correlation_type=CorrelationType.SPEARMAN,
)

selected_features, results = roc.fit(...)

# ROC doesn't have a predictive model (model_ = None)
# Use permutation importance if you have a downstream model
# or implement custom feature ranking
```

### Boruta (Feature Selection Module)

```python
from octopus.modules import Boruta
from octopus.types import ModelName

boruta = Boruta(
    task_id=0,
    model=ModelName.RandomForestClassifier,
    perc=100,
)

boruta.fit(...)

# Boruta uses RandomForest internally
fi_df = boruta.get_feature_importances(method="internal")
```

## Best Practices

### 1. Choose the Right Method

- **Tree-based models**: Use `method="internal"` for speed and interpretability
- **Linear models**: Use `method="coefficients"` for direct coefficient interpretation
- **Any model**: Use `method="permutation"` for model-agnostic importance
- **Feature selection modules**: Check if they have a model_ attribute first

### 2. Validate Importances

Always validate feature importances make sense:

```python
fi_df = module.get_feature_importances(method="internal")

# Check that importances sum to ~1.0 for tree models
total_importance = fi_df["importance"].sum()
print(f"Total importance: {total_importance}")

# Identify top features
top_features = fi_df.head(10)
print(f"Top 10 features:\n{top_features}")
```

### 3. Compare Across Methods

For tree-based models, compare internal and permutation importance:

```python
fi_internal = module.get_feature_importances(method="internal")
fi_permutation = module.get_feature_importances(
    method="permutation",
    data=test_data,
    target=test_data["target"],
)

# Merge and compare
import pandas as pd
comparison = pd.merge(
    fi_internal,
    fi_permutation,
    on="feature",
    suffixes=("_internal", "_permutation"),
)
print(comparison)
```

### 4. Save Importances for Later Analysis

```python
# Get feature importances
fi_df = module.get_feature_importances(method="internal")

# Save to disk
output_path = module.path_results / "feature_importances.parquet"
fi_df.to_parquet(output_path)

# Save to CSV for easy inspection
fi_df.to_csv(module.path_results / "feature_importances.csv", index=False)
```

## Integration with Workflow

Feature importances are automatically calculated and saved during workflow execution through `ModuleResults`. However, you can also access them directly:

```python
from octopus.manager.workflow_runner import WorkflowTaskRunner
from octopus.modules import Octo
from octopus.types import ModelName

# Define workflow
workflow = [
    Octo(task_id=0, models=[ModelName.RandomForestClassifier]),
]

# Run workflow
runner = WorkflowTaskRunner(
    study=study,
    workflow=workflow,
    cpus_per_outersplit=4,
    log_dir=log_dir,
)

runner.run(outersplit_id=0, data_train=train_data, data_test=test_data)

# After workflow completes, you can load modules and get importances
from octopus.modules import Octo

octo_dir = study.output_path / "outersplit0" / "task0" / "module"
loaded_octo = Octo.load(octo_dir)

# Get importances from loaded module
fi_df = loaded_octo.get_feature_importances(method="internal")
```

## Advanced Usage

### Handling GridSearchCV Models

The feature importance methods automatically unwrap `GridSearchCV` objects:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from octopus.types import ModelName

# Octo internally uses GridSearchCV
octo = Octo(task_id=0, models=[ModelName.RandomForestClassifier])
octo.fit(...)

# This automatically extracts best_estimator_ from GridSearchCV
fi_df = octo.get_feature_importances(method="internal")
```

### Custom Feature Importance

For modules with custom importance calculation needs, override the methods:

```python
from octopus.modules import Task
import pandas as pd

class CustomModule(Task):
    def _get_internal_importance(self) -> pd.DataFrame:
        """Custom importance calculation."""
        # Your custom logic here
        custom_scores = self._calculate_custom_scores()

        df = pd.DataFrame({
            "feature": self.selected_features_,
            "importance": custom_scores,
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
```

## Summary

The feature importance functionality in Octopus modules provides:

- **Unified interface** across all modules via `get_feature_importances()`
- **Multiple methods** (internal, permutation, coefficients) for different model types
- **Automatic error handling** with clear, actionable error messages
- **Standardized output** format (DataFrame with feature/importance columns)
- **Easy integration** with workflows and results saving

This makes it easy to understand which features are most important for your models and make informed decisions about feature selection and model interpretation.
