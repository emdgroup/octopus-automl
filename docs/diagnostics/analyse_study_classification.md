# Analyze Study (Binary Classification)

- version 0.4
- 1.3.2026


## ToDo

- [done] Study details
- [done] Target metric performance on all tasks
- [done] Selected features summary
- [done] Model performance on test dataset for a given task
- [done] AUCROC plots
- [done] Confusion matrix
- [done] Per-split predictions and probabilities (df=True)
- [done] Individual test feature importances (table + plot)
- [done] Merged test feature importances (table + plot)
- Summary confusion matrix
- create tests for notebook utils
- beeswarm plot (individual + merged!)


## Imports



```
from octopus.predict import TaskPredictorTest
from octopus.predict.notebook_utils import (
    display_table,
    find_latest_study,
    show_aucroc_plots,
    show_confusionmatrix,
    show_overall_fi_plot,
    show_overall_fi_table,
    show_selected_features,
    show_study_details,
    show_target_metric_performance,
    show_testset_performance,
)
```

## Input



```
# INPUT: Select study by name prefix (automatically picks the latest timestamped run)
study_name_prefix = "wf_octo_mrmr_octo"
study_directory = find_latest_study("../studies", study_name_prefix)
print(f"Using study: {study_directory}")
```

## Study Details



```
# Call the utility function to display and validate study details
study_info = show_study_details(study_directory)
```

## Target Metric Performance for all Tasks



```
# Display performance (target metric) for all workflow tasks
# Set report_test=True to include test-set columns (test_avg, test_ensemble)
performance_tables = show_target_metric_performance(study_info)
```

## Selected Features Summary



```
# Display the number of selected features across outer splits and tasks
sort_by_task = None
sort_by_key = None
feature_table, feature_frequency_table, raw_feature_table = show_selected_features(
    study_info, sort_task=sort_by_task, sort_key=sort_by_key
)
```

## Model Performance on Test Dataset for a given Task



```
# Input: selected metrics for performance overview
metrics = ["AUCROC", "ACCBAL", "ACC", "F1", "AUCPR", "NEGBRIERSCORE"]
print("Selected metrics: ", metrics)
```

### Test performance for given task and selected metrics



```
# load predictor object (TaskPredictorTest includes stored test data)
task_predictor_test = TaskPredictorTest(study_path=study_info["path"], task_id=0, result_type="best")
testset_performance = show_testset_performance(predictor=task_predictor_test, metrics=metrics)
```

### AUCROC Plots



```
show_aucroc_plots(task_predictor_test, show_individual=True)
```

### Confusion Matrix



```
show_confusionmatrix(task_predictor_test, threshold=0.5, metrics=metrics)
```

### Test Feature Importances


#### Calculate Permutation Feature Importances



```
# Permutation feature importances on test data using final models
# calculate_fi() returns the full FI table (per-split + ensemble rows)
print("PFI calculation running.....")
fi_table_perm = task_predictor_test.calculate_fi(fi_type="group_permutation", n_repeats=3)
```

#### Ensemble Feature Importances (table + plot)



```
# show_overall_fi_table filters to fi_source=="ensemble" automatically
fi_ensemble_perm = show_overall_fi_table(fi_table_perm)
fi_ensemble_perm.head(10)
```


```
show_overall_fi_plot(fi_table_perm)
```

#### Individual Per-Split Feature Importances



```
# Access per-split feature importances via the fi_source column
for split_id in task_predictor_test.outer_splits:
    split_fi = fi_table_perm[fi_table_perm["fi_source"] == split_id].copy()
    print(f"\n=== Outersplit {split_id} ===")
    display_table(split_fi[["feature", "importance_mean", "importance_std", "p_value"]].head(10))
```

#### Calculate SHAP Feature Importances

Available `shap_type` options:

- `'kernel'` — Model-agnostic (KernelExplainer). Works with any model. Default.
- `'permutation'` — Model-agnostic (PermutationExplainer). Permutation-based approach.
- `'exact'` — Model-agnostic (ExactExplainer). Exact SHAP values (slowest, most accurate).



```
# SHAP feature importances on test data using final models
# calculate_fi() returns the full FI table (per-split + ensemble rows)
# Available shap_type: 'kernel' (default), 'permutation', 'exact'
fi_table_shap = task_predictor_test.calculate_fi(fi_type="shap", shap_type="kernel")
```


```
fi_ensemble_shap = show_overall_fi_table(fi_table_shap)
fi_ensemble_shap.head(10)
```


```
show_overall_fi_plot(fi_table_shap)
```


