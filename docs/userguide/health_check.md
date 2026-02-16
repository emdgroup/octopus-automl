# Data Health Check

Every call to `study.fit()` automatically runs a data health check before training
begins. The health check inspects your dataset for common quality issues and
produces a report at `{output_path}/health_check_report.csv`.

## What it checks

Issues are classified by severity:

**Critical** (training stops with a `ValueError`):

- Dataset has fewer than 20 samples
- Duplicate values in the row ID column
- Missing values in target, duration, event, row ID, or stratification columns
- Features where every value is missing

**Warning** (logged, training continues):

- Class imbalance (majority class > 80% of samples)
- High missing value rate in feature columns (> 25%) or rows (> 50%)
- Features highly correlated with the target (> 0.95, potential data leakage)
- Highly correlated feature pairs (> 0.8)
- Duplicate rows or identical features
- Integer columns with very few unique values (may be categorical)
- Skewed or heavy-tailed target distributions (regression)

**Info** (recorded in the report only):

- Low missing value rates
- Infinity values in numeric features

## Customizing thresholds

All thresholds are configurable via `HealthCheckConfig`. Pass it to `fit()`:

```python
from octopus.study.healthChecker import HealthCheckConfig

config = HealthCheckConfig(
    missing_value_column_threshold=0.30,   # flag features with >30% missing (default: 0.25)
    class_imbalance_threshold=0.90,        # flag if majority class >90% (default: 0.80)
    minimum_samples_threshold=50,          # require at least 50 samples (default: 20)
    target_leakage_threshold=0.99,         # raise leakage threshold (default: 0.95)
    feature_correlation_threshold=0.90,    # flag correlated features >0.90 (default: 0.80)
)

study.fit(data=df, health_check_config=config)
```

## Reading the report

The report is saved as a CSV file with one row per issue found:

| Column | Description |
|--------|-------------|
| `Category` | Area of the issue (rows, columns, features, target) |
| `Issue Type` | Specific check that triggered (e.g. `high_missing_values`) |
| `Affected Items` | Names of problematic columns or rows |
| `Severity` | Critical, Warning, or Info |
| `Description` | Explanation of what was found |
| `Recommended Action` | Suggested fix |

If any critical issues are found, `fit()` raises a `ValueError` pointing to the
report file. Fix the issues and call `fit()` again on a new study instance.

## See also

- [Classification](classification.md), [Regression](regression.md), [Time to Event](time_to_event.md) — task-specific guides showing how to set up a study.
- [Understanding the Output](output_structure.md) — where the health check report file is saved.
