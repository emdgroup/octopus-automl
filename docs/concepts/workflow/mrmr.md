# MRMR -- Maximum Relevance Minimum Redundancy

*Based on: [mrmr](https://github.com/smazzanti/mrmr)*

MRMR is a greedy filter method that selects a fixed-size feature subset by
balancing two competing objectives: each selected feature should be highly
relevant to the target, and the selected features should be as uncorrelated with
each other as possible. This makes MRMR an effective bridge step between two
Octo tasks -- it uses the feature importances from a preceding Octo run as the
relevance signal and compresses the feature set for the next training round.

## How it works

1. **Obtain relevance scores.** In the default `"permutation"` mode, MRMR reads
   the permutation feature importances produced by a prior Octo task (specified
   via `fi_method`). It aggregates importance values across inner splits, keeps
   only features with positive importance, and uses these as the relevance
   vector. Alternatively, when `relevance_method="f_statistics"`, MRMR computes
   univariate F-statistics directly from the data (no prior task needed).

2. **Compute a pairwise correlation matrix.** Absolute Pearson, Spearman, or
   RDC correlations are calculated among all candidate features.

3. **Greedy iterative selection.** Starting from an empty set:
    - **Step 1:** Select the feature with the highest relevance.
    - **Step *k*:** For each remaining candidate, compute its MRMR score as
      *relevance / mean-correlation-with-already-selected* (ratio mode, the
      default). The candidate with the highest score is added.
    - Near-perfect correlations (>1 - 1e-8) are penalized with a large finite
      value to avoid numerical instability.
    - Ties are broken deterministically.

4. **Stop after `n_features`.** The algorithm returns exactly `n_features`
   features (or fewer if fewer candidates have positive importance).

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_features` | `30` | Number of features to select |
| `correlation_type` | `"spearman"` | `"pearson"`, `"spearman"`, or `"rdc"` |
| `relevance_method` | `"permutation"` | `"permutation"` (uses prior task's feature importances) or `"f_statistics"` |
| `fi_method` | `"permutation"` | `"permutation"`, `"shap"`, `"internal"`, or `"lofo"` |

## When to use

MRMR works best as a **middle step** in a multi-task workflow, typically placed
between two Octo runs:

```
Octo (full features) → MRMR (reduce to top N) → Octo (refined features)
```

It leverages the feature importances from the first Octo to build a compact,
low-redundancy feature set for the second. It is also useful when you have a
target feature count in mind and want a fast, non-iterative way to get there.

## Limitations

- When using `relevance_method="permutation"`, MRMR **cannot be the first task**
  in a workflow, it requires feature importances from a prior module.
- Only features with strictly positive importance are considered. If the prior
  model assigns zero or negative importance to many features, the effective
  candidate pool shrinks.
- The greedy selection is not globally optimal; the quality of the result
  depends heavily on accurate relevance estimates from the prior task.

## Examples

- [Multi-Step Classification](../../examples/wf_multi_step_classification.md) — Octo → MRMR → Octo pipeline for classification.
- [Multi-Step Regression](../../examples/wf_multi_step_regression.md) — same pattern applied to regression.
