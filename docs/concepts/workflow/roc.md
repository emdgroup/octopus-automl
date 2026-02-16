# ROC -- Removal of Correlated Features


ROC is a fast, filter-based preprocessing step that removes redundant features
before more expensive methods run. Highly correlated features carry overlapping
information; keeping all of them wastes compute and can destabilize models. ROC
detects these groups automatically and retains only the most informative member
of each group.

## How it works

1. **Compute pairwise correlations.** ROC calculates absolute Spearman rank
   correlations (default) or Randomized Dependence Coefficients (RDC) between
   every pair of features. Spearman is fast and captures monotonic
   relationships; RDC can detect non-linear dependencies at a higher
   computational cost.

2. **Build a correlation graph.** An undirected graph is constructed where each
   feature is a node. An edge is added between two features whenever their
   correlation exceeds the `threshold` (default 0.8).

3. **Find connected components.** Using NetworkX, ROC identifies the connected
   components of the graph. Each component forms a *correlation group* -- a
   cluster of features that are all transitively correlated above the threshold.

4. **Select the best feature per group.** Within each group, ROC scores every
   feature for univariate relevance to the target using either F-statistics
   (ANOVA F-value for classification, F-regression for regression) or mutual
   information. The feature with the highest relevance score is kept; the rest
   are dropped.

5. **Return the reduced feature set.** All features that were *not* part of any
   correlation group are kept unconditionally, together with the one
   representative per group.

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | `0.8` | Correlation above which features are considered redundant |
| `correlation_type` | `"spearmanr"` | `"spearmanr"` or `"rdc"` |
| `filter_type` | `"f_statistics"` | `"f_statistics"` or `"mutual_info"` -- used to pick the best feature in each group |

## When to use

ROC is ideal as the **first task** in a workflow. It is fast (no model training
required), deterministic, and typically removes 10-40% of features in datasets
with many correlated columns. Running ROC before heavier modules like Octo or
RFE reduces their runtime and search space significantly.

## Limitations

- Only detects *pairwise* (or transitively pairwise) redundancy. Complex
  multivariate redundancies are not captured.
- The Spearman correlation only captures monotonic relationships. Switch to
  `"rdc"` if non-linear correlations are expected, at the cost of longer
  computation.
- For time-to-event targets the relevance scoring is not available; ROC simply
  keeps the first feature in each group.
