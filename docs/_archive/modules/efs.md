# EFS -- Ensemble Feature Selection


EFS takes a fundamentally different approach to feature selection: instead of
evaluating features individually, it trains many models on random feature
subsets, then uses ensemble optimization to find the best *combination* of
models. Features that appear in the winning ensemble are selected. This
diversity-driven approach is especially effective for high-dimensional datasets
where individual feature rankings may be unstable.

## How it works

1. **Generate random feature subsets.** EFS creates `n_subsets` (default 100)
   random subsets, each containing `subset_size` (default 30) features drawn
   from the full feature set.

2. **Train a model per subset.** For each subset, a `GridSearchCV` tunes and
   trains the chosen model (CatBoost, XGBoost, RandomForest, or ExtraTrees).
   Cross-validated predictions are collected for every training sample.

3. **Build a model table.** Each trained model is recorded along with its CV
   performance, the features it used (excluding those with zero importance), and
   its out-of-fold predictions. Models are sorted by performance.

4. **Ensemble scan (hill-climbing).** Starting from the single best model, the
   module incrementally adds the next-best model and computes the ensemble
   performance (averaged predictions across models). This scan identifies the
   number of top models that, when ensembled, give the best combined score.

5. **Ensemble optimization with replacement.** Starting from the models found in
   the scan, the optimizer iteratively tests adding each of the top
   `max_n_models` models (with replacement) to the ensemble. At each iteration,
   the model that improves ensemble performance the most is added. The process
   repeats for up to `max_n_iterations` or until no improvement is found.

6. **Feature aggregation.** The final optimized ensemble is a weighted
   collection of models (weights = number of times each model appears). The
   union of all features used by the ensemble models becomes the selected
   feature set. Feature importance is reported as *counts* (how many times a
   feature appeared across ensemble models) and *relative counts* (counts /
   total models).

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `""` (auto) | Model name -- `CatBoost`, `XGB`, `RandomForest`, or `ExtraTrees` |
| `subset_size` | `30` | Number of features per random subset |
| `n_subsets` | `100` | Number of random subsets to create |
| `cv` | `5` | Cross-validation folds |
| `max_n_iterations` | `50` | Iterations for ensemble optimization |
| `max_n_models` | `30` | Maximum models to consider in optimization |

## When to use

EFS is ideal when:

- The dataset is **high-dimensional** (hundreds to thousands of features) and
  individual feature rankings are noisy or inconsistent.
- You want a **diversity-driven** selection that captures complementary sets of
  features rather than just the top-ranked ones.
- Compute resources are available for training many models in parallel.

## Limitations

- Computationally heavy: `n_subsets` models are trained, each with a grid
  search. With 100 subsets and a 4-parameter grid this can mean thousands of
  model fits.
- The random subset generation means results are seed-dependent. Different seeds
  may produce different feature sets, though the ensemble optimization helps
  stabilize this.
- Does not produce scores or predictions in the standard format (scores and
  predictions DataFrames are empty); it primarily returns feature counts as
  importance measures.
- Does not support time-to-event targets.
