"""Ensemble selection."""

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from attrs import define, field, validators
from upath import UPath

from octopus.logger import get_logger
from octopus.metrics import Metrics
from octopus.metrics.utils import get_performance_from_predictions
from octopus.modules.tako.bag import BagBase, recompute_prediction_from_probabilities
from octopus.modules.tako.objective_optuna import _BAG_SUFFIX, _PREDS_SUFFIX
from octopus.types import DataPartition, MetricDirection, MLType, PerformanceKey, PredictionKey
from octopus.utils import joblib_load

logger = get_logger()

_BAGGING_ROUNDS = 20
_BAGGING_FRACTION = 0.5
_BAGGING_MIN_ROWS = 4


def _average_and_quantize(weight_vectors: list[Counter[UPath]]) -> dict[UPath, int]:
    """Combine weight vectors from bagging rounds into a single ensemble.

    Averages model weights across rounds and quantizes to positive integers.
    Bags with averaged weight below 0.5 are omitted from the final ensemble.

    Args:
        weight_vectors: List of Counter dicts mapping bag path to integer weight.

    Returns:
        Dict mapping bag path to quantized positive integer weight.
    """
    all_paths: set[UPath] = set().union(*weight_vectors)

    n_rounds = len(weight_vectors)
    result: dict[UPath, int] = {}
    for path in all_paths:
        total = sum(wv.get(path, 0) for wv in weight_vectors)
        assert total >= 0, f"Counter weights must be non-negative, got {total} for {path}"
        quantized = int(total / n_rounds + 0.5)
        if quantized > 0:
            result[path] = quantized
    return result


def _class_balanced_subsample(
    row_ids: pd.Index,
    targets: pd.Series,
    n_subset: int,
    rng: np.random.Generator,
) -> pd.Index:
    """Subsample row IDs ensuring at least one representative per class.

    Prevents single-class subsets that cause NaN from AUCROC and similar
    metrics. Does NOT guarantee proportional class distribution — remaining
    slots after the per-class guarantee are filled by random draw.

    Falls back to plain random subsampling when n_subset < n_classes.

    Args:
        row_ids: Full set of row IDs (index-aligned with targets).
        targets: Target values for each row ID.
        n_subset: Desired subset size.
        rng: Numpy random generator.

    Returns:
        Subsampled row IDs as pd.Index.
    """
    classes = targets.unique()
    if n_subset < len(classes):
        idx = rng.choice(len(row_ids), size=n_subset, replace=False)
        return row_ids[idx]

    selected = []
    for c in classes:
        class_row_ids = row_ids[targets == c]
        pick = rng.choice(len(class_row_ids), size=1, replace=False)
        selected.append(class_row_ids[pick[0]])

    remaining = n_subset - len(selected)
    pool = row_ids.difference(pd.Index(selected))
    if remaining > 0 and len(pool) > 0:
        extra_idx = rng.choice(len(pool), size=min(remaining, len(pool)), replace=False)
        selected.extend(pool[extra_idx])

    result: pd.Index = pd.Index(selected)
    assert len(result) == n_subset, (
        f"_class_balanced_subsample contract violated: expected {n_subset} rows, got {len(result)}. "
        f"len(row_ids)={len(row_ids)}, n_classes={len(classes)}."
    )
    return result


@define
class EnSel:
    """Ensemble Selection."""

    target_metric: str = field(validator=[validators.instance_of(str)])
    target_assignments: dict = field(validator=[validators.instance_of(dict)])
    path_trials: UPath = field(validator=[validators.instance_of(UPath)])
    max_n_iterations: int = field(validator=[validators.instance_of(int)])
    row_id_col: str = field(validator=[validators.instance_of(str)])
    n_assigned_cpus: int = field(validator=[validators.instance_of(int)])
    ml_type: MLType = field(validator=[validators.instance_of(MLType)])
    positive_class: str | int | None = field(default=None)
    model_table: pd.DataFrame = field(
        init=False,
        default=pd.DataFrame(),
        validator=[validators.instance_of(pd.DataFrame)],
    )
    scan_table: pd.DataFrame = field(
        init=False,
        default=pd.DataFrame(),
        validator=[validators.instance_of(pd.DataFrame)],
    )
    start_ensemble: dict = field(init=False, validator=[validators.instance_of(dict)])
    optimized_ensemble: dict = field(init=False, validator=[validators.instance_of(dict)])
    bags: dict[UPath, dict[str, Any]] = field(init=False, validator=[validators.instance_of(dict)])
    _pred_arrays: dict[UPath, dict[DataPartition, pd.DataFrame]] = field(
        init=False, validator=[validators.instance_of(dict)]
    )
    _target_dtypes: dict[str, Any] = field(init=False, factory=dict)

    @property
    def direction(self) -> MetricDirection:
        """Optuna direction."""
        return Metrics.get_direction(self.target_metric)

    def __attrs_post_init__(self):
        # initialization here due to "Python immutable default"
        self.bags = {}
        self.start_ensemble = {}
        self.optimized_ensemble = {}
        self._pred_arrays = {}
        # get a trials existing in path_trials
        self._collect_trials()
        self._create_model_table()
        self._ensemble_scan()
        self._ensemble_optimization()

    def _collect_trials(self):
        """Load trial data into self.bags.

        For each trial, prefers the lightweight prediction file (*_preds.joblib)
        when available, falling back to the full bag file (*_bag.joblib).
        """
        for bag_file in self.path_trials.glob(f"*{_BAG_SUFFIX}"):
            preds_file = bag_file.with_name(bag_file.name.replace(_BAG_SUFFIX, _PREDS_SUFFIX))
            if preds_file.exists():
                self._load_trial_from_preds(preds_file)
            else:
                self._load_trial_from_bag(bag_file)

        if not self.bags:
            raise FileNotFoundError(
                f"No ensemble candidate bag files (*{_BAG_SUFFIX}) found in {self.path_trials}. "
                "Ensure the optimization phase completed and n_ensemble_candidates > 0."
            )
        logger.info(f"Collected {len(self.bags)} ensemble candidates from {self.path_trials}")

    def _load_trial_from_preds(self, file: UPath):
        """Fast path: load a single lightweight prediction file."""
        preds_data = joblib_load(file)

        predictions = preds_data["predictions"]
        if not self._target_dtypes and "target_dtypes" in preds_data:
            self._target_dtypes = preds_data["target_dtypes"]

        performance = self._compute_trial_performance(predictions)

        bag_path = file.with_name(file.name.removesuffix(_PREDS_SUFFIX) + _BAG_SUFFIX)
        self.bags[bag_path] = {
            "id": preds_data["bag_id"],
            "performance": performance,
            "predictions": predictions,
            "n_features_used_mean": preds_data["n_features_used_mean"],
        }

    def _load_trial_from_bag(self, file: UPath):
        """Slow path: load a full bag file."""
        bag: BagBase = joblib_load(file)

        predictions = bag.get_predictions(n_assigned_cpus=self.n_assigned_cpus)

        if not self._target_dtypes and bag.trainings:
            first_train_data = bag.trainings[0].data_train
            for col in self.target_assignments.values():
                if col in first_train_data.columns:
                    self._target_dtypes[col] = first_train_data[col].dtype

        performance = self._compute_trial_performance(predictions)

        self.bags[file] = {
            "id": bag.bag_id,
            "performance": performance,
            "predictions": predictions,
            "n_features_used_mean": bag.n_features_used_mean,
        }

    def _compute_trial_performance(self, predictions: dict) -> dict[str, float]:
        """Compute performance metrics from a predictions dict."""
        performance_raw = get_performance_from_predictions(
            predictions=predictions,
            target_metric=self.target_metric,
            target_assignments=self.target_assignments,
            positive_class=self.positive_class,
        )

        performance: dict[str, float] = {}
        if PredictionKey.ENSEMBLE in performance_raw:
            performance[PerformanceKey.DEV_ENSEMBLE] = performance_raw[PredictionKey.ENSEMBLE][DataPartition.DEV]
            performance[PerformanceKey.TEST_ENSEMBLE] = performance_raw[PredictionKey.ENSEMBLE][DataPartition.TEST]

        dev_scores = [v[DataPartition.DEV] for k, v in performance_raw.items() if k != PredictionKey.ENSEMBLE]
        test_scores = [v[DataPartition.TEST] for k, v in performance_raw.items() if k != PredictionKey.ENSEMBLE]
        performance[PerformanceKey.DEV_AVG] = sum(dev_scores) / len(dev_scores) if dev_scores else 0.0
        performance[PerformanceKey.TEST_AVG] = sum(test_scores) / len(test_scores) if test_scores else 0.0

        return performance

    def _create_model_table(self):
        """Create model table."""
        records = [
            {
                "id": value["id"],
                PerformanceKey.DEV_ENSEMBLE: value["performance"][PerformanceKey.DEV_ENSEMBLE],
                PerformanceKey.TEST_ENSEMBLE: value["performance"][PerformanceKey.TEST_ENSEMBLE],
                PerformanceKey.DEV_AVG: value["performance"][PerformanceKey.DEV_AVG],
                PerformanceKey.TEST_AVG: value["performance"][PerformanceKey.TEST_AVG],
                "n_features_used_mean": value["n_features_used_mean"],
                "path": key,
            }
            for key, value in self.bags.items()
        ]

        self.model_table = pd.DataFrame.from_records(records)

        # order of table is important, depending on metric,
        # (a) direction
        ascending = self.direction != MetricDirection.MAXIMIZE

        self.model_table = self.model_table.sort_values(
            by=PerformanceKey.DEV_ENSEMBLE, ascending=ascending
        ).reset_index(drop=True)

        logger.info("Model Table:")
        logger.info(f"\n{self.model_table.head(20)}")

    def _restore_target_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Restore target column dtypes after averaging.

        After groupby().mean(), integer/boolean target columns become float.
        This restores them to their original dtype for consistency with
        Bag.get_predictions() behavior.

        Args:
            df: DataFrame with averaged predictions.

        Returns:
            DataFrame with target columns cast to original dtypes.
        """
        for col, dtype in self._target_dtypes.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        return df

    def _ensemble_models(self, bag_keys):
        """Esemble using all bags and their corresponding models provided by input."""
        # collect all predictions over inner splits and bags
        predictions: dict[str, dict[str, pd.DataFrame]] = {}
        performance_output = {}
        ensemble_pool: dict[DataPartition, list[pd.DataFrame]] = {DataPartition.DEV: [], DataPartition.TEST: []}

        for key in bag_keys:
            bag_predictions = self.bags[key]["predictions"]
            # concatenate and average dev and test predictions from inner models
            for pred_key, pred in bag_predictions.items():
                if pred_key == PredictionKey.ENSEMBLE:
                    continue
                for part, part_preds in ensemble_pool.items():
                    part_preds.append(pred[part])

        # created ensembled predictions (all inner models from all bags) for each partition
        predictions[PredictionKey.ENSEMBLE] = {}
        for part, part_preds in ensemble_pool.items():
            combined = pd.concat(part_preds, axis=0)
            # Identify numeric columns to average (exclude metadata and row_id)
            numeric_cols = combined.select_dtypes(include=["number"]).columns.tolist()
            if self.row_id_col in numeric_cols:
                numeric_cols.remove(self.row_id_col)
            for col in ["outer_split_id", "inner_split_id", "task_id"]:
                if col in numeric_cols:
                    numeric_cols.remove(col)

            ensemble = combined.groupby(by=self.row_id_col)[numeric_cols].mean().reset_index()
            self._restore_target_dtypes(ensemble)
            recompute_prediction_from_probabilities(ensemble)
            predictions[PredictionKey.ENSEMBLE][part] = ensemble

        # Calculate performance using the utility function
        performance = get_performance_from_predictions(
            predictions=predictions,
            target_metric=self.target_metric,
            target_assignments=self.target_assignments,
            positive_class=self.positive_class,
        )

        # Add ensemble performance with renamed keys
        if PredictionKey.ENSEMBLE in performance:
            performance_output[PerformanceKey.DEV_ENSEMBLE] = performance[PredictionKey.ENSEMBLE][DataPartition.DEV]
            performance_output[PerformanceKey.TEST_ENSEMBLE] = performance[PredictionKey.ENSEMBLE][DataPartition.TEST]

        return performance_output

    def _ensemble_scan(self):
        """Scan for highest performing ensemble consisting of best N bags."""
        self.scan_table = pd.DataFrame(
            columns=[
                "#models",
                PerformanceKey.DEV_ENSEMBLE,
                PerformanceKey.TEST_ENSEMBLE,
            ]
        )

        for i in range(len(self.model_table)):
            bag_keys = self.model_table[: i + 1]["path"].tolist()
            scores = self._ensemble_models(bag_keys)
            self.scan_table.loc[i] = [
                i + 1,
                scores[PerformanceKey.DEV_ENSEMBLE],
                scores[PerformanceKey.TEST_ENSEMBLE],
            ]
        logger.info("Scan Table:")
        logger.info(f"\n{self.scan_table.head(20)}")

    def _precompute_prediction_arrays(self):
        """Precompute per-bag averaged prediction DataFrames for fast incremental ensembling.

        Creates self._pred_arrays: dict mapping bag path to dict with
        "dev" and "test" DataFrames (one row per sample, indexed by row_id).
        Each bag's predictions are averaged across its inner folds.
        """
        self._pred_arrays = {}
        for key, bag_data in self.bags.items():
            predictions = bag_data["predictions"]
            dev_preds: list[pd.DataFrame] = []
            test_preds: list[pd.DataFrame] = []

            for pred_key, pred in predictions.items():
                if pred_key == PredictionKey.ENSEMBLE:
                    continue
                dev_preds.append(pred[DataPartition.DEV])
                test_preds.append(pred[DataPartition.TEST])

            for part_name, part_preds in [(DataPartition.DEV, dev_preds), (DataPartition.TEST, test_preds)]:
                combined = pd.concat(part_preds, axis=0)
                # Identify numeric columns to average (exclude metadata and row_id)
                numeric_cols = combined.select_dtypes(include=["number"]).columns.tolist()
                if self.row_id_col in numeric_cols:
                    numeric_cols.remove(self.row_id_col)
                for col in ["outer_split_id", "inner_split_id", "task_id"]:
                    if col in numeric_cols:
                        numeric_cols.remove(col)

                avg = combined.groupby(self.row_id_col)[numeric_cols].mean()
                self._restore_target_dtypes(avg)
                recompute_prediction_from_probabilities(avg)
                if key not in self._pred_arrays:
                    self._pred_arrays[key] = {}
                self._pred_arrays[key][part_name] = avg

    def _compute_metric_from_avg(self, avg_dev: pd.DataFrame) -> float:
        """Compute target metric from an averaged dev prediction DataFrame.

        Args:
            avg_dev: DataFrame indexed by row_id with prediction and target columns.

        Returns:
            Metric value (float).
        """
        avg_dev = avg_dev.copy()

        # Renormalize class probability columns that may not sum to 1 after averaging.
        prob_cols = [c for c in avg_dev.columns if isinstance(c, int)]
        if len(prob_cols) >= 2:
            row_sums = avg_dev[prob_cols].sum(axis=1)
            # Only renormalize if sums deviate from 1
            needs_renorm = (row_sums - 1.0).abs() > 1e-9
            if needs_renorm.any():
                avg_dev.loc[needs_renorm, prob_cols] = avg_dev.loc[needs_renorm, prob_cols].div(
                    row_sums[needs_renorm], axis=0
                )

        recompute_prediction_from_probabilities(avg_dev)

        predictions = {PredictionKey.ENSEMBLE: {DataPartition.DEV: avg_dev.reset_index()}}
        perf = get_performance_from_predictions(
            predictions=predictions,
            target_metric=self.target_metric,
            target_assignments=self.target_assignments,
            positive_class=self.positive_class,
        )
        result: float = perf[PredictionKey.ENSEMBLE][DataPartition.DEV]
        return result

    def _is_better(self, score_a: float, score_b: float) -> bool:
        """Return True if score_a is strictly better than score_b."""
        if self.direction == MetricDirection.MAXIMIZE:
            return score_a > score_b
        return score_a < score_b

    def _diversity_columns(self) -> list[str | int]:
        """Return the columns used for diversity measurement.

        For classification with probability columns: all integer-named class
        probability columns (captures full diversity for both binary and
        multiclass). For regression/T2E: the 'prediction' column.

        Returns:
            Non-empty list of column names. Each column is verified to exist
            in the precomputed prediction arrays.

        Raises:
            RuntimeError: If no usable diversity columns are found in
                self._pred_arrays. Indicates that prediction precomputation
                produced an unexpected schema.
        """
        first_key = next(iter(self._pred_arrays))
        cols = self._pred_arrays[first_key][DataPartition.DEV].columns
        prob_cols: list[str | int] = [c for c in cols if isinstance(c, int)]
        if prob_cols:
            return prob_cols
        if "prediction" in cols:
            return ["prediction"]
        raise RuntimeError(
            f"No diversity column found in prediction arrays. "
            f"Expected integer-named probability columns or 'prediction' column, "
            f"got columns: {list(cols)}"
        )

    @staticmethod
    def _avg_abs_corr(a: pd.DataFrame, b: pd.DataFrame, cols: list[str | int]) -> float:
        """Average |Pearson correlation| across a list of columns.

        Excludes NaN correlations (constant columns). Returns 1.0 if every
        column-pair correlation is NaN (treat as fully redundant).

        Args:
            a: First prediction frame.
            b: Second prediction frame (must share index with ``a``).
            cols: Columns to correlate.

        Returns:
            Mean of ``|corr|`` over non-NaN columns, or 1.0 if all NaN.
        """
        corrs: list[float] = []
        for c in cols:
            corr = a[c].corr(b[c])
            if not pd.isna(corr):
                corrs.append(abs(float(corr)))
        if not corrs:
            return 1.0
        return float(np.mean(corrs))

    def _run_single_greedy(
        self,
        start_bags: list[UPath],
        pred_arrays: dict[UPath, dict[DataPartition, pd.DataFrame]],
        all_candidates: list[UPath],
        div_cols: list[str | int],
    ) -> Counter:
        """Run one greedy forward selection pass with use_best backtracking (Caruana et al., ICML 2004).

        Args:
            start_bags: Starting ensemble from scan.
            pred_arrays: Per-bag averaged prediction DataFrames. May be subsetted
                (fewer rows) for bagged ensemble selection.
            all_candidates: Candidate bag paths to consider at each step.
                Hoisted out of the loop to avoid rebuilding per bagging round.
            div_cols: Columns used for diversity tie-breaking. Computed once
                in :py:meth:`_ensemble_optimization` and shared across rounds.

        Returns:
            Counter mapping bag path to selection count.
        """
        first_key = start_bags[0]
        running_dev = pd.DataFrame(
            0.0,
            index=pred_arrays[first_key][DataPartition.DEV].index,
            columns=pred_arrays[first_key][DataPartition.DEV].columns,
        )
        for bag_path in start_bags:
            running_dev += pred_arrays[bag_path][DataPartition.DEV]
        n_members = len(start_bags)

        bags_ensemble = list(start_bags)
        start_perf = self._compute_metric_from_avg(running_dev / n_members)
        trajectory: list[tuple[float, list[UPath]]] = [(start_perf, list(bags_ensemble))]

        for i in range(self.max_n_iterations):
            current_avg: pd.DataFrame | None = None
            best_score_this_iter: float | None = None
            best_model_this_iter: UPath | None = None

            for model in all_candidates:
                candidate_avg = running_dev.add(pred_arrays[model][DataPartition.DEV]) / (n_members + 1)
                try:
                    score = self._compute_metric_from_avg(candidate_avg)
                except (ValueError, ZeroDivisionError):
                    continue

                if pd.isna(score):
                    continue

                if i > 0 and best_score_this_iter is not None and abs(best_score_this_iter) > 1e-4:
                    score = round(score, 6)

                if best_score_this_iter is None or self._is_better(score, best_score_this_iter):
                    best_score_this_iter = score
                    best_model_this_iter = model
                elif score == best_score_this_iter:
                    if model in bags_ensemble and best_model_this_iter not in bags_ensemble:
                        best_model_this_iter = model
                    elif (model in bags_ensemble) == (best_model_this_iter in bags_ensemble):
                        if current_avg is None:
                            current_avg = running_dev / n_members
                        corr_new = self._avg_abs_corr(current_avg, pred_arrays[model][DataPartition.DEV], div_cols)
                        corr_old = self._avg_abs_corr(
                            current_avg, pred_arrays[best_model_this_iter][DataPartition.DEV], div_cols
                        )
                        if corr_new < corr_old:
                            best_model_this_iter = model

            if best_model_this_iter is None or best_score_this_iter is None:
                break
            bags_ensemble.append(best_model_this_iter)
            running_dev += pred_arrays[best_model_this_iter][DataPartition.DEV]
            n_members += 1
            trajectory.append((best_score_this_iter, list(bags_ensemble)))

        best_traj_idx = 0
        for idx in range(1, len(trajectory)):
            if self._is_better(trajectory[idx][0], trajectory[best_traj_idx][0]):
                best_traj_idx = idx

        return Counter(trajectory[best_traj_idx][1])

    def _ensemble_optimization(self):
        """Greedy forward selection, optionally bagged (Caruana et al., ICML 2006).

        Starts from the best N models found by the scan phase. Each iteration
        tries adding every candidate model and keeps the one that maximizes
        ensemble performance on the dev set. With use_best backtracking.

        When bagging is active, runs the greedy selection B times on random
        dev-row subsets and averages the resulting weight vectors.
        """
        self._precompute_prediction_arrays()

        if self.direction == MetricDirection.MAXIMIZE:
            best_value = self.scan_table[PerformanceKey.DEV_ENSEMBLE].max()
            best_idx = self.scan_table[self.scan_table[PerformanceKey.DEV_ENSEMBLE] == best_value].index[-1]
        else:
            best_value = self.scan_table[PerformanceKey.DEV_ENSEMBLE].min()
            best_idx = self.scan_table[self.scan_table[PerformanceKey.DEV_ENSEMBLE] == best_value].index[-1]
        start_n = int(self.scan_table.loc[best_idx, "#models"])
        logger.info(f"Ensemble scan, number of included best models: {start_n}")
        logger.info(f"Ensemble scan, dev_ensemble value: {self.scan_table.loc[best_idx, PerformanceKey.DEV_ENSEMBLE]}")
        logger.info(
            f"Ensemble scan, test_ensemble value: {self.scan_table.loc[best_idx, PerformanceKey.TEST_ENSEMBLE]}"
        )

        start_bags = self.model_table.head(start_n)["path"].tolist()
        start_perf = self._ensemble_models(start_bags)[PerformanceKey.DEV_ENSEMBLE]
        logger.info("Ensemble optimization")
        logger.info(f"Start performance: {start_perf}")

        self.start_ensemble = dict(Counter(start_bags))

        all_candidates: list[UPath] = self.model_table["path"].tolist()
        div_cols = self._diversity_columns()

        first_key = start_bags[0]
        all_row_ids = self._pred_arrays[first_key][DataPartition.DEV].index
        n_rows = len(all_row_ids)

        weights: dict
        if n_rows < _BAGGING_MIN_ROWS:
            logger.info(
                f"Dev set has {n_rows} rows (< {_BAGGING_MIN_ROWS}); skipping bagging and running a single greedy pass."
            )
            weights = self._run_single_greedy(start_bags, self._pred_arrays, all_candidates, div_cols)
        else:
            n_subset = max(2, min(int(n_rows * _BAGGING_FRACTION), n_rows))
            rng = np.random.default_rng(42)

            first_dev = self._pred_arrays[first_key][DataPartition.DEV]
            stratify_targets = self._stratification_targets(first_dev)

            weight_vectors: list[Counter[UPath]] = []
            for b in range(_BAGGING_ROUNDS):
                if stratify_targets is not None:
                    subset = _class_balanced_subsample(all_row_ids, stratify_targets, n_subset, rng)
                else:
                    subset_idx = rng.choice(len(all_row_ids), size=n_subset, replace=False)
                    subset = all_row_ids[subset_idx]

                pred_arrays_subset = {
                    key: {DataPartition.DEV: arr[DataPartition.DEV].loc[subset]}
                    for key, arr in self._pred_arrays.items()
                }

                weights_b = self._run_single_greedy(start_bags, pred_arrays_subset, all_candidates, div_cols)
                weight_vectors.append(weights_b)
                logger.debug(f"Bagging round {b}: {len(weights_b)} distinct bags")

            logger.info(
                f"Bagging completed: {_BAGGING_ROUNDS} rounds, "
                f"avg distinct bags per round = "
                f"{sum(len(w) for w in weight_vectors) / _BAGGING_ROUNDS:.1f}"
            )
            weights = _average_and_quantize(weight_vectors)

            final_bags = [k for k, v in weights.items() for _ in range(v)]
            final_perf = self._ensemble_models(final_bags)[PerformanceKey.DEV_ENSEMBLE]
            if not self._is_better(final_perf, start_perf):
                logger.info(
                    f"Bagged ensemble ({final_perf:.4f}) not better than scan start ({start_perf:.4f}); "
                    f"falling back to scan start."
                )
                weights = dict(Counter(start_bags))

        self.optimized_ensemble = {k: v for k, v in weights.items() if v > 0}
        logger.info("Ensemble optimization completed.")

    def _stratification_targets(self, dev_df: pd.DataFrame) -> pd.Series | None:
        """Return the per-row target series to stratify on, or None for plain random.

        For binary/multiclass classification: stratify on the prediction target column.
        For time-to-event: stratify on the event indicator column (preserves the
        event/censoring ratio across bagging subsets, which matters for CI / IBS).
        For regression: no stratification.

        Args:
            dev_df: Any of the precomputed dev prediction DataFrames. Expected to
                contain the relevant target column(s).

        Returns:
            Series indexed by row_id with target/event values, or None if the
            ml_type does not warrant stratification.
        """
        if self.ml_type in (MLType.BINARY, MLType.MULTICLASS):
            target_col = list(self.target_assignments.values())[0]
            return pd.Series(dev_df[target_col])
        if self.ml_type == MLType.TIMETOEVENT:
            event_col = self.target_assignments.get("event")
            if event_col is None or event_col not in dev_df.columns:
                logger.warning(
                    "T2E ml_type but event column not available in dev predictions; "
                    "falling back to random subsampling for bagging."
                )
                return None
            return pd.Series(dev_df[event_col])
        return None

    def get_ens_input(self):
        """Get ensemble dict."""
        return self.bags
