"""Ensemble selection."""

from collections import Counter
from typing import Any

import pandas as pd
from attrs import define, field, validators
from upath import UPath

from octopus.logger import get_logger
from octopus.metrics import Metrics
from octopus.metrics.utils import get_performance_from_predictions
from octopus.modules.octo.bag import BagBase, recompute_prediction_from_probabilities
from octopus.types import DataPartition, MetricDirection
from octopus.utils import joblib_load

logger = get_logger()


@define
class EnSel:
    """Ensemble Selection."""

    target_metric: str = field(validator=[validators.instance_of(str)])
    target_assignments: dict = field(validator=[validators.instance_of(dict)])
    path_trials: UPath = field(validator=[validators.instance_of(UPath)])
    max_n_iterations: int = field(validator=[validators.instance_of(int)])
    row_id_col: str = field(validator=[validators.instance_of(str)])
    n_assigned_cpus: int = field(validator=[validators.instance_of(int)])
    positive_class = field(default=None)
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
        """Get all trials saved in path_trials and store properties in self.bags.

        Computes predictions once per bag and derives performance from them,
        avoiding the double computation that occurs when calling both
        bag.get_performance() and bag.get_predictions() separately.
        """
        # Get all .joblib files in the directory
        joblib_files = [file for file in self.path_trials.iterdir() if file.is_file() and file.suffix == ".joblib"]

        # fill bags dict
        for file in joblib_files:
            bag: BagBase = joblib_load(file)

            # Compute predictions once (get_performance internally calls get_predictions again)
            predictions = bag.get_predictions(n_assigned_cpus=self.n_assigned_cpus)

            # Derive performance from already-computed predictions
            performance_raw = get_performance_from_predictions(
                predictions=predictions,
                target_metric=self.target_metric,
                target_assignments=self.target_assignments,
                positive_class=self.positive_class,
            )

            # Restructure into the format expected by _create_model_table
            performance: dict[str, float] = {}
            if "ensemble" in performance_raw:
                performance["dev_ensemble"] = performance_raw["ensemble"][DataPartition.DEV]
                performance["test_ensemble"] = performance_raw["ensemble"][DataPartition.TEST]

            # Compute average across individual trainings
            dev_scores = [v[DataPartition.DEV] for k, v in performance_raw.items() if k != "ensemble"]
            test_scores = [v[DataPartition.TEST] for k, v in performance_raw.items() if k != "ensemble"]
            performance["dev_avg"] = sum(dev_scores) / len(dev_scores) if dev_scores else 0.0
            performance["test_avg"] = sum(test_scores) / len(test_scores) if test_scores else 0.0

            # Cache target column dtypes from the first bag for dtype restoration
            if not self._target_dtypes and bag.trainings:
                first_train_data = bag.trainings[0].data_train
                for col in self.target_assignments.values():
                    if col in first_train_data.columns:
                        self._target_dtypes[col] = first_train_data[col].dtype

            self.bags[file] = {
                "id": bag.bag_id,
                "performance": performance,
                "predictions": predictions,
                "n_features_used_mean": bag.n_features_used_mean,
            }

    def _create_model_table(self):
        """Create model table."""
        df_lst = []
        for key, value in self.bags.items():
            s = pd.Series()
            s["id"] = value["id"]
            s["dev_ensemble"] = value["performance"]["dev_ensemble"]  # relevant
            s["test_ensemble"] = value["performance"]["test_ensemble"]
            s["dev_avg"] = value["performance"]["dev_avg"]
            s["test_avg"] = value["performance"]["test_avg"]
            s["n_features_used_mean"] = value["n_features_used_mean"]
            s["path"] = key
            df_lst.append(s)

        self.model_table = pd.concat(df_lst, axis=1).T

        # order of table is important, depending on metric,
        # (a) direction
        ascending = self.direction != MetricDirection.MAXIMIZE

        self.model_table = self.model_table.sort_values(by="dev_ensemble", ascending=ascending).reset_index(drop=True)

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
                if pred_key == "ensemble":
                    continue
                for part, part_preds in ensemble_pool.items():
                    part_preds.append(pred[part])

        # created ensembled predictions (all inner models from all bags) for each partition
        predictions["ensemble"] = {}
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
            predictions["ensemble"][part] = ensemble

        # Calculate performance using the utility function
        performance = get_performance_from_predictions(
            predictions=predictions,
            target_metric=self.target_metric,
            target_assignments=self.target_assignments,
            positive_class=self.positive_class,
        )

        # Add ensemble performance with renamed keys
        if "ensemble" in performance:
            performance_output["dev_ensemble"] = performance["ensemble"][DataPartition.DEV]
            performance_output["test_ensemble"] = performance["ensemble"][DataPartition.TEST]

        return performance_output

    def _ensemble_scan(self):
        """Scan for highest performing ensemble consisting of best N bags."""
        self.scan_table = pd.DataFrame(
            columns=[
                "#models",
                "dev_ensemble",
                "test_ensemble",
            ]
        )

        for i in range(len(self.model_table)):
            bag_keys = self.model_table[: i + 1]["path"].tolist()
            scores = self._ensemble_models(bag_keys)
            self.scan_table.loc[i] = [
                i + 1,
                scores["dev_ensemble"],
                scores["test_ensemble"],
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
                if pred_key == "ensemble":
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

        predictions = {"ensemble": {DataPartition.DEV: avg_dev.reset_index()}}
        perf = get_performance_from_predictions(
            predictions=predictions,
            target_metric=self.target_metric,
            target_assignments=self.target_assignments,
            positive_class=self.positive_class,
        )
        result: float = perf["ensemble"][DataPartition.DEV]
        return result

    def _is_better(self, score_a: float, score_b: float) -> bool:
        """Return True if score_a is strictly better than score_b."""
        if self.direction == MetricDirection.MAXIMIZE:
            return score_a > score_b
        return score_a < score_b

    def _ensemble_optimization(self):
        """Greedy forward selection with replacement and use_best backtracking.

        Starts from the best prefix found by _ensemble_scan. Each iteration
        tries adding every candidate model and keeps the one that maximizes
        ensemble performance on the dev set. Runs for max_n_iterations and
        returns the prefix with best overall performance (backtracking).

        Reference: Caruana et al. (2004) "Ensemble selection from libraries of
        models", with use_best backtracking from AutoGluon.
        """
        # Precompute per-bag averaged predictions for fast incremental computation
        self._precompute_prediction_arrays()

        # Determine starting ensemble from scan
        if self.direction == MetricDirection.MAXIMIZE:
            best_value = self.scan_table["dev_ensemble"].max()
            best_idx = self.scan_table[self.scan_table["dev_ensemble"] == best_value].index[-1]
        else:
            best_value = self.scan_table["dev_ensemble"].min()
            best_idx = self.scan_table[self.scan_table["dev_ensemble"] == best_value].index[-1]
        start_n = int(self.scan_table.loc[best_idx, "#models"])
        logger.info(f"Ensemble scan, number of included best models: {start_n}")
        logger.info(f"Ensemble scan, dev_ensemble value: {self.scan_table.loc[best_idx, 'dev_ensemble']}")
        logger.info(f"Ensemble scan, test_ensemble value: {self.scan_table.loc[best_idx, 'test_ensemble']}")

        # Build starting bags list
        start_bags = self.model_table.head(start_n)["path"].tolist()
        start_perf = self._ensemble_models(start_bags)["dev_ensemble"]
        logger.info("Ensemble optimization")
        logger.info(f"Start performance: {start_perf}")

        # Store scan result
        self.start_ensemble = dict(Counter(start_bags))

        # Initialize running dev sum for incremental computation
        first_key = start_bags[0]
        running_dev = pd.DataFrame(
            0.0,
            index=self._pred_arrays[first_key][DataPartition.DEV].index,
            columns=self._pred_arrays[first_key][DataPartition.DEV].columns,
        )
        for bag_path in start_bags:
            running_dev = running_dev.add(self._pred_arrays[bag_path][DataPartition.DEV])
        n_members = len(start_bags)

        # Greedy optimization with replacement and trajectory tracking
        bags_ensemble = list(start_bags)
        all_candidates = self.model_table["path"].tolist()
        trajectory: list[tuple[float, list[UPath]]] = [(start_perf, list(bags_ensemble))]

        for i in range(self.max_n_iterations):
            best_score_this_iter: float | None = None
            best_model_this_iter: UPath | None = None

            for model in all_candidates:
                # Fantasy prediction: (running_sum + candidate) / (n + 1)
                fantasy_dev = running_dev.add(self._pred_arrays[model][DataPartition.DEV]) / (n_members + 1)
                score = self._compute_metric_from_avg(fantasy_dev)

                # Round scores to 6 decimals to collapse near-ties from float noise
                if i > 0 and best_score_this_iter is not None and abs(best_score_this_iter) > 1e-4:
                    score = round(score, 6)

                if best_score_this_iter is None or self._is_better(score, best_score_this_iter):
                    best_score_this_iter = score
                    best_model_this_iter = model
                elif score == best_score_this_iter:
                    # Tie-breaking: prefer model already in ensemble (smaller final ensemble)
                    if model in bags_ensemble and best_model_this_iter not in bags_ensemble:
                        best_model_this_iter = model

            # Add best model to ensemble
            assert best_model_this_iter is not None
            bags_ensemble.append(best_model_this_iter)
            running_dev = running_dev.add(self._pred_arrays[best_model_this_iter][DataPartition.DEV])
            n_members += 1

            assert best_score_this_iter is not None
            trajectory.append((best_score_this_iter, list(bags_ensemble)))
            logger.info(
                f"Optimization iter {i}: dev_ensemble={best_score_this_iter:.6f}, ensemble_size={len(bags_ensemble)}"
            )

        # use_best backtracking: find the trajectory entry with the best score
        best_traj_idx = 0
        for idx in range(1, len(trajectory)):
            if self._is_better(trajectory[idx][0], trajectory[best_traj_idx][0]):
                best_traj_idx = idx

        best_bags = trajectory[best_traj_idx][1]
        best_score = trajectory[best_traj_idx][0]
        logger.info(f"Backtracking: best at step {best_traj_idx} (score={best_score:.6f}, {len(best_bags)} bags)")

        # Prune zero-weight models
        self.optimized_ensemble = {k: v for k, v in Counter(best_bags).items() if v > 0}
        logger.info("Ensemble selection completed.")

    def get_ens_input(self):
        """Get ensemble dict."""
        return self.bags
