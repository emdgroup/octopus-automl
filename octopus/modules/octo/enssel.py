"""Ensemble selection."""

# TODO
# - issue: ACC and BALACC need integer pooling values!
# - potential issue: check start_n, +1 or not
# - get FI and counts
# - display results summary
# - save results to experiment
# - create finale ensemble_bag containing all the models
# - ensemble models needs to provide finale predictions (dev, test)

import copy
from collections import Counter
from typing import Any

import pandas as pd
from attrs import define, field, validators
from upath import UPath

from octopus.logger import get_logger
from octopus.manager import ParallelResources
from octopus.metrics import Metrics
from octopus.metrics.utils import get_performance_from_predictions
from octopus.modules.octo.bag import BagBase
from octopus.types import MetricDirection
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
    resources: ParallelResources = field(validator=[validators.instance_of(ParallelResources)])
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

    @property
    def direction(self) -> MetricDirection:
        """Optuna direction."""
        return Metrics.get_direction(self.target_metric)

    def __attrs_post_init__(self):
        # initialization here due to "Python immutable default"
        self.bags = {}
        self.start_ensemble = {}
        self.optimized_ensemble = {}
        # get a trials existing in path_trials
        self._collect_trials()
        self._create_model_table()
        self._ensemble_scan()
        self._ensemble_optimization()

    def _collect_trials(self):
        """Get all trials save in path_trials and store properties in self.bags."""
        # Get all .joblib files in the directory
        joblib_files = [file for file in self.path_trials.iterdir() if file.is_file() and file.suffix == ".joblib"]

        # fill bags dict
        for file in joblib_files:
            bag: BagBase = joblib_load(file)
            self.bags[file] = {
                "id": bag.bag_id,
                "performance": bag.get_performance(resources=self.resources),
                "predictions": bag.get_predictions(resources=self.resources),
                "n_features_used_mean": bag.n_features_used_mean,
            }

    def _create_model_table(self):
        """Create model table."""
        df_lst = []
        for key, value in self.bags.items():
            s = pd.Series()
            s["id"] = value["id"]
            s["dev_pool"] = value["performance"]["dev_pool"]  # relevant
            s["test_pool"] = value["performance"]["test_pool"]
            s["dev_avg"] = value["performance"]["dev_avg"]
            s["test_avg"] = value["performance"]["test_avg"]
            s["n_features_used_mean"] = value["n_features_used_mean"]
            s["path"] = key
            df_lst.append(s)

        self.model_table = pd.concat(df_lst, axis=1).T

        # order of table is important, depending on metric,
        # (a) direction
        ascending = self.direction != MetricDirection.MAXIMIZE

        self.model_table = self.model_table.sort_values(by="dev_pool", ascending=ascending).reset_index(drop=True)

        logger.info("Model Table:")
        logger.info(f"\n{self.model_table.head(20)}")

    def _ensemble_models(self, bag_keys):
        """Esemble using all bags and their corresponding models provided by input."""
        # collect all predictions over inner folds and bags
        predictions: dict[str, dict[str, pd.DataFrame]] = {}
        performance_output = {}
        pool: dict[str, list[pd.DataFrame]] = {key: [] for key in ["dev", "test"]}

        for key in bag_keys:
            bag_predictions = self.bags[key]["predictions"]
            # remove 'ensemble'
            bag_predictions.pop("ensemble", 0)
            # concatenate and averag dev and test predictions from inner models
            for pred in bag_predictions.values():
                for part, pool_value in pool.items():
                    pool_value.append(pred[part])

        # created ensembled predictions (all inner models from all bags) for each partition
        predictions["ensemble"] = {}
        # Load the first bag once to determine target column dtype
        first_bag_key = bag_keys[0]
        first_bag = joblib_load(first_bag_key)
        for part, pool_value in pool.items():
            combined = pd.concat(pool_value, axis=0)
            # Identify numeric columns to average (exclude metadata and row_id)
            numeric_cols = combined.select_dtypes(include=["number"]).columns.tolist()
            if self.row_id_col in numeric_cols:
                numeric_cols.remove(self.row_id_col)
            for col in ["outer_split_id", "inner_split_id", "task_id"]:
                if col in numeric_cols:
                    numeric_cols.remove(col)

            ensemble = combined.groupby(by=self.row_id_col)[numeric_cols].mean().reset_index()
            for column in list(self.target_assignments.values()):
                if column in ensemble.columns:
                    ensemble[column] = ensemble[column].astype(first_bag.trainings[0].data_train[column].dtype)
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
            performance_output["dev_pool"] = performance["ensemble"]["dev"]
            performance_output["test_pool"] = performance["ensemble"]["test"]

        return performance_output

    def _ensemble_scan(self):
        """Scan for highest performing ensemble consisting of best N bags."""
        self.scan_table = pd.DataFrame(
            columns=[
                "#models",
                "dev_pool",
                "test_pool",
            ]
        )

        for i in range(len(self.model_table)):
            bag_keys = self.model_table[: i + 1]["path"].tolist()
            scores = self._ensemble_models(bag_keys)
            self.scan_table.loc[i] = [
                i + 1,
                scores["dev_pool"],
                scores["test_pool"],
            ]
        logger.info("Scan Table:")
        logger.info(f"\n{self.scan_table.head(20)}")

    def _ensemble_optimization(self):
        """Ensembling optimization with replacement.

        We start with the best N models derived from self.scan_table,
        assuming that it is sorted correctly. When there are multiple rows
        with the same best value, we take the last one (more models).
        """
        if self.direction == MetricDirection.MAXIMIZE:
            # Get all indices with the max value, then take the last one
            best_value = self.scan_table["dev_pool"].max()
            best_idx = self.scan_table[self.scan_table["dev_pool"] == best_value].index[-1]
        else:
            # Get all indices with the min value, then take the last one
            best_value = self.scan_table["dev_pool"].min()
            best_idx = self.scan_table[self.scan_table["dev_pool"] == best_value].index[-1]
        start_n = int(self.scan_table.loc[best_idx, "#models"])
        logger.info(f"Ensemble scan, number of included best models: {start_n}")
        logger.info(f"Ensemble scan, dev_pool value: {self.scan_table.loc[best_idx, 'dev_pool']}")
        logger.info(f"Ensemble scan, test_pool value: {self.scan_table.loc[best_idx, 'test_pool']}")

        # startn_bags dict with path as key and repeats=1 as value
        escan_ensemble: dict[UPath, int] = {}
        for _, row in self.model_table.head(start_n).iterrows():
            escan_ensemble[row["path"]] = 1

        # ensemble_optimmization, reference score
        # we start with the bags found in ensemble scan
        results: list[tuple[str | UPath, float, list[UPath]]] = []
        start_bags = list(escan_ensemble.keys())
        start_perf = self._ensemble_models(start_bags)["dev_pool"]
        logger.info("Ensemble optimization")
        logger.info(f"Start performance: {start_perf}")
        # record start performance
        results.append(("ensemble scan", start_perf, copy.deepcopy(start_bags)))

        # optimization
        bags_ensemble = copy.deepcopy(start_bags)
        best_global = copy.deepcopy(start_perf)

        for i in range(self.max_n_iterations):
            df = pd.DataFrame(columns=["model", "performance_dev", "performance_test"])
            # find additional model
            for model in self.model_table["path"].tolist():
                bags_lst = copy.deepcopy(bags_ensemble)
                bags_lst.append(model)
                perf = self._ensemble_models(bags_lst)
                df.loc[len(df)] = [model, perf["dev_pool"], perf["test_pool"]]

            if self.direction == MetricDirection.MAXIMIZE:
                idx_best = df["performance_dev"].idxmax()
                best_model: UPath = df.loc[idx_best]["model"]
                best_performance: float = df.loc[idx_best]["performance_dev"]  # type: ignore[assignment]
                performance_test: float = df.loc[idx_best]["performance_test"]  # type: ignore[assignment]
                if best_performance < best_global:  # stop if performance worsens
                    break  # stop ensembling
                else:
                    best_global = best_performance
                    logger.info(
                        f"iteration: {i}, performance_dev: {best_performance}, performance_test {performance_test}"
                    )
            else:  # minimize
                idx_best = df["performance_dev"].idxmin()
                best_model = df.loc[idx_best]["model"]
                best_performance = df.loc[idx_best]["performance_dev"]  # type: ignore[assignment]
                performance_test = df.loc[idx_best]["performance_test"]  # type: ignore[assignment]
                if best_performance > best_global:  # stop if performance worsens
                    break
                else:
                    best_global = best_performance
                    logger.info(
                        f"iteration: {i}, performance_dev: {best_performance}, performance_test {performance_test}"
                    )

            # add best model to ensemble
            bags_ensemble.append(best_model)

            # record results
            results.append((best_model, best_performance, copy.deepcopy(bags_ensemble)))

        # store start bargs from ensemble scan
        self.start_ensemble = dict(Counter(start_bags))

        # store optimization results
        results_df = pd.DataFrame(results, columns=["model", "performance", "bags_lst"])
        self.optimized_ensemble = dict(Counter(results_df.iloc[-1]["bags_lst"]))
        logger.info("Ensemble selection completed.")

    def get_ens_input(self):
        """Get ensemble dict."""
        return self.bags
