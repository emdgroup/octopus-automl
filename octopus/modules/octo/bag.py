"""Octo Bags."""

# import concurrent.futures
# import logging
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd
from attrs import define, field, validators
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from upath import UPath

# sklearn imports for compatibility
from octopus.logger import get_logger
from octopus.manager import ray_parallel
from octopus.metrics.utils import get_performance_from_predictions
from octopus.modules.octo.training import Training, fi_storage_key, parse_fi_storage_key

# Adjust this import path as needed depending on your package layout
from octopus.types import DataPartition, FIComputeMethod, LogGroup, MLType

logger = get_logger()


class TrainingWithLogging:
    """Logging class for trainings."""

    def __init__(self, inner_training: Training, idx: int, logger, log_group_cls, log_prefix: str = "EXP"):
        self._inner = inner_training
        self._idx = idx
        self._logger = logger
        self._log_group_cls = log_group_cls
        self._log_prefix = log_prefix

    def fit(self) -> Training:
        """Fit function."""
        # Your logging policy lives here
        self._logger.set_log_group(self._log_group_cls.PROCESSING, f"{self._log_prefix} {self._idx}")
        self._logger.info("Starting execution")
        try:
            result = self._inner.fit()
            self._logger.set_log_group(self._log_group_cls.PREPARE_EXECUTION, f"{self._log_prefix} {self._idx}")
            self._logger.info("Completed successfully")
            return result
        except Exception as e:
            self._logger.exception(f"Exception occurred while executing training {self._idx}: {e!s}")
            raise e


class FeatureImportanceWithLogging:
    """Logging wrapper for feature importance calculations."""

    def __init__(self, training: Training, idx, fi_type, partition, logger, log_group_cls, log_prefix="FI"):
        self._training = training
        self._idx = idx
        self._fi_type = fi_type
        self._partition = partition
        self._logger = logger
        self._log_group_cls = log_group_cls
        self._log_prefix = log_prefix

    def fit(self) -> Training:
        """Calculate feature importance for the training.

        Uses fit() method for Ray compatibility.
        """
        training_id = getattr(self._training, "training_id", self._idx)
        self._logger.set_log_group(self._log_group_cls.PROCESSING, f"{self._log_prefix} {training_id}")
        self._logger.info(f"Starting {self._fi_type} feature importance calculation")

        try:
            self._training.calculate_fi(self._fi_type, partition=self._partition)

            self._logger.set_log_group(self._log_group_cls.PREPARE_EXECUTION, f"{self._log_prefix} {training_id}")
            self._logger.info(f"Completed {self._fi_type} feature importance calculation")
            return self._training
        except Exception as e:
            self._logger.exception(f"Exception in {self._fi_type} FI calculation for training {training_id}: {e!s}")
            # Return the training object even if FI calculation failed
            return self._training


@define
class BagBase(BaseEstimator):
    """Base Container for Trainings.

    Supports:
    - execution of trainings, sequential/parallel
    - saving/loading
    - sklearn compatibility for inference tools like SHAP and permutation importance
    """

    bag_id: str = field(validator=[validators.instance_of(str)])
    trainings: list["Training"] = field(validator=[validators.instance_of(list)])
    target_metric: str = field(validator=[validators.instance_of(str)])
    target_assignments: dict = field(validator=[validators.instance_of(dict)])
    row_id_col: str = field(validator=[validators.instance_of(str)])
    ml_type: MLType = field(validator=[validators.instance_of(MLType)])
    log_dir: UPath = field(validator=[validators.instance_of(UPath)])
    train_status: bool = field(default=False)

    # bag training outputs, initialized in post_init
    feature_importances: dict[str, pd.DataFrame | dict[str, pd.DataFrame]] = field(
        init=False, validator=[validators.instance_of(dict)]
    )
    n_features_used_mean: float = field(init=False, validator=[validators.instance_of(float)])

    @property
    def feature_groups(self) -> dict[str, list[str]]:
        """Experiment wide feature groups."""
        # assuming that there is at least one training
        return self.trainings[0].feature_groups

    @property
    def classes_(self):
        """Get unique classes from all trainings in the bag.

        Returns:
            numpy.ndarray or None: Array of unique class labels sorted in ascending order
                                  for classification/multiclass tasks, None otherwise.
        """
        # Return None for non-classification tasks
        if self.ml_type not in (MLType.BINARY, MLType.MULTICLASS):
            return None

        # Check if we have any trainings
        if not self.trainings:
            return None

        # Collect all unique classes from all trainings
        all_classes = set()

        for training in self.trainings:
            # Check if the training has a fitted model with classes_ attribute
            if hasattr(training, "model") and hasattr(training.model, "classes_"):
                # Add classes from this training to the set
                training_classes = training.model.classes_
                if training_classes is not None:
                    all_classes.update(training_classes)

        # If no classes found, return None
        if not all_classes:
            return None

        # Convert to numpy array and sort (sklearn compatible)
        return np.array(sorted(all_classes))

    @property
    def positive_class(self):
        """Get the positive class for binary classification from trainings.

        Returns:
            The positive class value, or None if not applicable/determinable.

        Raises:
            ValueError: If training is missing config_training, missing positive_class
                       in config_training, or if trainings have inconsistent positive_class values.

        For binary classification, this method requires that each training has
        positive_class explicitly set in config_training. No fallback solutions are provided.
        """
        # Only applicable for binary classification
        if self.ml_type != MLType.BINARY:
            return None

        # Initialize _positive_class if it doesn't exist (e.g., when loaded from pickle)
        if not hasattr(self, "_positive_class"):
            self._positive_class = None

        # Return cached value if already computed
        if self._positive_class is not None:
            return self._positive_class

        # Check if we have any trainings
        if not self.trainings:
            return None

        # Require explicit positive_class in training configurations
        # All trainings must have this property set
        for training in self.trainings:
            if not (hasattr(training, "config_training") and isinstance(training.config_training, dict)):
                raise ValueError(f"Training {getattr(training, 'training_id', 'unknown')} missing config_training")

            positive_class = training.config_training.get("positive_class")
            if positive_class is None:
                raise ValueError(
                    f"Training {getattr(training, 'training_id', 'unknown')} missing positive_class in config_training"
                )

            # Cache the first valid positive_class found
            if self._positive_class is None:
                self._positive_class = positive_class

            # Verify all trainings have the same positive_class
            elif self._positive_class != positive_class:
                raise ValueError(
                    f"Inconsistent positive_class values across trainings: {self._positive_class} vs {positive_class}"
                )

        return self._positive_class

    def __attrs_post_init__(self):
        # initialization here due to "Python immutable default"
        self.feature_importances = {}
        self.n_features_used_mean = 0.0
        self._positive_class = None  # Will be inferred when needed

    def get_params(self, deep=True):
        """Get parameters for this estimator (sklearn requirement)."""
        return {
            "bag_id": self.bag_id,
            "target_metric": self.target_metric,
            "ml_type": self.ml_type,
        }

    def set_params(self, **params):
        """Set parameters for this estimator (sklearn requirement)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def _more_tags(self):
        """Provide additional tags for sklearn compatibility."""
        return {
            "requires_fit": True,
            "no_validation": True,  # We handle our own validation
            "_xfail_checks": {
                "check_estimators_dtypes": "Custom data handling",
                "check_fit_score_takes_y": "Custom scoring method",
            },
        }

    def _train_parallel(self, num_assigned_cpus: int):
        """Run trainings in parallel using Ray (delegated to ray_parallel)."""
        # Orchestrate with Ray; exceptions propagate only if your wrapper re-raises.
        self.trainings = ray_parallel.run_parallel_inner(
            self.bag_id,
            trainings=[
                TrainingWithLogging(
                    inner_training=t,
                    idx=idx,
                    logger=logger,
                    log_group_cls=LogGroup,
                    log_prefix="EXP",
                )
                for idx, t in enumerate(self.trainings)
            ],
            log_dir=self.log_dir,
            num_assigned_cpus=num_assigned_cpus,
        )

    def _train_sequential(self):
        """Run trainings sequentially in the current process."""
        successful_trainings = []
        failed_trainings = []

        for idx, training in enumerate(self.trainings):
            training_id = getattr(training, "training_id", idx)
            try:
                training.fit()
                successful_trainings.append(training)
                logger.info(
                    f"Inner sequential training completed for bag_id {self.bag_id} and training id {training_id}"
                )
            except Exception as e:  # pylint: disable=broad-except
                failed_trainings.append((training_id, str(e), type(e).__name__))
                logger.error(
                    f"Training failed for bag_id {self.bag_id}, training_id {training_id}: {e}, type: {type(e).__name__}"
                )

        # Log summary of training results
        total_trainings = len(self.trainings)
        successful_count = len(successful_trainings)
        failed_count = len(failed_trainings)

        logger.info(
            f"Bag {self.bag_id} training summary: {successful_count}/{total_trainings} successful, {failed_count} failed"
        )

        if failed_trainings:
            logger.warning(f"Failed trainings in bag {self.bag_id}:")
            for training_id, error_msg, error_type in failed_trainings:
                logger.warning(f"  - Training {training_id}: {error_type} - {error_msg}")

        self.trainings = successful_trainings

    def fit(self, num_assigned_cpus: int):
        """Run all available trainings."""
        if num_assigned_cpus > 1:
            self._train_parallel(num_assigned_cpus)
        else:
            self._train_sequential()

        self.train_status = True

        # get used features in bag
        n_feat_lst = []
        for training in self.trainings:
            n_feat_lst.append(float(len(training.features_used)))

        if not n_feat_lst:
            raise ValueError(f"Empty feature list in bag: '{self.bag_id}'.")

        self.n_features_used_mean = mean(n_feat_lst)

    def get_predictions(self, num_assigned_cpus: int):
        """Extract bag predictions for train, dev, and test.

        Args:
            num_assigned_cpus: Number of CPUs to use for parallel processing.

        Returns:
            dict: Dictionary containing predictions for each training and ensemble.
                 Keys are training IDs plus 'ensemble', values are dicts with 'train', 'dev', 'test' keys.
        """
        if not self.train_status:
            logger.set_log_group(LogGroup.TRAINING)
            logger.info("Running trainings first to be able to get scores")
            self.fit(num_assigned_cpus)

        predictions = {}
        pool: dict[str, list[pd.DataFrame]] = {key: [] for key in ["train", "dev", "test"]}

        for training in self.trainings:
            predictions[training.training_id] = training.predictions
            for part, pool_value in pool.items():
                pool_value.append(training.predictions[part])

        # Create ensemble predictions for each partition
        predictions["ensemble"] = {}
        for part, pool_value in pool.items():
            # Get metadata from first training (all have same outer_split_id and task_id)
            first_pred = pool_value[0]
            outer_split_id = first_pred["outer_split_id"].iloc[0]
            task_id = first_pred["task_id"].iloc[0]

            # Concatenate and group by row_id, averaging only numeric columns
            combined = pd.concat(pool_value, axis=0)
            # Identify numeric columns to average (exclude metadata and row_id)
            numeric_cols = combined.select_dtypes(include=["number"]).columns.tolist()
            if self.row_id_col in numeric_cols:
                numeric_cols.remove(self.row_id_col)
            # Remove metadata columns from averaging
            for col in ["outer_split_id", "inner_split_id", "task_id"]:
                if col in numeric_cols:
                    numeric_cols.remove(col)

            # Group by row_id and average only numeric prediction columns
            ensemble = combined.groupby(self.row_id_col)[numeric_cols].mean().reset_index()

            # Restore target column dtype
            for column in list(self.target_assignments.values()):
                if column in ensemble.columns:
                    ensemble[column] = ensemble[column].astype(self.trainings[0].data_train[column].dtype)

            # Add metadata columns for ensemble predictions
            ensemble["outer_split_id"] = outer_split_id
            ensemble["inner_split_id"] = "ensemble"
            ensemble["partition"] = part
            ensemble["task_id"] = task_id

            predictions["ensemble"][part] = ensemble

        return predictions

    def get_performance(self, num_assigned_cpus: int, metric: str | None = None):
        """Get performance using get_performance_from_predictions utility.

        Args:
            num_assigned_cpus: Number of CPUs to use for parallel processing when getting predictions.
            metric: The metric to evaluate. Defaults to self.target_metric when None.

        Returns:
            dict: Dictionary with performance values in the same format as get_performance()
        """
        if metric is None:
            metric = self.target_metric

        # Get predictions from the bag
        predictions = self.get_predictions(num_assigned_cpus=num_assigned_cpus)

        # Calculate performance using the utility function
        performance = get_performance_from_predictions(
            predictions=predictions,
            target_metric=metric,
            target_assignments=self.target_assignments,
            positive_class=self.positive_class,
        )

        # Create performance_output dictionary with restructured data
        performance_output: dict[str, list[float] | float] = {}

        # Collect lists for train, dev, test from all non-ensemble trainings
        train_lst = []
        dev_lst = []
        test_lst = []

        for training_id, partitions in performance.items():
            if training_id != "ensemble":
                train_lst.append(partitions["train"])
                dev_lst.append(partitions["dev"])
                test_lst.append(partitions["test"])

        # Calculate averages
        performance_output["train_avg"] = mean(train_lst)
        performance_output["train_lst"] = train_lst
        performance_output["dev_avg"] = mean(dev_lst)
        performance_output["dev_lst"] = dev_lst
        performance_output["test_avg"] = mean(test_lst)
        performance_output["test_lst"] = test_lst

        # Add ensemble performance with renamed keys
        if "ensemble" in performance:
            performance_output["train_ensemble"] = performance["ensemble"]["train"]
            performance_output["dev_ensemble"] = performance["ensemble"]["dev"]
            performance_output["test_ensemble"] = performance["ensemble"]["test"]

        return performance_output

    def get_performance_df(self, num_assigned_cpus: int, metric: str) -> pd.DataFrame:
        """Convert get_performance() dict to standard scores DataFrame.

        Args:
            num_assigned_cpus: Number of CPUs to use for parallel processing.
            metric: The metric name (e.g. "MAE", "accuracy").

        Returns:
            DataFrame with columns: metric, partition, aggregation, fold, value
        """
        perf = self.get_performance(num_assigned_cpus=num_assigned_cpus, metric=metric)
        rows = []

        # Per-fold scores
        for partition in ["train", "dev", "test"]:
            lst_key = f"{partition}_lst"
            avg_key = f"{partition}_avg"
            ensemble_key = f"{partition}_ensemble"

            if lst_key in perf:
                for fold_idx, val in enumerate(perf[lst_key]):
                    rows.append(
                        {
                            "metric": metric,
                            "partition": partition,
                            "aggregation": "per_fold",
                            "fold": fold_idx,
                            "value": val,
                        }
                    )
            if avg_key in perf:
                rows.append(
                    {
                        "metric": metric,
                        "partition": partition,
                        "aggregation": "avg",
                        "fold": None,
                        "value": perf[avg_key],
                    }
                )
            if ensemble_key in perf:
                rows.append(
                    {
                        "metric": metric,
                        "partition": partition,
                        "aggregation": "ensemble",
                        "fold": None,
                        "value": perf[ensemble_key],
                    }
                )

        return pd.DataFrame(rows)

    def get_predictions_df(self, num_assigned_cpus: int) -> pd.DataFrame:
        """Concat all training predictions into a single DataFrame.

        Args:
            num_assigned_cpus: Number of CPUs to use for parallel processing.

        Returns:
            DataFrame with all predictions from get_predictions().
        """
        predictions = self.get_predictions(num_assigned_cpus=num_assigned_cpus)
        all_dfs = []
        for _split_id, partitions in predictions.items():
            if isinstance(partitions, dict):
                for _part_name, df in partitions.items():
                    all_dfs.append(df)
            elif isinstance(partitions, pd.DataFrame):
                all_dfs.append(partitions)
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return pd.DataFrame()

    def get_feature_importances_df(self) -> pd.DataFrame:
        """Concat per-training FI into a single DataFrame with fi_method, fi_dataset, and training_id columns.

        Returns:
            DataFrame with columns: feature, importance, fi_method, fi_dataset, training_id
        """
        all_dfs: list[pd.DataFrame] = []
        for key, value in self.feature_importances.items():
            # Skip aggregated keys like "internal_mean", "permutation_dev_count", etc.
            if key.endswith("_mean") or key.endswith("_count"):
                continue
            if isinstance(value, dict):
                # Per-training: {fi_key: DataFrame}
                training_id = key
                for fi_key, df in value.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        temp = df[["feature", "importance"]].copy()
                        method, dataset = parse_fi_storage_key(fi_key)
                        temp["fi_method"] = method
                        temp["fi_dataset"] = dataset
                        temp["training_id"] = training_id
                        all_dfs.append(temp)
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return pd.DataFrame()

    def _calculate_fi_parallel(self, fi_type: FIComputeMethod, partition: str, num_assigned_cpus: int):
        """Calculate feature importance in parallel using Ray."""
        # Execute feature importance calculations in parallel
        # Use the same pattern as training execution
        results = ray_parallel.run_parallel_inner(
            bag_id=self.bag_id,
            trainings=[
                FeatureImportanceWithLogging(
                    training=t,
                    idx=idx,
                    fi_type=fi_type,
                    partition=partition,
                    logger=logger,
                    log_group_cls=LogGroup,
                    log_prefix="FI",
                )
                for idx, t in enumerate(self.trainings)
            ],
            log_dir=self.log_dir,
            num_assigned_cpus=num_assigned_cpus,
        )

        # Update trainings with results (should be the same objects with FI calculated)
        self.trainings = results

    def _calculate_fi_sequential(self, fi_type: FIComputeMethod, partition: str):
        """Calculate feature importance sequentially."""
        successful_calculations = []
        failed_calculations = []

        for idx, training in enumerate(self.trainings):
            training_id = getattr(training, "training_id", idx)
            try:
                training.calculate_fi(fi_type, partition=partition)

                successful_calculations.append(training)
                logger.info(
                    f"Feature importance ({fi_type}) calculation completed for bag_id {self.bag_id} and training id {training_id}"
                )
            except Exception as e:  # pylint: disable=broad-except
                failed_calculations.append((training_id, str(e), type(e).__name__))
                logger.error(
                    f"Feature importance ({fi_type}) calculation failed for bag_id {self.bag_id}, training_id {training_id}: {e}, type: {type(e).__name__}"
                )
                # Still include the training even if FI calculation failed
                successful_calculations.append(training)

        # Log summary of FI calculation results
        total_trainings = len(self.trainings)
        successful_count = len(successful_calculations)
        failed_count = len(failed_calculations)

        logger.info(
            f"Bag {self.bag_id} feature importance ({fi_type}) summary: {successful_count}/{total_trainings} successful, {failed_count} failed"
        )

        if failed_calculations:
            logger.warning(f"Failed feature importance ({fi_type}) calculations in bag {self.bag_id}:")
            for training_id, error_msg, error_type in failed_calculations:
                logger.warning(f"  - Training {training_id}: {error_type} - {error_msg}")

        self.trainings = successful_calculations

    def _calculate_fi(self, fi_type: FIComputeMethod, num_assigned_cpus: int, partition=DataPartition.DEV):
        """Calculate feature importance using parallel or sequential execution."""
        if num_assigned_cpus > 1:
            self._calculate_fi_parallel(fi_type=fi_type, partition=partition, num_assigned_cpus=num_assigned_cpus)
        else:
            self._calculate_fi_sequential(fi_type=fi_type, partition=partition)

    def get_selected_features(self, fi_methods: list[FIComputeMethod] | None = None) -> list[str]:
        """Get features selected by model, depending on fi method.

        The list of selected features will be derived only from one feature
        importance method out of the ones specified in fi_methods,
        with the following ranking: (1) permutation (2) shap (3) internal,
        (4) constant.
        """
        # we assume that feature_importances were previously calculated
        if fi_methods is None:
            fi_methods = []

        if FIComputeMethod.PERMUTATION in fi_methods:
            fi_df = self.feature_importances[fi_storage_key(FIComputeMethod.PERMUTATION, "dev", "mean")]
        elif FIComputeMethod.SHAP in fi_methods:
            fi_df = self.feature_importances[fi_storage_key(FIComputeMethod.SHAP, "dev", "mean")]
        elif FIComputeMethod.INTERNAL in fi_methods:
            fi_df = self.feature_importances[fi_storage_key(FIComputeMethod.INTERNAL, stat="mean")]
        elif FIComputeMethod.CONSTANT in fi_methods:
            fi_df = self.feature_importances[fi_storage_key(FIComputeMethod.CONSTANT, stat="mean")]
        else:
            logger.set_log_group(LogGroup.RESULTS)
            logger.info("No features selected, return empty list")
            return []

        # only keep nonzero features
        fi_df = fi_df[fi_df["importance"] != 0]  # type: ignore[index]

        # store group features
        groups_df = fi_df[fi_df["feature"].str.startswith("group")].copy()

        # remove all group features -> single features
        fi_df = fi_df[~fi_df["feature"].str.startswith("group")]
        feat_single = fi_df["feature"].tolist()

        # For each feature group with positive importance (only),
        # check if any feature is in feat_single. In not, add the
        # one with the largest feature importance
        groups = groups_df[groups_df["importance"] > 0]["feature"].tolist()
        feat_additional = []
        for key in groups:
            features = self.feature_groups.get(key, [])
            if features and not any(feature in feat_single for feature in features):
                # Find the feature with the highest importance in fi_df
                feature_importances = fi_df[fi_df["feature"].isin(features)]
                if not feature_importances.empty:
                    best_feature = feature_importances.loc[feature_importances["importance"].idxmax(), "feature"]
                    feat_additional.append(best_feature)

        # Add the additional features to feat_single and remove duplicates
        feat_all = list(set(feat_single + feat_additional))

        logger.set_log_group(LogGroup.RESULTS, f"BAG {self.bag_id}")
        logger.info(f"Number of selected features: {len(feat_all)}")
        logger.info(f"Number of single features: {len(feat_single)}")
        logger.info(f"Number of features from groups: {len(feat_additional)}")

        return sorted(feat_all, key=lambda x: (len(x), sorted(x)))

    def calculate_feature_importances(
        self,
        fi_methods: list[FIComputeMethod] | None,
        partitions: list[DataPartition | str] | None,
        num_assigned_cpus: int,
    ):
        """Extract feature importances of all models in bag."""
        # we always extract internal feature importances, if available
        if fi_methods is None:
            fi_methods = []
        if partitions is None:
            partitions = [DataPartition.DEV, DataPartition.TEST]

        self._calculate_fi(fi_type=FIComputeMethod.INTERNAL, num_assigned_cpus=num_assigned_cpus)

        for method in fi_methods:
            if method == FIComputeMethod.INTERNAL:
                continue  # already done
            elif method in (FIComputeMethod.SHAP, FIComputeMethod.PERMUTATION):
                for partition in partitions:
                    self._calculate_fi(fi_type=method, partition=partition, num_assigned_cpus=num_assigned_cpus)
            elif method in (FIComputeMethod.LOFO, FIComputeMethod.CONSTANT):
                self._calculate_fi(fi_type=method, num_assigned_cpus=num_assigned_cpus)
            else:
                raise ValueError(f"Feature importance method {method} not supported.")

        # save feature importances for every training in bag
        for training in self.trainings:
            self.feature_importances[training.training_id] = training.feature_importances

        # summary feature importances for all trainings (mean + count)
        # Aggregate all computed partitions dynamically (not just "dev")
        for method in fi_methods:
            if method in (FIComputeMethod.INTERNAL, FIComputeMethod.CONSTANT):
                keys_to_aggregate = [fi_storage_key(method)]
            else:
                keys_to_aggregate = [fi_storage_key(method, p) for p in partitions]

            for method_key in keys_to_aggregate:
                fi_pool = []
                for training in self.trainings:
                    fi_df = training.feature_importances.get(method_key)
                    if fi_df is not None and not fi_df.empty:
                        fi_pool.append(fi_df)

                if not fi_pool:
                    logger.warning(f"No FI data found for key '{method_key}' across trainings.")
                    continue

                fi = pd.concat(fi_pool, axis=0)

                # calculate mean feature importances, keep zero entries
                mean_key = method_key + "_mean"
                self.feature_importances[mean_key] = (
                    fi[["feature", "importance"]]
                    .groupby(by="feature")
                    .sum()
                    .div(len(fi_pool))  # mean over trainings that produced FI (not all may succeed)
                    .sort_values(by="importance", ascending=False)
                    .reset_index()
                )

                # calculate count feature importances, keep zero entries
                non_zero_importances = (
                    fi[fi["importance"] != 0][["feature", "importance"]].groupby(by="feature").count()
                )
                # Create a DataFrame with all features, init importance counts to zero
                all_features = pd.DataFrame(fi["feature"].unique(), columns=["feature"])
                all_features["importance"] = 0
                # Update the importance counts for non-zero importances
                all_features = all_features.set_index("feature")
                all_features.update(non_zero_importances)
                all_features = all_features.reset_index()
                # Sort and reset index
                count_key = method_key + "_count"
                self.feature_importances[count_key] = all_features.sort_values(
                    by="importance", ascending=False
                ).reset_index(drop=True)

        return self.feature_importances

    def predict(self, x):
        """Predict with sklearn compatibility."""
        # Check if the bag has fitted trainings
        if not self.trainings:
            raise ValueError("No trainings available in bag")

        # Check if all trainings are fitted
        for training in self.trainings:
            if not getattr(training, "is_fitted", False):
                raise ValueError(f"Training {training.training_id} is not fitted")

        preds_lst = []
        weights_lst = []
        for training in self.trainings:
            train_w = training.training_weight
            weights_lst.append(train_w)
            preds_lst.append(train_w * training.predict(x))

        # return mean of weighted predictions
        return np.sum(np.array(preds_lst), axis=0) / sum(weights_lst)

    def predict_proba(self, x):
        """Predict_proba with sklearn compatibility."""
        # Only available for classification tasks
        if self.ml_type not in (MLType.BINARY, MLType.MULTICLASS):
            raise AttributeError(f"predict_proba is not available for ml_type '{self.ml_type}'")

        # Check if the bag has fitted trainings
        if not self.trainings:
            raise ValueError("No trainings available in bag")

        # Check if all trainings are fitted
        for training in self.trainings:
            if not getattr(training, "is_fitted", False):
                raise ValueError(f"Training {training.training_id} is not fitted")

        preds_lst = []
        weights_lst = []
        for training in self.trainings:
            train_w = training.training_weight
            weights_lst.append(train_w)
            preds_lst.append(train_w * training.predict_proba(x))

        # return mean of weighted predictions
        return np.sum(np.array(preds_lst), axis=0) / sum(weights_lst)


@define
class BagClassifier(BagBase, ClassifierMixin):
    """Bag for classification tasks with sklearn ClassifierMixin."""

    @property
    def _estimator_type(self) -> str:  # type: ignore[override]
        """Return the estimator type for sklearn compatibility."""
        return "classifier"


@define
class BagRegressor(BagBase, RegressorMixin):
    """Bag for regression tasks with sklearn RegressorMixin."""

    @property
    def _estimator_type(self) -> str:  # type: ignore[override]
        """Return the estimator type for sklearn compatibility."""
        return "regressor"

    def predict_proba(self, x):
        """Predict_proba not available for regression tasks."""
        raise AttributeError("predict_proba is not available for regression tasks")


class _BagFunction:
    """Class to create appropriate Bag instance based on ml_type (factory function).

    Args:
        **kwargs: Arguments to pass to the Bag constructor

    Returns:
        BagClassifier or BagRegressor: Appropriate Bag instance based on ml_type
    """

    def __call__(self, **kwargs: Any) -> BagClassifier | BagRegressor:
        ml_type = kwargs.get("ml_type", MLType.REGRESSION)
        if ml_type in (MLType.BINARY, MLType.MULTICLASS):
            return BagClassifier(**kwargs)
        else:
            return BagRegressor(**kwargs)


Bag = _BagFunction()
