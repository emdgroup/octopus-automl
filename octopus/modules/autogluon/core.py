"""AutoGluon execution module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from attrs import Factory, define, field
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from upath import UPath

from octopus._optional.autogluon import (
    TabularPredictor,
    accuracy,
    average_precision,
    balanced_accuracy,
    f1,
    log_loss,
    mcc,
    mean_absolute_error,
    precision,
    r2,
    recall,
    roc_auc,
    root_mean_squared_error,
)
from octopus.logger import LogGroup, get_logger
from octopus.manager.ray_parallel import setup_ray_for_external_library
from octopus.modules.base import FIDataset, FIMethod, MLModuleExecution, ResultType
from octopus.modules.utils import get_score_from_model
from octopus.study.context import StudyContext

if TYPE_CHECKING:
    from octopus.modules.autogluon.module import AutoGluon  # noqa: F401

logger = get_logger()


class SklearnClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn classifier wrapper for AutoGluon predictor."""

    def __init__(self, predictor: TabularPredictor):
        self.predictor = predictor
        self.classes_ = self.predictor.class_labels
        self.n_classes_ = len(self.classes_)
        self.feature_names_in_ = self.predictor.original_features
        self.n_features_in_ = len(self.feature_names_in_)
        self._is_fitted = True

    def predict(self, x):
        """Predict class labels."""
        return self.predictor.predict(x, as_pandas=False)

    def predict_proba(self, x):
        """Predict class probabilities."""
        probabilities = self.predictor.predict_proba(x, as_pandas=False, as_multiclass=True)
        return probabilities

    def fit(self, x, y):
        """Fit method - not implemented as predictor is already fitted."""
        raise NotImplementedError("This classifier is already fitted. Only for inference use.")


class SklearnRegressor(BaseEstimator, RegressorMixin):
    """Sklearn regressor wrapper for AutoGluon predictor."""

    def __init__(self, predictor: TabularPredictor):
        self.predictor = predictor
        self.feature_names_in_ = self.predictor.original_features
        self.n_features_in_ = len(self.feature_names_in_)
        self.n_outputs_ = 1
        self._is_fitted = True

    def predict(self, x):
        """Predict target values."""
        return self.predictor.predict(x, as_pandas=False)

    def fit(self, x, y):
        """Fit method - not implemented as predictor is already fitted."""
        raise NotImplementedError("This regressor is already fitted. Only for inference use.")


# Mapping of Octopus metrics to AutoGluon metrics
metrics_inventory_autogluon = {
    "AUCROC": roc_auc,
    "ACC": accuracy,
    "ACCBAL": balanced_accuracy,
    "AUCPR": average_precision,
    "F1": f1,
    "LOGLOSS": log_loss,
    "MAE": mean_absolute_error,
    "MCC": mcc,
    "MSE": root_mean_squared_error,
    "NEGBRIERSCORE": "brier_score_loss",
    "PRECISION": precision,
    "RECALL": recall,
    "R2": r2,
    "RMSE": root_mean_squared_error,
}


@define
class AutoGluonModule(MLModuleExecution["AutoGluon"]):
    """AutoGluon execution module. Created by AutoGluon.create_module()."""

    # Internal state (temporary, available during fit)
    _num_cpus_allocated: int = field(init=False, default=1)
    """Allocated CPU count after validation."""

    _study: StudyContext | None = field(init=False, default=None)
    """StudyContext reference (temporary state during fit)."""

    _output_dir: Any = field(init=False, default=None)
    """Output directory (temporary state during fit)."""

    _feature_groups: dict = field(init=False, default=Factory(dict))
    """Feature groups (temporary state during fit)."""

    _x_traindev: pd.DataFrame | None = field(init=False, default=None)
    """Training/development features (temporary state during fit)."""

    _y_traindev: pd.DataFrame | None = field(init=False, default=None)
    """Training/development target (temporary state during fit)."""

    _x_test: pd.DataFrame | None = field(init=False, default=None)
    """Test features (temporary state during fit)."""

    _y_test: pd.DataFrame | None = field(init=False, default=None)
    """Test target (temporary state during fit)."""

    _row_traindev: pd.Series | None = field(init=False, default=None)
    """Row IDs for traindev (temporary state during fit)."""

    _row_test: pd.Series | None = field(init=False, default=None)
    """Row IDs for test (temporary state during fit)."""

    _feature_cols: list[str] | None = field(init=False, default=None)
    """Feature columns (temporary state during fit)."""

    _outersplit_id: int | None = field(init=False, default=None)
    """Fold ID (temporary state during fit)."""

    @property
    def target_assignments(self) -> dict:
        """Target column assignments (available during fit)."""
        if self._study is None:
            raise ValueError("StudyContext not available - fit() not called")
        return self._study.target_assignments

    @property
    def target_metric(self) -> str:
        """Target metric (available during fit)."""
        if self._study is None:
            raise ValueError("StudyContext not available - fit() not called")
        return self._study.target_metric

    @property
    def metrics(self) -> list[str]:
        """All metrics (available during fit)."""
        if self._study is None:
            raise ValueError("StudyContext not available - fit() not called")
        return self._study.metrics

    @property
    def ml_type(self) -> str:
        """ML type (available during fit)."""
        if self._study is None:
            raise ValueError("StudyContext not available - fit() not called")
        return self._study.ml_type

    @property
    def positive_class(self) -> Any:
        """Positive class (available during fit). None for regression."""
        if self._study is None:
            raise ValueError("StudyContext not available - fit() not called")
        return self._study.positive_class

    @property
    def row_column(self) -> str:
        """Row ID column name (available during fit)."""
        if self._study is None:
            raise ValueError("StudyContext not available - fit() not called")
        return self._study.row_id_col

    @property
    def row_traindev(self) -> pd.Series:
        """Row IDs for traindev (available during fit)."""
        if self._row_traindev is None:
            raise ValueError("row_traindev not available - fit() not called")
        return self._row_traindev

    @property
    def row_test(self) -> pd.Series:
        """Row IDs for test (available during fit)."""
        if self._row_test is None:
            raise ValueError("row_test not available - fit() not called")
        return self._row_test

    @property
    def feature_groups(self) -> dict:
        """Feature groups (available during fit)."""
        return self._feature_groups

    @property
    def path_results(self) -> Any:
        """Results path (available during fit)."""
        if self._output_dir is None:
            raise ValueError("Output dir not available - fit() not called")
        return self._output_dir / "results"

    @property
    def x_traindev(self) -> pd.DataFrame:
        """Training/development features (available during fit)."""
        if self._x_traindev is None:
            raise ValueError("x_traindev not available - fit() not called or already completed")
        return self._x_traindev

    @property
    def y_traindev(self) -> pd.DataFrame:
        """Training/development target (available during fit)."""
        if self._y_traindev is None:
            raise ValueError("y_traindev not available - fit() not called or already completed")
        return self._y_traindev

    @property
    def x_test(self) -> pd.DataFrame:
        """Test features (available during fit)."""
        if self._x_test is None:
            raise ValueError("x_test not available - fit() not called or already completed")
        return self._x_test

    @property
    def y_test(self) -> pd.DataFrame:
        """Test target (available during fit)."""
        if self._y_test is None:
            raise ValueError("y_test not available - fit() not called or already completed")
        return self._y_test

    @property
    def ag_train_data(self) -> pd.DataFrame:
        """Combined training data (features + target)."""
        return pd.concat([self.x_traindev, self.y_traindev], axis=1)

    @property
    def ag_test_data(self) -> pd.DataFrame:
        """Combined test data (features + target)."""
        return pd.concat([self.x_test, self.y_test], axis=1)

    def fit(
        self,
        data_traindev: pd.DataFrame,
        data_test: pd.DataFrame,
        feature_cols: list[str],
        study: StudyContext,
        outersplit_id: int,
        output_dir: UPath,
        num_assigned_cpus: int = 1,
        feature_groups: dict | None = None,
        prior_results: dict | None = None,
    ) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fit AutoGluon TabularPredictor."""
        # Store temporary execution state (available during fit)
        self._study = study
        self._output_dir = output_dir
        self._feature_groups = feature_groups or {}
        self._outersplit_id = outersplit_id
        self._feature_cols = feature_cols

        target_cols = list(study.target_assignments.values())
        row_id_col = study.row_id_col

        self._x_traindev = data_traindev[feature_cols]
        self._y_traindev = data_traindev[target_cols]
        self._x_test = data_test[feature_cols]
        self._y_test = data_test[target_cols]
        self._row_traindev = data_traindev[row_id_col]
        self._row_test = data_test[row_id_col]

        # Set up logging and resources
        logger.set_log_group(LogGroup.AUTOGLUON, f"OUTER {self._outersplit_id}")

        # Allocate CPUs
        if self.config.num_cpus == "auto":
            self._num_cpus_allocated = num_assigned_cpus
        else:
            self._num_cpus_allocated = min(num_assigned_cpus, self.config.num_cpus)

        logger.info(
            f"CPU Resources | Available: {num_assigned_cpus} | "
            f"Requested: {self.config.num_cpus} | "
            f"Allocated: {self._num_cpus_allocated}"
        )

        # Ensure AutoGluon uses existing Ray instance if available
        setup_ray_for_external_library()

        # Get target column
        if len(self.target_assignments) == 1:
            target = next(iter(self.target_assignments.values()))
        else:
            raise ValueError(f"Single target expected. Got keys: {self.target_assignments.keys()}")

        # Get scoring metric
        assert self.target_metric is not None, "target_metric should be set during fit()"
        scoring_type = metrics_inventory_autogluon[self.target_metric]

        # Initialize TabularPredictor (store temporarily for fit operations)
        _predictor = TabularPredictor(
            label=target,
            eval_metric=scoring_type,
            verbosity=self.config.verbosity,
            log_to_file=False,
        )

        # Fit predictor
        _predictor.fit(
            self.ag_train_data,
            time_limit=self.config.time_limit,
            infer_limit=self.config.infer_limit,
            memory_limit=self.config.memory_limit,
            presets=self.config.presets,
            fit_strategy=self.config.fit_strategy,
            num_bag_folds=self.config.num_bag_folds,
            included_model_types=self.config.included_model_types,
        )

        logger.set_log_group(LogGroup.AUTOGLUON, f"OUTER {self._outersplit_id}")
        logger.info("Fitting completed")

        # Create results directory and save outputs
        assert self.path_results is not None, "path_results should be set during fit()"
        self.path_results.mkdir(parents=True, exist_ok=True)

        # Save failure info
        with (self.path_results / "debug_info.txt").open("w", encoding="utf-8") as text_file:
            print(_predictor.model_failures(), file=text_file)

        # Temporarily store predictor for helper methods
        self.model_ = _predictor

        # Save leaderboard and model info
        self._save_leaderboard_info()

        # Get raw results (uses self.model_ internally)
        raw_scores = self._get_scores()
        raw_predictions = self._get_predictions()
        raw_fi = self._get_feature_importances()

        # AutoGluon doesn't do feature selection, so return all features
        selected_features = feature_cols
        self.selected_features_ = selected_features

        # Store fitted state - wrap with sklearn-compatible model
        self.feature_importances_ = raw_fi
        self.model_ = self._get_sklearn_model()

        # Build flat scores DataFrame
        scores_rows = []
        for key, value in raw_scores.items():
            if isinstance(value, (int, float)):
                scores_rows.append(
                    {
                        "result_type": ResultType.BEST,
                        "metric": key,
                        "partition": "combined",
                        "aggregation": "single",
                        "fold": None,
                        "value": value,
                    }
                )
        scores = pd.DataFrame(scores_rows)

        # Build flat predictions DataFrame
        pred_dfs = []
        for _part_name, df in raw_predictions.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                temp = df.copy()
                temp["result_type"] = ResultType.BEST
                pred_dfs.append(temp)
        predictions = pd.concat(pred_dfs, ignore_index=True) if pred_dfs else pd.DataFrame()

        # Build flat feature_importances DataFrame
        fi_dfs = []
        for _fi_key, fi_df in raw_fi.items():
            if isinstance(fi_df, pd.DataFrame) and not fi_df.empty:
                temp = fi_df.reset_index() if fi_df.index.name is not None else fi_df.copy()
                if "feature" not in temp.columns and temp.index.name is None:
                    temp = temp.reset_index()
                    temp.columns = ["feature", "importance", *list(temp.columns[2:])]
                temp = (
                    temp[["feature", "importance"]].copy()
                    if "feature" in temp.columns and "importance" in temp.columns
                    else temp
                )
                temp["fi_method"] = FIMethod.PERMUTATION
                temp["fi_dataset"] = FIDataset.TEST
                temp["training_id"] = "autogluon"
                temp["result_type"] = ResultType.BEST
                fi_dfs.append(temp)
        feature_importances = pd.concat(fi_dfs, ignore_index=True) if fi_dfs else pd.DataFrame()

        return (selected_features, scores, predictions, feature_importances)

    def _get_sklearn_model(self):
        """Get sklearn-compatible wrapper for the AutoGluon model."""
        if self.ml_type == "classification":
            return SklearnClassifier(self.model_)
        elif self.ml_type == "regression":
            return SklearnRegressor(self.model_)
        else:
            raise ValueError(f"ML type {self.ml_type} not supported")

    def _get_feature_importances(self):
        """Calculate feature importances using AutoGluon's permutation importance."""
        logger.set_log_group(LogGroup.AUTOGLUON, f"OUTER {self._outersplit_id}")
        logger.info("Calculating test permutation feature importances...")

        fi = {}
        np.random.seed(42)

        # AutoGluon permutation feature importances
        fi["autogluon_permutation_test"] = self.model_.feature_importance(
            data=self.ag_test_data,
            subsample_size=5000,
            time_limit=None,
            include_confidence_band=True,
            confidence_level=0.95,
            num_shuffle_sets=15,
            silent=True,
        )

        # Calculate group feature importances if feature_groups provided
        if self.feature_groups:
            group_importances = {}
            for group_name, features in self.feature_groups.items():
                group_importance = self.model_.feature_importance(
                    data=self.ag_test_data,
                    features=[(group_name, features)],
                    subsample_size=5000,
                    time_limit=None,
                    include_confidence_band=True,
                    confidence_level=0.95,
                    num_shuffle_sets=15,
                    silent=True,
                )
                group_importances[group_name] = group_importance

            # Combine feature and group importances
            combined_feature_importances = [fi["autogluon_permutation_test"]]

            for group_name, importance in group_importances.items():
                group_row = importance.copy()
                group_row.index = [f"{group_name}"] * len(group_row)
                combined_feature_importances.append(group_row)

            fi["autogluon_permutation_test"] = pd.concat(combined_feature_importances)

        fi["autogluon_permutation_test"] = fi["autogluon_permutation_test"].sort_values(
            by="importance", ascending=False
        )

        # Save combined importances
        combined_importances = {
            "autogluon_permutation": fi["autogluon_permutation_test"].to_dict(orient="index"),
        }

        assert self.path_results is not None, "path_results should be set during fit()"
        with (self.path_results / "combined_feature_importances.json").open("w", encoding="utf-8") as f:
            json.dump(combined_importances, f, indent=4)

        return fi

    def _get_scores(self):
        """Calculate performance scores on train/dev/test sets."""
        # Test performance
        test_performance = self.model_.evaluate(self.ag_test_data, detailed_report=True, silent=True)
        test_performance_with_suffix = {f"{key}_test": value for key, value in test_performance.items()}

        # Dev performance from leaderboard
        leaderboard = self.model_.leaderboard(silent=True)
        best_model_info = leaderboard.iloc[0].to_dict()
        dev_performance = {key: value for key, value in best_model_info.items() if "val" in key or "score" in key}
        dev_performance_with_suffix = {f"{key}_dev": value for key, value in dev_performance.items()}

        # Train performance
        train_performance = self.model_.evaluate(self.ag_train_data, detailed_report=True, silent=True)
        train_performance_with_suffix = {f"{key}_train": value for key, value in train_performance.items()}

        # Test scores using Octopus metrics for comparison
        assert self.target_metric is not None, "target_metric should be set during fit()"
        all_metrics = list(dict.fromkeys([*self.metrics, self.target_metric]))
        test_performance_octo = {}
        for metric in all_metrics:
            assert self._feature_cols is not None, "feature_cols should be set during fit()"
            assert metric is not None, "metric should not be None"
            performance = get_score_from_model(
                self.model_,
                self.ag_test_data,
                self._feature_cols,
                metric,
                self.target_assignments,
                positive_class=self.positive_class,
            )
            test_performance_octo[metric + "_test_octo"] = performance

        # Combine all performance metrics
        combined_performance = {
            **dev_performance_with_suffix,
            **train_performance_with_suffix,
            **test_performance_with_suffix,
            **test_performance_octo,
        }

        # Save performance results
        if isinstance(combined_performance, dict):
            for key, value in combined_performance.items():
                if isinstance(value, pd.DataFrame):
                    combined_performance[key] = value.to_dict(orient="records")

        assert self.path_results is not None, "path_results should be set during fit()"
        with (self.path_results / "performance_results.json").open("w", encoding="utf-8") as f:
            json.dump(combined_performance, f, indent=4)

        return combined_performance

    def _save_leaderboard_info(self):
        """Save AutoGluon leaderboard and model information."""
        assert self.path_results is not None, "path_results should be set during fit()"

        # Save leaderboard
        leaderboard = self.model_.leaderboard(self.ag_test_data, extra_info=True)
        leaderboard_path = self.path_results / "leaderboard.csv"
        leaderboard.to_csv(
            str(leaderboard_path),
            storage_options=dict(leaderboard_path.storage_options),
        )

        # Save best model results
        best_model = leaderboard.iloc[0]
        best_result_df = pd.DataFrame(best_model).transpose()
        best_result_path = self.path_results / "best_model_result.csv"
        best_result_df.to_csv(
            str(best_result_path),
            storage_options=dict(best_result_path.storage_options),
        )

        # Save model info
        model_info = self.model_.info()
        with (self.path_results / "model_info.json").open("w", encoding="utf-8") as f:
            json.dump(model_info, f, default=str, indent=4)

        # Save fit summary
        fit_summary = self.model_.fit_summary()
        with (self.path_results / "model_stats.txt").open("w", encoding="utf-8") as text_file:
            print(fit_summary, file=text_file)

    def _get_predictions(self):
        """Get out-of-fold and test predictions with metadata."""
        predictions = {}
        best_model_name = self.model_.model_best
        problem_type = self.model_.problem_type
        row_column = self.row_column

        # Metadata for all predictions (AutoGluon doesn't use inner splits)
        outersplit_id = self._outersplit_id
        task_id = self.config.task_id
        inner_split_id = "autogluon"  # AutoGluon doesn't have inner CV splits

        # Test predictions
        rowid_test = pd.DataFrame({row_column: self.row_test})

        if problem_type == "regression":
            test_pred_data = self.model_.predict(self.ag_test_data)
            test_pred = pd.DataFrame({"prediction": test_pred_data})
        elif problem_type in ["binary", "multiclass"]:
            test_pred = self.model_.predict_proba(self.ag_test_data)
            class_labels = self.model_.class_labels
            test_pred.columns = class_labels
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")

        assert len(rowid_test) == len(test_pred), "Mismatch in number of test rows!"
        test_df = pd.concat(
            [rowid_test.reset_index(drop=True), test_pred.reset_index(drop=True)],
            axis=1,
        )
        # Add metadata
        test_df["outersplit_id"] = outersplit_id
        test_df["inner_split_id"] = inner_split_id
        test_df["partition"] = "test"
        test_df["task_id"] = task_id
        predictions["test"] = test_df

        # Out-of-fold validation predictions
        rowid_dev = pd.DataFrame({row_column: self.row_traindev})

        if problem_type == "regression":
            oof_pred_data = self.model_.predict_oof(model=best_model_name)
            oof_pred = pd.DataFrame({"prediction": oof_pred_data})
        elif problem_type in ["binary", "multiclass"]:
            oof_pred = self.model_.predict_proba_oof(model=best_model_name)
            class_labels = self.model_.class_labels
            oof_pred.columns = class_labels
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")

        assert len(rowid_dev) == len(oof_pred), "Mismatch in number of dev rows!"
        dev_df = pd.concat(
            [rowid_dev.reset_index(drop=True), oof_pred.reset_index(drop=True)],
            axis=1,
        )
        # Add metadata
        dev_df["outersplit_id"] = outersplit_id
        dev_df["inner_split_id"] = inner_split_id
        dev_df["partition"] = "dev"
        dev_df["task_id"] = task_id
        predictions["dev"] = dev_df

        return predictions
