"""AutoGluon execution module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from attrs import define
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
from octopus.metrics.utils import get_score_from_model
from octopus.modules import FIDataset, FIMethod, ModuleExecution, ModuleResult, ResultType, StudyContext
from octopus.types import MLType

if TYPE_CHECKING:
    from octopus.modules import (
        AutoGluon,  # noqa: F401
        StudyContext,
    )

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
class AutoGluonModule(ModuleExecution["AutoGluon"]):
    """AutoGluon execution module. Created by AutoGluon.create_module()."""

    def fit(
        self,
        *,
        data_traindev: pd.DataFrame,
        data_test: pd.DataFrame,
        feature_cols: list[str],
        study_context: StudyContext,
        outersplit_id: int,
        results_dir: UPath,
        num_assigned_cpus: int,
        feature_groups: dict[str, list[str]],
        **kwargs,
    ) -> dict[ResultType, ModuleResult]:
        """Fit AutoGluon TabularPredictor."""
        target_cols = list(study_context.target_assignments.values())
        row_id_col = study_context.row_id_col

        x_traindev = data_traindev[feature_cols]
        y_traindev = data_traindev[target_cols]
        x_test = data_test[feature_cols]
        y_test = data_test[target_cols]
        row_traindev = data_traindev[row_id_col]
        row_test = data_test[row_id_col]

        ag_train_data = pd.concat([x_traindev, y_traindev], axis=1)
        ag_test_data = pd.concat([x_test, y_test], axis=1)

        # Set up logging and resources
        logger.set_log_group(LogGroup.AUTOGLUON, f"OUTER {outersplit_id}")

        # Allocate CPUs
        if self.config.num_cpus == "auto":
            num_cpus_allocated = num_assigned_cpus
        else:
            num_cpus_allocated = min(num_assigned_cpus, self.config.num_cpus)

        logger.info(
            f"CPU Resources | Available: {num_assigned_cpus} | Requested: {self.config.num_cpus} | Allocated: {num_cpus_allocated}"
        )

        # Ensure AutoGluon uses existing Ray instance if available
        setup_ray_for_external_library()

        # Get target column
        if len(study_context.target_assignments) == 1:
            target = next(iter(study_context.target_assignments.values()))
        else:
            raise ValueError(f"Single target expected. Got keys: {study_context.target_assignments.keys()}")

        # Get scoring metric
        assert study_context.target_metric is not None, "target_metric should be set during fit()"
        scoring_type = metrics_inventory_autogluon[study_context.target_metric]

        # Initialize TabularPredictor (store temporarily for fit operations)
        predictor = TabularPredictor(
            label=target,
            eval_metric=scoring_type,
            verbosity=self.config.verbosity,
            log_to_file=False,
        )

        # Fit predictor
        predictor.fit(
            ag_train_data,
            time_limit=self.config.time_limit,
            infer_limit=self.config.infer_limit,
            memory_limit=self.config.memory_limit,
            presets=self.config.presets,
            fit_strategy=self.config.fit_strategy,
            num_bag_folds=self.config.num_bag_folds,
            included_model_types=self.config.included_model_types,
        )

        logger.set_log_group(LogGroup.AUTOGLUON, f"OUTER {outersplit_id}")
        logger.info("Fitting completed")

        # Save failure info
        with (results_dir / "autogluon_debug_info.txt").open("w", encoding="utf-8") as text_file:
            print(predictor.model_failures(), file=text_file)

        # Save leaderboard and model info
        self._save_leaderboard_info(predictor, ag_test_data, outersplit_id, results_dir)

        # Get raw results
        raw_scores = self._get_scores(predictor, study_context, ag_train_data, ag_test_data, feature_cols, results_dir)
        raw_predictions = self._get_predictions(
            predictor, study_context, ag_test_data, row_test, row_traindev, outersplit_id
        )
        raw_fi = self._get_feature_importances(predictor, ag_test_data, outersplit_id, feature_groups, results_dir)

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

        return {
            ResultType.BEST: ModuleResult(
                result_type=ResultType.BEST,
                module=self.config.module,
                selected_features=feature_cols,  # AutoGluon doesn't do feature selection, so return all features
                scores=scores,
                predictions=predictions,
                feature_importances=feature_importances,
                model=self._get_sklearn_model(predictor, study_context),
            )
        }

    def _get_sklearn_model(self, model: TabularPredictor, study_context: StudyContext) -> BaseEstimator:
        """Get sklearn-compatible wrapper for the AutoGluon model."""
        if study_context.ml_type in (MLType.BINARY, MLType.MULTICLASS):
            return SklearnClassifier(model)
        elif study_context.ml_type == MLType.REGRESSION:
            return SklearnRegressor(model)
        else:
            raise ValueError(f"ML type {study_context.ml_type} not supported")

    def _get_feature_importances(
        self,
        model: TabularPredictor,
        ag_test_data: pd.DataFrame,
        outersplit_id: int,
        feature_groups: dict[str, list[str]],
        results_dir: UPath,
    ) -> dict[str, pd.DataFrame]:
        """Calculate feature importances using AutoGluon's permutation importance."""
        logger.set_log_group(LogGroup.AUTOGLUON, f"OUTER {outersplit_id}")
        logger.info("Calculating test permutation feature importances...")

        fi: dict[str, pd.DataFrame] = {}
        np.random.seed(42)

        # AutoGluon permutation feature importances
        fi["autogluon_permutation_test"] = model.feature_importance(
            data=ag_test_data,
            subsample_size=5000,
            time_limit=None,
            include_confidence_band=True,
            confidence_level=0.95,
            num_shuffle_sets=15,
            silent=True,
        )

        # Calculate group feature importances if feature_groups provided
        if feature_groups:
            group_importances = {}
            for group_name, features in feature_groups.items():
                group_importance = model.feature_importance(
                    data=ag_test_data,
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

        with (results_dir / "combined_feature_importances.json").open("w", encoding="utf-8") as f:
            json.dump(combined_importances, f, indent=4)

        return fi

    def _get_scores(
        self,
        model: TabularPredictor,
        study_context: StudyContext,
        ag_train_data: pd.DataFrame,
        ag_test_data: pd.DataFrame,
        feature_cols: list[str],
        results_dir: UPath,
    ) -> dict[str, Any]:
        """Calculate performance scores on train/dev/test sets."""
        # Test performance
        test_performance = model.evaluate(ag_test_data, detailed_report=True, silent=True)
        test_performance_with_suffix = {f"{key}_test": value for key, value in test_performance.items()}

        # Dev performance from leaderboard
        leaderboard = model.leaderboard(silent=True)
        best_model_info = leaderboard.iloc[0].to_dict()
        dev_performance = {key: value for key, value in best_model_info.items() if "val" in key or "score" in key}
        dev_performance_with_suffix = {f"{key}_dev": value for key, value in dev_performance.items()}

        # Train performance
        train_performance = model.evaluate(ag_train_data, detailed_report=True, silent=True)
        train_performance_with_suffix = {f"{key}_train": value for key, value in train_performance.items()}

        # Test scores using Octopus metrics for comparison
        assert study_context.target_metric is not None, "target_metric should be set during fit()"
        all_metrics = list(dict.fromkeys([*study_context.metrics, study_context.target_metric]))
        test_performance_octo = {}
        for metric in all_metrics:
            assert feature_cols is not None, "feature_cols should be set during fit()"
            assert metric is not None, "metric should not be None"
            performance = get_score_from_model(
                model,
                ag_test_data,
                feature_cols,
                metric,
                study_context.target_assignments,
                positive_class=study_context.positive_class,
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

        with (results_dir / "performance_results.json").open("w", encoding="utf-8") as f:
            json.dump(combined_performance, f, indent=4)

        return combined_performance

    def _save_leaderboard_info(
        self,
        model: TabularPredictor,
        ag_test_data: pd.DataFrame,
        outersplit_id: int,
        results_dir: UPath,
    ) -> None:
        """Save AutoGluon leaderboard and model information."""
        # Save leaderboard
        leaderboard = model.leaderboard(ag_test_data, extra_info=True)
        leaderboard_path = results_dir / "leaderboard.csv"
        leaderboard.to_csv(
            str(leaderboard_path),
            storage_options=dict(leaderboard_path.storage_options),
        )

        # Save best model results
        best_model = leaderboard.iloc[0]
        best_result_df = pd.DataFrame(best_model).transpose()
        best_result_path = results_dir / "best_model_result.csv"
        best_result_df.to_csv(
            str(best_result_path),
            storage_options=dict(best_result_path.storage_options),
        )

        # Save model info
        model_info = model.info()
        with (results_dir / "model_info.json").open("w", encoding="utf-8") as f:
            json.dump(model_info, f, default=str, indent=4)

        # Save fit summary
        fit_summary = model.fit_summary()
        with (results_dir / "model_stats.txt").open("w", encoding="utf-8") as text_file:
            print(fit_summary, file=text_file)

    def _get_predictions(
        self,
        model: TabularPredictor,
        study_context: StudyContext,
        ag_test_data: pd.DataFrame,
        row_test: pd.Series,
        row_traindev: pd.Series,
        outersplit_id: int,
    ) -> dict[str, pd.DataFrame]:
        """Get out-of-fold and test predictions with metadata."""
        predictions = {}
        best_model_name = model.model_best
        problem_type = model.problem_type
        row_column = study_context.row_id_col

        # Metadata for all predictions (AutoGluon doesn't use inner splits)
        task_id = self.config.task_id
        inner_split_id = "autogluon"  # AutoGluon doesn't have inner CV splits

        # Test predictions
        rowid_test = pd.DataFrame({row_column: row_test})

        if problem_type == "regression":
            test_pred_data = model.predict(ag_test_data)
            test_pred = pd.DataFrame({"prediction": test_pred_data})
        elif problem_type in ["binary", "multiclass"]:
            test_pred = model.predict_proba(ag_test_data)
            class_labels = model.class_labels
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
        rowid_dev = pd.DataFrame({row_column: row_traindev})

        if problem_type == "regression":
            oof_pred_data = model.predict_oof(model=best_model_name)
            oof_pred = pd.DataFrame({"prediction": oof_pred_data})
        elif problem_type in ["binary", "multiclass"]:
            oof_pred = model.predict_proba_oof(model=best_model_name)
            class_labels = model.class_labels
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
