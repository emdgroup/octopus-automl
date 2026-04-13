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
    mean_squared_error,
    precision,
    r2,
    recall,
    roc_auc,
    roc_auc_ovr,
    roc_auc_ovr_weighted,
    root_mean_squared_error,
)
from octopus.logger import get_logger
from octopus.metrics import Metrics
from octopus.metrics.utils import get_performance_from_model
from octopus.modules import ModuleExecution, ModuleResult, StudyContext
from octopus.types import DataPartition, FIResultLabel, LogGroup, MLType, PredictionType, ResultType
from octopus.utils import csv_save

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
    "ACC": accuracy,
    "ACCBAL": balanced_accuracy,
    "ACCBAL_MC": balanced_accuracy,
    "AUCPR": average_precision,
    "AUCROC": roc_auc,
    "AUCROC_MACRO": roc_auc_ovr,
    "AUCROC_WEIGHTED": roc_auc_ovr_weighted,
    "F1": f1,
    "LOGLOSS": log_loss,
    "MAE": mean_absolute_error,
    "MCC": mcc,
    "MSE": mean_squared_error,
    "NEGBRIERSCORE": "brier_score_loss",
    "PRECISION": precision,
    "R2": r2,
    "RECALL": recall,
    "RMSE": root_mean_squared_error,
}

_AG_INNER_SPLIT_ID = "autogluon"


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
        outer_split_id: int,
        results_dir: UPath,
        scratch_dir: UPath,
        n_assigned_cpus: int,
        feature_groups: dict[str, list[str]],
        prior_results: dict[str, pd.DataFrame],
        **kwargs,
    ) -> dict[ResultType, ModuleResult]:
        """Fit AutoGluon TabularPredictor."""
        if study_context.ml_type == MLType.TIMETOEVENT:
            raise ValueError(
                "AutoGluon does not support time-to-event tasks. "
                "Use the Octo module with CatBoostCoxSurvival or XGBoostCoxSurvival models."
            )

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
        logger.set_log_group(LogGroup.AUTOGLUON, f"OUTER {outer_split_id}")

        # Get target column
        if len(study_context.target_assignments) == 1:
            target = next(iter(study_context.target_assignments.values()))
        else:
            raise ValueError(f"Single target expected. Got keys: {study_context.target_assignments.keys()}")

        # Get scoring metric
        if study_context.target_metric is None:
            raise ValueError("target_metric should be set during fit()")

        scoring_type = metrics_inventory_autogluon[study_context.target_metric]

        # Initialize TabularPredictor (store temporarily for fit operations)
        predictor = TabularPredictor(
            label=target,
            eval_metric=scoring_type,
            verbosity=0,
            log_to_file=False,
            learner_kwargs={"cache_data": True},
        )

        # Log configuration summary
        logger.info(
            "Starting fit: presets=%s, time_limit=%s, model_types=%s",
            self.config.presets,
            self.config.time_limit,
            self.config.included_model_types or "all",
        )

        # Fit predictor
        predictor.fit(
            ag_train_data,
            time_limit=self.config.time_limit,
            infer_limit=self.config.infer_limit,
            memory_limit=self.config.memory_limit,
            presets=self.config.presets,
            fit_strategy="sequential",
            num_bag_folds=self.config.n_bag_splits,
            included_model_types=self.config.included_model_types,
            num_cpus=n_assigned_cpus,
        )

        logger.set_log_group(LogGroup.AUTOGLUON, f"OUTER {outer_split_id}")
        logger.info("Fitting completed")

        # Log best model summary
        leaderboard_summary = predictor.leaderboard(silent=True)
        best = leaderboard_summary.iloc[0]
        logger.info("Best model: %s (score_val=%.4f)", best["model"], best["score_val"])

        # Save failure info
        with (results_dir / "autogluon_debug_info.txt").open("w", encoding="utf-8") as text_file:
            print(predictor.model_failures(), file=text_file)

        # Save leaderboard and model info
        self._save_leaderboard_info(predictor, outer_split_id, results_dir)

        # Get raw results (predictions first — needed for dev scores)
        raw_predictions = self._get_predictions(
            predictor, study_context, ag_test_data, row_test, row_traindev, outer_split_id
        )
        scores = self._get_scores(
            predictor,
            study_context,
            ag_train_data,
            ag_test_data,
            feature_cols,
            raw_predictions,
            data_traindev,
            data_test,
            results_dir,
        )
        raw_fi = self._get_fi(predictor, ag_train_data, outer_split_id, feature_cols, feature_groups, results_dir)

        # Build flat predictions DataFrame
        pred_dfs = []
        for _part_name, df in raw_predictions.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                temp = df.copy()
                temp["result_type"] = ResultType.BEST
                pred_dfs.append(temp)
        predictions = pd.concat(pred_dfs, ignore_index=True) if pred_dfs else pd.DataFrame()

        # Build flat feature importance DataFrame
        fi_dfs = []
        for _fi_key, fi_df in raw_fi.items():
            if isinstance(fi_df, pd.DataFrame) and not fi_df.empty:
                temp = fi_df[["importance"]].reset_index()
                temp.columns = ["feature", "importance"]
                temp["fi_method"] = FIResultLabel.PERMUTATION
                temp["fi_dataset"] = DataPartition.DEV
                temp["training_id"] = "mean"
                temp["result_type"] = ResultType.BEST
                fi_dfs.append(temp)
        fi_df = pd.concat(fi_dfs, ignore_index=True) if fi_dfs else pd.DataFrame()

        return {
            ResultType.BEST: ModuleResult(
                result_type=ResultType.BEST,
                module=self.config.module,
                selected_features=feature_cols,  # AutoGluon doesn't do feature selection, so return all features
                scores=scores,
                predictions=predictions,
                fi=fi_df,
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

    def _get_fi(
        self,
        model: TabularPredictor,
        ag_train_data: pd.DataFrame,
        outer_split_id: int,
        feature_cols: list[str],
        feature_groups: dict[str, list[str]],
        results_dir: UPath,
    ) -> dict[str, pd.DataFrame]:
        """Calculate feature importances using OOF or traindev permutation importance.

        Attempts OOF (out-of-fold) FI via the best level-1 bagged model,
        which provides truly unbiased importance scores on original features.
        Falls back to permutation FI on traindev data if OOF is unavailable.

        Args:
            model: Fitted AutoGluon TabularPredictor.
            ag_train_data: Training data (traindev split) for fallback FI.
            outer_split_id: Outer split index for logging.
            feature_cols: Original feature column names for safety check.
            feature_groups: Feature groups (logged as unsupported, not used).
            results_dir: Directory to save diagnostic output.

        Returns:
            Dict with single key mapping to the FI DataFrame.
        """
        logger.set_log_group(LogGroup.AUTOGLUON, f"OUTER {outer_split_id}")

        if feature_groups:
            logger.info(
                "Group feature importances not supported for AutoGluon — "
                "only individual feature importances will be computed."
            )

        np.random.seed(42)

        fi_df = self._compute_oof_fi(model, feature_cols)

        if fi_df is None:
            logger.info("Computing permutation FI on traindev data (fallback)...")
            fi_df = model.feature_importance(
                data=ag_train_data,
                feature_stage="original",
                subsample_size=5000,
                time_limit=None,
                include_confidence_band=True,
                confidence_level=0.95,
                num_shuffle_sets=15,
                silent=True,
            )

        fi_df = fi_df.sort_values(by="importance", ascending=False)

        with (results_dir / "feature_importances_raw.json").open("w", encoding="utf-8") as f:
            json.dump(fi_df.to_dict(orient="index"), f, indent=4)

        return {"autogluon_permutation_dev": fi_df}

    def _compute_oof_fi(
        self,
        model: TabularPredictor,
        feature_cols: list[str],
    ) -> pd.DataFrame | None:
        """Compute OOF feature importance via the best level-1 bagged model.

        Returns None if OOF FI is unavailable (no L1 models, feature mismatch,
        or any error), signalling the caller to use the traindev fallback.

        Args:
            model: Fitted AutoGluon TabularPredictor (must have cache_data=True).
            feature_cols: Original feature column names for validation.

        Returns:
            FI DataFrame indexed by feature name, or None on failure.
        """
        l1_models = model.model_names(level=1)
        if not l1_models:
            logger.info("No level-1 models found — cannot compute OOF FI.")
            return None

        leaderboard = model.leaderboard(silent=True)
        l1_leaderboard = leaderboard[leaderboard["model"].isin(l1_models)]
        if l1_leaderboard.empty:
            logger.info("No level-1 models in leaderboard — cannot compute OOF FI.")
            return None

        best_l1 = l1_leaderboard.iloc[0]["model"]
        logger.info("Computing OOF permutation FI via L1 model: %s", best_l1)

        try:
            fi_df: pd.DataFrame = model.feature_importance(
                data=None,
                model=best_l1,
                feature_stage="transformed_model",
                subsample_size=5000,
                time_limit=None,
                include_confidence_band=True,
                confidence_level=0.95,
                num_shuffle_sets=15,
                silent=True,
            )
        except Exception:
            logger.warning("OOF FI failed — falling back to traindev FI", exc_info=True)
            return None

        fi_features = set(fi_df.index)
        original_features = set(feature_cols)
        if not fi_features <= original_features:
            unexpected = fi_features - original_features
            logger.warning(
                "OOF FI returned unexpected features %s — falling back to traindev FI",
                unexpected,
            )
            return None

        return fi_df

    def _get_scores(
        self,
        model: TabularPredictor,
        study_context: StudyContext,
        ag_train_data: pd.DataFrame,
        ag_test_data: pd.DataFrame,
        feature_cols: list[str],
        raw_predictions: dict[str, pd.DataFrame],
        data_traindev: pd.DataFrame,
        data_test: pd.DataFrame,
        results_dir: UPath,
    ) -> pd.DataFrame:
        """Calculate performance scores on dev and test sets using octopus metrics.

        Dev scores are computed from OOF predictions joined with traindev targets.
        Test scores are computed via model prediction on test data.
        AG-native metrics are saved separately for diagnostics.

        Args:
            model: Fitted AutoGluon TabularPredictor.
            study_context: Study configuration and metadata.
            ag_train_data: Training data (features + target) for AG diagnostics.
            ag_test_data: Test data (features + target) for AG diagnostics.
            feature_cols: Feature column names.
            raw_predictions: Dict with "dev" and "test" prediction DataFrames.
            data_traindev: Original traindev data (with row_id and target cols).
            data_test: Original test data (with row_id and target cols).
            results_dir: Directory to save diagnostic output.

        Returns:
            Scores DataFrame with columns: result_type, metric, partition,
            aggregation, split, value.
        """
        self._save_ag_diagnostics(model, ag_train_data, ag_test_data, results_dir)

        all_metric_names = Metrics.get_by_type(study_context.ml_type)
        target_col = list(study_context.target_assignments.values())[0]
        row_id_col = study_context.row_id_col

        dev_pred = raw_predictions["dev"]
        dev_with_target = dev_pred.merge(data_traindev[[row_id_col, target_col]], on=row_id_col)

        rows = []
        for metric_name in all_metric_names:
            dev_value = self._compute_metric_from_predictions(
                metric_name,
                dev_with_target,
                target_col,
                study_context,
            )
            rows.append(
                {
                    "result_type": ResultType.BEST,
                    "metric": metric_name,
                    "partition": DataPartition.DEV,
                    "aggregation": "ensemble",
                    "split": None,
                    "value": dev_value,
                }
            )

            test_value = get_performance_from_model(
                model,
                ag_test_data,
                feature_cols,
                metric_name,
                study_context.target_assignments,
                positive_class=study_context.positive_class,
            )
            rows.append(
                {
                    "result_type": ResultType.BEST,
                    "metric": metric_name,
                    "partition": DataPartition.TEST,
                    "aggregation": "ensemble",
                    "split": None,
                    "value": test_value,
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _compute_metric_from_predictions(
        metric_name: str,
        pred_df: pd.DataFrame,
        target_col: str,
        study_context: StudyContext,
    ) -> float:
        """Compute a single octopus metric from a predictions DataFrame.

        Args:
            metric_name: Registered metric name (e.g. "ACCBAL").
            pred_df: DataFrame with target column and prediction/probability columns.
            target_col: Name of the target column.
            study_context: Study context for positive_class and ml_type.

        Returns:
            Metric value as float.
        """
        metric = Metrics.get_instance(metric_name)
        target = pred_df[target_col]
        positive_class = study_context.positive_class

        if positive_class is not None and metric.supports_ml_type(MLType.BINARY):
            probabilities = pred_df[positive_class]
            if metric.prediction_type == PredictionType.PROBABILITIES:
                return metric.calculate(target, probabilities)
            predictions = (probabilities >= 0.5).astype(int)
            return metric.calculate(target, predictions)

        if metric.supports_ml_type(MLType.MULTICLASS) and positive_class is None:
            if metric.prediction_type == PredictionType.PROBABILITIES:
                prob_columns = sorted(c for c in pred_df.columns if isinstance(c, int))
                prob_matrix = pred_df[prob_columns].to_numpy()
                return metric.calculate(target, prob_matrix)
            predictions = pred_df["prediction"].astype(int)
            return metric.calculate(target, predictions)

        if metric.supports_ml_type(MLType.REGRESSION):
            predictions = pred_df["prediction"]
            return metric.calculate(target, predictions)

        raise ValueError(f"Cannot compute metric '{metric_name}' for ml_type={study_context.ml_type}")

    @staticmethod
    def _save_ag_diagnostics(
        model: TabularPredictor,
        ag_train_data: pd.DataFrame,
        ag_test_data: pd.DataFrame,
        results_dir: UPath,
    ) -> None:
        """Save AG-native performance metrics for diagnostics only."""
        test_performance = model.evaluate(ag_test_data, detailed_report=True, silent=True)
        leaderboard = model.leaderboard(silent=True)
        best_model_info = leaderboard.iloc[0].to_dict()
        train_performance = model.evaluate(ag_train_data, detailed_report=True, silent=True)

        diagnostics: dict[str, Any] = {
            "ag_test": {k: v for k, v in test_performance.items() if isinstance(v, (int, float))},
            "ag_train": {k: v for k, v in train_performance.items() if isinstance(v, (int, float))},
            "ag_best_model": {k: v for k, v in best_model_info.items() if isinstance(v, (int, float, str))},
        }

        with (results_dir / "performance_results.json").open("w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=4, default=str)

    def _save_leaderboard_info(
        self,
        model: TabularPredictor,
        outer_split_id: int,
        results_dir: UPath,
    ) -> None:
        """Save AutoGluon leaderboard and model information."""
        # Save leaderboard (without test data — diagnostic only, uses AG internal val scores)
        leaderboard = model.leaderboard(extra_info=True, silent=True)
        leaderboard_path = results_dir / "leaderboard.csv"
        csv_save(leaderboard, leaderboard_path)

        # Save best model results
        best_model = leaderboard.iloc[0]
        best_result_df = pd.DataFrame(best_model).transpose()
        best_result_path = results_dir / "best_model_result.csv"
        csv_save(best_result_df, best_result_path)

        # Save model info
        model_info = model.info()
        with (results_dir / "model_info.json").open("w", encoding="utf-8") as f:
            json.dump(model_info, f, default=str, indent=4)

        # Save fit summary
        fit_summary = model.fit_summary(verbosity=0)
        with (results_dir / "model_stats.txt").open("w", encoding="utf-8") as text_file:
            print(fit_summary, file=text_file)

    def _get_predictions(
        self,
        model: TabularPredictor,
        study_context: StudyContext,
        ag_test_data: pd.DataFrame,
        row_test: pd.Series,
        row_traindev: pd.Series,
        outer_split_id: int,
    ) -> dict[str, pd.DataFrame]:
        """Get out-of-split and test predictions with metadata."""
        predictions = {}
        best_model_name = model.model_best
        problem_type = model.problem_type
        row_column = study_context.row_id_col

        # Metadata for all predictions (AutoGluon doesn't use inner splits)
        task_id = self.config.task_id
        inner_split_id = _AG_INNER_SPLIT_ID

        # Test predictions
        rowid_test = pd.DataFrame({row_column: row_test})

        if problem_type == "regression":
            test_pred_data = model.predict(ag_test_data)
            test_pred = pd.DataFrame({"prediction": test_pred_data})
        elif problem_type in ["binary", "multiclass"]:
            test_pred = model.predict_proba(ag_test_data)
            class_labels = model.class_labels
            test_pred.columns = class_labels
            test_pred["prediction"] = test_pred[class_labels].values.argmax(axis=1)
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")

        assert len(rowid_test) == len(test_pred), "Mismatch in number of test rows!"
        test_df = pd.concat(
            [rowid_test.reset_index(drop=True), test_pred.reset_index(drop=True)],
            axis=1,
        )
        # Add metadata
        test_df["outer_split_id"] = outer_split_id
        test_df["inner_split_id"] = inner_split_id
        test_df["partition"] = DataPartition.TEST
        test_df["task_id"] = task_id
        predictions["test"] = test_df

        # Out-of-split validation predictions
        rowid_dev = pd.DataFrame({row_column: row_traindev})

        if problem_type == "regression":
            oof_pred_data = model.predict_oof(model=best_model_name)
            oof_pred = pd.DataFrame({"prediction": oof_pred_data})
        elif problem_type in ["binary", "multiclass"]:
            oof_pred = model.predict_proba_oof(model=best_model_name)
            class_labels = model.class_labels
            oof_pred.columns = class_labels
            oof_pred["prediction"] = oof_pred[class_labels].values.argmax(axis=1)
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")

        assert len(rowid_dev) == len(oof_pred), "Mismatch in number of dev rows!"
        dev_df = pd.concat(
            [rowid_dev.reset_index(drop=True), oof_pred.reset_index(drop=True)],
            axis=1,
        )
        # Add metadata
        dev_df["outer_split_id"] = outer_split_id
        dev_df["inner_split_id"] = inner_split_id
        dev_df["partition"] = DataPartition.DEV
        dev_df["task_id"] = task_id
        predictions["dev"] = dev_df

        return predictions
