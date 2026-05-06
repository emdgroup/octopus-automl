"""AutoGluon execution module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import pandas as pd
from attrs import define
from sklearn.base import BaseEstimator
from upath import UPath

from octopus._optional.autogluon import (
    TabularPredictor,
    accuracy,
    average_precision,
    balanced_accuracy,
    brier_score,
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
from octopus.metrics.utils import get_performance_from_predictions
from octopus.modules import ModuleExecution, ModuleResult, StudyContext
from octopus.modules.autogluon.adapters import SklearnClassifier, SklearnRegressor
from octopus.modules.autogluon.feature_importance import compute_fi
from octopus.modules.autogluon.predictions import build_predictions
from octopus.types import DataPartition, FIResultLabel, LogGroup, MLType, ResultType

if TYPE_CHECKING:
    from octopus.modules import AutoGluon  # noqa: F401

logger = get_logger()


_AG_METRIC_MAP: Final[dict[str, Any]] = {
    "ACC": accuracy,
    "ACCBAL": balanced_accuracy,
    "ACCBAL_MC": balanced_accuracy,
    "AUCPR": average_precision,
    "AUCROC": roc_auc,
    "AUCROC_MACRO": roc_auc_ovr,
    "AUCROC_WEIGHTED": roc_auc_ovr_weighted,
    "BRIERSCORE": brier_score,
    "F1": f1,
    "LOGLOSS": log_loss,
    "MAE": mean_absolute_error,
    "MCC": mcc,
    "MSE": mean_squared_error,
    "PRECISION": precision,
    "R2": r2,
    "RECALL": recall,
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
        outer_split_id: int,
        results_dir: UPath,
        n_assigned_cpus: int,
        feature_groups: dict[str, list[str]],
        **kwargs,
    ) -> dict[ResultType, ModuleResult]:
        """Fit AutoGluon TabularPredictor.

        AG fits in a single phase and persists directly under `results_dir`
        (specifically `results_dir/best/model/ag_predictor/`); it does not
        use `scratch_dir`. `dependency_results` is unused because AG has no
        upstream task contract. Both are accepted via `**kwargs` to satisfy
        the `ModuleExecution` ABC.
        """
        if study_context.ml_type == MLType.TIMETOEVENT:
            raise ValueError(
                "AutoGluon does not support time-to-event tasks. "
                "Use the Tako module with CatBoostCoxSurvival or XGBoostCoxSurvival models."
            )

        if len(study_context.target_assignments) != 1:
            raise ValueError(f"Single target expected. Got keys: {study_context.target_assignments.keys()}")
        target_col = next(iter(study_context.target_assignments.values()))

        scoring = _AG_METRIC_MAP.get(study_context.target_metric) if study_context.target_metric else None
        if scoring is None:
            raise ValueError(
                f"target_metric={study_context.target_metric!r} is not supported by AutoGluon. "
                f"Supported metrics: {sorted(_AG_METRIC_MAP)}"
            )

        logger.set_log_group(LogGroup.AUTOGLUON, f"OUTER {outer_split_id}")

        ag_train_data = data_traindev[[*feature_cols, target_col]]
        ag_predictor_path = results_dir / ResultType.BEST.value / "model" / "ag_predictor"

        predictor = TabularPredictor(
            label=target_col,
            eval_metric=scoring,
            verbosity=0,
            log_to_file=False,
            path=str(ag_predictor_path),
        )

        logger.info(
            "Starting fit: presets=%s, time_limit=%s, model_types=%s",
            self.config.presets,
            self.config.time_limit,
            self.config.included_model_types or "all",
        )
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
        logger.info("Fitting completed")

        leaderboard = predictor.leaderboard(silent=True)
        best_row = leaderboard.iloc[0]
        logger.info("Best model: %s (score_val=%.4f)", best_row["model"], best_row["score_val"])

        prediction_frames = build_predictions(
            predictor,
            study_context=study_context,
            data_traindev=data_traindev,
            data_test=data_test,
            outer_split_id=outer_split_id,
            task_id=self.config.task_id,
        )
        scores = _compute_scores(prediction_frames, study_context=study_context)
        raw_fi = compute_fi(
            predictor,
            feature_cols=feature_cols,
            feature_groups=feature_groups,
            leaderboard=leaderboard,
        )

        predictions = pd.concat(list(prediction_frames.values()), ignore_index=True)
        if raw_fi.empty:
            fi_df = pd.DataFrame()
        else:
            fi_df = raw_fi[["importance"]].reset_index()
            fi_df.columns = ["feature", "importance"]
            fi_df["fi_method"] = FIResultLabel.PERMUTATION
            fi_df["fi_dataset"] = DataPartition.DEV
            fi_df["training_id"] = raw_fi.attrs["training_id"]

        return {
            ResultType.BEST: ModuleResult(
                result_type=ResultType.BEST,
                module=self.config.module,
                selected_features=feature_cols,
                scores=scores,
                predictions=predictions,
                fi=fi_df,
                model=_build_sklearn_wrapper(predictor, study_context.ml_type),
            )
        }


def _build_sklearn_wrapper(predictor: TabularPredictor, ml_type: MLType) -> BaseEstimator:
    """Wrap the fitted AG predictor as an sklearn-compatible inference object."""
    if ml_type in (MLType.BINARY, MLType.MULTICLASS):
        return SklearnClassifier(predictor)
    if ml_type == MLType.REGRESSION:
        return SklearnRegressor(predictor)
    raise ValueError(f"ML type {ml_type} not supported")


def _compute_scores(
    predictions: dict[DataPartition, pd.DataFrame],
    *,
    study_context: StudyContext,
) -> pd.DataFrame:
    """Compute scores for every metric supported by `study_context.ml_type`.

    Delegates entirely to `octopus.metrics.utils.get_performance_from_predictions`
    so AG and Tako share one metric router.

    Args:
        predictions: Mapping from DataPartition to a Tako-schema prediction
            frame (must include the target column).
        study_context: Study configuration (ml_type, target_assignments,
            positive_class).

    Returns:
        Long-format DataFrame with columns
        `{metric, partition, aggregation, split, value}` and one row per
        (metric, partition).
    """
    metric_names = Metrics.get_by_type(study_context.ml_type)
    rows: list[dict[str, object]] = []
    predictions_for_helper = {"ensemble": dict(predictions.items())}
    for metric_name in metric_names:
        performance = get_performance_from_predictions(
            predictions_for_helper,
            target_metric=metric_name,
            target_assignments=study_context.target_assignments,
            positive_class=study_context.positive_class,
        )
        for partition, value in performance["ensemble"].items():
            rows.append(
                {
                    "metric": metric_name,
                    "partition": partition,
                    "aggregation": "ensemble",
                    "split": None,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)
