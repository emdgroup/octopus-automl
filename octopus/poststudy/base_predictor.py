"""Shared base class for OctoPredictor and OctoTestEvaluator.

Not part of the public API.  Both predictor classes inherit from
``_PredictorBase``, which provides artifact loading, typed properties,
per-split scoring, summary scoring, and FI dispatch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from attrs import define, field

from octopus.metrics.utils import get_performance_from_model
from octopus.poststudy.study_io import StudyInfo, load_task_artifacts
from octopus.types import FIType, MLType, ResultType


@define(slots=False)
class _PredictorBase:
    """Shared loading, fields, and typed properties for predictors.

    Not part of the public API. Both ``OctoPredictor`` and
    ``OctoTestEvaluator`` inherit from this class.

    Args:
        study_info: ``StudyInfo`` returned by ``load_study_information()``.
        task_id: Concrete workflow task index (must be >= 0).
        result_type: Result type for filtering results (default: 'best').
    """

    _study_info: StudyInfo = field(alias="study_info")
    _task_id: int = field(alias="task_id")
    _result_type: ResultType = field(default=ResultType.BEST, alias="result_type", converter=ResultType)

    _models: dict[int, Any] = field(init=False, factory=dict, repr=False)
    _feature_cols_per_split: dict[int, list[str]] = field(init=False, factory=dict, repr=False)
    _feature_groups_per_split: dict[int, dict[str, list[str]]] = field(init=False, factory=dict, repr=False)
    _feature_cols: list[str] = field(init=False, factory=list, repr=False)

    def __attrs_post_init__(self) -> None:
        """Validate task_id, load artifacts from the study directory."""
        if self._task_id < 0:
            raise ValueError(f"task_id must be >= 0, got {self._task_id}")
        valid_task_ids = {t["task_id"] for t in self._study_info.workflow_tasks}
        if self._task_id not in valid_task_ids:
            raise ValueError(f"task_id {self._task_id} not found in study, available: {sorted(valid_task_ids)}")
        if self._study_info.row_id_col is None:
            raise ValueError(
                "config['prepared']['row_id_col'] is missing. Studies created with an older version must be re-run."
            )

        self._models, self._feature_cols_per_split, self._feature_groups_per_split = load_task_artifacts(
            self._study_info.path, self._study_info.outersplits, self._task_id, self._result_type
        )

        if not self._models:
            raise ValueError(f"No models found for task {self._task_id}.")

        all_feature_cols: set[str] = set()
        for split_fcols in self._feature_cols_per_split.values():
            if split_fcols:
                all_feature_cols.update(split_fcols)

        if all_feature_cols:
            self._feature_cols = sorted(all_feature_cols)
        else:
            self._feature_cols = self._study_info.feature_cols

    @property
    def study_info(self) -> StudyInfo:
        """Study metadata (ml_type, target_metric, target_col, etc.).

        On loaded predictors (via ``OctoPredictor.load``), the returned
        ``StudyInfo`` has empty ``workflow_tasks`` and ``outersplit_dirs``.
        Do not pass it to analysis functions like ``get_performance()``
        which iterate ``outersplit_dirs`` — they will silently return
        empty results.
        """
        return self._study_info

    @property
    def task_id(self) -> int:
        """Workflow task index."""
        return self._task_id

    @property
    def result_type(self) -> ResultType:
        """Result type (e.g. 'best', 'ensemble_selection')."""
        return self._result_type

    @property
    def feature_cols(self) -> list[str]:
        """Union of input feature columns across all outersplits."""
        return self._feature_cols

    @property
    def classes_(self) -> np.ndarray:
        """Class labels from the first model (classification only).

        Raises:
            AttributeError: If the model does not have a classes_ attribute.
        """
        model = self._models[self._study_info.outersplits[0]]
        if not hasattr(model, "classes_"):
            raise AttributeError(f"Not a classification model: {type(model).__name__}")
        result: np.ndarray = model.classes_
        return result

    @property
    def feature_cols_per_split(self) -> dict[int, list[str]]:
        """Input feature columns per outersplit (loaded from disk)."""
        return self._feature_cols_per_split

    @property
    def feature_groups_per_split(self) -> dict[int, dict[str, list[str]]]:
        """Feature groups per outersplit (loaded from disk)."""
        return self._feature_groups_per_split

    def get_model(self, outersplit_id: int) -> Any:
        """Get the fitted model for an outersplit.

        Args:
            outersplit_id: Outer split index.

        Returns:
            The fitted model object.
        """
        return self._models[outersplit_id]

    def _get_row_ids(self, data: pd.DataFrame) -> Any:
        """Return row identifiers from *data*.

        Uses ``row_id_col`` when it is set and present in the DataFrame,
        otherwise falls back to ``data.index``.

        Args:
            data: Input DataFrame.

        Returns:
            Array-like of row identifiers.
        """
        row_id_col = self._study_info.row_id_col
        if row_id_col and row_id_col in data.columns:
            return data[row_id_col].values
        return data.index

    def _get_target_columns(self, data: pd.DataFrame) -> dict[str, Any]:
        """Build target column(s) dict from a DataFrame.

        For single-target tasks: ``{"target": <array>}``.
        For multi-target tasks (T2E): ``{"target_duration": ..., "target_event": ...}``.

        Args:
            data: DataFrame containing the target column(s).

        Returns:
            Dict mapping output column names to arrays of target values.
        """
        assignments = self._study_info.target_assignments
        if len(assignments) == 1:
            col = next(iter(assignments.values()))
            return {"target": data[col].values}
        return {f"target_{role}": data[col].values for role, col in assignments.items()}

    def _predict_raw(self, split_id: int, data: pd.DataFrame) -> np.ndarray:
        """Run model.predict for a single outersplit on the given data."""
        features = self._feature_cols_per_split[split_id]
        result: np.ndarray = self._models[split_id].predict(data[features])
        return result

    def _predict_proba_raw(self, split_id: int, data: pd.DataFrame) -> np.ndarray:
        """Run model.predict_proba for a single outersplit, returning an ndarray."""
        features = self._feature_cols_per_split[split_id]
        probas = self._models[split_id].predict_proba(data[features])
        if isinstance(probas, pd.DataFrame):
            probas = probas.values
        result: np.ndarray = probas
        return result

    def _check_classification_only(self, method_name: str) -> None:
        """Raise TypeError if ml_type is not classification or multiclass."""
        if self._study_info.ml_type not in (MLType.BINARY, MLType.MULTICLASS):
            raise TypeError(
                f"{method_name}() is only available for classification and multiclass tasks, "
                f"but this study has ml_type='{self._study_info.ml_type}'."
            )

    def _resolve_metrics(self, metrics: list[str] | None) -> list[str]:
        """Return explicit metric list, auto-detecting from ML type if None."""
        if metrics is not None:
            return metrics
        from octopus.metrics.core import Metrics  # noqa: PLC0415

        return Metrics.get_by_type(self._study_info.ml_type)

    def _needs_proba(self, metrics: list[str]) -> bool:
        """Check if probability predictions are needed for scoring.

        Return True for all classification tasks. Binary metrics threshold
        probabilities, and multiclass metrics need probabilities for both
        probability-type metrics and argmax-based class predictions.
        """
        return self._study_info.ml_type in (MLType.BINARY, MLType.MULTICLASS)

    def _compute_per_split_scores(
        self,
        data_per_split: dict[int, pd.DataFrame],
        metrics: list[str],
        threshold: float,
    ) -> pd.DataFrame:
        """Score each outer-split model on its corresponding data.

        Args:
            data_per_split: Dict mapping outersplit_id to scoring DataFrame.
            metrics: Resolved list of metric names (must not be None).
            threshold: Classification threshold for threshold-dependent metrics.

        Returns:
            Wide DataFrame with outersplit IDs as index, metrics as columns.
        """
        rows = []
        for split_id in self._study_info.outersplits:
            model = self._models[split_id]
            features = self._feature_cols_per_split[split_id]
            row: dict[str, Any] = {}
            for metric_name in metrics:
                row[metric_name] = get_performance_from_model(
                    model=model,
                    data=data_per_split[split_id],
                    feature_cols=features,
                    target_metric=metric_name,
                    target_assignments=self._study_info.target_assignments,
                    threshold=threshold,
                    positive_class=self._study_info.positive_class,
                )
            rows.append(row)
        return pd.DataFrame(rows, index=self._study_info.outersplits)

    def _compute_summary_scores(
        self,
        predictions: pd.DataFrame,
        probabilities: pd.DataFrame | None,
        metrics: list[str],
        threshold: float,
    ) -> dict[str, float]:
        """Score predictions against ground truth for summary rows.

        Unified scoring for both Merged (evaluator) and Ensemble (predictor).

        Args:
            predictions: DataFrame with ``prediction`` column and target
                column(s): ``target`` for classification/regression, or
                ``target_duration`` + ``target_event`` for T2E.
            probabilities: Optional DataFrame with class probability columns.
                For binary: column named by positive_class value.
                For multiclass: columns named by class labels.
                None for regression/T2E.
            metrics: Resolved list of metric names.
            threshold: Classification threshold for threshold-dependent metrics.

        Returns:
            Dict mapping metric name to score.
        """
        from octopus.metrics.core import Metrics  # noqa: PLC0415
        from octopus.types import PredictionType  # noqa: PLC0415

        ml_type = self._study_info.ml_type
        positive_class = self._study_info.positive_class
        scores: dict[str, float] = {}

        if ml_type == MLType.TIMETOEVENT:
            for m in metrics:
                metric = Metrics.get_instance(m)
                scores[m] = metric.calculate_t2e(
                    predictions["target_event"].astype(bool),
                    predictions["target_duration"].astype(float),
                    predictions["prediction"],
                )
            return scores

        target = np.asarray(predictions["target"])
        preds = np.asarray(predictions["prediction"])

        proba: np.ndarray | None = None
        if probabilities is not None:
            if ml_type == MLType.BINARY:
                proba = np.asarray(probabilities[positive_class])
            else:
                proba = np.asarray(probabilities[list(self.classes_)])

        target_binary: np.ndarray | None = None
        if ml_type == MLType.BINARY and positive_class is not None:
            target_binary = (target == positive_class).astype(int)

        for m in metrics:
            metric = Metrics.get_instance(m)
            if metric.prediction_type == PredictionType.PROBABILITIES and proba is not None:
                scoring_target = target_binary if target_binary is not None else target
                scores[m] = metric.calculate(scoring_target, proba)
            elif ml_type == MLType.BINARY and positive_class is not None:
                thresholded = (proba >= threshold).astype(int) if proba is not None else preds
                assert target_binary is not None
                scores[m] = metric.calculate(target_binary, thresholded)
            elif ml_type == MLType.MULTICLASS and proba is not None:
                class_labels = list(self.classes_)
                argmax_indices = np.argmax(proba, axis=1)
                mapped_preds = np.array([class_labels[i] for i in argmax_indices])
                scores[m] = metric.calculate(target, mapped_preds)
            else:
                scores[m] = metric.calculate(target, preds)

        return scores

    def _dispatch_fi(
        self,
        test_data: dict[int, pd.DataFrame],
        train_data: dict[int, pd.DataFrame],
        fi_type: FIType,
        *,
        n_repeats: int = 10,
        feature_groups: dict[str, list[str]] | None = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Forward to the stateless dispatch_fi with instance attributes."""
        from octopus.poststudy.feature_importance import dispatch_fi as _dispatch  # noqa: PLC0415

        return _dispatch(
            models=self._models,
            feature_cols_per_split=self._feature_cols_per_split,
            test_data=test_data,
            train_data=train_data,
            target_assignments=self._study_info.target_assignments,
            target_metric=self._study_info.target_metric,
            positive_class=self._study_info.positive_class,
            feature_cols=self._feature_cols,
            feature_groups_per_split=self._feature_groups_per_split,
            fi_type=fi_type,
            n_repeats=n_repeats,
            feature_groups=feature_groups,
            random_state=random_state,
            ml_type=self._study_info.ml_type,
            **kwargs,
        )
