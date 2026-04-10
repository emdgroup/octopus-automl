"""OctoTestEvaluator — evaluate models on held-out test folds.

Standalone evaluator with stored test/train data for evaluating study results
on held-out test data.  Each outer-split model predicts ONLY on its
corresponding test data — models never see test data from other splits.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from attrs import define, field

from octopus.metrics.utils import get_performance_from_model
from octopus.predict.study_io import (
    StudyInfo,
    load_feature_cols,
    load_feature_groups,
    load_model,
    load_prepared_data,
    load_selected_features,
    load_split_data,
)
from octopus.types import FIType, MLType, ResultType


@define(slots=False)
class OctoTestEvaluator:
    """Predictor for analysing study results on held-out test data.

    Stores test and train data alongside models.  Uses stored
    test data implicitly — the caller never needs to pass data.

    Each outer-split model predicts **only** on its corresponding test data.
    No averaging across splits.

    Attributes:
        study: Validated study (from ``load_study_info()``).
        task_id: Concrete workflow task index (must be >= 0).
        result_type: Result type for filtering results (default: 'best').

    Raises:
        ValueError: If task_id is out of range or no models found.
        FileNotFoundError: If expected model artifacts are missing.

    Example:
        >>> study = load_study_info("studies/my_study")
        >>> tp = OctoTestEvaluator(study, task_id=0)
        >>> scores = tp.performance(metrics=["AUCROC", "ACC"])
    """

    study: StudyInfo = field()
    """Validated study (from ``load_study_info()``)."""

    task_id: int = field()
    """Workflow task index (>= 0)."""

    result_type: ResultType = field(default=ResultType.BEST, converter=ResultType)
    """Result type: 'best' or 'ensemble_selection'."""

    def __attrs_post_init__(self) -> None:
        """Load models, per-split artifacts, and test/train data."""
        self._config = self.study.config
        self._models: dict[int, Any] = {}
        self._selected_features: dict[int, list[str]] = {}
        self._training_features: dict[int, list[str]] = {}
        self._feature_groups: dict[int, dict[str, list[str]]] = {}
        self._test_data: dict[int, pd.DataFrame] = {}
        self._train_data: dict[int, pd.DataFrame] = {}

        valid_task_ids = {t["task_id"] for t in self.study.workflow_tasks}
        if self.task_id not in valid_task_ids:
            raise ValueError(f"task_id {self.task_id} not found in study, available: {sorted(valid_task_ids)}")

        split_ids = [int(d.name.removeprefix("outersplit")) for d in self.study.outer_split_dirs]
        prepared_data = load_prepared_data(self.study)
        for split_id in split_ids:
            self._models[split_id] = load_model(self.study, split_id, self.task_id, self.result_type)
            self._selected_features[split_id] = load_selected_features(
                self.study, split_id, self.task_id, self.result_type
            )
            self._training_features[split_id] = load_feature_cols(self.study, split_id, self.task_id)
            self._feature_groups[split_id] = load_feature_groups(self.study, split_id, self.task_id)

            test_data, train_data = load_split_data(self.study, split_id, prepared_data=prepared_data)
            self._test_data[split_id] = test_data
            self._train_data[split_id] = train_data

        # Compute union of training features across all outersplits
        all_feature_cols: set[str] = set()
        for split_id in self._models:
            all_feature_cols.update(self._training_features[split_id])
        self._feature_cols = sorted(all_feature_cols)

    def _get_target_columns(self, test: pd.DataFrame) -> dict[str, Any]:
        """Build target column(s) for ``df=True`` output.

        Returns a dict suitable for unpacking into a DataFrame constructor.

        For single-target tasks (regression, binary, multiclass):
            ``{"target": <array>}``

        For multi-target tasks (T2E):
            ``{"target_duration": <array>, "target_event": <array>}``
            — one key per role in ``target_assignments``, prefixed with
            ``"target_"``.

        Args:
            test: DataFrame containing the target column(s).

        Returns:
            Dict mapping output column names to arrays of target values.
        """
        assignments = self._config.get("prepared", {}).get("target_assignments", {})
        if len(assignments) == 1:
            col = next(iter(assignments.values()))
            return {"target": test[col].values}
        return {f"target_{role}": test[col].values for role, col in assignments.items()}

    def predict(self) -> pd.DataFrame:
        """Predict on stored test data.  Each model predicts only on its own test data.

        No ensemble averaging — results are collected per split.

        Returns:
            DataFrame with columns: outer_split, row_id, prediction,
            and target column(s).  For T2E tasks the target columns are
            ``target_duration`` and ``target_event`` instead of ``target``.
        """
        row_id_col = self._config["prepared"]["row_id_col"]
        all_rows = []

        for split_id in self._models:
            test = self._test_data[split_id]
            preds = self._models[split_id].predict(test[self._training_features[split_id]])
            all_rows.append(
                pd.DataFrame(
                    {
                        "outer_split": split_id,
                        "row_id": test[row_id_col].values,
                        "prediction": preds,
                        **self._get_target_columns(test),
                    }
                )
            )

        return pd.concat(all_rows, ignore_index=True)

    def predict_proba(self) -> pd.DataFrame:
        """Predict probabilities on stored test data (classification/multiclass only).

        Each model predicts only on its own test data.  No averaging.

        Returns:
            DataFrame with columns: outer_split, row_id, one probability
            column per class, and target column(s).

        Raises:
            TypeError: If ml_type is not classification or multiclass.
        """
        ml_type = MLType(self._config["ml_type"])
        if ml_type not in (MLType.BINARY, MLType.MULTICLASS):
            raise TypeError(
                f"predict_proba() is only available for classification and multiclass tasks, "
                f"but this study has ml_type='{ml_type}'."
            )
        row_id_col = self._config["prepared"]["row_id_col"]
        class_labels = next(iter(self._models.values())).classes_
        all_rows = []

        for split_id in self._models:
            test = self._test_data[split_id]
            probas = self._models[split_id].predict_proba(test[self._training_features[split_id]])
            if isinstance(probas, pd.DataFrame):
                probas = probas.values
            split_df = pd.DataFrame(probas, columns=class_labels)
            split_df.insert(0, "outer_split", split_id)
            split_df.insert(1, "row_id", test[row_id_col].values)
            for col_name, col_values in self._get_target_columns(test).items():
                split_df[col_name] = col_values
            all_rows.append(split_df)

        return pd.concat(all_rows, ignore_index=True)

    def performance(
        self,
        metrics: list[str] | None = None,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Compute performance scores on stored test data.

        Each outer-split model is scored **only on its own test data**.
        Scores are computed fresh — never read from disk.

        Args:
            metrics: List of metric names to compute.
                If None, uses the study target metric.
            threshold: Classification threshold for threshold-dependent metrics.

        Returns:
            DataFrame with columns: outer_split, metric, score.
        """
        if metrics is None:
            metrics = [self._config.get("target_metric", "")]
        target_assignments = self._config.get("prepared", {}).get("target_assignments", {})

        rows = []
        for split_id in self._models:
            for metric_name in metrics:
                score = get_performance_from_model(
                    model=self._models[split_id],
                    data=self._test_data[split_id],
                    feature_cols=self._training_features[split_id],
                    target_metric=metric_name,
                    target_assignments=target_assignments,
                    threshold=threshold,
                    positive_class=self._config.get("positive_class"),
                )
                rows.append({"outer_split": split_id, "metric": metric_name, "score": score})

        return pd.DataFrame(rows)

    def calculate_fi(
        self,
        fi_type: FIType = FIType.PERMUTATION,
        *,
        n_repeats: int = 10,
        feature_groups: dict[str, list[str]] | None = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Calculate feature importance using stored test data and models.

        Each split's model permutes features only in its own test data.

        Args:
            fi_type: Type of feature importance. One of:
                - ``FIType.PERMUTATION`` — Per-feature permutation importance.
                - ``FIType.GROUP_PERMUTATION`` — Per-feature + per-group permutation
                  importance.  Uses ``feature_groups`` (from study config or
                  explicitly provided) to also compute group-level importance.
                - ``FIType.SHAP`` — SHAP-based importance.  Pass ``shap_type`` as a
                  kwarg to select the explainer: ``"kernel"`` (default),
                  ``"permutation"``, or ``"exact"``.
            n_repeats: Number of permutation repeats.
            feature_groups: Dict mapping group names to feature lists.
                If None and fi_type is ``FIType.GROUP_PERMUTATION``, groups are
                loaded from the study.
            random_state: Random seed.
            **kwargs: Additional keyword arguments passed to the FI function.
                For ``fi_type=FIType.SHAP``, supported kwargs include:
                ``shap_type`` (``"kernel"``, ``"permutation"``,
                ``"exact"``),
                ``max_samples``, ``background_size``.

        Returns:
            DataFrame with feature importance results including a ``fi_type``
            column and per-split + ensemble rows.

        Raises:
            ValueError: If fi_type is unknown.
        """
        from octopus.predict.feature_importance import dispatch_fi  # noqa: PLC0415

        fi_type = FIType(fi_type)

        return dispatch_fi(
            models=self._models,
            selected_features=self._selected_features,
            test_data=self._test_data,
            train_data=self._train_data,
            target_assignments=self._config.get("prepared", {}).get("target_assignments", {}),
            target_metric=self._config.get("target_metric", ""),
            positive_class=self._config.get("positive_class"),
            feature_cols=self._feature_cols,
            feature_groups_per_split=self._feature_groups,
            fi_type=fi_type,
            n_repeats=n_repeats,
            feature_groups=feature_groups,
            random_state=random_state,
            ml_type=MLType(self._config["ml_type"]),
            **kwargs,
        )
