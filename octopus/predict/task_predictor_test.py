"""TaskPredictorTest — test-data analysis predictor.

Extends TaskPredictor with stored test/train data for analysing study results
on held-out test data.  Each outer-split model predicts ONLY on its
corresponding test data — models never see test data from other splits.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from attrs import define, field
from upath import UPath

from octopus.metrics.utils import get_performance_from_model
from octopus.predict.study_io import StudyLoader
from octopus.predict.task_predictor import TaskPredictor
from octopus.types import FIType, MLType


@define(slots=False)
class TaskPredictorTest(TaskPredictor):
    """Predictor for analysing study results on held-out test data.

    Inherits from ``TaskPredictor`` and additionally stores test and train
    data.  Overrides ``predict``, ``predict_proba``, ``performance``, and
    ``calculate_fi`` to use stored test data implicitly — the caller never
    needs to pass data.

    Each outer-split model predicts **only** on its corresponding test data.
    No averaging across splits.

    Args:
        study_path: Path to the study directory.
        task_id: Concrete workflow task index (must be >= 0).
        result_type: Result type for filtering results (default: 'best').

    Raises:
        ValueError: If task_id is negative, out of range, or no models found.
        FileNotFoundError: If expected study artifacts are missing.

    Example:
        >>> tp = TaskPredictorTest("studies/my_study", task_id=0)
        >>> scores = tp.performance(metrics=["AUCROC", "ACC"])
    """

    # Additional fields for test/train data (populated in __attrs_post_init__)
    _test_data: dict[int, pd.DataFrame] = field(init=False, factory=dict, repr=False)
    _train_data: dict[int, pd.DataFrame] = field(init=False, factory=dict, repr=False)

    def __attrs_post_init__(self) -> None:
        """Load base artifacts via parent, then additionally load test/train data."""
        # Call parent __attrs_post_init__ to load config, validate, and load models
        super().__attrs_post_init__()

        # Additionally load test and train data per split via StudyLoader factory
        loader = StudyLoader(self._study_path)
        for split_id in self._outersplits:
            split_loader = loader.get_outersplit_loader(
                outersplit_id=split_id,
                task_id=self._task_id,
                result_type=self._result_type,
            )
            self._test_data[split_id] = split_loader.load_test_data()
            self._train_data[split_id] = split_loader.load_train_data()

    # ── Prediction (per-split on own test data) ─────────────────

    def predict(self, df: bool = False) -> np.ndarray | pd.DataFrame:  # type: ignore[override]
        """Predict on stored test data.  Each model predicts only on its own test data.

        No ensemble averaging — results are collected per split.

        Args:
            df: If True, return a DataFrame with outersplit, row_id, prediction,
                and target columns.  If False (default), return concatenated ndarray.

        Returns:
            Per-split predictions as ndarray or DataFrame.
        """
        target_col = self._resolve_target_col()
        row_id_col = self.row_id_col

        all_preds = []
        all_rows = []

        for split_id in self._outersplits:
            features = self._selected_features[split_id]
            test = self._test_data[split_id]
            preds = self._models[split_id].predict(test[features])
            all_preds.append(preds)

            if df:
                row_ids = test[row_id_col] if row_id_col and row_id_col in test.columns else pd.RangeIndex(len(test))
                split_df = pd.DataFrame(
                    {
                        "outersplit": split_id,
                        "row_id": row_ids.values if hasattr(row_ids, "values") else row_ids,
                        "prediction": preds,
                        "target": test[target_col].values,
                    }
                )
                all_rows.append(split_df)

        if df:
            return pd.concat(all_rows, ignore_index=True)
        return np.concatenate(all_preds)

    def predict_proba(self, df: bool = False) -> np.ndarray | pd.DataFrame:  # type: ignore[override]
        """Predict probabilities on stored test data (classification/multiclass only).

        Each model predicts only on its own test data.  No averaging.

        Args:
            df: If True, return a DataFrame with outersplit, row_id, probability
                columns per class, and target.  If False (default), return
                concatenated ndarray.

        Returns:
            Per-split probabilities as ndarray or DataFrame.

        Raises:
            TypeError: If ml_type is not classification or multiclass.
        """
        if self.ml_type not in (MLType.BINARY, MLType.MULTICLASS):
            raise TypeError(
                f"predict_proba() is only available for classification and multiclass tasks, "
                f"but this study has ml_type='{self.ml_type}'."
            )
        target_col = self._resolve_target_col()
        row_id_col = self.row_id_col
        class_labels = self.classes_

        all_probas = []
        all_rows = []

        for split_id in self._outersplits:
            features = self._selected_features[split_id]
            test = self._test_data[split_id]
            probas = self._models[split_id].predict_proba(test[features])
            if isinstance(probas, pd.DataFrame):
                probas = probas.values
            all_probas.append(probas)

            if df:
                row_ids = test[row_id_col] if row_id_col and row_id_col in test.columns else pd.RangeIndex(len(test))
                split_df = pd.DataFrame(probas, columns=class_labels)
                split_df.insert(0, "outersplit", split_id)
                row_vals: Any = row_ids.values if hasattr(row_ids, "values") else row_ids
                split_df.insert(1, "row_id", row_vals)
                split_df["target"] = test[target_col].values
                all_rows.append(split_df)

        if df:
            return pd.concat(all_rows, ignore_index=True)
        return np.concatenate(all_probas)

    # ── Scoring (per-split on own test data) ────────────────────

    def performance(  # type: ignore[override]
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
            DataFrame with columns: outersplit, metric, score.
        """
        if metrics is None:
            metrics = [self.target_metric]

        target_col = self._resolve_target_col()

        rows = []
        for split_id in self._outersplits:
            model = self._models[split_id]
            features = self._selected_features[split_id]
            test = self._test_data[split_id]

            for metric_name in metrics:
                target_assignments = {target_col: target_col}
                score = get_performance_from_model(
                    model=model,
                    data=test,
                    feature_cols=features,
                    target_metric=metric_name,
                    target_assignments=target_assignments,
                    threshold=threshold,
                    positive_class=self.positive_class,
                )
                rows.append({"outersplit": split_id, "metric": metric_name, "score": score})

        return pd.DataFrame(rows)

    # ── Feature Importance (per-split on own test data) ─────────

    def calculate_fi(  # type: ignore[override]
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
                  kwarg to select the explainer: ``ShapExplainerType.KERNEL`` (default),
                  ``ShapExplainerType.PERMUTATION``, or ``ShapExplainerType.EXACT``.
            n_repeats: Number of permutation repeats.
            feature_groups: Dict mapping group names to feature lists.
                If None and fi_type is ``FIType.GROUP_PERMUTATION``, groups are
                loaded from the study.
            random_state: Random seed.
            **kwargs: Additional keyword arguments passed to the FI function.
                For ``fi_type=FIType.SHAP``, supported kwargs include:
                ``shap_type`` (``ShapExplainerType.KERNEL``, ``ShapExplainerType.PERMUTATION``,
                ``ShapExplainerType.EXACT``),
                ``max_samples``, ``background_size``.

        Returns:
            DataFrame with feature importance results including a ``fi_type``
            column and per-split + ensemble rows.

        Raises:
            ValueError: If fi_type is unknown.
        """
        from octopus.predict.feature_importance import (  # noqa: PLC0415
            calculate_fi_permutation,
            calculate_fi_shap,
        )

        fi_type = FIType(fi_type)
        target_col = self._resolve_target_col()

        if fi_type in (FIType.PERMUTATION, FIType.GROUP_PERMUTATION):
            resolved_groups = None
            if fi_type == FIType.GROUP_PERMUTATION:
                if feature_groups is not None:
                    resolved_groups = feature_groups
                else:
                    resolved_groups = self._compute_feature_groups()

            result = calculate_fi_permutation(
                models=self._models,
                selected_features=self._selected_features,
                test_data=self._test_data,
                train_data=self._train_data,
                target_col=target_col,
                target_metric=self.target_metric,
                positive_class=self.positive_class,
                n_repeats=n_repeats,
                random_state=random_state,
                feature_groups=resolved_groups,
                feature_cols=self._feature_cols,
            )
        elif fi_type == FIType.SHAP:
            result = calculate_fi_shap(
                models=self._models,
                selected_features=self._selected_features,
                test_data=self._test_data,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown fi_type '{fi_type}'. Use FIType.PERMUTATION, FIType.GROUP_PERMUTATION, or FIType.SHAP."
            )

        result.insert(0, "fi_type", fi_type.value)
        return result

    # ── Serialization — not supported ───────────────────────────

    def save(self, path: str | UPath) -> None:
        """Not supported for TaskPredictorTest.

        Args:
            path: Ignored — not used.

        Raises:
            NotImplementedError: Always. The study directory is the
                persistent artifact for test predictors.
        """
        raise NotImplementedError(
            "TaskPredictorTest does not support save(). "
            "The study directory is the persistent artifact. "
            "Use TaskPredictor for standalone deployment."
        )

    @classmethod
    def load(cls, path: str | UPath) -> TaskPredictorTest:
        """Not supported for TaskPredictorTest.

        Args:
            path: Ignored — not used.

        Returns:
            Never returns — always raises.

        Raises:
            NotImplementedError: Always. Use TaskPredictor.load() for
                loading saved predictors.
        """
        raise NotImplementedError(
            "TaskPredictorTest does not support load(). "
            "Construct from a study directory instead, or use TaskPredictor.load() "
            "for standalone deployment."
        )
