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

        loader = StudyLoader(self._study_path)

        for split_id in self._outersplits:
            split_loader = loader.get_outersplit_loader(
                outersplit_id=split_id,
                task_id=self._task_id,
                result_type=self._result_type,
            )
            self._train_data[split_id] = split_loader.load_partition("traindev")
            self._test_data[split_id] = split_loader.load_partition("test")

    # ── Prediction (per-split on own test data) ─────────────────

    def _get_target_columns(self, test: pd.DataFrame) -> dict[str, Any]:
        """Build target column(s) for ``df=True`` output.

        Returns a dict suitable for unpacking into a DataFrame constructor.

        For single-target tasks (regression, binary, multiclass):
            ``{"target": <array>}``

        For multi-target tasks (T2E):
            ``{"target_duration": <array>, "target_event": <array>}``
            — one key per role in ``target_assignments``, prefixed with
            ``"target_"``.

        The single-target form uses the bare name ``"target"`` (no role
        suffix) to preserve backwards compatibility with existing callers.

        Args:
            test: DataFrame containing the target column(s).

        Returns:
            Dict mapping output column names to arrays of target values.
        """
        assignments = self.target_assignments
        if len(assignments) == 1:
            col = next(iter(assignments.values()))
            return {"target": test[col].values}
        return {f"target_{role}": test[col].values for role, col in assignments.items()}

    def predict(self, df: bool = False) -> np.ndarray | pd.DataFrame:  # type: ignore[override]
        """Predict on stored test data.  Each model predicts only on its own test data.

        No ensemble averaging — results are collected per split.

        Args:
            df: If True, return a DataFrame with outersplit, row_id, prediction,
                and target columns.  For T2E tasks the target columns are
                ``target_duration`` and ``target_event`` instead of ``target``.
                If False (default), return concatenated ndarray.

        Returns:
            Per-split predictions as ndarray or DataFrame.
        """
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
                        **self._get_target_columns(test),
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
                columns per class, and target column(s).  If False (default),
                return concatenated ndarray.

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
                for col_name, col_values in self._get_target_columns(test).items():
                    split_df[col_name] = col_values
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

        rows = []
        for split_id in self._outersplits:
            model = self._models[split_id]
            features = self._selected_features[split_id]
            test = self._test_data[split_id]

            for metric_name in metrics:
                score = get_performance_from_model(
                    model=model,
                    data=test,
                    feature_cols=features,
                    target_metric=metric_name,
                    target_assignments=self.target_assignments,
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
        Delegates to ``_dispatch_fi()`` (inherited from ``TaskPredictor``)
        with stored per-split test and train data.

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
        fi_type = FIType(fi_type)

        return self._dispatch_fi(
            self._test_data,
            self._train_data,
            fi_type,
            n_repeats=n_repeats,
            feature_groups=feature_groups,
            random_state=random_state,
            **kwargs,
        )

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
