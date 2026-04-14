"""OctoTestEvaluator — test-data analysis predictor.

Stores test/train data per outer split for analysing study results on
held-out test data.  Each outer-split model predicts ONLY on its
corresponding test data — models never see test data from other splits.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from attrs import define, field

from octopus.poststudy.base_predictor import _PredictorBase
from octopus.types import FIType
from octopus.utils import parquet_load


@define(slots=False)
class OctoTestEvaluator(_PredictorBase):
    """Predictor for analysing study results on held-out test data.

    Stores test and train data per outer split.  Methods use stored test
    data implicitly — the caller never needs to pass data.

    Each outer-split model predicts **only** on its corresponding test data.
    No averaging across splits.

    Args:
        study_info: ``StudyInfo`` returned by ``load_study_information()``.
        task_id: Concrete workflow task index (must be >= 0).
        result_type: Result type for filtering results (default: 'best').

    Raises:
        ValueError: If task_id is negative, out of range, or no models found.
        FileNotFoundError: If expected study artifacts are missing.

    Example:
        >>> info = load_study_information("studies/my_study")
        >>> tp = OctoTestEvaluator(study_info=info, task_id=0)
        >>> scores = tp.performance(metrics=["AUCROC", "ACC"])
    """

    _test_data: dict[int, pd.DataFrame] = field(init=False, factory=dict, repr=False)
    _train_data: dict[int, pd.DataFrame] = field(init=False, factory=dict, repr=False)

    def __attrs_post_init__(self) -> None:
        """Load base artifacts via parent, then additionally load test/train data."""
        super().__attrs_post_init__()

        from octopus.poststudy.study_io import load_partition  # noqa: PLC0415

        prepared_data = parquet_load(self._study_info.path / "data_prepared.parquet")

        for split_id in self._study_info.outersplits:
            self._train_data[split_id] = load_partition(self._study_info.path, split_id, "traindev", prepared_data)
            self._test_data[split_id] = load_partition(self._study_info.path, split_id, "test", prepared_data)

    def predict(self) -> pd.DataFrame:
        """Predict on stored test data.  Each model predicts only on its own test data.

        No ensemble averaging — results are collected per split.

        Returns:
            DataFrame with columns: ``outersplit``, ``row_id``, ``prediction``,
            and target column(s).  For T2E tasks the target columns are
            ``target_duration`` and ``target_event`` instead of ``target``.
        """
        all_rows: list[pd.DataFrame] = []

        for split_id in self._study_info.outersplits:
            test = self._test_data[split_id]
            preds = self._predict_raw(split_id, test)

            split_df = pd.DataFrame(
                {
                    "outersplit": split_id,
                    "row_id": self._get_row_ids(test),
                    "prediction": preds,
                    **self._get_target_columns(test),
                }
            )
            all_rows.append(split_df)

        return pd.concat(all_rows, ignore_index=True)

    def predict_proba(self) -> pd.DataFrame:
        """Predict probabilities on stored test data (classification/multiclass only).

        Each model predicts only on its own test data.  No averaging.

        Returns:
            DataFrame with columns: ``outersplit``, ``row_id``, one probability
            column per class label, and target column(s).

        Raises:
            TypeError: If ml_type is not classification or multiclass.
        """
        self._check_classification_only("predict_proba")
        class_labels = self.classes_
        all_rows: list[pd.DataFrame] = []

        for split_id in self._study_info.outersplits:
            test = self._test_data[split_id]
            probas = self._predict_proba_raw(split_id, test)

            split_df = pd.DataFrame(probas, columns=class_labels)
            split_df.insert(0, "outersplit", split_id)
            split_df.insert(1, "row_id", self._get_row_ids(test))
            for col_name, col_values in self._get_target_columns(test).items():
                split_df[col_name] = col_values
            all_rows.append(split_df)

        return pd.concat(all_rows, ignore_index=True)

    def performance(
        self,
        metrics: list[str] | None = None,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Compute performance on stored test data per outersplit with Mean and Merged.

        Each outer-split model is scored **only on its own test data**.
        The ``Mean`` row averages per-split scores.  The ``Merged`` row
        pools all test predictions and scores them as one set.

        Args:
            metrics: List of metric names to compute.
                If None, auto-detected from the ML type.
            threshold: Classification threshold for threshold-dependent metrics.

        Returns:
            Wide DataFrame with outersplit IDs as index (plus ``Mean`` and
            ``Merged``), metrics as columns.
        """
        metrics = self._resolve_metrics(metrics)
        df = self._compute_per_split_scores(self._test_data, metrics, threshold)
        df.loc["Mean"] = df.mean()

        pred_df = self.predict()
        proba_df = self.predict_proba() if self._needs_proba(metrics) else None
        df.loc["Merged"] = self._compute_summary_scores(pred_df, proba_df, metrics, threshold)
        return df

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
        Delegates to ``_dispatch_fi()`` (inherited from ``_PredictorBase``)
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
