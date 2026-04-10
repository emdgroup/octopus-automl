"""OctoTestEvaluator — test-data analysis predictor.

Extends OctoPredictor with stored test/train data for analysing study results
on held-out test data.  Each outer-split model predicts ONLY on its
corresponding test data — models never see test data from other splits.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from attrs import define, field

from octopus.poststudy.task_predictor import _PredictorBase
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

        prepared_data = parquet_load(self._study_info.path / "data_prepared.parquet")

        for split_id, loader in self._loaders.items():
            self._train_data[split_id] = loader.load_partition("traindev", prepared_data)
            self._test_data[split_id] = loader.load_partition("test", prepared_data)

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
        assignments = self._study_info.target_assignments
        if len(assignments) == 1:
            col = next(iter(assignments.values()))
            return {"target": test[col].values}
        return {f"target_{role}": test[col].values for role, col in assignments.items()}

    def predict(self, df: bool = False) -> np.ndarray | pd.DataFrame:
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
        row_id_col = self._study_info.row_id_col

        all_preds = []
        all_rows = []

        for split_id in self._study_info.outersplits:
            features = self._feature_cols_per_split[split_id]
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

    def predict_proba(self, df: bool = False) -> np.ndarray | pd.DataFrame:
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
        self._check_classification_only("predict_proba")
        row_id_col = self._study_info.row_id_col
        class_labels = self.classes_

        all_probas = []
        all_rows = []

        for split_id in self._study_info.outersplits:
            features = self._feature_cols_per_split[split_id]
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
            DataFrame with columns: outersplit, metric, score.
        """
        if metrics is None:
            metrics = [self._study_info.target_metric]
        return self._compute_performance(self._test_data, metrics, threshold)

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
