"""TaskPredictorTest — test-data analysis predictor.

Standalone predictor with stored test/train data for analysing study results
on held-out test data.  Each outer-split model predicts ONLY on its
corresponding test data — models never see test data from other splits.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from attrs import define, field
from octopus.metrics.utils import get_performance_from_model
from octopus.predict.study_io import StudyInfo
from octopus.types import FIType, MLType, ResultType
from octopus.utils import joblib_load, parquet_load


@define(slots=False)
class TaskPredictorTest:
    """Predictor for analysing study results on held-out test data.

    Stores test and train data alongside models.  Uses stored
    test data implicitly — the caller never needs to pass data.

    Each outer-split model predicts **only** on its corresponding test data.
    No averaging across splits.

    Args:
        study: Validated study (from ``load_study()``).
        task_id: Concrete workflow task index (must be >= 0).
        result_type: Result type for filtering results (default: 'best').

    Raises:
        ValueError: If task_id is out of range or no models found.
        FileNotFoundError: If expected model artifacts are missing.

    Example:
        >>> study = load_study("studies/my_study")
        >>> tp = TaskPredictorTest(study, task_id=0)
        >>> scores = tp.performance(metrics=["AUCROC", "ACC"])
    """

    study: StudyInfo = field()
    """Validated study (from ``load_study()``)."""

    task_id: int = field()
    """Workflow task index (>= 0)."""

    result_type: ResultType = field(default=ResultType.BEST, converter=ResultType)
    """Result type: 'best' or 'ensemble_selection'."""

    def __attrs_post_init__(self) -> None:
        """Load models, per-split artifacts, and test/train data."""
        self._config = self.study.config
        self._models: dict[int, Any] = {}
        self._selected_features: dict[int, list[str]] = {}
        self._feature_cols_per_split: dict[int, list[str]] = {}
        self._feature_groups_per_split: dict[int, dict[str, list[str]]] = {}
        self._test_data: dict[int, pd.DataFrame] = {}
        self._train_data: dict[int, pd.DataFrame] = {}

        valid_task_ids = {t["task_id"] for t in self.study.workflow_tasks}
        if self.task_id not in valid_task_ids:
            raise ValueError(f"task_id {self.task_id} not found in study, available: {sorted(valid_task_ids)}")

        for fold_dir in self.study.outersplit_dirs:
            split_id = int(fold_dir.name.replace("outersplit", ""))
            task_dir = fold_dir / f"task{self.task_id}"
            result_dir = task_dir / "results" / self.result_type

            # Load model
            model_path = result_dir / "model" / "model.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}. Has the study completed successfully?")
            self._models[split_id] = joblib_load(model_path)

            # Load selected features (required)
            sf_path = result_dir / "selected_features.json"
            if not sf_path.exists():
                raise FileNotFoundError(f"Selected features not found: {sf_path}")
            with sf_path.open() as f:
                self._selected_features[split_id] = json.load(f)

            # Load optional artifacts
            config_dir = task_dir / "config"
            fc_path = config_dir / "feature_cols.json"
            self._feature_cols_per_split[split_id] = json.load(fc_path.open()) if fc_path.exists() else []
            fg_path = config_dir / "feature_groups.json"
            self._feature_groups_per_split[split_id] = json.load(fg_path.open()) if fg_path.exists() else {}

            # Load test/train data via split_row_ids.json + data_prepared.parquet
            split_ids_path = fold_dir / "split_row_ids.json"
            if not split_ids_path.exists():
                raise FileNotFoundError(f"Split row IDs not found: {split_ids_path}")
            prepared_path = self.study.path / "data_prepared.parquet"
            if not prepared_path.exists():
                raise FileNotFoundError(f"Prepared data not found: {prepared_path}")

            with split_ids_path.open() as f:
                split_info: dict[str, Any] = json.load(f)
            row_id_col: str = split_info["row_id_col"]
            prepared_data = parquet_load(prepared_path)
            indexed = prepared_data.set_index(row_id_col)

            self._test_data[split_id] = indexed.loc[split_info["test_row_ids"]].reset_index()
            self._train_data[split_id] = indexed.loc[split_info["traindev_row_ids"]].reset_index()

        # Compute union of feature_cols across all outersplits
        all_feature_cols: set[str] = set()
        for split_id in self._models:
            split_fcols = self._feature_cols_per_split.get(split_id, [])
            if split_fcols:
                all_feature_cols.update(split_fcols)

        if all_feature_cols:
            self._feature_cols = sorted(all_feature_cols)
        else:
            self._feature_cols = self._config.get("feature_cols", [])

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
        row_id_col = self._config.get("prepared", {}).get("row_id_col") or self._config.get("row_id_col") or "row_id"

        all_preds = []
        all_rows = []

        for split_id in self._models:
            features = self._feature_cols_per_split.get(split_id) or self._selected_features[split_id]
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
        ml_type = MLType(self._config["ml_type"])
        if ml_type not in (MLType.BINARY, MLType.MULTICLASS):
            raise TypeError(
                f"predict_proba() is only available for classification and multiclass tasks, "
                f"but this study has ml_type='{ml_type}'."
            )
        row_id_col = self._config.get("prepared", {}).get("row_id_col") or self._config.get("row_id_col") or "row_id"
        class_labels = next(iter(self._models.values())).classes_

        all_probas = []
        all_rows = []

        for split_id in self._models:
            features = self._feature_cols_per_split.get(split_id) or self._selected_features[split_id]
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
            metrics = [self._config.get("target_metric", "")]
        target_assignments = self._config.get("prepared", {}).get("target_assignments", {})

        rows = []
        for split_id in self._models:
            model = self._models[split_id]
            features = self._feature_cols_per_split.get(split_id) or self._selected_features[split_id]
            test = self._test_data[split_id]

            for metric_name in metrics:
                score = get_performance_from_model(
                    model=model,
                    data=test,
                    feature_cols=features,
                    target_metric=metric_name,
                    target_assignments=target_assignments,
                    threshold=threshold,
                    positive_class=self._config.get("positive_class"),
                )
                rows.append({"outersplit": split_id, "metric": metric_name, "score": score})

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
        from octopus.feature_importance import dispatch_fi  # noqa: PLC0415

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
            feature_groups_per_split=self._feature_groups_per_split,
            fi_type=fi_type,
            n_repeats=n_repeats,
            feature_groups=feature_groups,
            random_state=random_state,
            ml_type=MLType(self._config["ml_type"]),
            **kwargs,
        )

