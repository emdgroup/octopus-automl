"""TaskPredictor — unified predictor for a single task across all outer splits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from upath import UPath

from octopus.metrics.utils import get_performance_from_model
from octopus.predict.study_io import OuterSplitLoader, StudyLoader


class TaskPredictor:
    """Unified predictor for a single task across all outer splits.

    Usage:

        tp = TaskPredictor(study_path, task_id)
        tp.predict(data)

    Args:
        study_path: Path to the study directory.
        task_id: Task ID to load. Use -1 for the last task in the workflow.
        module: Module name for filtering results (default: 'octo').
        result_type: Result type for filtering results (default: 'best').

    Raises:
        ValueError: If no models are found for the specified task.

    Example:
        >>> tp = TaskPredictor("studies/my_study", task_id=2)
        >>> scores = tp.performance_test(metrics=["AUCROC", "ACC"])
    """

    def __init__(
        self,
        study_path: str | UPath,
        task_id: int = -1,
        module: str = "octo",
        result_type: str = "best",
    ) -> None:
        self._study_path = UPath(study_path)
        self._module_name = module
        self._result_type = result_type

        # Load config
        loader = StudyLoader(self._study_path)
        self._config = loader.load_config()

        # Resolve task_id (-1 -> last task)
        if task_id < 0:
            task_id = len(self._config["workflow"]) - 1
        self._task_id = task_id

        # Extract config values
        self._ml_type: str = self._config.get("ml_type", "")
        self._target_metric: str = self._config.get("target_metric", "")
        self._target_col: str = self._config.get("target_col", "")
        self._target_assignments: dict[str, str] = self._config.get("prepared", {}).get("target_assignments", {})
        self._positive_class: Any = self._config.get("positive_class")
        self._row_id_col: str | None = self._config.get("prepared", {}).get("row_id_col")
        if not self._row_id_col:
            self._row_id_col = self._config.get("row_id_col") or "row_id"

        # Load per-outersplit data
        self._outersplits: list[int] = []
        self._models: dict[int, Any] = {}
        self._selected_features: dict[int, list[str]] = {}
        self._feature_cols_per_split: dict[int, list[str]] = {}
        self._feature_groups_per_split: dict[int, dict[str, list[str]]] = {}
        self._test_data: dict[int, pd.DataFrame] = {}
        self._train_data: dict[int, pd.DataFrame] = {}

        n_outersplits = self._config.get("n_folds_outer", 0)
        for split_id in range(n_outersplits):
            try:
                split_loader = OuterSplitLoader(
                    self._study_path,
                    split_id,
                    self._task_id,
                    module,
                    result_type,
                )
                if not split_loader.has_model():
                    continue

                self._outersplits.append(split_id)
                self._models[split_id] = split_loader.load_model()
                self._selected_features[split_id] = split_loader.load_selected_features()
                self._feature_cols_per_split[split_id] = split_loader.load_feature_cols()
                self._feature_groups_per_split[split_id] = split_loader.load_feature_groups()
                self._test_data[split_id] = split_loader.load_test_data()
                self._train_data[split_id] = split_loader.load_train_data()
            except (FileNotFoundError, OSError):
                continue

        if not self._outersplits:
            raise ValueError(f"No models found for task {task_id}. Check that the study has been run.")

        # Compute union of feature_cols across all outersplits.
        # Falls back to study config if per-split files are not available.
        all_feature_cols: set[str] = set()
        for split_id in self._outersplits:
            split_fcols = self._feature_cols_per_split.get(split_id, [])
            if split_fcols:
                all_feature_cols.update(split_fcols)

        if all_feature_cols:
            self._feature_cols: list[str] = sorted(all_feature_cols)
        else:
            # Fallback: use feature_cols from study config
            self._feature_cols = self._config.get("feature_cols", [])

        # FI results cache
        self._fi_results: dict[str, pd.DataFrame] = {}

    # ── Properties ──────────────────────────────────────────────

    @property
    def ml_type(self) -> str:
        """Machine learning type (classification, regression, timetoevent)."""
        return self._ml_type

    @property
    def target_metric(self) -> str:
        """Target metric name."""
        return self._target_metric

    @property
    def target_col(self) -> str:
        """Target column name from config."""
        return self._target_col

    @property
    def target_assignments(self) -> dict[str, str]:
        """Target column assignments from prepared config."""
        return self._target_assignments

    @property
    def positive_class(self) -> Any:
        """Positive class label for classification."""
        return self._positive_class

    @property
    def row_id_col(self) -> str | None:
        """Row ID column name."""
        return self._row_id_col

    @property
    def feature_cols(self) -> list[str]:
        """Input feature column names from study config."""
        return self._feature_cols

    @property
    def n_outersplits(self) -> int:
        """Number of loaded outersplits."""
        return len(self._outersplits)

    @property
    def outersplits(self) -> list[int]:
        """List of loaded outersplit IDs."""
        return list(self._outersplits)

    @property
    def config(self) -> dict[str, Any]:
        """Full study configuration dictionary."""
        return self._config

    @property
    def classes_(self) -> np.ndarray:
        """Class labels from the first model (classification only).

        Raises:
            AttributeError: If the model does not have a classes_ attribute.
        """
        model = self._models[self._outersplits[0]]
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

    @property
    def fi_results(self) -> dict[str, pd.DataFrame]:
        """Cached feature importance results keyed by fi_type."""
        return self._fi_results

    # ── Per-outersplit access ───────────────────────────────────

    def get_model(self, outersplit_id: int) -> Any:
        """Get the fitted model for an outersplit.

        Args:
            outersplit_id: Outer split index.

        Returns:
            The fitted model object.
        """
        return self._models[outersplit_id]

    def get_selected_features(self, outersplit_id: int) -> list[str]:
        """Get selected features for an outersplit.

        Args:
            outersplit_id: Outer split index.

        Returns:
            List of selected feature names.
        """
        return self._selected_features[outersplit_id]

    def get_test_data(self, outersplit_id: int) -> pd.DataFrame:
        """Get test data for an outersplit.

        Args:
            outersplit_id: Outer split index.

        Returns:
            Test data DataFrame.
        """
        return self._test_data[outersplit_id]

    def get_train_data(self, outersplit_id: int) -> pd.DataFrame:
        """Get train data for an outersplit.

        Args:
            outersplit_id: Outer split index.

        Returns:
            Train data DataFrame.
        """
        return self._train_data[outersplit_id]

    # ── Prediction ──────────────────────────────────────────────

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict on new data (mean across outer splits).

        Args:
            data: DataFrame containing feature columns.

        Returns:
            Array of predictions averaged across outer splits.
        """
        preds = []
        for split_id in self._outersplits:
            features = self._selected_features[split_id]
            preds.append(self._models[split_id].predict(data[features]))
        return np.mean(preds, axis=0)  # type: ignore[no-any-return]

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Predict probabilities on new data (mean across outer splits).

        Args:
            data: DataFrame containing feature columns.

        Returns:
            Array of predicted probabilities averaged across outer splits.
        """
        probas = []
        for split_id in self._outersplits:
            features = self._selected_features[split_id]
            probas.append(self._models[split_id].predict_proba(data[features]))
        return np.mean(probas, axis=0)  # type: ignore[no-any-return]

    # ── Scoring ─────────────────────────────────────────────────

    def _resolve_target_col(self) -> str:
        """Resolve the actual target column name from assignments or config.

        Returns:
            The target column name to use for scoring.
        """
        if self._target_assignments:
            return list(self._target_assignments.values())[0]
        return self._target_col

    def performance_test(
        self,
        metrics: list[str] | None = None,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Score test predictions per outer split.

        Args:
            metrics: List of metric names to evaluate.
                If None, uses the study target metric.
            threshold: Classification threshold for binary prediction
                metrics (e.g., ACC, F1). Probability-based metrics
                (e.g., AUCROC, AUCPR) are unaffected. Default: 0.5.

        Returns:
            DataFrame with columns: outersplit, metric, score.
        """
        if metrics is None:
            metrics = [self._target_metric]

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
                    positive_class=self._positive_class,
                )
                rows.append({"outersplit": split_id, "metric": metric_name, "score": score})

        return pd.DataFrame(rows)

    # ── Feature Importance ──────────────────────────────────────

    def calculate_fi(
        self,
        fi_type: str = "permutation",
        *,
        n_repeats: int = 10,
        feature_groups: dict[str, list[str]] | None = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        """Calculate feature importance across all outer splits.

        Computes FI fresh from loaded models, providing p-values,
        confidence intervals, and group permutation support.  Results are
        stored in the ``fi_results`` cache (keyed by *fi_type*) and can be
        retrieved via ``self.fi_results[fi_type]``.

        Args:
            fi_type: Type of feature importance. One of 'permutation',
                'group_permutation', or 'shap'.
            n_repeats: Number of permutation repeats (for permutation FI).
            feature_groups: Dict mapping group names to feature lists
                (for group_permutation).
            random_state: Random seed.
            **kwargs: Additional keyword arguments passed to the FI function.

        Raises:
            ValueError: If fi_type is unknown.
        """
        from octopus.predict.feature_importance import (  # noqa: PLC0415
            calculate_fi_permutation,
            calculate_fi_shap,
        )

        target_col = self._resolve_target_col()

        if fi_type in ("permutation", "group_permutation"):
            # Auto-compute feature groups from training data if not provided
            resolved_groups = None
            if fi_type == "group_permutation":
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
                target_metric=self._target_metric,
                positive_class=self._positive_class,
                n_repeats=n_repeats,
                random_state=random_state,
                feature_groups=resolved_groups,
                feature_cols=self._feature_cols,
            )
        elif fi_type == "shap":
            result = calculate_fi_shap(
                models=self._models,
                selected_features=self._selected_features,
                test_data=self._test_data,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown fi_type '{fi_type}'. Use 'permutation', 'group_permutation', or 'shap'.")

        self._fi_results[fi_type] = result

    def _compute_feature_groups(self) -> dict[str, list[str]]:
        """Compute merged feature groups from all outersplits.

        Merges the per-split feature groups loaded from disk into a single
        dict. Groups with the same name across splits are merged by taking
        the union of their features.

        Returns:
            Dict mapping group names to lists of feature names.
        """
        all_groups: dict[str, list[str]] = {}
        for split_id in self._outersplits:
            split_groups = self._feature_groups_per_split.get(split_id, {})
            for group_name, group_features in split_groups.items():
                if group_name in all_groups:
                    existing = set(all_groups[group_name])
                    existing.update(group_features)
                    all_groups[group_name] = sorted(existing)
                else:
                    all_groups[group_name] = sorted(group_features)
        return all_groups
