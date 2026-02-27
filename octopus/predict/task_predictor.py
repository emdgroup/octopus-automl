"""TaskPredictor — ensemble model for predicting on new, unseen data.

Wraps the fitted models from a single task within an octopus study.
The caller always provides data explicitly.  Stores models + metadata only
(no test/train data) to enable lightweight save/load for deployment.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from attrs import define, field
from upath import UPath

from octopus.metrics.utils import get_performance_from_model
from octopus.predict.study_io import StudyLoader, StudyMetadata


def _get_octopus_version() -> str:
    """Get the current octopus package version, or 'unknown' if unavailable."""
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return version("octopus-automl")
    except Exception:
        return "unknown"


def _to_upath(value: str | UPath) -> UPath:
    """Convert a string or UPath to UPath."""
    return UPath(value)


@define(slots=False)
class TaskPredictor:
    """Ensemble model for predicting on new, unseen data.

    Wraps the fitted models from a single task across all outer splits.
    All methods require **explicit data** — no test/train data is stored.
    All results are computed fresh from loaded models.

    Args:
        study_path: Path to the study directory.
        task_id: Concrete workflow task index (must be >= 0).
        result_type: Result type for filtering results (default: 'best').

    Raises:
        ValueError: If task_id is negative, out of range, or no models found.
        FileNotFoundError: If expected study artifacts are missing.

    Example:
        >>> tp = TaskPredictor("studies/my_study", task_id=0)
        >>> predictions = tp.predict(new_data, df=True)
    """

    _study_path: UPath = field(converter=_to_upath, alias="study_path")
    _task_id: int = field(alias="task_id")
    _result_type: str = field(default="best", alias="result_type")

    # Computed fields — populated in __attrs_post_init__
    _config: dict[str, Any] = field(init=False, factory=dict, repr=False)
    _metadata: StudyMetadata = field(init=False)

    # Flattened from artifacts for fast access
    _outersplits: list[int] = field(init=False, factory=list)
    _models: dict[int, Any] = field(init=False, factory=dict, repr=False)
    _selected_features: dict[int, list[str]] = field(init=False, factory=dict, repr=False)
    _feature_cols_per_split: dict[int, list[str]] = field(init=False, factory=dict, repr=False)
    _feature_groups_per_split: dict[int, dict[str, list[str]]] = field(init=False, factory=dict, repr=False)
    _feature_cols: list[str] = field(init=False, factory=list, repr=False)

    # FI results cache
    _fi_results: dict[str, pd.DataFrame] = field(init=False, factory=dict, repr=False)

    def __attrs_post_init__(self) -> None:
        """Load config, validate, and load artifacts from the study directory."""
        loader = StudyLoader(self._study_path)
        self._config = loader.load_config()

        # Validate task_id via I/O layer
        loader.validate_task_id(self._task_id, self._config)

        # Extract metadata via I/O layer
        self._metadata = loader.extract_metadata(self._config)

        # Load all per-split artifacts via I/O layer
        artifacts = loader.load_task_artifacts(
            self._task_id,
            self._result_type,
            self._metadata.n_outersplits,
        )

        # Flatten artifacts for fast per-split access
        self._outersplits = list(artifacts.outersplit_ids)
        for split_id, sa in artifacts.splits.items():
            self._models[split_id] = sa.model
            self._selected_features[split_id] = sa.selected_features
            self._feature_cols_per_split[split_id] = sa.feature_cols
            self._feature_groups_per_split[split_id] = sa.feature_groups

        # Compute union of feature_cols across all outersplits
        all_feature_cols: set[str] = set()
        for split_id in self._outersplits:
            split_fcols = self._feature_cols_per_split.get(split_id, [])
            if split_fcols:
                all_feature_cols.update(split_fcols)

        if all_feature_cols:
            self._feature_cols = sorted(all_feature_cols)
        else:
            self._feature_cols = self._metadata.feature_cols

    # ── Properties ──────────────────────────────────────────────

    @property
    def ml_type(self) -> str:
        """Machine learning type (classification, regression, timetoevent)."""
        return self._metadata.ml_type

    @property
    def target_metric(self) -> str:
        """Target metric name."""
        return self._metadata.target_metric

    @property
    def target_col(self) -> str:
        """Target column name from config."""
        return self._metadata.target_col

    @property
    def target_assignments(self) -> dict[str, str]:
        """Target column assignments from prepared config."""
        return self._metadata.target_assignments

    @property
    def positive_class(self) -> Any:
        """Positive class label for classification."""
        return self._metadata.positive_class

    @property
    def row_id_col(self) -> str | None:
        """Row ID column name."""
        return self._metadata.row_id_col

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

    # ── Prediction ──────────────────────────────────────────────

    def predict(self, data: pd.DataFrame, df: bool = False) -> np.ndarray | pd.DataFrame:
        """Predict on new data by averaging predictions across all outer-split models.

        Args:
            data: DataFrame containing feature columns.
            df: If True, return a DataFrame with a ``prediction`` column.
                If False (default), return an ndarray.

        Returns:
            Ensemble-averaged predictions as ndarray or DataFrame.
        """
        preds = []
        for split_id in self._outersplits:
            features = self._selected_features[split_id]
            preds.append(self._models[split_id].predict(data[features]))
        result: np.ndarray = np.mean(preds, axis=0)

        if df:
            return pd.DataFrame({"prediction": result}, index=data.index)
        return result

    def predict_proba(self, data: pd.DataFrame, df: bool = False) -> np.ndarray | pd.DataFrame:
        """Predict probabilities on new data (classification only).

        Ensemble average across all outer-split models.

        Args:
            data: DataFrame containing feature columns.
            df: If True, return a DataFrame with class-name columns.
                If False (default), return an ndarray.

        Returns:
            Ensemble-averaged probabilities as ndarray or DataFrame.
        """
        probas = []
        for split_id in self._outersplits:
            features = self._selected_features[split_id]
            probas.append(self._models[split_id].predict_proba(data[features]))
        result: np.ndarray = np.mean(probas, axis=0)

        if df:
            class_labels = self.classes_
            return pd.DataFrame(result, columns=class_labels, index=data.index)
        return result

    # ── Scoring ─────────────────────────────────────────────────

    def _resolve_target_col(self) -> str:
        """Resolve the actual target column name from assignments or config.

        Returns:
            The target column name to use for scoring.
        """
        if self.target_assignments:
            return list(self.target_assignments.values())[0]
        return self.target_col

    def performance(
        self,
        data: pd.DataFrame,
        metrics: list[str] | None = None,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Compute performance scores on provided data for each outer split.

        Each outer-split model is scored independently on the **same** data.
        Scores are computed fresh — never read from disk.

        Args:
            data: Data to score on; must contain feature columns + target column.
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

            for metric_name in metrics:
                target_assignments = {target_col: target_col}
                score = get_performance_from_model(
                    model=model,
                    data=data,
                    feature_cols=features,
                    target_metric=metric_name,
                    target_assignments=target_assignments,
                    threshold=threshold,
                    positive_class=self.positive_class,
                )
                rows.append({"outersplit": split_id, "metric": metric_name, "score": score})

        return pd.DataFrame(rows)

    # ── Feature Importance ──────────────────────────────────────

    def calculate_fi(
        self,
        data: pd.DataFrame,
        fi_type: str = "permutation",
        *,
        n_repeats: int = 10,
        feature_groups: dict[str, list[str]] | None = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        """Calculate feature importance on provided data across all outer splits.

        Computes FI fresh from loaded models, providing p-values,
        confidence intervals, and group permutation support.  Results are
        stored in the ``fi_results`` cache (keyed by *fi_type*) and can be
        retrieved via ``self.fi_results[fi_type]``.

        Args:
            data: Data to compute FI on (must contain features + target).
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

        # Build per-split data dicts (same data for all splits)
        test_data = dict.fromkeys(self._outersplits, data)
        train_data = dict.fromkeys(self._outersplits, data)

        if fi_type in ("permutation", "group_permutation"):
            resolved_groups = None
            if fi_type == "group_permutation":
                if feature_groups is not None:
                    resolved_groups = feature_groups
                else:
                    resolved_groups = self._compute_feature_groups()

            result = calculate_fi_permutation(
                models=self._models,
                selected_features=self._selected_features,
                test_data=test_data,
                train_data=train_data,
                target_col=target_col,
                target_metric=self.target_metric,
                positive_class=self.positive_class,
                n_repeats=n_repeats,
                random_state=random_state,
                feature_groups=resolved_groups,
                feature_cols=self._feature_cols,
            )
        elif fi_type == "shap":
            result = calculate_fi_shap(
                models=self._models,
                selected_features=self._selected_features,
                test_data=test_data,
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

    # ── Serialization ───────────────────────────────────────────

    def save(self, path: str | UPath) -> None:
        """Save the predictor for standalone deployment.

        Writes a self-contained directory with models + metadata only
        (no data). The saved predictor can be loaded later without the
        original study directory.

        Args:
            path: Directory path to save to. Created if it doesn't exist.
        """
        import joblib  # noqa: PLC0415

        save_dir = UPath(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "task_id": self._task_id,
            "ml_type": self.ml_type,
            "target_metric": self.target_metric,
            "target_col": self.target_col,
            "target_assignments": self.target_assignments,
            "positive_class": self.positive_class,
            "row_id_col": self.row_id_col,
            "feature_cols": self._feature_cols,
            "outersplits": self._outersplits,
            "result_type": self._result_type,
            "feature_cols_per_split": {str(k): v for k, v in self._feature_cols_per_split.items()},
            "feature_groups_per_split": {str(k): v for k, v in self._feature_groups_per_split.items()},
        }
        with (save_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save models
        models_dir = save_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        for split_id in self._outersplits:
            joblib.dump(self._models[split_id], models_dir / f"model_{split_id:03d}.joblib")

        # Save selected features
        features_dir = save_dir / "selected_features"
        features_dir.mkdir(parents=True, exist_ok=True)
        for split_id in self._outersplits:
            with (features_dir / f"split_{split_id:03d}.json").open("w") as f:
                json.dump(self._selected_features[split_id], f)

        # Save version info
        version = _get_octopus_version()
        with (save_dir / "version.json").open("w") as f:
            json.dump({"octopus_version": version}, f, indent=2)

    @classmethod
    def load(cls, path: str | UPath) -> TaskPredictor:
        """Load a previously saved predictor.

        Args:
            path: Directory path containing the saved predictor.

        Returns:
            A new TaskPredictor instance that can predict without the
            original study directory.
        """
        import joblib  # noqa: PLC0415

        load_dir = UPath(path)

        # Load metadata
        with (load_dir / "metadata.json").open() as f:
            metadata_dict = json.load(f)

        # Load version and warn if mismatch
        version_path = load_dir / "version.json"
        if version_path.exists():
            with version_path.open() as f:
                version_info = json.load(f)
            saved_version = version_info.get("octopus_version", "unknown")
            current_version = _get_octopus_version()
            if saved_version not in ("unknown", current_version):
                import warnings  # noqa: PLC0415

                warnings.warn(
                    f"Predictor was saved with octopus {saved_version}, "
                    f"but current version is {current_version}. "
                    f"Predictions may differ.",
                    stacklevel=2,
                )

        # Create instance without calling __init__ / __attrs_post_init__
        # Use TaskPredictor explicitly (not cls) to avoid subclass issues
        instance = TaskPredictor.__new__(TaskPredictor)

        instance._study_path = UPath(load_dir)
        instance._result_type = metadata_dict.get("result_type", "best")
        instance._task_id = metadata_dict["task_id"]
        instance._config = {}
        instance._metadata = StudyMetadata(
            ml_type=metadata_dict["ml_type"],
            target_metric=metadata_dict["target_metric"],
            target_col=metadata_dict["target_col"],
            target_assignments=metadata_dict.get("target_assignments", {}),
            positive_class=metadata_dict.get("positive_class"),
            row_id_col=metadata_dict.get("row_id_col"),
            feature_cols=metadata_dict.get("feature_cols", []),
            n_outersplits=len(metadata_dict.get("outersplits", [])),
        )
        instance._feature_cols = metadata_dict.get("feature_cols", [])
        instance._outersplits = metadata_dict.get("outersplits", [])
        instance._fi_results = {}

        # Restore per-split data
        instance._feature_cols_per_split = {
            int(k): v for k, v in metadata_dict.get("feature_cols_per_split", {}).items()
        }
        instance._feature_groups_per_split = {
            int(k): v for k, v in metadata_dict.get("feature_groups_per_split", {}).items()
        }

        # Load models
        instance._models = {}
        models_dir = load_dir / "models"
        for split_id in instance._outersplits:
            instance._models[split_id] = joblib.load(models_dir / f"model_{split_id:03d}.joblib")

        # Load selected features
        instance._selected_features = {}
        features_dir = load_dir / "selected_features"
        for split_id in instance._outersplits:
            with (features_dir / f"split_{split_id:03d}.json").open() as f:
                instance._selected_features[split_id] = json.load(f)

        return instance
