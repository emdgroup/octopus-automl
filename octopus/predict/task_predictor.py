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
from octopus.predict.study_io import StudyLoader, StudyMetadata, _to_upath
from octopus.types import FIType, MLType
from octopus.utils import get_version, joblib_load, joblib_save


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
    _target_col_resolved: str = field(init=False, default="")

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

        # Cache resolved target column name
        self._target_col_resolved = self._resolve_target_col()

    # ── Properties ──────────────────────────────────────────────

    @property
    def ml_type(self) -> MLType:
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
        """Full study configuration dictionary.

        Note:
            After ``TaskPredictor.load()``, this returns an empty dict
            because the full config is not serialized — only the metadata
            fields needed for prediction are saved.
        """
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
        """Predict on new data using all outer-split models.

        Args:
            data: DataFrame containing feature columns.
            df: If True, return a DataFrame with per-outersplit predictions
                and ensemble (averaged) predictions, with columns
                ``outersplit``, ``row_id``, ``prediction``.
                If False (default), return ensemble-averaged ndarray.

        Returns:
            Ensemble-averaged predictions as ndarray, or a DataFrame with
            per-split and ensemble rows when ``df=True``.
        """
        per_split_preds: list[np.ndarray] = []
        all_rows: list[pd.DataFrame] = []

        for split_id in self._outersplits:
            features = self._selected_features[split_id]
            preds = self._models[split_id].predict(data[features])
            per_split_preds.append(preds)

            if df:
                split_df = pd.DataFrame(
                    {
                        "outersplit": split_id,
                        "row_id": data.index,
                        "prediction": preds,
                    }
                )
                all_rows.append(split_df)

        ensemble: np.ndarray = np.mean(per_split_preds, axis=0)

        if df:
            ensemble_df = pd.DataFrame(
                {
                    "outersplit": "ensemble",
                    "row_id": data.index,
                    "prediction": ensemble,
                }
            )
            all_rows.append(ensemble_df)
            return pd.concat(all_rows, ignore_index=True)
        return ensemble

    def predict_proba(self, data: pd.DataFrame, df: bool = False) -> np.ndarray | pd.DataFrame:
        """Predict probabilities on new data (classification/multiclass only).

        Args:
            data: DataFrame containing feature columns.
            df: If True, return a DataFrame with per-outersplit probabilities
                and ensemble (averaged) probabilities, with columns
                ``outersplit``, ``row_id``, plus one column per class label.
                If False (default), return ensemble-averaged ndarray.

        Returns:
            Ensemble-averaged probabilities as ndarray, or a DataFrame with
            per-split and ensemble rows when ``df=True``.

        Raises:
            TypeError: If ml_type is not classification or multiclass.
        """
        if self.ml_type not in (MLType.BINARY, MLType.MULTICLASS):
            raise TypeError(
                f"predict_proba() is only available for classification and multiclass tasks, "
                f"but this study has ml_type='{self.ml_type}'."
            )
        per_split_probas: list[np.ndarray] = []
        all_rows: list[pd.DataFrame] = []
        class_labels = self.classes_

        for split_id in self._outersplits:
            features = self._selected_features[split_id]
            probas = self._models[split_id].predict_proba(data[features])
            if isinstance(probas, pd.DataFrame):
                probas = probas.values
            per_split_probas.append(probas)

            if df:
                split_df = pd.DataFrame(probas, columns=class_labels)
                split_df.insert(0, "outersplit", split_id)
                split_df.insert(1, "row_id", data.index.values)
                all_rows.append(split_df)

        ensemble: np.ndarray = np.mean(per_split_probas, axis=0)

        if df:
            ensemble_df = pd.DataFrame(ensemble, columns=class_labels)
            ensemble_df.insert(0, "outersplit", "ensemble")
            ensemble_df.insert(1, "row_id", data.index.values)
            all_rows.append(ensemble_df)
            return pd.concat(all_rows, ignore_index=True)
        return ensemble

    # ── Scoring ─────────────────────────────────────────────────

    def _resolve_target_col(self) -> str:
        """Resolve the actual target column name from assignments or config.

        Called once during init/load and cached in ``_target_col_resolved``.

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

        target_col = self._target_col_resolved

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
        fi_type: FIType | str = FIType.PERMUTATION,
        *,
        n_repeats: int = 10,
        feature_groups: dict[str, list[str]] | None = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Calculate feature importance on provided data across all outer splits.

        Computes FI fresh from loaded models, providing p-values,
        confidence intervals, and group permutation support.

        Args:
            data: Data to compute FI on (must contain features + target).
            fi_type: Type of feature importance. One of:
                - ``FIType.PERMUTATION`` — Per-feature permutation importance.
                - ``FIType.GROUP_PERMUTATION`` — Per-feature + per-group permutation
                  importance.  Uses ``feature_groups`` (from study config or
                  explicitly provided) to also compute group-level importance.
                - ``FIType.SHAP`` — SHAP-based importance.  Pass ``shap_type`` as a
                  kwarg to select the explainer: ``ShapExplainerType.KERNEL`` (default),
                  ``ShapExplainerType.PERMUTATION``, or ``ShapExplainerType.EXACT``.
            n_repeats: Number of permutation repeats (for permutation FI).
            feature_groups: Dict mapping group names to feature lists
                (for group_permutation).  If None and fi_type is
                ``FIType.GROUP_PERMUTATION``, groups are loaded from the study.
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
        target_col = self._target_col_resolved

        # Build per-split data dicts (same data for all splits)
        test_data = dict.fromkeys(self._outersplits, data)
        train_data = dict.fromkeys(self._outersplits, data)

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
        elif fi_type == FIType.SHAP:
            result = calculate_fi_shap(
                models=self._models,
                selected_features=self._selected_features,
                test_data=test_data,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown fi_type '{fi_type}'. Use FIType.PERMUTATION, FIType.GROUP_PERMUTATION, or FIType.SHAP."
            )

        result.insert(0, "fi_type", fi_type)
        return result

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
            joblib_save(self._models[split_id], models_dir / f"model_{split_id:03d}.joblib")

        # Save selected features
        features_dir = save_dir / "selected_features"
        features_dir.mkdir(parents=True, exist_ok=True)
        for split_id in self._outersplits:
            with (features_dir / f"split_{split_id:03d}.json").open("w") as f:
                json.dump(self._selected_features[split_id], f)

        # Save version info
        with (save_dir / "version.json").open("w") as f:
            json.dump({"octopus_version": get_version()}, f, indent=2)

    @classmethod
    def load(cls, path: str | UPath) -> TaskPredictor:
        """Load a previously saved predictor.

        Args:
            path: Directory path containing the saved predictor.

        Returns:
            A new TaskPredictor instance that can predict without the
            original study directory.
        """
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
            current_version = get_version()
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
            instance._models[split_id] = joblib_load(models_dir / f"model_{split_id:03d}.joblib")

        # Load selected features
        instance._selected_features = {}
        features_dir = load_dir / "selected_features"
        for split_id in instance._outersplits:
            with (features_dir / f"split_{split_id:03d}.json").open() as f:
                instance._selected_features[split_id] = json.load(f)

        # Cache resolved target column name
        instance._target_col_resolved = instance._resolve_target_col()

        return instance
