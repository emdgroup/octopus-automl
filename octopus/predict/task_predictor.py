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
from octopus.predict.study_io import StudyInfo
from octopus.types import FIType, MLType, ResultType
from octopus.utils import get_version, joblib_load, joblib_save, parquet_load


@define(slots=False)
class TaskPredictor:
    """Ensemble model for predicting on new, unseen data.

    Wraps the fitted models from a single task across all outer splits.
    All methods require **explicit data** — no test/train data is stored.
    All results are computed fresh from loaded models.

    Args:
        study: Validated study (from ``load_study()``).
        task_id: Concrete workflow task index (must be >= 0).
        result_type: Result type for filtering results (default: 'best').

    Raises:
        ValueError: If task_id is out of range or no models found.
        FileNotFoundError: If expected model artifacts are missing.

    Example:
        >>> study = load_study("studies/my_study")
        >>> tp = TaskPredictor(study, task_id=0)
        >>> predictions = tp.predict(new_data, df=True)
    """

    study: StudyInfo = field()
    """Validated study (from ``load_study()``)."""

    task_id: int = field()
    """Workflow task index (>= 0)."""

    result_type: ResultType = field(default=ResultType.BEST, converter=ResultType)
    """Result type: 'best' or 'ensemble_selection'."""

    def __attrs_post_init__(self) -> None:
        """Load models and per-split artifacts from the study directory."""
        self._config = self.study.config
        self._models: dict[int, Any] = {}
        self._selected_features: dict[int, list[str]] = {}
        self._feature_cols_per_split: dict[int, list[str]] = {}
        self._feature_groups_per_split: dict[int, dict[str, list[str]]] = {}

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

        for split_id in self._models:
            features = self._feature_cols_per_split.get(split_id) or self._selected_features[split_id]
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
        ml_type = MLType(self._config["ml_type"])
        if ml_type not in (MLType.BINARY, MLType.MULTICLASS):
            raise TypeError(
                f"predict_proba() is only available for classification and multiclass tasks, "
                f"but this study has ml_type='{ml_type}'."
            )
        per_split_probas: list[np.ndarray] = []
        all_rows: list[pd.DataFrame] = []
        class_labels = next(iter(self._models.values())).classes_

        for split_id in self._models:
            features = self._feature_cols_per_split.get(split_id) or self._selected_features[split_id]
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
            metrics = [self._config.get("target_metric", "")]
        target_assignments = self._config.get("prepared", {}).get("target_assignments", {})

        rows = []
        for split_id in self._models:
            model = self._models[split_id]
            features = self._feature_cols_per_split.get(split_id) or self._selected_features[split_id]

            for metric_name in metrics:
                score = get_performance_from_model(
                    model=model,
                    data=data,
                    feature_cols=features,
                    target_metric=metric_name,
                    target_assignments=target_assignments,
                    threshold=threshold,
                    positive_class=self._config.get("positive_class"),
                )
                rows.append({"outersplit": split_id, "metric": metric_name, "score": score})

        return pd.DataFrame(rows)

    def _build_pool_data(self, data: pd.DataFrame) -> dict[int, pd.DataFrame]:
        """Build per-split pool data for permutation FI.

        In study-connected mode (constructed via ``TaskPredictor(study,
        task_id)``), loads per-split train/dev data from the study directory
        via ``split_row_ids.json``.  This provides a richer pool of
        replacement values for permutation FI, better approximating the
        marginal distribution of each feature.

        In deployment mode (loaded via ``TaskPredictor.load(path)``), the
        original study directory is not available, so the user-provided
        ``data`` is used as the pool for all splits.

        Args:
            data: User-provided data (used as fallback for all splits).

        Returns:
            Dict mapping outersplit_id to pool DataFrame.
        """
        pool: dict[int, pd.DataFrame] = {}

        for split_id in self._models:
            if self.study is not None:
                split_ids_path = self.study.path / f"outersplit{split_id}" / "split_row_ids.json"
                prepared_path = self.study.path / "data_prepared.parquet"
                if split_ids_path.exists() and prepared_path.exists():
                    with split_ids_path.open() as f:
                        split_info = json.load(f)
                    row_id_col = split_info["row_id_col"]
                    prepared = parquet_load(prepared_path)
                    indexed = prepared.set_index(row_id_col)
                    pool[split_id] = indexed.loc[split_info["traindev_row_ids"]].reset_index()
                    continue
            pool[split_id] = data

        return pool

    def calculate_fi(
        self,
        data: pd.DataFrame,
        fi_type: FIType = FIType.PERMUTATION,
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
                  kwarg to select the explainer: ``"kernel"`` (default),
                  ``"permutation"``, or ``"exact"``.
            n_repeats: Number of permutation repeats (for permutation FI).
            feature_groups: Dict mapping group names to feature lists
                (for group_permutation).  If None and fi_type is
                ``FIType.GROUP_PERMUTATION``, groups are loaded from the study.
            random_state: Random seed.
            **kwargs: Additional keyword arguments passed to the FI function.
                For ``fi_type=FIType.SHAP``, supported kwargs include:
                ``shap_type`` (``"kernel"``, ``"permutation"``, ``"exact"``),
                ``max_samples``, ``background_size``.

        Returns:
            DataFrame with feature importance results including a ``fi_type``
            column and per-split + ensemble rows.

        Raises:
            ValueError: If fi_type is unknown.
        """
        from octopus.feature_importance import dispatch_fi  # noqa: PLC0415

        fi_type = FIType(fi_type)

        # Build per-split data dicts
        test_data = dict.fromkeys(self._models, data)
        train_data = self._build_pool_data(data)

        return dispatch_fi(
            models=self._models,
            selected_features=self._selected_features,
            test_data=test_data,
            train_data=train_data,
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

        # Save study config (the real one — all metadata comes from here)
        with (save_dir / "study_config.json").open("w") as f:
            json.dump(self._config, f, indent=2, default=str)

        # Save predictor state (per-split artifacts not in the study config)
        state = {
            "task_id": self.task_id,
            "result_type": self.result_type,
            "outersplits": list(self._models.keys()),
            "feature_cols": self._feature_cols,
            "feature_cols_per_split": {str(k): v for k, v in self._feature_cols_per_split.items()},
            "feature_groups_per_split": {str(k): v for k, v in self._feature_groups_per_split.items()},
        }
        with (save_dir / "predictor_state.json").open("w") as f:
            json.dump(state, f, indent=2)

        # Save models
        models_dir = save_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        for split_id in self._models:
            joblib_save(self._models[split_id], models_dir / f"model_{split_id:03d}.joblib")

        # Save selected features
        features_dir = save_dir / "selected_features"
        features_dir.mkdir(parents=True, exist_ok=True)
        for split_id in self._models:
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

        # Load study config
        with (load_dir / "study_config.json").open() as f:
            config = json.load(f)

        # Load predictor state
        with (load_dir / "predictor_state.json").open() as f:
            state = json.load(f)

        instance = TaskPredictor.__new__(TaskPredictor)

        instance.study = None
        instance.task_id = state["task_id"]
        instance.result_type = ResultType(state.get("result_type", "best"))
        instance._config = config
        instance._feature_cols = state.get("feature_cols", [])
        instance._feature_cols_per_split = {int(k): v for k, v in state.get("feature_cols_per_split", {}).items()}
        instance._feature_groups_per_split = {int(k): v for k, v in state.get("feature_groups_per_split", {}).items()}

        # Load models
        outersplits = state.get("outersplits", [])
        instance._models = {}
        models_dir = load_dir / "models"
        for split_id in outersplits:
            instance._models[split_id] = joblib_load(models_dir / f"model_{split_id:03d}.joblib")

        # Load selected features
        instance._selected_features = {}
        features_dir = load_dir / "selected_features"
        for split_id in outersplits:
            with (features_dir / f"split_{split_id:03d}.json").open() as f:
                instance._selected_features[split_id] = json.load(f)

        return instance
