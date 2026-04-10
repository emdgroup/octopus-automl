"""OctoPredictor — ensemble model for predicting on new, unseen data.

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
from octopus.poststudy.study_io import StudyInfo, TaskOutersplitLoader
from octopus.types import FIType, MLType, ResultType
from octopus.utils import get_version, joblib_load, joblib_save, parquet_load


@define(slots=False)
class _PredictorBase:
    """Shared loading, fields, and typed properties for predictors.

    Not part of the public API. Both ``OctoPredictor`` and
    ``OctoTestEvaluator`` inherit from this class.

    Args:
        study_info: ``StudyInfo`` returned by ``load_study_information()``.
        task_id: Concrete workflow task index (must be >= 0).
        result_type: Result type for filtering results (default: 'best').
    """

    _study_info: StudyInfo = field(alias="study_info")
    _task_id: int = field(alias="task_id")
    _result_type: ResultType = field(default=ResultType.BEST, alias="result_type", converter=ResultType)

    _loaders: dict[int, TaskOutersplitLoader] = field(init=False, factory=dict, repr=False)
    _models: dict[int, Any] = field(init=False, factory=dict, repr=False)
    _feature_cols_per_split: dict[int, list[str]] = field(init=False, factory=dict, repr=False)
    _feature_groups_per_split: dict[int, dict[str, list[str]]] = field(init=False, factory=dict, repr=False)
    _feature_cols: list[str] = field(init=False, factory=list, repr=False)

    def __attrs_post_init__(self) -> None:
        """Validate task_id, load artifacts from the study directory."""
        if self._task_id < 0:
            raise ValueError(f"task_id must be >= 0, got {self._task_id}")
        if self._task_id >= len(self._study_info.workflow_tasks):
            raise ValueError(
                f"task_id {self._task_id} out of range, study has {len(self._study_info.workflow_tasks)} tasks"
            )
        if self._study_info.row_id_col is None:
            raise ValueError(
                "config['prepared']['row_id_col'] is missing. Studies created with an older version must be re-run."
            )

        for split_id in self._study_info.outersplits:
            loader = TaskOutersplitLoader(
                self._study_info.path,
                split_id,
                self._task_id,
                self._result_type,
            )
            self._loaders[split_id] = loader
            self._models[split_id] = loader.load_model()
            self._feature_cols_per_split[split_id] = loader.load_feature_cols()
            self._feature_groups_per_split[split_id] = loader.load_feature_groups()

        if not self._models:
            raise ValueError(f"No models found for task {self._task_id}.")

        all_feature_cols: set[str] = set()
        for split_fcols in self._feature_cols_per_split.values():
            if split_fcols:
                all_feature_cols.update(split_fcols)

        if all_feature_cols:
            self._feature_cols = sorted(all_feature_cols)
        else:
            self._feature_cols = self._study_info.feature_cols

    @property
    def study_info(self) -> StudyInfo:
        """Study metadata (ml_type, target_metric, target_col, etc.)."""
        return self._study_info

    @property
    def feature_cols(self) -> list[str]:
        """Union of input feature columns across all outersplits."""
        return self._feature_cols

    @property
    def classes_(self) -> np.ndarray:
        """Class labels from the first model (classification only).

        Raises:
            AttributeError: If the model does not have a classes_ attribute.
        """
        model = self._models[self._study_info.outersplits[0]]
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

    def get_model(self, outersplit_id: int) -> Any:
        """Get the fitted model for an outersplit.

        Args:
            outersplit_id: Outer split index.

        Returns:
            The fitted model object.
        """
        return self._models[outersplit_id]

    def _check_classification_only(self, method_name: str) -> None:
        """Raise TypeError if ml_type is not classification or multiclass."""
        if self._study_info.ml_type not in (MLType.BINARY, MLType.MULTICLASS):
            raise TypeError(
                f"{method_name}() is only available for classification and multiclass tasks, "
                f"but this study has ml_type='{self._study_info.ml_type}'."
            )

    def _compute_performance(
        self,
        data_per_split: dict[int, pd.DataFrame],
        metrics: list[str],
        threshold: float,
    ) -> pd.DataFrame:
        """Score each outer-split model on its corresponding data.

        Args:
            data_per_split: Dict mapping outersplit_id to scoring DataFrame.
            metrics: List of metric names to compute.
            threshold: Classification threshold for threshold-dependent metrics.

        Returns:
            DataFrame with columns: outersplit, metric, score.
        """
        rows = []
        for split_id in self._study_info.outersplits:
            model = self._models[split_id]
            features = self._feature_cols_per_split[split_id]
            for metric_name in metrics:
                score = get_performance_from_model(
                    model=model,
                    data=data_per_split[split_id],
                    feature_cols=features,
                    target_metric=metric_name,
                    target_assignments=self._study_info.target_assignments,
                    threshold=threshold,
                    positive_class=self._study_info.positive_class,
                )
                rows.append({"outersplit": split_id, "metric": metric_name, "score": score})
        return pd.DataFrame(rows)

    def _dispatch_fi(
        self,
        test_data: dict[int, pd.DataFrame],
        train_data: dict[int, pd.DataFrame],
        fi_type: FIType,
        *,
        n_repeats: int = 10,
        feature_groups: dict[str, list[str]] | None = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Forward to the stateless dispatch_fi with instance attributes."""
        from octopus.poststudy.feature_importance import dispatch_fi as _dispatch  # noqa: PLC0415

        return _dispatch(
            models=self._models,
            feature_cols_per_split=self._feature_cols_per_split,
            test_data=test_data,
            train_data=train_data,
            target_assignments=self._study_info.target_assignments,
            target_metric=self._study_info.target_metric,
            positive_class=self._study_info.positive_class,
            feature_cols=self._feature_cols,
            feature_groups_per_split=self._feature_groups_per_split,
            fi_type=fi_type,
            n_repeats=n_repeats,
            feature_groups=feature_groups,
            random_state=random_state,
            ml_type=self._study_info.ml_type,
            **kwargs,
        )


@define(slots=False)
class OctoPredictor(_PredictorBase):
    """Ensemble model for predicting on new, unseen data.

    Wraps the fitted models from a single task across all outer splits.
    All methods require **explicit data** — no test/train data is stored.
    All results are computed fresh from loaded models.

    Args:
        study_info: ``StudyInfo`` returned by ``load_study_information()``.
        task_id: Concrete workflow task index (must be >= 0).
        result_type: Result type for filtering results (default: 'best').

    Raises:
        ValueError: If task_id is negative, out of range, or no models found.
        FileNotFoundError: If expected study artifacts are missing.

    Example:
        >>> info = load_study_information("studies/my_study")
        >>> tp = OctoPredictor(study_info=info, task_id=0)
        >>> predictions = tp.predict(new_data, df=True)
    """

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

        for split_id in self._study_info.outersplits:
            features = self._feature_cols_per_split[split_id]
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
        self._check_classification_only("predict_proba")
        per_split_probas: list[np.ndarray] = []
        all_rows: list[pd.DataFrame] = []
        class_labels = self.classes_

        for split_id in self._study_info.outersplits:
            features = self._feature_cols_per_split[split_id]
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
            metrics = [self._study_info.target_metric]
        data_per_split = dict.fromkeys(self._study_info.outersplits, data)
        return self._compute_performance(data_per_split, metrics, threshold)

    def _build_pool_data(self, data: pd.DataFrame) -> dict[int, pd.DataFrame]:
        """Build per-split pool data for permutation FI.

        In study-connected mode, filters ``data_prepared.parquet`` using the
        stored split row IDs to recover per-split traindev data.  In deployment
        mode (loaded via ``OctoPredictor.load``), the user-provided ``data`` is
        used as the pool for all splits.

        Args:
            data: User-provided data (used as fallback for all splits).

        Returns:
            Dict mapping outersplit_id to pool DataFrame.
        """
        prepared_path = self._study_info.path / "data_prepared.parquet"

        preloaded: pd.DataFrame | None = None
        if prepared_path.exists():
            preloaded = parquet_load(prepared_path)

        pool: dict[int, pd.DataFrame] = {}
        for split_id in self._study_info.outersplits:
            loader = self._loaders.get(split_id)
            if loader is None:
                pool[split_id] = data
                continue
            try:
                pool[split_id] = loader.load_partition("traindev", preloaded)
            except (FileNotFoundError, KeyError):
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
        fi_type = FIType(fi_type)

        # All splits share the same DataFrame reference.  Safe because
        # compute_permutation_single / compute_shap_single copy data before mutating.
        test_data = dict.fromkeys(self._study_info.outersplits, data)
        train_data = self._build_pool_data(data)

        return self._dispatch_fi(
            test_data,
            train_data,
            fi_type,
            n_repeats=n_repeats,
            feature_groups=feature_groups,
            random_state=random_state,
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

        si = self._study_info
        metadata = {
            "task_id": self._task_id,
            "ml_type": si.ml_type,
            "target_metric": si.target_metric,
            "target_col": si.target_col,
            "target_assignments": si.target_assignments,
            "positive_class": si.positive_class,
            "row_id_col": si.row_id_col,
            "feature_cols": self._feature_cols,
            "outersplits": si.outersplits,
            "result_type": self._result_type,
            "feature_cols_per_split": {str(k): v for k, v in self._feature_cols_per_split.items()},
            "feature_groups_per_split": {str(k): v for k, v in self._feature_groups_per_split.items()},
        }
        with (save_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2, default=str)

        models_dir = save_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        for split_id in si.outersplits:
            joblib_save(self._models[split_id], models_dir / f"model_{split_id:03d}.joblib")

        with (save_dir / "version.json").open("w") as f:
            json.dump({"octopus_version": get_version()}, f, indent=2)

    @classmethod
    def load(cls, path: str | UPath) -> OctoPredictor:
        """Load a previously saved predictor.

        Args:
            path: Directory path containing the saved predictor.

        Returns:
            A new OctoPredictor instance that can predict without the
            original study directory.
        """
        load_dir = UPath(path)

        with (load_dir / "metadata.json").open() as f:
            metadata_dict = json.load(f)

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

        instance = OctoPredictor.__new__(OctoPredictor)

        try:
            instance._study_info = StudyInfo(
                path=UPath(load_dir),
                n_folds_outer=len(metadata_dict.get("outersplits", [])),
                workflow_tasks=(),
                outersplit_dirs=(),
                ml_type=MLType(metadata_dict["ml_type"]),
                target_metric=metadata_dict["target_metric"],
                target_col=metadata_dict["target_col"],
                target_assignments=metadata_dict.get("target_assignments", {}),
                positive_class=metadata_dict.get("positive_class"),
                row_id_col=metadata_dict.get("row_id_col"),
                feature_cols=metadata_dict.get("feature_cols", []),
            )
            instance._result_type = ResultType(metadata_dict.get("result_type", "best"))
            instance._task_id = metadata_dict["task_id"]
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"Saved predictor metadata is incomplete or corrupted: {e}. "
                f"Re-save the predictor or check {load_dir / 'metadata.json'}."
            ) from e
        instance._feature_cols = metadata_dict.get("feature_cols", [])
        instance._loaders = {}
        instance._feature_cols_per_split = {
            int(k): v for k, v in metadata_dict.get("feature_cols_per_split", {}).items()
        }
        instance._feature_groups_per_split = {
            int(k): v for k, v in metadata_dict.get("feature_groups_per_split", {}).items()
        }

        instance._models = {}
        models_dir = load_dir / "models"
        for split_id in instance._study_info.outersplits:
            instance._models[split_id] = joblib_load(models_dir / f"model_{split_id:03d}.joblib")

        return instance
