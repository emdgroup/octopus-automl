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
from attrs import define
from upath import UPath

from octopus.poststudy.base_predictor import _PredictorBase
from octopus.poststudy.study_io import StudyInfo, load_partition
from octopus.types import FIType, MLType, ResultType
from octopus.utils import get_version, joblib_load, joblib_save, parquet_load


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
        >>> predictions = tp.predict(new_data)
    """

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict on new data using all outer-split models.

        Return a wide-format DataFrame with one row per sample.
        Columns: ``row_id``, ``split_0``, ``split_1``, ..., ``ensemble``.

        For regression and time-to-event, the ``ensemble`` column is the
        arithmetic mean of per-split predictions.  For classification, it
        contains class labels derived from the argmax of ensemble-averaged
        probabilities.

        Args:
            data: DataFrame containing feature columns.

        Returns:
            Wide-format DataFrame with per-split and ensemble predictions.
        """
        per_split_preds = {sid: self._predict_raw(sid, data) for sid in self._study_info.outersplits}

        result = pd.DataFrame({"row_id": self._get_row_ids(data)})
        for split_id, preds in per_split_preds.items():
            result[f"split_{split_id}"] = preds

        if self._study_info.ml_type in (MLType.BINARY, MLType.MULTICLASS):
            per_split_probas = {sid: self._predict_proba_raw(sid, data) for sid in self._study_info.outersplits}
            avg_proba = np.mean(list(per_split_probas.values()), axis=0)
            result["ensemble"] = self.classes_[np.argmax(avg_proba, axis=1)]
        else:
            result["ensemble"] = np.mean(list(per_split_preds.values()), axis=0)

        return result

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict probabilities on new data (classification/multiclass only).

        Return a wide-format DataFrame with one row per sample.
        Columns: ``row_id``, one column per class label (ensemble-averaged),
        then ``<class>_split_0``, ``<class>_split_1``, ... for per-split detail.

        Args:
            data: DataFrame containing feature columns.

        Returns:
            Wide-format DataFrame with ensemble and per-split probabilities.

        Raises:
            TypeError: If ml_type is not classification or multiclass.
        """
        self._check_classification_only("predict_proba")
        per_split_probas = {sid: self._predict_proba_raw(sid, data) for sid in self._study_info.outersplits}
        class_labels = self.classes_
        ensemble: np.ndarray = np.mean(list(per_split_probas.values()), axis=0)

        result = pd.DataFrame({"row_id": self._get_row_ids(data)})
        for i, label in enumerate(class_labels):
            result[label] = ensemble[:, i]
        for split_id, probas in per_split_probas.items():
            for i, label in enumerate(class_labels):
                result[f"{label}_split_{split_id}"] = probas[:, i]
        return result

    def performance(
        self,
        data: pd.DataFrame,
        metrics: list[str] | None = None,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Compute performance on provided data per outersplit with Mean and Ensemble.

        Each outer-split model is scored independently on the **same** data.
        The ``Mean`` row averages per-split scores.  The ``Ensemble`` row
        scores the ensemble-averaged predictions against ground truth.

        Args:
            data: Data to score on; must contain feature columns + target column.
            metrics: List of metric names to compute.
                If None, auto-detected from the ML type.
            threshold: Classification threshold for threshold-dependent metrics.

        Returns:
            Wide DataFrame with outersplit IDs as index (plus ``Mean`` and
            ``Ensemble``), metrics as columns.
        """
        metrics = self._resolve_metrics(metrics)
        data_per_split = dict.fromkeys(self._study_info.outersplits, data)
        df = self._compute_per_split_scores(data_per_split, metrics, threshold)
        df.loc["Mean"] = df.mean()

        pred_df = self.predict(data)
        pred_with_target = pd.DataFrame(
            {
                "prediction": pred_df["ensemble"].values,
                **self._get_target_columns(data),
            }
        )
        proba_df = self.predict_proba(data) if self._needs_proba(metrics) else None
        df.loc["Ensemble"] = self._compute_summary_scores(pred_with_target, proba_df, metrics, threshold)
        return df

    def _build_pool_data(self, data: pd.DataFrame) -> dict[int, pd.DataFrame]:
        """Build per-split pool data for permutation FI.

        In study-connected mode, filters ``data_prepared.parquet`` using the
        stored split row IDs to recover per-split traindev data.  In deployment
        mode (loaded via ``OctoPredictor.load``), the user-provided ``data`` is
        used as the pool for all splits.

        The prepared data is cached on first read so repeated ``calculate_fi``
        calls (e.g. permutation then SHAP) do not re-read from disk.

        Args:
            data: User-provided data (used as fallback for all splits).

        Returns:
            Dict mapping outersplit_id to pool DataFrame.
        """
        if not hasattr(self, "_prepared_data"):
            prepared_path = self._study_info.path / "data_prepared.parquet"
            self._prepared_data: pd.DataFrame | None = parquet_load(prepared_path) if prepared_path.exists() else None

        pool: dict[int, pd.DataFrame] = {}
        for split_id in self._study_info.outersplits:
            try:
                pool[split_id] = load_partition(self._study_info.path, split_id, "traindev", self._prepared_data)
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

        outersplits = metadata_dict.get("outersplits", [])
        try:
            instance._study_info = StudyInfo(
                path=UPath(load_dir),
                n_outer_splits=len(outersplits),
                workflow_tasks=(),
                outersplit_dirs=(),
                ml_type=MLType(metadata_dict["ml_type"]),
                target_metric=metadata_dict["target_metric"],
                target_col=metadata_dict["target_col"],
                target_assignments=metadata_dict.get("target_assignments", {}),
                positive_class=metadata_dict.get("positive_class"),
                row_id_col=metadata_dict.get("row_id_col"),
                feature_cols=metadata_dict.get("feature_cols", []),
                outersplit_ids=tuple(outersplits),
            )
            instance._result_type = ResultType(metadata_dict.get("result_type", "best"))
            instance._task_id = metadata_dict["task_id"]
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"Saved predictor metadata is incomplete or corrupted: {e}. "
                f"Re-save the predictor or check {load_dir / 'metadata.json'}."
            ) from e
        instance._feature_cols = metadata_dict.get("feature_cols", [])
        instance._feature_cols_per_split = {
            int(k): v for k, v in metadata_dict.get("feature_cols_per_split", {}).items()
        }
        instance._feature_groups_per_split = {
            int(k): v for k, v in metadata_dict.get("feature_groups_per_split", {}).items()
        }

        instance._models = {}
        models_dir = load_dir / "models"
        for split_id in outersplits:
            instance._models[split_id] = joblib_load(models_dir / f"model_{split_id:03d}.joblib")

        return instance
