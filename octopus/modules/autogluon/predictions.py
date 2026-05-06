"""Build prediction frames from a fitted AutoGluon TabularPredictor.

The frames produced here match Tako's prediction schema so that
`octopus.metrics.utils.get_performance_from_predictions` can score AG and Tako
output through the same code path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from octopus._optional.autogluon import TabularPredictor
from octopus.types import DataPartition

if TYPE_CHECKING:
    from octopus.modules import StudyContext

_AG_INNER_SPLIT_ID = "autogluon"


def build_predictions(
    predictor: TabularPredictor,
    *,
    study_context: StudyContext,
    data_traindev: pd.DataFrame,
    data_test: pd.DataFrame,
    outer_split_id: int,
    task_id: int,
) -> dict[DataPartition, pd.DataFrame]:
    """Build DEV (OOF) and TEST prediction frames in Tako's canonical schema.

    Each frame carries:
        - row_id_col, target_col (so the canonical scoring helper can read both),
        - prediction (hard label for classification, value for regression),
        - one column per integer class label with predicted probabilities
          (classification only; named with `predictor.class_labels`),
        - outer_split_id, inner_split_id="autogluon", partition, task_id.

    Args:
        predictor: Fitted AutoGluon TabularPredictor.
        study_context: Study configuration; supplies row_id_col, target column,
            ml_type, and positive_class.
        data_traindev: Train/dev frame used for OOF predictions. Must include
            the row_id_col and target column(s).
        data_test: Test frame used for held-out predictions. Must include the
            row_id_col and target column(s).
        outer_split_id: Outer split index.
        task_id: Task identifier.

    Returns:
        Mapping `{DataPartition.DEV: dev_df, DataPartition.TEST: test_df}`.

    Raises:
        RuntimeError: If AG OOF predictions miss any traindev row index.
        ValueError: If `problem_type` is unsupported, or AG class labels are
            not integer-convertible.
    """
    target_col = next(iter(study_context.target_assignments.values()))
    row_id_col = study_context.row_id_col
    problem_type = predictor.problem_type
    best_model_name = predictor.model_best

    return {
        DataPartition.DEV: _build_one(
            partition=DataPartition.DEV,
            source_data=data_traindev,
            predictor=predictor,
            problem_type=problem_type,
            best_model_name=best_model_name,
            row_id_col=row_id_col,
            target_col=target_col,
            outer_split_id=outer_split_id,
            task_id=task_id,
        ),
        DataPartition.TEST: _build_one(
            partition=DataPartition.TEST,
            source_data=data_test,
            predictor=predictor,
            problem_type=problem_type,
            best_model_name=best_model_name,
            row_id_col=row_id_col,
            target_col=target_col,
            outer_split_id=outer_split_id,
            task_id=task_id,
        ),
    }


def _build_one(
    *,
    partition: DataPartition,
    source_data: pd.DataFrame,
    problem_type: str,
    predictor: TabularPredictor,
    best_model_name: str,
    row_id_col: str,
    target_col: str,
    outer_split_id: int,
    task_id: int,
) -> pd.DataFrame:
    """Build one prediction frame (DEV or TEST) in Tako's schema.

    `partition == DataPartition.DEV` triggers the OOF path
    (`predict_oof` / `predict_proba_oof`); otherwise the in-memory
    `predict` / `predict_proba` path runs against `source_data`.
    """
    is_oof = partition == DataPartition.DEV
    if problem_type == "regression":
        if is_oof:
            pred_values = _aligned_oof_predict(predictor, best_model_name, source_data.index).to_numpy()
        else:
            x_data = source_data[list(predictor.original_features)]
            pred_values = predictor.predict(x_data, as_pandas=False)
        proba_df: pd.DataFrame | None = None
    elif problem_type in ("binary", "multiclass"):
        if is_oof:
            proba_df = _aligned_oof_predict_proba(predictor, best_model_name, source_data.index)
        else:
            x_data = source_data[list(predictor.original_features)]
            proba_df = predictor.predict_proba(x_data, as_multiclass=True)
        int_labels = _coerce_int_labels(predictor.class_labels)
        proba_df.columns = int_labels
        pred_values = np.asarray(int_labels)[proba_df.to_numpy().argmax(axis=1)]
    else:
        raise ValueError(f"Unsupported problem_type: {problem_type!r}")

    base_cols = source_data[[row_id_col, target_col]].reset_index(drop=True)
    pred_col = pd.DataFrame({"prediction": pred_values})
    parts = [base_cols, pred_col]
    if proba_df is not None:
        parts.append(proba_df.reset_index(drop=True))
    out = pd.concat(parts, axis=1)
    out["outer_split_id"] = outer_split_id
    out["inner_split_id"] = _AG_INNER_SPLIT_ID
    out["partition"] = partition
    out["task_id"] = task_id
    return out


def _coerce_int_labels(class_labels: list) -> list[int]:
    """Convert AG class labels to ints, raising a clear error on string labels."""
    try:
        return [int(c) for c in class_labels]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"AutoGluon class labels {class_labels!r} are not integer-convertible. "
            "Octopus requires integer-encoded class labels (e.g. [0, 1] or [0, 1, 2])."
        ) from exc


def _aligned_oof_predict(predictor: TabularPredictor, best_model_name: str, source_index: pd.Index) -> pd.Series:
    """Return OOF predictions reindexed to the traindev source index.

    AG's `predict_oof` returns one row per training row, indexed by the
    training-frame index. Reindexing locks the output to the caller's row
    order so downstream `reset_index(drop=True)` of predictions and target
    are aligned by construction rather than by AG-internal coincidence.
    """
    oof: pd.Series = predictor.predict_oof(model=best_model_name)
    oof = oof.reindex(source_index)
    if oof.isna().any():
        n_missing = int(oof.isna().sum())
        raise RuntimeError(
            f"AG OOF predictions are missing rows for {n_missing} of {len(source_index)} traindev "
            f"indices. predict_oof did not return a row per training row; this indicates an "
            f"AG-internal index contract change."
        )
    return oof


def _aligned_oof_predict_proba(
    predictor: TabularPredictor, best_model_name: str, source_index: pd.Index
) -> pd.DataFrame:
    """Return OOF probabilities reindexed to the traindev source index.

    See `_aligned_oof_predict` for rationale; this is the multiclass-aware
    counterpart that preserves the per-class column structure.
    """
    proba: pd.DataFrame = predictor.predict_proba_oof(model=best_model_name)
    proba = proba.reindex(source_index)
    if proba.isna().any().any():
        n_missing = int(proba.isna().any(axis=1).sum())
        raise RuntimeError(
            f"AG OOF probabilities are missing rows for {n_missing} of {len(source_index)} traindev "
            f"indices. predict_proba_oof did not return a row per training row; this indicates an "
            f"AG-internal index contract change."
        )
    return proba
