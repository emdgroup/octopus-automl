"""Level 3 — notebook convenience wrappers.

Each function combines a Level 1 data function with a Level 2 plot function
and handles display (``display_table`` / ``.show()``).  These are the
functions that notebook users call directly.

Layer contract:

- L1 (tables.py) returns DataFrames/dicts — pure data, no side effects.
- L2 (plots.py) returns ``go.Figure`` — pure visualization, no side effects.
- L3 (this module) composes L1+L2 with contextual print output and display.
  **Always returns the computed data** so callers can build on it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from octopus.poststudy.analysis.plots import confusion_matrix_plot
from octopus.poststudy.analysis.tables import confusion_matrix_data
from octopus.poststudy.study_io import StudyInfo
from octopus.types import PerformanceKey

try:
    from IPython.display import display as _ipython_display
except ImportError:
    _ipython_display = None

if TYPE_CHECKING:
    from typing import Any

    from octopus.poststudy.analysis.evaluator import OctoTestEvaluator
    from octopus.poststudy.base_predictor import _PredictorBase


def display_table(data: Any) -> None:
    """Display a table in Jupyter notebooks or print in other environments.

    Args:
        data: The data to display (DataFrame, Series, or any printable object).
    """
    if _ipython_display is not None:
        _ipython_display(data)
    else:
        print(data)


def display_study_overview(study_info: StudyInfo) -> None:
    """Display a concise summary of the study configuration.

    Prints study path, ML type, target metric, and outer split count.

    Args:
        study_info: ``StudyInfo`` returned by ``load_study_information()``.
    """
    print(f"Selected study path: {study_info.path}")
    print(f"ML type: {study_info.ml_type}")
    print(f"Target metric: {study_info.target_metric}")
    print(f"Number of outer splits: {study_info.n_outersplits}")
    print(f"Outer split IDs: {study_info.outersplits}")


def display_performance_tables(
    perf_df: pd.DataFrame,
    metric: str | None = None,
) -> None:
    """Display performance tables per workflow task for a single metric.

    Filter the DataFrame from ``get_performance()`` to the selected metric,
    then split into per-task/key groups with headers.

    Args:
        perf_df: DataFrame from ``get_performance()`` with ``task``,
            ``key``, and ``metric`` columns.
        metric: Metric name to display (e.g. ``"AUCROC"``).  If None,
            defaults to ``perf_df.attrs["target_metric"]``.
    """
    if perf_df.empty:
        return

    if metric is None:
        metric = perf_df.attrs.get("target_metric", "")
    if not metric:
        return

    filtered = perf_df[perf_df["metric"] == metric].copy()
    if filtered.empty:
        print(f"No data for metric '{metric}'")
        return

    target_metric = perf_df.attrs.get("target_metric", "")

    _col_order = [
        PerformanceKey.TRAIN_AVG,
        PerformanceKey.DEV_AVG,
        PerformanceKey.DEV_ENSEMBLE,
        PerformanceKey.TEST_AVG,
        PerformanceKey.TEST_ENSEMBLE,
    ]
    _hidden = {"metric", "task", "key", PerformanceKey.TRAIN_ENSEMBLE}
    available_cols = [c for c in filtered.columns if c not in _hidden]
    display_cols: list[str] = [c for c in _col_order if c in available_cols]
    display_cols += [c for c in available_cols if c not in display_cols]

    task_key_pairs: list[tuple[int, str]] = []
    task_to_keys: dict[int, list[str]] = {}
    for _, row in filtered[["task", "key"]].drop_duplicates().iterrows():
        task_id = int(row["task"])
        key = str(row["key"])
        task_key_pairs.append((task_id, key))
        task_to_keys.setdefault(task_id, []).append(key)

    print(f"Display metric: {metric}")
    print(f"Target metric: {target_metric}\n")
    current_task: int | None = None

    for task_id, key in task_key_pairs:
        if task_id != current_task:
            current_task = task_id
            keys = task_to_keys[task_id]
            print(f"\033[1mWorkflow task: {task_id}\033[0m")
            print(f"Available results keys: {keys}")

        mask = (filtered["task"] == task_id) & (filtered["key"] == key)
        sub_df = filtered.loc[mask, display_cols]

        print(f"Selected results key: {key}")
        with pd.option_context("display.float_format", "{:.3f}".format):
            display_table(sub_df)


def display_confusionmatrix(
    predictor: OctoTestEvaluator,
    threshold: float = 0.5,
    metrics: list[str] | None = None,
) -> None:
    """Display confusion matrices and performance metrics for all outersplits.

    Args:
        predictor: ``OctoTestEvaluator`` instance for a binary classification task.
        threshold: Probability threshold for binary classification.
        metrics: List of metric names to evaluate.  If None, uses defaults.
    """
    cm_data = confusion_matrix_data(predictor, threshold=threshold, metrics=metrics)

    for outersplit_id, split_data in cm_data["per_split"].items():
        confusion_matrix_plot(
            split_data["cm_abs"],
            split_data["cm_rel"],
            split_data["class_names"],
            f"Outersplit {outersplit_id}",
            scores=split_data["scores"],
        ).show()


def display_feature_groups_table(predictor: _PredictorBase) -> pd.DataFrame:
    """Display feature groups with left-aligned formatting.

    Args:
        predictor: ``OctoTestEvaluator`` (or ``OctoPredictor``) instance.

    Returns:
        DataFrame with columns ``group`` and ``features``.
    """
    from octopus.poststudy.analysis.tables import feature_groups_table  # noqa: PLC0415

    df = feature_groups_table(predictor)
    if not df.empty:
        print("Feature correlation groups and their content")
        styled = df.style.set_properties(subset=["features"], **{"text-align": "left"})
        display_table(styled)
    return df
