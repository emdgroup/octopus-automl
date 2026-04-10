"""Level 3 — notebook convenience wrappers.

Each function combines a Level 1 data function with a Level 2 plot function
and handles display (``display_table`` / ``.show()``).  These are the
functions that notebook users call directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from octopus.poststudy.plots import (
    _aucroc_individual_plot,
    aucroc_averaged_plot,
    aucroc_merged_plot,
    confusion_matrix_plot,
    fi_plot,
    testset_performance_plot,
)
from octopus.poststudy.tables import (
    StudyInfo,
    aucroc_data,
    confusion_matrix_data,
    fi_ensemble_table,
    testset_performance_table,
)

try:
    from IPython.display import display as _ipython_display
except ImportError:
    _ipython_display = None

if TYPE_CHECKING:
    from typing import Any

    from octopus.poststudy.task_evaluator_test import OctoTestEvaluator


def display_table(data: Any) -> None:
    """Display a table in Jupyter notebooks or print in other environments.

    Args:
        data: The data to display (DataFrame, Series, or any printable object).
    """
    if _ipython_display is not None:
        _ipython_display(data)
    else:
        print(data)


def show_study_overview(study_info: StudyInfo) -> None:
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


def show_performance_tables(
    perf_df: pd.DataFrame,
    metric: str | None = None,
) -> None:
    """Display performance tables per workflow task for a single metric.

    Filters the DataFrame from ``get_performance()`` to the selected metric,
    then splits into per-task/key groups with headers.

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

    # Filter to selected metric
    filtered = perf_df[perf_df["metric"] == metric].copy()
    if filtered.empty:
        print(f"No data for metric '{metric}'")
        return

    target_metric = perf_df.attrs.get("target_metric", "")

    # Preferred column order for display (columns may not all exist)
    _col_order = ["task", "key", "train_avg", "train_ensemble", "dev_avg", "dev_ensemble", "test_avg", "test_ensemble"]
    available_cols = [c for c in filtered.columns if c != "metric"]
    display_cols = [c for c in _col_order if c in available_cols]
    display_cols += [c for c in available_cols if c not in display_cols]

    # Collect unique (task, key) pairs in order
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
        display_table(sub_df)


def show_selected_features_tables(
    feat_table: pd.DataFrame,
    freq_table: pd.DataFrame,
) -> None:
    """Display feature count and frequency tables.

    Args:
        feat_table: Feature count DataFrame from ``get_selected_features()[0]``.
        freq_table: Frequency DataFrame from ``get_selected_features()[1]``.
    """
    print("\n" + "=" * 40)
    print("NUMBER OF SELECTED FEATURES")
    print("=" * 40)
    display_table(feat_table)

    print("\n" + "=" * 40)
    print("FEATURE FREQUENCY ACROSS OUTER SPLITS")
    print("=" * 40)
    display_table(freq_table)


def show_testset_performance(
    predictor: OctoTestEvaluator,
    metrics: list[str] | None = None,
    *,
    show_plot: bool = True,
) -> pd.DataFrame:
    """Display test-set performance table and optional plot.

    Args:
        predictor: ``OctoTestEvaluator`` instance for the task to evaluate.
        metrics: List of metric names to evaluate.  If None, uses defaults.
        show_plot: If True, display bar chart alongside the table.

    Returns:
        DataFrame with outersplits as rows (plus Mean), metrics as columns.
    """
    print("Performance on test dataset by outer split")
    df = testset_performance_table(predictor, metrics=metrics)
    display_table(df)
    if show_plot:
        fig = testset_performance_plot(df)
        fig.show()
    return df


def show_aucroc(
    predictor: OctoTestEvaluator,
    *,
    show_individual: bool = False,
) -> None:
    """Display merged and averaged ROC curves.

    Args:
        predictor: ``OctoTestEvaluator`` instance for a binary classification task.
        show_individual: If True, also display individual ROC curves per outersplit.
    """
    roc = aucroc_data(predictor)

    print("=" * 60)
    print("1. MERGED ROC CURVE (All Predictions Pooled)")
    print("=" * 60)
    print(f"Merged AUC-ROC: {roc['merged_auc']:.3f}\n")
    aucroc_merged_plot(roc).show()

    print("\n" + "=" * 60)
    print("2. AVERAGED ROC CURVE (Mean +/- Std. Dev.)")
    print("=" * 60)
    print(f"Mean AUC-ROC: {roc['mean_auc']:.3f}")
    print(f"Std. Dev. AUC-ROC: {roc['std_auc']:.3f}\n")
    aucroc_averaged_plot(roc).show()

    if show_individual:
        print("\n" + "=" * 60)
        print("3. INDIVIDUAL ROC CURVES")
        print("=" * 60)
        for split_info in roc["per_split"]:
            split_id = split_info["split_id"]
            auc_score = split_info["auc"]
            print(f"\nOutersplit {split_id}: AUC = {auc_score:.3f}")
            _aucroc_individual_plot(
                split_info["fpr"],
                split_info["tpr"],
                auc_score,
                split_id,
            ).show()


def show_confusionmatrix(
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

    print("=" * 80)
    print("CONFUSION MATRICES AND PERFORMANCE METRICS")
    print("=" * 80)
    print(f"Threshold: {cm_data['threshold']}")
    print(f"Metrics: {cm_data['metrics']}\n")

    for outersplit_id, split_data in cm_data["per_split"].items():
        print(f"\n{'=' * 60}")
        print(f"OUTERSPLIT {outersplit_id}")
        print("=" * 60)

        fig = confusion_matrix_plot(
            split_data["cm_abs"],
            split_data["cm_rel"],
            split_data["class_names"],
            f"Confusion Matrices - Outersplit {outersplit_id}",
        )
        print("\nConfusion Matrices:")
        fig.show()

        print("\nPerformance Metrics:")
        for _, row in split_data["scores"].iterrows():
            print(f"  {row['metric']:<15}: {row['score']:.4f}")

    print(f"\n{'=' * 80}")
    print("OVERALL PERFORMANCE (Mean across all outersplits)")
    print("=" * 80)
    mean_scores = cm_data["mean_scores"]
    for metric_name, value in mean_scores.items():
        print(f"  {metric_name!s:<15}: {value:.4f}")


def show_fi(
    fi_table: pd.DataFrame,
    *,
    top_n: int | None = None,
    show_plot: bool = True,
) -> pd.DataFrame:
    """Display ensemble FI table and optional bar plot.

    Args:
        fi_table: DataFrame returned by ``calculate_fi()``.
        top_n: Number of top features to display in the plot.  None shows all.
        show_plot: If True, display bar chart alongside the table.

    Returns:
        DataFrame with ensemble feature importance (sorted descending).
    """
    ensemble_df = fi_ensemble_table(fi_table)
    display_table(ensemble_df)
    if show_plot:
        fig = fi_plot(fi_table, top_n=top_n)
        fig.show()
    return ensemble_df
