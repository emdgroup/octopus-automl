"""Level 1 — data functions that return DataFrames or dicts.

Pure computation, no display, no side effects. Independently testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve

from octopus.poststudy.study_io import StudyInfo, TaskOutersplitLoader, discover_result_types
from octopus.types import MLType

if TYPE_CHECKING:
    from octopus.poststudy.task_evaluator_test import OctoTestEvaluator


def workflow_graph(study_info: StudyInfo) -> str:
    """Return an ASCII tree of workflow tasks and their dependencies.

    Each node shows the task ID, module type (uppercased), and description.
    Children are indented with tree connectors to visualise the dependency
    chain.

    Args:
        study_info: ``StudyInfo`` returned by ``load_study_information()``.

    Returns:
        Multi-line string representing the workflow tree.

    Example::

        [task 0] OCTO "step1_octo_full"
        └── [task 1] MRMR "step2_mrmr"
            └── [task 2] OCTO "step3_octo_reduced"
    """
    tasks = study_info.workflow_tasks
    if not tasks:
        return "(no workflow tasks)"

    # Build parent → children mapping
    children: dict[int | None, list[dict[str, Any]]] = {}
    for task in tasks:
        parent = task.get("depends_on")
        children.setdefault(parent, []).append(task)

    lines: list[str] = []
    roots = children.get(None, [])
    for i, root in enumerate(roots):
        if i > 0:
            lines.append("")
        _walk_tree(root, "", lines, children, connector="")
    return "\n".join(lines)


def _walk_tree(
    task: dict[str, Any],
    prefix: str,
    lines: list[str],
    children: dict[int | None, list[dict[str, Any]]],
    connector: str = "",
) -> None:
    """Recursively render a task node and its children."""
    module = str(task.get("module", "")).upper()
    desc = task.get("description", "")
    label = f'[task {task["task_id"]}] {module} "{desc}"'
    lines.append(f"{prefix}{connector}{label}")

    child_prefix = prefix + ("    " if connector in ("", "└── ") else "│   ")

    child_list = children.get(task["task_id"], [])
    for i, child in enumerate(child_list):
        is_last = i == len(child_list) - 1
        _walk_tree(child, child_prefix, lines, children, "└── " if is_last else "├── ")


# Module types that produce performance scores (prediction modules).
# Feature selection modules (mrmr, roc, boruta) are skipped.
_PREDICTION_MODULES = {"octo", "autogluon"}


def get_performance(
    study_info: StudyInfo,
    report_test: bool = False,
) -> pd.DataFrame:
    """Return performance per outersplit for all prediction tasks and metrics.

    Reads ``scores.parquet`` directly from each outersplit directory.
    Feature selection tasks (mrmr, roc, boruta) are skipped automatically.
    All result types (e.g. ``best``, ``ensemble_selection``) are discovered
    from disk.

    Args:
        study_info: ``StudyInfo`` returned by ``load_study_information()``.
        report_test: If True, include test-set columns (``test_avg``,
            ``test_ensemble``).  Default False to prevent accidental
            data leakage during model selection.

    Returns:
        DataFrame with columns: ``outersplit``, ``task``, ``key``,
        ``metric``, plus partition/aggregation columns (``dev_avg``,
        ``dev_ensemble``, ``train_avg``, ``train_ensemble``, and
        optionally ``test_avg``, ``test_ensemble``).
        ``Mean`` rows per (task, key, metric) group are appended.
        ``attrs["target_metric"]`` is set from the study config.
    """
    study_path = study_info.path
    frames: list[pd.DataFrame] = []

    for item in study_info.workflow_tasks:
        if item.get("module", "").lower() not in _PREDICTION_MODULES:
            continue

        task_id = item["task_id"]
        result_types = discover_result_types(study_info.outersplit_dirs, task_id, "scores.parquet")
        if not result_types:
            continue

        for rt in result_types:
            for split_dir in study_info.outersplit_dirs:
                split_id = int(split_dir.name.replace("outersplit", ""))
                loader = TaskOutersplitLoader(study_path, split_id, task_id, rt)
                scores = loader.load_scores()
                if scores.empty:
                    continue

                # Keep only avg and ensemble aggregations
                scores = scores[scores["aggregation"].isin(["avg", "ensemble"])]
                scores["perf_key"] = scores["partition"] + "_" + scores["aggregation"]
                pivoted = scores.pivot(index="metric", columns="perf_key", values="value")
                pivoted.index.name = None

                pivoted.insert(0, "outersplit", split_id)
                pivoted.insert(1, "task", task_id)
                pivoted.insert(2, "key", rt)
                pivoted.insert(3, "metric", pivoted.index)
                pivoted = pivoted.reset_index(drop=True)
                frames.append(pivoted)

            # Append Mean rows for this (task, key)
            if frames:
                task_key_frames = [
                    f
                    for f in frames
                    if not f.empty and int(f["task"].iloc[0]) == task_id and str(f["key"].iloc[0]) == rt
                ]
                if task_key_frames:
                    combined = pd.concat(task_key_frames, ignore_index=True)
                    numeric_cols = combined.select_dtypes(include="number").columns.tolist()
                    numeric_cols = [c for c in numeric_cols if c not in ("outersplit", "task")]
                    means = combined.groupby("metric")[numeric_cols].mean().reset_index()
                    means["outersplit"] = "Mean"
                    means["task"] = task_id
                    means["key"] = rt
                    frames.append(means)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.set_index("outersplit")

    if not report_test:
        test_cols = [c for c in result.columns if "_test_" in c or c.startswith("test_")]
        result = result.drop(columns=[c for c in test_cols if c in result.columns])

    result.attrs["target_metric"] = study_info.target_metric
    return result


def get_selected_features(
    study_info: StudyInfo,
    sort_task: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return feature count table and frequency table.

    Reads ``selected_features.json`` from each outersplit, task, and
    result_type directory.  Discovers all result_types on disk (e.g.
    ``best``, ``ensemble_selection``).  Tasks without selected features
    are skipped automatically.

    Args:
        study_info: ``StudyInfo`` returned by ``load_study_information()``.
        sort_task: Task ID whose column is used to sort the frequency
            table descending.  Defaults to the first task.

    Returns:
        Tuple of two DataFrames:

        - **feature_table**: outersplit (index), columns include ``task``,
          ``result_type``, ``n_features``.  Includes Mean rows.
        - **frequency_table**: feature name (index) x (task, result_type)
          columns, values = count of outersplits in which the feature was
          selected.  Sorted descending by *sort_task* column.
    """
    raw_rows: list[dict[str, Any]] = []
    task_rt_keys: list[tuple[int, str]] = []

    for t in study_info.workflow_tasks:
        tid = t["task_id"]
        result_types = discover_result_types(study_info.outersplit_dirs, tid)
        if not result_types:
            continue

        for rt in result_types:
            task_rt_keys.append((tid, rt))
            for split_dir in study_info.outersplit_dirs:
                split_id = int(split_dir.name.replace("outersplit", ""))
                loader = TaskOutersplitLoader(study_info.path, split_id, tid, rt)
                try:
                    features: list[str] = loader.load_selected_features()
                except FileNotFoundError:
                    continue
                raw_rows.append(
                    {
                        "outersplit": split_id,
                        "task": tid,
                        "result_type": rt,
                        "features": features,
                    }
                )

    if not raw_rows:
        return pd.DataFrame(columns=["n_features"]), pd.DataFrame()

    # --- feature_table: outersplit x (task, result_type), values = n_features ---
    feature_table = pd.DataFrame(
        [
            {
                "outersplit": r["outersplit"],
                "task": r["task"],
                "result_type": r["result_type"],
                "n_features": len(r["features"]),
            }
            for r in raw_rows
        ],
    )
    group_cols = ["task", "result_type"]
    means = feature_table.groupby(group_cols)[["n_features"]].mean().reset_index()
    means["outersplit"] = "Mean"
    feature_table = pd.concat([feature_table, means], ignore_index=True)
    feature_table = feature_table.set_index("outersplit")

    if len(task_rt_keys) == 1:
        feature_table = feature_table.drop(columns=group_cols)

    # --- frequency_table: feature name x (task, result_type) ---
    freq_rows = [
        {"feature": feat, "task_key": f"{r['task']}_{r['result_type']}"} for r in raw_rows for feat in r["features"]
    ]
    frequency_table = pd.DataFrame(freq_rows).groupby(["feature", "task_key"]).size().unstack(fill_value=0)
    frequency_table.columns.name = None

    # Determine sort column
    if sort_task is not None:
        sort_candidates = [c for c in frequency_table.columns if c.startswith(f"{sort_task}_")]
        sort_col = sort_candidates[0] if sort_candidates else frequency_table.columns[0]
    else:
        sort_col = frequency_table.columns[0]
    frequency_table = frequency_table.sort_values(sort_col, ascending=False)

    if len(task_rt_keys) == 1:
        frequency_table.columns = ["frequency"]

    return feature_table, frequency_table


def testset_performance_table(
    predictor: OctoTestEvaluator,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Return test-set performance per outersplit with Mean row.

    Args:
        predictor: ``OctoTestEvaluator`` instance for the task to evaluate.
        metrics: List of metric names to evaluate.  If None, uses defaults
            based on ML type.

    Returns:
        DataFrame with outersplits as rows (plus ``Mean``), metrics as columns.
    """
    if metrics is None:
        if predictor.study_info.ml_type == MLType.BINARY:
            metrics = ["AUCROC", "ACCBAL", "ACC"]
        elif predictor.study_info.ml_type == MLType.REGRESSION:
            metrics = ["R2", "MAE", "RMSE"]
        else:
            metrics = []

    performance_long = predictor.performance(metrics=metrics)
    df = performance_long.pivot(index="outersplit", columns="metric", values="score")

    available_cols = [m for m in metrics if m in df.columns]
    df = df[available_cols]
    df.loc["Mean"] = df.mean()
    return df


def aucroc_data(predictor: OctoTestEvaluator) -> dict[str, Any]:
    """Compute ROC curve data for all outersplits.

    Args:
        predictor: ``OctoTestEvaluator`` instance for a binary classification task.

    Returns:
        Dictionary with keys:
            - ``merged_fpr``, ``merged_tpr``, ``merged_auc``: Pooled ROC data.
            - ``mean_fpr``, ``mean_tpr``, ``std_tpr``: Averaged ROC data.
            - ``mean_auc``, ``std_auc``: AUC statistics.
            - ``tprs_upper``, ``tprs_lower``: +/- 1 std band boundaries.
            - ``per_split``: List of per-split dicts with ``split_id``,
              ``fpr``, ``tpr``, ``auc``.

    Raises:
        ValueError: If the task is not binary classification.
    """
    if predictor.study_info.ml_type != MLType.BINARY:
        raise ValueError("AUCROC data is only available for binary classification tasks")

    proba_df = predictor.predict_proba(df=True)
    positive_class = predictor.study_info.positive_class

    per_split: list[dict[str, Any]] = []
    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs: list[np.ndarray] = []

    for split_id in predictor.study_info.outersplits:
        split_df = proba_df[proba_df["outersplit"] == split_id]
        probabilities = np.asarray(split_df[positive_class])
        target = np.asarray(split_df["target"])

        fpr, tpr, _ = roc_curve(target, probabilities, drop_intermediate=True)
        auc_score = float(auc(fpr, tpr))
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

        per_split.append({"split_id": split_id, "fpr": fpr, "tpr": tpr, "auc": auc_score})

    # Merged (pooled) ROC
    all_probabilities = np.asarray(proba_df[positive_class])
    all_targets = np.asarray(proba_df["target"])
    fpr_merged, tpr_merged, _ = roc_curve(all_targets, all_probabilities, drop_intermediate=True)
    auc_merged = float(auc(fpr_merged, tpr_merged))

    # Averaged ROC
    tprs_array = np.array(interp_tprs)
    aucs = [s["auc"] for s in per_split]
    mean_tpr = np.mean(tprs_array, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs_array, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    return {
        "merged_fpr": fpr_merged,
        "merged_tpr": tpr_merged,
        "merged_auc": auc_merged,
        "mean_fpr": mean_fpr,
        "mean_tpr": mean_tpr,
        "std_tpr": std_tpr,
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "tprs_upper": tprs_upper,
        "tprs_lower": tprs_lower,
        "per_split": per_split,
    }


def confusion_matrix_data(
    predictor: OctoTestEvaluator,
    threshold: float = 0.5,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Compute confusion matrices and performance scores per outersplit.

    Args:
        predictor: ``OctoTestEvaluator`` instance for a binary classification task.
        threshold: Probability threshold for binary classification.
        metrics: List of metric names to evaluate.  If None, uses defaults.

    Returns:
        Dictionary with keys:
            - ``per_split``: Dict mapping outersplit_id to a dict with
              ``cm_abs``, ``cm_rel``, ``class_names``, ``scores`` (DataFrame).
            - ``mean_scores``: Series indexed by metric with mean performance
              across splits.

    Raises:
        ValueError: If the task is not binary classification.
    """
    if metrics is None:
        metrics = ["AUCROC", "ACCBAL", "ACC", "F1", "AUCPR", "NEGBRIERSCORE"]

    if predictor.study_info.ml_type != MLType.BINARY:
        raise ValueError("Confusion matrix data is only available for binary classification tasks")

    proba_df = predictor.predict_proba(df=True)
    positive_class = predictor.study_info.positive_class
    all_scores = predictor.performance(metrics=metrics, threshold=threshold)

    per_split_data: dict[int, dict[str, Any]] = {}
    result_rows: list[dict[str, Any]] = []

    for outersplit_id in predictor.study_info.outersplits:
        split_df = proba_df[proba_df["outersplit"] == outersplit_id]
        probabilities = np.asarray(split_df[positive_class])
        target = np.asarray(split_df["target"])

        predictions = (probabilities > threshold).astype(int)
        cm_abs = confusion_matrix(target, predictions)
        cm_rel = confusion_matrix(target, predictions, normalize="true")
        class_names = [str(c) for c in predictor.classes_]

        split_scores = all_scores[all_scores["outersplit"] == outersplit_id]

        per_split_data[outersplit_id] = {
            "cm_abs": cm_abs,
            "cm_rel": cm_rel,
            "class_names": class_names,
            "scores": split_scores,
        }

        for _, row in split_scores.iterrows():
            result_rows.append(
                {
                    "metric": row["metric"],
                    "performance": row["score"],
                    "outersplit_id": outersplit_id,
                }
            )

    df_results = pd.DataFrame(result_rows)
    mean_scores = df_results.groupby("metric")["performance"].mean()

    return {
        "per_split": per_split_data,
        "mean_scores": mean_scores,
        "threshold": threshold,
        "metrics": metrics,
    }


def fi_ensemble_table(fi_table: pd.DataFrame) -> pd.DataFrame:
    """Filter feature importance table to ensemble rows, sorted by importance.

    Args:
        fi_table: DataFrame returned by ``calculate_fi()``, containing
            per-split and ensemble rows with a ``fi_source`` column.

    Returns:
        DataFrame with ensemble feature importance results sorted by
        importance descending.
    """
    if "fi_source" in fi_table.columns:
        ensemble_df = fi_table[fi_table["fi_source"] == "ensemble"].copy()
        ensemble_df = ensemble_df.sort_values("importance_mean", ascending=False)
        return ensemble_df.reset_index(drop=True)
    return fi_table
