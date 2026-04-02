"""Post-hoc analysis tables: DataFrames and dicts, no figures.

All public functions return DataFrames, dicts, or plain values.
No ``print()``, ``display()``, or ``fig.show()`` calls.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd


if TYPE_CHECKING:
    from octopus.predict.study_io import StudyInfo

__all__ = [
    "get_details",
    "performance",
    "selected_features",
    "workflow_graph",
]


def workflow_graph(study: StudyInfo) -> str:
    """Return an ASCII tree diagram of the workflow tasks.

    Each task has at most one parent (``depends_on``), so the workflow is
    a forest of trees.  Example with a branch::

        [task 0] octo "step_1"
        ├── [task 1] mrmr "step_2"
        │   └── [task 2] octo "step_3"
        └── [task 3] octo "step_4"

    Args:
        study: A ``StudyInfo`` instance returned by ``load_study()``.

    Returns:
        Multi-line string with the workflow tree.
    """
    tasks = study.workflow_tasks
    if not tasks:
        return "(empty workflow)"

    children_of: dict[int | None, list[dict]] = {}
    for t in tasks:
        parent = t.get("depends_on")
        children_of.setdefault(parent, []).append(t)

    lines: list[str] = []

    def _walk(task: dict, prefix: str, connector: str) -> None:
        desc = task.get("description", "")
        module = task.get("module", "?").upper()
        label = f'[task {task["task_id"]}] {module} "{desc}"'
        lines.append(f"{prefix}{connector}{label}")

        children = children_of.get(task["task_id"], [])
        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            child_prefix = prefix + ("    " if connector in ("", "└── ") else "│   ")
            _walk(child, child_prefix, "└── " if is_last else "├── ")

    roots = children_of.get(None, [])
    for i, root in enumerate(roots):
        if i > 0:
            lines.append("")
        _walk(root, "", "")

    return "\n".join(lines)


def get_details(study: StudyInfo) -> dict:
    """Return a summary of key study metadata.

    Args:
        study: A ``StudyInfo`` instance returned by ``load_study()``.

    Returns:
        Dictionary with keys: ``ml_type``, ``target_metric``, ``n_folds_outer``,
        ``n_outersplits``, ``task_ids``.
    """
    return {
        "ml_type": study.config["ml_type"],
        "target_metric": study.config.get("target_metric", ""),
        "n_folds_outer": study.config["n_folds_outer"],
        "n_outersplits": len(study.outersplit_dirs),
    }


def _tasks_with_scores(study_info: StudyInfo) -> list[int]:
    """Return task IDs that have scores (i.e. not pure feature selection tasks)."""
    task_ids = []
    first_split = study_info.outersplit_dirs[0]
    for t in study_info.workflow_tasks:
        scores_file = first_split / f"task{t['task_id']}" / "results" / "best" / "scores.parquet"
        if scores_file.exists():
            task_ids.append(t["task_id"])
    return task_ids


def performance(
    study_info: StudyInfo,
    task: int | None = None,
    metric: str | None = None,
    partition: str = "dev",
    aggregation: str = "avg",
) -> pd.DataFrame:
    """Performance per outer split for one or all tasks.

    Reads ``scores.parquet`` from each outersplit directory and pivots
    metrics into columns.  Tasks without scores (e.g. pure feature
    selection steps) are skipped automatically.

    Args:
        study_info: ``StudyInfo`` instance returned by ``load_study()``.
        task: Workflow task ID, or ``None`` for all tasks (default).
        metric: Metric name to report (e.g. ``"AUCROC"``).  If None,
            all available metrics are shown.
        partition: Data partition (``"dev"``, ``"test"``, or ``"train"``).
        aggregation: Aggregation method (``"avg"`` or ``"pool"``).

    Returns:
        DataFrame with one row per outer split (plus a Mean row per task),
        one column per metric.  A ``task`` column is included when
        multiple tasks are returned.

    Example:
        >>> study_info = load_study("./studies/my_study/")
        >>> performance(study_info)
        >>> performance(study_info, task=2)
        >>> performance(study_info, metric="AUCROC", partition="test")
    """
    scored_tasks = _tasks_with_scores(study_info)

    if task is None:
        task_ids = scored_tasks
    else:
        task_ids = [task]

    rows: list[dict] = []

    for tid in task_ids:
        for split_dir in study_info.outersplit_dirs:
            split_id = int(split_dir.name.replace("outersplit", ""))
            scores_file = split_dir / f"task{tid}" / "results" / "best" / "scores.parquet"

            if not scores_file.exists():
                continue

            df = pd.read_parquet(scores_file)
            df = df[(df["partition"] == partition) & (df["aggregation"] == aggregation)]

            if metric is not None:
                df = df[df["metric"] == metric]

            row: dict = {"outersplit": split_id, "task": tid}
            for _, s in df.iterrows():
                row[s["metric"]] = s["value"]
            rows.append(row)

    result = pd.DataFrame(rows)

    # Add mean rows per task
    metric_cols = [c for c in result.columns if c not in ("outersplit", "task")]
    means = result.groupby("task")[metric_cols].mean().reset_index()
    means["outersplit"] = "Mean"
    result = pd.concat([result, means], ignore_index=True)
    result = result.set_index("outersplit")

    # Drop task column for single-task results
    if len(task_ids) == 1:
        result = result.drop(columns="task")

    return result


def _tasks_with_selected_features(study_info: StudyInfo) -> list[int]:
    """Return task IDs that have selected_features.json."""
    task_ids = []
    first_split = study_info.outersplit_dirs[0]
    for t in study_info.workflow_tasks:
        sf_file = first_split / f"task{t['task_id']}" / "results" / "best" / "selected_features.json"
        if sf_file.exists():
            task_ids.append(t["task_id"])
    return task_ids


def selected_features(
    study_info: StudyInfo,
    sort_task: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Selected features per outer split for all tasks.

    Reads ``selected_features.json`` from each outersplit directory.

    Args:
        study_info: ``StudyInfo`` instance returned by ``load_study()``.
        sort_task: Task ID whose column is used to sort the frequency
            table descending.  Defaults to the first task.

    Returns:
        A tuple of two DataFrames:

        - **feature_table**: outersplit (index), values = number of
          selected features.  Includes a Mean row.  A ``task`` column
          is included when multiple tasks are returned.
        - **frequency_table**: feature name (index) x task (columns),
          values = count of outersplits in which the feature was selected.
          Sorted descending by ``sort_task`` column.

    Example:
        >>> study_info = load_study("./studies/my_study/")
        >>> feature_table, frequency_table = selected_features(study_info)
    """
    task_ids = _tasks_with_selected_features(study_info)

    raw_rows: list[dict] = []
    for tid in task_ids:
        for split_dir in study_info.outersplit_dirs:
            split_id = int(split_dir.name.replace("outersplit", ""))
            sf_file = split_dir / f"task{tid}" / "results" / "best" / "selected_features.json"
            if not sf_file.exists():
                continue
            with sf_file.open() as f:
                features: list[str] = json.load(f)
            raw_rows.append({"outersplit": split_id, "task": tid, "features": features})

    # --- feature_table (same flat pattern as performance()) ---
    feature_table = pd.DataFrame(
        [{"outersplit": r["outersplit"], "task": r["task"], "n_features": len(r["features"])} for r in raw_rows],
    )
    means = feature_table.groupby("task")[["n_features"]].mean().reset_index()
    means["outersplit"] = "Mean"
    feature_table = pd.concat([feature_table, means], ignore_index=True)
    feature_table = feature_table.set_index("outersplit")

    if len(task_ids) == 1:
        feature_table = feature_table.drop(columns="task")

    # --- frequency_table: feature name x task, values = count across outersplits ---
    freq_rows = [
        {"feature": feat, "task": r["task"]}
        for r in raw_rows
        for feat in r["features"]
    ]
    frequency_table = (
        pd.DataFrame(freq_rows)
        .groupby(["feature", "task"])
        .size()
        .unstack(fill_value=0)
    )
    frequency_table.columns.name = None

    sort_col = sort_task if sort_task is not None else task_ids[0]
    if sort_col in frequency_table.columns:
        frequency_table = frequency_table.sort_values(sort_col, ascending=False)

    if len(task_ids) == 1:
        frequency_table.columns = ["frequency"]

    return feature_table, frequency_table
