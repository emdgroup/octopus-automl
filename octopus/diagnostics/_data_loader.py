"""Data loading utilities for diagnostics — pandas-based parquet glob reader.

Replaces DuckDB's ``read_parquet(..., hive_partitioning=true)`` with
:func:`load_parquet_glob` which iterates directories, reads individual
parquet files, and extracts outersplit/task IDs from directory names.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pandas as pd

from octopus.utils import parquet_load

if TYPE_CHECKING:
    from upath import UPath


def _extract_id_from_dirname(dirname: str, prefix: str) -> int | None:
    """Extract numeric ID from a directory name like 'outersplit0' or 'task2'.

    Args:
        dirname: Directory name (e.g. 'outersplit0', 'task2').
        prefix: Expected prefix (e.g. 'outersplit', 'task').

    Returns:
        Integer ID, or None if the directory name does not match.
    """
    match = re.match(rf"^{prefix}(\d+)$", dirname)
    if match:
        return int(match.group(1))
    return None


def load_parquet_glob(study_path: UPath, pattern: str) -> pd.DataFrame:
    """Load and concatenate parquet files matching a glob pattern.

    Extracts ``outersplit_id`` and ``task_id`` from the directory structure,
    equivalent to DuckDB's ``hive_partitioning=true``.

    Args:
        study_path: Root path of the study directory.
        pattern: Glob pattern relative to study_path
            (e.g. ``"outersplit*/task*/scores.parquet"``).

    Returns:
        Concatenated DataFrame with ``outersplit_id`` and ``task_id``
        columns added from directory names. Empty DataFrame if no
        files match.
    """
    dfs: list[pd.DataFrame] = []
    for parquet_file in sorted(study_path.glob(pattern)):
        try:
            df = parquet_load(parquet_file)
        except Exception:
            continue

        # Extract IDs from path components
        parts = parquet_file.relative_to(study_path).parts
        for part in parts[:-1]:  # exclude filename
            outer_id = _extract_id_from_dirname(part, "outersplit")
            if outer_id is not None and "outersplit_id" not in df.columns:
                df["outersplit_id"] = outer_id
            task_id = _extract_id_from_dirname(part, "task")
            if task_id is not None and "task_id" not in df.columns:
                df["task_id"] = task_id

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_predictions(study_path: UPath) -> pd.DataFrame:
    """Load all predictions parquet files across outersplits and tasks.

    Searches in ``outersplit*/task*/results/*/predictions.parquet`` to pick up
    results stored under ``results/best/`` and ``results/ensemble_selection/`` sub-directories.

    Args:
        study_path: Root path of the study directory.

    Returns:
        Combined predictions DataFrame.
    """
    return load_parquet_glob(study_path, "outersplit*/task*/results/*/predictions.parquet")


def load_feature_importances(study_path: UPath) -> pd.DataFrame:
    """Load all feature importance parquet files across outersplits and tasks.

    Searches in ``outersplit*/task*/results/*/feature_importances.parquet`` to pick up
    results stored under ``results/best/`` and ``results/ensemble_selection/`` sub-directories.

    Args:
        study_path: Root path of the study directory.

    Returns:
        Combined feature importances DataFrame.
    """
    return load_parquet_glob(study_path, "outersplit*/task*/results/*/feature_importances.parquet")


def load_optuna(study_path: UPath) -> pd.DataFrame:
    """Load all Optuna parquet files across outersplits and tasks.

    Args:
        study_path: Root path of the study directory.

    Returns:
        Combined Optuna results DataFrame.
    """
    return load_parquet_glob(study_path, "outersplit*/task*/results/optuna_results.parquet")


def load_scores(study_path: UPath) -> pd.DataFrame:
    """Load all scores parquet files across outersplits and tasks.

    Searches in ``outersplit*/task*/results/*/scores.parquet`` to pick up
    results stored under ``results/best/`` and ``results/ensemble_selection/`` sub-directories.

    Args:
        study_path: Root path of the study directory.

    Returns:
        Combined scores DataFrame.
    """
    return load_parquet_glob(study_path, "outersplit*/task*/results/*/scores.parquet")
