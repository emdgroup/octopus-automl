"""Data loading utilities for diagnostics.

Provides :func:`load_parquet_glob` as a generic glob-based parquet reader,
:func:`load_optuna` for Optuna trial results, and
:func:`load_feature_importances` for saved feature importance parquet files.
"""

from __future__ import annotations

import re
import warnings
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
            (e.g. ``"outersplit*/task*/results/optuna_results.parquet"``).

    Returns:
        Concatenated DataFrame with ``outersplit_id`` and ``task_id``
        columns added from directory names (unless already present in
        the parquet data). Empty DataFrame if no files match.
    """
    dfs: list[pd.DataFrame] = []
    for parquet_file in sorted(study_path.glob(pattern)):
        try:
            df = parquet_load(parquet_file)
        except Exception:
            warnings.warn(
                f"Failed to read parquet file, skipping: {parquet_file}",
                stacklevel=2,
            )
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


def load_optuna(study_path: UPath) -> pd.DataFrame:
    """Load all Optuna parquet files across outersplits and tasks.

    Args:
        study_path: Root path of the study directory.

    Returns:
        Combined Optuna results DataFrame.
    """
    return load_parquet_glob(study_path, "outersplit*/task*/results/optuna_results.parquet")


def load_feature_importances(study_path: UPath) -> pd.DataFrame:
    """Load all saved feature importance parquet files across outersplits and tasks.

    Reads ``feature_importances.parquet`` from both ``best/`` and
    ``ensemble_selection/`` result directories.

    Args:
        study_path: Root path of the study directory.

    Returns:
        Combined feature importances DataFrame with columns including
        ``feature``, ``importance``, ``fi_method``, ``fi_dataset``,
        ``training_id``, ``module``, ``result_type``, plus injected
        ``outersplit_id`` and ``task_id``.
    """
    return load_parquet_glob(study_path, "outersplit*/task*/results/*/feature_importances.parquet")
