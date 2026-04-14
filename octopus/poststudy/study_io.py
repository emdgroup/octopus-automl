"""File I/O for reading study directories.

Provides ``StudyInfo`` for typed study-level metadata and flat functions
for loading per-split artifacts from disk.

All study I/O is centralized here so that consumers never construct
filesystem paths directly.
"""

from __future__ import annotations

import json
import re
import warnings
from typing import Any, Literal

import pandas as pd
from attrs import field, frozen
from upath import UPath

from octopus.types import MLType, ResultType
from octopus.utils import parquet_load

__all__ = [
    "StudyInfo",
    "discover_result_types",
    "find_latest_study",
    "load_study_information",
]

_SENTINEL = object()


@frozen
class StudyInfo:
    """Validated, immutable view of a completed study directory.

    Returned by ``load_study_information()``. Accepted by both analysis
    functions and predictor constructors. Does NOT store the raw config
    dict — all values are typed extractions.
    """

    path: UPath
    n_outer_splits: int
    workflow_tasks: tuple[dict[str, Any], ...]
    outersplit_dirs: tuple[UPath, ...]
    ml_type: MLType
    target_metric: str
    target_col: str
    target_assignments: dict[str, str]
    positive_class: Any
    row_id_col: str | None
    feature_cols: list[str]
    _outersplit_ids: tuple[int, ...] | None = field(default=None, alias="outersplit_ids")

    @property
    def outersplits(self) -> list[int]:
        """Outersplit IDs.

        Resolution order:
        1. Explicit ``outersplit_ids`` (set by ``OctoPredictor.load()``).
        2. Derived from ``outersplit_dirs`` (normal study path).
        3. ``range(n_outer_splits)`` (last resort).
        """
        if self._outersplit_ids is not None:
            return list(self._outersplit_ids)
        if self.outersplit_dirs:
            return [int(d.name.replace("outersplit", "")) for d in self.outersplit_dirs]
        return list(range(self.n_outer_splits))

    @property
    def n_outersplits(self) -> int:
        """Number of outer splits."""
        if self._outersplit_ids is not None:
            return len(self._outersplit_ids)
        if self.outersplit_dirs:
            return len(self.outersplit_dirs)
        return self.n_outer_splits


def _split_dir(study_path: UPath, split_id: int) -> UPath:
    return study_path / f"outersplit{split_id}"


def _task_result_dir(
    study_path: UPath,
    split_id: int,
    task_id: int,
    result_type: ResultType,
) -> UPath:
    return _split_dir(study_path, split_id) / f"task{task_id}" / "results" / str(result_type)


def _task_config_dir(study_path: UPath, split_id: int, task_id: int) -> UPath:
    return _split_dir(study_path, split_id) / f"task{task_id}" / "config"


def _load_json(path: UPath, *, default: Any = _SENTINEL) -> Any:
    """Load a JSON file, returning *default* if the file is missing."""
    if not path.exists():
        if default is not _SENTINEL:
            return default
        raise FileNotFoundError(f"File not found: {path}")
    with path.open() as f:
        return json.load(f)


def load_model(
    study_path: UPath,
    split_id: int,
    task_id: int,
    result_type: ResultType = ResultType.BEST,
) -> Any:
    """Load fitted model from ``result_dir/model/model.joblib``."""
    from octopus.utils import joblib_load  # noqa: PLC0415

    path = _task_result_dir(study_path, split_id, task_id, result_type) / "model" / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib_load(path)


def load_selected_features(
    study_path: UPath,
    split_id: int,
    task_id: int,
    result_type: ResultType = ResultType.BEST,
) -> list[str]:
    """Load ``selected_features.json`` from result_dir."""
    return _load_json(  # type: ignore[no-any-return]
        _task_result_dir(study_path, split_id, task_id, result_type) / "selected_features.json"
    )


def load_feature_cols(study_path: UPath, split_id: int, task_id: int) -> list[str]:
    """Load ``feature_cols.json`` (input features for this task)."""
    return _load_json(_task_config_dir(study_path, split_id, task_id) / "feature_cols.json", default=[])  # type: ignore[no-any-return]


def load_feature_groups(study_path: UPath, split_id: int, task_id: int) -> dict[str, list[str]]:
    """Load ``feature_groups.json`` (correlation-based groups)."""
    return _load_json(_task_config_dir(study_path, split_id, task_id) / "feature_groups.json", default={})  # type: ignore[no-any-return]


def load_scores(
    study_path: UPath,
    split_id: int,
    task_id: int,
    result_type: ResultType = ResultType.BEST,
) -> pd.DataFrame:
    """Load ``scores.parquet`` from result_dir."""
    path = _task_result_dir(study_path, split_id, task_id, result_type) / "scores.parquet"
    return parquet_load(path) if path.exists() else pd.DataFrame()


def load_partition(
    study_path: UPath,
    split_id: int,
    partition: Literal["traindev", "test"],
    prepared_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load one data partition by filtering prepared data with stored row IDs.

    Args:
        study_path: Path to the study directory.
        split_id: Outer-split index.
        partition: ``"traindev"`` or ``"test"``.
        prepared_data: Pre-loaded DataFrame. If None, loaded from disk.

    Returns:
        DataFrame for the requested partition.

    Raises:
        FileNotFoundError: If split_row_ids.json or data_prepared.parquet
            is missing.
        KeyError: If row IDs are not found in the prepared data.
        ValueError: If duplicate row IDs are detected.
    """
    if partition not in ("traindev", "test"):
        raise ValueError(f"partition must be 'traindev' or 'test', got {partition!r}")

    split_ids_path = _split_dir(study_path, split_id) / "split_row_ids.json"
    if not split_ids_path.exists():
        raise FileNotFoundError(f"Split row IDs not found: {split_ids_path}")

    with split_ids_path.open() as f:
        split_info: dict[str, Any] = json.load(f)
    row_ids: list = split_info[f"{partition}_row_ids"]
    row_id_col: str = split_info["row_id_col"]

    if prepared_data is None:
        prepared_path = study_path / "data_prepared.parquet"
        if not prepared_path.exists():
            raise FileNotFoundError(f"Prepared data not found: {prepared_path}")
        prepared_data = parquet_load(prepared_path)

    if len(set(row_ids)) != len(row_ids):
        n_dupes = len(row_ids) - len(set(row_ids))
        raise ValueError(
            f"{partition} split contains {n_dupes} duplicate row IDs. '{row_id_col}' must be unique within each split."
        )
    if prepared_data[row_id_col].duplicated().any():
        n_dupes = int(prepared_data[row_id_col].duplicated().sum())
        raise ValueError(f"data_prepared.parquet has {n_dupes} duplicate values in '{row_id_col}'.")

    indexed = prepared_data.set_index(row_id_col)
    missing = set(row_ids) - set(indexed.index)
    if missing:
        raise KeyError(
            f"{len(missing)} {partition} row IDs not found in data_prepared.parquet (first 5: {sorted(missing)[:5]})"
        )
    return indexed.loc[row_ids].reset_index()


def load_task_artifacts(
    study_path: UPath,
    outersplits: list[int],
    task_id: int,
    result_type: ResultType = ResultType.BEST,
) -> tuple[dict[int, Any], dict[int, list[str]], dict[int, dict[str, list[str]]]]:
    """Load models, feature columns, and feature groups for all splits.

    Args:
        study_path: Path to the study directory.
        outersplits: List of outersplit IDs.
        task_id: Workflow task index.
        result_type: Result type (default: ``'best'``).

    Returns:
        Tuple of ``(models, feature_cols_per_split, feature_groups_per_split)``.
    """
    models: dict[int, Any] = {}
    feature_cols_per_split: dict[int, list[str]] = {}
    feature_groups_per_split: dict[int, dict[str, list[str]]] = {}
    for split_id in outersplits:
        models[split_id] = load_model(study_path, split_id, task_id, result_type)
        feature_cols_per_split[split_id] = load_feature_cols(study_path, split_id, task_id)
        feature_groups_per_split[split_id] = load_feature_groups(study_path, split_id, task_id)
    return models, feature_cols_per_split, feature_groups_per_split


def load_study_config(study_path: str | UPath) -> dict[str, Any]:
    """Load ``study_config.json`` from the study directory."""
    path = UPath(study_path) / "study_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Study config not found: {path}")
    with path.open() as f:
        result: dict[str, Any] = json.load(f)
        return result


def find_latest_study(studies_root: str | UPath, prefix: str) -> str:
    """Find the latest study directory matching a name prefix.

    Study directories are named ``<prefix>-YYYYMMDD_HHMMSS``.  This function
    finds all directories matching the given *prefix* and returns the one with
    the most recent timestamp (lexicographic sort).  Falls back to an exact
    match (no timestamp suffix) when no timestamped directories are found.

    Args:
        studies_root: Path to the parent directory containing study directories.
        prefix: The study name prefix, e.g. ``"wf_octo_mrmr_octo"``.

    Returns:
        Path string to the latest matching study directory.

    Raises:
        FileNotFoundError: If no matching study directory is found.
    """
    root = UPath(studies_root)
    timestamp_pattern = re.compile(re.escape(prefix) + r"-\d{8}_\d{6}$")
    candidates = sorted(
        [d for d in root.glob(f"{prefix}-*") if d.is_dir() and timestamp_pattern.match(d.name)],
        key=lambda p: p.name,
        reverse=True,
    )
    if candidates:
        return str(candidates[0])
    exact = root / prefix
    if exact.is_dir():
        return str(exact)
    raise FileNotFoundError(f"Study directory not found for prefix '{prefix}' in {root}")


def load_study_information(study_directory: str | UPath) -> StudyInfo:
    """Load and validate a study directory.

    Reads ``study_config.json``, discovers outersplit directories,
    validates structure, and extracts typed metadata into a frozen
    ``StudyInfo``.

    Args:
        study_directory: Path to the study directory.

    Returns:
        Frozen ``StudyInfo`` with validated study metadata.

    Raises:
        ValueError: If no outersplit directories are found.
        FileNotFoundError: If the study directory or config does not exist.
    """
    study_path = UPath(study_directory)
    if not study_path.exists():
        raise FileNotFoundError(f"Study directory not found: {study_path}")

    config = load_study_config(study_path)

    n_outer_splits = config["n_outer_splits"]
    workflow_tasks = config["workflow"]

    outersplit_dirs = sorted(
        [d for d in study_path.glob("outersplit*") if d.is_dir()],
        key=lambda x: int(x.name.replace("outersplit", "")),
    )
    if not outersplit_dirs:
        raise ValueError(
            f"No outersplit directories found in study path.\n"
            f"Study path: {study_path}\nThe study may not have been run yet."
        )

    missing_outersplits = [i for i in range(n_outer_splits) if not (study_path / f"outersplit{i}").exists()]
    if missing_outersplits:
        warnings.warn(
            f"Missing outersplit directories: {missing_outersplits}\nStudy path: {study_path}",
            stacklevel=2,
        )

    task_ids = [t["task_id"] for t in workflow_tasks]
    missing_task_dirs = [
        f"{d.name}/task{tid}" for d in outersplit_dirs for tid in task_ids if not (d / f"task{tid}").exists()
    ]
    if missing_task_dirs:
        warnings.warn(
            f"Missing workflow task directories: {missing_task_dirs}\nStudy path: {study_path}",
            stacklevel=2,
        )

    prepared = config.get("prepared", {})

    return StudyInfo(
        path=study_path,
        n_outer_splits=n_outer_splits,
        workflow_tasks=tuple(workflow_tasks),
        outersplit_dirs=tuple(outersplit_dirs),
        ml_type=MLType(config["ml_type"]),
        target_metric=config.get("target_metric", ""),
        target_col=config.get("target_col", ""),
        target_assignments=prepared.get("target_assignments", {}),
        positive_class=config.get("positive_class"),
        row_id_col=prepared.get("row_id_col"),
        feature_cols=prepared.get("feature_cols", []),
    )


def discover_result_types(
    outersplit_dirs: tuple[UPath, ...] | list[UPath],
    task_id: int,
    artifact: str = "selected_features.json",
) -> list[str]:
    """Discover result_type directories containing *artifact*, checking all outersplits.

    Takes the union across all outersplits so that an incomplete first split
    does not cause result types to be silently skipped.

    Args:
        outersplit_dirs: List of outersplit directory paths.
        task_id: Workflow task index.
        artifact: Filename to look for inside each result_type directory.

    Returns:
        Sorted list of result_type names.
    """
    found: set[str] = set()
    for split_dir in outersplit_dirs:
        results_dir = split_dir / f"task{task_id}" / "results"
        if not results_dir.exists():
            continue
        for d in results_dir.iterdir():
            if d.is_dir() and (d / artifact).exists():
                found.add(d.name)
    return sorted(found)
