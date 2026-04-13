"""Study I/O: load config, resolve paths, validate study directories.

The ``StudyInfo`` dataclass and ``load_study_info`` entry-point are the public
interface for post-hoc analysis.  ``load_config`` is used by the predictor
classes to read study configuration.
"""

from __future__ import annotations

import json
import re
import warnings
from typing import Any

import pandas as pd
from attrs import frozen
from upath import UPath

from octopus.types import ResultType

__all__ = [
    "StudyInfo",
    "load_config",
    "load_feature_cols",
    "load_feature_groups",
    "load_model",
    "load_prepared_data",
    "load_scores",
    "load_selected_features",
    "load_split_data",
    "load_study_info",
]


def _split_id(outer_split_dir: UPath) -> int:
    return int(outer_split_dir.name.removeprefix("outersplit"))


def _split_dir(study: StudyInfo, split_id: int) -> UPath:
    return study.path / f"outersplit{split_id}"


def _task_result_dir(study: StudyInfo, split_id: int, task_id: int, result_type: ResultType) -> UPath:
    return _split_dir(study, split_id) / f"task{task_id}" / "results" / str(result_type)


def _task_config_dir(study: StudyInfo, split_id: int, task_id: int) -> UPath:
    return _split_dir(study, split_id) / f"task{task_id}" / "config"


def load_model(study: StudyInfo, split_id: int, task_id: int, result_type: ResultType = ResultType.BEST) -> Any:
    """Load model.joblib for a task within an outer split.

    Args:
        study: Validated study (from ``load_study_info()``).
        split_id: Outer-split index.
        task_id: Workflow task index.
        result_type: Result type (default ``"best"``).

    Returns:
        The deserialized model object.
    """
    from octopus.utils import joblib_load  # noqa: PLC0415

    model_path = _task_result_dir(study, split_id, task_id, result_type) / "model" / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Has the study completed successfully?")
    return joblib_load(model_path)


def load_selected_features(
    study: StudyInfo, split_id: int, task_id: int, result_type: ResultType = ResultType.BEST
) -> list[str]:
    """Load selected_features.json for a task within an outer split.

    Args:
        study: Validated study (from ``load_study_info()``).
        split_id: Outer-split index.
        task_id: Workflow task index.
        result_type: Result type (default ``"best"``).

    Returns:
        List of selected feature names.
    """
    sf_path = _task_result_dir(study, split_id, task_id, result_type) / "selected_features.json"
    if not sf_path.exists():
        raise FileNotFoundError(f"Selected features not found: {sf_path}")
    with sf_path.open() as f:
        result: list[str] = json.load(f)
    return result


def load_feature_cols(study: StudyInfo, split_id: int, task_id: int) -> list[str]:
    """Load feature_cols.json for a task within an outer split.

    Args:
        study: Validated study (from ``load_study_info()``).
        split_id: Outer-split index.
        task_id: Workflow task index.

    Returns:
        List of feature column names.
    """
    fc_path = _task_config_dir(study, split_id, task_id) / "feature_cols.json"
    if not fc_path.exists():
        raise FileNotFoundError(f"Feature columns not found: {fc_path}")
    with fc_path.open() as f:
        result: list[str] = json.load(f)
    return result


def load_feature_groups(study: StudyInfo, split_id: int, task_id: int) -> dict[str, list[str]]:
    """Load feature_groups.json for a task within an outer split, returning {} if not found.

    Args:
        study: Validated study (from ``load_study_info()``).
        split_id: Outer-split index.
        task_id: Workflow task index.

    Returns:
        Mapping of group name to feature lists, or ``{}`` if not found.
    """
    fg_path = _task_config_dir(study, split_id, task_id) / "feature_groups.json"
    if not fg_path.exists():
        return {}
    with fg_path.open() as f:
        result: dict[str, list[str]] = json.load(f)
    return result


def load_scores(
    study: StudyInfo, split_id: int, task_id: int, result_type: ResultType = ResultType.BEST
) -> pd.DataFrame:
    """Load scores.parquet for a task within an outer split.

    Args:
        study: Validated study (from ``load_study_info()``).
        split_id: Outer-split index.
        task_id: Workflow task index.
        result_type: Result type (default ``"best"``).

    Returns:
        DataFrame with scores.

    Raises:
        FileNotFoundError: If scores.parquet is missing.
    """
    scores_path = _task_result_dir(study, split_id, task_id, result_type) / "scores.parquet"
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores not found: {scores_path}")
    return pd.read_parquet(scores_path)


def load_prepared_data(study: StudyInfo) -> pd.DataFrame:
    """Load data_prepared.parquet from a study directory.

    Args:
        study: Validated study (from ``load_study_info()``).

    Returns:
        The prepared data DataFrame.

    Raises:
        FileNotFoundError: If data_prepared.parquet is missing.
    """
    from octopus.utils import parquet_load  # noqa: PLC0415

    prepared_path = study.path / "data_prepared.parquet"
    if not prepared_path.exists():
        raise FileNotFoundError(f"Prepared data not found: {prepared_path}")
    return parquet_load(prepared_path)


def load_split_data(
    study: StudyInfo, split_id: int, *, prepared_data: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load test/train data via split_row_ids.json + data_prepared.parquet.

    Args:
        study: Validated study (from ``load_study_info()``).
        split_id: Outer-split index.
        prepared_data: Pre-loaded prepared data to avoid repeated parquet reads.
            If None, data_prepared.parquet is read from disk.

    Returns:
        Tuple of (test_data, train_data).
    """
    outer_split_dir = _split_dir(study, split_id)
    split_ids_path = outer_split_dir / "split_row_ids.json"
    if not split_ids_path.exists():
        raise FileNotFoundError(f"Split row IDs not found: {split_ids_path}")

    if prepared_data is None:
        prepared_data = load_prepared_data(study)

    with split_ids_path.open() as f:
        split_info: dict[str, Any] = json.load(f)
    row_id_col: str = split_info["row_id_col"]
    indexed = prepared_data.set_index(row_id_col)

    test_data = indexed.loc[split_info["test_row_ids"]].reset_index()
    train_data = indexed.loc[split_info["traindev_row_ids"]].reset_index()
    return test_data, train_data


def load_config(study_path: str | UPath) -> dict[str, Any]:
    """Load study_config.json from a study directory.

    Args:
        study_path: Path to the study directory.

    Returns:
        Study configuration dictionary.

    Raises:
        FileNotFoundError: If study_config.json is missing.
    """
    path = UPath(study_path) / "study_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Study config not found at {path}")
    with path.open() as f:
        result: dict[str, Any] = json.load(f)
        return result


@frozen
class StudyInfo:
    """Validated, loaded view of a completed study directory."""

    path: UPath
    config: dict[str, Any]
    workflow_tasks: tuple[dict[str, Any], ...]
    outer_split_dirs: tuple[UPath, ...]


def _resolve_study_path(path: UPath) -> UPath:
    """Resolve a study path, searching for the latest timestamped match if needed.

    If *path* is an existing directory, return it as-is.  Otherwise, treat the
    last component as a name prefix and search the parent directory for
    ``<prefix>-YYYYMMDD_HHMMSS`` directories, returning the latest one.

    Args:
        path: Study directory path or name prefix.

    Returns:
        Resolved path to the study directory.

    Raises:
        FileNotFoundError: If the path does not exist and no timestamped match
            is found in the parent directory.
    """
    if path.is_dir():
        return path

    root = path.parent
    prefix = path.name
    timestamp_pattern = re.compile(re.escape(prefix) + r"-\d{8}_\d{6}$")
    candidates = sorted(
        [d for d in root.glob(f"{prefix}-*") if d.is_dir() and timestamp_pattern.match(d.name)],
        key=lambda p: p.name,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"No study directory found for '{prefix}' in {root}")


def load_study_info(study_directory: str | UPath) -> StudyInfo:
    """Load and validate a completed study directory.

    Accepts either a full path to a study directory or a path whose last
    component is a name prefix (e.g. ``"./studies/wf_octo_mrmr_octo"``).
    In the latter case the latest timestamped directory matching
    ``<prefix>-YYYYMMDD_HHMMSS`` is used automatically.

    Args:
        study_directory: Path to the study directory, or a path ending in a
            study name prefix.

    Returns:
        A ``StudyInfo`` instance containing study metadata and validation results.

    Raises:
        ValueError: If no outersplit directories exist at all (study not run).
        FileNotFoundError: If the study directory does not exist and no
            timestamped match is found.
    """
    path_study = _resolve_study_path(UPath(study_directory))

    config = load_config(path_study)

    workflow_tasks = config["workflow"]
    n_outer_splits = config["n_outer_splits"]

    outer_split_dirs = sorted(
        [d for d in path_study.glob("outersplit*") if d.is_dir()],
        key=_split_id,
    )
    if not outer_split_dirs:
        raise ValueError(
            f"No outersplit directories found in study path.\n"
            f"Study path: {path_study}\nThe study may not have been run yet."
        )

    missing_outersplits = [i for i in range(n_outer_splits) if not (path_study / f"outersplit{i}").exists()]
    if missing_outersplits:
        warnings.warn(f"Missing outersplit directories: {missing_outersplits}\nStudy path: {path_study}", stacklevel=2)

    task_ids = [task["task_id"] for task in workflow_tasks]
    missing_workflow_dirs: list[str] = []

    for split_dir in outer_split_dirs:
        for task_id in task_ids:
            if not (split_dir / f"task{task_id}").exists():
                missing_workflow_dirs.append(f"{split_dir.name}/task{task_id}")

    if missing_workflow_dirs:
        warnings.warn(
            f"Missing workflow task directories: {missing_workflow_dirs}\nStudy path: {path_study}", stacklevel=2
        )

    return StudyInfo(
        path=path_study,
        config=config,
        workflow_tasks=tuple(workflow_tasks),
        outer_split_dirs=tuple(outer_split_dirs),
    )
