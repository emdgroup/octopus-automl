"""Study I/O: load config, resolve paths, validate study directories.

The ``StudyInfo`` dataclass and ``load_study`` entry-point are the public
interface for post-hoc analysis.  ``load_config`` is used by the predictor
classes to read study configuration.
"""

from __future__ import annotations

import json
import re
from typing import Any

from attrs import frozen
from upath import UPath

__all__ = [
    "StudyInfo",
    "load_config",
    "load_study",
]


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
    outersplit_dirs: tuple[UPath, ...]


def _resolve_study_path(path: UPath) -> UPath:
    """Resolve a study path, searching for the latest timestamped match if needed.

    If *path* is an existing directory, return it as-is.  Otherwise, treat the
    last component as a name prefix and search the parent directory for
    ``<prefix>-YYYYMMDD_HHMMSS`` directories, returning the latest one.

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


def load_study(study_directory: str | UPath) -> StudyInfo:
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
        ValueError: If outersplit directories or workflow task directories
            are missing.
        FileNotFoundError: If the study directory does not exist and no
            timestamped match is found.
    """
    path_study = _resolve_study_path(UPath(study_directory))

    config = load_config(path_study)

    workflow_tasks = config["workflow"]
    n_folds_outer = config["n_folds_outer"]

    outersplit_dirs = sorted(
        [d for d in path_study.glob("outersplit*") if d.is_dir()],
        key=lambda x: int(x.name.replace("outersplit", "")),
    )
    if not outersplit_dirs:
        raise ValueError(
            f"No outersplit directories found in study path.\n"
            f"Study path: {path_study}\nThe study may not have been run yet."
        )

    missing_outersplits = [i for i in range(n_folds_outer) if not (path_study / f"outersplit{i}").exists()]
    if missing_outersplits:
        raise ValueError(f"Missing outersplit directories: {missing_outersplits}\nStudy path: {path_study}")

    task_ids = [task["task_id"] for task in workflow_tasks]
    missing_workflow_dirs: list[str] = []

    for split_dir in outersplit_dirs:
        for task_id in task_ids:
            if not (split_dir / f"task{task_id}").exists():
                missing_workflow_dirs.append(f"{split_dir.name}/task{task_id}")

    if missing_workflow_dirs:
        raise ValueError(f"Missing workflow task directories: {missing_workflow_dirs}\nStudy path: {path_study}")

    return StudyInfo(
        path=path_study,
        config=config,
        workflow_tasks=tuple(workflow_tasks),
        outersplit_dirs=tuple(outersplit_dirs),
    )
