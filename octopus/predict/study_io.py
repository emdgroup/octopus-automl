"""File I/O for reading study directories.

Provides StudyLoader and OuterSplitLoader classes for accessing study artifacts
from disk. These handle the actual directory structure where data files are at
the outersplit level and model artifacts are in task/module/ subdirectories.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from upath import UPath


class OuterSplitLoader:
    """Load data for a single outersplit from disk.

    Matches actual disk structure:
    - data_test.parquet, data_train.parquet at outersplit level
    - model.joblib, predictor.json inside task/module/ subdirectory
    - selected_features.json, scores.parquet at task level

    Args:
        study_path: Path to the study directory.
        outersplit_id: Outer split index.
        task_id: Workflow task index.
        module: Module name for filtering results (default: 'octo').
        result_type: Result type for filtering results (default: 'best').
    """

    def __init__(
        self,
        study_path: str | UPath,
        outersplit_id: int,
        task_id: int,
        module: str = "octo",
        result_type: str = "best",
    ) -> None:
        self.study_path = UPath(study_path)
        self.outersplit_id = outersplit_id
        self.task_id = task_id
        self.module = module
        self.result_type = result_type

    @property
    def fold_dir(self) -> UPath:
        """Outersplit directory path."""
        return self.study_path / f"outersplit{self.outersplit_id}"

    @property
    def task_dir(self) -> UPath:
        """Task directory path."""
        return self.fold_dir / f"task{self.task_id}"

    @property
    def module_dir(self) -> UPath:
        """Module artifact directory path."""
        return self.task_dir / "module"

    def load_test_data(self) -> pd.DataFrame:
        """Load test data (at outersplit level).

        Returns:
            DataFrame with test data.
        """
        return pd.read_parquet(self.fold_dir / "data_test.parquet")

    def load_train_data(self) -> pd.DataFrame:
        """Load train data (at outersplit level).

        Returns:
            DataFrame with train data.
        """
        return pd.read_parquet(self.fold_dir / "data_train.parquet")

    def load_model(self) -> Any:
        """Load fitted model from module/model.joblib.

        Returns:
            The deserialized fitted model object.
        """
        import joblib

        return joblib.load(self.module_dir / "model.joblib")

    def has_model(self) -> bool:
        """Check if this task has a fitted model.

        Returns:
            True if model.joblib exists in the module directory.
        """
        return (self.module_dir / "model.joblib").exists()

    def load_selected_features(self) -> list[str]:
        """Load selected_features.json from task directory.

        Returns:
            List of selected feature names, or empty list if not found.
        """
        path = self.task_dir / "selected_features.json"
        if not path.exists():
            return []
        with path.open() as f:
            return json.load(f)

    def load_feature_cols(self) -> list[str]:
        """Load feature_cols.json from module directory.

        These are the input feature columns used by this task (before feature
        selection). Saved by ``ModuleExecution.save()`` during study execution.

        Returns:
            List of input feature column names, or empty list if not found.
        """
        path = self.module_dir / "feature_cols.json"
        if not path.exists():
            return []
        with path.open() as f:
            return json.load(f)

    def load_feature_groups(self) -> dict[str, list[str]]:
        """Load feature_groups.json from module directory.

        These are correlation-based feature groups computed from training data.
        Saved by ``ModuleExecution.save()`` during study execution.

        Returns:
            Dict mapping group names to lists of feature names, or empty dict.
        """
        path = self.module_dir / "feature_groups.json"
        if not path.exists():
            return {}
        with path.open() as f:
            return json.load(f)

    def load_scores(self) -> pd.DataFrame:
        """Load scores.parquet.

        Returns:
            DataFrame with scores, or empty DataFrame if not found.
        """
        path = self.task_dir / "scores.parquet"
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    def load_predictions(self) -> pd.DataFrame:
        """Load predictions.parquet with optional filtering by module and result_type.

        Returns:
            DataFrame with predictions.
        """
        path = self.task_dir / "predictions.parquet"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        mask = pd.Series(True, index=df.index)
        if "module" in df.columns and self.module:
            mask &= df["module"] == self.module
        if "result_type" in df.columns and self.result_type:
            mask &= df["result_type"] == self.result_type
        filtered = df[mask]
        return filtered if not filtered.empty else df

    def load_feature_importance(self) -> pd.DataFrame:
        """Load feature_importances.parquet with optional filtering.

        Returns:
            DataFrame with feature importance data.
        """
        path = self.task_dir / "feature_importances.parquet"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        mask = pd.Series(True, index=df.index)
        if "module" in df.columns and self.module:
            mask &= df["module"] == self.module
        if "result_type" in df.columns and self.result_type:
            mask &= df["result_type"] == self.result_type
        filtered = df[mask]
        return filtered if not filtered.empty else df

    def load_task_config(self) -> dict[str, Any]:
        """Load task_config.json from task directory.

        Returns:
            Task configuration dictionary.
        """
        path = self.task_dir / "task_config.json"
        with path.open() as f:
            return json.load(f)


class StudyLoader:
    """Study-level data access for reading study configuration and structure.

    Used by study-level notebook functions (show_study_details,
    show_target_metric_performance, show_selected_features).

    Args:
        study_path: Path to the study directory.
    """

    def __init__(self, study_path: str | UPath) -> None:
        self.study_path = UPath(study_path)

    def load_config(self) -> dict[str, Any]:
        """Load study config.json.

        Returns:
            Study configuration dictionary.
        """
        with (self.study_path / "config.json").open() as f:
            return json.load(f)

    def get_outersplit_loader(
        self,
        outersplit_id: int,
        task_id: int,
        module: str = "octo",
        result_type: str = "best",
    ) -> OuterSplitLoader:
        """Get an OuterSplitLoader for a specific outersplit and task.

        Args:
            outersplit_id: Outer split index.
            task_id: Task index.
            module: Module name for filtering.
            result_type: Result type for filtering.

        Returns:
            OuterSplitLoader instance.
        """
        return OuterSplitLoader(self.study_path, outersplit_id, task_id, module, result_type)

    def get_available_outersplits(self) -> list[int]:
        """Get list of available outersplit IDs.

        Returns:
            Sorted list of outersplit IDs found on disk.
        """
        dirs = sorted(
            [d for d in self.study_path.glob("outersplit*") if d.is_dir()],
            key=lambda x: int(x.name.replace("outersplit", "")),
        )
        return [int(d.name.replace("outersplit", "")) for d in dirs]

    def get_task_directories(self, outersplit_id: int) -> list[tuple[int, UPath]]:
        """Get task directories for a given outersplit.

        Args:
            outersplit_id: Outer split index.

        Returns:
            Sorted list of (task_id, task_path) tuples.
        """
        fold_dir = self.study_path / f"outersplit{outersplit_id}"
        if not fold_dir.exists():
            return []
        task_dirs = []
        for task_dir in fold_dir.glob("task*"):
            if task_dir.is_dir():
                match = re.search(r"\d+$", task_dir.name)
                if match:
                    task_dirs.append((int(match.group()), task_dir))
        return sorted(task_dirs)
