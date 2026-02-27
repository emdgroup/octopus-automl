"""File I/O for reading study directories.

Provides StudyLoader and TaskOutersplitLoader classes for accessing study artifacts
from disk. These handle the actual directory structure where data files are at
the outersplit level and model artifacts are in task/module/ subdirectories.

Also provides frozen data classes (StudyMetadata, SplitArtifacts, TaskArtifacts)
for type-safe, immutable data bundles at the boundary between I/O and prediction.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from attrs import field, frozen
from upath import UPath

__all__ = [
    "SplitArtifacts",
    "StudyLoader",
    "StudyMetadata",
    "TaskArtifacts",
    "TaskOutersplitLoader",
]


def _to_upath(value: str | UPath) -> UPath:
    """Convert a string or UPath to UPath."""
    return UPath(value)


# ═══════════════════════════════════════════════════════════════
# Frozen data classes — immutable bundles at the I/O ↔ prediction boundary
# ═══════════════════════════════════════════════════════════════


@frozen
class StudyMetadata:
    """Immutable study-level metadata extracted from config.

    Attributes:
        ml_type: Machine learning type (classification, regression, etc.).
        target_metric: Primary metric name.
        target_col: Target column name.
        target_assignments: Target column assignments from prepared config.
        positive_class: Positive class label for classification.
        row_id_col: Row ID column name.
        feature_cols: Union of feature columns across outer splits.
        n_outersplits: Number of outer folds from config.
    """

    ml_type: str
    target_metric: str
    target_col: str
    target_assignments: dict[str, str]
    positive_class: Any
    row_id_col: str | None
    feature_cols: list[str]
    n_outersplits: int


@frozen
class SplitArtifacts:
    """All artifacts loaded for one outersplit.

    Attributes:
        model: The fitted model object.
        selected_features: List of selected feature names.
        feature_cols: Input feature columns (before selection).
        feature_groups: Feature groups dict (group name → feature list).
    """

    model: Any
    selected_features: list[str]
    feature_cols: list[str]
    feature_groups: dict[str, list[str]]


@frozen
class TaskArtifacts:
    """All artifacts for a task across all outersplits.

    Attributes:
        splits: Dict mapping outersplit_id to SplitArtifacts.
        outersplit_ids: Sorted list of outersplit IDs.
    """

    splits: dict[int, SplitArtifacts]
    outersplit_ids: list[int]


# ═══════════════════════════════════════════════════════════════
# TaskOutersplitLoader — frozen path resolver + artifact loader
# ═══════════════════════════════════════════════════════════════


@frozen
class TaskOutersplitLoader:
    """Load data for a single outersplit from disk.

    Matches actual disk structure:
    - data_test.parquet, data_train.parquet at outersplit level
    - feature_cols.json, feature_groups.json inside task/module/
    - model.joblib, predictor.json inside task/{result_type}/model/
    - selected_features.json, scores.parquet, predictions.parquet,
      feature_importances.parquet inside task/{result_type}/

    Attributes:
        study_path: Path to the study directory.
        outersplit_id: Outer split index.
        task_id: Workflow task index.
        result_type: Result type for filtering results (default: 'best').
    """

    study_path: UPath = field(converter=_to_upath)
    outersplit_id: int
    task_id: int
    result_type: str = "best"

    @property
    def fold_dir(self) -> UPath:
        """Outersplit directory path."""
        return self.study_path / f"outersplit{self.outersplit_id}"

    @property
    def task_dir(self) -> UPath:
        """Task directory path."""
        return self.fold_dir / f"task{self.task_id}"

    @property
    def result_dir(self) -> UPath:
        """Result type directory path (e.g. task0/best/)."""
        return self.task_dir / self.result_type

    @property
    def module_dir(self) -> UPath:
        """Module artifact directory path (feature_cols, feature_groups)."""
        return self.task_dir / "module"

    @property
    def model_dir(self) -> UPath:
        """Model directory path inside result_dir (model.joblib, predictor.json)."""
        return self.result_dir / "model"

    # ── Validation ──────────────────────────────────────────────

    def validate_directories(self) -> None:
        """Check that fold_dir and task_dir exist on disk.

        Raises:
            FileNotFoundError: If the outer split or task directory is missing.
        """
        if not self.fold_dir.exists():
            raise FileNotFoundError(f"Outer split directory not found: {self.fold_dir}")
        if not self.task_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {self.task_dir}")

    # ── Artifact loading ────────────────────────────────────────

    def load_all_artifacts(self) -> SplitArtifacts:
        """Load all artifacts for this outersplit in one call.

        Validates directories, loads required artifacts (model,
        selected_features), and optional artifacts (feature_cols,
        feature_groups).

        Returns:
            SplitArtifacts with model, selected_features, feature_cols,
            feature_groups.

        Raises:
            FileNotFoundError: If directories or required files are missing.
        """
        self.validate_directories()
        return SplitArtifacts(
            model=self.load_model(),
            selected_features=self.load_selected_features(),
            feature_cols=self.load_feature_cols(),
            feature_groups=self.load_feature_groups(),
        )

    def load_test_data(self) -> pd.DataFrame:
        """Load test data (at outersplit level).

        Returns:
            DataFrame with test data.

        Raises:
            FileNotFoundError: If test data file is missing.
        """
        path = self.fold_dir / "data_test.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Test data not found: {path}")
        return pd.read_parquet(path)

    def load_train_data(self) -> pd.DataFrame:
        """Load train data (at outersplit level).

        Returns:
            DataFrame with train data.

        Raises:
            FileNotFoundError: If train data file is missing.
        """
        path = self.fold_dir / "data_train.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Train data not found: {path}")
        return pd.read_parquet(path)

    def load_model(self) -> Any:
        """Load fitted model from result_dir/model/model.joblib.

        Returns:
            The deserialized fitted model object.

        Raises:
            FileNotFoundError: If model file is missing.
        """
        import joblib  # noqa: PLC0415

        path = self.model_dir / "model.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}. Has the study completed successfully?")
        return joblib.load(path)

    def has_model(self) -> bool:
        """Check if this task has a fitted model.

        Returns:
            True if model.joblib exists in the model directory.
        """
        return bool((self.model_dir / "model.joblib").exists())

    def load_selected_features(self) -> list[str]:
        """Load selected_features.json from result_dir.

        Returns:
            List of selected feature names.

        Raises:
            FileNotFoundError: If selected features file is missing.
        """
        path = self.result_dir / "selected_features.json"
        if not path.exists():
            raise FileNotFoundError(f"Selected features not found: {path}")
        with path.open() as f:
            result: list[str] = json.load(f)
            return result

    def load_feature_cols(self) -> list[str]:
        """Load feature_cols.json from module directory.

        These are the input feature columns used by this task (before feature
        selection). Saved by ``WorkflowTaskRunner._save_task_context()``
        during study execution.

        Returns:
            List of input feature column names, or empty list if not found.
        """
        path = self.module_dir / "feature_cols.json"
        if not path.exists():
            return []
        with path.open() as f:
            result: list[str] = json.load(f)
            return result

    def load_feature_groups(self) -> dict[str, list[str]]:
        """Load feature_groups.json from module directory.

        These are correlation-based feature groups computed from training data.
        Saved by ``WorkflowTaskRunner._save_task_context()`` during study
        execution.

        Returns:
            Dict mapping group names to lists of feature names, or empty dict.
        """
        path = self.module_dir / "feature_groups.json"
        if not path.exists():
            return {}
        with path.open() as f:
            result: dict[str, list[str]] = json.load(f)
            return result

    def load_scores(self) -> pd.DataFrame:
        """Load scores.parquet from result_dir.

        Returns:
            DataFrame with scores, or empty DataFrame if not found.
        """
        path = self.result_dir / "scores.parquet"
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    def load_predictions(self) -> pd.DataFrame:
        """Load predictions.parquet from result_dir.

        Returns:
            DataFrame with predictions.
        """
        path = self.result_dir / "predictions.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_feature_importance(self) -> pd.DataFrame:
        """Load feature_importances.parquet from result_dir.

        Returns:
            DataFrame with feature importance data.
        """
        path = self.result_dir / "feature_importances.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_task_config(self) -> dict[str, Any]:
        """Load task_config.json from task directory.

        Returns:
            Task configuration dictionary.
        """
        path = self.task_dir / "task_config.json"
        with path.open() as f:
            result: dict[str, Any] = json.load(f)
            return result


# ═══════════════════════════════════════════════════════════════
# StudyLoader — frozen study-level reader with validation + aggregation
# ═══════════════════════════════════════════════════════════════


@frozen
class StudyLoader:
    """Study-level data access for reading study configuration and structure.

    Used by study-level notebook functions (show_study_details,
    show_target_metric_performance, show_selected_features) and by
    TaskPredictor for loading study artifacts.

    Attributes:
        study_path: Path to the study directory.
    """

    study_path: UPath = field(converter=_to_upath)

    # ── Config + validation ─────────────────────────────────────

    def load_config(self) -> dict[str, Any]:
        """Load study config.json.

        Returns:
            Study configuration dictionary.

        Raises:
            FileNotFoundError: If config.json is missing.
        """
        path = self.study_path / "config.json"
        if not path.exists():
            raise FileNotFoundError(f"Study config not found at {path}")
        with path.open() as f:
            result: dict[str, Any] = json.load(f)
            return result

    def validate_task_id(self, task_id: int, config: dict[str, Any]) -> None:
        """Validate task_id against the study config.

        Args:
            task_id: Task index to validate.
            config: Study configuration dictionary.

        Raises:
            ValueError: If task_id is negative or out of range.
        """
        if task_id < 0:
            raise ValueError(f"task_id must be >= 0, got {task_id}")

        workflow = config.get("workflow", [])
        if task_id >= len(workflow):
            raise ValueError(f"task_id {task_id} out of range, study has {len(workflow)} tasks")

    def extract_metadata(self, config: dict[str, Any]) -> StudyMetadata:
        """Extract study-level metadata from config into a frozen data class.

        Args:
            config: Study configuration dictionary.

        Returns:
            StudyMetadata with all study-level properties.
        """
        row_id_col = config.get("prepared", {}).get("row_id_col")
        if not row_id_col:
            row_id_col = config.get("row_id_col") or "row_id"

        return StudyMetadata(
            ml_type=config.get("ml_type", ""),
            target_metric=config.get("target_metric", ""),
            target_col=config.get("target_col", ""),
            target_assignments=config.get("prepared", {}).get("target_assignments", {}),
            positive_class=config.get("positive_class"),
            row_id_col=row_id_col,
            feature_cols=config.get("feature_cols", []),
            n_outersplits=config.get("n_folds_outer", 0),
        )

    def load_task_artifacts(
        self,
        task_id: int,
        result_type: str,
        n_outersplits: int,
    ) -> TaskArtifacts:
        """Load all models and features for a task across all outer splits.

        Validates directories exist, loads models and features, and bundles
        them into an immutable TaskArtifacts.

        Args:
            task_id: Concrete task index.
            result_type: Result type for filtering.
            n_outersplits: Number of outer folds from config.

        Returns:
            TaskArtifacts with per-split models and features.

        Raises:
            FileNotFoundError: If any expected directory or artifact is missing.
            ValueError: If no models are found.
        """
        splits: dict[int, SplitArtifacts] = {}
        outersplit_ids: list[int] = []

        for split_id in range(n_outersplits):
            loader = TaskOutersplitLoader(self.study_path, split_id, task_id, result_type)
            splits[split_id] = loader.load_all_artifacts()
            outersplit_ids.append(split_id)

        if not outersplit_ids:
            raise ValueError(f"No models found for task {task_id}. Check that the study has been run.")

        return TaskArtifacts(splits=splits, outersplit_ids=outersplit_ids)

    # ── Outersplit loader factory ───────────────────────────────

    def get_outersplit_loader(
        self,
        outersplit_id: int,
        task_id: int,
        result_type: str = "best",
    ) -> TaskOutersplitLoader:
        """Get a TaskOutersplitLoader for a specific outersplit and task.

        Args:
            outersplit_id: Outer split index.
            task_id: Task index.
            result_type: Result type for filtering.

        Returns:
            TaskOutersplitLoader instance.
        """
        return TaskOutersplitLoader(self.study_path, outersplit_id, task_id, result_type)

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

    # ── Aggregation (used by notebook_utils) ────────────────────

    def build_performance_summary(self) -> pd.DataFrame:
        """Aggregate scores across all outer splits and tasks.

        Iterates over all outersplit directories and task directories found on
        disk, loading scores and selected features for each combination.

        Returns:
            DataFrame with columns: OuterSplit, Task, Task_name, Module,
            Results_key, Performance_dict, n_features, Selected_features.
            Sorted by Task, OuterSplit.
        """
        _ = self.load_config()  # Validate config exists
        rows_list: list[dict[str, Any]] = []

        available_splits = self.get_available_outersplits()

        for split_num in available_splits:
            task_dirs = self.get_task_directories(split_num)

            for workflow_num, path_workflow in task_dirs:
                workflow_name = str(path_workflow.name)

                try:
                    loader = self.get_outersplit_loader(outersplit_id=split_num, task_id=workflow_num)
                    try:
                        selected_features = loader.load_selected_features()
                    except FileNotFoundError:
                        selected_features = []

                    perf_df = loader.load_scores()

                    # Filter out per_fold rows — only keep avg and pool aggregations
                    if not perf_df.empty and "aggregation" in perf_df.columns:
                        perf_df = perf_df[perf_df["aggregation"] != "per_fold"]

                    if not perf_df.empty and "result_type" in perf_df.columns:
                        group_cols = ["result_type"]
                        if "module" in perf_df.columns:
                            group_cols = ["module", "result_type"]

                        unique_combos = perf_df[group_cols].drop_duplicates()
                        for _, combo in unique_combos.iterrows():
                            mask = pd.Series(True, index=perf_df.index)
                            for col in group_cols:
                                mask &= perf_df[col] == combo[col]
                            combo_perf = perf_df[mask]
                            performance_dict = {}
                            for _, row in combo_perf.iterrows():
                                perf_key = f"{row['partition']}_{row['aggregation']}"
                                performance_dict[perf_key] = row["value"]

                            module_name = combo.get("module", "") if "module" in group_cols else ""
                            rows_list.append(
                                {
                                    "OuterSplit": split_num,
                                    "Task": workflow_num,
                                    "Task_name": workflow_name,
                                    "Module": module_name,
                                    "Results_key": str(combo["result_type"]),
                                    "Performance_dict": performance_dict,
                                    "n_features": len(selected_features),
                                    "Selected_features": sorted(selected_features),
                                }
                            )
                    else:
                        rows_list.append(
                            {
                                "OuterSplit": split_num,
                                "Task": workflow_num,
                                "Task_name": workflow_name,
                                "Module": "",
                                "Results_key": "",
                                "Performance_dict": {},
                                "n_features": len(selected_features),
                                "Selected_features": sorted(selected_features),
                            }
                        )

                except (FileNotFoundError, KeyError) as e:
                    print(f"Warning: Could not load data for {workflow_name} in outersplit{split_num}: {e}")
                    continue

        df = pd.DataFrame(
            rows_list,
            columns=[
                "OuterSplit",
                "Task",
                "Task_name",
                "Module",
                "Results_key",
                "Performance_dict",
                "n_features",
                "Selected_features",
            ],
        )
        df = df.sort_values(by=["Task", "OuterSplit"], ignore_index=True)
        return df

    def build_feature_summary(
        self,
        sort_task: int | None = None,
        sort_key: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregate feature selection info across splits and tasks.

        Args:
            sort_task: Task ID to use for sorting the frequency table.
            sort_key: Results key to use for sorting the frequency table.

        Returns:
            Tuple of two DataFrames:
            - feature_table: Number of features per outer split for each
              task-key combination (with Mean row).
            - frequency_table: Feature frequency across outersplits.
        """
        raw = self.build_performance_summary()

        # Feature count table
        feature_table = raw.pivot_table(
            index="OuterSplit",
            columns=["Task", "Results_key"],
            values="n_features",
            aggfunc="first",
        )
        mean_row = feature_table.mean(axis=0)
        feature_table.loc["Mean"] = mean_row
        feature_table = feature_table.astype(int)
        feature_table.index.name = "OuterSplit"

        # Determine sort defaults
        task_key_combinations = raw[["Task", "Results_key"]].drop_duplicates().sort_values(["Task", "Results_key"])

        if sort_task is None:
            sort_task = int(task_key_combinations.iloc[0]["Task"])
            sort_key = str(task_key_combinations.iloc[0]["Results_key"])
        elif sort_key is None:
            task_keys = raw[raw["Task"] == sort_task]["Results_key"].unique()
            sort_key = str(task_keys[0]) if len(task_keys) > 0 else None

        # Feature frequency table
        frequency_data: dict[tuple[int, str], dict[str, int]] = {}
        for _, row in task_key_combinations.iterrows():
            task = int(row["Task"])
            key = str(row["Results_key"])
            task_key = (task, key)
            frequency_data[task_key] = {}

            task_key_data = raw[(raw["Task"] == task) & (raw["Results_key"] == key)]
            for _, data_row in task_key_data.iterrows():
                for feature in data_row["Selected_features"]:
                    if feature not in frequency_data[task_key]:
                        frequency_data[task_key][feature] = 0
                    frequency_data[task_key][feature] += 1

        frequency_table = pd.DataFrame(frequency_data)
        frequency_table = frequency_table.fillna(0).astype(int)

        if sort_key is not None:
            sort_col = (sort_task, sort_key)
            if sort_col in frequency_table.columns:
                frequency_table = frequency_table.sort_values(
                    by=[sort_col],  # type: ignore[list-item]
                    ascending=False,
                )

        frequency_table.index.name = "Feature"

        return feature_table, frequency_table
