"""Data loading abstractions for prediction module.

This module provides classes to load study data from the new parquet/joblib
file structure without relying on pickle-based storage classes.
"""

import json
from typing import Any

import pandas as pd
from attrs import define, field, validators
from upath import UPath

# Import will be done locally in load_task_modules to avoid circular dependency


@define
class OuterSplitLoader:
    """Loads data for a single outersplit (fold) from disk.

    Attributes:
        study_path: Path to the study directory
        outersplit_id: Outersplit (fold) ID
        task_id: Task ID within the workflow
        module: Module name to filter results (e.g., 'octo', 'autogluon')
        result_type: Result type to filter results (e.g., 'best', 'ensemble_selection')
    """

    study_path: UPath = field(converter=lambda x: UPath(x))
    outersplit_id: int
    task_id: int
    module: str = "octo"
    result_type: str = "best"

    @property
    def fold_dir(self) -> UPath:
        """Get the outersplit directory path."""
        return self.study_path / f"outersplit{self.outersplit_id}"

    @property
    def task_dir(self) -> UPath:
        """Get the task directory path.

        Returns:
            Path to task directory

        Raises:
            FileNotFoundError: If task directory doesn't exist
        """
        task_dir = self.fold_dir / f"task{self.task_id}"
        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")
        return task_dir

    @property
    def module_dir(self) -> UPath:
        """Get the module directory path (contains model and module state)."""
        module_dir = self.task_dir / "module"
        if not module_dir.exists():
            raise FileNotFoundError(f"Module directory not found: {module_dir}")
        return module_dir

    def load_test_data(self) -> pd.DataFrame:
        """Load test data for this outersplit.

        Returns:
            Test dataset as DataFrame

        Raises:
            FileNotFoundError: If test data file doesn't exist
        """
        data_test_path = self.fold_dir / "data_test.parquet"
        if not data_test_path.exists():
            raise FileNotFoundError(f"Test data not found: {data_test_path}")
        return pd.read_parquet(data_test_path, engine="pyarrow")

    def load_train_data(self) -> pd.DataFrame:
        """Load training data for this outersplit.

        Returns:
            Training dataset as DataFrame

        Raises:
            FileNotFoundError: If training data file doesn't exist
        """
        data_train_path = self.fold_dir / "data_train.parquet"
        if not data_train_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_train_path}")
        return pd.read_parquet(data_train_path, engine="pyarrow")

    def load_module_state(self) -> dict:
        """Load module fitted state (selected features, results).

        Returns:
            Module state dictionary with keys:
                - selected_features: List of selected feature names
                - results: Dictionary of module results

        Raises:
            FileNotFoundError: If module state file doesn't exist
        """
        state_path = self.module_dir / "module_state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"Module state not found: {state_path}")

        with open(state_path) as f:
            result: dict = json.load(f)
        return result

    def load_selected_features(self) -> list[str]:
        """Load selected features for this task.

        Returns:
            List of selected feature names

        Raises:
            FileNotFoundError: If selected_features.json doesn't exist
        """
        sf_path = self.task_dir / "selected_features.json"
        if not sf_path.exists():
            raise FileNotFoundError(f"Selected features not found: {sf_path}")
        with open(sf_path) as f:
            features: list[str] = json.load(f)
        return features

    def load_scores(self) -> pd.DataFrame:
        """Load scores DataFrame for this task.

        Returns:
            DataFrame with columns: result_type, metric, partition, aggregation, fold, value.
            Returns empty DataFrame if no scores file exists.
        """
        scores_path = self.task_dir / "scores.parquet"
        if scores_path.exists():
            return pd.read_parquet(scores_path, engine="pyarrow")
        return pd.DataFrame()

    def load_predictions(self) -> pd.DataFrame:
        """Load predictions for this task, optionally filtered by result_type.

        Returns:
            DataFrame with predictions including columns:
            - result_type: Result type identifier
            - row_id: Sample identifier
            - prediction: Model prediction
            - target: True target value
            - partition: Data partition ('train', 'dev', or 'test')
            - inner_split_id: Inner CV split ID
            - outer_split_id: Outer fold ID
            - task_id: Task ID

        Raises:
            FileNotFoundError: If predictions file doesn't exist
        """
        predictions_path = self.task_dir / "predictions.parquet"
        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions not found: {predictions_path}")
        df = pd.read_parquet(predictions_path, engine="pyarrow")
        mask = pd.Series(True, index=df.index)
        if "module" in df.columns and self.module:
            mask &= df["module"] == self.module
        if "result_type" in df.columns and self.result_type:
            mask &= df["result_type"] == self.result_type
        filtered = df[mask]
        return filtered if not filtered.empty else df

    def load_feature_importance(self) -> pd.DataFrame:
        """Load feature importance for this task, optionally filtered by result_type.

        Returns:
            DataFrame with columns: feature, importance, fi_method, fi_dataset, training_id, result_type

        Raises:
            FileNotFoundError: If feature importance file doesn't exist
        """
        fi_path = self.task_dir / "feature_importances.parquet"
        if not fi_path.exists():
            raise FileNotFoundError(f"Feature importance not found: {fi_path}")
        df = pd.read_parquet(fi_path, engine="pyarrow")
        mask = pd.Series(True, index=df.index)
        if "module" in df.columns and self.module:
            mask &= df["module"] == self.module
        if "result_type" in df.columns and self.result_type:
            mask &= df["result_type"] == self.result_type
        filtered = df[mask]
        return filtered if not filtered.empty else df

    def load_metrics(self) -> dict[str, float]:
        """Load performance metrics for this task and result_type.

        Returns:
            Dictionary of metric keys (e.g. "dev_avg", "test_refit") to values.
            Returns empty dict if no scores file exists.
        """
        scores_df = self.load_scores()
        if scores_df.empty:
            return {}

        if "module" in scores_df.columns and self.module:
            scores_df = scores_df[scores_df["module"] == self.module]
        if "result_type" in scores_df.columns and self.result_type:
            scores_df = scores_df[scores_df["result_type"] == self.result_type]

        metrics: dict[str, float] = {}
        for _, row in scores_df.iterrows():
            key = f"{row['partition']}_{row['aggregation']}"
            metrics[key] = row["value"]
        return metrics


@define
class StudyLoader:
    """Loads study-level data and coordinates loading across outersplits.

    This class provides high-level methods to access study configuration
    and create loaders for specific outersplits and tasks.

    Attributes:
        study_path: Path to the study directory
    """

    study_path: UPath = field(converter=lambda x: UPath(x), validator=[validators.instance_of(UPath)])

    def load_config(self) -> dict[str, Any]:
        """Load study configuration from config.json.

        Returns:
            Study configuration dictionary

        Raises:
            FileNotFoundError: If config.json doesn't exist
        """
        config_path = self.study_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Study config not found: {config_path}")

        with open(config_path) as f:
            config: dict[str, Any] = json.load(f)
        return config

    def get_outersplit_loader(
        self, outersplit_id: int, task_id: int, module: str = "octo", result_type: str = "best"
    ) -> OuterSplitLoader:
        """Create an OuterSplitLoader for a specific fold and task.

        Args:
            outersplit_id: Outersplit (fold) ID
            task_id: Task ID within the workflow
            module: Module name to filter results (default: 'octo')
            result_type: Result type to filter results (default: 'best')

        Returns:
            Configured OuterSplitLoader instance
        """
        return OuterSplitLoader(
            study_path=self.study_path,
            outersplit_id=outersplit_id,
            task_id=task_id,
            module=module,
            result_type=result_type,
        )

    def get_available_outersplits(self) -> list[int]:
        """Scan directory to find available outersplit IDs.

        Returns:
            Sorted list of available outersplit IDs
        """
        outersplit_dirs = sorted(
            [d for d in self.study_path.glob("outersplit*") if d.is_dir()],
            key=lambda x: int(x.name.replace("outersplit", "")),
        )

        return [int(d.name.replace("outersplit", "")) for d in outersplit_dirs]

    def get_task_directories(self, outersplit_id: int) -> list[tuple[int, UPath]]:
        """Get all task directories for a fold.

        Args:
            outersplit_id: Outersplit (fold) ID

        Returns:
            List of tuples (task_id, task_directory_path), sorted by task_id
        """
        fold_dir = self.study_path / f"outersplit{outersplit_id}"
        if not fold_dir.exists():
            return []

        task_dirs = []
        for task_dir in fold_dir.glob("task*"):
            if not task_dir.is_dir():
                continue
            task_id = int(task_dir.name.replace("task", ""))
            task_dirs.append((task_id, task_dir))

        return sorted(task_dirs, key=lambda x: x[0])
