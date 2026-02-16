"""Workflow task runner for processing tasks."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
import ray
from attrs import define, field, validators
from upath import UPath

from octopus.logger import get_logger
from octopus.modules.base import Task
from octopus.utils import calculate_feature_groups

if TYPE_CHECKING:
    from octopus.study.core import OctoStudy

logger = get_logger()


@define
class WorkflowTaskRunner:
    """Runs workflow tasks for a single fold.

    Handles the lifecycle of processing workflow tasks:
    - Saving fold data
    - Running tasks with dependencies
    - Saving task results

    Attributes:
        study: OctoStudy instance containing ML configuration (includes workflow, log_dir, etc.)
        cpus_per_outersplit: Number of CPUs allocated to each task
    """

    study: OctoStudy = field(validator=[validators.instance_of(object)])  # type: ignore[assignment]
    cpus_per_outersplit: int = field(validator=[validators.instance_of(int)])

    def run(self, outersplit_id: int, data_train: pd.DataFrame, data_test: pd.DataFrame) -> None:
        """Process all workflow tasks for a single fold.

        Args:
            outersplit_id: Current fold ID
            data_train: Training DataFrame
            data_test: Test DataFrame

        Raises:
            RuntimeError: If Ray is not initialized.
        """
        if not ray.is_initialized():
            raise RuntimeError(
                "Ray is not initialized. WorkflowTaskRunner.run() must be called "
                "after Ray initialization by OctoManager.run_outersplits()."
            )

        # Save fold data
        fold_dir = self.study.output_path / f"outersplit{outersplit_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_path = fold_dir / "data_train.parquet"
        data_train.to_parquet(str(train_path), storage_options=train_path.storage_options, engine="pyarrow")
        test_path = fold_dir / "data_test.parquet"
        data_test.to_parquet(str(test_path), storage_options=test_path.storage_options, engine="pyarrow")

        # task_results: dict[task_id -> (selected_features, prior_results_dict)]
        # prior_results_dict has keys: "scores", "predictions", "feature_importances" (DataFrames)
        task_results: dict[int, tuple[list[str], dict[str, pd.DataFrame]]] = {}

        for task in self.study.workflow:
            self._log_task_info(task)

            if task.load_task:
                # Load pre-existing task results
                result = self._load_task(outersplit_id, task)
                task_results[task.task_id] = result
            else:
                # Run and save task
                result = self._run_task(outersplit_id, data_train, data_test, task, task_results)
                task_results[task.task_id] = result

    def _run_task(
        self,
        outersplit_id: int,
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        task: Task,
        task_results: dict[int, tuple[list[str], dict[str, pd.DataFrame]]],
    ) -> tuple[list[str], dict[str, pd.DataFrame]]:
        """Run a single workflow task.

        Args:
            outersplit_id: Current fold ID
            data_train: Training DataFrame
            data_test: Test DataFrame
            task: Task to run
            task_results: Dictionary of results from previous tasks

        Returns:
            Tuple of (selected_features, results_dict) where results_dict
            has keys "scores", "predictions", "feature_importances" as DataFrames.

        Raises:
            ValueError: If task depends on a task that has not run yet
        """
        # Resolve upstream dependencies
        if task.depends_on is not None:
            if task.depends_on not in task_results:
                raise ValueError(f"Task {task.task_id} depends on task {task.depends_on} which has not run yet")
            feature_cols, prior_results = task_results[task.depends_on]
            logger.info(f"Prior results keys: {prior_results.keys()}")
        else:
            feature_cols = self.study.prepared.feature_cols
            prior_results = {}

        # Calculate feature groups
        feature_groups = calculate_feature_groups(data_train, feature_cols)

        # Create output directory
        output_dir = self.study.output_path / f"outersplit{outersplit_id}" / f"task{task.task_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running task {task.task_id} for fold {outersplit_id}")

        # Create execution module from config and run fit()
        module = task.create_module()
        selected_features, scores, predictions, feature_importances = module.fit(
            data_traindev=data_train,
            data_test=data_test,
            feature_cols=feature_cols,
            study=self.study,
            outersplit_id=outersplit_id,
            output_dir=output_dir,
            num_assigned_cpus=self.cpus_per_outersplit,
            feature_groups=feature_groups,
            prior_results=prior_results,
        )

        # Stamp module column on all result DataFrames
        module_name = task.module
        for df in [scores, predictions, feature_importances]:
            if isinstance(df, pd.DataFrame) and not df.empty:
                df["module"] = module_name

        # Save module state (model, config, fitted state)
        module.save(output_dir / "module")

        # Save task configuration and results
        self._save_task_config(task, output_dir)
        self._save_task_results(selected_features, scores, predictions, feature_importances, output_dir)

        results = {
            "scores": scores,
            "predictions": predictions,
            "feature_importances": feature_importances,
        }
        return (selected_features, results)

    def _load_task(
        self,
        outersplit_id: int,
        task: Task,
    ) -> tuple[list[str], dict[str, pd.DataFrame]]:
        """Load a pre-existing task from disk.

        Reads selected_features.json and loads DataFrames from parquet files.

        Args:
            outersplit_id: Current fold ID
            task: Task config for the task to load

        Returns:
            Tuple of (selected_features, results_dict)

        Raises:
            FileNotFoundError: If the task directory or required files don't exist
        """
        output_dir = self.study.output_path / f"outersplit{outersplit_id}" / f"task{task.task_id}"

        # Load selected features
        sf_path = output_dir / "selected_features.json"
        if not sf_path.exists():
            raise FileNotFoundError(
                f"Cannot load task {task.task_id}: selected_features.json not found at {output_dir}"
            )
        with sf_path.open() as f:
            selected_features = json.load(f)

        results = self._load_task_results(output_dir)

        logger.info(f"Loaded task {task.task_id}: {len(selected_features)} features")

        return (selected_features, results)

    def _load_task_results(self, output_dir: UPath) -> dict[str, pd.DataFrame]:
        """Load task results from saved parquet files.

        Args:
            output_dir: Task output directory

        Returns:
            Dict with keys "scores", "predictions", "feature_importances",
            each a DataFrame (or empty DataFrame if not found).
        """
        results: dict[str, pd.DataFrame] = {}

        for name in ["scores", "predictions", "feature_importances"]:
            path = output_dir / f"{name}.parquet"
            if path.exists():
                results[name] = pd.read_parquet(str(path), storage_options=path.storage_options, engine="pyarrow")
            else:
                results[name] = pd.DataFrame()

        return results

    def _save_task_config(self, task: Task, output_dir: UPath) -> None:
        """Save task configuration to JSON.

        Args:
            task: Task to save configuration for
            output_dir: Directory to save configuration in
        """
        config_path = output_dir / "task_config.json"

        # Convert task to dict for JSON serialization
        from attrs import asdict  # noqa: PLC0415

        # Exclude temporary state fields (start with _) and non-init fields to avoid circular refs
        config_dict = asdict(task, recurse=True, filter=lambda attr, value: attr.init and not attr.name.startswith("_"))

        with config_path.open("w") as f:
            json.dump(config_dict, f, indent=2)

    def _save_task_results(
        self,
        selected_features: list[str],
        scores: pd.DataFrame,
        predictions: pd.DataFrame,
        feature_importances: pd.DataFrame,
        output_dir: UPath,
    ) -> None:
        """Save task results to parquet files.

        Args:
            selected_features: List of selected features
            scores: Scores DataFrame
            predictions: Predictions DataFrame
            feature_importances: Feature importances DataFrame
            output_dir: Directory to save results in
        """
        # Save selected features to JSON
        with (output_dir / "selected_features.json").open("w") as f:
            json.dump(selected_features, f)

        # Save each non-empty DataFrame as parquet
        for name, df in [
            ("scores", scores),
            ("predictions", predictions),
            ("feature_importances", feature_importances),
        ]:
            if isinstance(df, pd.DataFrame) and not df.empty:
                path = output_dir / f"{name}.parquet"
                df.to_parquet(str(path), storage_options=path.storage_options, engine="pyarrow")

    def _log_task_info(self, task: Task) -> None:
        """Log information about a workflow task.

        Args:
            task: Task to log information about
        """
        logger.info(
            f"Processing workflow task: {task.task_id} | "
            f"Input task: {task.depends_on} | "
            f"Module: {task.module} | "
            f"Description: {task.description} | "
            f"Load existing: {task.load_task}"
        )
