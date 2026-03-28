"""Workflow task runner for processing tasks."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import ray
from attrs import asdict, define, field, validators
from upath import UPath

from octopus.datasplit import OuterSplit
from octopus.logger import get_logger
from octopus.modules import ModuleResult, StudyContext, Task
from octopus.types import ResultType
from octopus.utils import calculate_feature_groups, parquet_save, rmtree

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger()


@define
class WorkflowTaskRunner:
    """Runs workflow tasks for a single fold.

    Handles the lifecycle of processing workflow tasks:
    - Saving fold data
    - Running tasks with dependencies
    - Saving task results

    Attributes:
        study_context: Frozen runtime context containing study configuration.
        workflow: List of workflow tasks to execute.
        cpus_per_outersplit: Number of CPUs allocated to each task.
    """

    study_context: StudyContext = field(validator=[validators.instance_of(StudyContext)])
    workflow: Sequence[Task] = field(validator=[validators.instance_of(list)])
    cpus_per_outersplit: int = field(validator=[validators.instance_of(int)])

    def run(self, outersplit_id: int, outersplit: OuterSplit) -> None:
        """Process all workflow tasks for a single fold.

        Args:
            outersplit_id: Current fold ID
            outersplit: OuterSplit containing traindev and test DataFrames

        Raises:
            RuntimeError: If Ray is not initialized.
        """
        if not ray.is_initialized():
            raise RuntimeError(
                "Ray is not initialized. WorkflowTaskRunner.run() must be called after Ray initialization by OctoManager.run_outersplits()."
            )

        # Save fold data
        fold_dir = self.study_context.output_path / f"outersplit{outersplit_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_path = fold_dir / "data_traindev.parquet"
        parquet_save(outersplit.traindev, train_path)
        test_path = fold_dir / "data_test.parquet"
        parquet_save(outersplit.test, test_path)

        # task_results: dict[task_id -> dict[ResultType, ModuleResult]]
        task_results: dict[int, dict[ResultType, ModuleResult]] = {}

        for task in self.workflow:
            self._log_task_info(task)

            result = self._run_task(outersplit_id, outersplit, task, task_results)
            task_results[task.task_id] = result

    def _run_task(
        self,
        outersplit_id: int,
        outersplit: OuterSplit,
        task: Task,
        task_results: dict[int, dict[ResultType, ModuleResult]],
    ) -> dict[ResultType, ModuleResult]:
        """Run a single workflow task.

        Args:
            outersplit_id: Current fold ID
            outersplit: OuterSplit containing traindev and test DataFrames
            task: Task to run
            task_results: Dictionary of results from previous tasks

        Returns:
            Dict mapping ResultType to ModuleResult.

        Raises:
            ValueError: If task depends on a task that has not run yet
        """
        # Resolve upstream dependencies
        if task.depends_on is not None:
            if task.depends_on not in task_results:
                raise ValueError(f"Task {task.task_id} depends on task {task.depends_on} which has not run yet")
            dependency_results = task_results[task.depends_on]
            feature_cols = dependency_results[ResultType.BEST].selected_features
        else:
            feature_cols = self.study_context.feature_cols
            dependency_results = {}

        # Calculate feature groups
        feature_groups = calculate_feature_groups(outersplit.traindev, feature_cols)

        # Create output directory
        output_dir = self.study_context.output_path / f"outersplit{outersplit_id}" / f"task{task.task_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        results_dir = output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        scratch_dir = output_dir / "scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running task {task.task_id} for fold {outersplit_id}")

        # Create execution module from config and run fit()
        module = task.create_module()
        results = module.fit(
            data_traindev=outersplit.traindev,
            data_test=outersplit.test,
            feature_cols=feature_cols,
            study_context=self.study_context,
            outersplit_id=outersplit_id,
            results_dir=results_dir,
            scratch_dir=scratch_dir,
            num_assigned_cpus=self.cpus_per_outersplit,
            feature_groups=feature_groups,
            dependency_results=dependency_results,
        )

        self._save_task_context(output_dir, feature_cols, feature_groups)
        self._save_task_config(task, output_dir)
        for result_type, module_result in results.items():
            module_result.save(results_dir / result_type.value)

        # Clean up scratch directory
        rmtree(scratch_dir)

        return results

    def _save_task_config(self, task: Task, output_dir: UPath) -> None:
        """Save task configuration to JSON.

        Args:
            task: Task to save configuration for
            output_dir: Directory to save configuration in
        """
        config_path = output_dir / "config" / "task_config.json"

        config_dict = asdict(task)

        with config_path.open("w") as f:
            json.dump(config_dict, f, indent=2)

    def _save_task_context(
        self,
        output_dir: UPath,
        feature_cols: list[str],
        feature_groups: dict | None,
    ) -> None:
        """Save task runtime context to disk.

        Saves the input feature columns and correlation-based feature groups
        that were used when running this task. These are needed by
        ``TaskPredictor`` for prediction and feature importance computation.

        Files are written to a ``config/`` subdirectory to match the path
        expected by ``OuterSplitLoader``.

        Args:
            output_dir: Task output directory (e.g. outersplit0/task0/).
            feature_cols: Input feature columns used by this task.
            feature_groups: Correlation-based feature groups, or None.
        """
        config_dir = output_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        with (config_dir / "feature_cols.json").open("w") as f:
            json.dump(feature_cols, f, indent=2)

        if feature_groups:
            with (config_dir / "feature_groups.json").open("w") as f:
                json.dump(feature_groups, f, indent=2)

    def _log_task_info(self, task: Task) -> None:
        """Log information about a workflow task.

        Args:
            task: Task to log information about
        """
        logger.info(
            f"Processing workflow task: {task.task_id} | Input task: {task.depends_on} | Module: {task.module} | Description: {task.description}"
        )
