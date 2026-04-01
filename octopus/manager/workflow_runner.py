"""Workflow task runner for processing tasks."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
from attrs import asdict, define, field, validators
from upath import UPath

from octopus.datasplit import OuterSplit
from octopus.logger import get_logger
from octopus.modules import ModuleResult, StudyContext, Task
from octopus.types import ResultType
from octopus.utils import calculate_feature_groups, rmtree

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger()


@define
class WorkflowTaskRunner:
    """Runs workflow tasks for a single outer split.

    Handles the lifecycle of processing workflow tasks:
    - Saving split data
    - Running tasks with dependencies
    - Saving task results

    Attributes:
        study_context: Frozen runtime context containing study configuration.
        workflow: List of workflow tasks to execute.
    """

    study_context: StudyContext = field(validator=[validators.instance_of(StudyContext)])
    workflow: Sequence[Task] = field(validator=[validators.instance_of(list)])

    def run(self, outer_split_id: int, outer_split: OuterSplit, n_assigned_cpus: int) -> None:
        """Process all workflow tasks for a single outer split.

        Args:
            outer_split_id: Current outer split ID
            outer_split: OuterSplit containing traindev and test DataFrames
            n_assigned_cpus: Number of CPUs assigned to this outer split for inner parallel processing
        """
        # Save split row IDs (not full datasets) for reproducibility
        outer_split_dir = self.study_context.output_path / f"outersplit{outer_split_id}"
        outer_split_dir.mkdir(parents=True, exist_ok=True)
        row_id_col = self.study_context.row_id_col
        split_ids = {
            "row_id_col": row_id_col,
            "traindev_row_ids": outer_split.traindev[row_id_col].tolist(),
            "test_row_ids": outer_split.test[row_id_col].tolist(),
        }
        with (outer_split_dir / "split_row_ids.json").open("w") as f:
            json.dump(split_ids, f)

        # task_results: dict[task_id -> dict[ResultType, ModuleResult]]
        task_results: dict[int, dict[ResultType, ModuleResult]] = {}

        for task in self.workflow:
            self._log_task_info(task)
            result = self._run_task(outer_split_id, outer_split, task, n_assigned_cpus, task_results, outer_split_dir)
            task_results[task.task_id] = result

    def _run_task(
        self,
        outer_split_id: int,
        outer_split: OuterSplit,
        task: Task,
        n_assigned_cpus: int,
        task_results: dict[int, dict[ResultType, ModuleResult]],
        outer_split_dir: UPath,
    ) -> dict[ResultType, ModuleResult]:
        """Run a single workflow task.

        Args:
            outer_split_id: Current outer split ID
            outer_split: OuterSplit containing traindev and test DataFrames
            task: Task to run
            n_assigned_cpus: Number of CPUs assigned to this outer split for inner parallel processing
            task_results: Dictionary of results from previous tasks
            outer_split_dir: directory where all data/results relevant for this outer
              split reside / should be saved to

        Returns:
            Dict mapping ResultType to ModuleResult.

        Raises:
            ValueError: If task depends on a task that has not run yet
        """
        # Resolve upstream dependencies
        if task.depends_on is not None:
            if task.depends_on not in task_results:
                raise ValueError(f"Task {task.task_id} depends on task {task.depends_on} which has not run yet")
            upstream_results = task_results[task.depends_on]
            feature_cols = upstream_results[ResultType.BEST].selected_features
            # Build prior_results by concatenating DataFrames from all upstream ModuleResult values
            prior_results: dict[str, pd.DataFrame] = {}
            for attr_name, key in [("scores", "scores"), ("predictions", "predictions"), ("fi", "fi")]:
                dfs = []
                for module_result in upstream_results.values():
                    df = getattr(module_result, attr_name)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        out = df.copy()
                        out["module"] = module_result.module
                        dfs.append(out)
                prior_results[key] = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            logger.info(f"Prior results keys: {prior_results.keys()}")
        else:
            feature_cols = self.study_context.feature_cols
            prior_results = {}

        # Calculate feature groups
        feature_groups = calculate_feature_groups(outer_split.traindev, feature_cols)

        # Create output directory
        module_output_dir = outer_split_dir / f"task{task.task_id}"
        module_output_dir.mkdir(parents=True, exist_ok=True)
        module_results_dir = module_output_dir / "results"
        module_results_dir.mkdir(parents=True, exist_ok=True)
        module_scratch_dir = module_output_dir / "scratch"
        module_scratch_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running task {task.task_id} for outer split {outer_split_id}")

        # Create execution module from config and run fit()
        module = task.create_module()
        results = module.fit(
            data_traindev=outer_split.traindev,
            data_test=outer_split.test,
            feature_cols=feature_cols,
            study_context=self.study_context,
            outer_split_id=outer_split_id,
            results_dir=module_results_dir,
            scratch_dir=module_scratch_dir,
            n_assigned_cpus=n_assigned_cpus,
            feature_groups=feature_groups,
            prior_results=prior_results,
        )

        self._save_task_context(module_output_dir, feature_cols, feature_groups)
        self._save_task_config(task, module_output_dir)
        for result_type, module_result in results.items():
            module_result.save(module_results_dir / result_type.value)

        # Clean up scratch directory
        rmtree(module_scratch_dir)

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
