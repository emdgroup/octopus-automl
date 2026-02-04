"""Workflow task runner for processing experiments."""

import copy

import ray
from attrs import define, field, validators
from upath import UPath

from octopus.experiment import OctoExperiment
from octopus.logger import get_logger
from octopus.modules import modules_inventory
from octopus.task import Task

logger = get_logger()


@define
class WorkflowTaskRunner:
    """Runs workflow tasks for a single base experiment.

    Handles the lifecycle of processing workflow tasks:
    - Creating new experiments from templates
    - Loading existing experiments
    - Running ML modules and saving results

    Attributes:
        workflow: List of workflow tasks to process.
        cpus_per_experiment: Number of CPUs allocated to each experiment for inner parallelization.
        log_dir: Directory for individual worker logs.
    """

    workflow: list[Task] = field(validator=[validators.instance_of(list)])
    cpus_per_experiment: int = field(validator=[validators.instance_of(int)])
    log_dir: UPath = field(validator=[validators.instance_of(UPath)])

    def run(self, base_experiment: OctoExperiment) -> None:
        """Process all workflow tasks for a base experiment.

        Args:
            base_experiment: The base experiment to process.

        Raises:
            RuntimeError: If Ray is not initialized. Ray must be initialized by
                OctoManager.run_outer_experiments() before calling this method.
        """
        if not ray.is_initialized():
            raise RuntimeError(
                "Ray is not initialized. WorkflowTaskRunner.run() must be called "
                "after Ray initialization by OctoManager.run_outer_experiments()."
            )

        exp_path_dict: dict[int, UPath] = {}

        for task in self.workflow:
            self._log_task_info(task)

            if task.load_task:
                self._load_experiment(base_experiment, task)
            else:
                self._run_task(base_experiment, task, exp_path_dict)

    def _run_task(
        self,
        base_experiment: OctoExperiment,
        task: Task,
        exp_path_dict: dict[int, UPath],
    ) -> None:
        """Run a single workflow task."""
        experiment = self._create_experiment(base_experiment, task)
        workflow_dir = self._ensure_workflow_dir(experiment)
        save_path = workflow_dir / f"exp{experiment.experiment_id}_{experiment.task_id}.pkl"
        assert experiment.task_id is not None  # Set in _create_experiment
        exp_path_dict[experiment.task_id] = save_path

        self._apply_dependencies(experiment, exp_path_dict)
        self._execute_and_save(experiment, workflow_dir, save_path)

    def _create_experiment(self, base_experiment: OctoExperiment, task: Task) -> OctoExperiment:
        """Create a new experiment from base experiment and task."""
        experiment = copy.deepcopy(base_experiment)
        experiment.ml_module = task.module  # type: ignore[attr-defined]  # ClassVar in Task subclasses
        experiment.ml_config = task
        experiment.id = f"{experiment.id}_{task.task_id}"
        experiment.task_id = task.task_id
        experiment.depends_on_task = task.depends_on_task
        experiment._task_path = UPath(
            f"outersplit{experiment.experiment_id}",
            f"workflowtask{task.task_id}",
        )
        experiment.num_assigned_cpus = self.cpus_per_experiment
        return experiment

    def _ensure_workflow_dir(self, experiment: OctoExperiment) -> UPath:
        """Create and return the workflow directory for an experiment."""
        workflow_dir = experiment.path_study / experiment.task_path
        workflow_dir.mkdir(parents=True, exist_ok=True)
        return workflow_dir

    def _apply_dependencies(self, experiment: OctoExperiment, exp_path_dict: dict[int, UPath]) -> None:
        """Apply dependencies from previous workflow tasks."""
        if experiment.depends_on_task is not None and experiment.depends_on_task >= 0:
            input_path = exp_path_dict[experiment.depends_on_task]
            if not input_path.exists():
                raise FileNotFoundError("Workflow task to be loaded does not exist")

            input_experiment = OctoExperiment.from_pickle(input_path)
            experiment.feature_cols = input_experiment.selected_features
            experiment.prior_results = input_experiment.results
            logger.info(f"Prior results keys: {experiment.prior_results.keys()}")

        experiment.feature_groups = experiment.calculate_feature_groups(experiment.feature_cols)

    def _execute_and_save(
        self,
        experiment: OctoExperiment,
        workflow_dir: UPath,
        save_path: UPath,
    ) -> None:
        """Execute the ML module and save results."""
        logger.info(f"Running experiment: {experiment.id}")
        experiment.to_pickle(save_path)

        module = self._get_module(experiment)
        experiment = module.run_experiment()

        self._save_results(experiment, workflow_dir)
        experiment.to_pickle(save_path)

    def _get_module(self, experiment: OctoExperiment):
        """Get the ML module for an experiment."""
        if experiment.ml_module not in modules_inventory:
            raise ValueError(f"ml_module {experiment.ml_module} not supported")
        return modules_inventory[experiment.ml_module](experiment=experiment, log_dir=self.log_dir)

    def _save_results(self, experiment: OctoExperiment, workflow_dir: UPath) -> None:
        """Save experiment results (predictions and feature importance)."""
        if not experiment.results:
            return

        for key in experiment.results:
            result = experiment.results[key]

            # Save predictions
            predictions_path = (
                workflow_dir / f"predictions_{experiment.experiment_id}_{experiment.task_id}_{key}.parquet"
            )
            result.create_prediction_df().to_parquet(
                str(predictions_path),
                storage_options=predictions_path.storage_options,
                engine="pyarrow",
            )

            # Save feature importance
            fi_path = workflow_dir / f"feature-importance_{experiment.experiment_id}_{experiment.task_id}_{key}.parquet"
            result.create_feature_importance_df().to_parquet(
                str(fi_path),
                storage_options=fi_path.storage_options,
                engine="pyarrow",
            )

    def _load_experiment(self, base_experiment: OctoExperiment, task: Task) -> None:
        """Validate that an existing experiment exists on disk.

        Args:
            base_experiment: The base experiment to determine the path.
            task: The workflow task to load.

        Raises:
            FileNotFoundError: If the experiment file does not exist.
        """
        workflow_dir = (
            base_experiment.path_study / f"outersplit{base_experiment.experiment_id}" / f"workflowtask{task.task_id}"
        )
        load_path = workflow_dir / f"exp{base_experiment.experiment_id}_{task.task_id}.pkl"

        if not load_path.exists():
            raise FileNotFoundError("Workflow task to be loaded does not exist")

        OctoExperiment.from_pickle(load_path)
        logger.info(f"Validated existing experiment at: {load_path}")

    def _log_task_info(self, task: Task) -> None:
        """Log information about a workflow task."""
        logger.info(
            f"Processing workflow task: {task.task_id} | "
            f"Input item: {task.depends_on_task} | "
            f"Module: {task.module} | "  # type: ignore[attr-defined]
            f"Description: {task.description} | "
            f"Load existing workflow task: {task.load_task}"
        )
