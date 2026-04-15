"""OctoManager."""

from collections.abc import Sequence

from attrs import define, field, validators

from octopus.datasplit import OuterSplits
from octopus.logger import get_logger
from octopus.manager import ray_parallel
from octopus.manager.execution import (
    ExecutionStrategy,
    ParallelRayStrategy,
    SequentialStrategy,
    SingleOuterSplitStrategy,
)
from octopus.manager.workflow_runner import WorkflowTaskRunner
from octopus.modules import StudyContext, Task

logger = get_logger()


@define
class OctoManager:
    """Orchestrates the execution of outer splits."""

    outer_split_data: OuterSplits = field(validator=[validators.instance_of(dict)])
    """Preprocessed data for each outer split, keyed by outer split identifier."""

    study_context: StudyContext = field(validator=[validators.instance_of(StudyContext)])
    """Frozen runtime context containing study configuration."""

    workflow: Sequence[Task] = field(validator=[validators.instance_of(list)])
    """Workflow tasks to execute."""

    n_cpus: int = field(validator=validators.instance_of(int))
    """Number of CPUs to use for parallel processing. n_cpus=0 uses all available CPUs.
       Negative values indicate abs(n_cpus) to leave free, e.g. -1 means use all but one CPU.
       Set to 1 to disable all parallel processing and run sequentially."""

    single_outer_split: int | None = field(
        validator=validators.optional(validators.and_(validators.instance_of(int), validators.ge(0)))
    )
    """Index of single outer split to run (None for all)."""

    def run_outer_splits(self) -> None:
        """Run all outer splits."""
        if not self.outer_split_data:
            raise ValueError("No outer split data defined")

        if self.single_outer_split is not None and not (0 <= self.single_outer_split < len(self.outer_split_data)):
            raise ValueError(
                f"single_outer_split must be between 0 and n_outer_splits-1"
                f" ({len(self.outer_split_data) - 1}), got {self.single_outer_split}"
            )

        # Initialize Ray upfront to ensure worker setup hooks are registered before any workflows execute.
        # This is critical for:
        # 1. Inner parallelization: ML modules (e.g., Tako, AutoGluon) may spawn Ray workers for their
        #    internal operations (bagging, hyperparameter tuning)
        # 2. Lifecycle clarity: Explicit init → run → shutdown at the manager level makes the
        #    Ray lifecycle predictable and easier to reason about
        resources = ray_parallel.init(
            n_cpus_user=self.n_cpus,
            n_outer_splits=len(self.outer_split_data),
            run_single_outer_split=self.single_outer_split is not None,
            namespace=f"octopus_study_{self.study_context.output_path}",
        )

        logger.info(f"Preparing execution | {resources}")

        try:
            runner = WorkflowTaskRunner(
                study_context=self.study_context,
                workflow=self.workflow,
            )
            strategy = self._select_strategy(resources)
            strategy.execute(self.outer_split_data, runner.run)
        finally:
            ray_parallel.shutdown()

    def _select_strategy(self, resources: ray_parallel.ResourceConfig) -> ExecutionStrategy:
        """Select execution strategy based on configuration.

        Args:
            resources: Resource configuration for execution.

        Returns:
            Appropriate execution strategy based on configuration.
        """
        if self.single_outer_split is not None:
            return SingleOuterSplitStrategy(
                outer_split_index=self.single_outer_split,
                n_cpus=resources.cpus_per_worker,
            )
        elif resources.n_workers > 1:
            return ParallelRayStrategy(
                n_cpus_per_worker=resources.cpus_per_worker,
                log_dir=self.study_context.log_dir,
            )
        else:
            return SequentialStrategy(
                n_cpus=resources.cpus_per_worker,
            )
