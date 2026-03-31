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
    SingleOutersplitStrategy,
)
from octopus.manager.workflow_runner import WorkflowTaskRunner
from octopus.modules import StudyContext, Task

from . import ParallelResources

logger = get_logger()


@define
class OctoManager:
    """Orchestrates the execution of outersplits."""

    outersplit_data: OuterSplits = field(validator=[validators.instance_of(dict)])
    """Preprocessed data for each outersplit, keyed by outersplit identifier."""

    study_context: StudyContext = field(validator=[validators.instance_of(StudyContext)])
    """Frozen runtime context containing study configuration."""

    workflow: Sequence[Task] = field(validator=[validators.instance_of(list)])
    """Workflow tasks to execute."""

    num_cpus: int = field(validator=validators.instance_of(int))
    """Number of CPUs to use for parallel processing. num_cpus=0 uses all available CPUs.
       Negative values indicate abs(num_cpus) to leave free, e.g. -1 means use all but one CPU.
       Set to 1 to disable all parallel processing and run sequentially."""

    run_single_outersplit_num: int | None = field(
        validator=validators.optional([validators.instance_of(int), validators.ge(0)])
    )
    """Index of single outersplit to run (None for all)."""

    def run_outersplits(self) -> None:
        """Run all outersplits."""
        if not self.outersplit_data:
            raise ValueError("No outersplit data defined")

        if self.run_single_outersplit_num is not None and not (
            0 <= self.run_single_outersplit_num < len(self.outersplit_data)
        ):
            raise ValueError(
                f"run_single_outersplit_num must be between 0 and num_outersplits-1 ({len(self.outersplit_data) - 1}), got {self.run_single_outersplit_num}"
            )

        resources = ray_parallel.init(
            num_cpus_user=self.num_cpus,
            num_outersplits=len(self.outersplit_data),
            run_single_outersplit=self.run_single_outersplit_num is not None,
            log_dir=self.study_context.log_dir,
            namespace=f"octopus_study_{self.study_context.output_path}",
        )

        logger.info(f"Preparing execution | {resources}")

        try:
            runner = WorkflowTaskRunner(
                study_context=self.study_context,
                workflow=self.workflow,
            )
            strategy = self._select_strategy(resources)
            strategy.execute(self.outersplit_data, runner.run)
        finally:
            ray_parallel.shutdown()

    def _select_strategy(self, resources: ParallelResources) -> ExecutionStrategy:
        """Select execution strategy based on configuration.

        Args:
            resources: Resource configuration for execution.

        Returns:
            Appropriate execution strategy based on configuration.
        """
        if self.run_single_outersplit_num is not None:
            return SingleOutersplitStrategy(
                outersplit_index=self.run_single_outersplit_num,
                resources=resources,
            )
        elif len(resources.placement_group.bundle_specs) > 1:
            return ParallelRayStrategy(
                resources=resources,
            )
        else:
            return SequentialStrategy(
                resources=resources,
            )
