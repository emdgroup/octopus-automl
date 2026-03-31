"""Execution strategies for running outersplits."""

from collections.abc import Callable
from typing import Protocol

from attrs import define, field, validators

from octopus.datasplit import OuterSplit, OuterSplits
from octopus.logger import get_logger
from octopus.types import LogGroup

from . import ParallelResources, ray_parallel

logger = get_logger()


@define
class ExecutionStrategy(Protocol):
    """Protocol for outersplit execution strategies."""

    resources: ParallelResources = field(validator=validators.instance_of(ParallelResources))
    """Resources for parallel execution, including CPU counts and Ray placement group."""

    def execute(
        self,
        outersplit_data: OuterSplits,
        run_fn: "Callable[[int, OuterSplit, ParallelResources], None]",
    ) -> None:
        """Execute outersplits using this strategy."""
        ...


@define
class SingleOutersplitStrategy(ExecutionStrategy):
    """Run a single outersplit by index."""

    outersplit_index: int = field(validator=[validators.instance_of(int), validators.ge(0)])
    """Index of the single outersplit to run."""

    def execute(
        self,
        outersplit_data: OuterSplits,
        run_fn: "Callable[[int, OuterSplit, ParallelResources], None]",
    ) -> None:
        """Execute only the outersplit at outersplit_index."""
        logger.set_log_group(LogGroup.PROCESSING)
        logger.info(f"Running single outersplit: {self.outersplit_index}")
        outersplit_id = self.outersplit_index
        run_fn(outersplit_id, outersplit_data[outersplit_id], self.resources)


@define
class SequentialStrategy(ExecutionStrategy):
    """Run outersplits one after another."""

    def execute(
        self,
        outersplit_data: OuterSplits,
        run_fn: "Callable[[int, OuterSplit, ParallelResources], None]",
    ) -> None:
        """Execute all outersplits sequentially."""
        logger.set_log_group(LogGroup.PROCESSING)
        for outersplit_id in outersplit_data:
            logger.info(f"Running outer split: {outersplit_id}")
            run_fn(outersplit_id, outersplit_data[outersplit_id], self.resources)


@define
class ParallelRayStrategy(ExecutionStrategy):
    """Run outersplits in parallel using Ray.

    This strategy starts as many parallel workers as allowed by the resource
    configuration set up in ray_parallel.init() and executes one outer split per worker.
    """

    def execute(
        self,
        outersplit_data: OuterSplits,
        run_fn: Callable[[int, OuterSplit, ParallelResources], None],
    ) -> None:
        """Execute all outer splits in parallel using Ray."""

        def wrapped_run(outersplit_id: int, outersplit: OuterSplit, resources: ParallelResources) -> None:
            logger.set_log_group(LogGroup.PROCESSING, f"OUTER {outersplit_id}")
            logger.info(f"Starting execution for outer split {outersplit_id}")
            try:
                run_fn(outersplit_id, outersplit, resources)
                logger.set_log_group(LogGroup.PREPARE_EXECUTION, f"OUTER {outersplit_id}")
                logger.info(f"Completed successfully for outer split {outersplit_id}")
            except Exception as e:
                logger.exception(f"Exception in task {outersplit_id}: {e!s}")
                raise e

        ray_parallel.run_parallel_outer(outersplit_data=outersplit_data, run_fn=wrapped_run, resources=self.resources)
