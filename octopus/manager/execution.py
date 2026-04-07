"""Execution strategies for running outer splits."""

from typing import TYPE_CHECKING, Protocol

from attrs import define, field, validators
from upath import UPath

from octopus.datasplit import OuterSplit, OuterSplits
from octopus.logger import get_logger
from octopus.manager import ray_parallel
from octopus.types import LogGroup

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger()


class ExecutionStrategy(Protocol):
    """Protocol for outer split execution strategies."""

    def execute(
        self,
        outer_split_data: OuterSplits,
        run_fn: "Callable[[int, OuterSplit, int], None]",
    ) -> None:
        """Execute outer splits using this strategy."""
        ...


@define
class SingleOuterSplitStrategy(ExecutionStrategy):
    """Run a single outer split by index."""

    outer_split_index: int = field(validator=[validators.instance_of(int), validators.ge(0)])
    num_cpus: int = field(validator=[validators.instance_of(int), validators.ge(1)])
    """Number of CPUs to use for parallel processing within the single outer split."""

    def execute(
        self,
        outer_split_data: OuterSplits,
        run_fn: "Callable[[int, OuterSplit, int], None]",
    ) -> None:
        """Execute only the outer split at outer_split_index."""
        logger.set_log_group(LogGroup.PROCESSING)
        logger.info(f"Running single outer split: {self.outer_split_index}")
        outer_split_id = self.outer_split_index
        run_fn(outer_split_id, outer_split_data[outer_split_id], self.num_cpus)


@define
class SequentialStrategy(ExecutionStrategy):
    """Run outer splits one after another."""

    num_cpus: int = field(validator=[validators.instance_of(int), validators.ge(1)])
    """Number of CPUs to use for parallel processing in each sequential step."""

    def execute(
        self,
        outer_split_data: OuterSplits,
        run_fn: "Callable[[int, OuterSplit, int], None]",
    ) -> None:
        """Execute all outer splits sequentially."""
        logger.set_log_group(LogGroup.PROCESSING)
        for outer_split_id in outer_split_data:
            logger.info(f"Running outer split: {outer_split_id}")
            run_fn(outer_split_id, outer_split_data[outer_split_id], self.num_cpus)


@define
class ParallelRayStrategy(ExecutionStrategy):
    """Run outer splits in parallel using Ray.

    This strategy starts as many parallel workers as allowed by the resource
    configuration set up in ray_parallel.init() and executes one outer split per worker.
    """

    num_cpus_per_worker: int = field(validator=[validators.instance_of(int), validators.ge(1)])
    """Number of CPUs to use for parallel processing within each parallel worker."""
    log_dir: UPath = field(validator=validators.instance_of(UPath))

    def execute(
        self,
        outer_split_data: OuterSplits,
        run_fn: "Callable[[int, OuterSplit, int], None]",
    ) -> None:
        """Execute all outer splits in parallel using Ray."""

        def wrapped_run(outer_split_id: int, outer_split: OuterSplit, num_cpus_per_worker: int) -> None:
            logger.set_log_group(LogGroup.PROCESSING, f"OUTER {outer_split_id}")
            logger.info(f"Starting execution for outer split {outer_split_id}")
            try:
                run_fn(outer_split_id, outer_split, num_cpus_per_worker)
                logger.set_log_group(LogGroup.PREPARE_EXECUTION, f"OUTER {outer_split_id}")
                logger.info(f"Completed successfully for outer split {outer_split_id}")
            except Exception as e:
                logger.exception(f"Exception in task {outer_split_id}: {e!s}")
                raise e

        ray_parallel.run_parallel_outer(
            outer_split_data=outer_split_data,
            run_fn=wrapped_run,
            log_dir=self.log_dir,
            num_cpus_per_worker=self.num_cpus_per_worker,
        )
