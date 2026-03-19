"""Execution strategies for running outersplits."""

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
    """Protocol for outersplit execution strategies."""

    def execute(
        self,
        outersplit_data: OuterSplits,
        run_fn: "Callable[[int, OuterSplit, int], None]",
    ) -> None:
        """Execute outersplits using this strategy."""
        ...


@define
class SingleOutersplitStrategy(ExecutionStrategy):
    """Run a single outersplit by index."""

    outersplit_index: int = field(validator=[validators.instance_of(int), validators.ge(0)])
    num_cpus: int = field(validator=[validators.instance_of(int), validators.ge(1)])
    """Number of CPUs to use for parallel processing within the single outersplit."""

    def execute(
        self,
        outersplit_data: OuterSplits,
        run_fn: "Callable[[int, OuterSplit, int], None]",
    ) -> None:
        """Execute only the outersplit at outersplit_index."""
        logger.set_log_group(LogGroup.PROCESSING)
        logger.info(f"Running single outersplit: {self.outersplit_index}")
        outersplit_id = self.outersplit_index
        run_fn(outersplit_id, outersplit_data[outersplit_id], self.num_cpus)


@define
class SequentialStrategy(ExecutionStrategy):
    """Run outersplits one after another."""

    num_cpus: int = field(validator=[validators.instance_of(int), validators.ge(1)])
    """Number of CPUs to use for parallel processing in each sequential step."""

    def execute(
        self,
        outersplit_data: OuterSplits,
        run_fn: "Callable[[int, OuterSplit, int], None]",
    ) -> None:
        """Execute all outersplits sequentially."""
        logger.set_log_group(LogGroup.PROCESSING)
        for outersplit_id in outersplit_data:
            logger.info(f"Running outer split: {outersplit_id}")
            run_fn(outersplit_id, outersplit_data[outersplit_id], self.num_cpus)


@define
class ParallelRayStrategy(ExecutionStrategy):
    """Run outersplits in parallel using Ray.

    This strategy starts as many parallel workers as allowed by the resource
    configuration set up in ray_parallel.init() and executes one outer split per worker.
    """

    num_cpus_per_worker: int = field(validator=[validators.instance_of(int), validators.ge(1)])
    """Number of CPUs to use for parallel processing within each parallel worker."""
    log_dir: UPath = field(validator=validators.instance_of(UPath))

    def execute(
        self,
        outersplit_data: OuterSplits,
        run_fn: "Callable[[int, OuterSplit, int], None]",
    ) -> None:
        """Execute all outer splits in parallel using Ray."""

        def wrapped_run(outersplit_id: int, outersplit: OuterSplit, num_cpus_per_worker: int) -> None:
            logger.set_log_group(LogGroup.PROCESSING, f"OUTER {outersplit_id}")
            logger.info(f"Starting execution for outer split {outersplit_id}")
            try:
                run_fn(outersplit_id, outersplit, num_cpus_per_worker)
                logger.set_log_group(LogGroup.PREPARE_EXECUTION, f"OUTER {outersplit_id}")
                logger.info(f"Completed successfully for outer split {outersplit_id}")
            except Exception as e:
                logger.exception(f"Exception in task {outersplit_id}: {e!s}")
                raise e

        ray_parallel.run_parallel_outer(
            outersplit_data=outersplit_data,
            run_fn=wrapped_run,
            log_dir=self.log_dir,
            num_cpus_per_worker=self.num_cpus_per_worker,
        )
