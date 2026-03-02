"""Execution strategies for running outersplits."""

from typing import TYPE_CHECKING, Protocol

from attrs import define
from upath import UPath

from octopus.datasplit import OuterSplit, OuterSplits
from octopus.logger import LogGroup, get_logger
from octopus.manager.ray_parallel import run_parallel_outer_ray

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger()


class ExecutionStrategy(Protocol):
    """Protocol for outersplit execution strategies."""

    def execute(
        self,
        outersplit_data: OuterSplits,
        run_fn: "Callable[[int, OuterSplit], None]",
    ) -> None:
        """Execute outersplits using this strategy."""
        ...


@define
class SingleOutersplitStrategy:
    """Run a single outersplit by index."""

    outersplit_index: int

    def execute(
        self,
        outersplit_data: OuterSplits,
        run_fn: "Callable[[int, OuterSplit], None]",
    ) -> None:
        """Execute only the outersplit at outersplit_index."""
        logger.set_log_group(LogGroup.PROCESSING)
        logger.info(f"Running single outersplit: {self.outersplit_index}")
        outersplit_id = self.outersplit_index
        run_fn(outersplit_id, outersplit_data[outersplit_id])


@define
class SequentialStrategy:
    """Run outersplits one after another."""

    def execute(
        self,
        outersplit_data: OuterSplits,
        run_fn: "Callable[[int, OuterSplit], None]",
    ) -> None:
        """Execute all outersplits sequentially."""
        logger.set_log_group(LogGroup.PROCESSING)
        for outersplit_id in outersplit_data:
            logger.info(f"Running outer split: {outersplit_id}")
            run_fn(outersplit_id, outersplit_data[outersplit_id])


@define
class ParallelRayStrategy:
    """Run outersplits in parallel using Ray."""

    num_workers: int
    log_dir: UPath

    def execute(
        self,
        outersplit_data: OuterSplits,
        run_fn: "Callable[[int, OuterSplit], None]",
    ) -> None:
        """Execute all outersplits in parallel using Ray."""

        def wrapped_run(outersplit_id: int, outersplit: OuterSplit) -> None:
            logger.set_log_group(LogGroup.PROCESSING, f"OUTER {outersplit_id}")
            logger.info("Starting execution")
            try:
                run_fn(outersplit_id, outersplit)
                logger.set_log_group(LogGroup.PREPARE_EXECUTION, f"OUTER {outersplit_id}")
                logger.info("Completed successfully")
            except Exception as e:
                logger.exception(f"Exception in task {outersplit_id}: {e!s}")

        run_parallel_outer_ray(
            outersplit_data=outersplit_data,
            run_fn=wrapped_run,
            log_dir=self.log_dir,
            num_workers=self.num_workers,
        )
