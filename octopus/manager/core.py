"""OctoManager."""

import math
import os
from typing import TYPE_CHECKING

from attrs import define, field, validators

from octopus.logger import get_logger
from octopus.manager.execution import (
    ExecutionStrategy,
    ParallelRayStrategy,
    SequentialStrategy,
    SingleOutersplitStrategy,
)
from octopus.manager.ray_parallel import init_ray, shutdown_ray
from octopus.manager.workflow_runner import WorkflowTaskRunner

if TYPE_CHECKING:
    from octopus.study.core import OctoStudy

logger = get_logger()


def get_available_cpus() -> int:
    """Get available CPUs on the system."""
    total_cpus = os.cpu_count()
    if total_cpus is None:
        raise RuntimeError("Could not determine number of CPUs.")
    return total_cpus


@define(frozen=True)
class ResourceConfig:
    """Immutable configuration for CPU resources.

    Attributes:
        num_cpus: Total available CPUs on the system.
        num_workers: Number of parallel outer workers.
        cpus_per_outersplit: CPUs allocated to each outersplit for inner parallelization.
        outer_parallelization: Whether outer parallelization is enabled.
        run_single_outersplit_num: Index of single outersplit to run (-1 for all).
        num_outersplits: Total number of outersplits in the study.
    """

    num_cpus: int
    num_workers: int
    cpus_per_outersplit: int
    outer_parallelization: bool
    run_single_outersplit_num: int
    num_outersplits: int

    @classmethod
    def create(
        cls,
        num_outersplits: int,
        outer_parallelization: bool,
        run_single_outersplit_num: int,
        num_cpus: int | None = None,
    ) -> "ResourceConfig":
        """Create ResourceConfig with computed values.

        Args:
            num_outersplits: Total number of outersplits in the study.
            outer_parallelization: Whether to run outersplits in parallel.
            run_single_outersplit_num: Index of single outersplit to run (-1 for all).
            num_cpus: Total CPUs available (auto-detected if None).

        Returns:
            ResourceConfig with computed worker and CPU allocation.

        Raises:
            ValueError: If any input parameter is invalid.
        """
        if num_outersplits <= 0:
            raise ValueError(f"num_outersplits must be positive, got {num_outersplits}")

        if run_single_outersplit_num < -1:
            raise ValueError(
                f"run_single_outersplit_num must be -1 (all outersplits) or a valid index >= 0, "
                f"got {run_single_outersplit_num}"
            )
        if run_single_outersplit_num >= num_outersplits:
            raise ValueError(
                f"run_single_outersplit_num ({run_single_outersplit_num}) must be less than "
                f"num_outersplits ({num_outersplits})"
            )

        # Get or validate num_cpus
        if num_cpus is None:
            num_cpus = get_available_cpus()
        elif num_cpus <= 0:
            raise ValueError(f"num_cpus must be positive, got {num_cpus}")

        # Calculate effective number of outersplits for resource allocation
        effective_num_outersplits = 1 if run_single_outersplit_num != -1 else num_outersplits

        # Calculate resource allocation
        num_workers = min(effective_num_outersplits, num_cpus)
        if num_workers == 0:
            raise ValueError(
                f"Cannot allocate resources: num_workers computed as 0 "
                f"(effective_num_outersplits={effective_num_outersplits}, num_cpus={num_cpus})"
            )

        cpus_per_outersplit = max(1, math.floor(num_cpus / num_workers)) if outer_parallelization else num_cpus

        return cls(
            num_cpus=num_cpus,
            num_workers=num_workers,
            cpus_per_outersplit=cpus_per_outersplit,
            outer_parallelization=outer_parallelization,
            run_single_outersplit_num=run_single_outersplit_num,
            num_outersplits=num_outersplits,
        )

    def __str__(self) -> str:
        """Return string representation of resource configuration."""
        return (
            f"Parallelization: {self.outer_parallelization} | "
            f"Single outersplit: {self.run_single_outersplit_num} | "
            f"Outersplits: {self.num_outersplits} | "
            f"CPUs: {self.num_cpus} | "
            f"Workers: {self.num_workers} | "
            f"CPUs/outersplit: {self.cpus_per_outersplit}"
        )


@define
class OctoManager:
    """Orchestrates the execution of outersplits."""

    study: "OctoStudy" = field(validator=[validators.instance_of(object)])  # type: ignore[assignment]
    outersplit_data: dict = field(validator=[validators.instance_of(dict)])

    def run_outersplits(self) -> None:
        """Run all outersplits."""
        if not self.outersplit_data:
            raise ValueError("No outersplit data defined")

        # Initialize Ray upfront to ensure worker setup hooks are registered before any workflows execute.
        # This is critical for:
        # 1. Inner parallelization: ML modules (e.g., Octo, AutoGluon) may spawn Ray workers for their
        #    internal operations (bagging, hyperparameter tuning) even when outer_parallelization=False
        # 2. Safety checks: The worker setup hook (_check_parallelization_disabled) must be configured
        #    before any Ray workers start, to detect and prevent thread-level parallelization issues
        # 3. Lifecycle clarity: Explicit init → run → shutdown at the manager level makes the
        #    Ray lifecycle predictable and easier to reason about
        init_ray(start_local_if_missing=True)

        resources = ResourceConfig.create(
            num_outersplits=len(self.outersplit_data),
            outer_parallelization=self.study.outer_parallelization,
            run_single_outersplit_num=self.study.run_single_outersplit_num,
        )
        logger.info(f"Preparing execution | {resources}")

        try:
            runner = WorkflowTaskRunner(self.study, resources.cpus_per_outersplit)
            strategy = self._select_strategy(resources.num_workers)
            strategy.execute(self.outersplit_data, runner.run)
        finally:
            shutdown_ray()

    def _select_strategy(self, num_workers: int) -> ExecutionStrategy:
        """Select execution strategy based on configuration.

        Args:
            num_workers: Number of parallel workers for ParallelRayStrategy.

        Returns:
            Appropriate execution strategy based on configuration.
        """
        if self.study.run_single_outersplit_num != -1:
            return SingleOutersplitStrategy(self.study.run_single_outersplit_num)
        if self.study.outer_parallelization:
            return ParallelRayStrategy(num_workers, self.study.log_dir)
        return SequentialStrategy()
