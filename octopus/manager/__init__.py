"""OctoManager."""

from octopus.manager.core import OctoManager, ResourceConfig, get_available_cpus
from octopus.manager.execution import (
    ExecutionStrategy,
    ParallelRayStrategy,
    SequentialStrategy,
    SingleOutersplitStrategy,
)
from octopus.manager.ray_parallel import (
    init_ray,
    run_parallel_inner,
    run_parallel_outer_ray,
    setup_ray_for_external_library,
    shutdown_ray,
)
from octopus.manager.workflow_runner import WorkflowTaskRunner

__all__ = [
    "ExecutionStrategy",
    "OctoManager",
    "ParallelRayStrategy",
    "ResourceConfig",
    "SequentialStrategy",
    "SingleOutersplitStrategy",
    "WorkflowTaskRunner",
    "get_available_cpus",
    "init_ray",
    "run_parallel_inner",
    "run_parallel_outer_ray",
    "setup_ray_for_external_library",
    "shutdown_ray",
]
