"""Ray parallelization for outer and inner loops."""

import os
from collections import deque
from collections.abc import Callable, Sequence
from functools import partial
from typing import Protocol

import ray
import threadpoolctl
from ray.runtime_env import RuntimeEnv
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from upath import UPath

from octopus.datasplit import OuterSplit, OuterSplits
from octopus.logger import get_logger, set_logger_filename
from octopus.modules.octo.training import Training

from . import ParallelResources

logger = get_logger()


def _get_locally_available_cpus() -> int:
    """Get available CPUs on the system."""
    if (total_cpus := os.cpu_count()) is not None:
        return total_cpus
    else:
        raise RuntimeError("Could not determine number of CPUs.")


class SupportsFit(Protocol):
    """Protocol for objects that have a fit() method."""

    def fit(self) -> Training:
        """Generic fit method that returns a Training object."""
        ...


def init(
    num_cpus_user: int,
    num_outersplits: int,
    run_single_outersplit: bool,
    log_dir: UPath,
    address: str | None = None,
    namespace: str | None = None,
) -> ParallelResources:
    """Initialize Ray for the current process.

    Connects to an existing cluster if an address is provided or set via
    environment variables; otherwise, optionally starts a local Ray instance.

    If a local Ray instance is started, configure environment to enable
    external libraries to use the existing Ray instance:
    Sets RAY_ADDRESS to the current Ray GCS address, preventing external libraries
    (e.g., AutoGluon, Ray Tune) from creating separate Ray instances that would
    cause resource conflicts.

    Args:
        num_cpus_user: Number of CPUs requested by the user. for parallel processing. num_cpus=0 uses all available CPUs.
          Negative values indicate abs(num_cpus) to leave free, e.g. -1 means use all but one CPU.
          Set to 1 to disable all parallel processing and run sequentially.
        num_outersplits: Total number of outersplits in the study.
        run_single_outersplit: Whether to run a single outer split instead of all. This is mainly used for testing and debugging.
        log_dir: Directory for Ray worker logs.
        address: Ray head address (e.g., "auto", "127.0.0.1:6379", "local"). If None, uses
            env vars RAY_ADDRESS or RAY_HEAD_ADDRESS if set.
        namespace: Ray namespace to use for all operations. If None, uses the default namespace.

    Returns:
        ParallelResources with details about the initialized Ray cluster and resource allocation.

    Raises:
        ValueError: If num_cpus_user is set to a value that leaves no CPUs available in case of starting a local ray instance.
    """
    if ray.is_initialized():
        logger.debug("Ray is already initialized. Skipping initialization.")

    elif (addr := address or os.environ.get("RAY_ADDRESS") or os.environ.get("RAY_HEAD_ADDRESS")) not in (
        None,
        "local",
    ):
        logger.info(f"Connecting to existing Ray cluster at {addr}.")
        ray.init(address=addr, namespace=namespace)

    else:
        total_cpus_local = _get_locally_available_cpus()

        if num_cpus_user == 0:
            num_cpus = total_cpus_local
        elif num_cpus_user < 0:
            num_cpus = total_cpus_local + num_cpus_user  # Negative means leave some CPUs free
            if num_cpus <= 0:
                raise ValueError(
                    f"num_cpus is set to {num_cpus_user}, which leaves no CPUs available (total_cpus={total_cpus_local})."
                )
        elif num_cpus_user > total_cpus_local:
            raise ValueError(
                f"Requested num_cpus={num_cpus_user} exceeds total locally available CPUs ({total_cpus_local}). This may "
                "lead to oversubscription and degraded performance. Reduce num_cpus or set it to 0 to use all available CPUs. "
            )
        else:
            num_cpus = num_cpus_user

        logger.info(f"Creating a local ray instance with {num_cpus} CPUs.")
        ray.init(address="local", num_cpus=num_cpus, namespace=namespace)

    resources = ParallelResources.create(
        num_outersplits=1 if run_single_outersplit else num_outersplits,
        log_dir=log_dir,
    )

    if ray_address := ray.get_runtime_context().gcs_address:
        os.environ["RAY_ADDRESS"] = ray_address
    else:
        os.environ.pop("RAY_ADDRESS", None)

    return resources


def shutdown() -> None:
    """Shut down Ray if initialized. Safe to call multiple times."""
    if ray.is_initialized():
        ray.shutdown()

    # Clear RAY_ADDRESS to avoid stale references after shutdown
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_ADDRESS", None)


def _setup_worker_logging(log_dir: UPath):
    """Setup logging for Ray worker processes."""
    # We could log to individual files, e.g. per task:
    # task_id = ray.get_runtime_context().get_task_id()
    # worker_log_file = log_dir / f"octo_worker.{task_id}.log"
    # but for now every worker just logs into the same file
    worker_log_file = log_dir / "study.log"
    set_logger_filename(log_file=worker_log_file)


def run_parallel_outer(
    outersplit_data: OuterSplits,
    run_fn: Callable[[int, OuterSplit, ParallelResources], None],
    resources: ParallelResources,
) -> None:
    """Execute run_fn(outersplit_id, outersplit, num_cpus_per_worker) in parallel using Ray.

    Preserves input order. Essentially, one Ray actor is created per outer task, and each
    actor executes run_fn for its assigned outersplit_id. The runtime environment of the
    subprocesses is configured to allow inner parallelism (e.g. by AutoGluon)
    without oversubscribing CPUs through setting environment variables that many
    libraries respect (e.g. OpenBLAS, MKL, NumExpr, etc.) to
    num_cpus_per_worker and via a threadpoolctl limit.

    Args:
        outersplit_data: Dictionary mapping outersplit_id to OuterSplit(traindev, test).
        run_fn: Function called as run_fn(outersplit_id, outersplit, num_cpus_per_worker).
        resources: Resource configuration for parallel execution, including CPU counts and Ray placement group.
    """
    run(
        context="outer",
        tasks=[
            partial(
                run_fn,
                outersplit_id,
                outersplit,
            )
            for outersplit_id, outersplit in outersplit_data.items()
        ],
        resources=resources,
    )


def run_parallel_inner(
    bag_id: str,
    trainings: Sequence[SupportsFit],
    resources: ParallelResources,
) -> list[Training]:
    """Run training.fit() for each item inside trainings in parallel.

    Args:
        bag_id: Identifier for the bag.
        trainings: Objects with fit() method.
        resources: Allocated resources for the parallel task.

    Returns:
        Results from each training.fit() in input order.
    """

    class Wrapper:
        def __init__(self, training: SupportsFit):
            self.training = training

        def __call__(self, resources: ParallelResources) -> Training:
            return self.training.fit()

    return run(
        context=f"bag_{bag_id}_inner",
        tasks=[Wrapper(training) for training in trainings],
        resources=resources,
    )


def run[T](
    context: str,
    tasks: list[Callable[[ParallelResources], T]],
    resources: ParallelResources,
) -> list[T]:
    """Run tasks in parallel using Ray if num_workers > 1, otherwise run sequentially.

    Args:
        context: Description of the task context for logging purposes.
        tasks: List of callables that take a ParallelResources argument for potential sub-task parallelization and return a result.
        resources: Allocated resources for the parallel task.

    Returns:
        List of results from each task in input order.
    """
    worker_resources = resources.split(num_tasks=len(tasks))

    if len(worker_resources) == 1:
        res = worker_resources[0]

        logger.debug(f"Running {context} sequentially without Ray as num_workers=1.")
        _setup_worker_logging(res.log_dir)
        # TODO: can we locally set the environment variables and threadpoolctl limits properly here to allow inner parallelism even in the sequential case? Do we need a subprocess for that?
        with threadpoolctl.threadpool_limits(limits=res.num_cpus):
            try:
                return [task(res) for task in tasks]
            except Exception as e:
                logger.exception(f"Exception in sequential execution of {context}: {e!s}")
                raise e

    else:
        logger.debug(
            f"Running {context} in parallel with Ray using the following resources: {len(worker_resources)} workers "
            f"with {','.join(str(res.num_cpus) for res in worker_resources)} CPUs per worker."
        )

        if not ray.is_initialized():
            raise RuntimeError("Ray is not initialized. Call ray_parallel.init() first.")

        @ray.remote
        def run_task(
            resources: ParallelResources, task_idx: int, task: Callable[[ParallelResources], T]
        ) -> tuple[ParallelResources, int, T]:
            _setup_worker_logging(resources.log_dir)
            with threadpoolctl.threadpool_limits(limits=resources.num_cpus):
                try:
                    result = task(resources)
                    logger.debug(f"Completed task {task_idx} in parallel execution of {context}.")
                    return resources, task_idx, result
                except Exception as e:
                    logger.exception(
                        f"Exception in task {task_idx} during parallel execution of {context}_task{task_idx}: {e!s}"
                    )
                    raise e

        # Fill task queue and limit concurrency to num_assigned_cpus to avoid oversubscription.
        # Approach from https://docs.ray.io/en/latest/ray-core/patterns/limit-pending-tasks.html

        results: list[T] = [None] * len(tasks)  # type: ignore[list-item]
        free_resources = deque(worker_resources)

        inflight_refs: list[ray.ObjectRef] = []
        for task_idx, task in enumerate(tasks):
            if not free_resources:
                # wait for at least one task to complete before launching more to limit resource usage
                ready_refs, inflight_refs = ray.wait(inflight_refs, num_returns=1)

                for ref in ready_refs:
                    res, result_idx, result = ray.get(ref)
                    results[result_idx] = result
                    free_resources.append(res)  # make the resources of the completed task available for new tasks

            res = free_resources.popleft()

            inflight_refs.append(
                run_task.options(
                    name=f"{context}_task_{task_idx}",
                    # logically do not reserve any CPUs on sub-tasksas we take care of scheduling/resource allocation ourselves here...
                    num_cpus=res.num_cpus if res.is_root else 0,
                    # ... and set environment variables to prevent oversubscription inside the tasks
                    runtime_env=RuntimeEnv(env_vars=res.get_env_vars()),
                    # Ensure all child tasks (e.g. from inner parallelism in AutoGluon) are scheduled on the same placement group to keep resources together
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=res.placement_group, placement_group_capture_child_tasks=True
                    ),
                ).remote(res, task_idx, task)
            )

        # Wait for any remaining tasks to complete
        for res, result_idx, result in ray.get(inflight_refs):
            results[result_idx] = result
            free_resources.append(res)

        if len(free_resources) != len(worker_resources):
            logger.warning(
                f"Resource leak detected in parallel execution of {context}: {len(worker_resources) - len(free_resources)} resources not returned. "
                f"Final resource state: {len(free_resources)} available resources, results: [{results}]"
            )

        return results
