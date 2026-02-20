"""Ray parallelization for outer and inner loops."""

import os
from collections.abc import Callable, Iterable
from typing import Any

import ray
import threadpoolctl
from ray import ObjectRef
from upath import UPath

from octopus.datasplit import OuterSplit, OuterSplits
from octopus.logger import set_logger_filename


def init_ray(
    address: str | None = None,
    num_cpus: int | None = None,
    start_local_if_missing: bool = False,
    **kwargs,
) -> None:
    """Initialize Ray for the current process.

    Connects to an existing cluster if an address is provided or set via
    environment variables; otherwise, optionally starts a local Ray instance.

    Args:
        address: Ray head address (e.g., "auto", "127.0.0.1:6379"). If None, uses
            env vars RAY_ADDRESS or RAY_HEAD_ADDRESS if set.
        num_cpus: CPU limit when starting a local Ray instance (only used if starting locally).
        start_local_if_missing: If True and no address is available, start a local Ray instance.
        **kwargs: Extra args forwarded to ray.init (e.g., runtime_env, log_to_driver, namespace).

    Raises:
        RuntimeError: If no address is available and start_local_if_missing is False.
    """
    if ray.is_initialized():
        return

    addr = address or os.environ.get("RAY_ADDRESS") or os.environ.get("RAY_HEAD_ADDRESS")
    if addr:
        ray.init(
            address=addr,
            runtime_env={"worker_process_setup_hook": _check_parallelization_disabled},
            **kwargs,
        )
        return

    if start_local_if_missing:
        ray.init(
            num_cpus=num_cpus,
            runtime_env={"worker_process_setup_hook": _check_parallelization_disabled},
            **kwargs,
        )
        return

    raise RuntimeError(
        "No Ray address provided. Set RAY_ADDRESS env, pass address='auto', or call init_ray(..., start_local_if_missing=True) once in the driver."
    )


def shutdown_ray() -> None:
    """Shut down Ray if initialized. Safe to call multiple times."""
    if ray.is_initialized():
        ray.shutdown()
    # Clear RAY_ADDRESS to avoid stale references after shutdown
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_ADDRESS", None)


def setup_ray_for_external_library() -> None:
    """Configure environment to enable external libraries to use the existing Ray instance.

    Sets RAY_ADDRESS to the current Ray GCS address, preventing external libraries
    (e.g., AutoGluon, Ray Tune) from creating separate Ray instances that would
    cause resource conflicts.

    Should be called before using external libraries that may use Ray.
    """
    if ray.is_initialized():
        ray_address = ray.get_runtime_context().gcs_address
        if ray_address:
            os.environ["RAY_ADDRESS"] = ray_address
    else:
        # If Ray is not initialized, clear the RAY_ADDRESS to avoid stale references
        os.environ.pop("RAY_ADDRESS", None)


def _check_parallelization_disabled() -> None:
    """Raise an error if any kind of active parallelization (OMP, MKL, threadpools, ...) can be detected.

    This is required to prevent accidental OMP parallelization inside ray processes that can lead to oversubscription.
    """
    from octopus.modules import _PARALLELIZATION_ENV_VARS  # noqa: PLC0415

    for lib in threadpoolctl.threadpool_info():
        if lib["num_threads"] > 1:
            raise RuntimeError(
                f"Active thread-level parallelization detected in {lib}."
                "This may lead to resource oversubscription and slow execution. "
                "Please disable thread-level parallelization by setting respective "
                "environment variables."
            )

    for env_var in _PARALLELIZATION_ENV_VARS:
        if os.environ.get(env_var, None) != "1":
            raise RuntimeError(
                f"Environment variable {env_var} is set to {os.environ.get(env_var)}. "
                "This may lead to resource oversubscription and slow execution. "
                "Please set it to 1 to disable thread-level parallelization."
            )


def _setup_worker_logging(log_dir: UPath):
    """Setup logging for Ray worker processes."""
    # We could log to individual files, e.g. per task:
    # task_id = ray.get_runtime_context().get_task_id()
    # worker_log_file = log_dir / f"octo_worker.{task_id}.log"
    # but for now every worker just logs into the same file
    worker_log_file = log_dir / "octo_manager.log"
    set_logger_filename(log_file=worker_log_file)


def run_parallel_outer_ray(
    outersplit_data: OuterSplits,
    run_fn: Callable[[int, OuterSplit], None],
    log_dir: UPath,
    num_workers: int,
) -> None:
    """Execute run_fn(outersplit_id, outersplit) in parallel using Ray.

    Preserves input order and limits concurrency to num_workers. Outer tasks reserve
    0 CPUs so inner Ray work can use available CPUs.

    Args:
        outersplit_data: Dictionary mapping outersplit_id to OuterSplit(traindev, test).
        run_fn: Function called as run_fn(outersplit_id, outersplit).
        log_dir: Directory to store individual Ray worker logs.
        num_workers: Maximum number of concurrent outer tasks.
    """
    # Ensure Ray is ready in the driver (connect or start local)
    init_ray(start_local_if_missing=True)

    @ray.remote(num_cpus=0)
    def outer_task(outersplit_id: int, outersplit: OuterSplit, log_dir: UPath) -> int:
        _setup_worker_logging(log_dir)
        run_fn(outersplit_id, outersplit)
        return outersplit_id

    outersplit_ids = list(outersplit_data.keys())
    n = len(outersplit_ids)
    if n == 0:
        return

    max_concurrent = max(1, min(num_workers, n))
    inflight: list[ObjectRef] = []
    next_i = 0

    # Prime up to max_concurrent tasks
    while next_i < n and len(inflight) < max_concurrent:
        outersplit_id = outersplit_ids[next_i]
        inflight.append(
            outer_task.remote(outersplit_id, outersplit_data[outersplit_id], log_dir)
        )
        next_i += 1

    # Drain with backpressure
    while inflight:
        done, inflight = ray.wait(inflight, num_returns=1)
        ray.get(done[0])
        if next_i < n:
            outersplit_id = outersplit_ids[next_i]
            inflight.append(
                outer_task.remote(outersplit_id, outersplit_data[outersplit_id], log_dir)
            )
            next_i += 1


def run_parallel_inner(trainings: Iterable[Any], log_dir: UPath, num_cpus: int = 1) -> list[Any]:
    """Run training.fit() for each item in parallel.

    Args:
        trainings: Objects with fit() method.
        log_dir: Directory to store individual Ray worker logs.
        num_cpus: CPUs per training task.

    Returns:
        Results from each training.fit() in input order.

    Raises:
        RuntimeError: If Ray is not initialized.
    """
    if not ray.is_initialized():
        raise RuntimeError("Ray is not initialized. Call init_ray() first.")

    @ray.remote(num_cpus=num_cpus)
    def execute_training(training: Any, idx: int, log_dir: UPath) -> Any:
        _setup_worker_logging(log_dir)
        return training.fit()

    futures = [execute_training.remote(training, idx, log_dir) for idx, training in enumerate(trainings)]
    return ray.get(futures)
