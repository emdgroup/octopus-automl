"""Ray parallelization for outer and inner loops."""

import os
from collections.abc import Callable, Sequence
from typing import Any, TypedDict

import ray
import threadpoolctl
from attrs import define, field, validators
from ray.runtime_env import RuntimeEnv
from upath import UPath

from octopus.datasplit import OuterSplit, OuterSplits
from octopus.logger import get_logger, set_logger_filename
from octopus.modules.octo.bag import FeatureImportanceWithLogging, TrainingWithLogging
from octopus.modules.octo.training import Training

logger = get_logger()

_PARALLELIZATION_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _get_locally_available_cpus() -> int:
    """Get available CPUs on the system."""
    if (total_cpus := os.cpu_count()) is not None:
        return total_cpus
    else:
        raise RuntimeError("Could not determine number of CPUs.")


class _NodeResources(TypedDict):
    """Compute resources available on a Ray node."""

    memory: float
    CPU: float
    object_store_memory: float


def _get_ray_nodes() -> dict[str, _NodeResources]:
    if not ray.is_initialized():
        raise RuntimeError("Ray is not initialized. Call ray_parallel.init() first.")

    ray_nodes: dict[str, _NodeResources] = {}
    for node in ray.nodes():
        res = node.get("Resources", {})
        if name := (", ".join(k[5:] for k in res if k.startswith("node:"))).strip():
            ray_nodes[name] = {
                "CPU": res.get("CPU", 0.0),
                "memory": res.get("memory", 0.0),
                "object_store_memory": res.get("object_store_memory", 0.0),
            }

    return ray_nodes


@define(frozen=True)
class ResourceConfig:
    """Immutable configuration for CPU resources."""

    available_cpus: int = field(validator=validators.instance_of(int))
    """Total number of CPUs available for parallel processing (inner * outer parallelization)."""

    num_workers: int = field(validator=validators.instance_of(int))
    """Number of parallel outer workers."""

    cpus_per_worker: int = field(validator=validators.instance_of(int))
    """CPUs allocated to each worker for inner parallelization."""

    ray_nodes: dict[str, _NodeResources] = field(validator=validators.instance_of(dict))
    """Dictionary of Ray nodes and their resources, used for calculating available_cpus and num_workers."""

    num_outer_splits: int = field(validator=validators.instance_of(int))
    """Total number of outer splits in the study."""

    run_single_outer_split: bool = field(validator=validators.instance_of(bool))
    """Whether to run a single outer split instead of all. This is mainly used for testing and debugging."""

    @classmethod
    def create(
        cls,
        ray_nodes: dict[str, _NodeResources],
        num_outer_splits: int,
        run_single_outer_split: bool,
    ) -> "ResourceConfig":
        """Create ResourceConfig with computed values.

        Args:
            ray_nodes: Dictionary of Ray nodes and their available CPU resources.
            num_outer_splits: Total number of outer splits in the study.
            run_single_outer_split: Whether to run a single outer split instead of all.
              This is mainly used for testing and debugging.

        Returns:
            ResourceConfig with computed worker and CPU allocation.

        Raises:
            ValueError: If any input parameter is invalid.
        """
        if num_outer_splits <= 0:
            raise ValueError(f"num_outer_splits must be positive, got {num_outer_splits}")

        # Calculate effective number of outer splits for resource allocation
        effective_num_outer_splits = 1 if run_single_outer_split else num_outer_splits

        # TODO: instead of summing all CPUs we should properly use the node/resource architecture, i.e. num_workers should be a multiple of len(nodes)
        available_cpus = sum(int(node["CPU"]) for node in ray_nodes.values())

        # Calculate resource allocation
        num_workers = min(effective_num_outer_splits, available_cpus)
        if num_workers == 0:
            raise ValueError(
                f"Cannot allocate resources: num_workers computed as 0 "
                f"(effective_num_outer_splits={effective_num_outer_splits}, "
                f"available_cpus={available_cpus})"
            )

        return cls(
            available_cpus=available_cpus,
            num_workers=num_workers,
            cpus_per_worker=max(1, available_cpus // num_workers),
            ray_nodes=ray_nodes,
            num_outer_splits=num_outer_splits,
            run_single_outer_split=run_single_outer_split,
        )

    def __str__(self) -> str:
        """Return string representation of resource configuration."""
        nodes = "\n\t".join(f"Node {node}:  {res['CPU']} CPUs" for node, res in self.ray_nodes.items())

        return (
            f"\nSingle outer split: {self.run_single_outer_split}"
            f"\nOuter splits:      {self.num_outer_splits}"
            f"\nAvailable CPUs:    {self.available_cpus}"
            f"\nWorkers:           {self.num_workers}"
            f"\nCPUs/outer split:  {self.cpus_per_worker}"
            f"\nRay Nodes:\n\t{nodes}"
        )


def init(
    num_cpus_user: int,
    num_outer_splits: int,
    run_single_outer_split: bool,
    address: str | None = None,
    namespace: str | None = None,
):
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
        num_outer_splits: Total number of outer splits in the study.
        run_single_outer_split: Whether to run a single outer split instead of all. This is mainly used for testing and debugging.
        address: Ray head address (e.g., "auto", "127.0.0.1:6379", "local"). If None, uses
            env vars RAY_ADDRESS or RAY_HEAD_ADDRESS if set.
        namespace: Ray namespace to use for all operations. If None, uses the default namespace.

    Returns:
        ResourceConfig with details about the initialized Ray cluster and resource allocation.

    Raises:
        ValueError: If num_cpus_user is set to a value that leaves no CPUs available in case of starting a local ray instance.
    """
    if ray.is_initialized():
        logger.info("Ray is already initialized. Skipping initialization.")

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

    ray_nodes = _get_ray_nodes()

    resource_config = ResourceConfig.create(
        ray_nodes=ray_nodes,
        num_outer_splits=num_outer_splits,
        run_single_outer_split=run_single_outer_split,
    )

    if ray_address := ray.get_runtime_context().gcs_address:
        os.environ["RAY_ADDRESS"] = ray_address
    else:
        os.environ.pop("RAY_ADDRESS", None)

    return resource_config


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
    outer_split_data: OuterSplits,
    run_fn: Callable[[int, OuterSplit, int], None],
    log_dir: UPath,
    num_cpus_per_worker: int,
) -> None:
    """Execute run_fn(outer_split_id, outer_split, num_cpus_per_worker) in parallel using Ray.

    Preserves input order. Essentially, one Ray actor is created per outer task, and each
    actor executes run_fn for its assigned outer_split_id. The runtime environment of the
    subprocesses is configured to allow inner parallelism (e.g. by AutoGluon)
    without oversubscribing CPUs through setting environment variables that many
    libraries respect (e.g. OpenBLAS, MKL, NumExpr, etc.) to
    num_cpus_per_worker and via a threadpoolctl limit.

    Args:
        outer_split_data: Dictionary mapping outer_split_id to OuterSplit(traindev, test).
        run_fn: Function called as run_fn(outer_split_id, outer_split, num_cpus_per_worker).
        log_dir: Directory to store individual Ray worker logs.
        num_cpus_per_worker: CPUs used for each outer task to prevent
          oversubscription during inner parallel work. Outer workers do not reserve these
          CPUs by themselves but set them in the environment for libraries to respect and
          enforce via threadpoolctl in inner parallel code. This allows inner parallelism
          (e.g. by AutoGluon) without oversubscribing CPUs.
    """
    if not ray.is_initialized():
        raise RuntimeError("Ray is not initialized. Call ray_parallel.init() first.")

    class OuterTask:
        def __init__(self, outer_split_id: int, outer_split: OuterSplit, log_dir: UPath, num_cpus: int):
            _setup_worker_logging(log_dir)
            self.outer_split_id = outer_split_id
            self.outer_split = outer_split
            self.num_cpus = num_cpus

        @ray.method
        def run(self):
            with threadpoolctl.threadpool_limits(limits=self.num_cpus):
                run_fn(self.outer_split_id, self.outer_split, self.num_cpus)
            return self.outer_split_id

    OuterTaskActor = ray.remote(OuterTask)

    # do our best to prevent oversubscription of CPUs by setting environment variables that many libraries respect (e.g. OpenBLAS, MKL, NumExpr, etc.)
    runtime_env = RuntimeEnv(env_vars={var: str(num_cpus_per_worker) for var in _PARALLELIZATION_ENV_VARS})

    futures = [
        OuterTaskActor.options(
            name=f"outer_task_{outer_split_id}",
            num_cpus=num_cpus_per_worker,  # Outer task reserves all CPUs required for individual inner parallelization
            runtime_env=runtime_env,
        )
        .remote(outer_split_id, outer_split, log_dir, num_cpus_per_worker)
        .run.remote()
        for outer_split_id, outer_split in outer_split_data.items()
    ]
    ray.get(futures)


def run_parallel_inner(
    bag_id: str,
    trainings: Sequence[TrainingWithLogging | FeatureImportanceWithLogging],
    log_dir: UPath,
    num_assigned_cpus: int,
) -> list[Training]:
    """Run training.fit() for each item inside trainings in parallel.

    Args:
        bag_id: Identifier for the bag.
        trainings: Objects with fit() method.
        log_dir: Directory to store individual Ray worker logs.
        num_assigned_cpus: CPUs for parallel execution of the fit() methods.

    Returns:
        Results from each training.fit() in input order.

    Raises:
        RuntimeError: If Ray is not initialized.
    """
    if not ray.is_initialized():
        raise RuntimeError("Ray is not initialized. Call ray_parallel.init() first.")

    # do our best to prevent oversubscription of CPUs by setting environment variables that many libraries respect (e.g. OpenBLAS, MKL, NumExpr, etc.)
    runtime_env = RuntimeEnv(env_vars=dict.fromkeys(_PARALLELIZATION_ENV_VARS, "1"))

    # num_assigned_cpus inner tasks will run in parallel (See below), each task only gets one CPU
    @ray.remote
    def execute_training(training: Any, idx: int, log_dir: UPath) -> tuple[int, Training]:
        _setup_worker_logging(log_dir)
        with threadpoolctl.threadpool_limits(limits=1):
            return idx, training.fit()

    # Fill task queue and limit concurrency to num_assigned_cpus to avoid oversubscription.
    # Approach from https://docs.ray.io/en/latest/ray-core/patterns/limit-pending-tasks.html

    results: list[Training] = [None] * len(trainings)  # type: ignore[list-item]

    inflight_refs: list[ray.ObjectRef] = []
    for training_idx, training in enumerate(trainings):
        if len(inflight_refs) >= num_assigned_cpus:
            # wait for at least one task to complete before launching more to limit resource usage
            ready_refs, inflight_refs = ray.wait(inflight_refs, num_returns=1)

            for ref in ready_refs:
                idx, result = ray.get(ref)
                results[idx] = result

        inflight_refs.append(
            execute_training.options(
                name=f"{bag_id}_inner_task_{training_idx}",
                num_cpus=0,  # logically do not reserve any CPUs for the inner tasks as the outer task reserved enough CPUs for the inner parallelization.
                runtime_env=runtime_env,
            ).remote(training, training_idx, log_dir)
        )

    # Wait for any remaining tasks to complete
    for idx, result in ray.get(inflight_refs):
        results[idx] = result

    return results
