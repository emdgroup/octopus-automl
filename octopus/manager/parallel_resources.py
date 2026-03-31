"""Classes for managing and representing parallel resources/CPU partitions in Ray-based parallelization."""

from typing import TypedDict

import ray
from attrs import define
from ray.util.placement_group import PlacementGroup
from upath import UPath

from octopus.logger import get_logger

logger = get_logger()


_PARALLELIZATION_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class NodeResources(TypedDict):
    """Compute resources available on a Ray node."""

    memory: float
    CPU: float
    object_store_memory: float


def _get_ray_nodes() -> dict[str, NodeResources]:
    if not ray.is_initialized():
        raise RuntimeError("Ray is not initialized. Call ray_parallel.init() first.")

    ray_nodes: dict[str, NodeResources] = {}
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
class ParallelResources:
    """Helper Class to represent allocated resources for a parallel task."""

    log_dir: UPath
    num_cpus: int
    placement_group: PlacementGroup
    ray_nodes: dict[str, NodeResources]
    is_root: bool

    def split(self, num_tasks: int) -> list["ParallelResources"]:
        """Distribute available resources across num_tasks tasks."""
        num_workers = min(num_tasks, self.num_cpus)
        cpus_per_worker = max(1, self.num_cpus // num_workers)

        return [
            ParallelResources(
                log_dir=self.log_dir,
                num_cpus=cpus_per_worker,
                placement_group=self.placement_group,
                ray_nodes=self.ray_nodes,
                is_root=False,
            )
            for _ in range(num_workers)
        ]

    def get_env_vars(self) -> dict[str, str]:
        """Get environment variables to set for parallel tasks to prevent CPU oversubscription."""
        return dict.fromkeys(_PARALLELIZATION_ENV_VARS, str(self.num_cpus))

    @classmethod
    def create(
        cls,
        num_outersplits: int,
        log_dir: UPath,
    ) -> "ParallelResources":
        """Create ParallelResources with computed values.

        Args:
            num_outersplits: Total number of outersplits in the study.
            log_dir: Directory for Ray worker logging.

        Returns:
            ParallelResources with computed worker and CPU allocation.

        Raises:
            ValueError: If any input parameter is invalid.
        """
        if num_outersplits <= 0:
            raise ValueError(f"num_outersplits must be positive, got {num_outersplits}")

        ray_nodes = _get_ray_nodes()

        # TODO: instead of summing all CPUs we should properly use the node/resource architecture, i.e. num_workers should be a multiple of len(nodes)
        available_cpus = sum(int(node["CPU"]) for node in ray_nodes.values())
        cpus_on_smallest_node = min(int(node["CPU"]) for node in ray_nodes.values())

        # Calculate resource allocation
        num_workers = min(num_outersplits, available_cpus)
        if num_workers == 0:
            raise ValueError(
                f"Cannot allocate resources: num_workers computed as 0 (effective_num_outersplits={num_outersplits}, available_cpus={available_cpus})"
            )

        num_cpus_per_worker = min(cpus_on_smallest_node, max(1, available_cpus // num_workers))

        # Reserve a placement group of num_workers bundle that reserves num_cpus_per_worker CPUs
        pg_bundles = [{"CPU": num_cpus_per_worker}] * num_workers

        logger.info(
            f"Creating Ray placement group with {num_workers} bundles, each reserving {num_cpus_per_worker} CPUs for parallel workers: {pg_bundles}"
        )
        pg = ray.util.placement_group(pg_bundles)

        logger.info("Waiting for Ray placement group to be ready...")
        while not pg.wait(timeout_seconds=5):
            logger.info("... still waiting ...")

        logger.info("Ray placement group is ready.")

        return cls(
            log_dir=log_dir,
            num_cpus=num_cpus_per_worker * num_workers,
            placement_group=pg,
            ray_nodes=ray_nodes,
            is_root=True,
        )

    def __str__(self) -> str:
        """Return string representation of resource configuration."""
        nodes = "\n\t".join(
            f"Node {node} --> " + ", ".join(f"{key}: {value}" for key, value in res.items())
            for node, res in self.ray_nodes.items()
        )
        pg_specs = "\n\t".join(
            ", ".join(f"{key}: {value}" for key, value in bundle.items())
            for bundle in self.placement_group.bundle_specs
        )

        return f"""
Available CPUs:    {self.num_cpus}
Ray Nodes:
\t{nodes}
Placement Group Specs:
\t{pg_specs}"""
