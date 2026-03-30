import os
import subprocess

import pandas as pd
import pytest
import ray
import threadpoolctl
from upath import UPath

from octopus.datasplit import OuterSplit
from octopus.manager import ray_parallel


@pytest.fixture
def outersplits():
    return {
        0: OuterSplit(traindev=pd.DataFrame(), test=pd.DataFrame()),
        1: OuterSplit(traindev=pd.DataFrame(), test=pd.DataFrame()),
        2: OuterSplit(traindev=pd.DataFrame(), test=pd.DataFrame()),
        3: OuterSplit(traindev=pd.DataFrame(), test=pd.DataFrame()),
    }


def test_inner_parallelization_setup_in_workers(tmp_path, outersplits):
    resources = ray_parallel.init(
        num_cpus_user=0,
        num_outersplits=len(outersplits),
        run_single_outersplit=False,
        namespace="test_namespace",
    )

    def run_fn(outersplit_id: int, outersplit: OuterSplit, num_cpus_per_worker: int):
        assert num_cpus_per_worker == resources.cpus_per_worker

        for var in ray_parallel._PARALLELIZATION_ENV_VARS:
            assert os.environ.get(var, None) == str(resources.cpus_per_worker), (
                f"Expected {var}={resources.cpus_per_worker} in worker environment, but got {os.environ.get(var, None)}"
            )

        threadpool_info = threadpoolctl.threadpool_info()

        # we expect the following openmp libraries to be loaded: shipped with torch, shipped with sklearn, system openmp
        assert len(threadpool_info) >= 2

        for lib in threadpool_info:
            assert lib["num_threads"] == 1

    ray_parallel.run_parallel_outer(
        outersplit_data=outersplits,
        run_fn=run_fn,
        log_dir=UPath(tmp_path),
        num_workers=resources.num_workers,
        num_cpus_per_worker=resources.cpus_per_worker,
    )

    ray_parallel.shutdown()


@pytest.mark.skip(reason="Deadlock in Github CI to be investigated")
@pytest.mark.parametrize("num_nodes", [1, 2, 3], ids=lambda n: f"{n}_node(s)")
def test_connect_to_running_ray_cluster(tmp_path, outersplits, num_nodes):
    HOST = "127.0.0.1"
    PORT = 6379
    CPUS_PER_NODE = 4

    # 1. Start a Ray head node as a subprocess (separate process, survives ray.shutdown())
    subprocess.run(
        [
            "ray",
            "start",
            "--head",
            f"--num-cpus={CPUS_PER_NODE}",
            f"--port={PORT}",
            f"--dashboard-host={HOST}",
        ],
        check=True,
    )

    for _ in range(1, num_nodes):
        subprocess.run(
            [
                "ray",
                "start",
                f"--address={HOST}:{PORT}",
                f"--num-cpus={CPUS_PER_NODE}",
            ],
            check=True,
        )

    try:
        resources = ray_parallel.init(
            num_cpus_user=0,
            num_outersplits=len(outersplits),
            run_single_outersplit=False,
            address=f"{HOST}:{PORT}",
            namespace="test_namespace",
        )

        def run_fn(outersplit_id: int, outersplit: OuterSplit, num_cpus_per_worker: int):
            assert num_cpus_per_worker == resources.cpus_per_worker

        ray_parallel.run_parallel_outer(
            outersplit_data=outersplits,
            run_fn=run_fn,
            log_dir=UPath(tmp_path),
            num_workers=resources.num_workers,
            num_cpus_per_worker=resources.cpus_per_worker,
        )
    finally:
        ray.shutdown()
        subprocess.run(["ray", "stop"], check=True)
