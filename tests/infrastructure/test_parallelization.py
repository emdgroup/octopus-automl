import os
import subprocess
import sys
from pathlib import Path

import pytest
import threadpoolctl

import octopus  # noqa: F401
from octopus.modules import _PARALLELIZATION_ENV_VARS


def test_parallelization_inactive_in_threadpoolctl():
    threadpool_info = threadpoolctl.threadpool_info()

    # we expect the following openmp libraries to be loaded: shipped with torch, shipped with sklearn, system openmp
    assert len(threadpool_info) >= 2

    for lib in threadpool_info:
        assert lib["num_threads"] == 1


def test_parallelization_limited_by_env():
    # these vars are being set in octopus/modules/__init__.py

    for env_var in _PARALLELIZATION_ENV_VARS:
        assert os.environ.get(env_var, None) == "1"


@pytest.mark.skip
def test_ray_workers_detect_active_parallelization():
    """Test that ray workers abort if they detect active parallelization.

    This test spawns a subprocess that runs a test workflow with OMP_NUM_THREADS set to 42
    to make sure the modification of the ENV var is not accidentally carried over to
    other tests.
    """
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "42"  # Activate thread-level parallelization

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            Path(__file__).parent.parent / "workflows" / "test_ag_workflows.py",
            "-k",
            "full_regression_workflow",
        ],
        check=False,
        env=env,
        capture_output=True,
    )

    assert res.returncode == 1  # Expecting failure due to active thread-level parallelization
    assert b"RuntimeError: Environment variable OMP_NUM_THREADS is set to 42." in res.stdout
