"""Test ResourceConfig, OctoManager, and WorkflowTaskRunner from octopus.manager."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest
from upath import UPath

from octopus.datasplit import OuterSplit
from octopus.manager import OctoManager, ray_parallel
from octopus.manager.ray_parallel import ResourceConfig, _NodeResources
from octopus.manager.workflow_runner import WorkflowTaskRunner
from octopus.modules import StudyContext
from octopus.types import MLType

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_workflow():
    """Create mock workflow."""
    return [
        Mock(
            task_id=1,
            depends_on=None,
            module="test_module",
            description="Test",
        ),
        Mock(
            task_id=2,
            depends_on=1,
            module="test_module",
            description="Test",
        ),
    ]


@pytest.fixture
def mock_outersplit_data():
    """Create mock fold splits."""
    return {
        0: OuterSplit(
            traindev=pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4], "target": [0, 1]}),
            test=pd.DataFrame({"feature1": [5], "feature2": [6], "target": [1]}),
        ),
        1: OuterSplit(
            traindev=pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4], "target": [0, 1]}),
            test=pd.DataFrame({"feature1": [5], "feature2": [6], "target": [1]}),
        ),
    }


@pytest.fixture
def study():
    """Create StudyContext for testing."""
    return StudyContext(
        ml_type=MLType.BINARY,
        target_metric="AUCROC",
        target_assignments={"default": "target"},
        positive_class=1,
        stratification_col=None,
        sample_id_col="sample_id",
        feature_cols=["feature1", "feature2"],
        row_id_col="row_id",
        output_path=UPath("/tmp/test_study"),
        log_dir=UPath("/tmp/test_study"),
    )


@pytest.fixture
def octo_manager(study, mock_workflow, mock_outersplit_data):
    """Create octo manager."""
    return OctoManager(
        outersplit_data=mock_outersplit_data,
        study_context=study,
        workflow=mock_workflow,
        num_cpus=0,
        run_single_outersplit_num=None,
    )


# =============================================================================
# ResourceConfig Tests
# =============================================================================


class TestResourceConfig:
    """Tests for ResourceConfig."""

    @pytest.mark.parametrize("num_cpus", [1, 4, 14], ids=lambda n: f"{n}_cpus")
    @pytest.mark.parametrize("num_outersplits", [1, 4, 14, 47], ids=lambda n: f"{n}_outersplits")
    @pytest.mark.parametrize("run_single_outersplit", [False, True], ids=["parallel", "single_outersplit"])
    def test_create_with_parallelization(self, num_cpus, num_outersplits, run_single_outersplit):
        """Test resource creation and proper computation of num_workers and num_cpus_per_worker."""
        effective_num_outersplits = num_outersplits if not run_single_outersplit else 1
        expected_num_workers = min(effective_num_outersplits, num_cpus)
        expected_cpus_per_worker = max(1, num_cpus // expected_num_workers)
        expected_num_cpus = expected_cpus_per_worker * expected_num_workers
        ray_nodes: dict[str, _NodeResources] = {
            "local": {
                "CPU": num_cpus,
                "memory": 16 * 1024**3,
                "object_store_memory": 8 * 1024**3,
            }
        }  # Simulate a single-node Ray cluster with all CPUs and 16GB RAM

        config = ResourceConfig.create(
            ray_nodes=ray_nodes,
            num_outersplits=num_outersplits,
            run_single_outersplit=run_single_outersplit,
        )
        assert config.available_cpus == num_cpus
        assert config.used_cpus == expected_num_cpus
        assert config.num_workers == expected_num_workers
        assert config.cpus_per_worker == expected_cpus_per_worker

        str_repr = str(config)

        # Verify all key information is in the string
        assert f"Single outersplit: {run_single_outersplit}" in str_repr
        assert f"Outersplits:       {num_outersplits}" in str_repr
        assert f"Available CPUs:    {num_cpus}" in str_repr
        assert f"Used CPUs:         {expected_num_cpus}" in str_repr
        assert f"Workers:           {expected_num_workers}" in str_repr
        assert f"CPUs/outersplit:   {expected_cpus_per_worker}" in str_repr

    @pytest.mark.parametrize("num_outersplits", [0, -5], ids=["zero_outersplits", "negative_outersplits"])
    def test_create_rejects_invalid_outersplits(self, num_outersplits):
        """Test that zero or negative outersplits raises ValueError."""
        ray_nodes: dict[str, _NodeResources] = {
            "local": {
                "CPU": 4,
                "memory": 16 * 1024**3,
                "object_store_memory": 8 * 1024**3,
            }
        }  # Simulate a single-node Ray cluster with all CPUs and 16GB RAM

        with pytest.raises(ValueError, match="num_outersplits must be positive"):
            ResourceConfig.create(
                num_outersplits=num_outersplits,
                run_single_outersplit=False,
                ray_nodes=ray_nodes,
            )


# =============================================================================
# OctoManager Tests
# =============================================================================


class TestOctoManager:
    """Tests for OctoManager orchestration."""

    def test_run_outersplits_sequential(self, octo_manager):
        """Test run outersplits sequential."""
        octo_manager.num_cpus = 1

        with (
            patch.object(WorkflowTaskRunner, "run") as mock_run,
        ):
            octo_manager.run_outersplits()
            assert mock_run.call_count == len(octo_manager.outersplit_data)  # Called once per outersplit

    def test_run_outersplits_parallel(self, study, mock_workflow, mock_outersplit_data):
        """Test run outersplits with parallelization."""
        manager = OctoManager(
            outersplit_data=mock_outersplit_data,
            study_context=study,
            workflow=mock_workflow,
            num_cpus=0,
            run_single_outersplit_num=None,
        )

        with (
            patch("octopus.manager.ray_parallel.run_parallel_outer", return_value=[True]) as mock_ray,
        ):
            manager.run_outersplits()
            mock_ray.assert_called_once()

    def test_run_single_outersplit(self, study, mock_workflow, mock_outersplit_data):
        """Test run single outersplit."""
        manager = OctoManager(
            outersplit_data=mock_outersplit_data,
            study_context=study,
            workflow=mock_workflow,
            num_cpus=1,
            run_single_outersplit_num=0,
        )

        with (
            patch.object(WorkflowTaskRunner, "run") as mock_run,
        ):
            manager.run_outersplits()
            # Verify that run was called with outersplit_id and OuterSplit
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0]
            assert len(call_args) == 3
            assert call_args[0] == 0  # outersplit_id
            assert isinstance(call_args[1], OuterSplit)  # outersplit
            assert call_args[2] == 1  # num_assigned_cpus

    def test_no_outersplits_raises_error(self, study, mock_workflow):
        """Test that empty fold splits raises ValueError."""
        manager = OctoManager(
            outersplit_data={},
            study_context=study,
            workflow=mock_workflow,
            num_cpus=1,
            run_single_outersplit_num=None,
        )
        with pytest.raises(ValueError, match="No outersplit data defined"):
            manager.run_outersplits()

    def test_ray_shutdown_on_error(self, octo_manager):
        """Test that Ray is shut down even if execution fails."""
        octo_manager.num_cpus = 1

        with (
            patch("octopus.manager.ray_parallel.shutdown") as mock_shutdown,
            patch.object(WorkflowTaskRunner, "run", side_effect=RuntimeError("Test error")),
        ):
            with pytest.raises(RuntimeError):
                octo_manager.run_outersplits()
            mock_shutdown.assert_called_once()

        ray_parallel.shutdown()  # Ensure Ray is shut down after test

    def test_single_outersplit_resource_allocation(self, study, mock_workflow):
        """Test that single outersplit gets all CPUs when run_single_outersplit_num is set.

        This is a regression test: previously, when running a single outersplit from
        a set of 8 outersplits on 8 CPUs, the single outersplit would only get 1 CPU
        instead of all 8.
        """
        # Create 8 mock fold splits
        outersplit_data = {
            i: OuterSplit(
                traindev=pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4], "target": [0, 1]}),
                test=pd.DataFrame({"feature1": [5], "feature2": [6], "target": [1]}),
            )
            for i in range(8)
        }

        manager = OctoManager(
            outersplit_data=outersplit_data,
            study_context=study,
            workflow=mock_workflow,
            num_cpus=0,
            run_single_outersplit_num=0,
        )

        with (
            patch("octopus.manager.ray_parallel._get_locally_available_cpus", return_value=8),
            patch.object(WorkflowTaskRunner, "__init__", return_value=None),
            patch.object(WorkflowTaskRunner, "run") as mock_run,
        ):
            manager.run_outersplits()

            # Verify that WorkflowTaskRunner was initialized with cpus_per_outersplit that allocates all CPUs
            # to the single outersplit (not 8/8=1)
            call_args = mock_run.call_args
            num_assigned_cpus = call_args[0][2]  # Now a keyword arg
            assert num_assigned_cpus == 8  # All CPUs for that outersplit


# =============================================================================
# WorkflowTaskRunner Tests
# =============================================================================
