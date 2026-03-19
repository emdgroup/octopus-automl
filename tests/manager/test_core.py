"""Test ResourceConfig, OctoManager, and WorkflowTaskRunner from octopus.manager."""

import os
from typing import ClassVar
from unittest.mock import Mock, patch

import attrs
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

    TOTAL_CPUS = os.cpu_count() or 4  # Default to 4 if os.cpu_count() returns None
    RAY_NODES: ClassVar[dict[str, _NodeResources]] = {
        "local": {
            "CPU": TOTAL_CPUS,
            "memory": 16 * 1024**3,
            "object_store_memory": 8 * 1024**3,
        }
    }  # Simulate a single-node Ray cluster with all CPUs and 16GB RAM

    def test_create_with_parallelization(self):
        """Test resource creation with outer parallelization."""
        num_outersplits = 4

        config = ResourceConfig.create(
            ray_nodes=self.RAY_NODES,
            num_outersplits=num_outersplits,
            run_single_outersplit=False,
        )
        assert config.available_cpus == self.RAY_NODES["local"]["CPU"]
        assert config.num_workers == min(num_outersplits, self.RAY_NODES["local"]["CPU"])
        assert config.cpus_per_worker == self.RAY_NODES["local"]["CPU"] // num_outersplits

    def test_create_without_parallelization(self):
        """Test resource creation without outer parallelization."""
        config = ResourceConfig.create(
            ray_nodes=self.RAY_NODES,
            num_outersplits=4,
            run_single_outersplit=True,
        )
        assert config.available_cpus == self.RAY_NODES["local"]["CPU"]
        assert config.num_workers == 1
        assert config.cpus_per_worker == self.RAY_NODES["local"]["CPU"]  # All CPUs for sequential

    def test_create_more_outersplits_than_cpus(self):
        """Test when outersplits exceed available CPUs."""
        config = ResourceConfig.create(
            ray_nodes=self.RAY_NODES,
            num_outersplits=16,
            run_single_outersplit=False,
        )
        assert config.num_workers == min(16, self.RAY_NODES["local"]["CPU"])
        assert config.cpus_per_worker == max(1, self.RAY_NODES["local"]["CPU"] // 16)

    def test_frozen(self):
        """Test that ResourceConfig is immutable (attrs frozen)."""
        config = ResourceConfig(
            available_cpus=4,
            num_workers=2,
            cpus_per_worker=2,
            ray_nodes=self.RAY_NODES,
            run_single_outersplit=False,
            num_outersplits=4,
        )
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            config.available_cpus = 8  # type: ignore[misc]

    def test_create_single_outersplit_gets_all_cpus(self):
        """Test that when running a single outersplit, it gets all CPUs.

        This tests the fix for the regression where single outersplits were
        getting limited CPUs based on the total number of outersplits.
        """
        # Simulate: 8 CPUs, 8 total outersplits, but running only outersplit 0
        config = ResourceConfig.create(
            num_outersplits=8,
            ray_nodes=self.RAY_NODES,
            run_single_outersplit=True,
        )
        assert config.available_cpus == self.TOTAL_CPUS
        assert config.num_workers == 1  # Only 1 outersplit running
        assert config.cpus_per_worker == self.TOTAL_CPUS  # Gets all CPUs, not 8/8=1

    def test_create_rejects_zero_outersplits(self):
        """Test that zero outersplits raises ValueError."""
        with pytest.raises(ValueError, match="num_outersplits must be positive"):
            ResourceConfig.create(
                num_outersplits=0,
                run_single_outersplit=False,
                ray_nodes=self.RAY_NODES,
            )

    def test_create_rejects_negative_outersplits(self):
        """Test that negative outersplits raises ValueError."""
        with pytest.raises(ValueError, match="num_outersplits must be positive"):
            ResourceConfig.create(
                num_outersplits=-5,
                run_single_outersplit=False,
                ray_nodes=self.RAY_NODES,
            )

    def test_str_representation(self):
        """Test string representation of ResourceConfig."""
        num_outersplits = 4

        config = ResourceConfig.create(
            num_outersplits=num_outersplits,
            run_single_outersplit=False,
            ray_nodes=self.RAY_NODES,
        )
        str_repr = str(config)

        # Verify all key information is in the string
        assert "Single outersplit: False" in str_repr
        assert f"Outersplits:       {num_outersplits}" in str_repr
        assert f"Available CPUs:    {self.TOTAL_CPUS}" in str_repr
        assert f"Workers:           {min(num_outersplits, self.TOTAL_CPUS)}" in str_repr
        assert f"CPUs/outersplit:   {self.TOTAL_CPUS // num_outersplits}" in str_repr

        # Verify "Preparing execution" is NOT in the string
        assert "Preparing execution" not in str_repr


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
