"""Test ResourceConfig, OctoManager, and WorkflowTaskRunner from octopus.manager."""

from unittest.mock import Mock, patch

import attrs
import pandas as pd
import pytest
from upath import UPath

from octopus.datasplit import OuterSplit
from octopus.manager import OctoManager
from octopus.manager.core import ResourceConfig
from octopus.manager.workflow_runner import WorkflowTaskRunner
from octopus.study.context import StudyContext
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
            depends_on=0,
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
        )
    }


@pytest.fixture
def study():
    """Create StudyContext for testing."""
    return StudyContext(
        ml_type=MLType.BINARY,
        target_metric="AUCROC",
        metrics=["AUCROC"],
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
        outer_parallelization=False,
        run_single_outersplit_num=-1,
    )


# =============================================================================
# ResourceConfig Tests
# =============================================================================


class TestResourceConfig:
    """Tests for ResourceConfig."""

    def test_create_with_parallelization(self):
        """Test resource creation with outer parallelization."""
        config = ResourceConfig.create(
            num_outersplits=4,
            outer_parallelization=True,
            run_single_outersplit_num=-1,
            num_cpus=8,
        )
        assert config.num_cpus == 8
        assert config.num_workers == 4  # min(4 outersplits, 8 cpus)
        assert config.cpus_per_outersplit == 2  # 8 / 4

    def test_create_without_parallelization(self):
        """Test resource creation without outer parallelization."""
        config = ResourceConfig.create(
            num_outersplits=4,
            outer_parallelization=False,
            run_single_outersplit_num=-1,
            num_cpus=8,
        )
        assert config.num_cpus == 8
        assert config.num_workers == 4
        assert config.cpus_per_outersplit == 8  # All CPUs for sequential

    def test_create_more_outersplits_than_cpus(self):
        """Test when outersplits exceed available CPUs."""
        config = ResourceConfig.create(
            num_outersplits=16,
            outer_parallelization=True,
            run_single_outersplit_num=-1,
            num_cpus=4,
        )
        assert config.num_workers == 4  # Limited by CPUs
        assert config.cpus_per_outersplit == 1

    def test_frozen(self):
        """Test that ResourceConfig is immutable (attrs frozen)."""
        config = ResourceConfig(
            num_cpus=4,
            num_workers=2,
            cpus_per_outersplit=2,
            outer_parallelization=True,
            run_single_outersplit_num=-1,
            num_outersplits=4,
        )
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            config.num_cpus = 8

    def test_create_single_outersplit_gets_all_cpus(self):
        """Test that when running a single outersplit, it gets all CPUs.

        This tests the fix for the regression where single outersplits were
        getting limited CPUs based on the total number of outersplits.
        """
        # Simulate: 8 CPUs, 8 total outersplits, but running only outersplit 0
        config = ResourceConfig.create(
            num_outersplits=8,
            outer_parallelization=True,
            run_single_outersplit_num=0,
            num_cpus=8,
        )
        assert config.num_cpus == 8
        assert config.num_workers == 1  # Only 1 outersplit running
        assert config.cpus_per_outersplit == 8  # Gets all CPUs, not 8/8=1

    def test_create_rejects_zero_outersplits(self):
        """Test that zero outersplits raises ValueError."""
        with pytest.raises(ValueError, match="num_outersplits must be positive"):
            ResourceConfig.create(
                num_outersplits=0,
                outer_parallelization=True,
                run_single_outersplit_num=-1,
                num_cpus=8,
            )

    def test_create_rejects_negative_outersplits(self):
        """Test that negative outersplits raises ValueError."""
        with pytest.raises(ValueError, match="num_outersplits must be positive"):
            ResourceConfig.create(
                num_outersplits=-5,
                outer_parallelization=True,
                run_single_outersplit_num=-1,
                num_cpus=8,
            )

    def test_create_rejects_zero_cpus(self):
        """Test that zero CPUs raises ValueError."""
        with pytest.raises(ValueError, match="num_cpus must be positive"):
            ResourceConfig.create(
                num_outersplits=4,
                outer_parallelization=True,
                run_single_outersplit_num=-1,
                num_cpus=0,
            )

    def test_create_rejects_negative_cpus(self):
        """Test that negative CPUs raises ValueError."""
        with pytest.raises(ValueError, match="num_cpus must be positive"):
            ResourceConfig.create(
                num_outersplits=4,
                outer_parallelization=True,
                run_single_outersplit_num=-1,
                num_cpus=-2,
            )

    def test_create_rejects_invalid_single_outersplit_index(self):
        """Test that single outersplit index >= num_outersplits raises ValueError."""
        with pytest.raises(
            ValueError,
            match=r"run_single_outersplit_num \(5\) must be less than num_outersplits \(3\)",
        ):
            ResourceConfig.create(
                num_outersplits=3,
                outer_parallelization=True,
                run_single_outersplit_num=5,
                num_cpus=8,
            )

    def test_create_rejects_negative_single_outersplit_index(self):
        """Test that single outersplit index < -1 raises ValueError."""
        with pytest.raises(
            ValueError,
            match=r"run_single_outersplit_num must be -1 .* or a valid index",
        ):
            ResourceConfig.create(
                num_outersplits=3,
                outer_parallelization=True,
                run_single_outersplit_num=-5,
                num_cpus=8,
            )

    def test_create_accepts_valid_single_outersplit_index(self):
        """Test that valid single outersplit indices work correctly."""
        # Test last valid index
        config = ResourceConfig.create(
            num_outersplits=5,
            outer_parallelization=True,
            run_single_outersplit_num=4,
            num_cpus=8,
        )
        assert config.num_workers == 1
        assert config.run_single_outersplit_num == 4

        # Test first valid index
        config = ResourceConfig.create(
            num_outersplits=5,
            outer_parallelization=True,
            run_single_outersplit_num=0,
            num_cpus=8,
        )
        assert config.num_workers == 1
        assert config.run_single_outersplit_num == 0

    def test_str_representation(self):
        """Test string representation of ResourceConfig."""
        config = ResourceConfig.create(
            num_outersplits=4,
            outer_parallelization=True,
            run_single_outersplit_num=-1,
            num_cpus=8,
        )
        str_repr = str(config)

        # Verify all key information is in the string
        assert "Parallelization: True" in str_repr
        assert "Single outersplit: -1" in str_repr
        assert "Outersplits: 4" in str_repr
        assert "CPUs: 8" in str_repr
        assert "Workers: 4" in str_repr
        assert "CPUs/outersplit: 2" in str_repr

        # Verify "Preparing execution" is NOT in the string
        assert "Preparing execution" not in str_repr


# =============================================================================
# OctoManager Tests
# =============================================================================


class TestOctoManager:
    """Tests for OctoManager orchestration."""

    def test_run_outersplits_sequential(self, octo_manager):
        """Test run outersplits sequential."""
        with (
            patch("octopus.manager.core.shutdown_ray"),
            patch.object(WorkflowTaskRunner, "run") as mock_run,
        ):
            octo_manager.run_outersplits()
            assert mock_run.call_count == 1

    def test_run_outersplits_parallel(self, study, mock_workflow, mock_outersplit_data):
        """Test run outersplits with parallelization."""
        manager = OctoManager(
            outersplit_data=mock_outersplit_data,
            study_context=study,
            workflow=mock_workflow,
            outer_parallelization=True,
            run_single_outersplit_num=-1,
        )

        with (
            patch("octopus.manager.core.shutdown_ray"),
            patch("octopus.manager.execution.run_parallel_outer_ray", return_value=[True]) as mock_ray,
        ):
            manager.run_outersplits()
            mock_ray.assert_called_once()

    def test_run_single_outersplit(self, study, mock_workflow, mock_outersplit_data):
        """Test run single outersplit."""
        manager = OctoManager(
            outersplit_data=mock_outersplit_data,
            study_context=study,
            workflow=mock_workflow,
            outer_parallelization=False,
            run_single_outersplit_num=0,
        )

        with (
            patch("octopus.manager.core.shutdown_ray"),
            patch.object(WorkflowTaskRunner, "run") as mock_run,
        ):
            manager.run_outersplits()
            # Verify that run was called with outersplit_id and OuterSplit
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0]
            assert len(call_args) == 2
            assert call_args[0] == 0  # outersplit_id
            assert isinstance(call_args[1], OuterSplit)  # outersplit

    def test_no_outersplits_raises_error(self, study, mock_workflow):
        """Test that empty fold splits raises ValueError."""
        manager = OctoManager(
            outersplit_data={},
            study_context=study,
            workflow=mock_workflow,
            outer_parallelization=False,
            run_single_outersplit_num=-1,
        )
        with pytest.raises(ValueError, match="No outersplit data defined"):
            manager.run_outersplits()

    def test_ray_shutdown_on_error(self, octo_manager):
        """Test that Ray is shut down even if execution fails."""
        with (
            patch("octopus.manager.core.shutdown_ray") as mock_shutdown,
            patch.object(WorkflowTaskRunner, "run", side_effect=RuntimeError("Test error")),
        ):
            with pytest.raises(RuntimeError):
                octo_manager.run_outersplits()
            mock_shutdown.assert_called_once()

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
            outer_parallelization=True,
            run_single_outersplit_num=0,
        )

        with (
            patch("octopus.manager.core.shutdown_ray"),
            patch("octopus.manager.core.get_available_cpus", return_value=8),
            patch.object(WorkflowTaskRunner, "__init__", return_value=None) as mock_runner_init,
            patch.object(WorkflowTaskRunner, "run"),
        ):
            manager.run_outersplits()

            # Verify that WorkflowTaskRunner was initialized with cpus_per_outersplit that allocates all CPUs
            # to the single outersplit (not 8/8=1)
            call_args = mock_runner_init.call_args
            cpus_per_outersplit = call_args[1]["cpus_per_outersplit"]  # Now a keyword arg
            assert cpus_per_outersplit == 8  # All CPUs for that outersplit


# =============================================================================
# WorkflowTaskRunner Tests
# =============================================================================
