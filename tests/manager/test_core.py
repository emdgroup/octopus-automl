"""Test ResourceConfig and OctoManager from octopus.manager.core."""

from unittest.mock import Mock, patch

import attrs
import pytest
from upath import UPath

from octopus.experiment import OctoExperiment
from octopus.manager import OctoManager
from octopus.manager.core import ResourceConfig
from octopus.manager.workflow_runner import WorkflowTaskRunner

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_workflow():
    """Create mock workflow."""
    return [
        Mock(
            task_id=1,
            depends_on_task=0,
            module="test_module",
            description="Test",
            load_task=False,
        ),
        Mock(
            task_id=2,
            depends_on_task=1,
            module="test_module",
            description="Test",
            load_task=False,
        ),
    ]


@pytest.fixture
def mock_experiment():
    """Create mock experiment."""
    experiment = Mock(spec=OctoExperiment)
    experiment.experiment_id = "test_exp"
    experiment.path_study = UPath("/tmp/test_study")
    experiment.feature_cols = ["feature1", "feature2"]
    experiment.calculate_feature_groups = Mock(return_value=["group1"])
    return experiment


@pytest.fixture
def octo_manager(mock_workflow, mock_experiment):
    """Create octo manager."""
    return OctoManager(
        base_experiments=[mock_experiment],
        workflow=mock_workflow,
        outer_parallelization=False,
        run_single_experiment_num=-1,
        log_dir=mock_experiment.path_study,
    )


# =============================================================================
# ResourceConfig Tests
# =============================================================================


class TestResourceConfig:
    """Tests for ResourceConfig."""

    def test_create_with_parallelization(self):
        """Test resource creation with outer parallelization."""
        config = ResourceConfig.create(
            num_experiments=4,
            outer_parallelization=True,
            run_single_experiment_num=-1,
            num_cpus=8,
        )
        assert config.num_cpus == 8
        assert config.num_workers == 4  # min(4 experiments, 8 cpus)
        assert config.cpus_per_experiment == 2  # 8 / 4

    def test_create_without_parallelization(self):
        """Test resource creation without outer parallelization."""
        config = ResourceConfig.create(
            num_experiments=4,
            outer_parallelization=False,
            run_single_experiment_num=-1,
            num_cpus=8,
        )
        assert config.num_cpus == 8
        assert config.num_workers == 4
        assert config.cpus_per_experiment == 8  # All CPUs for sequential

    def test_create_more_experiments_than_cpus(self):
        """Test when experiments exceed available CPUs."""
        config = ResourceConfig.create(
            num_experiments=16,
            outer_parallelization=True,
            run_single_experiment_num=-1,
            num_cpus=4,
        )
        assert config.num_workers == 4  # Limited by CPUs
        assert config.cpus_per_experiment == 1

    def test_frozen(self):
        """Test that ResourceConfig is immutable (attrs frozen)."""
        config = ResourceConfig(
            num_cpus=4,
            num_workers=2,
            cpus_per_experiment=2,
            outer_parallelization=True,
            run_single_experiment_num=-1,
            num_experiments=4,
        )
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            config.num_cpus = 8

    def test_create_single_experiment_gets_all_cpus(self):
        """Test that when running a single experiment, it gets all CPUs.

        This tests the fix for the regression where single experiments were
        getting limited CPUs based on the total number of experiments.
        """
        # Simulate: 8 CPUs, 8 total experiments, but running only experiment 0
        config = ResourceConfig.create(
            num_experiments=8,
            outer_parallelization=True,
            run_single_experiment_num=0,
            num_cpus=8,
        )
        assert config.num_cpus == 8
        assert config.num_workers == 1  # Only 1 experiment running
        assert config.cpus_per_experiment == 8  # Gets all CPUs, not 8/8=1

    def test_create_rejects_zero_experiments(self):
        """Test that zero experiments raises ValueError."""
        with pytest.raises(ValueError, match="num_experiments must be positive"):
            ResourceConfig.create(
                num_experiments=0,
                outer_parallelization=True,
                run_single_experiment_num=-1,
                num_cpus=8,
            )

    def test_create_rejects_negative_experiments(self):
        """Test that negative experiments raises ValueError."""
        with pytest.raises(ValueError, match="num_experiments must be positive"):
            ResourceConfig.create(
                num_experiments=-5,
                outer_parallelization=True,
                run_single_experiment_num=-1,
                num_cpus=8,
            )

    def test_create_rejects_zero_cpus(self):
        """Test that zero CPUs raises ValueError."""
        with pytest.raises(ValueError, match="num_cpus must be positive"):
            ResourceConfig.create(
                num_experiments=4,
                outer_parallelization=True,
                run_single_experiment_num=-1,
                num_cpus=0,
            )

    def test_create_rejects_negative_cpus(self):
        """Test that negative CPUs raises ValueError."""
        with pytest.raises(ValueError, match="num_cpus must be positive"):
            ResourceConfig.create(
                num_experiments=4,
                outer_parallelization=True,
                run_single_experiment_num=-1,
                num_cpus=-2,
            )

    def test_create_rejects_invalid_single_experiment_index(self):
        """Test that single experiment index >= num_experiments raises ValueError."""
        with pytest.raises(
            ValueError,
            match=r"run_single_experiment_num \(5\) must be less than num_experiments \(3\)",
        ):
            ResourceConfig.create(
                num_experiments=3,
                outer_parallelization=True,
                run_single_experiment_num=5,
                num_cpus=8,
            )

    def test_create_rejects_negative_single_experiment_index(self):
        """Test that single experiment index < -1 raises ValueError."""
        with pytest.raises(
            ValueError,
            match=r"run_single_experiment_num must be -1 .* or a valid index",
        ):
            ResourceConfig.create(
                num_experiments=3,
                outer_parallelization=True,
                run_single_experiment_num=-5,
                num_cpus=8,
            )

    def test_create_accepts_valid_single_experiment_index(self):
        """Test that valid single experiment indices work correctly."""
        # Test last valid index
        config = ResourceConfig.create(
            num_experiments=5,
            outer_parallelization=True,
            run_single_experiment_num=4,
            num_cpus=8,
        )
        assert config.num_workers == 1
        assert config.run_single_experiment_num == 4

        # Test first valid index
        config = ResourceConfig.create(
            num_experiments=5,
            outer_parallelization=True,
            run_single_experiment_num=0,
            num_cpus=8,
        )
        assert config.num_workers == 1
        assert config.run_single_experiment_num == 0

    def test_str_representation(self):
        """Test string representation of ResourceConfig."""
        config = ResourceConfig.create(
            num_experiments=4,
            outer_parallelization=True,
            run_single_experiment_num=-1,
            num_cpus=8,
        )
        str_repr = str(config)

        # Verify all key information is in the string
        assert "Parallelization: True" in str_repr
        assert "Single exp: -1" in str_repr
        assert "Outer folds: 4" in str_repr
        assert "CPUs: 8" in str_repr
        assert "Workers: 4" in str_repr
        assert "CPUs/exp: 2" in str_repr

        # Verify "Preparing execution" is NOT in the string
        assert "Preparing execution" not in str_repr


# =============================================================================
# OctoManager Tests
# =============================================================================


class TestOctoManager:
    """Tests for OctoManager orchestration."""

    def test_run_outer_experiments_sequential(self, octo_manager):
        """Test run outer experiments sequential."""
        with (
            patch("octopus.manager.core.shutdown_ray"),
            patch.object(WorkflowTaskRunner, "run") as mock_run,
        ):
            octo_manager.run_outer_experiments()
            assert mock_run.call_count == 1

    def test_run_outer_experiments_parallel(self, octo_manager):
        """Test run outer experiments with parallelization."""
        octo_manager.outer_parallelization = True

        with (
            patch("octopus.manager.core.shutdown_ray"),
            patch("octopus.manager.execution.run_parallel_outer_ray", return_value=[True]) as mock_ray,
        ):
            octo_manager.run_outer_experiments()
            mock_ray.assert_called_once()

    def test_run_single_experiment(self, octo_manager):
        """Test run single experiment."""
        octo_manager.run_single_experiment_num = 0

        with (
            patch("octopus.manager.core.shutdown_ray"),
            patch.object(WorkflowTaskRunner, "run") as mock_run,
        ):
            octo_manager.run_outer_experiments()
            mock_run.assert_called_once_with(octo_manager.base_experiments[0])

    def test_no_experiments_raises_error(self, mock_workflow):
        """Test that empty experiments raises ValueError."""
        manager = OctoManager(
            base_experiments=[],
            workflow=mock_workflow,
            log_dir=UPath("/tmp/test"),
        )
        with pytest.raises(ValueError, match="No experiments defined"):
            manager.run_outer_experiments()

    def test_ray_shutdown_on_error(self, octo_manager):
        """Test that Ray is shut down even if execution fails."""
        with (
            patch("octopus.manager.core.shutdown_ray") as mock_shutdown,
            patch.object(WorkflowTaskRunner, "run", side_effect=RuntimeError("Test error")),
        ):
            with pytest.raises(RuntimeError):
                octo_manager.run_outer_experiments()
            mock_shutdown.assert_called_once()

    def test_single_experiment_resource_allocation(self, mock_workflow):
        """Test that single experiment gets all CPUs when run_single_experiment_num is set.

        This is a regression test: previously, when running a single experiment from
        a set of 8 experiments on 8 CPUs, the single experiment would only get 1 CPU
        instead of all 8.
        """
        # Create 8 mock experiments
        experiments = [Mock(spec=OctoExperiment, experiment_id=f"exp_{i}") for i in range(8)]

        manager = OctoManager(
            base_experiments=experiments,
            workflow=mock_workflow,
            outer_parallelization=True,
            run_single_experiment_num=0,  # Run only first experiment
            log_dir=UPath("/tmp/test"),
        )

        with (
            patch("octopus.manager.core.shutdown_ray"),
            patch("octopus.manager.core.get_available_cpus", return_value=8),
            patch.object(WorkflowTaskRunner, "__init__", return_value=None) as mock_runner_init,
            patch.object(WorkflowTaskRunner, "run"),
        ):
            manager.run_outer_experiments()

            # Verify that WorkflowTaskRunner was initialized with cpus_per_experiment that allocates all CPUs
            # to the single experiment (not 8/8=1)
            call_args = mock_runner_init.call_args
            cpus_per_experiment = call_args[0][1]  # Second positional arg is cpus_per_experiment
            assert cpus_per_experiment == 8  # All CPUs for that experiment
