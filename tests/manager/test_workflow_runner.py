"""Test WorkflowTaskRunner from octopus.manager.workflow_runner."""

from unittest.mock import Mock, patch

import pytest
from upath import UPath
from upath.implementations.local import PosixUPath

from octopus.experiment import OctoExperiment
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


@pytest.fixture(autouse=True)
def mock_ray_initialized():
    """Mock ray.is_initialized() to return True for all tests.

    WorkflowTaskRunner expects Ray to be initialized by OctoManager.
    Since these are unit tests of internal methods, we mock the check.
    """
    with patch("ray.is_initialized", return_value=True):
        yield


@pytest.fixture
def cpus_per_experiment():
    """CPUs allocated per experiment for testing."""
    return 2


# =============================================================================
# WorkflowTaskRunner Tests
# =============================================================================


class TestWorkflowTaskRunner:
    """Tests for WorkflowTaskRunner."""

    def test_create_experiment(self, mock_workflow, mock_experiment, cpus_per_experiment):
        """Test experiment creation from base experiment."""
        runner = WorkflowTaskRunner(mock_workflow, cpus_per_experiment, UPath("/tmp/test"))
        task = mock_workflow[0]

        with patch("octopus.manager.workflow_runner.copy.deepcopy", return_value=mock_experiment):
            experiment = runner._create_experiment(mock_experiment, task)

        assert experiment.ml_module == task.module
        assert experiment.task_id == task.task_id
        assert experiment.num_assigned_cpus == cpus_per_experiment

    def test_load_experiment(self, mock_workflow, mock_experiment, cpus_per_experiment):
        """Test validating existing experiment."""
        runner = WorkflowTaskRunner(mock_workflow, cpus_per_experiment, UPath("/tmp/test"))
        task = Mock(task_id=3)

        with (
            patch.object(PosixUPath, "exists", return_value=True),
            patch.object(OctoExperiment, "from_pickle", return_value=mock_experiment) as mock_from_pickle,
        ):
            result = runner._load_experiment(mock_experiment, task)
            # Should validate the file can be loaded
            mock_from_pickle.assert_called_once()
            # Should return None since it's just validation
            assert result is None

    def test_load_experiment_not_found(self, mock_workflow, mock_experiment, cpus_per_experiment):
        """Test loading non-existent experiment raises error."""
        runner = WorkflowTaskRunner(mock_workflow, cpus_per_experiment, UPath("/tmp/test"))
        task = Mock(task_id=3)

        with (
            patch.object(UPath, "exists", return_value=False),
            pytest.raises(FileNotFoundError),
        ):
            runner._load_experiment(mock_experiment, task)

    def test_apply_dependencies(self, mock_workflow, mock_experiment, cpus_per_experiment):
        """Test applying dependencies from previous task."""
        runner = WorkflowTaskRunner(mock_workflow, cpus_per_experiment, UPath("/tmp/test"))
        input_experiment = Mock(spec=OctoExperiment)
        input_experiment.selected_features = ["new_feature"]
        input_experiment.results = {"key": "value"}

        mock_experiment.depends_on_task = 1
        exp_path_dict = {1: UPath("/tmp/input_exp.pkl")}

        with (
            patch.object(PosixUPath, "exists", return_value=True),
            patch.object(OctoExperiment, "from_pickle", return_value=input_experiment),
        ):
            runner._apply_dependencies(mock_experiment, exp_path_dict)

        assert mock_experiment.feature_cols == ["new_feature"]
        assert mock_experiment.prior_results == {"key": "value"}
