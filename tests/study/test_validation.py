"""Test OctoClassification workflow validation."""

import tempfile

import pytest

from octopus.modules import Octo
from octopus.modules.mrmr.module import Mrmr
from octopus.study import OctoClassification


@pytest.fixture
def octo_task():
    """Create fixture for Octo task."""
    return Octo(
        task_id=0,
        depends_on_task=-1,
        description="step_1",
        models=["RandomForestRegressor", "XGBRegressor"],
    )


@pytest.fixture
def mrmr_task():
    """Create fixture for Mrmr task."""
    return Mrmr(task_id=1, depends_on_task=0, description="step2_mrmr")


@pytest.fixture
def base_study_kwargs():
    """Base kwargs for creating OctoClassification instances."""
    return {
        "name": "test",
        "target_metric": "AUCROC",
        "feature_columns": ["f1"],
        "target": "target",
        "sample_id": "id",
    }


class TestWorkflowValidation:
    """Test workflow validation."""

    @pytest.mark.parametrize(
        "workflow_tasks,expected_exception",
        [
            (["octo_task"], None),
            (["octo_task", "mrmr_task"], None),
            ([], ValueError),
            (["octo_task", "invalid_item"], TypeError),
            (["octo_task", "mrmr_task", "string"], TypeError),
            (None, TypeError),
        ],
    )
    def test_workflow_initialization(self, request, base_study_kwargs, workflow_tasks, expected_exception):
        """Test workflow initialization with various inputs."""
        if workflow_tasks is None:
            test_workflow = None
        else:
            test_workflow = []
            for name in workflow_tasks:
                if name in request._fixturemanager._arg2fixturedefs:
                    test_workflow.append(request.getfixturevalue(name))
                else:
                    test_workflow.append(name)

        with tempfile.TemporaryDirectory() as temp_dir:
            if expected_exception is None:
                study = OctoClassification(**base_study_kwargs, path=temp_dir, workflow=test_workflow)
                assert study.workflow == test_workflow
            else:
                with pytest.raises(expected_exception):
                    OctoClassification(**base_study_kwargs, path=temp_dir, workflow=test_workflow)

    def test_workflow_first_task_not_zero(self, base_study_kwargs):
        """Test that workflow validation fails when first task doesn't have task_id=0."""
        workflow = [Octo(task_id=1, depends_on_task=-1)]
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            pytest.raises(ValueError, match="The first task must have 'task_id=0'"),
        ):
            OctoClassification(**base_study_kwargs, path=temp_dir, workflow=workflow)

    def test_workflow_non_increasing_task_ids(self, base_study_kwargs):
        """Test that workflow validation fails when task_ids are not in increasing order."""
        workflow = [
            Octo(task_id=0, depends_on_task=-1),
            Mrmr(task_id=2, depends_on_task=0),
            Octo(task_id=1, depends_on_task=0),
        ]
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            pytest.raises(ValueError, match="not greater than the previous 'task_id'"),
        ):
            OctoClassification(**base_study_kwargs, path=temp_dir, workflow=workflow)

    def test_workflow_missing_task_ids(self, base_study_kwargs):
        """Test that workflow validation fails when task_ids have gaps."""
        workflow = [
            Octo(task_id=0, depends_on_task=-1),
            Mrmr(task_id=2, depends_on_task=0),
        ]
        with tempfile.TemporaryDirectory() as temp_dir, pytest.raises(ValueError, match="Missing task_ids"):
            OctoClassification(**base_study_kwargs, path=temp_dir, workflow=workflow)

    def test_workflow_duplicate_task_ids(self, base_study_kwargs):
        """Test that workflow validation fails when there are duplicate task_ids."""
        workflow = [
            Octo(task_id=0, depends_on_task=-1),
            Mrmr(task_id=1, depends_on_task=0),
            Octo(task_id=1, depends_on_task=0),
        ]
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            pytest.raises(ValueError, match="not greater than the previous 'task_id'"),
        ):
            OctoClassification(**base_study_kwargs, path=temp_dir, workflow=workflow)

    def test_workflow_depends_on_nonexistent_task(self, base_study_kwargs):
        """Test that workflow validation fails when depends_on_task references non-existent task."""
        workflow = [
            Octo(task_id=0, depends_on_task=-1),
            Mrmr(task_id=1, depends_on_task=5),
        ]
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            pytest.raises(ValueError, match="does not correspond to any 'task_id'"),
        ):
            OctoClassification(**base_study_kwargs, path=temp_dir, workflow=workflow)

    def test_workflow_depends_on_later_task(self, base_study_kwargs):
        """Test that workflow validation fails when depends_on_task references a later task."""
        workflow = [
            Octo(task_id=0, depends_on_task=-1),
            Mrmr(task_id=1, depends_on_task=2),
            Octo(task_id=2, depends_on_task=0),
        ]
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            pytest.raises(ValueError, match="refers to an item that comes after it"),
        ):
            OctoClassification(**base_study_kwargs, path=temp_dir, workflow=workflow)

    def test_workflow_depends_on_minus_one_after_positive(self, base_study_kwargs):
        """Test that tasks with depends_on_task=-1 must be at the start."""
        workflow = [
            Octo(task_id=0, depends_on_task=-1),
            Mrmr(task_id=1, depends_on_task=0),
            Octo(task_id=2, depends_on_task=-1),
        ]
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            pytest.raises(ValueError, match="appears after items with 'depends_on_task>=0'"),
        ):
            OctoClassification(**base_study_kwargs, path=temp_dir, workflow=workflow)

    def test_valid_multi_task_workflow(self, base_study_kwargs):
        """Test a valid multi-task workflow."""
        workflow = [
            Octo(task_id=0, depends_on_task=-1),
            Mrmr(task_id=1, depends_on_task=0),
            Octo(task_id=2, depends_on_task=1),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            study = OctoClassification(**base_study_kwargs, path=temp_dir, workflow=workflow)
            assert len(study.workflow) == 3
            assert study.workflow[0].task_id == 0
            assert study.workflow[1].task_id == 1
            assert study.workflow[2].task_id == 2
