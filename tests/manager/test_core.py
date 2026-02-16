"""Test ResourceConfig, OctoManager, and WorkflowTaskRunner from octopus.manager."""

import json
from unittest.mock import Mock, patch

import attrs
import pandas as pd
import pytest
from upath import UPath

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
            depends_on=0,
            module="test_module",
            description="Test",
            load_task=False,
        ),
        Mock(
            task_id=2,
            depends_on=1,
            module="test_module",
            description="Test",
            load_task=False,
        ),
    ]


@pytest.fixture
def mock_outersplit_data():
    """Create mock fold splits."""
    return {
        0: {
            "train": pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4], "target": [0, 1]}),
            "test": pd.DataFrame({"feature1": [5], "feature2": [6], "target": [1]}),
        }
    }


@pytest.fixture
def mock_study(mock_workflow):
    """Create mock study with workflow and settings."""
    study = Mock()
    study.output_path = UPath("/tmp/test_study")
    study.log_dir = UPath("/tmp/test_study")
    study.workflow = mock_workflow
    study.outer_parallelization = False
    study.run_single_outersplit_num = -1
    study.prepared = Mock()
    study.prepared.feature_cols = ["feature1", "feature2"]
    return study


@pytest.fixture
def octo_manager(mock_study, mock_outersplit_data):
    """Create octo manager."""
    return OctoManager(
        study=mock_study,
        outersplit_data=mock_outersplit_data,
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

    def test_run_outersplits_parallel(self, octo_manager):
        """Test run outersplits with parallelization."""
        octo_manager.study.outer_parallelization = True

        with (
            patch("octopus.manager.core.shutdown_ray"),
            patch("octopus.manager.execution.run_parallel_outer_ray", return_value=[True]) as mock_ray,
        ):
            octo_manager.run_outersplits()
            mock_ray.assert_called_once()

    def test_run_single_outersplit(self, octo_manager):
        """Test run single outersplit."""
        octo_manager.study.run_single_outersplit_num = 0

        with (
            patch("octopus.manager.core.shutdown_ray"),
            patch.object(WorkflowTaskRunner, "run") as mock_run,
        ):
            octo_manager.run_outersplits()
            # Verify that run was called with outersplit_id, train_df, test_df
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0]
            assert call_args[0] == 0  # outersplit_id
            assert isinstance(call_args[1], pd.DataFrame)  # train df
            assert isinstance(call_args[2], pd.DataFrame)  # test df

    def test_no_outersplits_raises_error(self, mock_study):
        """Test that empty fold splits raises ValueError."""
        manager = OctoManager(
            study=mock_study,
            outersplit_data={},
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

    def test_single_outersplit_resource_allocation(self, mock_study):
        """Test that single outersplit gets all CPUs when run_single_outersplit_num is set.

        This is a regression test: previously, when running a single outersplit from
        a set of 8 outersplits on 8 CPUs, the single outersplit would only get 1 CPU
        instead of all 8.
        """
        # Create 8 mock fold splits
        outersplit_data = {
            i: {
                "train": pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4], "target": [0, 1]}),
                "test": pd.DataFrame({"feature1": [5], "feature2": [6], "target": [1]}),
            }
            for i in range(8)
        }

        mock_study.outer_parallelization = True
        mock_study.run_single_outersplit_num = 0  # Run only first outersplit

        manager = OctoManager(
            study=mock_study,
            outersplit_data=outersplit_data,
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
            cpus_per_outersplit = call_args[0][1]  # Second positional arg is cpus_per_outersplit
            assert cpus_per_outersplit == 8  # All CPUs for that outersplit


# =============================================================================
# WorkflowTaskRunner Tests
# =============================================================================


class TestLoadTaskResults:
    """Tests for WorkflowTaskRunner._load_task_results()."""

    @pytest.fixture
    def runner(self):
        """Create a WorkflowTaskRunner with a mock study."""
        study = Mock()
        study.output_path = UPath("/tmp/test_study")
        return WorkflowTaskRunner(study=study, cpus_per_outersplit=1)

    def test_no_parquet_files(self, runner, tmp_path):
        """Test that missing parquet files returns dict of empty DataFrames."""
        output_dir = UPath(tmp_path / "task0")
        output_dir.mkdir()

        result = runner._load_task_results(output_dir)
        assert "scores" in result
        assert "predictions" in result
        assert "feature_importances" in result
        assert result["scores"].empty
        assert result["predictions"].empty
        assert result["feature_importances"].empty

    def test_load_results_with_scores(self, runner, tmp_path):
        """Test loading results with scores parquet."""
        output_dir = UPath(tmp_path / "task0")
        output_dir.mkdir()

        scores_df = pd.DataFrame(
            {
                "result_type": ["best"],
                "module": ["octo"],
                "metric": ["MAE"],
                "partition": ["dev"],
                "aggregation": ["avg"],
                "fold": [None],
                "value": [0.85],
            }
        )
        scores_df.to_parquet(output_dir / "scores.parquet", engine="pyarrow")

        result = runner._load_task_results(output_dir)

        assert not result["scores"].empty
        assert result["scores"].iloc[0]["result_type"] == "best"
        assert result["scores"].iloc[0]["module"] == "octo"
        assert result["predictions"].empty
        assert result["feature_importances"].empty

    def test_load_results_with_fi(self, runner, tmp_path):
        """Test loading results with feature importance parquet."""
        output_dir = UPath(tmp_path / "task0")
        output_dir.mkdir()

        fi_df = pd.DataFrame(
            {
                "feature": ["f1", "f2"],
                "importance": [0.7, 0.3],
                "fi_method": ["internal"] * 2,
                "fi_dataset": ["train"] * 2,
                "training_id": ["rfe"] * 2,
                "result_type": ["best"] * 2,
                "module": ["rfe"] * 2,
            }
        )
        fi_df.to_parquet(output_dir / "feature_importances.parquet", engine="pyarrow")

        result = runner._load_task_results(output_dir)

        assert not result["feature_importances"].empty
        assert len(result["feature_importances"]) == 2
        assert result["scores"].empty

    def test_load_all_result_types(self, runner, tmp_path):
        """Test loading results with all three parquet files."""
        output_dir = UPath(tmp_path / "task0")
        output_dir.mkdir()

        scores_df = pd.DataFrame(
            {
                "result_type": ["best"],
                "module": ["octo"],
                "metric": ["MAE"],
                "partition": ["dev"],
                "aggregation": ["avg"],
                "fold": [None],
                "value": [0.9],
            }
        )
        scores_df.to_parquet(output_dir / "scores.parquet", engine="pyarrow")

        predictions_df = pd.DataFrame(
            {
                "result_type": ["best"],
                "module": ["octo"],
                "row_id": [1],
                "prediction": [0.5],
                "target": [0.6],
                "partition": ["dev"],
            }
        )
        predictions_df.to_parquet(output_dir / "predictions.parquet", engine="pyarrow")

        fi_df = pd.DataFrame(
            {
                "feature": ["f1"],
                "importance": [0.7],
                "fi_method": ["internal"],
                "fi_dataset": ["train"],
                "training_id": ["t0"],
                "result_type": ["best"],
                "module": ["octo"],
            }
        )
        fi_df.to_parquet(output_dir / "feature_importances.parquet", engine="pyarrow")

        result = runner._load_task_results(output_dir)

        assert not result["scores"].empty
        assert not result["predictions"].empty
        assert not result["feature_importances"].empty


class TestLoadTask:
    """Tests for WorkflowTaskRunner._load_task()."""

    @pytest.fixture
    def runner(self, tmp_path):
        """Create a WorkflowTaskRunner with output_path pointing to tmp_path."""
        study = Mock()
        study.output_path = UPath(tmp_path)
        return WorkflowTaskRunner(study=study, cpus_per_outersplit=1)

    def test_load_task_reads_selected_features_from_json(self, runner, tmp_path):
        """Test that _load_task reads selected_features from selected_features.json."""
        task_dir = tmp_path / "outersplit0" / "task0"
        task_dir.mkdir(parents=True)

        # Write selected_features.json (new format)
        with open(task_dir / "selected_features.json", "w") as f:
            json.dump(["f1", "f2", "f3"], f)

        task = Mock(task_id=0)
        selected_features, results = runner._load_task(outersplit_id=0, task=task)

        assert selected_features == ["f1", "f2", "f3"]
        assert isinstance(results, dict)
        assert "scores" in results
        assert "predictions" in results
        assert "feature_importances" in results

    def test_load_task_missing_selected_features_raises(self, runner, tmp_path):
        """Test that _load_task raises FileNotFoundError if selected_features.json missing."""
        task_dir = tmp_path / "outersplit0" / "task0"
        task_dir.mkdir(parents=True)

        task = Mock(task_id=0)
        with pytest.raises(FileNotFoundError, match=r"selected_features\.json not found"):
            runner._load_task(outersplit_id=0, task=task)

    def test_load_task_with_results(self, runner, tmp_path):
        """Test that _load_task loads results from parquet files."""
        task_dir = tmp_path / "outersplit0" / "task0"
        task_dir.mkdir(parents=True)

        # Write selected_features.json
        with open(task_dir / "selected_features.json", "w") as f:
            json.dump(["f1", "f2"], f)

        # Write feature importances parquet
        fi_df = pd.DataFrame(
            {
                "feature": ["f1", "f2"],
                "importance": [0.6, 0.4],
                "fi_method": ["internal"] * 2,
                "fi_dataset": ["train"] * 2,
                "training_id": ["rfe"] * 2,
                "result_type": ["best"] * 2,
                "module": ["rfe"] * 2,
            }
        )
        fi_df.to_parquet(task_dir / "feature_importances.parquet", engine="pyarrow")

        task = Mock(task_id=0)
        selected_features, results = runner._load_task(outersplit_id=0, task=task)

        assert selected_features == ["f1", "f2"]
        assert not results["feature_importances"].empty
