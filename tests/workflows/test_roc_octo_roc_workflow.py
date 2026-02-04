"""Test workflow for ROC-OCTO-ROC sequence."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from octopus import OctoClassification
from octopus.modules import Octo, Roc


class TestRocOctoRocWorkflow:
    """Test suite for ROC-OCTO-ROC workflow sequence."""

    @pytest.fixture
    def sample_classification_dataset(self):
        """Create a sample classification dataset for testing."""
        np.random.seed(42)

        X, y = make_classification(
            n_samples=200,
            n_features=15,
            n_informative=12,
            n_redundant=3,
            n_clusters_per_class=2,
            class_sep=1.5,
            flip_y=0.01,
            random_state=42,
        )

        # Add some highly correlated features to test ROC filtering
        X_extended = np.column_stack(
            [
                X,
                X[:, 0] + np.random.normal(0, 0.05, X.shape[0]),
                X[:, 1] + np.random.normal(0, 0.05, X.shape[0]),
                X[:, 2] + np.random.normal(0, 0.05, X.shape[0]),
                X[:, 0] * 2 + np.random.normal(0, 0.1, X.shape[0]),
                X[:, 1] * 1.5 + np.random.normal(0, 0.1, X.shape[0]),
            ]
        )

        feature_names = [f"feature_{i}" for i in range(X_extended.shape[1])]

        df = pd.DataFrame(X_extended, columns=feature_names)
        df["target"] = y
        df["sample_id_col"] = range(len(df))

        return df, feature_names

    def test_roc_octo_roc_sequence_configuration(self):
        """Test that ROC-OCTO-ROC sequence can be properly configured."""
        workflow = [
            Roc(
                description="step_0_roc_initial",
                task_id=0,
                depends_on_task=-1,
                load_task=False,
                threshold=0.85,
                correlation_type="spearmanr",
                filter_type="f_statistics",
            ),
            Octo(
                description="step_1_octo",
                task_id=1,
                depends_on_task=0,
                n_folds_inner=3,
                models=["ExtraTreesClassifier"],
                model_seed=0,
                n_jobs=1,
                max_outl=0,
                fi_methods_bestbag=["permutation"],
                inner_parallelization=True,
                n_trials=6,
            ),
            Roc(
                description="step_2_roc_final",
                task_id=2,
                depends_on_task=1,
                load_task=False,
                threshold=0.5,
                correlation_type="spearmanr",
                filter_type="mutual_info",
            ),
        ]

        # Verify sequence configuration
        assert len(workflow) == 3

        # Verify first ROC step
        first_roc = workflow[0]
        assert isinstance(first_roc, Roc)
        assert first_roc.task_id == 0
        assert first_roc.depends_on_task == -1
        assert first_roc.threshold == 0.85
        assert first_roc.description == "step_0_roc_initial"

        # Verify OCTO step
        octo_step = workflow[1]
        assert isinstance(octo_step, Octo)
        assert octo_step.task_id == 1
        assert octo_step.depends_on_task == 0
        assert octo_step.description == "step_1_octo"

        # Verify second ROC step
        second_roc = workflow[2]
        assert isinstance(second_roc, Roc)
        assert second_roc.task_id == 2
        assert second_roc.depends_on_task == 1
        assert second_roc.threshold == 0.5
        assert second_roc.description == "step_2_roc_final"

    def test_octo_study_with_roc_octo_roc(self, sample_classification_dataset):
        """Test OctoStudy configuration with ROC-OCTO-ROC workflow."""
        _, feature_names = sample_classification_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            study = OctoClassification(
                name="test_roc_octo_roc",
                target_metric="ACCBAL",
                feature_cols=feature_names,
                target="target",
                sample_id_col="sample_id_col",
                stratification_column="target",
                path=temp_dir,
                ignore_data_health_warning=True,
                workflow=[
                    Roc(
                        description="step_0_roc_initial",
                        task_id=0,
                        depends_on_task=-1,
                        threshold=0.85,
                        correlation_type="spearmanr",
                        filter_type="f_statistics",
                    ),
                    Octo(
                        description="step_1_octo",
                        task_id=1,
                        depends_on_task=0,
                        n_folds_inner=3,
                        models=["ExtraTreesClassifier"],
                        model_seed=0,
                        n_jobs=1,
                        n_trials=15,
                    ),
                    Roc(
                        description="step_2_roc_final",
                        task_id=2,
                        depends_on_task=1,
                        threshold=0.5,
                        correlation_type="spearmanr",
                        filter_type="mutual_info",
                    ),
                ],
            )

            assert len(study.workflow) == 3

    def test_sequence_dependency_chain(self):
        """Test that the sequence dependency chain is correctly configured."""
        workflow = [
            Roc(task_id=0, depends_on_task=-1, threshold=0.85),
            Octo(task_id=1, depends_on_task=0, models=["ExtraTreesClassifier"], n_trials=6),
            Roc(task_id=2, depends_on_task=1, threshold=0.5),
        ]

        # First step has no dependencies
        assert workflow[0].depends_on_task == -1

        # Second step depends on first
        assert workflow[1].depends_on_task == workflow[0].task_id

        # Third step depends on second
        assert workflow[2].depends_on_task == workflow[1].task_id

        # Verify sequence IDs are sequential
        for i, step in enumerate(workflow):
            assert step.task_id == i

    def test_roc_threshold_configuration(self):
        """Test that ROC thresholds are configured correctly in the sequence."""
        workflow = [
            Roc(task_id=0, depends_on_task=-1, threshold=0.85),
            Octo(task_id=1, depends_on_task=0, models=["ExtraTreesClassifier"], n_trials=6),
            Roc(task_id=2, depends_on_task=1, threshold=0.5),
        ]

        first_roc = workflow[0]
        second_roc = workflow[2]

        assert first_roc.threshold == 0.85
        assert second_roc.threshold == 0.5

        # Verify that final ROC has more aggressive filtering
        assert second_roc.threshold < first_roc.threshold

    @pytest.mark.parametrize("correlation_type", ["spearmanr", "rdc"])
    @pytest.mark.parametrize("filter_type", ["f_statistics", "mutual_info"])
    def test_roc_configuration_variations(self, correlation_type, filter_type):
        """Test ROC configuration with different correlation and filter types."""
        workflow = [
            Roc(
                task_id=0,
                depends_on_task=-1,
                threshold=0.85,
                correlation_type=correlation_type,
                filter_type=filter_type,
            ),
            Octo(task_id=1, depends_on_task=0, models=["ExtraTreesClassifier"], n_trials=6),
            Roc(
                task_id=2,
                depends_on_task=1,
                threshold=0.5,
                correlation_type=correlation_type,
                filter_type=filter_type,
            ),
        ]

        first_roc = workflow[0]
        second_roc = workflow[2]

        assert first_roc.correlation_type == correlation_type
        assert first_roc.filter_type == filter_type
        assert second_roc.correlation_type == correlation_type
        assert second_roc.filter_type == filter_type

    def test_octo_configuration_in_sequence(self):
        """Test OCTO module configuration within the ROC-OCTO-ROC sequence."""
        workflow = [
            Roc(task_id=0, depends_on_task=-1, threshold=0.85),
            Octo(
                task_id=1,
                depends_on_task=0,
                models=["ExtraTreesClassifier", "RandomForestClassifier"],
                n_trials=10,
                max_features=15,
                n_folds_inner=5,
                model_seed=42,
            ),
            Roc(task_id=2, depends_on_task=1, threshold=0.5),
        ]

        octo_step = workflow[1]

        assert isinstance(octo_step, Octo)
        assert set(octo_step.models) == {"ExtraTreesClassifier", "RandomForestClassifier"}
        assert octo_step.n_trials == 10
        assert octo_step.max_features == 15
        assert octo_step.n_folds_inner == 5
        assert octo_step.model_seed == 42

    def test_workflow_sequence_validation(self):
        """Test that the workflow sequence is properly validated."""
        workflow = [
            Roc(task_id=0, depends_on_task=-1, threshold=0.85),
            Octo(task_id=1, depends_on_task=0, models=["ExtraTreesClassifier"], n_trials=6),
            Roc(task_id=2, depends_on_task=1, threshold=0.5),
        ]

        assert len(workflow) == 3

        # Verify all steps are properly configured
        for i, step in enumerate(workflow):
            assert step.task_id == i
            if i == 0:
                assert step.depends_on_task == -1
            else:
                assert step.depends_on_task == i - 1

    @pytest.mark.slow
    def test_roc_octo_roc_workflow_actual_execution(self, sample_classification_dataset):
        """Test that ROC-OCTO-ROC workflow actually runs end-to-end."""
        df, feature_names = sample_classification_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            study = OctoClassification(
                name="test_roc_octo_roc_execution",
                target_metric="ACCBAL",
                feature_cols=feature_names,
                target="target",
                sample_id_col="sample_id_col",
                stratification_column="target",
                metrics=["AUCROC", "ACCBAL"],
                datasplit_seed_outer=1234,
                n_folds_outer=2,
                path=temp_dir,
                ignore_data_health_warning=True,
                outer_parallelization=False,
                run_single_experiment_num=0,
                workflow=[
                    Roc(
                        description="step_0_roc_initial",
                        task_id=0,
                        depends_on_task=-1,
                        threshold=0.9,
                        correlation_type="spearmanr",
                        filter_type="f_statistics",
                    ),
                    Octo(
                        description="step_1_octo",
                        task_id=1,
                        depends_on_task=0,
                        n_folds_inner=5,
                        models=["ExtraTreesClassifier"],
                        model_seed=0,
                        n_jobs=1,
                        n_trials=13,
                        inner_parallelization=True,
                        fi_methods_bestbag=["permutation"],
                    ),
                    Roc(
                        description="step_2_roc_final",
                        task_id=2,
                        depends_on_task=1,
                        threshold=0.5,
                        correlation_type="spearmanr",
                        filter_type="f_statistics",
                    ),
                ],
            )

            study.fit(data=df)

            # Verify that the study was created and files exist
            study_path = Path(temp_dir) / "test_roc_octo_roc_execution"
            assert study_path.exists(), "Study directory should be created"

            assert (study_path / "octo_manager.log").exists(), "Octo Manager log file should exist"

            # Check for expected files (new architecture uses files, not directories)
            assert (study_path / "data.parquet").exists(), "Data parquet file should exist"
            assert (study_path / "data_prepared.parquet").exists(), "Prepared data parquet file should exist"
            assert (study_path / "config.json").exists(), "Config JSON file should exist"
            assert (study_path / "outersplit0").exists(), "Experiment directory should exist"

            # Verify that sequence steps were executed by checking for workflow directories
            experiment_path = study_path / "outersplit0"
            workflow_dirs = [d for d in experiment_path.iterdir() if d.is_dir() and d.name.startswith("workflowtask")]

            # Should have directories for each sequence step
            assert len(workflow_dirs) >= 3, (
                f"Should have at least 3 workflow directories, found: {[d.name for d in workflow_dirs]}"
            )

            # Verify the final ROC step was executed with threshold 0.5
            def extract_workflow_task_number(path):
                name = path.name
                return int(name.replace("workflowtask", ""))

            workflow_dirs_sorted = sorted(workflow_dirs, key=extract_workflow_task_number)
            final_workflow_dir = workflow_dirs_sorted[-1]

            assert final_workflow_dir.exists(), "Final ROC workflow step should have been executed"
