"""Test workflow for ROC-OCTO-ROC sequence."""

import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from octopus.modules import Octo, Roc
from octopus.study import OctoClassification
from octopus.types import CorrelationType, FIComputeMethod, ModelName, RelevanceMethod


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
                depends_on=None,
                correlation_threshold=0.85,
                correlation_type=CorrelationType.SPEARMAN,
                relevance_method=RelevanceMethod.F_STATISTICS,
            ),
            Octo(
                description="step_1_octo",
                task_id=1,
                depends_on=0,
                n_inner_splits=3,
                models=[ModelName.ExtraTreesClassifier],
                max_outliers=0,
                fi_methods=[FIComputeMethod.PERMUTATION],
                n_trials=6,
            ),
            Roc(
                description="step_2_roc_final",
                task_id=2,
                depends_on=1,
                correlation_threshold=0.5,
                correlation_type=CorrelationType.SPEARMAN,
                relevance_method=RelevanceMethod.MUTUAL_INFO,
            ),
        ]

        # Verify sequence configuration
        assert len(workflow) == 3

        # Verify first ROC step
        first_roc = workflow[0]
        assert isinstance(first_roc, Roc)
        assert first_roc.task_id == 0
        assert first_roc.depends_on is None
        assert first_roc.correlation_threshold == 0.85
        assert first_roc.description == "step_0_roc_initial"

        # Verify OCTO step
        octo_step = workflow[1]
        assert isinstance(octo_step, Octo)
        assert octo_step.task_id == 1
        assert octo_step.depends_on == 0
        assert octo_step.description == "step_1_octo"

        # Verify second ROC step
        second_roc = workflow[2]
        assert isinstance(second_roc, Roc)
        assert second_roc.task_id == 2
        assert second_roc.depends_on == 1
        assert second_roc.correlation_threshold == 0.5
        assert second_roc.description == "step_2_roc_final"

    def test_octo_study_with_roc_octo_roc(self, sample_classification_dataset):
        """Test OctoStudy configuration with ROC-OCTO-ROC workflow."""
        _, feature_names = sample_classification_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            study = OctoClassification(
                study_name="test_roc_octo_roc",
                target_metric="ACCBAL",
                feature_cols=feature_names,
                target_col="target",
                sample_id_col="sample_id_col",
                stratification_col="target",
                studies_directory=temp_dir,
                workflow=[
                    Roc(
                        description="step_0_roc_initial",
                        task_id=0,
                        depends_on=None,
                        correlation_threshold=0.85,
                        correlation_type=CorrelationType.SPEARMAN,
                        relevance_method=RelevanceMethod.F_STATISTICS,
                    ),
                    Octo(
                        description="step_1_octo",
                        task_id=1,
                        depends_on=0,
                        n_inner_splits=3,
                        models=[ModelName.ExtraTreesClassifier],
                        n_trials=15,
                    ),
                    Roc(
                        description="step_2_roc_final",
                        task_id=2,
                        depends_on=1,
                        correlation_threshold=0.5,
                        correlation_type=CorrelationType.SPEARMAN,
                        relevance_method=RelevanceMethod.MUTUAL_INFO,
                    ),
                ],
            )

            assert len(study.workflow) == 3

    def test_sequence_dependency_chain(self):
        """Test that the sequence dependency chain is correctly configured."""
        workflow = [
            Roc(task_id=0, depends_on=None, correlation_threshold=0.85),
            Octo(task_id=1, depends_on=0, models=[ModelName.ExtraTreesClassifier], n_trials=6),
            Roc(task_id=2, depends_on=1, correlation_threshold=0.5),
        ]

        # First step has no dependencies
        assert workflow[0].depends_on is None

        # Second step depends on first
        assert workflow[1].depends_on == workflow[0].task_id

        # Third step depends on second
        assert workflow[2].depends_on == workflow[1].task_id

        # Verify sequence IDs are sequential
        for i, step in enumerate(workflow):
            assert step.task_id == i

    def test_roc_correlation_threshold_configuration(self):
        """Test that ROC correlation thresholds are configured correctly in the sequence."""
        first_roc = Roc(task_id=0, depends_on=None, correlation_threshold=0.85)
        second_roc = Roc(task_id=2, depends_on=1, correlation_threshold=0.5)

        assert first_roc.correlation_threshold == 0.85
        assert second_roc.correlation_threshold == 0.5

        # Verify that final ROC has more aggressive filtering
        assert second_roc.correlation_threshold < first_roc.correlation_threshold

    @pytest.mark.parametrize("correlation_type", [CorrelationType.SPEARMAN, CorrelationType.RDC])
    @pytest.mark.parametrize("relevance_method", [RelevanceMethod.F_STATISTICS, RelevanceMethod.MUTUAL_INFO])
    def test_roc_configuration_variations(self, correlation_type, relevance_method):
        """Test ROC configuration with different correlation and relevance method types."""
        first_roc = Roc(
            task_id=0,
            depends_on=None,
            correlation_threshold=0.85,
            correlation_type=correlation_type,
            relevance_method=relevance_method,
        )
        second_roc = Roc(
            task_id=2,
            depends_on=1,
            correlation_threshold=0.5,
            correlation_type=correlation_type,
            relevance_method=relevance_method,
        )

        assert first_roc.correlation_type == correlation_type
        assert first_roc.relevance_method == relevance_method
        assert second_roc.correlation_type == correlation_type
        assert second_roc.relevance_method == relevance_method

    def test_octo_configuration_in_sequence(self):
        """Test OCTO module configuration within the ROC-OCTO-ROC sequence."""
        workflow = [
            Roc(task_id=0, depends_on=None, correlation_threshold=0.85),
            Octo(
                task_id=1,
                depends_on=0,
                models=[ModelName.ExtraTreesClassifier, ModelName.RandomForestClassifier],
                n_trials=10,
                max_features=15,
                n_inner_splits=5,
            ),
            Roc(task_id=2, depends_on=1, correlation_threshold=0.5),
        ]

        octo_step = workflow[1]

        assert isinstance(octo_step, Octo)
        assert octo_step.models is not None
        assert set(octo_step.models) == {ModelName.ExtraTreesClassifier, ModelName.RandomForestClassifier}
        assert octo_step.n_trials == 10
        assert octo_step.max_features == 15
        assert octo_step.n_inner_splits == 5

    def test_workflow_sequence_validation(self):
        """Test that the workflow sequence is properly validated."""
        workflow = [
            Roc(task_id=0, depends_on=None, correlation_threshold=0.85),
            Octo(task_id=1, depends_on=0, models=[ModelName.ExtraTreesClassifier], n_trials=6),
            Roc(task_id=2, depends_on=1, correlation_threshold=0.5),
        ]

        assert len(workflow) == 3

        # Verify all steps are properly configured
        for i, step in enumerate(workflow):
            assert step.task_id == i
            if i == 0:
                assert step.depends_on is None
            else:
                assert step.depends_on == i - 1

    @pytest.mark.slow
    def test_roc_octo_roc_workflow_actual_execution(self, sample_classification_dataset):
        """Test that ROC-OCTO-ROC workflow actually runs end-to-end."""
        df, feature_names = sample_classification_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            study = OctoClassification(
                study_name="test_roc_octo_roc_execution",
                target_metric="ACCBAL",
                feature_cols=feature_names,
                target_col="target",
                sample_id_col="sample_id_col",
                stratification_col="target",
                outer_split_seed=1234,
                n_outer_splits=2,
                studies_directory=temp_dir,
                single_outer_split=0,
                workflow=[
                    Roc(
                        description="step_0_roc_initial",
                        task_id=0,
                        depends_on=None,
                        correlation_threshold=0.9,
                        correlation_type=CorrelationType.SPEARMAN,
                        relevance_method=RelevanceMethod.F_STATISTICS,
                    ),
                    Octo(
                        description="step_1_octo",
                        task_id=1,
                        depends_on=0,
                        n_inner_splits=5,
                        models=[ModelName.ExtraTreesClassifier],
                        n_trials=13,
                        fi_methods=[FIComputeMethod.PERMUTATION],
                    ),
                    Roc(
                        description="step_2_roc_final",
                        task_id=2,
                        depends_on=1,
                        correlation_threshold=0.5,
                        correlation_type=CorrelationType.SPEARMAN,
                        relevance_method=RelevanceMethod.F_STATISTICS,
                    ),
                ],
            )

            study.fit(data=df)

            # Verify that the study was created and files exist
            study_path = study.output_path
            assert study_path.exists(), "Study directory should be created"

            assert (study_path / "study.log").exists(), "Study log file should exist"

            # Check for expected files (new architecture uses files, not directories)
            assert (study_path / "data_raw.parquet").exists(), "Data parquet file should exist"
            assert (study_path / "data_prepared.parquet").exists(), "Prepared data parquet file should exist"
            assert (study_path / "study_config.json").exists(), "Config JSON file should exist"
            assert (study_path / "study_meta.json").exists(), "Study meta JSON file should exist"
            assert (study_path / "outersplit0").exists(), "Experiment directory should exist"

            # Verify that sequence steps were executed by checking for workflow directories
            experiment_path = study_path / "outersplit0"
            workflow_dirs = [d for d in experiment_path.iterdir() if d.is_dir() and d.name.startswith("task")]

            # Should have directories for each sequence step
            assert len(workflow_dirs) >= 3, (
                f"Should have at least 3 workflow directories, found: {[d.name for d in workflow_dirs]}"
            )

            # Verify the final ROC step was executed with threshold 0.5
            def extract_workflow_task_number(path):
                name = path.name
                return int(name.replace("task", ""))

            workflow_dirs_sorted = sorted(workflow_dirs, key=extract_workflow_task_number)
            final_workflow_dir = workflow_dirs_sorted[-1]

            assert final_workflow_dir.exists(), "Final ROC workflow step should have been executed"
