"""Test workflow for Octopus intro classification example."""

import tempfile

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from octopus.modules import Octo
from octopus.study import OctoClassification
from octopus.types import FIComputeMethod, ModelName


@pytest.mark.windows
class TestOctoIntroClassification:
    """Test suite for Octopus intro classification workflow."""

    @pytest.fixture
    def breast_cancer_dataset(self):
        """Create synthetic binary classification dataset for testing (faster than breast cancer dataset)."""
        # Create synthetic binary classification dataset with reduced size for faster testing
        X, y = make_classification(
            n_samples=30,
            n_features=5,
            n_informative=3,
            n_redundant=2,
            n_classes=2,
            random_state=42,
        )

        # Create DataFrame similar to breast cancer dataset structure
        feature_names = [f"feature_{i}" for i in range(5)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        df = df.reset_index()

        return df, feature_names

    def test_breast_cancer_dataset_loading(self, breast_cancer_dataset):
        """Test that the breast cancer dataset is loaded correctly."""
        df, features = breast_cancer_dataset

        assert isinstance(df, pd.DataFrame)
        assert "target" in df.columns
        assert "index" in df.columns
        assert len(features) == 5
        assert df.shape[0] == 30

        unique_targets = df["target"].unique()
        assert len(unique_targets) == 2
        assert set(unique_targets) == {0, 1}

        for feature in features:
            assert feature in df.columns

        assert not df[features].isnull().any().any()
        assert not df["target"].isnull().any()

    def test_octo_study_configuration(self, breast_cancer_dataset):
        """Test OctoStudy configuration for breast cancer dataset."""
        _, features = breast_cancer_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            study = OctoClassification(
                study_name="test_classification",
                target_metric="ACCBAL",
                feature_cols=features,
                target_col="target",
                sample_id_col="index",
                stratification_col="target",
                study_path=temp_dir,
            )

            assert study.target_col == "target"
            assert len(study.feature_cols) == 5
            assert study.sample_id_col == "index"
            assert study.stratification_col == "target"

    def test_octo_task_configuration(self):
        """Test that Octo task can be properly configured."""
        octo_task = Octo(
            description="step_1_octo",
            task_id=0,
            depends_on=None,
            n_inner_splits=3,
            models=[ModelName.ExtraTreesClassifier, ModelName.RandomForestClassifier],
            fi_methods=[FIComputeMethod.PERMUTATION],
            n_startup_trials=5,
            n_trials=6,
            max_features=5,
            ensemble_selection=True,
        )

        assert isinstance(octo_task, Octo)
        assert octo_task.task_id == 0
        assert octo_task.depends_on is None
        assert octo_task.description == "step_1_octo"
        assert octo_task.n_inner_splits == 3
        assert octo_task.models is not None
        assert set(octo_task.models) == {ModelName.ExtraTreesClassifier, ModelName.RandomForestClassifier}

    @pytest.mark.parametrize("model", [ModelName.ExtraTreesClassifier, ModelName.RandomForestClassifier])
    def test_single_model_configuration(self, model):
        """Test configuration with different single models."""
        octo_task = Octo(
            description="step_1_octo",
            task_id=0,
            depends_on=None,
            models=[model],
            n_trials=3,
            n_inner_splits=3,
        )

        assert octo_task.models == [model]

    def test_multiple_models_configuration(self):
        """Test configuration with multiple models."""
        models = [ModelName.ExtraTreesClassifier, ModelName.RandomForestClassifier]
        octo_task = Octo(
            description="step_1_octo",
            task_id=0,
            depends_on=None,
            models=models,
            n_trials=5,
            n_inner_splits=3,
        )
        assert octo_task.models is not None
        assert set(octo_task.models) == set(models)

    def test_feature_importance_configuration(self):
        """Test feature importance method configuration."""
        fi_methods = [FIComputeMethod.PERMUTATION]
        octo_task = Octo(
            description="step_1_octo",
            task_id=0,
            depends_on=None,
            models=[ModelName.ExtraTreesClassifier],
            fi_methods=fi_methods,
            n_trials=3,
        )

        assert octo_task.fi_methods == fi_methods

    def test_ensemble_selection_configuration(self):
        """Test ensemble selection configuration."""
        octo_task = Octo(
            description="step_1_octo",
            task_id=0,
            depends_on=None,
            models=[ModelName.ExtraTreesClassifier, ModelName.RandomForestClassifier],
            ensemble_selection=True,
            n_trials=5,
        )

        assert octo_task.ensemble_selection is True

    def test_hyperparameter_optimization_configuration(self):
        """Test hyperparameter optimization configuration."""
        octo_task = Octo(
            description="step_1_octo",
            task_id=0,
            depends_on=None,
            models=[ModelName.ExtraTreesClassifier],
            n_startup_trials=5,
            n_trials=5,
            max_features=5,
        )

        assert octo_task.n_startup_trials == 5
        assert octo_task.n_trials == 5
        assert octo_task.max_features == 5

    @pytest.mark.slow
    def test_octo_intro_classification_actual_execution(self, breast_cancer_dataset):
        """Test that the Octopus intro classification workflow actually runs end-to-end."""
        df, features = breast_cancer_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            study = OctoClassification(
                study_name="test_octo_intro_execution",
                target_metric="ACCBAL",
                feature_cols=features,
                target_col="target",
                sample_id_col="index",
                stratification_col="target",
                outer_split_seed=1,
                n_outer_splits=2,
                study_path=temp_dir,
                workflow=[
                    Octo(
                        description="step_1_octo",
                        task_id=0,
                        depends_on=None,
                        n_inner_splits=3,
                        models=[ModelName.ExtraTreesClassifier],
                        max_outliers=0,
                        fi_methods=[FIComputeMethod.PERMUTATION],
                        n_startup_trials=3,
                        n_trials=5,
                        max_features=5,
                        ensemble_selection=True,
                    )
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
            assert (study_path / "outersplit0").exists(), "Outersplit directory should exist"

            # Verify that the Octo step was executed by checking for task directories
            experiment_path = study_path / "outersplit0"
            task_dirs = [d for d in experiment_path.iterdir() if d.is_dir() and d.name.startswith("task")]

            assert len(task_dirs) >= 1, f"Should have at least 1 task directory, found: {[d.name for d in task_dirs]}"

    def test_full_configuration_parameters(self):
        """Test that all configuration parameters from the original workflow are supported."""
        octo_task = Octo(
            description="step_1_octo",
            task_id=0,
            depends_on=None,
            n_inner_splits=5,
            models=[ModelName.ExtraTreesClassifier, ModelName.RandomForestClassifier],
            max_outliers=0,
            fi_methods=[FIComputeMethod.PERMUTATION],
            n_startup_trials=10,
            n_trials=5,
            max_features=5,
            ensemble_selection=True,
        )

        # Verify all parameters are set correctly
        assert octo_task.description == "step_1_octo"
        assert octo_task.task_id == 0
        assert octo_task.depends_on is None

        assert octo_task.n_inner_splits == 5
        assert octo_task.models is not None
        assert set(octo_task.models) == {ModelName.ExtraTreesClassifier, ModelName.RandomForestClassifier}
        assert octo_task.max_outliers == 0
        assert octo_task.fi_methods == [FIComputeMethod.PERMUTATION]
        assert octo_task.n_startup_trials == 10
        assert octo_task.n_trials == 5
        assert octo_task.max_features == 5
        assert octo_task.ensemble_selection is True
