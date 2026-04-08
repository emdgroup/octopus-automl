"""Test workflow for Octopus regression example."""

import tempfile

import pandas as pd
import pytest
from sklearn.datasets import make_regression

from octopus.modules import Octo
from octopus.study import OctoRegression
from octopus.types import FIComputeMethod, ModelName


class TestOctoRegression:
    """Test suite for Octopus regression workflow."""

    @pytest.fixture
    def diabetes_dataset(self):
        """Create synthetic regression dataset for testing."""
        X, y, _ = make_regression(
            n_samples=30,
            n_features=5,
            n_informative=3,
            noise=0.1,
            random_state=42,
            coef=True,
        )

        features = [f"feature_{i}" for i in range(5)]
        df = pd.DataFrame(X, columns=features)
        df["target"] = y
        df = df.reset_index()

        return df, features

    def test_diabetes_dataset_loading(self, diabetes_dataset):
        """Test that the diabetes dataset is loaded correctly."""
        df, features = diabetes_dataset

        assert isinstance(df, pd.DataFrame)
        assert "target" in df.columns
        assert "index" in df.columns
        assert len(features) == 5
        assert df.shape[0] == 30
        assert df["target"].dtype in ["float64", "int64"]
        assert df["target"].nunique() > 20

        for feature in features:
            assert feature in df.columns

        assert not df[features].isnull().any().any()
        assert not df["target"].isnull().any()

    def test_octo_study_configuration(self, diabetes_dataset):
        """Test OctoStudy configuration for diabetes dataset."""
        _, features = diabetes_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            study = OctoRegression(
                study_name="test_regression",
                target_metric="MAE",
                feature_cols=features,
                target_col="target",
                sample_id_col="index",
                studies_directory=temp_dir,
            )

            assert study.target_col == "target"
            assert len(study.feature_cols) == 5
            assert study.sample_id_col == "index"

    def test_octo_task_configuration(self):
        """Test that Octo task can be properly configured."""
        octo_task = Octo(
            task_id=0,
            depends_on=None,
            description="step_1",
            models=[
                ModelName.RandomForestRegressor,
                ModelName.XGBRegressor,
                ModelName.ExtraTreesRegressor,
                ModelName.ElasticNetRegressor,
                ModelName.GradientBoostingRegressor,
                ModelName.CatBoostRegressor,
            ],
            n_trials=12,
            max_features=6,
            ensemble_selection=True,
        )

        assert isinstance(octo_task, Octo)
        assert octo_task.task_id == 0
        assert octo_task.depends_on is None
        assert octo_task.description == "step_1"
        assert octo_task.n_trials == 12
        assert octo_task.max_features == 6
        assert octo_task.ensemble_selection is True

        expected_models = {
            ModelName.RandomForestRegressor,
            ModelName.XGBRegressor,
            ModelName.ExtraTreesRegressor,
            ModelName.ElasticNetRegressor,
            ModelName.GradientBoostingRegressor,
            ModelName.CatBoostRegressor,
        }
        assert octo_task.models is not None
        assert set(octo_task.models) == expected_models

    @pytest.mark.parametrize(
        "model",
        [
            ModelName.RandomForestRegressor,
            ModelName.XGBRegressor,
            ModelName.ExtraTreesRegressor,
            ModelName.ElasticNetRegressor,
            ModelName.GradientBoostingRegressor,
            ModelName.CatBoostRegressor,
        ],
    )
    def test_single_model_configuration(self, model):
        """Test configuration with different single models."""
        octo_task = Octo(
            task_id=0,
            depends_on=None,
            description="step_1",
            models=[model],
            n_trials=12,
            max_features=6,
            ensemble_selection=True,
        )

        assert octo_task.models == [model]
        assert octo_task.n_trials == 12
        assert octo_task.max_features == 6
        assert octo_task.ensemble_selection is True

    def test_multiple_models_configuration(self):
        """Test configuration with multiple models."""
        models = [ModelName.RandomForestRegressor, ModelName.XGBRegressor, ModelName.ExtraTreesRegressor]
        octo_task = Octo(
            task_id=0,
            depends_on=None,
            description="step_1",
            models=models,
            n_trials=12,
            max_features=6,
            ensemble_selection=True,
        )
        assert octo_task.models is not None
        assert set(octo_task.models) == set(models)

    def test_ensemble_selection_configuration(self):
        """Test ensemble selection configuration."""
        octo_task = Octo(
            task_id=0,
            depends_on=None,
            description="step_1",
            models=[ModelName.RandomForestRegressor, ModelName.XGBRegressor],
            n_trials=12,
            max_features=6,
            ensemble_selection=True,
        )

        assert octo_task.ensemble_selection is True

    def test_hyperparameter_optimization_configuration(self):
        """Test hyperparameter optimization configuration."""
        octo_task = Octo(
            task_id=0,
            depends_on=None,
            description="step_1",
            models=[ModelName.RandomForestRegressor],
            n_trials=12,
            max_features=6,
            ensemble_selection=True,
            n_startup_trials=5,
        )

        assert octo_task.n_trials == 12
        assert octo_task.max_features == 6
        assert octo_task.n_startup_trials == 5

    @pytest.mark.slow
    def test_octo_regression_actual_execution(self, diabetes_dataset):
        """Test that the Octopus regression workflow actually runs end-to-end."""
        df, features = diabetes_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            study = OctoRegression(
                study_name="test_octo_regression_execution",
                target_metric="MAE",
                feature_cols=features,
                target_col="target",
                sample_id_col="index",
                outer_split_seed=1234,
                n_outer_splits=2,
                studies_directory=temp_dir,
                single_outer_split=0,
                workflow=[
                    Octo(
                        task_id=0,
                        depends_on=None,
                        description="step_1",
                        models=[ModelName.RandomForestRegressor, ModelName.XGBRegressor],
                        n_trials=12,
                        max_features=6,
                        ensemble_selection=True,
                        n_startup_trials=3,
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

            # Verify that the Octo step was executed by checking for workflow directories
            experiment_path = study_path / "outersplit0"
            workflow_dirs = [d for d in experiment_path.iterdir() if d.is_dir() and d.name.startswith("task")]

            assert len(workflow_dirs) >= 1, (
                f"Should have at least 1 workflow directory, found: {[d.name for d in workflow_dirs]}"
            )

    def test_full_configuration_parameters(self):
        """Test that all configuration parameters are supported."""
        octo_task = Octo(
            task_id=0,
            depends_on=None,
            description="step_1",
            models=[
                ModelName.RandomForestRegressor,
                ModelName.XGBRegressor,
                ModelName.ExtraTreesRegressor,
                ModelName.ElasticNetRegressor,
                ModelName.GradientBoostingRegressor,
                ModelName.CatBoostRegressor,
            ],
            n_trials=12,
            max_features=6,
            ensemble_selection=True,
            max_outliers=0,
            fi_methods=[FIComputeMethod.PERMUTATION],
            n_startup_trials=10,
            n_inner_splits=5,
        )

        # Verify all parameters are set correctly
        assert octo_task.task_id == 0
        assert octo_task.depends_on is None
        assert octo_task.description == "step_1"

        expected_models = {
            ModelName.RandomForestRegressor,
            ModelName.XGBRegressor,
            ModelName.ExtraTreesRegressor,
            ModelName.ElasticNetRegressor,
            ModelName.GradientBoostingRegressor,
            ModelName.CatBoostRegressor,
        }
        assert octo_task.models is not None
        assert set(octo_task.models) == expected_models
        assert octo_task.max_outliers == 0
        assert octo_task.fi_methods == [FIComputeMethod.PERMUTATION]
        assert octo_task.n_startup_trials == 10
        assert octo_task.n_trials == 12
        assert octo_task.max_features == 6
        assert octo_task.ensemble_selection is True
        assert octo_task.n_inner_splits == 5
