"""Test workflow for Octopus regression example."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_regression

from octopus import OctoRegression
from octopus.modules import Octo


class TestOctoRegression:
    """Test suite for Octopus regression workflow."""

    @pytest.fixture
    def diabetes_dataset(self):
        """Create synthetic regression dataset for testing."""
        X, y = make_regression(
            n_samples=30,
            n_features=5,
            n_informative=3,
            noise=0.1,
            random_state=42,
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
                name="test_regression",
                target_metric="MAE",
                feature_cols=features,
                target="target",
                sample_id="index",
                path=temp_dir,
                ignore_data_health_warning=True,
            )

            assert study.target_cols == ["target"]
            assert len(study.feature_cols) == 5
            assert study.sample_id == "index"

    def test_octo_task_configuration(self):
        """Test that Octo task can be properly configured."""
        octo_task = Octo(
            task_id=0,
            depends_on_task=-1,
            description="step_1",
            models=[
                "RandomForestRegressor",
                "XGBRegressor",
                "ExtraTreesRegressor",
                "ElasticNetRegressor",
                "GradientBoostingRegressor",
                "CatBoostRegressor",
            ],
            n_trials=12,
            max_features=6,
            ensemble_selection=True,
            ensel_n_save_trials=10,
        )

        assert isinstance(octo_task, Octo)
        assert octo_task.task_id == 0
        assert octo_task.depends_on_task == -1
        assert octo_task.description == "step_1"
        assert octo_task.n_trials == 12
        assert octo_task.max_features == 6
        assert octo_task.ensemble_selection is True
        assert octo_task.ensel_n_save_trials == 10

        expected_models = {
            "RandomForestRegressor",
            "XGBRegressor",
            "ExtraTreesRegressor",
            "ElasticNetRegressor",
            "GradientBoostingRegressor",
            "CatBoostRegressor",
        }
        assert set(octo_task.models) == expected_models

    @pytest.mark.parametrize(
        "model",
        [
            "RandomForestRegressor",
            "XGBRegressor",
            "ExtraTreesRegressor",
            "ElasticNetRegressor",
            "GradientBoostingRegressor",
            "CatBoostRegressor",
        ],
    )
    def test_single_model_configuration(self, model):
        """Test configuration with different single models."""
        octo_task = Octo(
            task_id=0,
            depends_on_task=-1,
            description="step_1",
            models=[model],
            n_trials=12,
            max_features=6,
            ensemble_selection=True,
            ensel_n_save_trials=10,
        )

        assert octo_task.models == [model]
        assert octo_task.n_trials == 12
        assert octo_task.max_features == 6
        assert octo_task.ensemble_selection is True
        assert octo_task.ensel_n_save_trials == 10

    def test_multiple_models_configuration(self):
        """Test configuration with multiple models."""
        models = ["RandomForestRegressor", "XGBRegressor", "ExtraTreesRegressor"]
        octo_task = Octo(
            task_id=0,
            depends_on_task=-1,
            description="step_1",
            models=models,
            n_trials=12,
            max_features=6,
            ensemble_selection=True,
            ensel_n_save_trials=10,
        )

        assert set(octo_task.models) == set(models)

    def test_ensemble_selection_configuration(self):
        """Test ensemble selection configuration."""
        octo_task = Octo(
            task_id=0,
            depends_on_task=-1,
            description="step_1",
            models=["RandomForestRegressor", "XGBRegressor"],
            n_trials=12,
            max_features=6,
            ensemble_selection=True,
            ensel_n_save_trials=10,
        )

        assert octo_task.ensemble_selection is True
        assert octo_task.ensel_n_save_trials == 10

    def test_hyperparameter_optimization_configuration(self):
        """Test hyperparameter optimization configuration."""
        octo_task = Octo(
            task_id=0,
            depends_on_task=-1,
            description="step_1",
            models=["RandomForestRegressor"],
            n_trials=12,
            max_features=6,
            ensemble_selection=True,
            ensel_n_save_trials=10,
            optuna_seed=42,
            n_optuna_startup_trials=5,
            penalty_factor=1.5,
        )

        assert octo_task.n_trials == 12
        assert octo_task.max_features == 6
        assert octo_task.optuna_seed == 42
        assert octo_task.n_optuna_startup_trials == 5
        assert octo_task.penalty_factor == 1.5

    @pytest.mark.slow
    def test_octo_regression_actual_execution(self, diabetes_dataset):
        """Test that the Octopus regression workflow actually runs end-to-end."""
        df, features = diabetes_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            study = OctoRegression(
                name="test_octo_regression_execution",
                target_metric="MAE",
                feature_cols=features,
                target="target",
                sample_id="index",
                metrics=["MAE", "MSE", "R2"],
                datasplit_seed_outer=1234,
                n_folds_outer=2,
                path=temp_dir,
                ignore_data_health_warning=True,
                outer_parallelization=False,
                run_single_experiment_num=0,
                workflow=[
                    Octo(
                        task_id=0,
                        depends_on_task=-1,
                        description="step_1",
                        models=["RandomForestRegressor", "XGBRegressor"],
                        n_trials=12,
                        max_features=6,
                        ensemble_selection=True,
                        ensel_n_save_trials=10,
                        model_seed=0,
                        n_jobs=1,
                        inner_parallelization=True,
                        n_workers=2,
                        optuna_seed=0,
                        n_optuna_startup_trials=3,
                        resume_optimization=False,
                        penalty_factor=1.0,
                    )
                ],
            )

            study.fit(data=df)

            # Verify that the study was created and files exist
            study_path = Path(temp_dir) / "test_octo_regression_execution"
            assert study_path.exists(), "Study directory should be created"

            assert (study_path / "octo_manager.log").exists(), "Octo Manager log file should exist"

            # Check for expected files (new architecture uses files, not directories)
            assert (study_path / "data.parquet").exists(), "Data parquet file should exist"
            assert (study_path / "data_prepared.parquet").exists(), "Prepared data parquet file should exist"
            assert (study_path / "config.json").exists(), "Config JSON file should exist"
            assert (study_path / "outersplit0").exists(), "Outersplit directory should exist"

            # Verify that the Octo step was executed by checking for workflow directories
            experiment_path = study_path / "outersplit0"
            workflow_dirs = [d for d in experiment_path.iterdir() if d.is_dir() and d.name.startswith("workflowtask")]

            assert len(workflow_dirs) >= 1, (
                f"Should have at least 1 workflow directory, found: {[d.name for d in workflow_dirs]}"
            )

    def test_full_configuration_parameters(self):
        """Test that all configuration parameters are supported."""
        octo_task = Octo(
            task_id=0,
            depends_on_task=-1,
            description="step_1",
            models=[
                "RandomForestRegressor",
                "XGBRegressor",
                "ExtraTreesRegressor",
                "ElasticNetRegressor",
                "GradientBoostingRegressor",
                "CatBoostRegressor",
            ],
            n_trials=12,
            max_features=6,
            ensemble_selection=True,
            ensel_n_save_trials=10,
            model_seed=0,
            n_jobs=1,
            max_outl=0,
            fi_methods_bestbag=["permutation"],
            inner_parallelization=True,
            n_workers=5,
            optuna_seed=0,
            n_optuna_startup_trials=10,
            resume_optimization=False,
            penalty_factor=1.0,
            n_folds_inner=5,
        )

        # Verify all parameters are set correctly
        assert octo_task.task_id == 0
        assert octo_task.depends_on_task == -1
        assert octo_task.description == "step_1"

        expected_models = {
            "RandomForestRegressor",
            "XGBRegressor",
            "ExtraTreesRegressor",
            "ElasticNetRegressor",
            "GradientBoostingRegressor",
            "CatBoostRegressor",
        }
        assert set(octo_task.models) == expected_models
        assert octo_task.model_seed == 0
        assert octo_task.n_jobs == 1
        assert octo_task.max_outl == 0
        assert octo_task.fi_methods_bestbag == ["permutation"]
        assert octo_task.inner_parallelization is True
        assert octo_task.n_workers == 5
        assert octo_task.optuna_seed == 0
        assert octo_task.n_optuna_startup_trials == 10
        assert octo_task.resume_optimization is False
        assert octo_task.n_trials == 12
        assert octo_task.max_features == 6
        assert octo_task.penalty_factor == 1.0
        assert octo_task.ensemble_selection is True
        assert octo_task.ensel_n_save_trials == 10
        assert octo_task.n_folds_inner == 5
