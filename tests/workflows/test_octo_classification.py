"""Test workflow for Octopus intro classification example."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from octopus import OctoClassification
from octopus.modules import Octo


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
                name="test_classification",
                target_metric="ACCBAL",
                feature_cols=features,
                target="target",
                sample_id="index",
                stratification_column="target",
                path=temp_dir,
                ignore_data_health_warning=True,
            )

            assert study.target_cols == ["target"]
            assert len(study.feature_cols) == 5
            assert study.sample_id == "index"
            assert study.stratification_column == "target"

    def test_octo_task_configuration(self):
        """Test that Octo task can be properly configured."""
        octo_task = Octo(
            description="step_1_octo",
            task_id=0,
            depends_on_task=-1,
            load_task=False,
            n_folds_inner=3,
            models=["ExtraTreesClassifier", "RandomForestClassifier"],
            fi_methods_bestbag=["permutation"],
            inner_parallelization=True,
            n_workers=3,
            optuna_seed=0,
            n_optuna_startup_trials=5,
            resume_optimization=False,
            n_trials=6,
            max_features=5,
            penalty_factor=1.0,
            ensemble_selection=True,
            ensel_n_save_trials=5,
        )

        assert isinstance(octo_task, Octo)
        assert octo_task.task_id == 0
        assert octo_task.depends_on_task == -1
        assert octo_task.description == "step_1_octo"
        assert octo_task.n_folds_inner == 3
        assert set(octo_task.models) == {"ExtraTreesClassifier", "RandomForestClassifier"}

    @pytest.mark.parametrize("model", ["ExtraTreesClassifier", "RandomForestClassifier"])
    def test_single_model_configuration(self, model):
        """Test configuration with different single models."""
        octo_task = Octo(
            description="step_1_octo",
            task_id=0,
            depends_on_task=-1,
            models=[model],
            n_trials=3,
            n_folds_inner=3,
        )

        assert octo_task.models == [model]

    def test_multiple_models_configuration(self):
        """Test configuration with multiple models."""
        models = ["ExtraTreesClassifier", "RandomForestClassifier"]
        octo_task = Octo(
            description="step_1_octo",
            task_id=0,
            depends_on_task=-1,
            models=models,
            n_trials=5,
            n_folds_inner=3,
        )

        assert set(octo_task.models) == set(models)

    def test_feature_importance_configuration(self):
        """Test feature importance method configuration."""
        fi_methods = ["permutation"]
        octo_task = Octo(
            description="step_1_octo",
            task_id=0,
            depends_on_task=-1,
            models=["ExtraTreesClassifier"],
            fi_methods_bestbag=fi_methods,
            n_trials=3,
        )

        assert octo_task.fi_methods_bestbag == fi_methods

    def test_ensemble_selection_configuration(self):
        """Test ensemble selection configuration."""
        octo_task = Octo(
            description="step_1_octo",
            task_id=0,
            depends_on_task=-1,
            models=["ExtraTreesClassifier", "RandomForestClassifier"],
            ensemble_selection=True,
            ensel_n_save_trials=15,
            n_trials=5,
        )

        assert octo_task.ensemble_selection is True
        assert octo_task.ensel_n_save_trials == 15

    def test_hyperparameter_optimization_configuration(self):
        """Test hyperparameter optimization configuration."""
        octo_task = Octo(
            description="step_1_octo",
            task_id=0,
            depends_on_task=-1,
            models=["ExtraTreesClassifier"],
            optuna_seed=42,
            n_optuna_startup_trials=5,
            n_trials=5,
            max_features=5,
            penalty_factor=1.5,
        )

        assert octo_task.optuna_seed == 42
        assert octo_task.n_optuna_startup_trials == 5
        assert octo_task.n_trials == 5
        assert octo_task.max_features == 5
        assert octo_task.penalty_factor == 1.5

    @pytest.mark.slow
    def test_octo_intro_classification_actual_execution(self, breast_cancer_dataset):
        """Test that the Octopus intro classification workflow actually runs end-to-end."""
        df, features = breast_cancer_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            study = OctoClassification(
                name="test_octo_intro_execution",
                target_metric="ACCBAL",
                feature_cols=features,
                target="target",
                sample_id="index",
                stratification_column="target",
                metrics=["AUCROC", "ACCBAL", "ACC", "LOGLOSS"],
                datasplit_seed_outer=1234,
                n_folds_outer=2,
                path=temp_dir,
                ignore_data_health_warning=True,
                outer_parallelization=False,
                run_single_experiment_num=0,
                workflow=[
                    Octo(
                        description="step_1_octo",
                        task_id=0,
                        depends_on_task=-1,
                        load_task=False,
                        n_folds_inner=3,
                        models=["ExtraTreesClassifier"],
                        model_seed=0,
                        n_jobs=1,
                        max_outl=0,
                        fi_methods_bestbag=["permutation"],
                        inner_parallelization=True,
                        n_workers=2,
                        optuna_seed=0,
                        n_optuna_startup_trials=3,
                        resume_optimization=False,
                        n_trials=5,
                        max_features=5,
                        penalty_factor=1.0,
                        ensemble_selection=True,
                        ensel_n_save_trials=5,
                    )
                ],
            )

            study.fit(data=df)

            # Verify that the study was created and files exist
            study_path = Path(temp_dir) / "test_octo_intro_execution"
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
        """Test that all configuration parameters from the original workflow are supported."""
        octo_task = Octo(
            description="step_1_octo",
            task_id=0,
            depends_on_task=-1,
            load_task=False,
            n_folds_inner=5,
            models=["ExtraTreesClassifier", "RandomForestClassifier"],
            model_seed=0,
            n_jobs=1,
            max_outl=0,
            fi_methods_bestbag=["permutation"],
            inner_parallelization=True,
            n_workers=5,
            optuna_seed=0,
            n_optuna_startup_trials=10,
            resume_optimization=False,
            n_trials=5,
            max_features=5,
            penalty_factor=1.0,
            ensemble_selection=True,
            ensel_n_save_trials=10,
        )

        # Verify all parameters are set correctly
        assert octo_task.description == "step_1_octo"
        assert octo_task.task_id == 0
        assert octo_task.depends_on_task == -1
        assert octo_task.load_task is False
        assert octo_task.n_folds_inner == 5
        assert set(octo_task.models) == {"ExtraTreesClassifier", "RandomForestClassifier"}
        assert octo_task.model_seed == 0
        assert octo_task.n_jobs == 1
        assert octo_task.max_outl == 0
        assert octo_task.fi_methods_bestbag == ["permutation"]
        assert octo_task.inner_parallelization is True
        assert octo_task.n_workers == 5
        assert octo_task.optuna_seed == 0
        assert octo_task.n_optuna_startup_trials == 10
        assert octo_task.resume_optimization is False
        assert octo_task.n_trials == 5
        assert octo_task.max_features == 5
        assert octo_task.penalty_factor == 1.0
        assert octo_task.ensemble_selection is True
        assert octo_task.ensel_n_save_trials == 10
