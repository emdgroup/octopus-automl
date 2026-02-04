"""Test workflow for Octopus time-to-event (survival analysis) example."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from octopus import OctoTimeToEvent
from octopus.modules import Octo


@pytest.mark.skip(reason="Temporarily disabled")
class TestOctoTimeToEvent:
    """Test suite for Octopus time-to-event workflow."""

    @pytest.fixture
    def survival_dataset(self):
        """Create synthetic time-to-event dataset for testing."""
        np.random.seed(42)

        n_samples = 100
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        features = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=features)

        risk_score = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]
        baseline_hazard = 0.1
        duration = np.random.exponential(scale=1.0 / (baseline_hazard * np.exp(risk_score)))

        censoring_time = np.random.exponential(scale=15, size=n_samples)
        observed_time = np.minimum(duration, censoring_time)
        event = (duration <= censoring_time).astype(int)

        df["duration"] = observed_time
        df["event"] = event
        df = df.reset_index()

        return df, features

    def test_survival_dataset_loading(self, survival_dataset):
        """Test that the survival dataset is loaded correctly."""
        df, features = survival_dataset

        assert isinstance(df, pd.DataFrame)
        assert "duration" in df.columns
        assert "event" in df.columns
        assert "index" in df.columns
        assert len(features) == 5
        assert df.shape[0] == 100

        assert (df["duration"] > 0).all()
        assert df["duration"].dtype in ["float64", "int64"]
        assert df["event"].dtype in ["int64", "int32", "bool"]
        assert set(df["event"].unique()).issubset({0, 1})

        for feature in features:
            assert feature in df.columns

        assert not df[features].isnull().any().any()
        assert not df["duration"].isnull().any()
        assert not df["event"].isnull().any()

        assert df["event"].sum() > 0, "Should have at least one event"
        assert (df["event"] == 0).sum() > 0, "Should have at least one censored observation"

    def test_octo_study_configuration(self, survival_dataset):
        """Test OctoStudy configuration for survival dataset."""
        _, features = survival_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            study = OctoTimeToEvent(
                name="test_t2e",
                target_metric="CI",
                feature_cols=features,
                duration_column="duration",
                event_column="event",
                sample_id_col="index",
                path=temp_dir,
                ignore_data_health_warning=True,
            )

            assert study.target_cols == ["duration", "event"]
            assert len(study.feature_cols) == 5
            assert study.sample_id_col == "index"
            assert study.target_assignments == {"duration": "duration", "event": "event"}

    def test_octo_task_configuration(self):
        """Test that Octo task can be properly configured for time-to-event."""
        octo_task = Octo(
            task_id=0,
            depends_on_task=-1,
            description="step_1",
            models=["ExtraTreesSurv"],
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
        assert octo_task.models == ["ExtraTreesSurv"]

    def test_single_model_configuration(self):
        """Test configuration with ExtraTreesSurv model."""
        octo_task = Octo(
            task_id=0,
            depends_on_task=-1,
            description="step_1",
            models=["ExtraTreesSurv"],
            n_trials=12,
            max_features=6,
            ensemble_selection=True,
            ensel_n_save_trials=10,
        )

        assert octo_task.models == ["ExtraTreesSurv"]
        assert octo_task.n_trials == 12
        assert octo_task.max_features == 6
        assert octo_task.ensemble_selection is True
        assert octo_task.ensel_n_save_trials == 10

    def test_ensemble_selection_configuration(self):
        """Test ensemble selection configuration."""
        octo_task = Octo(
            task_id=0,
            depends_on_task=-1,
            description="step_1",
            models=["ExtraTreesSurv"],
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
            models=["ExtraTreesSurv"],
            n_trials=12,
            max_features=6,
            ensemble_selection=True,
            ensel_n_save_trials=10,
            optuna_seed=42,
            n_optuna_startup_trials=5,
            penalty_factor=1.5,
        )

        assert "ExtraTreesSurv" in octo_task.models
        assert octo_task.n_trials == 12
        assert octo_task.max_features == 6
        assert octo_task.ensemble_selection is True
        assert octo_task.ensel_n_save_trials == 10
        assert octo_task.optuna_seed == 42
        assert octo_task.n_optuna_startup_trials == 5
        assert octo_task.penalty_factor == 1.5

    @pytest.mark.slow
    def test_octo_timetoevent_actual_execution(self, survival_dataset):
        """Test that the Octopus time-to-event workflow actually runs end-to-end."""
        df, features = survival_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            study = OctoTimeToEvent(
                name="test_octo_t2e_execution",
                target_metric="CI",
                feature_cols=features,
                duration_column="duration",
                event_column="event",
                sample_id_col="index",
                metrics=["CI"],
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
                        models=["ExtraTreesSurv"],
                        n_trials=12,
                        max_features=6,
                        ensemble_selection=True,
                        ensel_n_save_trials=10,
                        model_seed=0,
                        n_jobs=1,
                        fi_methods_bestbag=["shap"],
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
            study_path = Path(temp_dir) / "test_octo_t2e_execution"
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
            models=["ExtraTreesSurv"],
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

        assert octo_task.task_id == 0
        assert octo_task.depends_on_task == -1
        assert octo_task.description == "step_1"
        assert octo_task.models == ["ExtraTreesSurv"]
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
