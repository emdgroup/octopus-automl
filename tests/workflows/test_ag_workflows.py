"""Test AutoGluon workflows."""

import os
import shutil
import tempfile

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from octopus.modules import AutoGluon
from octopus.study import OctoClassification, OctoRegression


class TestAutogluonWorkflows:
    """Test the AutoGluon classification workflow."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for studies
        self.temp_dir = tempfile.mkdtemp()
        self.studies_path = os.path.join(self.temp_dir, "studies")
        os.makedirs(self.studies_path, exist_ok=True)

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
        self.features = [f"feature_{i}" for i in range(5)]
        self.df = pd.DataFrame(X, columns=self.features)
        self.df["target"] = y
        self.df = self.df.reset_index()

    def teardown_method(self):
        """Clean up after each test method."""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_full_classification_workflow(self):
        """Test the complete classification workflow execution."""
        study = OctoClassification(
            name="test_classification_workflow",
            target_metric="ACCBAL",
            feature_cols=self.features,
            target_col="target",
            sample_id_col="index",
            stratification_col="target",
            metrics=["AUCROC", "ACCBAL", "ACC", "LOGLOSS"],
            datasplit_seed_outer=1234,
            n_folds_outer=5,
            path=self.studies_path,
            ignore_data_health_warning=True,
            outer_parallelization=True,
            run_single_outersplit_num=0,
            workflow=[
                AutoGluon(
                    description="ag_test",
                    task_id=0,
                    depends_on=None,
                    presets=["medium_quality"],
                    time_limit=15,
                    verbosity=0,
                ),
            ],
        )

        study.fit(data=self.df)

        # Verify that study files were created
        study_path = study.output_path
        assert study_path.exists(), "Study directory should be created"

        # Verify core study files
        assert (study_path / "study_config.json").exists(), "Config JSON should exist"
        assert (study_path / "data_raw.parquet").exists(), "Data parquet should exist"
        assert (study_path / "data_prepared.parquet").exists(), "Prepared data parquet should exist"

        # Verify outersplit and task output
        outersplit_dir = study_path / "outersplit0"
        assert outersplit_dir.exists(), "Outersplit directory should exist"

        task_dirs = [d for d in outersplit_dir.iterdir() if d.is_dir() and d.name.startswith("task")]
        assert len(task_dirs) >= 1, "Should have at least 1 task directory"

        # Verify AutoGluon task artifacts
        task_dir = task_dirs[0]
        assert (task_dir / "config" / "task_config.json").exists(), "Task config should exist"
        assert (task_dir / "results" / "best").exists(), "Best result directory should exist"
        assert (task_dir / "results" / "best" / "model").exists(), "Model directory should exist"

    def test_full_regression_workflow(self):
        """Test the complete regression workflow execution."""
        # Create synthetic regression dataset with reduced size for faster testing
        X, y, _ = make_regression(
            n_samples=30,
            n_features=5,
            n_informative=3,
            noise=0.1,
            random_state=42,
            coef=True,
        )

        # Create DataFrame similar to diabetes dataset structure
        feature_names = [f"feature_{i}" for i in range(5)]
        df_regression = pd.DataFrame(X, columns=feature_names)
        df_regression["target"] = y
        df_regression = df_regression.reset_index()

        study = OctoRegression(
            name="test_regression_workflow",
            target_metric="MAE",
            feature_cols=feature_names,
            target_col="target",
            sample_id_col="index",
            metrics=["MAE", "MSE", "R2"],
            datasplit_seed_outer=1234,
            n_folds_outer=2,
            path=self.studies_path,
            ignore_data_health_warning=True,
            outer_parallelization=False,
            run_single_outersplit_num=0,
            workflow=[
                AutoGluon(
                    description="ag_regression_test",
                    task_id=0,
                    depends_on=None,
                    presets=["medium_quality"],
                    time_limit=15,
                    verbosity=0,
                ),
            ],
        )

        study.fit(data=df_regression)

        # Verify that study files were created
        study_path = study.output_path
        assert study_path.exists(), "Study directory should be created"

        # Verify core study files
        assert (study_path / "study_config.json").exists(), "Config JSON should exist"
        assert (study_path / "data_raw.parquet").exists(), "Data parquet should exist"
        assert (study_path / "data_prepared.parquet").exists(), "Prepared data parquet should exist"

        # Verify outersplit and task output
        outersplit_dir = study_path / "outersplit0"
        assert outersplit_dir.exists(), "Outersplit directory should exist"

        task_dirs = [d for d in outersplit_dir.iterdir() if d.is_dir() and d.name.startswith("task")]
        assert len(task_dirs) >= 1, "Should have at least 1 task directory"

        # Verify AutoGluon task artifacts
        task_dir = task_dirs[0]
        assert (task_dir / "config" / "task_config.json").exists(), "Task config should exist"
        assert (task_dir / "results" / "best").exists(), "Best result directory should exist"
        assert (task_dir / "results" / "best" / "model").exists(), "Model directory should exist"


if __name__ == "__main__":
    # Allow running the test directly
    pytest.main([__file__, "-v"])
