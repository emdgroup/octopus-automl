"""Test AutoGluon workflows."""

import os
import re
import shutil
import tempfile

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from upath import UPath

from octopus import OctoClassification, OctoRegression
from octopus.experiment import OctoExperiment
from octopus.modules import AutoGluon


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
            target="target",
            sample_id_col="index",
            stratification_column="target",
            metrics=["AUCROC", "ACCBAL", "ACC", "LOGLOSS"],
            datasplit_seed_outer=1234,
            n_folds_outer=5,
            path=self.studies_path,
            ignore_data_health_warning=True,
            outer_parallelization=True,
            run_single_experiment_num=0,
            workflow=[
                AutoGluon(
                    description="ag_test",
                    task_id=0,
                    depends_on_task=-1,
                    presets=["medium_quality"],
                    time_limit=15,
                    verbosity=0,
                ),
            ],
        )

        study.fit(data=self.df)

        # Verify that study files were created
        study_path = UPath(self.studies_path) / "test_classification_workflow"
        assert study_path.exists(), "Study directory should be created"

        # Test specific keys exist
        self._test_specific_keys(study_path)

        success = True
        assert success is True

    def test_full_regression_workflow(self):
        """Test the complete regression workflow execution."""
        # Create synthetic regression dataset with reduced size for faster testing
        X, y = make_regression(
            n_samples=30,
            n_features=5,
            n_informative=3,
            noise=0.1,
            random_state=42,
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
            target="target",
            sample_id_col="index",
            metrics=["MAE", "MSE", "R2"],
            datasplit_seed_outer=1234,
            n_folds_outer=2,
            path=self.studies_path,
            ignore_data_health_warning=True,
            outer_parallelization=False,
            run_single_experiment_num=0,
            workflow=[
                AutoGluon(
                    description="ag_regression_test",
                    task_id=0,
                    depends_on_task=-1,
                    presets=["medium_quality"],
                    time_limit=15,
                    verbosity=0,
                ),
            ],
        )

        study.fit(data=df_regression)

        # Verify that study files were created
        study_path = UPath(self.studies_path) / "test_regression_workflow"
        assert study_path.exists(), "Study directory should be created"

        # Test specific keys exist
        self._test_specific_keys(study_path)

        success = True
        assert success is True

    def _test_specific_keys(self, study_path):
        """Test that specific keys exist in the experiment results."""
        print("\n=== Testing Specific Keys ===")

        # Find experiment directories (now called outersplit)
        path_experiments = [f for f in study_path.glob("outersplit*") if f.is_dir()]

        assert len(path_experiments) > 0, "No experiment directories found"

        # Track if we found the required keys
        found_autogluon_result = False
        found_autogluon_permutation_test = False

        # Iterate through experiments
        for path_exp in path_experiments:
            exp_name = str(path_exp.name)
            match = re.search(r"\d+", exp_name)
            exp_num = int(match.group()) if match else None

            # Find workflow directories
            path_workflows = [f for f in path_exp.glob("workflowtask*") if f.is_dir()]

            # Iterate through sequences
            for path_workflow in path_workflows:
                seq_name = str(path_workflow.name)
                match = re.search(r"\d+", seq_name)
                seq_num = int(match.group()) if match else None

                # Look for experiment pickle file
                path_exp_pkl = path_workflow / f"exp{exp_num}_{seq_num}.pkl"

                if path_exp_pkl.exists():
                    try:
                        # Load experiment
                        exp = OctoExperiment.from_pickle(path_exp_pkl)

                        # Test for 'autogluon' results key
                        if "autogluon" in exp.results:
                            found_autogluon_result = True
                            print(f"✓ Found 'autogluon' results key in {path_exp_pkl}")

                            # Test for 'autogluon_permutation_test' feature importance key
                            result = exp.results["autogluon"]
                            if "autogluon_permutation_test" in getattr(result, "feature_importances", {}):
                                found_autogluon_permutation_test = True
                                print(f"✓ Found 'autogluon_permutation_test' feature importance key in {path_exp_pkl}")

                    except Exception as e:
                        print(f"Error loading experiment {path_exp_pkl}: {e}")

        # Assert that we found the required keys
        assert found_autogluon_result, "Expected 'autogluon' key not found in experiment results"
        assert found_autogluon_permutation_test, (
            "Expected 'autogluon_permutation_test' key not found in feature importances"
        )

        print("✓ All required keys found successfully")
        print("=== Key Testing Complete ===\n")


if __name__ == "__main__":
    # Allow running the test directly
    pytest.main([__file__, "-v"])
