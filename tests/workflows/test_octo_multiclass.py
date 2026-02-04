"""Test workflow for Octopus multiclass classification using Wine dataset."""

import tempfile
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from octopus import OctoClassification
from octopus.modules import Octo


class TestOctoMulticlass:
    """Test suite for Octopus multiclass classification workflow."""

    @pytest.fixture
    def wine_dataset(self):
        """Create synthetic multiclass dataset for testing."""
        X, y = make_classification(
            n_samples=150,
            n_features=5,
            n_informative=3,
            n_redundant=2,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42,
        )

        feature_names = [f"feature_{i}" for i in range(5)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        df = df.reset_index()

        class MockWine:
            target_names: ClassVar[list[str]] = ["class_0", "class_1", "class_2"]

        wine = MockWine()

        return df, feature_names, wine

    def test_wine_dataset_loading(self, wine_dataset):
        """Test that the synthetic dataset is loaded correctly."""
        df, features, wine = wine_dataset

        assert isinstance(df, pd.DataFrame)
        assert "target" in df.columns
        assert "index" in df.columns
        assert len(features) == 5
        assert df.shape[0] == 150

        unique_targets = df["target"].unique()
        assert len(unique_targets) == 3
        assert set(unique_targets) == {0, 1, 2}

        for feature in features:
            assert feature in df.columns

        assert not df[features].isnull().any().any()
        assert not df["target"].isnull().any()

        target_counts = df["target"].value_counts().sort_index()
        assert len(target_counts) == 3
        assert all(count > 0 for count in target_counts.values)
        assert len(wine.target_names) == 3

    def test_octo_study_configuration(self, wine_dataset):
        """Test OctoStudy configuration for synthetic dataset."""
        _, features, _wine = wine_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            study = OctoClassification(
                name="test_multiclass",
                target_metric="AUCROC_MACRO",
                feature_cols=features,
                target="target",
                sample_id_col="index",
                stratification_col="target",
                path=temp_dir,
                ignore_data_health_warning=True,
            )

            assert study.target_cols == ["target"]
            assert len(study.feature_cols) == 5
            assert study.sample_id_col == "index"
            assert study.stratification_col == "target"

    def test_multiclass_task_configuration(self):
        """Test that multiclass Octo task can be properly configured."""
        octo_task = Octo(
            description="step_1_octo_multiclass",
            task_id=0,
            depends_on_task=-1,
            load_task=False,
            n_folds_inner=5,
            models=[
                "ExtraTreesClassifier",
                "RandomForestClassifier",
                "XGBClassifier",
                "CatBoostClassifier",
            ],
            model_seed=0,
            n_jobs=1,
            max_outl=0,
            fi_methods_bestbag=["permutation"],
            inner_parallelization=True,
            n_workers=5,
            n_trials=20,
        )

        assert isinstance(octo_task, Octo)
        assert octo_task.task_id == 0
        assert octo_task.depends_on_task == -1
        assert octo_task.description == "step_1_octo_multiclass"
        assert octo_task.n_folds_inner == 5
        assert set(octo_task.models) == {
            "ExtraTreesClassifier",
            "RandomForestClassifier",
            "XGBClassifier",
            "CatBoostClassifier",
        }

    @pytest.mark.parametrize(
        "model", ["ExtraTreesClassifier", "RandomForestClassifier", "XGBClassifier", "CatBoostClassifier"]
    )
    def test_multiclass_single_model_configuration(self, model):
        """Test configuration with different single multiclass models."""
        octo_task = Octo(
            description="step_1_octo_multiclass",
            task_id=0,
            depends_on_task=-1,
            models=[model],
            n_trials=5,
            n_folds_inner=3,
        )

        assert octo_task.models == [model]

    def test_multiclass_multiple_models_configuration(self):
        """Test configuration with multiple multiclass models."""
        models = ["ExtraTreesClassifier", "RandomForestClassifier", "XGBClassifier", "CatBoostClassifier"]
        octo_task = Octo(
            description="step_1_octo_multiclass",
            task_id=0,
            depends_on_task=-1,
            models=models,
            n_trials=10,
            n_folds_inner=3,
        )

        assert set(octo_task.models) == set(models)

    def test_multiclass_metrics_configuration(self):
        """Test multiclass-specific metrics configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            study = OctoClassification(
                name="test_multiclass_metrics",
                target_metric="AUCROC_MACRO",
                feature_cols=["f1"],
                target="target",
                sample_id_col="index",
                metrics=["AUCROC_MACRO", "AUCROC_WEIGHTED", "ACCBAL_MC"],
                path=temp_dir,
                ignore_data_health_warning=True,
            )

            assert study.target_metric == "AUCROC_MACRO"
            assert "AUCROC_MACRO" in study.metrics
            assert "AUCROC_WEIGHTED" in study.metrics
            assert "ACCBAL_MC" in study.metrics

    def test_feature_importance_configuration(self):
        """Test feature importance method configuration for multiclass."""
        fi_methods = ["permutation"]
        octo_task = Octo(
            description="step_1_octo_multiclass",
            task_id=0,
            depends_on_task=-1,
            models=["ExtraTreesClassifier"],
            fi_methods_bestbag=fi_methods,
            n_trials=5,
        )

        assert octo_task.fi_methods_bestbag == fi_methods

    def test_hyperparameter_optimization_configuration(self):
        """Test hyperparameter optimization configuration for multiclass."""
        octo_task = Octo(
            description="step_1_octo_multiclass",
            task_id=0,
            depends_on_task=-1,
            models=["ExtraTreesClassifier"],
            n_trials=25,
            n_folds_inner=5,
        )

        assert octo_task.n_trials == 25
        assert octo_task.n_folds_inner == 5

    @pytest.mark.slow
    def test_multiclass_workflow_actual_execution(self, wine_dataset):
        """Test that the multiclass workflow actually runs end-to-end."""
        df, features, _wine = wine_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            study = OctoClassification(
                name="test_multiclass_execution",
                target_metric="AUCROC_MACRO",
                feature_cols=features,
                target="target",
                sample_id_col="index",
                stratification_col="target",
                metrics=["AUCROC_MACRO", "AUCROC_WEIGHTED", "ACCBAL_MC"],
                datasplit_seed_outer=1234,
                n_folds_outer=2,
                path=temp_dir,
                ignore_data_health_warning=True,
                outer_parallelization=False,
                run_single_experiment_num=0,
                workflow=[
                    Octo(
                        description="step_1_octo_multiclass",
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
                        n_workers=3,
                        n_trials=12,
                    )
                ],
            )

            study.fit(data=df)

            # Verify that the study was created and files exist
            study_path = Path(temp_dir) / "test_multiclass_execution"
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

    def test_full_multiclass_configuration_parameters(self):
        """Test that all configuration parameters from the multiclass workflow are supported."""
        octo_task = Octo(
            description="step_1_octo_multiclass",
            task_id=0,
            depends_on_task=-1,
            load_task=False,
            n_folds_inner=5,
            models=[
                "ExtraTreesClassifier",
                "RandomForestClassifier",
                "XGBClassifier",
                "CatBoostClassifier",
            ],
            model_seed=0,
            n_jobs=1,
            max_outl=0,
            fi_methods_bestbag=["permutation"],
            inner_parallelization=True,
            n_workers=5,
            n_trials=20,
        )

        # Verify all parameters are set correctly
        assert octo_task.description == "step_1_octo_multiclass"
        assert octo_task.task_id == 0
        assert octo_task.depends_on_task == -1
        assert octo_task.load_task is False
        assert octo_task.n_folds_inner == 5
        assert set(octo_task.models) == {
            "ExtraTreesClassifier",
            "RandomForestClassifier",
            "XGBClassifier",
            "CatBoostClassifier",
        }
        assert octo_task.model_seed == 0
        assert octo_task.n_jobs == 1
        assert octo_task.max_outl == 0
        assert octo_task.fi_methods_bestbag == ["permutation"]
        assert octo_task.inner_parallelization is True
        assert octo_task.n_workers == 5
        assert octo_task.n_trials == 20

    def test_multiclass_target_metric_options(self):
        """Test different target metrics suitable for multiclass classification."""
        target_metrics = ["AUCROC_MACRO", "AUCROC_WEIGHTED", "ACCBAL_MC"]

        for target_metric in target_metrics:
            with tempfile.TemporaryDirectory() as temp_dir:
                study = OctoClassification(
                    name=f"test_multiclass_{target_metric.lower()}",
                    target_metric=target_metric,
                    feature_cols=["f1"],
                    target="target",
                    sample_id_col="index",
                    metrics=["AUCROC_MACRO", "AUCROC_WEIGHTED", "ACCBAL_MC"],
                    path=temp_dir,
                    ignore_data_health_warning=True,
                )

                assert study.target_metric == target_metric
                # ml_type will be set to MULTICLASS or CLASSIFICATION in fit() based on actual data
                assert study.ml_type.value == "classification"  # Default before fit()
