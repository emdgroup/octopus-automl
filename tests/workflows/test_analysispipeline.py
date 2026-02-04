"""Test analysis pipeline for OctoPredict functionality."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import auc, roc_curve
from upath import UPath

from octopus import OctoClassification
from octopus.metrics.utils import get_performance_from_model
from octopus.modules import Octo
from octopus.predict import OctoPredict


@pytest.fixture(scope="session")
def classification_dataset():
    """Create synthetic binary classification dataset for testing."""
    # Create a highly separable dataset with 100 samples
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,  # Single cluster per class for maximum separation
        weights=[0.5, 0.5],
        flip_y=0.0,  # No label noise
        class_sep=10.0,  # Maximum class separation
        random_state=42,
        shuffle=True,
    )

    feature_names = [f"feature_{i}" for i in range(4)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    df = df.reset_index()

    return df, feature_names


@pytest.fixture(scope="session")
def trained_study(classification_dataset, tmp_path_factory):
    """Create and train a minimal study for testing analysis pipeline."""
    df, features = classification_dataset

    temp_dir = tmp_path_factory.mktemp("test_data")

    # Create OctoClassification study
    study = OctoClassification(
        name="test_analysis_pipeline",
        target_metric="ACCBAL",
        metrics=["AUCROC", "ACCBAL", "ACC", "F1", "AUCPR"],
        feature_cols=features,
        target="target",
        sample_id_col="index",
        datasplit_type="sample",
        stratification_column="target",
        datasplit_seed_outer=1234,
        n_folds_outer=5,
        start_with_empty_study=True,
        path=str(temp_dir),
        silently_overwrite_study=True,
        ignore_data_health_warning=True,
        positive_class=1,
        outer_parallelization=False,
        run_single_experiment_num=0,  # Run only experiment 0
        workflow=[
            Octo(
                description="step_1_octo",
                task_id=0,
                depends_on_task=-1,
                n_folds_inner=5,
                models=["ExtraTreesClassifier"],
                model_seed=0,
                n_jobs=1,
                n_trials=20,
                fi_methods_bestbag=["permutation"],
                max_features=4,
            )
        ],
    )

    # Run the study
    study.fit(df)

    study_path = UPath(temp_dir) / "test_analysis_pipeline"
    yield study_path


class TestAnalysisPipeline:
    """Test suite for analysis pipeline using OctoPredict."""

    @pytest.mark.slow
    def test_octopredict_initialization(self, trained_study):
        """Test OctoPredict initialization with trained study."""
        task_item = OctoPredict(
            study_path=trained_study,
            task_id=0,
            results_key="best",
        )

        # Verify initialization
        assert task_item.study_path == trained_study
        assert task_item.task_id == 0
        assert task_item.results_key == "best"
        assert len(task_item.experiments) > 0

        # Verify experiments structure
        for _exp_id, experiment in task_item.experiments.items():
            assert hasattr(experiment, "model")
            assert hasattr(experiment, "data_test")
            assert hasattr(experiment, "feature_cols")
            assert hasattr(experiment, "target_assignments")

    @pytest.mark.slow
    def test_predict_on_test_data(self, trained_study):
        """Test prediction on test data."""
        task_item = OctoPredict(
            study_path=trained_study,
            task_id=0,
            results_key="best",
        )

        # Test predict_test method
        predictions = task_item.predict_test()

        assert isinstance(predictions, pd.DataFrame)
        assert "prediction" in predictions.columns
        assert "prediction_std" in predictions.columns
        assert "n" in predictions.columns
        assert len(predictions) > 0

    @pytest.mark.slow
    def test_predict_proba_on_test_data(self, trained_study):
        """Test probability prediction on test data."""
        task_item = OctoPredict(
            study_path=trained_study,
            task_id=0,
            results_key="best",
        )

        # Test predict_proba_test method
        probabilities = task_item.predict_proba_test()

        assert isinstance(probabilities, pd.DataFrame)
        assert "probability" in probabilities.columns
        assert "probability_std" in probabilities.columns
        assert "n" in probabilities.columns
        assert len(probabilities) > 0

        # Verify probabilities are in valid range [0, 1]
        assert (probabilities["probability"] >= 0).all()
        assert (probabilities["probability"] <= 1).all()

    @pytest.mark.slow
    def test_performance_metrics_calculation(self, trained_study):
        """Test calculation of performance metrics on test data."""
        task_item = OctoPredict(
            study_path=trained_study,
            task_id=0,
            results_key="best",
        )

        metrics = ["AUCROC", "ACCBAL", "ACC", "F1", "AUCPR"]

        for _exp_id, experiment in task_item.experiments.items():
            for metric in metrics:
                performance = get_performance_from_model(
                    experiment.model,
                    experiment.data_test,
                    experiment.feature_cols,
                    metric,
                    experiment.target_assignments,
                    positive_class=1,
                )

                # Verify performance is a valid number
                assert isinstance(performance, (int, float))
                assert not np.isnan(performance)

                # Verify performance is in reasonable range
                if metric in ["AUCROC", "ACCBAL", "ACC", "F1", "AUCPR"]:
                    assert 0 <= performance <= 1

    @pytest.mark.slow
    def test_roc_curve_calculation(self, trained_study):
        """Test ROC curve calculation for classification."""
        task_item = OctoPredict(
            study_path=trained_study,
            task_id=0,
            results_key="best",
        )

        # Collect predictions from all experiments
        res_lst = []
        for _key, experiment in task_item.experiments.items():
            data_test = experiment.data_test
            feature_cols = experiment.feature_cols
            row_column = experiment.row_column
            target_col = list(experiment.target_assignments.values())[0]

            df = pd.DataFrame()
            df["row_id_col"] = data_test[row_column]
            df["prediction"] = experiment.model.predict(data_test[feature_cols])

            pred = experiment.model.predict_proba(data_test[feature_cols])
            if isinstance(pred, pd.DataFrame):
                df["probabilities"] = pred[1]
            elif isinstance(pred, np.ndarray):
                df["probabilities"] = pred[:, 1]

            df["target"] = data_test[target_col]
            res_lst.append(df)

        res_df = pd.concat(res_lst, axis=0)

        # Compute ROC curve and AUC
        fpr, tpr, _thresholds = roc_curve(res_df["target"], res_df["probabilities"], drop_intermediate=True)
        auc_roc = auc(fpr, tpr)

        # Verify ROC metrics
        assert isinstance(fpr, np.ndarray)
        assert isinstance(tpr, np.ndarray)
        assert 0 <= auc_roc <= 1
        assert len(fpr) == len(tpr)

    @pytest.mark.slow
    def test_feature_importance_permutation(self, trained_study):
        """Test permutation feature importance calculation."""
        task_item = OctoPredict(
            study_path=trained_study,
            task_id=0,
            results_key="best",
        )

        # Calculate permutation feature importance with reduced repeats for speed
        task_item.calculate_fi_test(
            fi_type="group_permutation",
            n_repeat=2,
            experiment_id=0,
        )

        # Verify results are stored
        assert len(task_item.results) > 0

        # Check for permutation FI results
        pfi_keys = [key for key in task_item.results if "permutation" in key]
        assert len(pfi_keys) > 0

        # Verify FI dataframe structure
        for key in pfi_keys:
            fi_df = task_item.results[key]
            assert isinstance(fi_df, pd.DataFrame)
            assert "feature" in fi_df.columns
            assert "importance" in fi_df.columns
            assert len(fi_df) > 0

    @pytest.mark.slow
    @pytest.mark.expensive
    def test_feature_importance_shap(self, trained_study):
        """Test SHAP feature importance calculation."""
        task_item = OctoPredict(
            study_path=trained_study,
            task_id=0,
            results_key="best",
        )

        # Calculate SHAP feature importance with kernel explainer
        task_item.calculate_fi_test(
            fi_type="shap",
            shap_type="kernel",
            experiment_id=0,
        )

        # Verify results are stored
        shap_keys = [key for key in task_item.results if "shap" in key]
        assert len(shap_keys) > 0

        # Verify SHAP FI dataframe structure
        for key in shap_keys:
            fi_df = task_item.results[key]
            assert isinstance(fi_df, pd.DataFrame)
            assert "feature" in fi_df.columns
            assert "importance" in fi_df.columns
            assert len(fi_df) > 0

    @pytest.mark.slow
    def test_experiment_info_attributes(self, trained_study):
        """Test that ExperimentInfo objects have correct attributes."""
        task_item = OctoPredict(
            study_path=trained_study,
            task_id=0,
            results_key="best",
        )

        for _exp_id, experiment in task_item.experiments.items():
            # Verify all required attributes exist
            assert hasattr(experiment, "id")
            assert hasattr(experiment, "model")
            assert hasattr(experiment, "data_traindev")
            assert hasattr(experiment, "data_test")
            assert hasattr(experiment, "feature_cols")
            assert hasattr(experiment, "row_column")
            assert hasattr(experiment, "target_assignments")
            assert hasattr(experiment, "target_metric")
            assert hasattr(experiment, "ml_type")
            assert hasattr(experiment, "feature_group_dict")
            assert hasattr(experiment, "positive_class")

            # Verify attribute types
            assert isinstance(experiment.id, int)
            assert isinstance(experiment.data_test, pd.DataFrame)
            assert isinstance(experiment.feature_cols, list)
            assert isinstance(experiment.target_assignments, dict)
            assert experiment.ml_type == "classification"

    @pytest.mark.slow
    def test_config_property(self, trained_study):
        """Test config property of OctoPredict."""
        task_item = OctoPredict(
            study_path=trained_study,
            task_id=0,
            results_key="best",
        )

        config = task_item.config

        # Verify config structure (now a dict from config.json)
        assert isinstance(config, dict)
        assert "ml_type" in config
        assert "workflow" in config
        assert config["ml_type"] == "classification"
        assert config["target_metric"] == "ACCBAL"

    @pytest.mark.slow
    def test_n_experiments_property(self, trained_study):
        """Test n_experiments property."""
        task_item = OctoPredict(
            study_path=trained_study,
            task_id=0,
            results_key="best",
        )

        n_experiments = task_item.n_experiments

        assert isinstance(n_experiments, int)
        assert n_experiments > 0
        # n_experiments returns the configured n_folds_outer (5 in this case)
        assert n_experiments == 5
        # With run_single_experiment_num=0, only 1 experiment is actually run
        assert len(task_item.experiments) == 1

    @pytest.mark.slow
    def test_predict_on_new_data(self, trained_study, classification_dataset):
        """Test prediction on new data."""
        task_item = OctoPredict(
            study_path=trained_study,
            task_id=0,
            results_key="best",
        )

        df, features = classification_dataset

        # Create new data (subset of original)
        new_data = df[features].head(10)

        # Test predict with return_df=False (returns numpy array)
        predictions_array = task_item.predict(new_data, return_df=False)
        assert isinstance(predictions_array, np.ndarray)
        assert len(predictions_array) == 10

        # Test predict with return_df=True (returns DataFrame)
        predictions_df = task_item.predict(new_data, return_df=True)
        assert isinstance(predictions_df, pd.DataFrame)
        assert len(predictions_df) == 10

    @pytest.mark.slow
    def test_predict_proba_on_new_data(self, trained_study, classification_dataset):
        """Test probability prediction on new data."""
        task_item = OctoPredict(
            study_path=trained_study,
            task_id=0,
            results_key="best",
        )

        df, features = classification_dataset
        new_data = df[features].head(10)

        # Test predict_proba with return_df=False
        proba_array = task_item.predict_proba(new_data, return_df=False)
        assert isinstance(proba_array, np.ndarray)
        assert len(proba_array) == 10

        # Test predict_proba with return_df=True
        proba_df = task_item.predict_proba(new_data, return_df=True)
        assert isinstance(proba_df, pd.DataFrame)
        assert len(proba_df) == 10

    @pytest.mark.slow
    def test_invalid_results_key(self, trained_study):
        """Test that invalid results_key raises appropriate error."""
        with pytest.raises(ValueError, match="Specified results key not found"):
            OctoPredict(
                study_path=trained_study,
                task_id=0,
                results_key="invalid_key",
            )

    @pytest.mark.slow
    def test_feature_importance_results_structure(self, trained_study):
        """Test that feature importance results have correct structure."""
        task_item = OctoPredict(
            study_path=trained_study,
            task_id=0,
            results_key="best",
        )

        # Calculate FI
        task_item.calculate_fi_test(
            fi_type="group_permutation",
            n_repeat=5,
            experiment_id=0,
        )

        # Get FI results
        fi_key = "fi_table_group_permutation_exp0"
        assert fi_key in task_item.results

        fi_df = task_item.results[fi_key]

        # Verify required columns
        required_columns = ["feature", "importance", "ci_low_95", "ci_high_95"]
        for col in required_columns:
            assert col in fi_df.columns

        # Verify data types
        assert fi_df["importance"].dtype in [np.float64, np.float32]
        assert all(fi_df["importance"] >= 0)

    @pytest.mark.slow
    def test_predict_return_formats(self, trained_study, classification_dataset):
        """Test predict and predict_proba with different return formats."""
        task_item = OctoPredict(
            study_path=trained_study,
            task_id=0,
            results_key="best",
        )

        df, features = classification_dataset
        new_data = df[features].head(10)

        # Test predict with return_df=True
        result_df = task_item.predict(new_data, return_df=True)
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 10
        assert "prediction" in result_df.columns

        # Test predict with return_df=False
        result_array = task_item.predict(new_data, return_df=False)
        assert isinstance(result_array, np.ndarray)
        assert len(result_array) == 10

        # Test predict_proba with return_df=True
        proba_df = task_item.predict_proba(new_data, return_df=True)
        assert isinstance(proba_df, pd.DataFrame)
        assert len(proba_df) == 10
        assert proba_df.columns.nlevels == 2

        # Test predict_proba with return_df=False
        proba_array = task_item.predict_proba(new_data, return_df=False)
        assert isinstance(proba_array, np.ndarray)
        assert len(proba_array) == 10
        assert proba_array.shape[1] >= 1

    @pytest.mark.slow
    def test_calculate_fi_permutation(self, trained_study, classification_dataset):
        """Test calculate_fi with fi_type='permutation'."""
        task_item = OctoPredict(
            study_path=trained_study,
            task_id=0,
            results_key="best",
        )

        df, features = classification_dataset
        # Include target column for calculate_fi
        new_data = df[[*features, "target"]].head(50)

        # Calculate permutation feature importance on new data with reduced repeats
        task_item.calculate_fi(
            data=new_data,
            n_repeat=2,
            fi_type="permutation",
        )

        # Verify results are stored for each experiment
        perm_keys = [key for key in task_item.results if "permutation" in key and "exp" in key]
        assert len(perm_keys) > 0

        # Verify structure of results
        for key in perm_keys:
            fi_df = task_item.results[key]
            assert isinstance(fi_df, pd.DataFrame)
            assert "feature" in fi_df.columns
            assert "importance" in fi_df.columns
            assert "ci_low_95" in fi_df.columns
            assert "ci_high_95" in fi_df.columns
            assert len(fi_df) == len(features)

        # Verify ensemble results also exist
        assert "fi_table_permutation_ensemble" in task_item.results

    @pytest.mark.slow
    @pytest.mark.expensive
    def test_calculate_fi_shap_kernel(self, trained_study, classification_dataset):
        """Test calculate_fi with fi_type='shap' and shap_type='kernel'."""
        task_item = OctoPredict(
            study_path=trained_study,
            task_id=0,
            results_key="best",
        )

        df, features = classification_dataset
        # Include target column for calculate_fi
        new_data = df[[*features, "target"]].head(20)  # Use smaller sample for SHAP

        # Calculate SHAP feature importance on new data
        task_item.calculate_fi(
            data=new_data,
            fi_type="shap",
            shap_type="kernel",
        )

        # Verify results are stored for each experiment
        shap_keys = [key for key in task_item.results if "shap" in key and "exp" in key]
        assert len(shap_keys) > 0

        # Verify structure of results
        for key in shap_keys:
            fi_df = task_item.results[key]
            assert isinstance(fi_df, pd.DataFrame)
            assert "feature" in fi_df.columns
            assert "importance" in fi_df.columns
            assert len(fi_df) == len(features)
            # SHAP importances should be non-negative after abs()
            assert all(fi_df["importance"] >= 0)

        # Verify ensemble results also exist
        assert "fi_table_shap_ensemble" in task_item.results
