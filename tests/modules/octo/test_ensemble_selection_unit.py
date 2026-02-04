"""Unit tests for EnSel (Ensemble Selection) individual methods."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from upath import UPath

from octopus.modules.octo.bag import Bag
from octopus.modules.octo.enssel import EnSel
from octopus.modules.octo.training import Training

# Utility functions for creating mock data and bags


def create_mock_training(training_id, performance_dev, performance_test, n_samples=100):
    """Create a mock Training object with controlled performance."""
    np.random.seed(42 + hash(training_id) % 1000)  # Deterministic but varied

    # Generate synthetic data
    X = np.random.randn(n_samples, 4)
    y = np.random.randn(n_samples)
    row_ids = np.arange(n_samples)

    # Split data
    n_train = int(0.6 * n_samples)
    n_dev = int(0.2 * n_samples)

    train_idx = slice(0, n_train)
    dev_idx = slice(n_train, n_train + n_dev)
    test_idx = slice(n_train + n_dev, n_samples)

    # Create dataframes
    train_df = pd.DataFrame(X[train_idx], columns=[f"feature_{i}" for i in range(4)])
    train_df["row_id_col"] = row_ids[train_idx]
    train_df["target"] = y[train_idx]

    dev_df = pd.DataFrame(X[dev_idx], columns=[f"feature_{i}" for i in range(4)])
    dev_df["row_id_col"] = row_ids[dev_idx]
    dev_df["target"] = y[dev_idx]

    test_df = pd.DataFrame(X[test_idx], columns=[f"feature_{i}" for i in range(4)])
    test_df["row_id_col"] = row_ids[test_idx]
    test_df["target"] = y[test_idx]

    # Create controlled predictions to achieve target performance
    # For MAE, we need |prediction - target| = performance_dev/performance_test
    # Use systematic offset to achieve precise MAE
    pred_train = train_df["target"] + np.full(len(train_df), 0.1)  # Small constant error
    pred_dev = dev_df["target"] + np.full(len(dev_df), performance_dev)  # Exact MAE control
    pred_test = test_df["target"] + np.full(len(test_df), performance_test)  # Exact MAE control

    predictions = {
        "train": pd.DataFrame(
            {"row_id_col": train_df["row_id_col"], "prediction": pred_train, "target": train_df["target"]}
        ),
        "dev": pd.DataFrame({"row_id_col": dev_df["row_id_col"], "prediction": pred_dev, "target": dev_df["target"]}),
        "test": pd.DataFrame(
            {"row_id_col": test_df["row_id_col"], "prediction": pred_test, "target": test_df["target"]}
        ),
    }

    training = Training(
        training_id=training_id,
        ml_type="regression",
        target_assignments={"default": "target"},
        feature_cols=[f"feature_{i}" for i in range(4)],
        row_column="row_id_col",
        data_train=train_df,
        data_dev=dev_df,
        data_test=test_df,
        target_metric="MAE",
        max_features=4,
        feature_groups={},
        config_training={"ml_model_type": "RidgeRegressor"},
    )

    training.predictions = predictions
    training.model = LinearRegression()

    return training


def create_mock_bag(log_dir, bag_id, target_dev_mae, target_test_mae, n_trainings=3, exact_performance=False):
    """Create a mock Bag with controlled performance."""
    trainings = []
    for i in range(n_trainings):
        training_id = f"{bag_id}_{i}"
        if exact_performance:
            # Use exact performance for testing identical cases
            dev_mae = target_dev_mae
            test_mae = target_test_mae
        else:
            # Add small variation to individual trainings
            dev_mae = target_dev_mae + np.random.normal(0, 0.05)
            test_mae = target_test_mae + np.random.normal(0, 0.05)
        training = create_mock_training(training_id, dev_mae, test_mae)
        trainings.append(training)

    bag = Bag(
        bag_id=bag_id,
        trainings=trainings,
        target_assignments={"default": "target"},
        row_column="row_id_col",
        target_metric="MAE",
        ml_type="regression",
        parallel_execution=False,
        num_workers=1,
        log_dir=log_dir,
    )

    bag.train_status = True
    return bag


def create_mock_trial_directory(
    tmp_path: Path, bag_performances: list[tuple[str, float, float]], exact_performance: bool = False
) -> UPath:
    """Create directory with mock trial bags.

    Args:
        tmp_path: pytest tmp_path fixture
        bag_performances: list of tuples (bag_id, dev_mae, test_mae)
        exact_performance: if True, use exact performance values

    Returns:
        Path to trials directory
    """
    trials_path = UPath(tmp_path / "trials")
    trials_path.mkdir()

    for i, (bag_id, dev_mae, test_mae) in enumerate(bag_performances):
        bag = create_mock_bag(trials_path, bag_id, dev_mae, test_mae, exact_performance=exact_performance)
        bag_file = trials_path / f"trial{i}_bag.pkl"
        bag.to_pickle(bag_file)

    return trials_path


def create_partial_ensel(trials_path, target_metric="MAE", methods_to_run=None):
    """Create EnSel instance that only runs specified methods."""

    class PartialEnSel(EnSel):
        def __attrs_post_init__(self):
            self.bags = {}
            self.start_ensemble = {}
            self.optimized_ensemble = {}

            if methods_to_run is None or "_collect_trials" in methods_to_run:
                self._collect_trials()
            if methods_to_run is None or "_create_model_table" in methods_to_run:
                self._create_model_table()
            if methods_to_run is None or "_ensemble_scan" in methods_to_run:
                self._ensemble_scan()
            if methods_to_run is None or "_ensemble_optimization" in methods_to_run:
                self._ensemble_optimization()

    return PartialEnSel(
        target_metric=target_metric,
        target_assignments={"default": "target"},
        path_trials=trials_path,
        max_n_iterations=10,
        row_column="row_id_col",
    )


# Tests for EnSel._collect_trials() method


def test_collect_trials_basic(tmp_path):
    """Test basic trial collection from directory."""
    # Create mock trials with known performance
    bag_performances = [("model_a", 1.0, 1.1), ("model_b", 1.5, 1.4), ("model_c", 2.0, 1.9)]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances)

    ensel = create_partial_ensel(trials_path, methods_to_run=["_collect_trials"])

    # Verify bags were collected correctly
    assert len(ensel.bags) == 3

    # Check that all expected keys are present in bags
    expected_keys = {"id", "performance", "predictions", "n_features_used_mean"}
    for _bag_path, bag_data in ensel.bags.items():
        assert set(bag_data.keys()) == expected_keys
        assert isinstance(bag_data["performance"], dict)
        assert isinstance(bag_data["predictions"], dict)
        assert "dev_pool" in bag_data["performance"]
        assert "test_pool" in bag_data["performance"]


# Tests for EnSel._create_model_table() method


def test_create_model_table_sorting_minimize(tmp_path):
    """Test model table creation and sorting for minimize metrics (MAE)."""
    bag_performances = [
        ("model_good", 1.0, 1.1),  # Best performance
        ("model_medium", 1.5, 1.4),  # Medium performance
        ("model_poor", 2.0, 1.9),  # Worst performance
    ]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances)

    ensel = create_partial_ensel(trials_path, methods_to_run=["_collect_trials", "_create_model_table"])

    # Verify model table structure
    assert len(ensel.model_table) == 3
    expected_columns = {"id", "dev_pool", "test_pool", "dev_avg", "test_avg", "n_features_used_mean", "path"}
    assert set(ensel.model_table.columns) == expected_columns

    # Verify sorting (ascending for minimize metrics)
    dev_scores = ensel.model_table["dev_pool"].values
    assert all(dev_scores[i] <= dev_scores[i + 1] for i in range(len(dev_scores) - 1))

    # Best model should be first
    assert ensel.model_table.iloc[0]["id"] == "model_good"


def test_create_model_table_identical_performance(tmp_path):
    """Test handling of models with identical performance."""
    bag_performances = [("model_a", 1.5, 1.5), ("model_b", 1.5, 1.5), ("model_c", 1.5, 1.5)]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances, exact_performance=True)

    ensel = create_partial_ensel(trials_path, methods_to_run=["_collect_trials", "_create_model_table"])

    # Should handle identical performance
    assert len(ensel.model_table) == 3
    # All should have same dev_pool score
    dev_scores = ensel.model_table["dev_pool"].values
    assert np.all(dev_scores == 1.5)


# Tests for EnSel._ensemble_models() core algorithm


def test_ensemble_models_single_bag(tmp_path):
    """Test ensemble calculation with single model."""
    bag_performances = [("single_model", 1.0, 1.1)]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances)

    ensel = create_partial_ensel(trials_path, methods_to_run=["_collect_trials"])

    # Test ensemble with single bag
    bag_keys = list(ensel.bags.keys())
    scores = ensel._ensemble_models(bag_keys)

    # Verify score structure
    assert isinstance(scores, dict)
    expected_keys = {"dev_pool", "test_pool"}
    assert expected_keys.issubset(set(scores.keys()))
    assert isinstance(scores["dev_pool"], (int | float))
    assert isinstance(scores["test_pool"], (int | float))


# Tests for EnSel._ensemble_scan() method


def test_ensemble_scan_creates_scan_table(tmp_path):
    """Test that ensemble scan creates and populates scan_table."""
    bag_performances = [
        ("model_poor", 3.0, 3.1),  # Worst performance
        ("model_best", 1.0, 1.1),  # Best performance
        ("model_medium", 2.0, 2.1),  # Medium performance
    ]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances, exact_performance=True)

    ensel = create_partial_ensel(
        trials_path, methods_to_run=["_collect_trials", "_create_model_table", "_ensemble_scan"]
    )

    # Verify scan_table was created
    assert hasattr(ensel, "scan_table")

    # Should have one row for each model (testing ensemble sizes 1, 2, 3)
    assert len(ensel.scan_table) == 3


# Tests for EnSel._ensemble_optimization() method


def test_ensemble_optimization_creates_ensembles(tmp_path):
    """Test that optimization creates both start and optimized ensembles."""
    bag_performances = [("model_a", 2.0, 2.1), ("model_b", 1.8, 1.9), ("model_c", 2.2, 2.0)]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances, exact_performance=True)

    ensel = create_partial_ensel(
        trials_path,
        methods_to_run=["_collect_trials", "_create_model_table", "_ensemble_scan", "_ensemble_optimization"],
    )

    # Both start and optimized ensembles should exist
    assert hasattr(ensel, "start_ensemble")
    assert hasattr(ensel, "optimized_ensemble")
    assert len(ensel.start_ensemble) >= 1
    assert len(ensel.optimized_ensemble) >= 1

    # All weights should be positive integers
    for weight in ensel.start_ensemble.values():
        assert isinstance(weight, int)
        assert weight >= 1

    for weight in ensel.optimized_ensemble.values():
        assert isinstance(weight, int)
        assert weight >= 1


def test_ensemble_optimization_single_model_case(tmp_path):
    """Test optimization behavior when only one model exists."""
    bag_performances = [("single_model", 1.5, 1.6)]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances)

    ensel = create_partial_ensel(
        trials_path,
        methods_to_run=["_collect_trials", "_create_model_table", "_ensemble_scan", "_ensemble_optimization"],
    )

    # Both ensembles should have the single model
    assert len(ensel.start_ensemble) == 1
    assert len(ensel.optimized_ensemble) == 1

    # Should maintain the same model
    start_model = list(ensel.start_ensemble.keys())[0]
    opt_model = list(ensel.optimized_ensemble.keys())[0]
    assert start_model == opt_model
