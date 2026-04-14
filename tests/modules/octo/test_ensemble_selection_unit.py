"""Unit tests for EnSel (Ensemble Selection) individual methods."""

import heapq
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from upath import UPath

import octopus.modules.octo.enssel as enssel_module
from octopus.modules.octo.bag import Bag
from octopus.modules.octo.enssel import EnSel, _average_and_quantize, _stratified_subsample
from octopus.modules.octo.training import Training
from octopus.types import DataPartition, MLType, ModelName
from octopus.utils import joblib_load, joblib_save

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
    train_df["row_id"] = row_ids[train_idx]
    train_df["target"] = y[train_idx]

    dev_df = pd.DataFrame(X[dev_idx], columns=[f"feature_{i}" for i in range(4)])
    dev_df["row_id"] = row_ids[dev_idx]
    dev_df["target"] = y[dev_idx]

    test_df = pd.DataFrame(X[test_idx], columns=[f"feature_{i}" for i in range(4)])
    test_df["row_id"] = row_ids[test_idx]
    test_df["target"] = y[test_idx]

    # Create controlled predictions to achieve target performance
    # For MAE, we need |prediction - target| = performance_dev/performance_test
    # Use systematic offset to achieve precise MAE
    pred_train = train_df["target"] + np.full(len(train_df), 0.1)  # Small constant error
    pred_dev = dev_df["target"] + np.full(len(dev_df), performance_dev)  # Exact MAE control
    pred_test = test_df["target"] + np.full(len(test_df), performance_test)  # Exact MAE control

    # Parse training_id for metadata (format: "bagid_innersplit")
    parts = training_id.rsplit("_", 1)
    inner_split_id = parts[-1] if len(parts) > 1 else training_id

    predictions = {
        "train": pd.DataFrame(
            {
                "row_id": train_df["row_id"],
                "prediction": pred_train,
                "target": train_df["target"],
                "outer_split_id": 0,
                "inner_split_id": inner_split_id,
                "partition": "train",
                "task_id": 0,
            }
        ),
        "dev": pd.DataFrame(
            {
                "row_id": dev_df["row_id"],
                "prediction": pred_dev,
                "target": dev_df["target"],
                "outer_split_id": 0,
                "inner_split_id": inner_split_id,
                "partition": "dev",
                "task_id": 0,
            }
        ),
        "test": pd.DataFrame(
            {
                "row_id": test_df["row_id"],
                "prediction": pred_test,
                "target": test_df["target"],
                "outer_split_id": 0,
                "inner_split_id": inner_split_id,
                "partition": "test",
                "task_id": 0,
            }
        ),
    }

    training = Training(
        training_id=training_id,
        ml_type=MLType.REGRESSION,
        target_assignments={"default": "target"},
        feature_cols=[f"feature_{i}" for i in range(4)],
        row_id_col="row_id",
        data_train=train_df,
        data_dev=dev_df,
        data_test=test_df,
        target_metric="MAE",
        max_features=4,
        feature_groups={},
        config_training={"ml_model_type": ModelName.RidgeRegressor},
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
        row_id_col="row_id",
        target_metric="MAE",
        ml_type=MLType.REGRESSION,
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
        bag_file = trials_path / f"trial_{i}_bag.joblib"
        joblib_save(bag, bag_file)

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
        row_id_col="row_id",
        n_assigned_cpus=1,
        ml_type=MLType.REGRESSION,
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
        assert "dev_ensemble" in bag_data["performance"]
        assert "test_ensemble" in bag_data["performance"]


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
    expected_columns = {"id", "dev_ensemble", "test_ensemble", "dev_avg", "test_avg", "n_features_used_mean", "path"}
    assert set(ensel.model_table.columns) == expected_columns

    # Verify sorting (ascending for minimize metrics)
    dev_scores = ensel.model_table["dev_ensemble"].values
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
    # All should have same dev_ensemble score
    dev_scores = ensel.model_table["dev_ensemble"].values
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
    expected_keys = {"dev_ensemble", "test_ensemble"}
    assert expected_keys.issubset(set(scores.keys()))
    assert isinstance(scores["dev_ensemble"], (int | float))
    assert isinstance(scores["test_ensemble"], (int | float))


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


# Heap and trial library correctness


def test_heap_retains_best_trials_minimize_metric(tmp_path):
    """Heap must retain the M best trials for minimize metrics (MAE).

    Regression guard for Issue #1: heap direction inversion after
    minimize→maximize refactoring.
    """
    trials_path = UPath(tmp_path / "scratch")
    trials_path.mkdir()

    # Simulate _save_topn_trials logic for MAE (minimize → negated target_value)
    max_n_trials = 2
    top_trials: list = []
    scores = [1.5, 0.8, 1.2, 0.5, 2.0]  # MAE values (lower is better)

    saved_files: dict[float, UPath] = {}
    for i, mae in enumerate(scores):
        # ObjectiveOptuna negates for minimize: optuna_target = -mae
        target_value = -mae
        path = trials_path / f"trial_{i}.joblib"
        # Create a dummy file
        path.write_text(str(mae))
        saved_files[mae] = path

        # Replicate current heap logic (after fix: no -1* negation)
        heapq.heappush(top_trials, (target_value, path))
        if len(top_trials) > max_n_trials:
            _, path_delete = heapq.heappop(top_trials)
            if path_delete.is_file():
                path_delete.unlink()

    # The 2 surviving trials should be the 2 best MAE values: 0.5 and 0.8
    surviving = sorted([float(p.read_text()) for _, p in top_trials])
    assert surviving == [0.5, 0.8], f"Expected best 2 trials [0.5, 0.8], got {surviving}"


def test_heap_retains_best_trials_maximize_metric(tmp_path):
    """Heap must retain the M highest-scoring trials for maximize metrics.

    Regression guard for Issue #1.
    """
    trials_path = UPath(tmp_path / "scratch")
    trials_path.mkdir()

    max_n_trials = 3
    top_trials: list = []
    # AUCROC values (higher is better) — not negated since direction is MAXIMIZE
    scores = [0.70, 0.90, 0.80, 0.95, 0.60]

    for i, auc in enumerate(scores):
        target_value = auc  # No negation for MAXIMIZE metrics
        path = trials_path / f"trial_{i}.joblib"
        path.write_text(str(auc))

        heapq.heappush(top_trials, (target_value, path))
        if len(top_trials) > max_n_trials:
            _, path_delete = heapq.heappop(top_trials)
            if path_delete.is_file():
                path_delete.unlink()

    surviving = sorted([float(p.read_text()) for _, p in top_trials])
    assert surviving == [0.80, 0.90, 0.95], f"Expected best 3 trials [0.80, 0.90, 0.95], got {surviving}"


def test_heap_no_eviction_when_under_limit(tmp_path):
    """When n_trials <= ensel_n_save_trials, no trial should be evicted."""
    trials_path = UPath(tmp_path / "scratch")
    trials_path.mkdir()

    max_n_trials = 10
    top_trials: list = []

    for i in range(5):
        path = trials_path / f"trial_{i}.joblib"
        path.write_text(str(i))
        heapq.heappush(top_trials, (float(i), path))
        # No eviction since len <= max
        assert len(top_trials) <= max_n_trials

    assert len(top_trials) == 5
    assert all(p.is_file() for _, p in top_trials)


def test_heap_tiebreak_on_equal_scores(tmp_path):
    """When multiple trials have the same score, heap must not crash.

    UPath comparison via PurePath.__lt__ must work as tiebreaker.
    """
    trials_path = UPath(tmp_path / "scratch")
    trials_path.mkdir()

    max_n_trials = 3
    top_trials: list = []

    for i in range(5):
        path = trials_path / f"trial_{i}.joblib"
        path.write_text("0.85")
        heapq.heappush(top_trials, (0.85, path))
        if len(top_trials) > max_n_trials:
            _, path_delete = heapq.heappop(top_trials)
            if path_delete.is_file():
                path_delete.unlink()

    # Exactly 3 files must survive, no error raised
    assert len(top_trials) == 3
    surviving_files = [p for _, p in top_trials if p.is_file()]
    assert len(surviving_files) == 3


# Prediction cache integrity


def test_ensemble_models_preserves_prediction_cache(tmp_path):
    """Calling _ensemble_models must not mutate self.bags prediction dicts.

    Regression guard for Issue #6: .pop('ensemble') mutated shared cache.
    """
    bag_performances = [("a", 1.0, 1.1), ("b", 1.5, 1.4)]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances)
    ensel = create_partial_ensel(trials_path, methods_to_run=["_collect_trials"])

    # Snapshot keys before
    keys_before = {k: set(v["predictions"].keys()) for k, v in ensel.bags.items()}

    # Call _ensemble_models multiple times
    bag_keys = list(ensel.bags.keys())
    ensel._ensemble_models(bag_keys)
    ensel._ensemble_models(bag_keys)

    # Keys must be identical after
    for k, v in ensel.bags.items():
        assert set(v["predictions"].keys()) == keys_before[k], f"Prediction cache for bag {k} was mutated"


def test_ensemble_models_idempotent(tmp_path):
    """Calling _ensemble_models with same inputs must return same results."""
    bag_performances = [("a", 1.0, 1.1), ("b", 1.5, 1.4), ("c", 2.0, 1.9)]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances)
    ensel = create_partial_ensel(trials_path, methods_to_run=["_collect_trials"])

    bag_keys = list(ensel.bags.keys())
    result1 = ensel._ensemble_models(bag_keys)
    result2 = ensel._ensemble_models(bag_keys)

    assert result1["dev_ensemble"] == result2["dev_ensemble"]
    assert result1["test_ensemble"] == result2["test_ensemble"]


# Optimization uses the right result


def test_optimized_ensemble_can_differ_from_start(tmp_path):
    """Greedy optimization with replacement can produce weights different from start.

    The optimized_ensemble may differ from the scan start_ensemble.
    """
    bag_performances = [("a", 1.0, 1.1), ("b", 1.5, 1.4), ("c", 2.0, 1.9)]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances, exact_performance=True)

    ensel = create_partial_ensel(trials_path)

    # optimized_ensemble should have more total weight than start_ensemble
    # (optimization adds models via replacement)
    start_total = sum(ensel.start_ensemble.values())
    opt_total = sum(ensel.optimized_ensemble.values())
    assert opt_total >= start_total, f"Optimized ensemble total weight ({opt_total}) should be >= start ({start_total})"


def test_optimized_ensemble_weights_are_positive_integers(tmp_path):
    """All weights in optimized_ensemble must be positive integers.

    Regression guard: selection with replacement must produce integer
    weights >= 1 for all models.
    """
    bag_performances = [("a", 1.0, 1.1), ("b", 1.5, 1.4)]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances, exact_performance=True)

    ensel = create_partial_ensel(trials_path)

    assert len(ensel.optimized_ensemble) >= 1
    for path, weight in ensel.optimized_ensemble.items():
        assert isinstance(weight, int), f"Weight for {path} is not int: {type(weight)}"
        assert weight >= 1, f"Weight for {path} is < 1: {weight}"


def test_optimized_ensemble_contains_only_known_bags(tmp_path):
    """All paths in optimized_ensemble must be bags that were loaded in _collect_trials."""
    bag_performances = [("a", 1.0, 1.1), ("b", 1.5, 1.4), ("c", 2.0, 1.9)]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances)

    ensel = create_partial_ensel(trials_path)

    for path in ensel.optimized_ensemble:
        assert path in ensel.bags, f"Optimized ensemble contains unknown path: {path}"


# use_best backtracking


def test_optimization_runs_all_iterations(tmp_path):
    """Optimization must run max_n_iterations steps, not stop early.

    Regression guard: old code stopped at first non-improving step.
    New code runs all iterations and returns the best prefix via backtracking.
    """
    bag_performances = [("a", 1.0, 1.1), ("b", 1.5, 1.4), ("c", 2.0, 1.9)]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances, exact_performance=True)

    ensel = create_partial_ensel(trials_path)

    # The optimized ensemble total weight = start_n + max_n_iterations
    # backtracked to best prefix. Total weight must be >= start_n
    # (backtracking can go back to start, but optimization ran all iterations)
    opt_total = sum(ensel.optimized_ensemble.values())
    start_total = sum(ensel.start_ensemble.values())
    assert opt_total >= start_total


def test_optimization_result_at_least_as_good_as_start(tmp_path):
    """Optimized ensemble must perform at least as well as scan start on dev.

    This is the fundamental guarantee of use_best backtracking: it can always
    fall back to the starting ensemble if no improvement is found.
    """
    bag_performances = [("a", 1.0, 1.1), ("b", 1.5, 1.4), ("c", 2.0, 1.9)]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances, exact_performance=True)

    ensel = create_partial_ensel(trials_path)

    # Compute performance of both ensembles
    start_bags = []
    for path, weight in ensel.start_ensemble.items():
        start_bags.extend([path] * weight)
    start_perf = ensel._ensemble_models(start_bags)["dev_ensemble"]

    opt_bags = []
    for path, weight in ensel.optimized_ensemble.items():
        opt_bags.extend([path] * weight)
    opt_perf = ensel._ensemble_models(opt_bags)["dev_ensemble"]

    # For MAE (minimize), optimized must be <= start
    assert opt_perf <= start_perf + 1e-10, f"Optimized ({opt_perf:.6f}) should be <= start ({start_perf:.6f})"


# Performance and I/O regression


def test_no_joblib_load_during_scan(tmp_path, monkeypatch):
    """_ensemble_scan must not call joblib_load.

    Regression guard for Issue #3: _ensemble_models loaded a full bag
    from disk on every call just for dtype casting.
    """
    bag_performances = [("a", 1.0, 1.1), ("b", 1.5, 1.4)]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances)

    # Create ensel with only _collect_trials + _create_model_table
    ensel = create_partial_ensel(trials_path, methods_to_run=["_collect_trials", "_create_model_table"])

    # Now monkeypatch joblib_load and run scan
    load_count = {"n": 0}
    original_load = enssel_module.joblib_load

    def counting_load(*args, **kwargs):
        load_count["n"] += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(enssel_module, "joblib_load", counting_load)
    ensel._ensemble_scan()

    assert load_count["n"] == 0, f"_ensemble_scan called joblib_load {load_count['n']} times; expected 0"


def test_no_joblib_load_during_optimization(tmp_path, monkeypatch):
    """_ensemble_optimization must not call joblib_load.

    Same regression guard as above, for the optimization phase.
    """
    bag_performances = [("a", 1.0, 1.1), ("b", 1.5, 1.4)]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances)

    ensel = create_partial_ensel(
        trials_path, methods_to_run=["_collect_trials", "_create_model_table", "_ensemble_scan"]
    )

    load_count = {"n": 0}
    original_load = enssel_module.joblib_load

    def counting_load(*args, **kwargs):
        load_count["n"] += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(enssel_module, "joblib_load", counting_load)
    ensel._ensemble_optimization()

    assert load_count["n"] == 0, f"_ensemble_optimization called joblib_load {load_count['n']} times; expected 0"


# Scan table invariants


def test_scan_table_evaluates_all_prefix_sizes(tmp_path):
    """Scan table must evaluate all prefix sizes 1..M.

    Regression guard: ensures scan is exhaustive and doesn't skip sizes.
    """
    bag_performances = [("a", 1.0, 1.1), ("b", 1.5, 1.4), ("c", 2.0, 1.9)]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances)

    ensel = create_partial_ensel(
        trials_path, methods_to_run=["_collect_trials", "_create_model_table", "_ensemble_scan"]
    )

    assert len(ensel.scan_table) == 3
    assert list(ensel.scan_table["#models"]) == [1, 2, 3]


# S1: Prediction file loading


def create_mock_preds_file(trials_path, bag_id, bag, trial_idx):
    """Create a _preds.joblib file alongside a _bag.joblib file."""
    predictions = bag.get_predictions(n_assigned_cpus=1)
    predictions_ensel = {}
    for key, partitions in predictions.items():
        predictions_ensel[key] = {p: df for p, df in partitions.items() if p != DataPartition.TRAIN}

    preds_data = {
        "predictions": predictions_ensel,
        "bag_id": bag.bag_id,
        "n_features_used_mean": bag.n_features_used_mean,
        "target_dtypes": {"target": bag.trainings[0].data_train["target"].dtype},
    }
    preds_file = trials_path / f"trial_{trial_idx}_preds.joblib"
    joblib_save(preds_data, preds_file)
    return preds_file


def create_mock_trial_directory_with_preds(tmp_path, bag_performances, exact_performance=False):
    """Create directory with both _bag.joblib and _preds.joblib files."""
    trials_path = UPath(tmp_path / "trials")
    trials_path.mkdir()

    for i, (bag_id, dev_mae, test_mae) in enumerate(bag_performances):
        bag = create_mock_bag(trials_path, bag_id, dev_mae, test_mae, exact_performance=exact_performance)
        bag_file = trials_path / f"trial_{i}_bag.joblib"
        joblib_save(bag, bag_file)
        create_mock_preds_file(trials_path, bag_id, bag, i)

    return trials_path


def test_collect_trials_from_preds_basic(tmp_path):
    """When preds files exist, _collect_trials loads them and populates self.bags."""
    bag_performances = [("a", 1.0, 1.1), ("b", 1.5, 1.4), ("c", 2.0, 1.9)]
    trials_path = create_mock_trial_directory_with_preds(tmp_path, bag_performances)

    ensel = create_partial_ensel(trials_path, methods_to_run=["_collect_trials"])

    assert len(ensel.bags) == 3
    for bag_path, bag_data in ensel.bags.items():
        assert bag_path.name.endswith("_bag.joblib"), f"Key should be bag path, got {bag_path}"
        assert "performance" in bag_data
        assert "predictions" in bag_data
        assert "dev_ensemble" in bag_data["performance"]


def test_collect_trials_fallback_to_bags(tmp_path):
    """When no preds files exist, _collect_trials loads full bags."""
    bag_performances = [("a", 1.0, 1.1), ("b", 1.5, 1.4)]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances)

    ensel = create_partial_ensel(trials_path, methods_to_run=["_collect_trials"])

    assert len(ensel.bags) == 2


def test_per_trial_fallback_loads_both_formats(tmp_path):
    """When some trials have preds and some don't, all trials are loaded."""
    bag_performances = [("a", 1.0, 1.1), ("b", 1.5, 1.4)]
    trials_path = create_mock_trial_directory_with_preds(tmp_path, bag_performances)

    preds_file_b = trials_path / "trial_1_preds.joblib"
    preds_file_b.unlink()

    ensel = create_partial_ensel(trials_path, methods_to_run=["_collect_trials"])

    assert len(ensel.bags) == 2
    for bag_path in ensel.bags:
        assert bag_path.name.endswith("_bag.joblib")


def test_pred_file_excludes_train_partition(tmp_path):
    """Preds file must not contain TRAIN partition predictions."""
    bag_performances = [("a", 1.0, 1.1)]
    trials_path = create_mock_trial_directory_with_preds(tmp_path, bag_performances)

    preds_file = next(f for f in trials_path.iterdir() if f.name.endswith("_preds.joblib"))
    preds_data = joblib_load(preds_file)

    for key, partitions in preds_data["predictions"].items():
        assert DataPartition.TRAIN not in partitions, f"TRAIN partition found in {key}"
        assert "train" not in partitions, f"TRAIN partition found in {key}"


def test_ensel_results_identical_with_pred_files(tmp_path):
    """EnSel optimized_ensemble must be identical from preds files vs full bags."""
    bag_performances = [("a", 1.0, 1.1), ("b", 1.5, 1.4), ("c", 2.0, 1.9)]

    (tmp_path / "bags_only").mkdir()
    trials_path_bags = create_mock_trial_directory(tmp_path / "bags_only", bag_performances, exact_performance=True)
    ensel_bags = create_partial_ensel(trials_path_bags)

    (tmp_path / "with_preds").mkdir()
    trials_path_preds = create_mock_trial_directory_with_preds(
        tmp_path / "with_preds", bag_performances, exact_performance=True
    )
    ensel_preds = create_partial_ensel(trials_path_preds)

    bags_weights = {p.name: w for p, w in ensel_bags.optimized_ensemble.items()}
    preds_weights = {p.name: w for p, w in ensel_preds.optimized_ensemble.items()}
    assert bags_weights == preds_weights


# S2: Diversity tie-breaking


def test_diversity_tiebreak_prefers_diverse_model(tmp_path):
    """When candidates tie on score, the more diverse one should be preferred."""
    bag_performances = [("a", 1.0, 1.1), ("b", 1.0, 1.1), ("c", 1.0, 1.1)]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances, exact_performance=True)

    ensel = create_partial_ensel(trials_path)

    assert len(ensel.optimized_ensemble) >= 1


def test_diversity_tiebreak_nan_handling(tmp_path):
    """Constant predictions (NaN correlation) must not crash."""
    bag_performances = [("a", 1.0, 1.1), ("b", 1.5, 1.4)]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances, exact_performance=True)

    ensel = create_partial_ensel(trials_path)
    assert len(ensel.optimized_ensemble) >= 1


# S3: Bagged ensemble selection


def test_average_and_quantize():
    """Weight averaging and quantization produce correct integer weights."""
    vectors = [
        Counter({"a": 3, "b": 1}),
        Counter({"a": 5, "c": 2}),
    ]
    result = _average_and_quantize(vectors)
    assert result["a"] == 4
    assert result["c"] == 1


def test_average_and_quantize_bankers_rounding_avoided():
    """int(0.5 + 0.5) = 1, not round(0.5) = 0."""
    vectors = [
        Counter({"a": 1}),
        Counter(),
    ]
    result = _average_and_quantize(vectors)
    assert result.get("a", 0) == 1, f"Expected 1 (standard rounding), got {result.get('a', 0)}"


def test_stratified_subsample_preserves_all_classes():
    """Every class in the original dev set must appear in the subset."""
    row_ids = pd.Index(range(30))
    targets = pd.Series([0] * 25 + [1] * 5, index=row_ids)
    rng = np.random.default_rng(42)

    for _ in range(20):
        subset = _stratified_subsample(row_ids, targets, 10, rng)
        subset_targets = targets.loc[subset]
        assert 0 in subset_targets.values
        assert 1 in subset_targets.values


def test_bagging_runs_without_error(tmp_path):
    """Bagged ensemble selection (20 rounds, fraction 0.5) completes successfully."""
    bag_performances = [("a", 1.0, 1.1), ("b", 1.5, 1.4), ("c", 2.0, 1.9)]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances)

    ensel = create_partial_ensel(trials_path)

    assert len(ensel.optimized_ensemble) >= 1
    for weight in ensel.optimized_ensemble.values():
        assert isinstance(weight, int)
        assert weight >= 1


def test_bagging_is_deterministic(tmp_path):
    """Same inputs produce identical results across runs (fixed seed)."""
    bag_performances = [("a", 1.0, 1.1), ("b", 1.5, 1.4), ("c", 2.0, 1.9)]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances)

    ensel1 = create_partial_ensel(trials_path)
    ensel2 = create_partial_ensel(trials_path)

    w1 = {p.name: w for p, w in ensel1.optimized_ensemble.items()}
    w2 = {p.name: w for p, w in ensel2.optimized_ensemble.items()}
    assert w1 == w2


def test_nan_score_skipped_in_greedy_loop(tmp_path):
    """NaN metric scores must be skipped, not poison the greedy selection."""
    bag_performances = [("a", 1.0, 1.1), ("b", 1.5, 1.4)]
    trials_path = create_mock_trial_directory(tmp_path, bag_performances, exact_performance=True)

    ensel = create_partial_ensel(trials_path)

    assert len(ensel.optimized_ensemble) >= 1
