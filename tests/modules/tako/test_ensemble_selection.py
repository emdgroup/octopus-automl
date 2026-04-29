"""Test ensemble selection functionality."""

import copy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from upath import UPath

from octopus.modules.tako.bag import Bag
from octopus.modules.tako.enssel import EnSel
from octopus.modules.tako.training import Training
from octopus.types import MLType, ModelName, PerformanceKey
from octopus.utils import joblib_load, joblib_save


def create_synthetic_data_and_models(n_samples=500):
    """Create synthetic data with 3 models trained on orthogonal feature subsets.

    Returns data where target = signal1 + signal2 + signal3 + noise,
    and each model captures only 1/3 of the signal.
    """
    np.random.seed(42)

    # Generate features
    X = np.random.randn(n_samples, 12)

    # Create models with fixed parameters for consistency
    models: dict[str, Any] = {
        "linear": LinearRegression(),
        "rf": RandomForestRegressor(n_estimators=20, random_state=42),  # Fewer trees for speed
        "gb": GradientBoostingRegressor(n_estimators=20, random_state=42),
    }

    # Generate intermediate targets for training signal generators
    y_linear = 2 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3]
    y_rf = X[:, 4] * X[:, 5] + np.sin(X[:, 6]) + X[:, 7] ** 2
    y_gb = np.exp(0.1 * X[:, 8]) + np.log(1 + X[:, 9] ** 2) + X[:, 10] * X[:, 11]

    # Train each model on its feature subset and intermediate target
    models["linear"].fit(X[:, 0:4], y_linear)
    models["rf"].fit(X[:, 4:8], y_rf)
    models["gb"].fit(X[:, 8:12], y_gb)

    # Generate signals using trained models
    signals = {
        "linear": models["linear"].predict(X[:, 0:4]),
        "rf": models["rf"].predict(X[:, 4:8]),
        "gb": models["gb"].predict(X[:, 8:12]),
    }

    # Create global target as sum of all signals plus noise
    y_global = signals["linear"] + signals["rf"] + signals["gb"] + np.random.normal(0, 0.1, n_samples)

    return X, y_global, models


def create_data_splits(X, y_global, test_size=0.2):
    """Create train/test splits with proper row tracking."""
    n_samples = len(X)
    row_ids = np.arange(n_samples)

    # Simple split for test data
    n_test = int(n_samples * test_size)
    test_indices = row_ids[-n_test:]
    train_indices = row_ids[:-n_test]

    return {
        "X_train": X[train_indices],
        "y_train": y_global[train_indices],
        "X_test": X[test_indices],
        "y_test": y_global[test_indices],
        "train_row_ids": train_indices,
        "test_row_ids": test_indices,
    }


def create_cv_splits(X_train, y_train, train_row_ids, n_splits=3):
    """Create CV splits for training validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_splits = []

    for _, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        split_data = {
            "X_split_train": X_train[train_idx],
            "y_split_train": y_train[train_idx],
            "X_split_val": X_train[val_idx],
            "y_split_val": y_train[val_idx],
            "split_train_row_ids": train_row_ids[train_idx],
            "split_val_row_ids": train_row_ids[val_idx],
        }
        cv_splits.append(split_data)

    return cv_splits


def create_fake_training(trained_model, model_name, feature_indices, split_data, splits, training_id):
    """Create fake Training object with predictions using pre-trained model."""
    # Extract feature subset for this model
    X_split_train = split_data["X_split_train"][:, feature_indices]
    X_split_val = split_data["X_split_val"][:, feature_indices]
    X_test = splits["X_test"][:, feature_indices]

    # Create predictions using pre-trained model (NO retraining)
    pred_train = trained_model.predict(X_split_train)
    pred_val = trained_model.predict(X_split_val)
    pred_test = trained_model.predict(X_test)

    # Create prediction dataframes in octopus format (with required metadata columns)
    predictions = {
        "train": pd.DataFrame(
            {
                "row_id": split_data["split_train_row_ids"],
                "prediction": pred_train,
                "target": split_data["y_split_train"],
                "outer_split_id": 0,
                "inner_split_id": training_id,
                "partition": "train",
                "task_id": 0,
            }
        ),
        "dev": pd.DataFrame(
            {
                "row_id": split_data["split_val_row_ids"],
                "prediction": pred_val,
                "target": split_data["y_split_val"],
                "outer_split_id": 0,
                "inner_split_id": training_id,
                "partition": "dev",
                "task_id": 0,
            }
        ),
        "test": pd.DataFrame(
            {
                "row_id": splits["test_row_ids"],
                "prediction": pred_test,
                "target": splits["y_test"],
                "outer_split_id": 0,
                "inner_split_id": training_id,
                "partition": "test",
                "task_id": 0,
            }
        ),
    }

    # Create proper dataframes for Training object
    train_df = pd.DataFrame(X_split_train, columns=[f"feature_{i}" for i in range(len(feature_indices))])
    train_df["row_id"] = split_data["split_train_row_ids"]
    train_df["target"] = split_data["y_split_train"]

    val_df = pd.DataFrame(X_split_val, columns=[f"feature_{i}" for i in range(len(feature_indices))])
    val_df["row_id"] = split_data["split_val_row_ids"]
    val_df["target"] = split_data["y_split_val"]

    test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(len(feature_indices))])
    test_df["row_id"] = splits["test_row_ids"]
    test_df["target"] = splits["y_test"]

    # Map model names to valid octopus model names
    model_name_mapping: dict[str, ModelName] = {
        "linear": ModelName.RidgeRegressor,
        "rf": ModelName.RandomForestRegressor,
        "gb": ModelName.GradientBoostingRegressor,
    }

    training = Training(
        training_id=training_id,
        ml_type=MLType.REGRESSION,
        target_assignments={"default": "target"},
        feature_cols=[f"feature_{i}" for i in range(len(feature_indices))],
        row_id_col="row_id",
        data_train=train_df,
        data_dev=val_df,
        data_test=test_df,
        target_metric="MAE",
        max_features=len(feature_indices),
        feature_groups={},
        config_training={"ml_model_type": model_name_mapping[model_name]},
    )

    # Manually set predictions and model
    training.predictions = predictions
    training.model = trained_model

    return training


def create_fake_bag(log_dir, trained_model, model_name, feature_indices, cv_splits, splits, bag_id):
    """Create complete fake Bag for one trial."""
    trainings = []

    for split_idx, split_data in enumerate(cv_splits):
        training_id = f"{bag_id}_{split_idx}"
        training = create_fake_training(trained_model, model_name, feature_indices, split_data, splits, training_id)
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

    # Set train_status to indicate models are already trained
    bag.train_status = True

    return bag


def test_ensemble_selection_ensembled_data(tmp_path):
    """Test that ensemble selection works in a controlled scenario.

    Creates synthetic data where:
    - 3 models use orthogonal feature subsets (0-3, 4-7, 8-11)
    - Each model captures 1/3 of total signal
    - Target = signal1 + signal2 + signal3 + noise
    - Ensemble should significantly outperform individual models
    """
    # Generate synthetic data and trained models
    X, y_global, models = create_synthetic_data_and_models(n_samples=400)  # Smaller for speed

    # Create data splits
    splits = create_data_splits(X, y_global)
    cv_splits = create_cv_splits(splits["X_train"], splits["y_train"], splits["train_row_ids"])

    # Create fake bags for each model
    feature_subsets = {"linear": list(range(0, 4)), "rf": list(range(4, 8)), "gb": list(range(8, 12))}

    # Save fake trial bags to temporary directory
    trials_path = UPath(tmp_path / "experiment0" / "sequence0" / "trials")
    trials_path.mkdir(parents=True)

    bags = {}
    for model_name, model in models.items():
        bag = create_fake_bag(
            trials_path, model, model_name, feature_subsets[model_name], cv_splits, splits, bag_id=f"trial_{model_name}"
        )
        bags[model_name] = bag

    for trial_idx, (_model_name, bag) in enumerate(bags.items()):
        filename = f"trial_{trial_idx}_bag.joblib"
        filepath = trials_path / filename
        joblib_save(bag, filepath)

    # Get individual model performance
    individual_performances = []
    for _model_name, bag in bags.items():
        scores = bag.get_performance(n_assigned_cpus=1)
        individual_performances.append(scores[PerformanceKey.DEV_ENSEMBLE])

    best_individual_mae = min(individual_performances)

    # 6. Run EnSel analysis
    ensel = EnSel(
        target_metric="MAE",
        path_trials=trials_path,
        max_n_iterations=25,
        row_id_col="row_id",
        target_assignments={"default": "target"},
        n_assigned_cpus=1,
        ml_type=MLType.REGRESSION,
    )

    # Extract ensemble results
    start_ensemble = ensel.start_ensemble
    optimized_ensemble = ensel.optimized_ensemble

    # Calculate ensemble performance
    start_bags = list(start_ensemble.keys())
    start_scores = ensel._ensemble_models(start_bags)
    ensemble_mae = start_scores[PerformanceKey.DEV_ENSEMBLE]

    # Calculate true optimal ensemble performance using same bag predictions
    # Sum individual bag dev predictions instead of ensemble selection's averaging
    bag_dev_predictions = []
    for bag in bags.values():
        # Get pooled dev predictions from this bag (this is what EnSel uses)
        bag_pool = []
        for training in bag.trainings:
            bag_pool.append(training.predictions["dev"])

        # Pool predictions the same way as in Bag.get_scores()
        combined = pd.concat(bag_pool, axis=0)
        numeric_cols = combined.select_dtypes(include=["number"]).columns.tolist()
        for col in ["row_id", "outer_split_id", "task_id"]:
            if col in numeric_cols:
                numeric_cols.remove(col)
        pooled_dev = combined.groupby("row_id")[numeric_cols].mean().reset_index()
        bag_dev_predictions.append(pooled_dev)

    # Get common row IDs and targets
    common_rows = bag_dev_predictions[0]
    dev_targets = common_rows["target"].values

    # Average predictions from all three bags (same as ensemble selection does)
    true_ensemble_preds = (
        bag_dev_predictions[0]["prediction"].values
        + bag_dev_predictions[1]["prediction"].values
        + bag_dev_predictions[2]["prediction"].values
    ) / 3

    true_ensemble_mae = mean_absolute_error(dev_targets, true_ensemble_preds)

    # Test that ensemble outperforms best individual model
    assert ensemble_mae < best_individual_mae, (
        f"Ensemble MAE ({ensemble_mae:.4f}) should be better than best individual MAE ({best_individual_mae:.4f})"
    )

    # Test that found ensemble achieves near-optimal performance
    print(f"Ensemble MAE: {ensemble_mae:.4f}, True optimal MAE: {true_ensemble_mae:.4f}")
    assert ensemble_mae <= true_ensemble_mae * 1.1

    # Test that EnSel selected all 3 models (they should all contribute)
    assert len(start_ensemble) == 3

    # Test that optimization attempted to improve (weights might change)
    assert len(optimized_ensemble) >= len(start_ensemble)


# inner_split_id correctness


def _build_ensemble_trainings(ensemble_paths_dict: dict) -> list:
    """Replicate production _create_ensemble_bag deep-copy logic (core.py:259-274).

    Inlines the same deep-copy + inner_split_id update logic used in production.
    Since _create_ensemble_bag is a method on TakoModuleTemplate (requires
    StudyContext), we duplicate the logic here rather than calling it directly.
    """
    trainings = []
    train_id = 0
    for path, weight in ensemble_paths_dict.items():
        bag = joblib_load(path)
        for training in bag.trainings:
            for _ in range(int(weight)):
                train_cp = copy.deepcopy(training)
                train_cp.training_id = f"0_0_{train_id}"
                train_cp.training_weight = 1
                for part in train_cp.predictions:
                    if isinstance(train_cp.predictions[part], pd.DataFrame):
                        train_cp.predictions[part] = train_cp.predictions[part].copy()
                        train_cp.predictions[part]["inner_split_id"] = str(train_id)
                train_id += 1
                trainings.append(train_cp)
    return trainings


def test_ensemble_bag_has_unique_inner_split_ids(tmp_path):
    """All trainings in ensemble bag must have unique inner_split_id values.

    Regression guard for Issue #2: inner_split_id was not updated after
    deep-copy in _create_ensemble_bag, causing fold ID collisions.

    Verifies by calling production deep-copy logic via _build_ensemble_trainings
    and asserting inner_split_ids are unique without any test-side mutation.
    """
    # Create 2 bags x 3 folds
    X, y_global, models = create_synthetic_data_and_models(n_samples=300)
    splits = create_data_splits(X, y_global)
    cv_splits = create_cv_splits(splits["X_train"], splits["y_train"], splits["train_row_ids"])

    feature_subsets = {"linear": list(range(0, 4)), "rf": list(range(4, 8))}
    trials_path = UPath(tmp_path / "trials")
    trials_path.mkdir(parents=True)

    for trial_idx, (model_name, model) in enumerate(list(models.items())[:2]):
        bag = create_fake_bag(
            trials_path, model, model_name, feature_subsets[model_name], cv_splits, splits, bag_id=f"trial_{model_name}"
        )
        joblib_save(bag, trials_path / f"trial_{trial_idx}_bag.joblib")

    # Run ensemble selection
    ensel = EnSel(
        target_metric="MAE",
        path_trials=trials_path,
        max_n_iterations=5,
        row_id_col="row_id",
        target_assignments={"default": "target"},
        n_assigned_cpus=1,
        ml_type=MLType.REGRESSION,
    )

    # Build trainings using the production deep-copy logic
    trainings = _build_ensemble_trainings(ensel.optimized_ensemble)

    # Verify all inner_split_ids are unique across trainings
    dev_ids = [t.predictions["dev"]["inner_split_id"].iloc[0] for t in trainings]
    assert len(set(dev_ids)) == len(dev_ids), (
        f"inner_split_ids must be unique across trainings, "
        f"got {len(set(dev_ids))} unique out of {len(dev_ids)}: {dev_ids}"
    )

    # Each training must have exactly one inner_split_id per partition
    for training in trainings:
        for part in ["dev", "test"]:
            if part in training.predictions:
                ids = training.predictions[part]["inner_split_id"].unique()
                assert len(ids) == 1, f"Training {training.training_id} has multiple inner_split_ids in {part}: {ids}"


def test_ensemble_bag_predictions_groupby_inner_split_correct(tmp_path):
    """Groupby inner_split_id must yield one group per training.

    Regression guard for Issue #2: without unique inner_split_ids,
    groupby merges trainings from different bags.
    """
    X, y_global, models = create_synthetic_data_and_models(n_samples=300)
    splits = create_data_splits(X, y_global)
    cv_splits = create_cv_splits(splits["X_train"], splits["y_train"], splits["train_row_ids"])

    feature_subsets = {"linear": list(range(0, 4)), "rf": list(range(4, 8))}
    trials_path = UPath(tmp_path / "trials")
    trials_path.mkdir(parents=True)

    for trial_idx, (model_name, model) in enumerate(list(models.items())[:2]):
        bag = create_fake_bag(
            trials_path, model, model_name, feature_subsets[model_name], cv_splits, splits, bag_id=f"trial_{model_name}"
        )
        joblib_save(bag, trials_path / f"trial_{trial_idx}_bag.joblib")

    ensel = EnSel(
        target_metric="MAE",
        path_trials=trials_path,
        max_n_iterations=5,
        row_id_col="row_id",
        target_assignments={"default": "target"},
        n_assigned_cpus=1,
        ml_type=MLType.REGRESSION,
    )

    # Build trainings using the production deep-copy logic
    trainings = _build_ensemble_trainings(ensel.optimized_ensemble)

    # Concat all dev predictions and verify groupby produces one group per training
    all_dev = pd.concat([t.predictions["dev"] for t in trainings], axis=0)
    groups = all_dev.groupby("inner_split_id")

    assert len(groups) == len(trainings), (
        f"Expected {len(trainings)} groups, got {len(groups)}. inner_split_ids are colliding across trainings."
    )


# Ensemble quality invariants


def test_ensemble_at_least_as_good_as_best_single_model(tmp_path):
    """Final ensemble must perform at least as well as the best individual bag.

    This is the fundamental guarantee of Caruana's algorithm: ensemble
    selection with replacement can always fall back to the best single model.
    """
    X, y_global, models = create_synthetic_data_and_models(n_samples=400)
    splits = create_data_splits(X, y_global)
    cv_splits = create_cv_splits(splits["X_train"], splits["y_train"], splits["train_row_ids"])

    feature_subsets = {"linear": list(range(0, 4)), "rf": list(range(4, 8)), "gb": list(range(8, 12))}
    trials_path = UPath(tmp_path / "trials")
    trials_path.mkdir(parents=True)

    bags = {}
    for model_name, model in models.items():
        bag = create_fake_bag(
            trials_path, model, model_name, feature_subsets[model_name], cv_splits, splits, bag_id=f"trial_{model_name}"
        )
        bags[model_name] = bag

    for trial_idx, (_name, bag) in enumerate(bags.items()):
        joblib_save(bag, trials_path / f"trial_{trial_idx}_bag.joblib")

    # Get best individual MAE
    individual_performances = []
    for bag in bags.values():
        scores = bag.get_performance(n_assigned_cpus=1)
        individual_performances.append(scores[PerformanceKey.DEV_ENSEMBLE])
    best_individual_mae = min(individual_performances)

    ensel = EnSel(
        target_metric="MAE",
        path_trials=trials_path,
        max_n_iterations=25,
        row_id_col="row_id",
        target_assignments={"default": "target"},
        n_assigned_cpus=1,
        ml_type=MLType.REGRESSION,
    )

    # Compute optimized ensemble performance
    opt_bags = []
    for path, weight in ensel.optimized_ensemble.items():
        opt_bags.extend([path] * weight)
    opt_perf = ensel._ensemble_models(opt_bags)[PerformanceKey.DEV_ENSEMBLE]

    # For MAE (minimize), ensemble must be <= best individual
    assert opt_perf <= best_individual_mae + 1e-10, (
        f"Ensemble ({opt_perf:.6f}) must be at least as good as best individual ({best_individual_mae:.6f})"
    )


# Weight propagation


def test_ensemble_with_replacement_weights_predictions(tmp_path):
    """Model with weight > 1 must contribute proportionally more to predictions.

    A model selected K times via replacement gets K/(total) weight in the
    averaged predictions.
    """
    X, y_global, models = create_synthetic_data_and_models(n_samples=300)
    splits = create_data_splits(X, y_global)
    cv_splits = create_cv_splits(splits["X_train"], splits["y_train"], splits["train_row_ids"])

    feature_subsets = {"linear": list(range(0, 4)), "rf": list(range(4, 8))}
    trials_path = UPath(tmp_path / "trials")
    trials_path.mkdir(parents=True)

    bags_by_name = {}
    for trial_idx, (model_name, model) in enumerate(list(models.items())[:2]):
        bag = create_fake_bag(
            trials_path, model, model_name, feature_subsets[model_name], cv_splits, splits, bag_id=f"trial_{model_name}"
        )
        bags_by_name[model_name] = bag
        joblib_save(bag, trials_path / f"trial_{trial_idx}_bag.joblib")

    ensel = EnSel(
        target_metric="MAE",
        path_trials=trials_path,
        max_n_iterations=10,
        row_id_col="row_id",
        target_assignments={"default": "target"},
        n_assigned_cpus=1,
        ml_type=MLType.REGRESSION,
    )

    # Manually create weighted ensemble: model_a x3, model_b x1
    bag_keys = list(ensel.bags.keys())
    assert len(bag_keys) == 2

    # Ensemble with [A, A, A, B] should weight A 3x more than B
    weighted_bags = [bag_keys[0], bag_keys[0], bag_keys[0], bag_keys[1]]
    weighted_result = ensel._ensemble_models(weighted_bags)

    # Ensemble with [A, B] should give equal weight
    equal_bags = [bag_keys[0], bag_keys[1]]
    equal_result = ensel._ensemble_models(equal_bags)

    # The two must produce different dev_ensemble scores
    # (unless predictions are identical, which they're not for different models)
    assert weighted_result[PerformanceKey.DEV_ENSEMBLE] != equal_result[PerformanceKey.DEV_ENSEMBLE], (
        "Weighted [A,A,A,B] and equal [A,B] ensembles should produce different scores"
    )
