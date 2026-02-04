"""Test ensemble selection functionality."""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from upath import UPath

from octopus.modules.octo.bag import Bag
from octopus.modules.octo.enssel import EnSel
from octopus.modules.octo.training import Training


def create_synthetic_data_and_models(n_samples=500):
    """Create synthetic data with 3 models trained on orthogonal feature subsets.

    Returns data where target = signal1 + signal2 + signal3 + noise,
    and each model captures only 1/3 of the signal.
    """
    np.random.seed(42)

    # Generate features
    X = np.random.randn(n_samples, 12)

    # Create models with fixed parameters for consistency
    models = {
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


def create_cv_folds(X_train, y_train, train_row_ids, n_folds=3):
    """Create CV folds for training validation."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_folds = []

    for _, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        fold_data = {
            "X_fold_train": X_train[train_idx],
            "y_fold_train": y_train[train_idx],
            "X_fold_val": X_train[val_idx],
            "y_fold_val": y_train[val_idx],
            "fold_train_row_ids": train_row_ids[train_idx],
            "fold_val_row_ids": train_row_ids[val_idx],
        }
        cv_folds.append(fold_data)

    return cv_folds


def create_fake_training(trained_model, model_name, feature_indices, fold_data, splits, training_id):
    """Create fake Training object with predictions using pre-trained model."""
    # Extract feature subset for this model
    X_fold_train = fold_data["X_fold_train"][:, feature_indices]
    X_fold_val = fold_data["X_fold_val"][:, feature_indices]
    X_test = splits["X_test"][:, feature_indices]

    # Create predictions using pre-trained model (NO retraining)
    pred_train = trained_model.predict(X_fold_train)
    pred_val = trained_model.predict(X_fold_val)
    pred_test = trained_model.predict(X_test)

    # Create prediction dataframes in octopus format
    predictions = {
        "train": pd.DataFrame(
            {"row_id": fold_data["fold_train_row_ids"], "prediction": pred_train, "target": fold_data["y_fold_train"]}
        ),
        "dev": pd.DataFrame(
            {"row_id": fold_data["fold_val_row_ids"], "prediction": pred_val, "target": fold_data["y_fold_val"]}
        ),
        "test": pd.DataFrame({"row_id": splits["test_row_ids"], "prediction": pred_test, "target": splits["y_test"]}),
    }

    # Create proper dataframes for Training object
    train_df = pd.DataFrame(X_fold_train, columns=[f"feature_{i}" for i in range(len(feature_indices))])
    train_df["row_id"] = fold_data["fold_train_row_ids"]
    train_df["target"] = fold_data["y_fold_train"]

    val_df = pd.DataFrame(X_fold_val, columns=[f"feature_{i}" for i in range(len(feature_indices))])
    val_df["row_id"] = fold_data["fold_val_row_ids"]
    val_df["target"] = fold_data["y_fold_val"]

    test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(len(feature_indices))])
    test_df["row_id"] = splits["test_row_ids"]
    test_df["target"] = splits["y_test"]

    # Map model names to valid octopus model names
    model_name_mapping = {"linear": "RidgeRegressor", "rf": "RandomForestRegressor", "gb": "GradientBoostingRegressor"}

    training = Training(
        training_id=training_id,
        ml_type="regression",
        target_assignments={"default": "target"},
        feature_cols=[f"feature_{i}" for i in range(len(feature_indices))],
        row_column="row_id",
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


def create_fake_bag(log_dir, trained_model, model_name, feature_indices, cv_folds, splits, bag_id):
    """Create complete fake Bag for one trial."""
    trainings = []

    for fold_idx, fold_data in enumerate(cv_folds):
        training_id = f"{bag_id}_{fold_idx}"
        training = create_fake_training(trained_model, model_name, feature_indices, fold_data, splits, training_id)
        trainings.append(training)

    bag = Bag(
        bag_id=bag_id,
        trainings=trainings,
        target_assignments={"default": "target"},
        row_column="row_id",
        target_metric="MAE",
        ml_type="regression",
        parallel_execution=False,
        num_workers=1,
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
    cv_folds = create_cv_folds(splits["X_train"], splits["y_train"], splits["train_row_ids"])

    # Create fake bags for each model
    feature_subsets = {"linear": list(range(0, 4)), "rf": list(range(4, 8)), "gb": list(range(8, 12))}

    # Save fake trial bags to temporary directory
    trials_path = UPath(tmp_path / "experiment0" / "sequence0" / "trials")
    trials_path.mkdir(parents=True)

    bags = {}
    for model_name, model in models.items():
        bag = create_fake_bag(
            trials_path, model, model_name, feature_subsets[model_name], cv_folds, splits, bag_id=f"trial_{model_name}"
        )
        bags[model_name] = bag

    for trial_idx, (_model_name, bag) in enumerate(bags.items()):
        filename = f"trial{trial_idx}_bag.pkl"
        filepath = trials_path / filename
        bag.to_pickle(filepath)

    # Get individual model performance
    individual_performances = []
    for _model_name, bag in bags.items():
        scores = bag.get_performance()
        individual_performances.append(scores["dev_pool"])

    best_individual_mae = min(individual_performances)

    # 6. Run EnSel analysis
    ensel = EnSel(
        target_metric="MAE",
        path_trials=trials_path,
        max_n_iterations=50,  # Fewer iterations for speed
        row_column="row_id",
        target_assignments={"default": "target"},
    )

    # Extract ensemble results
    start_ensemble = ensel.start_ensemble
    optimized_ensemble = ensel.optimized_ensemble

    # Calculate ensemble performance
    start_bags = list(start_ensemble.keys())
    start_scores = ensel._ensemble_models(start_bags)
    ensemble_mae = start_scores["dev_pool"]

    # Calculate true optimal ensemble performance using same bag predictions
    # Sum individual bag dev predictions instead of ensemble selection's averaging
    bag_dev_predictions = []
    for bag in bags.values():
        # Get pooled dev predictions from this bag (this is what EnSel uses)
        bag_pool = []
        for training in bag.trainings:
            bag_pool.append(training.predictions["dev"])

        # Pool predictions the same way as in Bag.get_scores()
        pooled_dev = pd.concat(bag_pool, axis=0).groupby("row_id").mean().reset_index()
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
