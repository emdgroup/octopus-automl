"""Helper functions to load modules for prediction and analysis."""

from typing import Any

import pandas as pd
from upath import UPath

from octopus.analysis.loaders import StudyLoader


def load_task_modules(
    study_path: str | UPath,
    task_id: int = -1,
    module: str = "octo",
    result_type: str = "best",
) -> dict[int, dict[str, Any]]:
    """Load all modules for a specific task across all outersplits.

    This function loads the fitted modules from all available outersplits
    for a given task, along with their associated data (test set, train set,
    feature information, etc.).

    Note: This function is intended for ML modules (Octo, AutoGluon) that have
    trained models. For feature selection modules (MRMR, RFE, etc.), use the
    loaders directly to get selected features.

    Args:
        study_path: Path to the study directory
        task_id: Task ID to load. If -1, loads the last task in workflow.
        module: Module name to filter results (e.g., 'octo', 'autogluon'). Default: 'octo'
        result_type: Result type to load (e.g., 'best', 'ensemble_selection'). Default: 'best'

    Returns:
        Dictionary mapping outersplit_id to a dict containing:
            - 'module': Loaded module instance (Predictor with fitted model)
            - 'data_test': Test dataset for this fold
            - 'data_train': Training dataset for this fold
            - 'outersplit_id': Fold/outersplit ID
            - 'ml_type': ML type from config
            - 'target_metric': Target metric from config
            - 'target_assignments': Target column assignments
            - 'positive_class': Positive class for classification (if applicable)
            - 'row_id_col': Row ID column name
            - 'is_ml_module': Boolean indicating if module can predict

    Example:
        >>> from octopus.analysis import load_task_modules
        >>> modules = load_task_modules("./studies/my_study/", task_id=0)
        >>> # Make predictions on new data (only works for ML modules)
        >>> for outersplit_id, module_info in modules.items():
        >>>     if module_info['is_ml_module']:
        >>>         predictions = module_info['module'].predict(new_data)
    """
    study_path = UPath(study_path)
    study_loader = StudyLoader(study_path)
    config = study_loader.load_config()

    # Determine task_id if not specified
    if task_id < 0:
        task_id = len(config["workflow"]) - 1

    modules = {}
    n_outersplits = config.get("n_folds_outer", 0)

    print(f"\nLoading modules for task {task_id} across {n_outersplits} outersplits...")

    for outersplit_id in range(n_outersplits):
        try:
            loader = study_loader.get_outersplit_loader(
                outersplit_id=outersplit_id,
                task_id=task_id,
                module=module,
                result_type=result_type,
            )

            # Determine if this is an ML module by checking for model.joblib
            model_path = loader.module_dir / "model.joblib"
            is_ml_module = model_path.exists()

            if is_ml_module:
                # Use lightweight Predictor (no config reconstruction needed)
                from octopus.modules.predictor import Predictor  # noqa: PLC0415

                loaded_module: Any = Predictor.load(loader.module_dir)
            else:
                # Feature selection module: load selected features only
                selected_features = loader.load_selected_features()
                loaded_module = type(
                    "FeatureSelectionResult",
                    (),
                    {
                        "selected_features_": selected_features,
                    },
                )()

            # Load associated data
            data_test = loader.load_test_data()
            data_train = loader.load_train_data()

            # Build module info dict
            modules[outersplit_id] = {
                "module": loaded_module,
                "data_test": data_test,
                "data_train": data_train,
                "outersplit_id": outersplit_id,
                "ml_type": config.get("ml_type", ""),
                "target_metric": config.get("target_metric", ""),
                "target_assignments": config.get("prepared", {}).get("target_assignments", {}),
                "positive_class": config.get("positive_class"),
                "row_id_col": config.get("prepared", {}).get("row_id_col", "row_id"),
                "is_ml_module": is_ml_module,
            }

            module_type = "ML module" if is_ml_module else "Feature selection module"
            print(f"  Loaded outersplit {outersplit_id} ({module_type})")

        except (FileNotFoundError, Exception) as e:
            print(f"  Outersplit {outersplit_id}: Could not load - {e}")
            continue

    print(f"Successfully loaded {len(modules)} out of {n_outersplits} outersplits\n")

    if not modules:
        raise ValueError(
            f"No modules could be loaded for task {task_id}. Check that the study has been run and results exist."
        )

    return modules


def ensemble_predict(
    modules: dict[int, dict[str, Any]],
    data: pd.DataFrame,
    method: str = "mean",
) -> pd.DataFrame:
    """Make ensemble predictions using multiple ML modules.

    Combines predictions from all loaded ML modules into a single prediction.
    Only works with ML modules (Octo, AutoGluon) that have predict() method.

    Args:
        modules: Dictionary of module info dicts from load_task_modules()
        data: DataFrame containing features for prediction
        method: Aggregation method:
            - 'mean': Average predictions across modules (default)
            - 'median': Median of predictions
            - 'vote': Majority vote (classification only)

    Returns:
        DataFrame with aggregated predictions. For regression/mean:
            - 'prediction': Mean prediction
            - 'prediction_std': Standard deviation across modules
        For classification with probabilities:
            - One column per class with mean probabilities
            - 'prediction': Final class prediction

    Raises:
        ValueError: If no ML modules found in the provided modules dict

    Example:
        >>> modules = load_task_modules("./studies/my_study/")
        >>> predictions = ensemble_predict(modules, new_data, method='mean')
    """
    predictions_list = []

    for outersplit_id, module_info in modules.items():
        # Skip non-ML modules (feature selection modules don't have predict)
        if not module_info.get("is_ml_module", False):
            continue

        loaded_module = module_info["module"]
        pred = loaded_module.predict(data)

        # Normalize to DataFrame with a "prediction" column
        if isinstance(pred, pd.DataFrame):
            pred_df = pred.copy()
        else:
            # np.ndarray or pd.Series
            pred_df = pd.DataFrame({"prediction": pred})

        pred_df["outersplit_id"] = outersplit_id
        pred_df["row_id"] = data.index.values
        predictions_list.append(pred_df)

    if not predictions_list:
        raise ValueError(
            "No ML modules found. ensemble_predict() only works with ML modules (Octo, AutoGluon). "
            "Feature selection modules (MRMR, RFE, etc.) do not have predict() capability."
        )

    # Combine predictions
    all_preds = pd.concat(predictions_list, ignore_index=True)

    # Aggregate based on method
    if method == "mean":
        result = (
            all_preds.groupby("row_id")["prediction"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "prediction", "std": "prediction_std"})
        )
    elif method == "median":
        result = all_preds.groupby("row_id")["prediction"].median().to_frame("prediction")
    else:
        raise ValueError(f"Aggregation method '{method}' not supported")

    return result.reset_index()


def ensemble_predict_proba(
    modules: dict[int, dict[str, Any]],
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Make ensemble probability predictions for classification using ML modules.

    Averages probability predictions from all loaded ML modules.
    Only works with classification ML modules that have predict_proba() method.

    Args:
        modules: Dictionary of module info dicts from load_task_modules()
        data: DataFrame containing features for prediction

    Returns:
        DataFrame with columns:
            - One column per class with mean probabilities
            - '{class}_std': Standard deviation for each class probability
            - 'prediction': Final predicted class (argmax of mean probabilities)

    Raises:
        ValueError: If no ML modules found in the provided modules dict

    Example:
        >>> modules = load_task_modules("./studies/my_study/")
        >>> proba_df = ensemble_predict_proba(modules, new_data)
    """
    proba_list = []

    for outersplit_id, module_info in modules.items():
        # Skip non-ML modules
        if not module_info.get("is_ml_module", False):
            continue

        module = module_info["module"]
        proba = module.predict_proba(data)

        # Convert to DataFrame
        if isinstance(proba, pd.DataFrame):
            proba_df = proba.copy()
        else:
            # NumPy array - get class names from module's model
            classes = module.model_.classes_
            proba_df = pd.DataFrame(proba, columns=classes)

        proba_df["row_id"] = data.index
        proba_df["outersplit_id"] = outersplit_id
        proba_list.append(proba_df)

    if not proba_list:
        raise ValueError(
            "No ML modules found. ensemble_predict_proba() only works with ML modules (Octo, AutoGluon). "
            "Feature selection modules (MRMR, RFE, etc.) do not have predict_proba() capability."
        )

    # Combine probabilities
    all_proba = pd.concat(proba_list, ignore_index=True)

    # Get class columns (exclude row_id and outersplit_id)
    class_cols = [col for col in all_proba.columns if col not in ["row_id", "outersplit_id"]]

    # Aggregate - mean and std for each class
    result = all_proba.groupby("row_id")[class_cols].agg(["mean", "std"])

    # Flatten MultiIndex column names from .agg(["mean", "std"])
    multi_index: pd.MultiIndex = result.columns  # type: ignore[assignment]
    result.columns = pd.Index(f"{col}_{agg}" if agg == "std" else str(col) for col, agg in multi_index)

    # Get final prediction (argmax of mean probabilities)
    mean_cols = [col for col in result.columns if not col.endswith("_std")]
    result["prediction"] = result[mean_cols].idxmax(axis=1)

    return result.reset_index()
