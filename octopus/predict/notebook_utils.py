"""Utility functions for Jupyter notebooks."""

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import auc, confusion_matrix, roc_curve

from octopus.experiment import OctoExperiment
from octopus.metrics.utils import get_performance_from_model

if TYPE_CHECKING:
    from octopus.predict.core import OctoPredict

try:
    from IPython.display import display as ipython_display
except ImportError:
    ipython_display = None


def display_table(data: Any) -> None:
    """Display a table in Jupyter notebooks or print in other environments.

    This function attempts to use IPython's display functionality when running
    in a Jupyter notebook environment. If not available, it falls back to using
    the standard print function.

    Args:
        data: The data to display. Can be a pandas DataFrame, Series, or any
            object that can be displayed by IPython.display or printed.

    Example:
        >>> import pandas as pd
        >>> from octopus.predict.notebook_utils import display_table
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> display_table(df)
    """
    if ipython_display is not None:
        ipython_display(data)
    else:
        print(data)


def show_study_details(study_directory: str | Path, verbose: bool = True) -> dict:
    """Display and validate study details including configuration and structure.

    This function reads the study configuration, validates the study structure,
    and displays information about the workflow tasks and experiment directories.

    Args:
        study_directory: Path to the study directory.
        verbose: If True, prints detailed information.

    Returns:
        Dictionary containing study information with keys:
            - 'path': Path object of the study directory
            - 'config': Study configuration dictionary
            - 'ml_type': Machine learning type
            - 'n_folds_outer': Number of outer folds
            - 'workflow_tasks': List of workflow task configurations
            - 'outersplit_dirs': List of outersplit directory paths
            - 'expected_task_ids': List of expected task IDs
            - 'octo_workflow_tasks': List of task IDs for octo modules
            - 'missing_outersplits': List of missing outersplit IDs
            - 'missing_workflow_dirs': List of missing workflow directories

    Raises:
        ValueError: If no outersplit directories are found, or if no workflow results are found.
        FileNotFoundError: If the study directory does not exist.

    Example:
        >>> from octopus.predict.notebook_utils import show_study_details
        >>> study_info = show_study_details("./studies/my_study/")
        >>> print(f"Study has {len(study_info['workflow_tasks'])} workflow tasks")
    """
    path_study = Path(study_directory)

    # Display path status
    if not path_study.exists():
        raise FileNotFoundError(f"⚠️ WARNING: Study path does not exist: {path_study}")

    if verbose:
        print(f"Selected study path: {path_study}\n")

    # Study information and available sequence items
    config_path = path_study / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    ml_type = config["ml_type"]
    n_folds_outer = config["n_folds_outer"]
    workflow_tasks = config["workflow"]

    # Validate study requirements
    if verbose:
        print("Validate study....")
        print(f"ML Type: {ml_type}")

    # Check 1: Verify study has been run (check for outersplit directories)
    outersplit = sorted(
        [d for d in path_study.glob("outersplit*") if d.is_dir()],
        key=lambda x: int(x.name.replace("outersplit", "")),
    )

    if not outersplit:
        raise ValueError(
            f"❌ ERROR: No experiment directories found in study path.\n"
            f"Study path: {path_study}\nThe study may not have been run yet."
        )
    if verbose:
        print(f"Found {len(outersplit)} outersplit directory/directories")

    # Check that all expected outersplit directories exist
    expected_outersplit_ids = list(range(n_folds_outer))
    if verbose:
        print(f"Expected outersplit IDs: {expected_outersplit_ids}")

    missing_outersplits = []
    for split_id in expected_outersplit_ids:
        expected_split_dir = path_study / f"outersplit{split_id}"
        if not expected_split_dir.exists():
            if verbose:
                print(f"⚠️  WARNING: Missing directory 'outersplit{split_id}'")
            missing_outersplits.append(split_id)

    if missing_outersplits:
        if verbose:
            print(f"⚠️  {len(missing_outersplits)} outersplit directory/directories missing")
    elif verbose:
        print("All expected outersplit directories found")

    # Check 3: Verify experiments contain results
    # Extract task_ids from workflow_tasks
    expected_task_ids = [task["task_id"] for task in workflow_tasks]
    if verbose:
        print(f"Expected workflow task IDs: {expected_task_ids}")

    has_results = False
    missing_workflow_dirs = []

    for split_dir in outersplit:
        workflow_dirs = list(split_dir.glob("workflowtask*"))
        if workflow_dirs:
            has_results = True

        # Check that all expected workflow task directories exist
        for task_id in expected_task_ids:
            expected_dir = split_dir / f"workflowtask{task_id}"
            if not expected_dir.exists():
                if verbose:
                    print(f"⚠️  WARNING: Missing directory '{expected_dir.name}' in {split_dir.name}")
                missing_workflow_dirs.append(str(expected_dir.relative_to(path_study)))

    if not has_results:
        raise ValueError(
            "❌ ERROR: No workflow results found in experiments.\nThe study may not have completed successfully."
        )
    elif missing_workflow_dirs:
        if verbose:
            print("⚠️  Study has completed workflow tasks, but some directories are missing (see warnings above)")
    elif verbose:
        print("Study has completed workflow tasks - all expected directories found")

    # Display workflow task information
    if verbose:
        print("\nInformation on workflow tasks in this study")
        print(f"Number of workflow tasks: {len(workflow_tasks)}")

    # Get octo workflows
    octo_workflow_lst = []
    for _item in workflow_tasks:
        if verbose:
            print(f"Task {_item['task_id']}: {_item['module']}")
        if _item["module"] == "octo":
            octo_workflow_lst.append(_item["task_id"])

    if verbose:
        print(f"Octo workflow tasks: {octo_workflow_lst}")

    # Return all collected information
    return {
        "path": path_study,
        "config": config,
        "ml_type": ml_type,
        "n_folds_outer": n_folds_outer,
        "workflow_tasks": workflow_tasks,
        "outersplit_dirs": outersplit,
        "expected_task_ids": expected_task_ids,
        "octo_workflow_tasks": octo_workflow_lst,
        "missing_outersplits": missing_outersplits,
        "missing_workflow_dirs": missing_workflow_dirs,
    }


def _build_performance_dataframe(study_info: dict) -> pd.DataFrame:
    """Build performance dataframe from study experiments.

    This is a utility function that loads experiments from all workflow tasks
    across outer splits and extracts performance metrics and feature information.

    Args:
        study_info: Dictionary returned by show_study_details() containing study information.

    Returns:
        DataFrame containing performance metrics with columns:
            - OuterSplit: Outer split number
            - Task: Workflow task number
            - Task_name: Name of the workflow directory
            - Results_key: Key identifying the result
            - Scores_dict: Dictionary of performance scores
            - n_features: Number of selected features
            - Selected_features: List of selected feature names
    """
    df = pd.DataFrame(
        columns=[
            "OuterSplit",
            "Task",
            "Task_name",
            "Results_key",
            "Scores_dict",
            "n_features",
            "Selected_features",
        ]
    )

    for path_split in study_info["outersplit_dirs"]:
        # Name of outer split
        split_name = path_split.name
        # Number of outer split
        match = re.search(r"\d+$", split_name)
        split_num = int(match.group()) if match else None

        # Workflows
        path_workflows = [f for f in path_split.glob("workflowtask*") if f.is_dir()]

        # Iterate through workflows
        for path_workflow in path_workflows:
            # Name of workflow task
            workflow_name = str(path_workflow.name)
            # Number of workflow task
            match = re.search(r"\d+", workflow_name)
            workflow_num = int(match.group()) if match else None
            path_exp_pkl = path_workflow.joinpath(f"exp{split_num}_{workflow_num}.pkl")

            if path_exp_pkl.exists():
                # Load experiment
                exp = OctoExperiment.from_pickle(path_exp_pkl)
                # Iterate through keys
                for key, result in exp.results.items():
                    new_row = pd.DataFrame(
                        [
                            {
                                "OuterSplit": split_num,
                                "Task": workflow_num,
                                "Task_name": workflow_name,
                                "Results_key": str(key),
                                "Scores_dict": result.scores,
                                "n_features": len(result.selected_features),
                                "Selected_features": sorted(result.selected_features),
                            }
                        ]
                    )
                    df = pd.concat([df, new_row], ignore_index=True)

    # Sort dataframe by Task, then by OuterSplit
    df = df.sort_values(by=["Task", "OuterSplit"], ignore_index=True)

    return df


def show_target_metric_performance(study_info: dict, details: bool = False) -> list[pd.DataFrame]:
    """Display performance metrics for all workflow tasks in a study.

    This function loads experiments from all workflow tasks across outer splits,
    extracts performance metrics, and displays aggregated results.

    Args:
        study_info: Dictionary returned by show_study_details() containing study information.
        details: If False, only shows performance overview. If True, shows performance
            overview first, then detailed information for each experiment.

    Returns:
        List of DataFrames, one for each task/key combination with performance metrics

    Example:
        >>> from octopus.predict.notebook_utils import show_study_details, show_target_metric_performance
        >>> study_info = show_study_details("./studies/my_study/")
        >>> tables = show_target_metric_performance(study_info, details=False)
    """
    # Build performance dataframe using utility function
    df = _build_performance_dataframe(study_info)

    # Initialize results list
    performance_tables = []

    # Performance overview (always shown)
    for _item in study_info["workflow_tasks"]:
        print(f"\033[1mWorkflow task: {_item['task_id']}\033[0m")

        df_workflow = df[df["Task"] == _item["task_id"]]

        # Available results keys
        res_keys = sorted(set(df_workflow["Results_key"].tolist()))
        print("Available results keys:", res_keys)

        for _key in res_keys:
            print("Selected results key:", _key)
            df_workflow_selected = df_workflow.copy()
            df_workflow_selected = df_workflow_selected[df_workflow_selected["Results_key"] == _key]
            # Expand the Scores_dict column into separate columns
            scores_df = df_workflow_selected["Scores_dict"].apply(pd.Series)
            # Combine with the original DataFrame, setting 'OuterSplit' as the index
            result_df = df_workflow_selected[["OuterSplit"]].join(scores_df).set_index("OuterSplit")
            # Remove columns that do not contain numeric values
            result_df = result_df.select_dtypes(include="number")
            # Add Task and Key columns
            result_df.insert(0, "Task", _item["task_id"])
            result_df.insert(1, "Key", _key)
            mean_values = {}
            # Iterate through the columns
            for column in result_df.columns:
                if result_df[column].dtype in ["float64", "int64"]:
                    mean_values[column] = result_df[column].mean()
                else:
                    mean_values[column] = ""
            # Set non-numeric values for Mean row
            mean_values["Task"] = _item["task_id"]
            mean_values["Key"] = _key
            # Append the mean values as a new row
            result_df.loc["Mean"] = mean_values
            # Collect the table
            performance_tables.append(result_df.copy())
            display_table(result_df)

    # Detailed printout (only if details=True)
    if details:
        print("\n" + "=" * 80)
        print("DETAILED INFORMATION")
        print("=" * 80 + "\n")
        print("Listing of outer splits available in this study")

        # Iterate through outer splits again for detailed output
        for path_split in study_info["outersplit_dirs"]:
            split_name = path_split.name
            match = re.search(r"\d+$", split_name)
            split_num = int(match.group()) if match else None

            print(f"Processing split {split_num} at {path_split} ...")

            # Workflows
            path_workflows = [f for f in path_split.glob("workflowtask*") if f.is_dir()]

            # Iterate through workflows
            for path_workflow in path_workflows:
                workflow_name = str(path_workflow.name)
                match = re.search(r"\d+", workflow_name)
                workflow_num = int(match.group()) if match else None
                path_exp_pkl = path_workflow.joinpath(f"exp{split_num}_{workflow_num}.pkl")

                print(f"\tWorkflow Task {workflow_num} at {path_exp_pkl}")

                if path_exp_pkl.exists():
                    exp = OctoExperiment.from_pickle(path_exp_pkl)
                    for key, result in exp.results.items():
                        print(f"\t\t{key}: {'\n\t\t\t'.join(f'{m}: {s}' for m, s in result.scores.items())}")

    return performance_tables


def show_selected_features(
    study_info: dict, sort_task: int | None = None, sort_key: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Display the number of selected features across outer splits, tasks, and result keys.

    This function creates two summary tables:
    1. Number of features per outer split for each task-key combination
    2. Feature frequency table showing how often each feature appears across outer splits for each task-key combination

    Args:
        study_info: Dictionary returned by show_study_details() containing study information.
        sort_task: Task ID to use for sorting the feature frequency table.
            If provided (along with sort_key), the table will be sorted by frequency for this task-key combination.
            If None, uses the first task in the dataframe.
        sort_key: Results key to use for sorting the feature frequency table.
            If provided (along with sort_task), the table will be sorted by frequency for this task-key combination.
            If None, uses the first key for the specified task.

    Returns:
        Tuple of three DataFrames:
        - feature_table: Number of features per outer split for each task-key combination
        - frequency_table: Features as rows, (task, key) combinations as columns, showing selection frequency
        - raw_feature_table: Raw performance dataframe with experiment data and selected features

    Example:
        >>> from octopus.predict.notebook_utils import show_study_details, show_selected_features
        >>> study_info = show_study_details("./studies/my_study/")
        >>> feat_table, freq_table, raw_table = show_selected_features(study_info, sort_task=0, sort_key="best")
    """
    # Build performance dataframe using utility function
    raw_feature_table = _build_performance_dataframe(study_info)

    # Create a pivot table with OuterSplit as rows and (Task, Results_key) as columns
    feature_table = raw_feature_table.pivot_table(
        index="OuterSplit", columns=["Task", "Results_key"], values="n_features", aggfunc="first"
    )

    # Calculate mean for each (task, key) combination and add as a new row
    mean_row = feature_table.mean(axis=0)
    feature_table.loc["Mean"] = mean_row

    # Convert to integers
    feature_table = feature_table.astype(int)
    feature_table.index.name = "OuterSplit"

    # Display the table
    print("\n" + "=" * 40)
    print("NUMBER OF SELECTED FEATURES")
    print("=" * 40)
    print("Rows: OuterSplit | Columns: (Task, Key) | Values: Number of Features")
    display_table(feature_table)

    # Create feature frequency table
    # Get all unique task-key combinations
    task_key_combinations = (
        raw_feature_table[["Task", "Results_key"]].drop_duplicates().sort_values(["Task", "Results_key"])
    )

    # Determine which task-key combination to use for sorting
    if sort_task is None:
        sort_task = int(task_key_combinations.iloc[0]["Task"])
        sort_key = str(task_key_combinations.iloc[0]["Results_key"])
    elif sort_key is None:
        # If sort_task is provided but not sort_key, use the first key for that task
        task_keys = raw_feature_table[raw_feature_table["Task"] == sort_task]["Results_key"].unique()
        sort_key = str(task_keys[0]) if len(task_keys) > 0 else None

    # Create a dictionary to store frequency counts
    frequency_data: dict[tuple[int, str], dict[str, int]] = {}

    # Count feature occurrences for each task-key combination across all outer splits
    for _, row in task_key_combinations.iterrows():
        task = int(row["Task"])
        key = str(row["Results_key"])
        task_key = (task, key)
        frequency_data[task_key] = {}

        task_key_data = raw_feature_table[
            (raw_feature_table["Task"] == task) & (raw_feature_table["Results_key"] == key)
        ]
        for _, data_row in task_key_data.iterrows():
            for feature in data_row["Selected_features"]:
                if feature not in frequency_data[task_key]:
                    frequency_data[task_key][feature] = 0
                frequency_data[task_key][feature] += 1

    # Create frequency table with multi-index columns
    frequency_table = pd.DataFrame(frequency_data)
    frequency_table = frequency_table.fillna(0).astype(int)

    # Sort by the specified task-key combination's frequency (highest first)
    if sort_key is not None:
        sort_col = (sort_task, sort_key)
        if sort_col in frequency_table.columns:
            frequency_table = frequency_table.sort_values(by=sort_col, ascending=False)  # type: ignore[arg-type]

    frequency_table.index.name = "Feature"

    # Display the frequency table
    print("\n" + "=" * 40)
    print("FEATURE FREQUENCY ACROSS OUTER SPLITS")
    print("=" * 40)
    print("Rows: Features | Columns: (Task, Key) | Values: Feature Frequency")
    if sort_key:
        print(f"Sorted by Task {sort_task}, Key '{sort_key}' frequency (highest first)")
    display_table(frequency_table)

    return feature_table, frequency_table, raw_feature_table


def testset_performance_overview(predictor: "OctoPredict", metrics: list[str]) -> pd.DataFrame:
    """Display test performance metrics across all experiments in a task predictor.

    This function evaluates each experiment's model on the test dataset using the
    specified metrics and creates a summary table showing performance across experiments.

    Args:
        predictor: OctoPredict object containing experiments with trained models.
        metrics: List of metric names to evaluate (e.g., ['roc_auc', 'accuracy']).

    Returns:
        DataFrame with experiments as rows (plus a 'Mean' row), metrics as columns,
        showing performance values for each metric-experiment combination.

    Example:
        >>> from octopus.predict import OctoPredict
        >>> from octopus.predict.notebook_utils import testset_performance_overview
        >>> task_predictor = OctoPredict(study_directory="./studies/my_study/", task_id=0)
        >>> df_test_perf = testset_performance_overview(predictor=task_predictor, metrics=["roc_auc", "accuracy"])
    """
    # Collect performance data
    data_list = []

    print("Performance on test dataset (pooling)")

    for exp_id, experiment in predictor.experiments.items():
        # Create a row dictionary for this experiment
        row_data: dict[str, int | float] = {"outersplit": exp_id}

        for metric in metrics:
            performance = get_performance_from_model(
                experiment.model,
                experiment.data_test,
                experiment.feature_cols,
                metric,
                experiment.target_assignments,
                positive_class=experiment.positive_class,
            )
            row_data[metric] = performance

        data_list.append(row_data)

    # Create DataFrame
    df = pd.DataFrame(data_list)

    # Set experiment_id as index
    df = df.set_index("outersplit")

    # Calculate mean for each metric and add as a new row
    mean_row = df.mean(axis=0)
    df.loc["Mean"] = mean_row

    display_table(df)
    return df


def _get_predictions_df(experiment: Any) -> pd.DataFrame:
    """Extract predictions and probabilities from an experiment.

    Args:
        experiment: Experiment object containing model and test data.

    Returns:
        DataFrame with row_id, prediction, probabilities, and target columns.
    """
    data_test = experiment.data_test
    feature_cols = experiment.feature_cols
    target_col = list(experiment.target_assignments.values())[0]

    pred_proba = experiment.model.predict_proba(data_test[feature_cols])
    # Get the index of the positive class
    positive_class_idx = list(experiment.model.classes_).index(experiment.positive_class)
    probabilities = (
        pred_proba[positive_class_idx] if isinstance(pred_proba, pd.DataFrame) else pred_proba[:, positive_class_idx]
    )

    return pd.DataFrame(
        {
            "row_id": data_test[experiment.row_column],
            "prediction": experiment.model.predict(data_test[feature_cols]),
            "probabilities": probabilities,
            "target": data_test[target_col],
        }
    )


def _create_roc_figure(
    fpr: np.ndarray, tpr: np.ndarray, auc_score: float, title: str, label: str, width: int, height: int
) -> go.Figure:
    """Create a Plotly ROC curve figure.

    Args:
        fpr: False positive rates.
        tpr: True positive rates.
        auc_score: AUC score.
        title: Plot title.
        label: Legend label for ROC curve.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line={"dash": "dash", "width": 2, "color": "gray"})
    )
    fig.add_trace(
        go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{label} (AUC = {auc_score:.3f})", line={"width": 2, "color": "blue"}
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=width,
        height=height,
        showlegend=True,
        legend={"x": 0.95, "y": 0.05, "xanchor": "right", "yanchor": "bottom"},
        xaxis={"range": [0, 1], "gridcolor": "rgba(128, 128, 128, 0.2)"},
        yaxis={"range": [0, 1], "gridcolor": "rgba(128, 128, 128, 0.2)"},
    )
    return fig


def plot_aucroc(
    predictor: "OctoPredict",
    figsize: tuple[int, int] = (8, 8),
    dpi: int = 100,
    show_individual: bool = False,
) -> None:
    """Plot ROC curves: merged predictions, averaged with confidence intervals, and optionally individual plots.

    This function creates comprehensive ROC curve visualizations:
    1. Merged plot: All predictions pooled into a single ROC curve
    2. Averaged plot: Mean ROC curve with ±1 std. dev. confidence bands
    3. Individual plots (optional): Separate ROC curve for each experiment

    Args:
        predictor: OctoPredict object containing experiments with trained models
            and test data. Must be for a classification task.
        figsize: Figure size as (width, height) in inches. Default is (8, 8).
        dpi: Dots per inch for the figure. Default is 100.
        show_individual: If True, plots individual ROC curves for each experiment
            in addition to merged and averaged plots. Default is False.

    Raises:
        ValueError: If the predictor is not for a classification task.

    Example:
        >>> from octopus.predict import OctoPredict
        >>> from octopus.predict.notebook_utils import plot_aucroc
        >>> predictor = OctoPredict(study_directory="./studies/my_study/", task_id=0)
        >>> plot_aucroc(predictor, show_individual=True)
    """
    # Validate classification task
    if next(iter(predictor.experiments.values())).ml_type != "classification":
        raise ValueError("AUCROC plots are only available for classification tasks")

    # Calculate figure dimensions (80% of specified size)
    width_px, height_px = int(figsize[0] * 80), int(figsize[1] * 80)

    # Collect predictions and compute ROC data for all experiments
    predictions_list = []
    roc_data = []  # Store (key, fpr, tpr, auc) for each experiment
    mean_fpr = np.linspace(0, 1, 100)

    for key, experiment in predictor.experiments.items():
        df_pred = _get_predictions_df(experiment)
        predictions_list.append(df_pred)

        fpr, tpr, _ = roc_curve(df_pred["target"], df_pred["probabilities"], drop_intermediate=True)
        auc_score = float(auc(fpr, tpr))
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0

        roc_data.append((key, fpr, tpr, auc_score, interp_tpr))

    # ===== PLOT 1: MERGED ROC CURVE =====
    print("=" * 60)
    print("1. MERGED ROC CURVE (All Predictions Pooled)")
    print("=" * 60)

    merged_df = pd.concat(predictions_list, axis=0)
    fpr_merged, tpr_merged, _ = roc_curve(merged_df["target"], merged_df["probabilities"], drop_intermediate=True)
    auc_merged = float(auc(fpr_merged, tpr_merged))
    print(f"Merged AUC-ROC: {auc_merged:.3f}\n")

    _create_roc_figure(
        fpr_merged, tpr_merged, auc_merged, "Merged ROC Curve (All Predictions Pooled)", "Merged", width_px, height_px
    ).show()

    # ===== PLOT 2: AVERAGED ROC CURVE =====
    print("\n" + "=" * 60)
    print("2. AVERAGED ROC CURVE (Mean ± Std. Dev.)")
    print("=" * 60)

    aucs = [auc_score for _, _, _, auc_score, _ in roc_data]
    tprs = np.array([interp_tpr for _, _, _, _, interp_tpr in roc_data])

    aucroc_mean, aucroc_std = float(np.mean(aucs)), float(np.std(aucs))
    print(f"Mean AUC-ROC: {aucroc_mean:.3f}")
    print(f"Std. Dev. AUC-ROC: {aucroc_std:.3f}\n")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    fig_avg = go.Figure()
    fig_avg.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line={"dash": "dash", "width": 2, "color": "gray"},
            opacity=0.8,
        )
    )
    fig_avg.add_trace(
        go.Scatter(
            x=np.concatenate([mean_fpr, mean_fpr[::-1]]),
            y=np.concatenate([tprs_upper, tprs_lower[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 0, 255, 0.2)",
            line={"color": "rgba(255,255,255,0)"},
            name="± 1 std. dev.",
        )
    )
    fig_avg.add_trace(
        go.Scatter(
            x=mean_fpr,
            y=mean_tpr,
            mode="lines",
            name=f"Mean ROC (AUC = {aucroc_mean:.3f} ± {aucroc_std:.3f})",
            line={"width": 2, "color": "blue"},
            opacity=0.8,
        )
    )
    fig_avg.update_layout(
        title="Averaged ROC Curve (All Outer Test Sets)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=width_px,
        height=height_px,
        showlegend=True,
        legend={"x": 0.95, "y": 0.05, "xanchor": "right", "yanchor": "bottom"},
        xaxis={"range": [0, 1], "gridcolor": "rgba(128, 128, 128, 0.2)"},
        yaxis={"range": [0, 1], "gridcolor": "rgba(128, 128, 128, 0.2)"},
    )
    fig_avg.show()

    # ===== PLOT 3 (OPTIONAL): INDIVIDUAL ROC CURVES =====
    if show_individual:
        print("\n" + "=" * 60)
        print("3. INDIVIDUAL ROC CURVES")
        print("=" * 60)

        for key, fpr, tpr, auc_score, _ in roc_data:
            print(f"\nExperiment {key}: AUC = {auc_score:.3f}")
            _create_roc_figure(
                fpr, tpr, auc_score, f"ROC Curve - Experiment {key}", f"Experiment {key}", width_px, height_px
            ).show()


def show_confusionmatrix(predictor: "OctoPredict", threshold: float = 0.5, metrics: list[str] | None = None) -> None:
    """Display confusion matrices and performance metrics for all experiments in a task predictor.

    This function evaluates each experiment's model on the test dataset, plots confusion matrices
    (both absolute and relative/percentage), and displays performance metrics for specified metrics.

    Args:
        predictor: OctoPredict object containing experiments with trained models.
        threshold: Probability threshold for binary classification (default: 0.5).
            Predictions above this value are classified as positive (class 1).
        metrics: List of metric names to evaluate (e.g., ['AUCROC', 'ACCBAL', 'ACC', 'F1']).
            If None, uses default metrics: ['AUCROC', 'ACCBAL', 'ACC', 'F1', 'AUCPR', 'NEGBRIERSCORE'].

    Raises:
        ValueError: If the task is not a classification task.

    Example:
        >>> from octopus.predict.notebook_utils import show_confusionmatrix
        >>> from octopus.predict import OctoPredict
        >>> predictor = OctoPredict.from_study("./studies/my_study/", task_id=0)
        >>> show_confusionmatrix(predictor, threshold=0.5, metrics=['AUCROC', 'ACC', 'F1'])
    """
    if metrics is None:
        metrics = ["AUCROC", "ACCBAL", "ACC", "F1", "AUCPR", "NEGBRIERSCORE"]

    # Verify it's a classification task
    if predictor.config.get("ml_type") != "classification":
        raise ValueError("show_confusionmatrix() is only applicable for classification tasks")

    # Initialize results dataframe
    df_results = pd.DataFrame(columns=["metric", "performance", "experiment_id"])

    print("=" * 80)
    print("CONFUSION MATRICES AND PERFORMANCE METRICS")
    print("=" * 80)
    print(f"Threshold: {threshold}")
    print(f"Metrics: {metrics}\n")

    # Iterate through experiments
    for exp_id, experiment in predictor.experiments.items():
        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT {exp_id}")
        print("=" * 60)

        # Extract experiment data
        data_test = experiment.data_test
        feature_cols = experiment.feature_cols
        target_col = list(experiment.target_assignments.values())[0]
        target = data_test[target_col]
        model = experiment.model

        # Get predicted probabilities for the positive class
        positive_class_idx = list(model.classes_).index(experiment.positive_class)  # type: ignore[attr-defined]
        model_proba = model.predict_proba(data_test[feature_cols])
        if isinstance(model_proba, pd.DataFrame):
            probabilities = model_proba[positive_class_idx].values
        elif isinstance(model_proba, np.ndarray):
            probabilities = model_proba[:, positive_class_idx]
        else:
            raise ValueError("Model predictions must be a DataFrame or NumPy array")

        # Apply threshold to get predicted labels
        predictions = (probabilities > threshold).astype(int)

        # Compute confusion matrices
        cm_abs = confusion_matrix(target, predictions)
        cm_rel = confusion_matrix(target, predictions, normalize="true")

        # Class labels for display
        class_names = ["0", "1"]

        # Create subplots with 1 row and 2 columns
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Confusion Matrix (Absolute)", "Confusion Matrix (Relative %)"],
            horizontal_spacing=0.25,
        )

        # Add absolute confusion matrix heatmap
        cm_rel_text = [[f"{val:.1f}%" for val in row] for row in cm_rel * 100]
        cm_abs_max = float(cm_abs.max())
        fig.add_trace(
            go.Heatmap(
                z=cm_abs,
                x=class_names,  # type: ignore[arg-type]
                y=class_names,  # type: ignore[arg-type]
                text=cm_abs,
                texttemplate="%{text}",
                textfont={"size": 12},
                colorscale="Blues",
                showscale=True,
                zmin=0,
                zmax=cm_abs_max,
                colorbar={
                    "x": 0.42,
                    "len": 0.75,
                    "thickness": 15,
                    "showticklabels": True,
                },
            ),
            row=1,
            col=1,
        )

        # Add relative confusion matrix heatmap
        fig.add_trace(
            go.Heatmap(
                z=cm_rel * 100,
                x=class_names,  # type: ignore[arg-type]
                y=class_names,  # type: ignore[arg-type]
                text=cm_rel_text,  # type: ignore[arg-type]
                texttemplate="%{text}",
                textfont={"size": 12},
                colorscale="Blues",
                showscale=True,
                zmin=0,
                zmax=100,
                colorbar={
                    "x": 1.05,
                    "len": 0.75,
                    "thickness": 15,
                    "showticklabels": True,
                },
            ),
            row=1,
            col=2,
        )

        # Update axes
        fig.update_xaxes(title_text="Predicted Label", row=1, col=1, side="bottom")
        fig.update_yaxes(title_text="True Label", row=1, col=1, autorange="reversed")
        fig.update_xaxes(title_text="Predicted Label", row=1, col=2, side="bottom")
        fig.update_yaxes(title_text="True Label", row=1, col=2, autorange="reversed")

        # Update layout
        fig.update_layout(
            title_text=f"Confusion Matrices - Experiment {exp_id}",
            width=900,
            height=420,
            showlegend=False,
        )

        # Display confusion matrices
        print("\nConfusion Matrices:")
        fig.show()

        # Calculate and display metrics
        print("\nPerformance Metrics:")
        for metric in metrics:
            performance = get_performance_from_model(
                model,
                data_test,
                feature_cols,
                metric,
                experiment.target_assignments,
                positive_class=experiment.positive_class,
                threshold=threshold,
            )
            print(f"  {metric:<15}: {performance:.4f}")
            # Create new row explicitly
            new_row_data = {"metric": metric, "performance": performance, "experiment_id": exp_id}
            df_results = pd.concat([df_results, pd.DataFrame([new_row_data])], ignore_index=True)

    # Display overall performance summary
    print(f"\n{'=' * 80}")
    print("OVERALL PERFORMANCE (Mean across all experiments)")
    print("=" * 80)
    performance_mean = df_results.groupby("metric").mean()["performance"]
    for metric_name, value in performance_mean.items():
        metric_str = str(metric_name)
        print(f"  {metric_str:<15}: {value:.4f}")


def show_overall_fi_table(predictor: "OctoPredict", fi_type: str = "group_permutation") -> pd.DataFrame:
    """Calculate and display overall feature importance table across all experiments.

    This function aggregates feature importance values from all experiments, averaging
    them and sorting by importance. It handles feature groups by joining group names.

    Args:
        predictor: OctoPredict object containing experiments and results.
        fi_type: Type of feature importance to extract. Options:
            - "group_permutation": Permutation feature importance with feature groups
            - "shap": SHAP feature importance
            - "permutation": Standard permutation feature importance
            Default: "group_permutation"


    Returns:
        DataFrame with columns ['feature', 'importance'] sorted by importance (descending).

    Raises:
        ValueError: If no feature importance results are found for the specified fi_type.

    Example:
        >>> from octopus.predict import OctoPredict
        >>> from octopus.predict.notebook_utils import show_overall_fi_table
        >>> predictor = OctoPredict.from_study("./studies/my_study/", task_id=0)
        >>> fi_table = show_overall_fi_table(predictor, fi_type="group_permutation")
    """
    # Get matching result keys
    matching_keys = [key for key in predictor.results if fi_type in key]
    if not matching_keys:
        raise ValueError(
            f"No feature importance results found for fi_type='{fi_type}'. "
            f"Available keys: {list(predictor.results.keys())}"
        )

    # Collect and process feature importance dataframes
    df_lst = []
    for key in matching_keys:
        # Extract experiment ID using regex
        match = re.search(r"\d+", key)
        if not match:
            continue
        exp_id = int(match.group())

        # Get feature importance dataframe
        df_fi = predictor.results[key].copy()

        # Replace features with group names if feature groups exist
        if exp_id in predictor.experiments and hasattr(predictor.experiments[exp_id], "feature_group_dict"):
            feature_groups = predictor.experiments[exp_id].feature_group_dict
            if feature_groups:

                def replace_feature(f: str, groups: dict = feature_groups) -> str:
                    return "_".join(groups[f]) if f in groups else f

                df_fi["feature"] = df_fi["feature"].apply(replace_feature)

        df_lst.append(df_fi)

    # Aggregate feature importance across experiments
    df_all = pd.concat(df_lst, axis=0)
    df_aggregated: pd.DataFrame = (
        df_all.groupby("feature")["importance"]
        .sum()
        .div(len(matching_keys))
        .reset_index()
        .sort_values(by="importance", ascending=False)
    )

    return df_aggregated


def show_overall_fi_plot(
    predictor: "OctoPredict",
    fi_type: str = "group_permutation",
    top_n: int | None = None,
) -> None:
    """Create and display a bar plot of overall feature importance.

    This function creates an interactive plotly bar chart showing feature importance
    averaged across all experiments.

    Args:
        predictor: OctoPredict object containing experiments and results.
        fi_type: Type of feature importance to plot. Options:
            - "group_permutation": Permutation feature importance with feature groups
            - "shap": SHAP feature importance
            - "permutation": Standard permutation feature importance
            Default: "group_permutation"
        top_n: Number of top features to display. If None, shows all features.
            Default: None (show all)

    Example:
        >>> from octopus.predict import OctoPredict
        >>> from octopus.predict.notebook_utils import show_overall_fi_plot
        >>> predictor = OctoPredict.from_study("./studies/my_study/", task_id=0)
        >>> show_overall_fi_plot(predictor, fi_type="group_permutation", top_n=20)
    """
    # Get feature importance table
    df = show_overall_fi_table(predictor, fi_type=fi_type)

    # Filter to top N features if specified
    if top_n is not None:
        df = df.head(top_n)
        title = f"Top {top_n} Feature Importances ({fi_type})"
    else:
        title = f"Feature Importances ({fi_type})"

    # Create and display plotly bar chart
    fig = go.Figure(
        data=go.Bar(
            x=df["feature"],
            y=df["importance"],
            marker={"color": "royalblue"},
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="Importance Value",
        xaxis={"tickangle": -45, "tickfont": {"size": 10}},
        height=600,
        width=max(800, len(df) * 20),
        showlegend=False,
    )

    fig.show()
