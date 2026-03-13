"""Utility functions for Jupyter notebooks — predict package version.

Provides high-level analysis functions for Jupyter notebooks:
- Study-level functions delegate data loading to StudyLoader
- Task-level functions use TaskPredictorTest for test-data analysis
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import auc, confusion_matrix, roc_curve
from upath import UPath

from octopus.predict.study_io import StudyLoader
from octopus.types import MLType

__all__ = [
    "display_table",
    "find_latest_study",
    "show_aucroc_plots",
    "show_confusionmatrix",
    "show_overall_fi_plot",
    "show_overall_fi_table",
    "show_selected_features",
    "show_study_details",
    "show_target_metric_performance",
    "show_testset_performance",
]


def find_latest_study(studies_root: str | UPath, prefix: str) -> str:
    """Find the latest study directory matching a name prefix.

    Study directories are named ``<prefix>-YYYYMMDD_HHMMSS``.  This function
    finds all directories matching the given *prefix* and returns the one with
    the most recent timestamp (lexicographic sort).  Falls back to an exact
    match (no timestamp suffix) when no timestamped directories are found.

    Args:
        studies_root: Path to the parent directory containing study directories.
        prefix: The study name prefix, e.g. ``"wf_octo_mrmr_octo"``.

    Returns:
        Path string to the latest matching study directory.

    Raises:
        FileNotFoundError: If no matching study directory is found.

    Example:
        >>> from octopus.predict.notebook_utils import find_latest_study
        >>> study_dir = find_latest_study("./studies", "wf_octo_mrmr_octo")
    """
    root = UPath(studies_root)
    # Match timestamped directories: prefix-YYYYMMDD_HHMMSS
    _timestamp_pattern = re.compile(re.escape(prefix) + r"-\d{8}_\d{6}$")
    candidates = sorted(
        [d for d in root.glob(f"{prefix}-*") if d.is_dir() and _timestamp_pattern.search(d.name)],
        key=lambda p: p.name,
        reverse=True,
    )
    if candidates:
        return str(candidates[0])
    # Fallback: exact match without timestamp
    exact = root / prefix
    if exact.is_dir():
        return str(exact)
    raise FileNotFoundError(f"No study directory found for prefix '{prefix}' in {root}")


try:
    from IPython.display import display as ipython_display
except ImportError:
    ipython_display = None

if TYPE_CHECKING:
    from octopus.predict.task_predictor_test import TaskPredictorTest


def display_table(data: Any) -> None:
    """Display a table in Jupyter notebooks or print in other environments.

    Args:
        data: The data to display (DataFrame, Series, or any printable object).

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


# ═══════════════════════════════════════════════════════════════
# STUDY-LEVEL FUNCTIONS (delegate to StudyLoader)
# ═══════════════════════════════════════════════════════════════


def _validate_study_structure(study_path: UPath, config: dict) -> dict:
    """Validate the study directory structure against the configuration.

    Checks that expected outersplit directories and workflow task directories
    exist.  Returns a dict of validation results without printing.

    Args:
        study_path: UPath to the study directory.
        config: Study configuration dictionary (from ``StudyLoader.load_config()``).

    Returns:
        Dictionary containing validation results with keys:
            - 'outersplit_dirs': List of found outersplit directory paths
            - 'expected_task_ids': List of expected task IDs
            - 'octo_workflow_tasks': List of task IDs for octo modules
            - 'missing_outersplits': List of missing outersplit IDs
            - 'missing_workflow_dirs': List of missing workflow directories

    Raises:
        ValueError: If no outersplit directories or workflow results are found.
    """
    n_folds_outer = config["n_folds_outer"]
    workflow_tasks = config["workflow"]

    outersplit = sorted(
        [d for d in study_path.glob("outersplit*") if d.is_dir()],
        key=lambda x: int(x.name.replace("outersplit", "")),
    )

    if not outersplit:
        raise ValueError(
            f"No outersplit directories found in study path.\n"
            f"Study path: {study_path}\nThe study may not have been run yet."
        )

    expected_outersplit_ids = list(range(n_folds_outer))
    missing_outersplits = [
        split_id for split_id in expected_outersplit_ids if not (study_path / f"outersplit{split_id}").exists()
    ]

    expected_task_ids = [task["task_id"] for task in workflow_tasks]

    has_results = False
    missing_workflow_dirs: list[str] = []

    for split_dir in outersplit:
        workflow_dirs = list(split_dir.glob("task*"))
        if workflow_dirs:
            has_results = True
        for task_id in expected_task_ids:
            expected_dir = split_dir / f"task{task_id}"
            if not expected_dir.exists():
                missing_workflow_dirs.append(f"{split_dir.name}/task{task_id}")

    if not has_results:
        raise ValueError("No workflow results found in outersplits.\nThe study may not have completed successfully.")

    octo_workflow_lst = [item["task_id"] for item in workflow_tasks if item["module"] == "octo"]

    return {
        "outersplit_dirs": outersplit,
        "expected_task_ids": expected_task_ids,
        "octo_workflow_tasks": octo_workflow_lst,
        "missing_outersplits": missing_outersplits,
        "missing_workflow_dirs": missing_workflow_dirs,
    }


def show_study_details(study_directory: str | UPath, verbose: bool = True) -> dict:
    """Display and validate study details including configuration and structure.

    This function reads the study configuration via ``StudyLoader``, validates
    the study structure, and displays information about the workflow tasks
    and outersplit directories.

    Args:
        study_directory: Path to the study directory.
        verbose: If True, prints detailed information.

    Returns:
        Dictionary containing study information with keys:
            - 'path': UPath object of the study directory
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
        ValueError: If no outersplit directories or workflow results are found.
        FileNotFoundError: If the study directory does not exist.

    Example:
        >>> from octopus.predict.notebook_utils import show_study_details
        >>> study_info = show_study_details("./studies/my_study/")
    """
    path_study = UPath(study_directory)

    if not path_study.exists():
        raise FileNotFoundError(f"Study path does not exist: {path_study}")

    if verbose:
        print(f"Selected study path: {path_study}\n")

    # Delegate config loading to I/O layer
    loader = StudyLoader(path_study)
    config = loader.load_config()

    ml_type = config["ml_type"]
    n_folds_outer = config["n_folds_outer"]
    workflow_tasks = config["workflow"]

    if verbose:
        print("Validate study....")
        print(f"ML Type: {ml_type}")

    # Delegate structure validation
    validation = _validate_study_structure(path_study, config)
    outersplit = validation["outersplit_dirs"]
    missing_outersplits = validation["missing_outersplits"]
    missing_workflow_dirs = validation["missing_workflow_dirs"]
    expected_task_ids = validation["expected_task_ids"]
    octo_workflow_lst = validation["octo_workflow_tasks"]

    if verbose:
        print(f"Found {len(outersplit)} outersplit directory/directories")
        expected_outersplit_ids = list(range(n_folds_outer))
        print(f"Expected outersplit IDs: {expected_outersplit_ids}")

        if missing_outersplits:
            for split_id in missing_outersplits:
                print(f"  WARNING: Missing directory 'outersplit{split_id}'")
            print(f"  {len(missing_outersplits)} outersplit directory/directories missing")
        else:
            print("All expected outersplit directories found")

        print(f"Expected workflow task IDs: {expected_task_ids}")

        if missing_workflow_dirs:
            for missing_dir in missing_workflow_dirs:
                split_name, task_name = missing_dir.split("/")
                print(f"  WARNING: Missing directory '{task_name}' in {split_name}")
            print("Study has completed workflow tasks, but some directories are missing (see warnings above)")
        else:
            print("Study has completed workflow tasks - all expected directories found")

        print("\nInformation on workflow tasks in this study")
        print(f"Number of workflow tasks: {len(workflow_tasks)}")

        for _item in workflow_tasks:
            print(f"Task {_item['task_id']}: {_item['module']}")

        print(f"Octo workflow tasks: {octo_workflow_lst}")

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


def show_target_metric_performance(study_info: dict, details: bool = False) -> list[pd.DataFrame]:
    """Display performance metrics for all workflow tasks in a study.

    Delegates data loading to ``StudyLoader.build_performance_summary()``.

    Args:
        study_info: Dictionary returned by show_study_details().
        details: If True, shows detailed information for each outersplit.

    Returns:
        List of DataFrames, one for each task/key combination with performance metrics.

    Example:
        >>> study_info = show_study_details("./studies/my_study/")
        >>> tables = show_target_metric_performance(study_info, details=False)
    """
    loader = StudyLoader(study_info["path"])
    df = loader.build_performance_summary()
    performance_tables = []

    for _item in study_info["workflow_tasks"]:
        print(f"\033[1mWorkflow task: {_item['task_id']}\033[0m")

        df_workflow = df[df["Task"] == _item["task_id"]]
        res_keys = sorted(set(df_workflow["Results_key"].tolist()))
        print("Available results keys:", res_keys)

        for _key in res_keys:
            print("Selected results key:", _key)
            df_workflow_selected = df_workflow[df_workflow["Results_key"] == _key].copy()
            perf_expanded = df_workflow_selected["Performance_dict"].apply(pd.Series)
            result_df = df_workflow_selected[["OuterSplit"]].join(perf_expanded).set_index("OuterSplit")
            result_df = result_df.select_dtypes(include="number")
            result_df.insert(0, "Task", _item["task_id"])
            result_df.insert(1, "Key", _key)

            mean_values = {}
            for column in result_df.columns:
                if result_df[column].dtype in ["float64", "int64"]:
                    mean_values[column] = result_df[column].mean()
                else:
                    mean_values[column] = ""
            mean_values["Task"] = _item["task_id"]
            mean_values["Key"] = _key
            result_df.loc["Mean"] = mean_values

            performance_tables.append(result_df.copy())
            display_table(result_df)

    return performance_tables


def show_selected_features(
    study_info: dict, sort_task: int | None = None, sort_key: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Display the number of selected features across outer splits, tasks, and result keys.

    Delegates data loading to ``StudyLoader.build_feature_summary()``
    and ``StudyLoader.build_performance_summary()``.

    Args:
        study_info: Dictionary returned by show_study_details().
        sort_task: Task ID to use for sorting the frequency table.
        sort_key: Results key to use for sorting the frequency table.

    Returns:
        Tuple of three DataFrames:
        - feature_table: Number of features per outer split for each task-key combination
        - frequency_table: Feature frequency across outersplits
        - raw_feature_table: Raw performance dataframe

    Example:
        >>> study_info = show_study_details("./studies/my_study/")
        >>> feat_table, freq_table, raw_table = show_selected_features(study_info)
    """
    loader = StudyLoader(study_info["path"])
    raw_feature_table = loader.build_performance_summary()
    feature_table, frequency_table = loader.build_feature_summary(sort_task=sort_task, sort_key=sort_key)

    print("\n" + "=" * 40)
    print("NUMBER OF SELECTED FEATURES")
    print("=" * 40)
    print("Rows: OuterSplit | Columns: (Task, Key) | Values: Number of Features")
    display_table(feature_table)

    print("\n" + "=" * 40)
    print("FEATURE FREQUENCY ACROSS OUTER SPLITS")
    print("=" * 40)
    print("Rows: Features | Columns: (Task, Key) | Values: Feature Frequency")
    if sort_key:
        print(f"Sorted by Task {sort_task}, Key '{sort_key}' frequency (highest first)")
    display_table(frequency_table)

    return feature_table, frequency_table, raw_feature_table


# ═══════════════════════════════════════════════════════════════
# TASK-LEVEL FUNCTIONS (take TaskPredictorTest)
# ═══════════════════════════════════════════════════════════════


def show_testset_performance(
    predictor: TaskPredictorTest,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Display test performance metrics across all outersplits for a task.

    Args:
        predictor: TaskPredictorTest instance for the task to evaluate.
        metrics: List of metric names to evaluate. If None, uses defaults
            based on ML type.

    Returns:
        DataFrame with outersplits as rows (plus a 'Mean' row), metrics as columns.

    Example:
        >>> from octopus.predict import TaskPredictorTest
        >>> tp = TaskPredictorTest("./studies/my_study/", task_id=0)
        >>> df = show_testset_performance(tp, metrics=["AUCROC", "ACCBAL", "ACC"])
    """
    if metrics is None:
        if predictor.ml_type == MLType.BINARY:
            metrics = ["AUCROC", "ACCBAL", "ACC"]
        elif predictor.ml_type == MLType.REGRESSION:
            metrics = ["R2", "MAE", "RMSE"]
        else:
            metrics = []

    print("Performance on test dataset (pooling)")

    performance_long = predictor.performance(metrics=metrics)
    df = performance_long.pivot(index="outersplit", columns="metric", values="score")

    # Reorder columns to match metrics order
    available_cols = [m for m in metrics if m in df.columns]
    df = df[available_cols]

    # Add Mean row
    df.loc["Mean"] = df.mean()

    display_table(df)
    return df


def _create_roc_figure(
    fpr: np.ndarray, tpr: np.ndarray, auc_score: float, title: str, label: str, width: int, height: int
) -> go.Figure:
    """Create a Plotly ROC curve figure.

    Args:
        fpr: False positive rates.
        tpr: True positive rates.
        auc_score: AUC score.
        title: Plot title.
        label: Legend label.
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


def show_aucroc_plots(
    predictor: TaskPredictorTest,
    figsize: tuple[int, int] = (8, 8),
    show_individual: bool = False,
) -> None:
    """Plot ROC curves: merged, averaged with confidence bands, and optionally individual.

    Uses ``predictor.predict_proba(df=True)`` to get per-split probabilities
    and targets in a single call, avoiding manual access to models/data.

    1. Merged ROC curve (all predictions pooled)
    2. Averaged ROC curve (mean +/- 1 std dev)
    3. Individual ROC curves per outersplit (if show_individual=True)

    Args:
        predictor: TaskPredictorTest instance for a classification task.
        figsize: Figure size as (width, height) in inches.
        show_individual: If True, also plot individual ROC curves per outersplit.

    Raises:
        ValueError: If the task is not for classification.

    Example:
        >>> tp = TaskPredictorTest("./studies/my_study/", task_id=0)
        >>> show_aucroc_plots(tp, show_individual=True)
    """
    if predictor.ml_type != MLType.BINARY:
        raise ValueError("AUCROC plots are only available for binary classification tasks")

    width_px, height_px = int(figsize[0] * 80), int(figsize[1] * 80)

    # Get all per-split probabilities + targets in one call
    proba_df = predictor.predict_proba(df=True)
    positive_class = predictor.positive_class

    roc_data = []
    mean_fpr = np.linspace(0, 1, 100)

    for split_id in predictor.outersplits:
        split_df = proba_df[proba_df["outersplit"] == split_id]
        probabilities = np.asarray(split_df[positive_class])
        target = np.asarray(split_df["target"])

        fpr, tpr, _ = roc_curve(target, probabilities, drop_intermediate=True)
        auc_score = float(auc(fpr, tpr))
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        roc_data.append((split_id, fpr, tpr, auc_score, interp_tpr))

    # === PLOT 1: MERGED ROC CURVE ===
    print("=" * 60)
    print("1. MERGED ROC CURVE (All Predictions Pooled)")
    print("=" * 60)

    all_probabilities = np.asarray(proba_df[positive_class])
    all_targets = np.asarray(proba_df["target"])
    fpr_merged, tpr_merged, _ = roc_curve(all_targets, all_probabilities, drop_intermediate=True)
    auc_merged = float(auc(fpr_merged, tpr_merged))
    print(f"Merged AUC-ROC: {auc_merged:.3f}\n")

    _create_roc_figure(
        fpr_merged, tpr_merged, auc_merged, "Merged ROC Curve (All Predictions Pooled)", "Merged", width_px, height_px
    ).show()

    # === PLOT 2: AVERAGED ROC CURVE ===
    print("\n" + "=" * 60)
    print("2. AVERAGED ROC CURVE (Mean +/- Std. Dev.)")
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
            name="+/- 1 std. dev.",
        )
    )
    fig_avg.add_trace(
        go.Scatter(
            x=mean_fpr,
            y=mean_tpr,
            mode="lines",
            name=f"Mean ROC (AUC = {aucroc_mean:.3f} +/- {aucroc_std:.3f})",
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

    # === PLOT 3 (OPTIONAL): INDIVIDUAL ROC CURVES ===
    if show_individual:
        print("\n" + "=" * 60)
        print("3. INDIVIDUAL ROC CURVES")
        print("=" * 60)

        for key, fpr, tpr, auc_score, _ in roc_data:
            print(f"\nOutersplit {key}: AUC = {auc_score:.3f}")
            _create_roc_figure(
                fpr, tpr, auc_score, f"ROC Curve - Outersplit {key}", f"Outersplit {key}", width_px, height_px
            ).show()


def _compute_confusion_matrices(
    target: np.ndarray | pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute absolute and relative confusion matrices.

    Args:
        target: True labels.
        probabilities: Predicted probabilities for the positive class.
        threshold: Classification threshold.

    Returns:
        Tuple of (absolute confusion matrix, relative confusion matrix).
    """
    predictions = (np.asarray(probabilities) > threshold).astype(int)
    cm_abs = confusion_matrix(target, predictions)
    cm_rel = confusion_matrix(target, predictions, normalize="true")
    return cm_abs, cm_rel


def _create_confusion_figure(
    cm_abs: np.ndarray,
    cm_rel: np.ndarray,
    class_names: list[str],
    title: str,
) -> go.Figure:
    """Create a Plotly figure with absolute and relative confusion matrices side-by-side.

    Args:
        cm_abs: Absolute confusion matrix.
        cm_rel: Relative (normalized) confusion matrix.
        class_names: List of class label strings.
        title: Figure title.

    Returns:
        Plotly Figure with two heatmap subplots.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Confusion Matrix (Absolute)", "Confusion Matrix (Relative %)"],
        horizontal_spacing=0.25,
    )

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

    fig.update_xaxes(title_text="Predicted Label", row=1, col=1, side="bottom")
    fig.update_yaxes(title_text="True Label", row=1, col=1, autorange="reversed")
    fig.update_xaxes(title_text="Predicted Label", row=1, col=2, side="bottom")
    fig.update_yaxes(title_text="True Label", row=1, col=2, autorange="reversed")

    fig.update_layout(
        title_text=title,
        width=900,
        height=420,
        showlegend=False,
    )

    return fig


def show_confusionmatrix(
    predictor: TaskPredictorTest,
    threshold: float = 0.5,
    metrics: list[str] | None = None,
) -> None:
    """Display confusion matrices and performance metrics for all outersplits.

    Uses ``predictor.predict_proba(df=True)`` to get per-split probabilities
    and targets, avoiding manual access to models/data/features.

    Shows absolute and relative confusion matrices side-by-side (plotly subplots)
    plus performance metrics for each outersplit and overall mean.

    Args:
        predictor: TaskPredictorTest instance for a classification task.
        threshold: Probability threshold for binary classification.
        metrics: List of metric names to evaluate. If None, uses defaults.

    Raises:
        ValueError: If the task is not a classification task.

    Example:
        >>> tp = TaskPredictorTest("./studies/my_study/", task_id=0)
        >>> show_confusionmatrix(tp, threshold=0.5, metrics=["AUCROC", "ACC", "F1"])
    """
    if metrics is None:
        metrics = ["AUCROC", "ACCBAL", "ACC", "F1", "AUCPR", "NEGBRIERSCORE"]

    if predictor.ml_type != MLType.BINARY:
        raise ValueError("show_confusionmatrix() is only applicable for binary classification tasks")

    # Get all per-split probabilities + targets in one call
    proba_df = predictor.predict_proba(df=True)
    positive_class = predictor.positive_class

    result_rows: list[dict[str, Any]] = []

    # Compute all scores once (covers all outersplits) instead of calling
    # predictor.performance() inside the loop for each outersplit.
    all_scores = predictor.performance(metrics=metrics, threshold=threshold)

    print("=" * 80)
    print("CONFUSION MATRICES AND PERFORMANCE METRICS")
    print("=" * 80)
    print(f"Threshold: {threshold}")
    print(f"Metrics: {metrics}\n")

    for outersplit_id in predictor.outersplits:
        print(f"\n{'=' * 60}")
        print(f"OUTERSPLIT {outersplit_id}")
        print("=" * 60)

        split_df = proba_df[proba_df["outersplit"] == outersplit_id]
        probabilities = np.asarray(split_df[positive_class])
        target = np.asarray(split_df["target"])

        cm_abs, cm_rel = _compute_confusion_matrices(target, probabilities, threshold)

        class_names = [str(c) for c in predictor.classes_]

        fig = _create_confusion_figure(cm_abs, cm_rel, class_names, f"Confusion Matrices - Outersplit {outersplit_id}")

        print("\nConfusion Matrices:")
        fig.show()

        # Filter pre-computed scores to this outersplit.  Threshold was passed
        # through to get_performance_from_model() so that prediction-type
        # metrics (ACC, F1) use the same threshold as the confusion matrix.
        split_scores = all_scores[all_scores["outersplit"] == outersplit_id]

        print("\nPerformance Metrics:")
        for _, row in split_scores.iterrows():
            print(f"  {row['metric']:<15}: {row['score']:.4f}")
            result_rows.append({"metric": row["metric"], "performance": row["score"], "outersplit_id": outersplit_id})

    df_results = pd.DataFrame(result_rows)

    print(f"\n{'=' * 80}")
    print("OVERALL PERFORMANCE (Mean across all outersplits)")
    print("=" * 80)
    performance_mean = df_results.groupby("metric")["performance"].mean()
    for metric_name, value in performance_mean.items():
        metric_str = str(metric_name)
        print(f"  {metric_str:<15}: {value:.4f}")


# ═══════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════


def show_overall_fi_table(
    fi_table: pd.DataFrame,
) -> pd.DataFrame:
    """Display ensemble feature importance table.

    Filters to ensemble rows (``fi_source == "ensemble"``) for the overview.

    Args:
        fi_table: DataFrame returned by ``calculate_fi()``, containing
            per-split and ensemble rows with a ``fi_source`` column.

    Returns:
        DataFrame with ensemble feature importance results sorted by importance
        (descending).

    Example:
        >>> tp = TaskPredictorTest("./studies/my_study/", task_id=0)
        >>> fi_perm = tp.calculate_fi(fi_type="group_permutation", n_repeats=3)
        >>> fi_ensemble = show_overall_fi_table(fi_perm)
    """
    # Filter to ensemble rows for the overview
    if "fi_source" in fi_table.columns:
        ensemble_df = fi_table[fi_table["fi_source"] == "ensemble"].copy()
        ensemble_df = ensemble_df.sort_values("importance_mean", ascending=False)
        return ensemble_df.reset_index(drop=True)
    return fi_table


def show_overall_fi_plot(
    fi_table: pd.DataFrame,
    top_n: int | None = None,
) -> None:
    """Display bar chart of ensemble feature importance.

    Filters to ensemble rows (``fi_source == "ensemble"``) for the plot.

    Args:
        fi_table: DataFrame returned by ``calculate_fi()``, containing
            per-split and ensemble rows with a ``fi_source`` column.
        top_n: Number of top features to display. None shows all.

    Example:
        >>> tp = TaskPredictorTest("./studies/my_study/", task_id=0)
        >>> fi_perm = tp.calculate_fi(fi_type="group_permutation", n_repeats=3)
        >>> show_overall_fi_plot(fi_perm, top_n=20)
    """
    fi_df = fi_table.copy()

    # Filter to ensemble rows for the overview plot
    if "fi_source" in fi_df.columns:
        fi_df = fi_df[fi_df["fi_source"] == "ensemble"].copy()

    # Determine fi_type from the DataFrame for the title
    fi_type = str(fi_df["fi_type"].iloc[0]) if "fi_type" in fi_df.columns and not fi_df.empty else "unknown"

    if top_n is not None:
        fi_df = fi_df.head(top_n)
        title = f"Top {top_n} Feature Importances ({fi_type})"
    else:
        title = f"Feature Importances ({fi_type})"

    # Determine the importance column name (handle both naming conventions)
    if "importance_mean" in fi_df.columns:
        importance_col = "importance_mean"
    elif "importance" in fi_df.columns:
        importance_col = "importance"
    else:
        raise ValueError(
            f"FI DataFrame has no 'importance_mean' or 'importance' column. Columns: {list(fi_df.columns)}"
        )

    # Build error bars if CI columns are available (skip for ensemble rows
    # where CI values are NaN)
    error_y_config = None
    if "ci_lower" in fi_df.columns and "ci_upper" in fi_df.columns:
        ci_lower_vals = fi_df["ci_lower"]
        ci_upper_vals = fi_df["ci_upper"]
        if not ci_lower_vals.isna().all():
            error_y_config = {
                "type": "data",
                "symmetric": False,
                "array": (ci_upper_vals - fi_df[importance_col]).values,
                "arrayminus": (fi_df[importance_col] - ci_lower_vals).values,
            }

    fig = go.Figure(
        data=go.Bar(
            x=fi_df["feature"],
            y=fi_df[importance_col],
            marker={"color": "royalblue"},
            error_y=error_y_config,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="Importance Value",
        xaxis={"tickangle": -45, "tickfont": {"size": 10}},
        height=600,
        width=max(800, len(fi_df) * 20),
        showlegend=False,
    )

    fig.show()
