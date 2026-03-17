"""Plotly chart functions for diagnostics — replaces Altair charts from evaluation notebooks."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

from octopus.types import FIResultLabel, MetricDirection


def plot_feature_importance_chart(
    df: pd.DataFrame,
    *,
    outersplit_id: int | str = 0,
    task_id: int | str = 0,
    training_id: str = "",
    fi_method: str | FIResultLabel = "",
) -> go.Figure:
    """Create a feature importance bar chart.

    Args:
        df: Feature importances DataFrame with columns:
            feature, importance, fi_method, training_id, outersplit_id, task_id.
        outersplit_id: Outer split to filter on.
        task_id: Task to filter on.
        training_id: Training ID to filter on. If empty, uses first available.
        fi_method: FI method to filter on. If empty, uses first available.

    Returns:
        Plotly Figure.
    """
    mask = (df["outersplit_id"] == int(outersplit_id)) & (df["task_id"] == int(task_id))
    if training_id:
        mask &= df["training_id"] == training_id
    if fi_method:
        mask &= df["fi_method"] == fi_method

    filtered = df[mask].copy()
    if filtered.empty:
        fig = go.Figure()
        fig.update_layout(title="No data for selected filters")
        return fig

    filtered = filtered.sort_values("importance", ascending=False)

    fig = go.Figure(
        data=go.Bar(
            x=filtered["feature"],
            y=filtered["importance"],
            marker={"color": "royalblue"},
            hovertemplate="Feature: %{x}<br>Importance: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Feature",
        yaxis_title="Importance",
        xaxis={"tickangle": -45, "tickfont": {"size": 10}},
        width=700,
        height=450,
    )
    return fig


def plot_confusion_matrix_chart(
    df_predictions: pd.DataFrame,
    *,
    outersplit_id: int | str = 0,
    task_id: int | str = 0,
    training_id: str = "",
) -> go.Figure:
    """Create a confusion matrix heatmap from saved predictions.

    Args:
        df_predictions: Predictions DataFrame with columns:
            prediction, target, partition, outersplit_id, task_id, inner_split_id.
        outersplit_id: Outer split to filter on.
        task_id: Task to filter on.
        training_id: Training ID / inner_split_id to filter on.

    Returns:
        Plotly Figure with absolute and relative confusion matrices.
    """
    mask = (
        (df_predictions["outersplit_id"] == int(outersplit_id))
        & (df_predictions["task_id"] == int(task_id))
        & (df_predictions["partition"] == "test")
    )
    if training_id and "inner_split_id" in df_predictions.columns:
        mask &= df_predictions["inner_split_id"].astype(str) == str(training_id)

    filtered = df_predictions[mask]
    if filtered.empty:
        fig = go.Figure()
        fig.update_layout(title="No test data for selected filters")
        return fig

    y_true = filtered["target"].to_numpy()
    y_pred = filtered["prediction"].astype(int).to_numpy()
    class_labels = sorted(set(y_true) | set(y_pred))
    class_names = [str(c) for c in class_labels]

    cm_abs = confusion_matrix(y_true, y_pred, labels=class_labels)
    cm_rel = confusion_matrix(y_true, y_pred, labels=class_labels, normalize="true")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Confusion Matrix (Absolute)", "Confusion Matrix (Relative %)"],
        horizontal_spacing=0.25,
    )

    cm_rel_text = [[f"{val:.1f}%" for val in row] for row in cm_rel * 100]
    cm_abs_max = float(cm_abs.max()) if cm_abs.size > 0 else 1

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
            colorbar={"x": 0.42, "len": 0.75, "thickness": 15},
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
            colorbar={"x": 1.05, "len": 0.75, "thickness": 15},
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Predicted Label", row=1, col=1, side="bottom")
    fig.update_yaxes(title_text="True Label", row=1, col=1, autorange="reversed")
    fig.update_xaxes(title_text="Predicted Label", row=1, col=2, side="bottom")
    fig.update_yaxes(title_text="True Label", row=1, col=2, autorange="reversed")

    fig.update_layout(
        title_text=f"Confusion Matrix — outersplit {outersplit_id}, task {task_id}",
        width=900,
        height=420,
        showlegend=False,
    )
    return fig


def plot_predictions_vs_truth_chart(
    df_predictions: pd.DataFrame,
    *,
    outersplit_id: int | str = 0,
    task_id: int | str = 0,
    training_id: str = "",
) -> go.Figure:
    """Create a prediction vs ground truth scatter plot (regression).

    Args:
        df_predictions: Predictions DataFrame.
        outersplit_id: Outer split to filter on.
        task_id: Task to filter on.
        training_id: Training ID / inner_split_id to filter on.

    Returns:
        Plotly Figure.
    """
    mask = (df_predictions["outersplit_id"] == int(outersplit_id)) & (df_predictions["task_id"] == int(task_id))
    if training_id and "inner_split_id" in df_predictions.columns:
        mask &= df_predictions["inner_split_id"].astype(str) == str(training_id)

    filtered = df_predictions[mask]
    if filtered.empty:
        fig = go.Figure()
        fig.update_layout(title="No data for selected filters")
        return fig

    # Color by partition (train/test/dev)
    partitions = filtered["partition"].unique() if "partition" in filtered.columns else ["all"]
    colors = {"train": "blue", "test": "red", "dev": "green"}

    fig = go.Figure()

    for part in sorted(partitions):
        part_data = filtered[filtered["partition"] == part] if "partition" in filtered.columns else filtered
        fig.add_trace(
            go.Scatter(
                x=part_data["target"],
                y=part_data["prediction"],
                mode="markers",
                name=str(part),
                marker={"color": colors.get(str(part), "gray"), "size": 5, "opacity": 0.6},
                hovertemplate="Target: %{x:.3f}<br>Prediction: %{y:.3f}<extra></extra>",
            )
        )

    # Diagonal reference line
    all_vals = pd.concat([filtered["target"], filtered["prediction"]])
    val_min, val_max = float(all_vals.min()), float(all_vals.max())
    fig.add_trace(
        go.Scatter(
            x=[val_min, val_max],
            y=[val_min, val_max],
            mode="lines",
            name="Perfect",
            line={"dash": "dash", "color": "black", "width": 1},
        )
    )

    fig.update_layout(
        title=f"Prediction vs Ground Truth — outersplit {outersplit_id}, task {task_id}",
        xaxis_title="Ground Truth",
        yaxis_title="Prediction",
        width=650,
        height=500,
    )
    return fig


def plot_optuna_trial_counts_chart(df_optuna: pd.DataFrame) -> go.Figure:
    """Create a bar chart of unique trial counts per model type, grouped by task.

    Args:
        df_optuna: Optuna results DataFrame.

    Returns:
        Plotly Figure.
    """
    if df_optuna.empty:
        fig = go.Figure()
        fig.update_layout(title="No Optuna data available")
        return fig

    grouped = (
        df_optuna.groupby(["outersplit_id", "task_id", "model_type"])["trial"].nunique().reset_index(name="trial_count")
    )

    # Create subplots: one row per task_id, one col per outersplit_id
    task_ids = sorted(grouped["task_id"].unique())
    outer_ids = sorted(grouped["outersplit_id"].unique())

    fig = make_subplots(
        rows=len(task_ids),
        cols=len(outer_ids),
        subplot_titles=[f"task {t}, outersplit {o}" for t in task_ids for o in outer_ids],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    model_types = sorted(grouped["model_type"].unique())
    colors = [
        "royalblue",
        "coral",
        "mediumseagreen",
        "mediumpurple",
        "goldenrod",
        "tomato",
        "teal",
        "orchid",
    ]

    for row_i, task in enumerate(task_ids, 1):
        for col_i, outer in enumerate(outer_ids, 1):
            sub = grouped[(grouped["task_id"] == task) & (grouped["outersplit_id"] == outer)]
            for m_i, model in enumerate(model_types):
                model_data = sub[sub["model_type"] == model]
                fig.add_trace(
                    go.Bar(
                        x=[model],
                        y=model_data["trial_count"].values if len(model_data) > 0 else [0],
                        name=model,
                        marker={"color": colors[m_i % len(colors)]},
                        showlegend=(row_i == 1 and col_i == 1),
                    ),
                    row=row_i,
                    col=col_i,
                )

    fig.update_layout(
        title="Number of Unique Trials by Model Type",
        height=300 * len(task_ids),
        width=300 * len(outer_ids),
        barmode="group",
    )
    return fig


def plot_optuna_trials_chart(
    df_optuna: pd.DataFrame,
    *,
    outersplit_id: int | str = 0,
    task_id: int | str = 0,
    direction: MetricDirection = MetricDirection.MINIMIZE,
) -> go.Figure:
    """Create scatter + best-value line plot for Optuna trials.

    Args:
        df_optuna: Optuna results DataFrame.
        outersplit_id: Outer split to filter on.
        task_id: Task to filter on.
        direction: Optimization direction ('minimize' or 'maximize').

    Returns:
        Plotly Figure.
    """
    mask = (df_optuna["outersplit_id"] == int(outersplit_id)) & (df_optuna["task_id"] == int(task_id))
    filtered = df_optuna[mask]
    if filtered.empty:
        fig = go.Figure()
        fig.update_layout(title="No Optuna data for selected filters")
        return fig

    # Get trial-level values (one value per trial)
    trial_values = filtered.groupby("trial")["value"].first().reset_index().sort_values("trial")

    # Cumulative best
    if direction == MetricDirection.MAXIMIZE:
        trial_values["best"] = trial_values["value"].cummax()
    else:
        trial_values["best"] = trial_values["value"].cummin()

    # Merge model_type for coloring
    trial_model = filtered.groupby("trial")["model_type"].first().reset_index()
    trial_values = trial_values.merge(trial_model, on="trial", how="left")

    fig = go.Figure()

    # Scatter per model type
    model_types = sorted(trial_values["model_type"].unique())
    colors = [
        "royalblue",
        "coral",
        "mediumseagreen",
        "mediumpurple",
        "goldenrod",
        "tomato",
        "teal",
        "orchid",
    ]
    for i, model in enumerate(model_types):
        sub = trial_values[trial_values["model_type"] == model]
        fig.add_trace(
            go.Scatter(
                x=sub["trial"],
                y=sub["value"],
                mode="markers",
                name=model,
                marker={"size": 7, "color": colors[i % len(colors)]},
                hovertemplate="Trial: %{x}<br>Value: %{y:.4f}<extra></extra>",
            )
        )

    # Best value line
    fig.add_trace(
        go.Scatter(
            x=trial_values["trial"],
            y=trial_values["best"],
            mode="lines",
            name="Best",
            line={"color": "green", "width": 2},
        )
    )

    fig.update_layout(
        title=f"Optuna Trials — outersplit {outersplit_id}, task {task_id}",
        xaxis_title="Trial",
        yaxis_title="Objective Value",
        yaxis_type="log",
        width=700,
        height=450,
    )
    return fig


def plot_optuna_hyperparameters_chart(
    df_optuna: pd.DataFrame,
    *,
    outersplit_id: int | str = 0,
    task_id: int | str = 0,
    model_type: str = "",
) -> go.Figure:
    """Create scatter plots of hyperparameter values vs objective.

    Args:
        df_optuna: Optuna results DataFrame.
        outersplit_id: Outer split to filter on.
        task_id: Task to filter on.
        model_type: Model type to filter on.

    Returns:
        Plotly Figure with subplots per hyperparameter.
    """
    mask = (df_optuna["outersplit_id"] == int(outersplit_id)) & (df_optuna["task_id"] == int(task_id))
    if model_type:
        mask &= df_optuna["model_type"] == model_type

    filtered = df_optuna[mask]
    if filtered.empty:
        fig = go.Figure()
        fig.update_layout(title="No Optuna data for selected filters")
        return fig

    params = sorted(filtered["hyper_param"].unique())
    n_params = len(params)
    cols = 2
    rows = (n_params + cols - 1) // cols

    fig = make_subplots(
        rows=max(rows, 1),
        cols=cols,
        subplot_titles=params,
        vertical_spacing=0.08,
        horizontal_spacing=0.12,
    )

    for i, param in enumerate(params):
        row = i // cols + 1
        col = i % cols + 1
        param_data = filtered[filtered["hyper_param"] == param].copy()

        # Try to convert param_value to numeric
        param_data["param_value_num"] = pd.to_numeric(param_data["param_value"], errors="coerce")

        fig.add_trace(
            go.Scatter(
                x=param_data["param_value_num"],
                y=param_data["value"],
                mode="markers",
                marker={
                    "size": 5,
                    "color": param_data["trial"],
                    "colorscale": "Blues",
                    "showscale": (i == 0),
                    "colorbar": {"title": "Trial"},
                },
                hovertemplate=f"{param}: %{{x}}<br>Value: %{{y:.4f}}<br>Trial: %{{marker.color}}<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="Parameter Value", row=row, col=col)
        fig.update_yaxes(title_text="Target Metric", row=row, col=col)

    fig.update_layout(
        title=f"Optuna Hyperparameters — outersplit {outersplit_id}, task {task_id}"
        + (f", {model_type}" if model_type else ""),
        height=300 * max(rows, 1),
        width=700,
    )
    return fig
