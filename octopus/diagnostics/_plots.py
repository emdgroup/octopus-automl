"""Plotly chart functions for Optuna diagnostics."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    direction: str = "minimize",
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
    if df_optuna.empty:
        fig = go.Figure()
        fig.update_layout(title="No Optuna data for selected filters")
        return fig

    mask = (df_optuna["outersplit_id"] == int(outersplit_id)) & (df_optuna["task_id"] == int(task_id))
    filtered = df_optuna[mask]
    if filtered.empty:
        fig = go.Figure()
        fig.update_layout(title="No Optuna data for selected filters")
        return fig

    # Get trial-level values (one value per trial)
    trial_values = filtered.groupby("trial")["value"].first().reset_index().sort_values("trial")

    # Cumulative best
    if direction == "maximize":
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
    if df_optuna.empty:
        fig = go.Figure()
        fig.update_layout(title="No Optuna data for selected filters")
        return fig

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
