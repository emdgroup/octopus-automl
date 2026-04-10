"""Level 2 — plot functions that return ``plotly.graph_objects.Figure``.

Each function accepts DataFrames or dicts produced by Level 1 (``tables.py``)
and returns a Figure.  None of these functions call ``.show()`` — display is
the responsibility of Level 3 (``notebook.py``) or the caller.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_COLORS = [
    "#4C78A8",  # steel blue (primary)
    "#F58518",  # orange
    "#E45756",  # coral red
    "#72B7B2",  # teal
    "#54A24B",  # green
    "#EECA3B",  # gold
    "#B279A2",  # mauve
    "#FF9DA6",  # pink
]

_DEFAULT_LAYOUT: dict[str, Any] = {
    "plot_bgcolor": "white",
    "paper_bgcolor": "white",
    "font": {"size": 13, "color": "#333333"},
    "xaxis": {"gridcolor": "#e0e0e0"},
    "yaxis": {"gridcolor": "#e0e0e0"},
    "margin": {"t": 60},
    "height": 400,
    "width": 600,
}


def performance_plot(
    perf_df: pd.DataFrame,
    value_col: str = "dev_ensemble",
) -> go.Figure:
    """Bar chart of performance per outersplit with a metric dropdown.

    One set of traces per metric.  Within each metric, one trace per
    (task, results_key) group.  A Plotly dropdown switches between
    metrics; the default is the study's target metric.

    Args:
        perf_df: DataFrame from ``get_performance()`` with ``task``,
            ``key``, ``metric`` columns and value columns like
            ``dev_ensemble``.  ``attrs["target_metric"]`` sets the
            default metric shown.
        value_col: Which value column to plot.  Default ``"dev_ensemble"``.

    Returns:
        Plotly Figure.
    """
    if perf_df.empty or value_col not in perf_df.columns:
        fig = go.Figure()
        fig.update_layout(title="No performance data to plot")
        return fig

    target_metric = perf_df.attrs.get("target_metric", "")
    metrics = sorted(perf_df["metric"].unique())
    groups = list(perf_df[["task", "key"]].drop_duplicates().itertuples(index=False))
    n_groups = len(groups)

    # Put target metric first in the list
    if target_metric in metrics:
        metrics = [target_metric, *[m for m in metrics if m != target_metric]]

    fig = go.Figure()

    for m_idx, metric in enumerate(metrics):
        metric_df = perf_df[perf_df["metric"] == metric]
        visible = m_idx == 0

        for g_idx, (raw_task, raw_key) in enumerate(groups):
            task_id = int(raw_task)
            key = str(raw_key)

            mask = (metric_df["task"] == task_id) & (metric_df["key"] == key)
            group = metric_df[mask]

            splits = group[group.index != "Mean"]
            mean_rows = group[group.index == "Mean"]
            mean_val = float(mean_rows[value_col].iloc[0]) if not mean_rows.empty else None

            x_labels = [f"Split {idx}" for idx in splits.index]
            y_values = splits[value_col].tolist()

            if mean_val is not None:
                x_labels.append("Mean")
                y_values.append(mean_val)

            color = _COLORS[g_idx % len(_COLORS)]
            n_splits = len(splits)
            opacities = [1.0] * n_splits + ([0.6] if mean_val is not None else [])

            fig.add_trace(
                go.Bar(
                    x=x_labels,
                    y=y_values,
                    name=f"Task {task_id} ({key})",
                    marker={"color": color, "opacity": opacities},
                    text=[f"{v:.3f}" for v in y_values],
                    textposition="outside",
                    textfont={"size": 12, "color": "#333333"},
                    cliponaxis=False,
                    visible=visible,
                    legendgroup=f"task_{task_id}_{key}",
                    showlegend=visible,
                )
            )

    # Dropdown to switch between metrics
    if len(metrics) > 1:
        buttons = []
        for m_idx, metric in enumerate(metrics):
            vis = []
            show = []
            for mi in range(len(metrics)):
                active = mi == m_idx
                vis.extend([active] * n_groups)
                show.extend([active] * n_groups)
            buttons.append(
                {
                    "label": metric,
                    "method": "update",
                    "args": [
                        {"visible": vis, "showlegend": show},
                        {"yaxis.title.text": metric},
                    ],
                }
            )
        fig.update_layout(
            updatemenus=[
                {
                    "active": 0,
                    "buttons": buttons,
                    "x": 1.02,
                    "xanchor": "left",
                    "y": 1.15,
                    "yanchor": "top",
                }
            ],
        )

    default_metric = metrics[0] if metrics else ""
    fig.update_layout(**_DEFAULT_LAYOUT)
    fig.update_layout(
        title={
            "text": f"Dev Performance<br><sup>Target metric: {target_metric}</sup>",
            "x": 0.0,
            "xanchor": "left",
        },
        xaxis_title="Outer Split",
        yaxis_title=default_metric,
        barmode="group",
        showlegend=True,
        legend={"yanchor": "top", "y": 1.0, "xanchor": "left", "x": 1.02},
    )
    return fig


def _col_label(col: str) -> str:
    """Convert column key like ``'2_best'`` to ``'Task 2 (best)'``."""
    if col == "frequency":
        return "frequency"
    parts = str(col).split("_", 1)
    if len(parts) == 2:
        return f"Task {parts[0]} ({parts[1]})"
    return f"Task {col}"


def feature_count_plot(feature_table: pd.DataFrame) -> go.Figure:
    """Plot of feature counts across tasks and outer splits.

    For multi-task workflows a line chart is shown: x-axis is the
    (task, result_type) combination, one line per outer split,
    visualising the feature count drop through the pipeline.
    For a single task a bar chart is shown with one bar per split.

    Args:
        feature_table: The ``feature_table`` DataFrame returned by
            ``get_selected_features()``.

    Returns:
        Plotly Figure.
    """
    df = feature_table.drop(index="Mean", errors="ignore")
    has_tasks = "task" in df.columns
    fig = go.Figure()

    if has_tasks:
        has_rt = "result_type" in df.columns
        if has_rt:
            group_keys = df[["task", "result_type"]].drop_duplicates().sort_values(["task", "result_type"])
            x_labels = [f"Task {r.task} ({r.result_type})" for r in group_keys.itertuples()]
        else:
            task_ids = sorted(df["task"].unique())
            x_labels = [f"Task {t}" for t in task_ids]

        for s_idx, split_id in enumerate(sorted(df.index.unique())):
            split_df = df.loc[[split_id]]
            if has_rt:
                values = []
                for r in group_keys.itertuples():
                    match = split_df[(split_df["task"] == r.task) & (split_df["result_type"] == r.result_type)]
                    values.append(match["n_features"].iloc[0] if not match.empty else None)
            else:
                values = [
                    split_df.loc[split_df["task"] == t, "n_features"].iloc[0] if t in split_df["task"].values else None
                    for t in task_ids
                ]
            color = _COLORS[s_idx % len(_COLORS)]
            y_values: list[float | None] = [float(v) if v is not None else None for v in values]
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=y_values,  # type: ignore[arg-type]
                    mode="lines+markers+text",
                    text=[str(v) if v is not None else "" for v in values],
                    textposition="top center",
                    textfont={"size": 12, "color": "#333333"},
                    marker={"size": 8, "color": color},
                    line={"color": color},
                    cliponaxis=False,
                    name=f"Split {split_id}",
                )
            )
        fig.update_layout(
            xaxis_title="Task",
            showlegend=True,
            legend={"yanchor": "top", "y": 1.0, "xanchor": "left", "x": 1.02},
        )
    else:
        x_labels = [f"Split {i}" for i in df.index]
        n_features = df["n_features"].tolist()
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=n_features,
                marker_color=_COLORS[0],
                text=[str(int(v)) for v in n_features],
                textposition="outside",
                textfont={"size": 14, "color": "#333333"},
                cliponaxis=False,
                name="n_features",
            )
        )
        fig.update_layout(xaxis_title="Data Split")

    fig.update_layout(**_DEFAULT_LAYOUT)
    fig.update_layout(yaxis_title="Number of Features")
    return fig


def feature_frequency_plot(
    freq_table: pd.DataFrame,
    top_n: int = 20,
) -> go.Figure:
    """Horizontal bar chart of the most frequently selected features.

    Shows the top *top_n* features sorted by selection frequency.  For
    multi-task DataFrames a dropdown switches between tasks.

    Args:
        freq_table: The ``frequency_table`` DataFrame returned by
            ``get_selected_features()``.
        top_n: Number of features to display (default 20).

    Returns:
        Plotly Figure.
    """
    task_cols = list(freq_table.columns)
    fig = go.Figure()

    for i, col in enumerate(task_cols):
        top = freq_table.nlargest(top_n, col)[[col]].sort_values(col)
        fig.add_trace(
            go.Bar(
                x=top[col].tolist(),
                y=top.index.tolist(),
                orientation="h",
                marker_color=_COLORS[i % len(_COLORS)],
                text=top[col].astype(str).tolist(),
                textposition="outside",
                textfont={"size": 12, "color": "#333333"},
                cliponaxis=False,
                visible=i == 0,
                name=_col_label(col),
            )
        )

    if len(task_cols) > 1:
        buttons = [
            {
                "label": _col_label(col),
                "method": "update",
                "args": [{"visible": [j == i for j in range(len(task_cols))]}],
            }
            for i, col in enumerate(task_cols)
        ]
        fig.update_layout(
            updatemenus=[
                {
                    "active": 0,
                    "buttons": buttons,
                    "x": 1.02,
                    "xanchor": "left",
                    "y": 1.15,
                    "yanchor": "top",
                }
            ],
        )

    height = max(400, top_n * 22)
    fig.update_layout(**_DEFAULT_LAYOUT)
    fig.update_layout(
        xaxis_title="Selection Frequency (across outer splits)",
        yaxis_title="Feature",
        height=height,
        width=700,
        margin={"l": 150, "t": 60},
    )
    return fig


def testset_performance_plot(perf_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of test-set performance: outersplits x metrics.

    Args:
        perf_df: DataFrame from ``testset_performance_table()``.  Index is
            ``outersplit`` (including ``Mean``), columns are metric names.

    Returns:
        Plotly Figure.
    """
    plot_df = perf_df[perf_df.index != "Mean"]

    fig = go.Figure()
    for col in plot_df.columns:
        fig.add_trace(
            go.Bar(
                x=[str(idx) for idx in plot_df.index],
                y=plot_df[col],
                name=col,
            )
        )

    fig.update_layout(
        title="Test-Set Performance per OuterSplit",
        xaxis_title="OuterSplit",
        yaxis_title="Score",
        barmode="group",
        showlegend=True,
    )
    return fig


def aucroc_merged_plot(roc_data: dict[str, Any]) -> go.Figure:
    """Merged ROC curve (all predictions pooled across outersplits).

    Args:
        roc_data: Dictionary from ``aucroc_data()``.

    Returns:
        Plotly Figure with merged ROC curve and chance line.
    """
    fpr = roc_data["merged_fpr"]
    tpr = roc_data["merged_tpr"]
    auc_score = roc_data["merged_auc"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line={"dash": "dash", "width": 2, "color": "gray"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Merged (AUC = {auc_score:.3f})",
            line={"width": 2, "color": "blue"},
        )
    )
    fig.update_layout(
        title="Merged ROC Curve (All Predictions Pooled)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        showlegend=True,
        legend={"x": 0.95, "y": 0.05, "xanchor": "right", "yanchor": "bottom"},
        xaxis={"range": [0, 1], "gridcolor": "rgba(128, 128, 128, 0.2)"},
        yaxis={"range": [0, 1], "gridcolor": "rgba(128, 128, 128, 0.2)"},
        width=640,
        height=640,
    )
    return fig


def aucroc_averaged_plot(roc_data: dict[str, Any]) -> go.Figure:
    """Averaged ROC curve with +/- 1 std deviation band.

    Args:
        roc_data: Dictionary from ``aucroc_data()``.

    Returns:
        Plotly Figure with averaged ROC, std band, and chance line.
    """
    mean_fpr = roc_data["mean_fpr"]
    mean_tpr = roc_data["mean_tpr"]
    tprs_upper = roc_data["tprs_upper"]
    tprs_lower = roc_data["tprs_lower"]
    mean_auc = roc_data["mean_auc"]
    std_auc = roc_data["std_auc"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line={"dash": "dash", "width": 2, "color": "gray"},
            opacity=0.8,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([mean_fpr, mean_fpr[::-1]]),
            y=np.concatenate([tprs_upper, tprs_lower[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 0, 255, 0.2)",
            line={"color": "rgba(255,255,255,0)"},
            name="+/- 1 std. dev.",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mean_fpr,
            y=mean_tpr,
            mode="lines",
            name=f"Mean ROC (AUC = {mean_auc:.3f} +/- {std_auc:.3f})",
            line={"width": 2, "color": "blue"},
            opacity=0.8,
        )
    )
    fig.update_layout(
        title="Averaged ROC Curve (All Outer Test Sets)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        showlegend=True,
        legend={"x": 0.95, "y": 0.05, "xanchor": "right", "yanchor": "bottom"},
        xaxis={"range": [0, 1], "gridcolor": "rgba(128, 128, 128, 0.2)"},
        yaxis={"range": [0, 1], "gridcolor": "rgba(128, 128, 128, 0.2)"},
        width=640,
        height=640,
    )
    return fig


def _aucroc_individual_plot(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    split_id: int,
) -> go.Figure:
    """ROC curve for a single outersplit.

    Args:
        fpr: False positive rates.
        tpr: True positive rates.
        auc_score: AUC score.
        split_id: Outersplit index.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line={"dash": "dash", "width": 2, "color": "gray"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Outersplit {split_id} (AUC = {auc_score:.3f})",
            line={"width": 2, "color": "blue"},
        )
    )
    fig.update_layout(
        title=f"ROC Curve - Outersplit {split_id}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        showlegend=True,
        legend={"x": 0.95, "y": 0.05, "xanchor": "right", "yanchor": "bottom"},
        xaxis={"range": [0, 1], "gridcolor": "rgba(128, 128, 128, 0.2)"},
        yaxis={"range": [0, 1], "gridcolor": "rgba(128, 128, 128, 0.2)"},
        width=640,
        height=640,
    )
    return fig


def confusion_matrix_plot(
    cm_abs: np.ndarray,
    cm_rel: np.ndarray,
    class_names: list[str],
    title: str,
) -> go.Figure:
    """Side-by-side absolute and relative confusion matrix heatmaps.

    Args:
        cm_abs: Absolute confusion matrix.
        cm_rel: Relative (row-normalized) confusion matrix.
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
            colorbar={"x": 0.42, "len": 0.75, "thickness": 15, "showticklabels": True},
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
            colorbar={"x": 1.05, "len": 0.75, "thickness": 15, "showticklabels": True},
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


def fi_plot(
    fi_table: pd.DataFrame,
    top_n: int | None = None,
) -> go.Figure:
    """Bar chart of ensemble feature importance with optional error bars.

    Args:
        fi_table: Ensemble-filtered DataFrame (from ``fi_ensemble_table()``
            or raw ``calculate_fi()`` output — ensemble rows are filtered
            automatically).
        top_n: Number of top features to display.  None shows all.

    Returns:
        Plotly Figure.
    """
    fi_df = fi_table.copy()

    # Filter to ensemble rows if not already filtered
    if "fi_source" in fi_df.columns:
        fi_df = fi_df[fi_df["fi_source"] == "ensemble"].copy()

    fi_df = fi_df.sort_values("importance_mean", ascending=False)

    # Determine fi_type for the title
    fi_type = str(fi_df["fi_type"].iloc[0]) if "fi_type" in fi_df.columns and not fi_df.empty else "unknown"

    if top_n is not None:
        fi_df = fi_df.head(top_n)
        title = f"Top {top_n} Feature Importances ({fi_type})"
    else:
        title = f"Feature Importances ({fi_type})"

    # Determine importance column
    if "importance_mean" in fi_df.columns:
        importance_col = "importance_mean"
    elif "importance" in fi_df.columns:
        importance_col = "importance"
    else:
        raise ValueError(
            f"FI DataFrame has no 'importance_mean' or 'importance' column. Columns: {list(fi_df.columns)}"
        )

    # Build error bars if CI columns are available
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
    return fig
