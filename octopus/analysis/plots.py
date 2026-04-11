"""Post-hoc analysis plots: Plotly figures, no display calls."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import auc, roc_curve

from octopus.types import MLType

if TYPE_CHECKING:
    from octopus.analysis.test_evaluator import OctoTestEvaluator

__all__ = ["aucroc_plot", "feature_count_plot", "feature_frequency_plot", "performance_plot"]

_COLORS = [
    "#4C78A8",
    "#F58518",
    "#E45756",
    "#72B7B2",
    "#54A24B",
    "#EECA3B",
    "#B279A2",
    "#FF9DA6",
]


def feature_count_plot(feature_df: pd.DataFrame) -> go.Figure:
    """Plot of feature counts across tasks and outer splits.

    For multi-task workflows a line chart is shown: x-axis is the workflow
    task, one line per outer split, visualising the feature count drop
    through the pipeline.  For a single task a bar chart is shown with one
    bar per outer split.

    Args:
        feature_df: The ``feature_table`` DataFrame returned by
            ``selected_features()``.

    Returns:
        A Plotly ``Figure``.
    """
    df = feature_df.drop(index="Mean", errors="ignore")
    has_tasks = "task" in df.columns
    fig = go.Figure()

    if has_tasks:
        task_ids = sorted(df["task"].unique())
        x_labels = [f"Task {t}" for t in task_ids]

        for s_idx, split_id in enumerate(sorted(df.index.unique())):
            split_df = df.loc[[split_id]]
            values: list[float | None] = [
                split_df.loc[split_df["task"] == t, "n_features"].iloc[0] if t in split_df["task"].values else None
                for t in task_ids
            ]
            color = _COLORS[s_idx % len(_COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=values,  # type: ignore[arg-type]
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
        counts: list[float] = df["n_features"].tolist()
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=counts,
                marker_color=_COLORS[0],
                text=[str(int(v)) for v in counts],
                textposition="outside",
                textfont={"size": 14, "color": "#333333"},
                cliponaxis=False,
                name="n_features",
            )
        )
        fig.update_layout(xaxis_title="Data Split")

    fig.update_layout(
        yaxis_title="Number of Features",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        width=600,
        font={"size": 13, "color": "#333333"},
        xaxis={"gridcolor": "#e0e0e0"},
        yaxis={"gridcolor": "#e0e0e0"},
        margin={"t": 60},
    )

    return fig


def feature_frequency_plot(
    freq_df: pd.DataFrame,
    top_n: int = 20,
) -> go.Figure:
    """Horizontal bar chart of the most frequently selected features.

    Shows the top *top_n* features sorted by selection frequency.  For
    multi-task DataFrames a dropdown switches between tasks.

    Args:
        freq_df: The ``frequency_table`` DataFrame returned by
            ``selected_features()``.
        top_n: Number of features to display (default 20).

    Returns:
        A Plotly ``Figure``.
    """
    task_cols = list(freq_df.columns)
    fig = go.Figure()

    for i, col in enumerate(task_cols):
        top = freq_df.nlargest(top_n, col)[[col]].sort_values(col)
        visible = i == 0
        fig.add_trace(
            go.Bar(
                x=top[col],
                y=top.index.tolist(),
                orientation="h",
                marker_color=_COLORS[i % len(_COLORS)],
                text=top[col].astype(str).tolist(),
                textposition="outside",
                textfont={"size": 12, "color": "#333333"},
                cliponaxis=False,
                visible=visible,
                name=str(col),
            )
        )

    if len(task_cols) > 1:
        buttons = []
        for i, col in enumerate(task_cols):
            vis = [j == i for j in range(len(task_cols))]
            buttons.append(
                {
                    "label": f"Task {col}",
                    "method": "update",
                    "args": [{"visible": vis}],
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
            ]
        )

    height = max(400, top_n * 22)
    fig.update_layout(
        xaxis_title="Selection Frequency (across outer splits)",
        yaxis_title="Feature",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=height,
        width=700,
        font={"size": 13, "color": "#333333"},
        xaxis={"gridcolor": "#e0e0e0"},
        yaxis={"gridcolor": "#e0e0e0"},
        margin={"l": 150, "t": 60},
    )

    return fig


def performance_plot(perf_df: pd.DataFrame, metric: str | None = None) -> go.Figure:
    """Bar chart of metric scores per outer split.

    If the DataFrame contains a ``task`` column (from
    ``performance(task=None)``), grouped bars are shown with one color per
    task.  If the DataFrame contains multiple metrics and *metric* is None,
    a dropdown menu is added so the user can switch between metrics.

    Args:
        perf_df: DataFrame returned by ``performance()``.  Outer splits as
            index (plus a ``"Mean"`` row), metric names as columns.
        metric: Metric column to display.  If None and multiple metrics
            exist, all are available via a dropdown (first shown by default).

    Returns:
        A Plotly ``Figure``.
    """
    df = perf_df.drop(index="Mean", errors="ignore")
    has_tasks = "task" in df.columns
    metric_cols = [c for c in df.columns if c != "task" and df[c].dtype in ("float64", "float32", "int64")]

    if metric is not None:
        metric_cols = [metric]

    fig = go.Figure()

    if has_tasks:
        task_ids = sorted(df["task"].unique())
        splits = sorted(df.index.unique())
        x_labels = [f"Split {s}" for s in splits]

        for m_idx, col in enumerate(metric_cols):
            for t_idx, tid in enumerate(task_ids):
                task_df = df[df["task"] == tid]
                values: list[float | None] = [task_df.loc[s, col] if s in task_df.index else None for s in splits]
                color = _COLORS[t_idx % len(_COLORS)]
                visible = m_idx == 0

                fig.add_trace(
                    go.Bar(
                        x=x_labels,
                        y=values,  # type: ignore[arg-type]
                        marker_color=color,
                        text=[f"{v:.3f}" if v is not None else "" for v in values],
                        textposition="outside",
                        textfont={"size": 12, "color": "#333333"},
                        cliponaxis=False,
                        visible=visible,
                        name=f"Task {tid}",
                        legendgroup=f"task_{tid}",
                        showlegend=visible,
                    )
                )

        n_tasks = len(task_ids)
        fig.update_layout(
            barmode="group",
            showlegend=True,
            legend={"yanchor": "top", "y": 1.0, "xanchor": "left", "x": 1.02},
        )

        if len(metric_cols) > 1:
            buttons = []
            for m_idx, col in enumerate(metric_cols):
                vis = []
                show = []
                for mi in range(len(metric_cols)):
                    active = mi == m_idx
                    vis.extend([active] * n_tasks)
                    show.extend([active] * n_tasks)
                buttons.append(
                    {
                        "label": col,
                        "method": "update",
                        "args": [
                            {"visible": vis, "showlegend": show},
                            {"yaxis.title.text": col},
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
                ]
            )
    else:
        x_labels = [f"Split {i}" for i in df.index]

        for i, col in enumerate(metric_cols):
            col_values = df[col].tolist()
            visible = i == 0

            fig.add_trace(
                go.Bar(
                    x=x_labels,
                    y=col_values,
                    marker_color=_COLORS[0],
                    text=[f"{v:.3f}" for v in col_values],
                    textposition="outside",
                    textfont={"size": 14, "color": "#333333"},
                    cliponaxis=False,
                    visible=visible,
                    name=col,
                )
            )

        if len(metric_cols) > 1:
            buttons = []
            for i, col in enumerate(metric_cols):
                vis = [j == i for j in range(len(metric_cols))]
                buttons.append(
                    {
                        "label": col,
                        "method": "update",
                        "args": [{"visible": vis}, {"yaxis.title.text": col}],
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
                ]
            )

    fig.update_layout(
        xaxis_title="Data Split",
        yaxis_title=metric_cols[0] if metric_cols else "",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        width=600,
        showlegend=has_tasks,
        font={"size": 13, "color": "#333333"},
        xaxis={"gridcolor": "#e0e0e0"},
        yaxis={"gridcolor": "#e0e0e0"},
        margin={"t": 60},
    )

    return fig


def aucroc_plot(
    predictor: OctoTestEvaluator,
    figsize: tuple[int, int] = (8, 8),
) -> tuple[go.Figure, go.Figure]:
    """Merged and averaged AUCROC curves for a binary classification task.

    Uses ``predictor.predict_proba(df=True)`` to obtain per-split
    probabilities and targets.

    Args:
        predictor: ``OctoTestEvaluator`` instance for a binary
            classification task.
        figsize: Figure size as ``(width, height)`` in inches.

    Returns:
        A tuple of two Plotly ``Figure`` objects:

        - **merged**: ROC curve computed from all predictions pooled.
        - **averaged**: Mean ROC curve with +/- 1 std dev band across
          outer splits.

    Raises:
        ValueError: If the predictor is not for a binary classification task.
    """
    if MLType(predictor._config["ml_type"]) != MLType.BINARY:
        raise ValueError("AUCROC plots are only available for binary classification tasks")

    width_px, height_px = int(figsize[0] * 80), int(figsize[1] * 80)
    proba_df = predictor.predict_proba(df=True)
    positive_class = predictor._config.get("positive_class")

    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs = []
    aucs = []

    for split_id in predictor._models:
        split_df = proba_df[proba_df["outersplit"] == split_id]
        fpr, tpr, _ = roc_curve(
            np.asarray(split_df["target"]),
            np.asarray(split_df[positive_class]),
            drop_intermediate=True,
        )
        aucs.append(float(auc(fpr, tpr)))
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

    # --- Merged ROC ---
    fpr_merged, tpr_merged, _ = roc_curve(
        np.asarray(proba_df["target"]),
        np.asarray(proba_df[positive_class]),
        drop_intermediate=True,
    )
    auc_merged = float(auc(fpr_merged, tpr_merged))

    fig_merged = _roc_figure(fpr_merged, tpr_merged, auc_merged, width_px, height_px, title="Merged ROC Curve")

    # --- Averaged ROC ---
    tprs_arr = np.array(interp_tprs)
    mean_tpr = np.mean(tprs_arr, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs_arr, axis=0)

    aucroc_mean = float(np.mean(aucs))
    aucroc_std = float(np.std(aucs))

    fig_avg = go.Figure()
    fig_avg.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line={"dash": "dash", "width": 2, "color": "gray"},
        )
    )
    fig_avg.add_trace(
        go.Scatter(
            x=np.concatenate([mean_fpr, mean_fpr[::-1]]),
            y=np.concatenate(
                [
                    np.minimum(mean_tpr + std_tpr, 1),
                    np.maximum(mean_tpr - std_tpr, 0)[::-1],
                ]
            ),
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
        )
    )
    fig_avg.update_layout(
        title="Averaged ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=width_px,
        height=height_px,
        showlegend=True,
        legend={"x": 0.95, "y": 0.05, "xanchor": "right", "yanchor": "bottom"},
        xaxis={"range": [0, 1], "gridcolor": "rgba(128, 128, 128, 0.2)"},
        yaxis={"range": [0, 1], "gridcolor": "rgba(128, 128, 128, 0.2)"},
    )

    return fig_merged, fig_avg


def _roc_figure(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    width: int,
    height: int,
    title: str = "ROC Curve",
) -> go.Figure:
    """Create a single ROC curve figure."""
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
            name=f"ROC (AUC = {auc_score:.3f})",
            line={"width": 2, "color": "blue"},
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
