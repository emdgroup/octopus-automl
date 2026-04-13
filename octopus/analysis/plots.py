"""Post-hoc analysis plots: Plotly figures, no display calls."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import auc, confusion_matrix, roc_curve

from octopus.types import MLType

if TYPE_CHECKING:
    from octopus.analysis.test_evaluator import OctoTestEvaluator

__all__ = [
    "aucroc_plot",
    "confusion_matrix_plot",
    "feature_count_plot",
    "feature_frequency_plot",
    "fi_plot",
    "performance_plot",
    "prediction_plot",
    "residual_plot",
]

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
    mode: str = "merged",
    figsize: tuple[int, int] = (8, 8),
) -> go.Figure:
    """AUCROC curve for a binary classification task.

    Uses ``predictor.predict_proba()`` to obtain per-split
    probabilities and targets.

    Args:
        predictor: ``OctoTestEvaluator`` instance for a binary
            classification task.
        mode: Plot type to return.  ``"merged"`` pools predictions from
            all outer splits into a single ROC curve.  ``"averaged"``
            computes per-split curves and plots the mean with a +/- 1
            std dev band.  ``"per_split"`` shows one subplot per outer
            split.
        figsize: Figure size as ``(width, height)`` in inches.

    Returns:
        A Plotly ``Figure``.

    Raises:
        ValueError: If the predictor is not for a binary classification
            task, or if *mode* is invalid.
    """
    if MLType(predictor._config["ml_type"]) != MLType.BINARY:
        raise ValueError("AUCROC plots are only available for binary classification tasks")
    if mode not in ("merged", "averaged", "per_split"):
        raise ValueError(f"mode must be 'merged', 'averaged', or 'per_split', got '{mode}'")

    width_px, height_px = int(figsize[0] * 80), int(figsize[1] * 80)
    proba_df = predictor.predict_proba()
    positive_class = predictor._config.get("positive_class")

    if mode == "merged":
        fpr_merged, tpr_merged, _ = roc_curve(
            np.asarray(proba_df["target"]),
            np.asarray(proba_df[positive_class]),
            drop_intermediate=True,
        )
        auc_merged = float(auc(fpr_merged, tpr_merged))
        return _roc_figure(fpr_merged, tpr_merged, auc_merged, width_px, height_px, title="Merged ROC Curve")

    if mode == "per_split":
        split_ids = sorted(predictor._models)
        n_splits = len(split_ids)
        n_cols = min(3, n_splits)
        n_rows = -(-n_splits // n_cols)  # ceiling division
        subplot_size = 300
        # Compute AUC per split first so we can put it in subplot titles.
        split_aucs: list[str] = []
        split_curves: list[tuple[np.ndarray, np.ndarray]] = []
        for split_id in split_ids:
            split_df = proba_df[proba_df["outer_split"] == split_id]
            fpr_s, tpr_s, _ = roc_curve(
                np.asarray(split_df["target"]),
                np.asarray(split_df[positive_class]),
                drop_intermediate=True,
            )
            auc_s = float(auc(fpr_s, tpr_s))
            split_aucs.append(f"Split {split_id} (AUC = {auc_s:.3f})")
            split_curves.append((fpr_s, tpr_s))

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=split_aucs,
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
        )
        for idx, (fpr_s, tpr_s) in enumerate(split_curves):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    line={"dash": "dash", "width": 2, "color": "gray"},
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=fpr_s,
                    y=tpr_s,
                    mode="lines",
                    line={"width": 2, "color": "blue"},
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.update_xaxes(range=[0, 1], gridcolor="rgba(128,128,128,0.2)", row=row, col=col)
            fig.update_yaxes(range=[0, 1], gridcolor="rgba(128,128,128,0.2)", row=row, col=col)
        fig.update_layout(
            title="Per-Split ROC Curves",
            width=subplot_size * n_cols,
            height=subplot_size * n_rows,
            showlegend=False,
        )
        return fig

    # --- Averaged ROC ---
    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs = []
    aucs = []

    for split_id in predictor._models:
        split_df = proba_df[proba_df["outer_split"] == split_id]
        fpr, tpr, _ = roc_curve(
            np.asarray(split_df["target"]),
            np.asarray(split_df[positive_class]),
            drop_intermediate=True,
        )
        aucs.append(float(auc(fpr, tpr)))
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

    tprs_arr = np.array(interp_tprs)
    mean_tpr = np.mean(tprs_arr, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs_arr, axis=0)

    aucroc_mean = float(np.mean(aucs))
    aucroc_std = float(np.std(aucs))

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
    fig.add_trace(
        go.Scatter(
            x=mean_fpr,
            y=mean_tpr,
            mode="lines",
            name=f"Mean ROC (AUC = {aucroc_mean:.3f} +/- {aucroc_std:.3f})",
            line={"width": 2, "color": "blue"},
        )
    )
    fig.update_layout(
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

    return fig


def confusion_matrix_plot(
    predictor: OctoTestEvaluator,
    mode: str = "count",
) -> go.Figure:
    """Per-split confusion matrices for a classification task.

    Each outer split is shown as a subplot.  The subplot title includes
    the split ID.

    Args:
        predictor: ``OctoTestEvaluator`` instance for a classification task.
        mode: ``"count"`` for absolute counts, ``"relative"`` for
            row-normalised percentages (recall per true class).

    Returns:
        A Plotly ``Figure`` with one heatmap subplot per outer split.

    Raises:
        ValueError: If the predictor is not for a classification task,
            or if *mode* is invalid.
    """
    ml_type = MLType(predictor._config["ml_type"])
    if ml_type not in (MLType.BINARY, MLType.MULTICLASS):
        raise ValueError("Confusion matrix plots are only available for classification tasks")
    if mode not in ("count", "relative"):
        raise ValueError(f"mode must be 'count' or 'relative', got '{mode}'")

    proba_df = predictor.predict_proba()
    class_labels = list(next(iter(predictor._models.values())).classes_)
    class_names = [str(c) for c in class_labels]

    split_ids = sorted(predictor._models)
    n_splits = len(split_ids)
    n_cols = min(3, n_splits)
    n_rows = -(-n_splits // n_cols)  # ceiling division
    subplot_size = 350

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"Split {s}" for s in split_ids],
        horizontal_spacing=0.12,
        vertical_spacing=0.12,
    )

    for idx, split_id in enumerate(split_ids):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        split_df = proba_df[proba_df["outer_split"] == split_id]

        y_true = split_df["target"].to_numpy()
        probas = split_df[class_labels].to_numpy()
        y_pred = np.array(class_labels)[np.argmax(probas, axis=1)]

        if mode == "relative":
            cm = confusion_matrix(y_true, y_pred, labels=class_labels, normalize="true") * 100
            texttemplate = "%{z:.1f}%"
            zmin, zmax = 0, 100
        else:
            cm = confusion_matrix(y_true, y_pred, labels=class_labels)
            texttemplate = "%{z}"
            zmin, zmax = 0, int(cm.max()) if cm.size > 0 else 1

        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=class_names,  # type: ignore[arg-type]
                y=class_names,  # type: ignore[arg-type]
                texttemplate=texttemplate,
                textfont={"size": 14},
                colorscale="Blues",
                showscale=False,
                zmin=zmin,
                zmax=zmax,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="Predicted", row=row, col=col, side="bottom")
        fig.update_yaxes(title_text="True", row=row, col=col, autorange="reversed")

    title = "Confusion Matrices (Relative %)" if mode == "relative" else "Confusion Matrices"
    fig.update_layout(
        title=title,
        width=subplot_size * n_cols,
        height=subplot_size * n_rows,
        showlegend=False,
    )
    return fig


def fi_plot(
    fi_table: pd.DataFrame,
    top_n: int = 20,
) -> go.Figure:
    """Horizontal bar chart of ensemble feature importance.

    Filters the DataFrame to ensemble rows (``fi_source == "ensemble"``),
    sorts by importance descending, and shows the top *top_n* features.
    Error bars are drawn when confidence interval columns are available.

    Args:
        fi_table: DataFrame returned by ``calculate_fi()``, containing
            per-split and ensemble rows with a ``fi_source`` column.
        top_n: Number of top features to display (default 20).

    Returns:
        A Plotly ``Figure``.
    """
    fi_df = fi_table.copy()
    if "fi_source" in fi_df.columns:
        fi_df = fi_df[fi_df["fi_source"] == "ensemble"].copy()

    fi_df = fi_df.sort_values("importance_mean", ascending=True).tail(top_n)

    fi_type = str(fi_df["fi_type"].iloc[0]) if "fi_type" in fi_df.columns and not fi_df.empty else ""
    title = f"Top {top_n} Feature Importances ({fi_type})" if fi_type else f"Top {top_n} Feature Importances"

    error_x_config = None
    if "ci_lower" in fi_df.columns and "ci_upper" in fi_df.columns:
        ci_lower = fi_df["ci_lower"]
        ci_upper = fi_df["ci_upper"]
        if not ci_lower.isna().all():
            error_x_config = {
                "type": "data",
                "symmetric": False,
                "array": (ci_upper - fi_df["importance_mean"]).values,
                "arrayminus": (fi_df["importance_mean"] - ci_lower).values,
            }

    fig = go.Figure(
        data=go.Bar(
            x=fi_df["importance_mean"],
            y=fi_df["feature"],
            orientation="h",
            marker_color=_COLORS[0],
            error_x=error_x_config,
        )
    )

    height = max(400, len(fi_df) * 22)
    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="Feature",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=height,
        width=700,
        font={"size": 13, "color": "#333333"},
        xaxis={"gridcolor": "#e0e0e0"},
        yaxis={"gridcolor": "#e0e0e0"},
        margin={"l": 150, "t": 60},
        showlegend=False,
    )
    return fig


def prediction_plot(
    predictor: OctoTestEvaluator,
) -> go.Figure:
    """Per-split prediction vs ground truth scatter plots for regression.

    Each outer split is shown as a subplot.  A dashed diagonal line
    marks perfect predictions.  Points close to this line indicate
    accurate predictions.

    Args:
        predictor: ``OctoTestEvaluator`` instance for a regression task.

    Returns:
        A Plotly ``Figure`` with one subplot per outer split.

    Raises:
        ValueError: If the predictor is not for a regression task.
    """
    if MLType(predictor._config["ml_type"]) != MLType.REGRESSION:
        raise ValueError("Prediction plots are only available for regression tasks")

    pred_df = predictor.predict()
    split_ids = sorted(predictor._models)
    n_splits = len(split_ids)
    n_cols = min(3, n_splits)
    n_rows = -(-n_splits // n_cols)
    subplot_size = 350

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"Split {s}" for s in split_ids],
        horizontal_spacing=0.10,
        vertical_spacing=0.12,
    )

    val_min = min(pred_df["target"].min(), pred_df["prediction"].min())
    val_max = max(pred_df["target"].max(), pred_df["prediction"].max())

    for idx, split_id in enumerate(split_ids):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        split_df = pred_df[pred_df["outer_split"] == split_id]

        fig.add_trace(
            go.Scatter(
                x=[val_min, val_max],
                y=[val_min, val_max],
                mode="lines",
                line={"dash": "dash", "color": "gray", "width": 1},
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=split_df["target"],
                y=split_df["prediction"],
                mode="markers",
                marker={"color": _COLORS[0], "size": 5, "opacity": 0.6},
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="Ground Truth", row=row, col=col)
        fig.update_yaxes(title_text="Prediction", row=row, col=col)

    fig.update_layout(
        title="Prediction vs Ground Truth",
        width=subplot_size * n_cols,
        height=subplot_size * n_rows,
        showlegend=False,
    )
    return fig


def residual_plot(
    predictor: OctoTestEvaluator,
) -> go.Figure:
    """Per-split residual plots for regression.

    Each outer split is shown as a subplot.  The x-axis shows predicted
    values, the y-axis shows residuals (prediction - target).  A
    horizontal dashed line at zero marks perfect predictions.

    Args:
        predictor: ``OctoTestEvaluator`` instance for a regression task.

    Returns:
        A Plotly ``Figure`` with one subplot per outer split.

    Raises:
        ValueError: If the predictor is not for a regression task.
    """
    if MLType(predictor._config["ml_type"]) != MLType.REGRESSION:
        raise ValueError("Residual plots are only available for regression tasks")

    pred_df = predictor.predict()
    pred_df["residual"] = pred_df["prediction"] - pred_df["target"]

    split_ids = sorted(predictor._models)
    n_splits = len(split_ids)
    n_cols = min(3, n_splits)
    n_rows = -(-n_splits // n_cols)
    subplot_size = 350

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"Split {s}" for s in split_ids],
        horizontal_spacing=0.10,
        vertical_spacing=0.12,
    )

    for idx, split_id in enumerate(split_ids):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        split_df = pred_df[pred_df["outer_split"] == split_id]

        fig.add_trace(
            go.Scatter(
                x=[split_df["prediction"].min(), split_df["prediction"].max()],
                y=[0, 0],
                mode="lines",
                line={"dash": "dash", "color": "gray", "width": 1},
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=split_df["prediction"],
                y=split_df["residual"],
                mode="markers",
                marker={"color": _COLORS[1], "size": 5, "opacity": 0.6},
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="Prediction", row=row, col=col)
        fig.update_yaxes(title_text="Residual", row=row, col=col)

    fig.update_layout(
        title="Residuals",
        width=subplot_size * n_cols,
        height=subplot_size * n_rows,
        showlegend=False,
    )
    return fig


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
