"""Post-hoc analysis utilities: tables (DataFrames) and plots (Plotly Figures)."""

from octopus.predict.study_io import StudyInfo, load_study
from octopus.analysis.plots import aucroc_plot, feature_count_plot, feature_frequency_plot, performance_plot
from octopus.analysis.tables import (
    get_details,
    performance,
    selected_features,
    workflow_graph,
)

__all__ = [
    "StudyInfo",
    "aucroc_plot",
    "feature_count_plot",
    "feature_frequency_plot",
    "get_details",
    "load_study",
    "performance",
    "performance_plot",
    "selected_features",
    "workflow_graph",
]
