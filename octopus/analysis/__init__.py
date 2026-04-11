"""Post-hoc analysis utilities: tables (DataFrames) and plots (Plotly Figures)."""

from octopus.analysis.plots import aucroc_plot, feature_count_plot, feature_frequency_plot, performance_plot
from octopus.analysis.tables import (
    get_details,
    performance,
    selected_features,
    workflow_graph,
)
from octopus.analysis.test_evaluator import OctoTestEvaluator
from octopus.predict.study_io import StudyInfo, load_study_info

__all__ = [
    "OctoTestEvaluator",
    "StudyInfo",
    "aucroc_plot",
    "feature_count_plot",
    "feature_frequency_plot",
    "get_details",
    "load_study_info",
    "performance",
    "performance_plot",
    "selected_features",
    "workflow_graph",
]
