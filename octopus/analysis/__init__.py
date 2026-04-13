"""Post-hoc analysis utilities: tables (DataFrames) and plots (Plotly Figures)."""

from octopus.analysis.plots import (
    aucroc_plot,
    confusion_matrix_plot,
    feature_count_plot,
    feature_frequency_plot,
    fi_plot,
    performance_plot,
    prediction_plot,
    residual_plot,
)
from octopus.analysis.tables import (
    get_details,
    performance,
    performance_table,
    selected_features,
    workflow_graph,
)
from octopus.analysis.test_evaluator import OctoTestEvaluator
from octopus.predict.study_io import StudyInfo, load_study_info

__all__ = [
    "OctoTestEvaluator",
    "StudyInfo",
    "aucroc_plot",
    "confusion_matrix_plot",
    "feature_count_plot",
    "feature_frequency_plot",
    "fi_plot",
    "get_details",
    "load_study_info",
    "performance",
    "performance_plot",
    "performance_table",
    "prediction_plot",
    "residual_plot",
    "selected_features",
    "workflow_graph",
]
