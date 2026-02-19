"""Predict module."""

import matplotlib.pyplot as plt
import shap
from matplotlib.backends.backend_pdf import PdfPages

from octopus.experiment import OctoExperiment
from octopus.modules.utils import get_fi_permutation
from octopus.predict.core import OctoPredict
from octopus.predict.notebook_utils import (
    display_table,
    plot_aucroc,
    show_confusionmatrix,
    show_overall_fi_plot,
    show_overall_fi_table,
    show_selected_features,
    show_study_details,
    show_target_metric_performance,
    testset_performance_overview,
)

__all__ = [
    "OctoExperiment",
    "OctoPredict",
    "PdfPages",
    "display_table",
    "get_fi_permutation",
    "plot_aucroc",
    "plt",
    "shap",
    "show_confusionmatrix",
    "show_overall_fi_plot",
    "show_overall_fi_table",
    "show_selected_features",
    "show_study_details",
    "show_target_metric_performance",
    "testset_performance_overview",
]
