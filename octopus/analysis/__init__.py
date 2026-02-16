"""Prediction and analysis utilities for Octopus studies.

This module provides functions for loading modules, making predictions,
and analyzing results from completed studies.
"""

import matplotlib.pyplot as plt
import shap
from matplotlib.backends.backend_pdf import PdfPages

from octopus.analysis.module_loader import ensemble_predict, ensemble_predict_proba, load_task_modules
from octopus.analysis.notebook_utils import (
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
from octopus.modules.utils import get_fi_permutation

__all__ = [
    "PdfPages",
    "display_table",
    "ensemble_predict",
    "ensemble_predict_proba",
    "get_fi_permutation",
    "load_task_modules",
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
