"""Octopus post-study package — prediction and analysis from saved studies.

Top-level imports provide the core prediction interface. Analysis functions
(tables, plots, notebook wrappers) are available via submodule imports::

    from octopus.poststudy import OctoPredictor, OctoTestEvaluator
    from octopus.poststudy.tables import get_performance
    from octopus.poststudy.plots import performance_plot
    from octopus.poststudy.notebook import show_study_overview
"""

from octopus.poststudy.study_io import StudyInfo, load_study_information
from octopus.poststudy.task_evaluator_test import OctoTestEvaluator
from octopus.poststudy.task_predictor import OctoPredictor

__all__ = ["OctoPredictor", "OctoTestEvaluator", "StudyInfo", "load_study_information"]
