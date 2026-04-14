"""Octopus post-study package — prediction and analysis from saved studies.

Top-level imports provide the core prediction interface. Analysis functions
(tables, plots, notebook wrappers) are available via submodule imports::

    from octopus.poststudy import OctoPredictor, OctoTestEvaluator
    from octopus.poststudy.analysis.tables import get_performance
    from octopus.poststudy.analysis.plots import dev_performance_plot, performance_plot
    from octopus.poststudy.analysis.notebook import display_study_overview
"""

from octopus.poststudy.analysis.evaluator import OctoTestEvaluator
from octopus.poststudy.predict.predictor import OctoPredictor
from octopus.poststudy.study_io import StudyInfo, load_study_information

__all__ = ["OctoPredictor", "OctoTestEvaluator", "StudyInfo", "load_study_information"]
