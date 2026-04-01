"""Diagnostics package — interactive study-level diagnostics from saved parquet files.

Provides :class:`StudyDiagnostics` for exploring predictions, feature importances,
and Optuna hyperparameter tuning results across all outer splits and tasks.

No model loading is performed — all data comes from saved parquet artifacts.

Example::

    from octopus.diagnostics import StudyDiagnostics

    diag = StudyDiagnostics("./studies/my_study/")
    diag.plot_fi()
    diag.plot_optuna_trials()
"""

from octopus.diagnostics.core import StudyDiagnostics

__all__ = ["StudyDiagnostics"]
