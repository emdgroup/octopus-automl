"""Diagnostics package — Optuna-only study-level diagnostics from saved parquet files.

Provides :class:`StudyDiagnostics` for exploring Optuna hyperparameter tuning
results across all outer splits and tasks. No model loading is performed.

Example::

    from octopus.diagnostics import StudyDiagnostics

    diag = StudyDiagnostics("./studies/my_study/")
    diag.plot_optuna_trial_counts()
    diag.plot_optuna_trials()
    diag.plot_optuna_hyperparameters()
"""

from octopus.diagnostics.core import StudyDiagnostics

__all__ = ["StudyDiagnostics"]
