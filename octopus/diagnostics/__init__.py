"""Diagnostics package — study diagnostics from saved parquet files.

Provides :class:`StudyDiagnostics` for exploring Optuna hyperparameter
tuning results and saved feature importances across all outer splits
and tasks.

No model loading is performed — all data comes from saved parquet artifacts.

Example::

    from octopus.diagnostics import StudyDiagnostics

    diag = StudyDiagnostics("./studies/my_study/")
    diag.plot_feature_importance(fi_method="internal", fi_dataset="dev")
    diag.plot_optuna_trial_counts()
    diag.plot_optuna_trials(outersplit_id=0, task_id=0)
"""

from octopus.diagnostics.core import StudyDiagnostics

__all__ = ["StudyDiagnostics"]
