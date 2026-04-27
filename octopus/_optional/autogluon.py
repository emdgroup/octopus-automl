"""Optional autogluon imports."""

from octopus.exceptions import OptionalImportError

try:
    from autogluon.core.metrics import (
        accuracy,
        average_precision,
        balanced_accuracy,
        f1,
        log_loss,
        make_scorer,
        mcc,
        mean_absolute_error,
        mean_squared_error,
        precision,
        r2,
        recall,
        roc_auc,
        roc_auc_ovr,
        roc_auc_ovr_weighted,
        root_mean_squared_error,
    )
    from autogluon.tabular import TabularPredictor
    from sklearn.metrics import brier_score_loss

    brier_score = make_scorer(
        "brier_score",
        brier_score_loss,
        greater_is_better=False,
        needs_proba=True,
    )

except ModuleNotFoundError as ex:
    raise OptionalImportError(
        "Autogluon is unavailable because the necessary optional "
        "dependencies are not installed. "
        'Consider installing Octopus with "autogluon" dependency, '
        'e.g. via `pip install -e ".[autogluon]"`.'
    ) from ex

__all__ = [
    "TabularPredictor",
    "accuracy",
    "average_precision",
    "balanced_accuracy",
    "brier_score",
    "f1",
    "log_loss",
    "mcc",
    "mean_absolute_error",
    "mean_squared_error",
    "precision",
    "r2",
    "recall",
    "roc_auc",
    "roc_auc_ovr",
    "roc_auc_ovr_weighted",
    "root_mean_squared_error",
]
