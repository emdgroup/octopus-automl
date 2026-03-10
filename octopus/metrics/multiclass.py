"""Multiclass metrics."""

from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from octopus.types import MLType, PredType

from .config import Metric
from .core import Metrics


@Metrics.register("ACCBAL_MC")
def accbal_multiclass_metric() -> Metric:
    """Balanced accuracy metric configuration for multiclass problems."""
    return Metric(
        name="ACCBAL_MC",
        metric_function=balanced_accuracy_score,
        ml_types=[MLType.MULTICLASS],
        higher_is_better=True,
        prediction_type=PredType.PREDICT,
        scorer_string="balanced_accuracy",
    )


@Metrics.register("AUCROC_MACRO")
def aucroc_macro_multiclass_metric() -> Metric:
    """AUCROC metric configuration for multiclass problems (macro-average)."""
    return Metric(
        name="AUCROC_MACRO",
        metric_function=roc_auc_score,
        metric_params={"multi_class": "ovr", "average": "macro"},
        ml_types=[MLType.MULTICLASS],
        higher_is_better=True,
        prediction_type=PredType.PREDICT_PROBA,
        scorer_string="roc_auc_ovr",
    )


@Metrics.register("AUCROC_WEIGHTED")
def aucroc_weighted_multiclass_metric() -> Metric:
    """AUCROC metric configuration for multiclass problems (weighted-average)."""
    return Metric(
        name="AUCROC_WEIGHTED",
        metric_function=roc_auc_score,
        metric_params={"multi_class": "ovr", "average": "weighted"},
        ml_types=[MLType.MULTICLASS],
        higher_is_better=True,
        prediction_type=PredType.PREDICT_PROBA,
        scorer_string="roc_auc_ovr_weighted",
    )
