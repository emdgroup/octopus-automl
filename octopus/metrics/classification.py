"""Classification metrics."""

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from octopus.types import MLType, PredType

from .config import Metric
from .core import Metrics


@Metrics.register("AUCROC")
def aucroc_metric() -> Metric:
    """AUCROC metric configuration."""
    return Metric(
        name="AUCROC",
        metric_function=roc_auc_score,
        ml_types=[MLType.BINARY],
        higher_is_better=True,
        prediction_type=PredType.PREDICT_PROBA,
        scorer_string="roc_auc",
    )


@Metrics.register("ACC")
def acc_metric() -> Metric:
    """Accuracy metric configuration."""
    return Metric(
        name="ACC",
        metric_function=accuracy_score,
        ml_types=[MLType.BINARY, MLType.MULTICLASS],
        higher_is_better=True,
        prediction_type=PredType.PREDICT,
        scorer_string="accuracy",
    )


@Metrics.register("ACCBAL")
def accbal_metric() -> Metric:
    """Balanced accuracy metric configuration."""
    return Metric(
        name="ACCBAL",
        metric_function=balanced_accuracy_score,
        ml_types=[MLType.BINARY, MLType.MULTICLASS],
        higher_is_better=True,
        prediction_type=PredType.PREDICT,
        scorer_string="balanced_accuracy",
    )


@Metrics.register("LOGLOSS")
def logloss_metric() -> Metric:
    """Log loss metric configuration."""
    return Metric(
        name="LOGLOSS",
        metric_function=log_loss,
        ml_types=[MLType.BINARY, MLType.MULTICLASS],
        higher_is_better=True,
        prediction_type=PredType.PREDICT_PROBA,
        scorer_string="neg_log_loss",
    )


@Metrics.register("F1")
def f1_metric() -> Metric:
    """F1 metric configuration."""
    return Metric(
        name="F1",
        metric_function=f1_score,
        ml_types=[MLType.BINARY],
        higher_is_better=True,
        prediction_type=PredType.PREDICT,
        scorer_string="f1",
    )


@Metrics.register("NEGBRIERSCORE")
def negbrierscore_metric() -> Metric:
    """Brier score metric configuration."""
    return Metric(
        name="NEGBRIERSCORE",
        metric_function=brier_score_loss,
        ml_types=[MLType.BINARY],
        higher_is_better=True,
        prediction_type=PredType.PREDICT_PROBA,
        scorer_string="neg_brier_score",
    )


@Metrics.register("AUCPR")
def aucpr_metric() -> Metric:
    """AUCPR metric configuration."""
    return Metric(
        name="AUCPR",
        metric_function=average_precision_score,
        ml_types=[MLType.BINARY],
        higher_is_better=True,
        prediction_type=PredType.PREDICT_PROBA,
        scorer_string="average_precision",
    )


@Metrics.register("MCC")
def mcc_metric() -> Metric:
    """Matthews Correlation Coefficient metric configuration."""
    return Metric(
        name="MCC",
        metric_function=matthews_corrcoef,
        ml_types=[MLType.BINARY, MLType.MULTICLASS],
        higher_is_better=True,
        prediction_type=PredType.PREDICT,
        scorer_string="matthews_corrcoef",
    )


@Metrics.register("PRECISION")
def precision_metric() -> Metric:
    """Precision metric configuration."""
    return Metric(
        name="PRECISION",
        metric_function=precision_score,
        ml_types=[MLType.BINARY],
        higher_is_better=True,
        prediction_type=PredType.PREDICT,
        scorer_string="precision",
    )


@Metrics.register("RECALL")
def recall_metric() -> Metric:
    """Recall metric configuration."""
    return Metric(
        name="RECALL",
        metric_function=recall_score,
        ml_types=[MLType.BINARY],
        higher_is_better=True,
        prediction_type=PredType.PREDICT,
        scorer_string="recall",
    )
