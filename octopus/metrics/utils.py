"""Metrics utility functions."""

from typing import Any

import numpy as np
import pandas as pd

from octopus.metrics import Metrics


def _to_numpy(data: Any) -> np.ndarray:
    """Convert data to numpy array if it's a DataFrame."""
    return data.to_numpy() if isinstance(data, pd.DataFrame) else np.asarray(data)


def _get_probability_columns(pred_df: pd.DataFrame, target_col: str) -> list[int]:
    """Extract and validate probability columns for multiclass predictions.

    Args:
        pred_df: Prediction DataFrame
        target_col: Target column name (string) to exclude

    Returns:
        Sorted list of probability column indices

    Raises:
        ValueError: If probability columns are not consecutive integers starting from 0
    """
    prob_columns: list[int] = []
    for col in pred_df.columns:
        if isinstance(col, int) and col not in [target_col, "prediction"]:
            prob_columns.append(col)

    if not prob_columns:
        return []

    prob_columns_sorted = sorted(prob_columns)
    expected_sequence = list(range(len(prob_columns_sorted)))

    if prob_columns_sorted != expected_sequence:
        raise ValueError(
            f"Probability columns must be consecutive integers starting from 0. "
            f"Found: {prob_columns_sorted}, expected: {expected_sequence}"
        )

    return prob_columns_sorted


def get_performance_from_model(
    model: Any,
    data: pd.DataFrame,
    feature_cols: list,
    target_metric: str,
    target_assignments: dict,
    threshold: float = 0.5,
    positive_class: int | str | None = None,
) -> float:
    """Calculate model performance on dataset for given metric.

    Args:
        model: Trained model with predict/predict_proba methods
        data: DataFrame containing features and targets
        feature_cols: List of feature column names
        target_metric: Name of the metric to calculate
        target_assignments: Dictionary mapping target types to column names
        threshold: Classification threshold (default: 0.5)
        positive_class: Positive class for binary classification

    Returns:
        Performance value as float

    Raises:
        ValueError: If input data is invalid, positive_class is missing for classification,
                   positive_class not found in model classes, predict_proba used for regression,
                   or ml_type is unknown
    """
    input_data = data[feature_cols]

    # Validate input data
    if input_data.empty or not all(col in input_data.columns for col in feature_cols):
        raise ValueError("Input data is empty or does not contain the required feature columns.")

    # Get metric instance
    metric = Metrics.get_instance(target_metric)

    # Time-to-event
    if metric.ml_type == "timetoevent":
        estimate = model.predict(input_data)
        event_time = data[target_assignments["duration"]].astype(float)
        event_indicator = data[target_assignments["event"]].astype(bool)
        return metric.calculate_t2e(event_indicator, event_time, estimate)

    # Get target for non-time-to-event tasks
    target_col = list(target_assignments.values())[0]
    target = data[target_col]

    # Binary classification
    if metric.ml_type == "classification":
        if positive_class is None:
            raise ValueError("positive_class must be provided for classification tasks")

        try:
            positive_class_idx = list(model.classes_).index(positive_class)
        except ValueError as e:
            raise ValueError(f"positive_class {positive_class} not found in model classes {model.classes_}") from e

        probabilities = _to_numpy(model.predict_proba(input_data))[:, positive_class_idx]

        if metric.prediction_type == "predict_proba":
            return metric.calculate(target, probabilities)

        predictions = (probabilities >= threshold).astype(int)
        return metric.calculate(target, predictions)

    # Multiclass classification
    if metric.ml_type == "multiclass":
        if metric.prediction_type == "predict_proba":
            probabilities = _to_numpy(model.predict_proba(input_data))
            return metric.calculate(target, probabilities)

        predictions = _to_numpy(model.predict(input_data))
        return metric.calculate(target, predictions)

    # Regression
    if metric.ml_type == "regression":
        if metric.prediction_type == "predict_proba":
            raise ValueError("predict_proba not supported for regression")

        predictions = _to_numpy(model.predict(input_data))
        return metric.calculate(target, predictions)

    raise ValueError(f"Unknown ml_type: {metric.ml_type}")


def get_performance_from_predictions(
    predictions: dict,
    target_metric: str,
    target_assignments: dict,
    threshold: float = 0.5,
    positive_class: int | str | None = None,
) -> dict:
    """Calculate model performance from predictions dict (from bag.get_predictions()).

    Args:
        predictions: Dictionary with predictions from bag.get_predictions().
                    Expected structure: {training_id: {partition: df}, 'ensemble': {partition: df}}
                    where partition is 'train', 'dev', or 'test'
        target_metric: Name of the metric to calculate
        target_assignments: Dictionary mapping target types to column names
        threshold: Classification threshold (default: 0.5)
        positive_class: Positive class for binary classification (required for classification)

    Returns:
        Dictionary with performance values for each training and ensemble

    Raises:
        ValueError: If positive_class is missing for classification, probability columns
                   are invalid for multiclass, predict_proba used for regression,
                   or ml_type is unknown
    """
    # Get metric instance
    metric = Metrics.get_instance(target_metric)

    performance: dict[Any, dict[str, float]] = {}
    for training_id, partitions in predictions.items():
        performance[training_id] = {}

        for part, pred_df in partitions.items():
            # Time-to-event
            if metric.ml_type == "timetoevent":
                estimate = pred_df["prediction"]
                event_time = pred_df[target_assignments["duration"]].astype(float)
                event_indicator = pred_df[target_assignments["event"]].astype(bool)
                perf_value = metric.calculate_t2e(event_indicator, event_time, estimate)

            else:
                # Get target for non-time-to-event tasks
                target_col = list(target_assignments.values())[0]
                target = pred_df[target_col]

                # Binary classification
                if metric.ml_type == "classification":
                    if positive_class is None:
                        raise ValueError("positive_class must be provided for classification tasks")

                    probabilities = pred_df[positive_class]

                    if metric.prediction_type == "predict_proba":
                        perf_value = metric.calculate(target, probabilities)
                    else:
                        predictions_binary = (probabilities >= threshold).astype(int)
                        perf_value = metric.calculate(target, predictions_binary)

                # Multiclass classification
                elif metric.ml_type == "multiclass":
                    if metric.prediction_type == "predict_proba":
                        prob_columns = _get_probability_columns(pred_df, target_col)
                        probabilities = pred_df[prob_columns].values
                        perf_value = metric.calculate(target, probabilities)
                    else:
                        predictions_class = pred_df["prediction"].astype(int)
                        perf_value = metric.calculate(target, predictions_class)

                # Regression
                elif metric.ml_type == "regression":
                    if metric.prediction_type == "predict_proba":
                        raise ValueError("predict_proba not supported for regression")

                    predictions_reg = pred_df["prediction"]
                    perf_value = metric.calculate(target, predictions_reg)

                else:
                    raise ValueError(f"Unknown ml_type: {metric.ml_type}")

            performance[training_id][part] = perf_value

    return performance


def get_score_from_prediction(
    predictions: dict,
    target_metric: str,
    target_assignments: dict,
    threshold: float = 0.5,
    positive_class: int | str | None = None,
) -> dict:
    """Calculate scores from predictions with optimization direction applied.

    Converts performance values to scores by applying the metric's optimization direction.
    For 'maximize' metrics, score = performance. For 'minimize' metrics, score = -performance.

    Args:
        predictions: Dictionary from bag.get_predictions()
        target_metric: Name of the metric to calculate
        target_assignments: Dictionary mapping target types to column names
        threshold: Classification threshold (default: 0.5)
        positive_class: Positive class for binary classification

    Returns:
        Dictionary with score values (direction-adjusted performance)
    """
    # Get performance values
    performance = get_performance_from_predictions(
        predictions=predictions,
        target_metric=target_metric,
        target_assignments=target_assignments,
        threshold=threshold,
        positive_class=positive_class,
    )

    # Convert performance to score based on optimization direction
    scores: dict[Any, dict[str, float]] = {}
    metric = Metrics.get_instance(target_metric)

    for training_id, partitions in performance.items():
        scores[training_id] = {}
        for part, perf_value in partitions.items():
            if metric.direction == "maximize":
                scores[training_id][part] = perf_value
            else:
                scores[training_id][part] = -perf_value

    return scores


def get_score_from_model(
    model: Any,
    data: pd.DataFrame,
    feature_cols: list,
    target_metric: str,
    target_assignments: dict,
    positive_class: int | str | None = None,
) -> float:
    """Calculate score from model with optimization direction applied.

    Converts performance value to score by applying the metric's optimization direction.
    For 'maximize' metrics, score = performance. For 'minimize' metrics, score = -performance.

    Args:
        model: Trained model
        data: DataFrame containing features and targets
        feature_cols: List of feature column names
        target_metric: Name of the metric to calculate
        target_assignments: Dictionary mapping target types to column names
        positive_class: Positive class for binary classification

    Returns:
        Score value (direction-adjusted performance)
    """
    performance = get_performance_from_model(
        model, data, feature_cols, target_metric, target_assignments, positive_class=positive_class
    )

    # Apply optimization direction
    metric = Metrics.get_instance(target_metric)
    return performance if metric.direction == "maximize" else -performance
