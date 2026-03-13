"""Base classes for survival analysis models.

Provides sklearn-compatible mixin and validation utilities for time-to-event models.
The SurvivalMixin provides a .score() method that returns the concordance index,
enabling compatibility with sklearn utilities like permutation_importance.
"""

import numpy as np


def check_y_survival(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and extract event indicator and time from structured survival array.

    Args:
        y: Structured numpy array with two fields. First field is the binary
           event indicator (bool), second field is the time (float).
           In octopus, these fields are named 'c1' (event) and 'c2' (duration).

    Returns:
        Tuple of (event, time) where:
            - event: boolean array, True = event observed
            - time: float array, time to event or censoring

    Raises:
        ValueError: If y is not a structured array with exactly 2 fields,
                   if any time is negative, or if all samples are censored.
    """
    if not isinstance(y, np.ndarray) or y.dtype.names is None or len(y.dtype.names) != 2:
        raise ValueError(
            "y must be a structured array with two fields: a boolean event indicator and a float time field."
        )

    event_field, time_field = y.dtype.names
    event = np.asarray(y[event_field], dtype=bool)
    time = np.asarray(y[time_field], dtype=float)

    if np.any(time < 0):
        raise ValueError("All event/censoring times must be non-negative.")

    if not np.any(event):
        raise ValueError("All samples are censored. Need at least one event.")

    return event, time


class SurvivalMixin:
    """Mixin class for survival analysis estimators.

    Provides a ``.score(X, y)`` method that returns the concordance index,
    enabling compatibility with sklearn utilities like ``permutation_importance``.

    Models using this mixin must implement:
        - ``.predict(X)`` returning an array of risk scores (higher = higher risk)

    The structured target array ``y`` must have two fields:
        - First field (bool): event indicator (True = event observed)
        - Second field (float): time to event or censoring

    In octopus, these are 'c1' and 'c2' respectively.
    """

    _estimator_type = "survival"

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the concordance index for predictions on X.

        Uses lifelines' concordance_index to evaluate the concordance
        between predicted risk scores and observed survival times.

        Higher risk scores from the model indicate shorter expected survival.
        The sign is flipped internally to match lifelines' convention where
        higher predicted_scores indicate longer survival.

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            y: Structured array with event indicator and time fields.

        Returns:
            Concordance index in [0, 1]. 1.0 = perfect, 0.5 = random.
        """
        try:
            from lifelines.utils import concordance_index  # noqa: PLC0415
        except ModuleNotFoundError as ex:
            from octopus.exceptions import OptionalImportError  # noqa: PLC0415

            raise OptionalImportError(
                "lifelines is required for survival model scoring but is not installed. "
                'Install survival dependencies with: pip install "octopus-automl[survival]"'
            ) from ex

        try:
            event, time = check_y_survival(y)
        except ValueError as exc:
            # Only treat "all samples are censored" as a benign degenerate scenario.
            # For other validation errors (e.g., negative times, malformed arrays),
            # re-raise so that data/target bugs are not silently masked.
            if "All samples are censored" in str(exc):
                return 0.5
            raise

        risk_scores = self.predict(X)  # type: ignore[attr-defined]

        # lifelines expects: higher predicted_scores = longer survival
        # Our models output: higher risk_scores = shorter survival (higher risk)
        # So we negate to match lifelines' convention
        result: float = concordance_index(
            event_times=time,
            predicted_scores=-risk_scores,
            event_observed=event,
        )
        return result
