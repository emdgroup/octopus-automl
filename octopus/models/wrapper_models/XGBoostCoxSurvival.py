"""XGBoost Cox survival wrapper model."""

import numpy as np
from attrs import define
from sklearn.base import BaseEstimator

from octopus.models.wrapper_models.survival_base import SurvivalMixin, check_y_survival


@define(slots=False)
class XGBoostCoxSurvival(SurvivalMixin, BaseEstimator):
    """XGBoost Cox proportional hazards model for survival analysis.

    Wraps XGBRegressor with ``objective="survival:cox"`` to produce risk scores.
    Accepts structured numpy array y with fields:
        - 'c1' (bool): event indicator (True = event observed)
        - 'c2' (float): duration (time to event or censoring)

    Converts these to XGBoost's signed-target format internally:
        +t = event at time t, -t = censored at time t

    Output: risk scores (exp(margin)) where higher = higher risk.

    Attributes:
        learning_rate: Learning rate (shrinkage).
        min_child_weight: Minimum sum of instance weight in a child.
        subsample: Subsample ratio of training instances.
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        validate_parameters: Whether to validate parameters.
        n_jobs: Number of parallel threads.
        random_state: Random seed.
    """

    learning_rate: float = 0.1
    min_child_weight: int = 2
    subsample: float = 1.0
    n_estimators: int = 200
    max_depth: int = 6
    validate_parameters: bool = True
    n_jobs: int = 1
    random_state: int | None = None

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances from the fitted XGBoost model."""
        return np.asarray(self.model_.feature_importances_)

    def fit(self, X: np.ndarray, y: np.ndarray, *args, **kwargs) -> "XGBoostCoxSurvival":
        """Fit XGBoost Cox model.

        Args:
            X: Feature matrix.
            y: Structured array with 'c1' (event bool) and 'c2' (duration float).
            *args: Additional positional arguments (unused, for API compatibility).
            **kwargs: Additional keyword arguments (unused, for API compatibility).

        Returns:
            self
        """
        from xgboost import XGBRegressor  # noqa: PLC0415

        event, duration = check_y_survival(y)

        self.model_ = XGBRegressor(
            objective="survival:cox",
            eval_metric="cox-nloglik",
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            validate_parameters=self.validate_parameters,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbosity=0,
        )

        # Convert to XGBoost signed-target format: +t = event, -t = censored
        y_signed = np.where(event, duration, -duration)
        self.model_.fit(X, y_signed)
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict risk scores. Higher = higher risk.

        Args:
            X: Feature matrix.
            **kwargs: Additional keyword arguments (unused, for API compatibility).

        Returns:
            Array of risk scores (exp(margin)).
        """
        return np.asarray(self.model_.predict(X))
