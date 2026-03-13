"""CatBoost Cox survival wrapper model."""

import numpy as np
from attrs import define
from sklearn.base import BaseEstimator

from octopus.models.wrapper_models.survival_base import SurvivalMixin, check_y_survival


@define(slots=False)
class CatBoostCoxSurvival(SurvivalMixin, BaseEstimator):
    """CatBoost Cox proportional hazards model for survival analysis.

    Wraps CatBoostRegressor with ``loss_function="Cox"`` to produce risk scores.
    Accepts structured numpy array y with fields:
        - 'c1' (bool): event indicator (True = event observed)
        - 'c2' (float): duration (time to event or censoring)

    Converts these to CatBoost's signed-target format internally:
        +t = event at time t, -t = censored at time t

    Output: risk scores (log-partial hazard) where higher = higher risk.

    Attributes:
        learning_rate: Learning rate (shrinkage).
        depth: Depth of trees.
        l2_leaf_reg: L2 regularization coefficient.
        random_strength: Random strength for scoring splits.
        rsm: Random subspace method (fraction of features per split).
        iterations: Maximum number of boosting iterations.
        allow_writing_files: Whether CatBoost can write temp files.
        logging_level: CatBoost logging level.
        thread_count: Number of threads for CatBoost.
        task_type: CatBoost computation device.
        random_state: Random seed for reproducibility.
    """

    learning_rate: float = 0.03
    depth: int = 6
    l2_leaf_reg: float = 3.0
    random_strength: float = 2.0
    rsm: float = 1.0
    iterations: int = 500
    allow_writing_files: bool = False
    logging_level: str = "Silent"
    thread_count: int = 1
    task_type: str = "CPU"
    random_state: int | None = None

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances from the fitted CatBoost model."""
        return np.asarray(self.model_.feature_importances_)

    def fit(self, X: np.ndarray, y: np.ndarray, *args, **kwargs) -> "CatBoostCoxSurvival":
        """Fit CatBoost Cox model.

        Args:
            X: Feature matrix.
            y: Structured array with 'c1' (event bool) and 'c2' (duration float).
            *args: Additional positional arguments (unused, for API compatibility).
            **kwargs: Additional keyword arguments (unused, for API compatibility).

        Returns:
            self
        """
        from catboost import CatBoostRegressor  # noqa: PLC0415

        event, duration = check_y_survival(y)

        self.model_ = CatBoostRegressor(
            loss_function="Cox",
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            random_strength=self.random_strength,
            rsm=self.rsm,
            iterations=self.iterations,
            allow_writing_files=self.allow_writing_files,
            logging_level=self.logging_level,
            thread_count=self.thread_count,
            task_type=self.task_type,
            random_state=self.random_state,
        )

        # Convert to CatBoost signed-target format: +t = event, -t = censored
        y_signed = np.where(event, duration, -duration)
        self.model_.fit(X, y_signed)
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict risk scores. Higher = higher risk.

        Args:
            X: Feature matrix.
            **kwargs: Additional keyword arguments (unused, for API compatibility).

        Returns:
            Array of risk scores (log-partial hazard).
        """
        return np.asarray(self.model_.predict(X))
