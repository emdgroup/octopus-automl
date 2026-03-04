"""Wrapper for Gaussian Process Regressor."""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel, Matern, RationalQuadratic
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class GPRegressorWrapper(RegressorMixin, BaseEstimator):
    """Wrapper for Gaussian Process Regressor."""

    _estimator_type = "regressor"

    def __init__(
        self,
        kernel: Literal["RBF", "Matern", "RationalQuadratic"] | Kernel = "RBF",
        alpha: float = 1e-10,
        optimizer: Literal["fmin_l_bfgs_b"] | Callable | None = "fmin_l_bfgs_b",
        n_restarts_optimizer: int = 0,
        normalize_y: bool = False,
        copy_X_train: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "GPRegressorWrapper":
        """Fit the Gaussian Process model."""
        X, y = check_X_y(X, y, y_numeric=True)
        kernel = self._get_kernel(self.kernel)
        self.model_ = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            optimizer=self.optimizer,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=self.normalize_y,
            copy_X_train=self.copy_X_train,
            random_state=self.random_state,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict using the Gaussian Process model."""
        check_is_fitted(self, "model_")
        X = check_array(X)
        return self.model_.predict(X)  # type: ignore[return-value]

    def _get_kernel(self, kernel_str: Literal["RBF", "Matern", "RationalQuadratic"] | Kernel) -> Kernel:
        """Get the kernel object based on the kernel string."""
        if isinstance(kernel_str, Kernel):
            return kernel_str
        elif kernel_str == "RBF":
            return RBF()
        elif kernel_str == "Matern":
            return Matern()
        elif kernel_str == "RationalQuadratic":
            return RationalQuadratic()
        else:
            raise ValueError(f"Unknown kernel: {kernel_str}")
