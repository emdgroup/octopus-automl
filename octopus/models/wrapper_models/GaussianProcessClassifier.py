"""Wrapper for Gaussian Process Classifier."""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Kernel, Matern, RationalQuadratic
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class GPClassifierWrapper(ClassifierMixin, BaseEstimator):
    """Wrapper for Gaussian Process Classifier."""

    _estimator_type = "classifier"

    def __init__(
        self,
        kernel: str | Kernel = "RBF",
        optimizer: Literal["fmin_l_bfgs_b"] | Callable | None = "fmin_l_bfgs_b",
        n_restarts_optimizer: int = 0,
        max_iter_predict: int = 100,
        warm_start: bool = False,
        copy_X_train: bool = True,
        random_state: int | None = None,
        multi_class: Literal["one_vs_rest", "one_vs_one"] = "one_vs_rest",
    ) -> None:
        self.kernel = kernel
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.warm_start = warm_start
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        self.multi_class = multi_class

    @property
    def classes_(self) -> np.ndarray:
        """Get the class labels."""
        check_is_fitted(self, "model_")
        return self.model_.classes_  # type: ignore[return-value]

    def fit(self, X: Any, y: Any) -> "GPClassifierWrapper":
        """Fit the Gaussian Process model."""
        X, y = check_X_y(X, y)
        kernel = self._get_kernel(self.kernel)
        self.model_ = GaussianProcessClassifier(
            kernel=kernel,
            optimizer=self.optimizer,
            n_restarts_optimizer=self.n_restarts_optimizer,
            max_iter_predict=self.max_iter_predict,
            warm_start=self.warm_start,
            copy_X_train=self.copy_X_train,
            random_state=self.random_state,
            multi_class=self.multi_class,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict using the Gaussian Process model."""
        check_is_fitted(self, "model_")
        X = check_array(X)
        return self.model_.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities using the Gaussian Process model."""
        check_is_fitted(self, "model_")
        X = check_array(X)
        return self.model_.predict_proba(X)

    def _get_kernel(self, kernel_str: str | Kernel) -> Kernel:
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
