"""Lightweight predictor for inference without config reconstruction."""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from attrs import define, field
from sklearn.inspection import permutation_importance
from upath import UPath


@define
class Predictor:
    """A lightweight wrapper around a fitted model and its selected features.

    This class provides prediction capabilities without requiring the full
    ModuleExecution class hierarchy or config reconstruction. It can be loaded
    from saved study directories using only model.joblib and module_state.json.

    Attributes:
        model_: Trained sklearn-compatible model
        selected_features_: Features the model was trained on
    """

    model_: Any = field()
    """Trained model (sklearn-compatible)."""

    selected_features_: list[str] = field()
    """Features the model was trained on."""

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict on new data using the fitted model.

        Args:
            data: DataFrame containing at least the selected features

        Returns:
            Predictions array

        Raises:
            ValueError: If data is missing required features
        """
        missing = set(self.selected_features_) - set(data.columns)
        if missing:
            raise ValueError(f"Data is missing required features: {missing}")
        result: np.ndarray = self.model_.predict(data[self.selected_features_])
        return result

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities on new data.

        Args:
            data: DataFrame containing at least the selected features

        Returns:
            Probability predictions array

        Raises:
            ValueError: If data is missing required features
            NotImplementedError: If the model doesn't support predict_proba
        """
        if not hasattr(self.model_, "predict_proba"):
            raise NotImplementedError(
                f"predict_proba() is not available for this model type: {type(self.model_).__name__}"
            )
        missing = set(self.selected_features_) - set(data.columns)
        if missing:
            raise ValueError(f"Data is missing required features: {missing}")
        result: np.ndarray = self.model_.predict_proba(data[self.selected_features_])
        return result

    def get_feature_importances(
        self,
        method: str = "internal",
        data: pd.DataFrame | None = None,
        target: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Get feature importance from fitted model.

        Args:
            method: "internal", "permutation", or "coefficients"
            data: Required for permutation importance
            target: Required for permutation importance

        Returns:
            DataFrame with columns ['feature', 'importance']
        """
        if method == "internal":
            return self._get_internal_importance()
        elif method == "permutation":
            if data is None or target is None:
                raise ValueError("Permutation importance requires data and target parameters")
            return self._get_permutation_importance(data, target)
        elif method == "coefficients":
            return self._get_coefficient_importance()
        else:
            raise ValueError(f"Feature importance method '{method}' not supported")

    def _get_internal_importance(self) -> pd.DataFrame:
        model = self.model_
        if hasattr(model, "best_estimator_"):
            model = model.best_estimator_

        if not hasattr(model, "feature_importances_"):
            raise ValueError(
                f"Model {model.__class__.__name__} does not have feature_importances_. "
                "Try method='permutation' or method='coefficients' instead."
            )

        importances = model.feature_importances_
        if len(self.selected_features_) != len(importances):
            raise ValueError(
                f"Feature count mismatch: {len(self.selected_features_)} features but {len(importances)} importances"
            )

        df = pd.DataFrame({"feature": self.selected_features_, "importance": importances})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def _get_permutation_importance(self, data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        X = data[self.selected_features_]
        result = permutation_importance(self.model_, X, target, n_repeats=10, random_state=42, n_jobs=1)

        df = pd.DataFrame({"feature": self.selected_features_, "importance": result.importances_mean})  # type: ignore[union-attr]
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def _get_coefficient_importance(self) -> pd.DataFrame:
        model = self.model_
        if hasattr(model, "best_estimator_"):
            model = model.best_estimator_

        if not hasattr(model, "coef_"):
            raise ValueError(
                f"Model {model.__class__.__name__} does not have coef_. "
                "Try method='internal' or method='permutation' instead."
            )

        coefficients = model.coef_
        if len(coefficients.shape) > 1:
            importances = np.abs(coefficients).mean(axis=0)
        else:
            importances = np.abs(coefficients)

        if len(self.selected_features_) != len(importances):
            raise ValueError(
                f"Feature count mismatch: {len(self.selected_features_)} features but {len(importances)} importances"
            )

        df = pd.DataFrame({"feature": self.selected_features_, "importance": importances})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, path: UPath) -> None:
        """Save predictor to disk.

        Writes model.joblib and predictor.json.

        Args:
            path: Directory to save files into
        """
        path.mkdir(parents=True, exist_ok=True)
        with (path / "model.joblib").open("wb") as f:
            joblib.dump(self.model_, f)
        state = {"selected_features": self.selected_features_}
        with (path / "predictor.json").open("w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: UPath) -> Predictor:
        """Load a predictor from disk.

        Reads model.joblib and predictor.json.

        Args:
            path: Directory containing saved model files

        Returns:
            Predictor instance

        Raises:
            FileNotFoundError: If model.joblib or predictor.json is not found
        """
        model_path = path / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        with model_path.open("rb") as f:
            model = joblib.load(f)

        predictor_path = path / "predictor.json"
        if not predictor_path.exists():
            raise FileNotFoundError(f"predictor.json not found in: {path}")
        with predictor_path.open() as f:
            state = json.load(f)

        selected_features = state.get("selected_features", [])
        return cls(model_=model, selected_features_=selected_features)
