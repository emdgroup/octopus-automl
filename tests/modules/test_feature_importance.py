"""Tests for feature importance methods on Predictor."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from octopus.modules.predictor import Predictor


def _make_test_data(n_samples=100, n_features=5):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    data = pd.DataFrame(X, columns=feature_names)
    data["target"] = y
    return data, feature_names


def _make_tree_predictor(data, feature_names):
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(data[feature_names], data["target"])
    return Predictor(model_=model, selected_features_=feature_names)


def _make_linear_predictor(data, feature_names):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(data[feature_names], data["target"])
    return Predictor(model_=model, selected_features_=feature_names)


def test_internal_importance():
    data, feature_names = _make_test_data()
    predictor = _make_tree_predictor(data, feature_names)

    importance_df = predictor.get_feature_importances(method="internal")

    assert "feature" in importance_df.columns
    assert "importance" in importance_df.columns
    assert len(importance_df) == len(feature_names)
    assert importance_df["importance"].sum() > 0


def test_permutation_importance():
    data, feature_names = _make_test_data()
    predictor = _make_tree_predictor(data, feature_names)

    importance_df = predictor.get_feature_importances(method="permutation", data=data, target=data["target"])

    assert "feature" in importance_df.columns
    assert "importance" in importance_df.columns
    assert len(importance_df) == len(feature_names)


def test_coefficient_importance():
    data, feature_names = _make_test_data()
    predictor = _make_linear_predictor(data, feature_names)

    importance_df = predictor.get_feature_importances(method="coefficients")

    assert "feature" in importance_df.columns
    assert "importance" in importance_df.columns
    assert len(importance_df) == len(feature_names)
    assert importance_df["importance"].sum() > 0


def test_error_handling():
    data, feature_names = _make_test_data()
    predictor = _make_tree_predictor(data, feature_names)

    with pytest.raises(ValueError, match="not supported"):
        predictor.get_feature_importances(method="invalid_method")

    with pytest.raises(ValueError, match="requires data and target"):
        predictor.get_feature_importances(method="permutation")

    # Internal importance on a model without feature_importances_
    linear_predictor = _make_linear_predictor(data, feature_names)
    with pytest.raises(ValueError, match="does not have feature_importances_"):
        linear_predictor.get_feature_importances(method="internal")

    # Coefficient importance on a model without coef_
    with pytest.raises(ValueError, match="does not have coef_"):
        predictor.get_feature_importances(method="coefficients")
