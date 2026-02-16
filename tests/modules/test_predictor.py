"""Tests for Predictor class and ModuleExecution.save()."""

import json
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from attrs import define
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from upath import UPath

from octopus.modules.base import FeatureSelectionExecution, MLModuleExecution, Task
from octopus.modules.predictor import Predictor


@pytest.fixture
def classification_data():
    """Create synthetic classification data."""
    np.random.seed(42)
    n_samples = 50
    feature_names = ["f1", "f2", "f3"]
    X = np.random.randn(n_samples, len(feature_names))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df, feature_names


@pytest.fixture
def regression_data():
    """Create synthetic regression data."""
    np.random.seed(42)
    n_samples = 50
    feature_names = ["f1", "f2", "f3"]
    X = np.random.randn(n_samples, len(feature_names))
    y = X[:, 0] * 2 + X[:, 1] + np.random.randn(n_samples) * 0.1
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df, feature_names


@pytest.fixture
def classifier_predictor(classification_data):
    """Create a Predictor with a trained classifier."""
    df, feature_names = classification_data
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(df[feature_names], df["target"])
    return Predictor(model_=model, selected_features_=feature_names)


@pytest.fixture
def regressor_predictor(regression_data):
    """Create a Predictor with a trained regressor."""
    df, feature_names = regression_data
    model = RandomForestRegressor(n_estimators=5, random_state=42)
    model.fit(df[feature_names], df["target"])
    return Predictor(model_=model, selected_features_=feature_names)


class TestPredict:
    """Tests for Predictor.predict()."""

    def test_predict_classification(self, classifier_predictor, classification_data):
        """Test classification predictions return correct shape and values."""
        df, _feature_names = classification_data
        predictions = classifier_predictor.predict(df)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(df)
        assert set(predictions).issubset({0, 1})

    def test_predict_regression(self, regressor_predictor, regression_data):
        """Test regression predictions return correct shape."""
        df, _feature_names = regression_data
        predictions = regressor_predictor.predict(df)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(df)

    def test_predict_missing_features(self, classifier_predictor):
        """Test that missing features raises ValueError."""
        df = pd.DataFrame({"f1": [1.0], "f2": [2.0]})  # missing f3
        with pytest.raises(ValueError, match="missing required features"):
            classifier_predictor.predict(df)

    def test_predict_extra_columns_ok(self, classifier_predictor, classification_data):
        """Test that extra columns in data are ignored."""
        df, _ = classification_data
        df["extra_col"] = 999
        predictions = classifier_predictor.predict(df)
        assert len(predictions) == len(df)


class TestPredictProba:
    """Tests for Predictor.predict_proba()."""

    def test_predict_proba_classification(self, classifier_predictor, classification_data):
        """Test probability predictions return correct shape and sum to 1."""
        df, _ = classification_data
        probas = classifier_predictor.predict_proba(df)
        assert isinstance(probas, np.ndarray)
        assert probas.shape == (len(df), 2)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0)

    def test_predict_proba_regression_raises(self, regressor_predictor, regression_data):
        """Test that predict_proba raises for regression models."""
        df, _ = regression_data
        with pytest.raises(NotImplementedError, match=r"predict_proba.*not available"):
            regressor_predictor.predict_proba(df)

    def test_predict_proba_missing_features(self, classifier_predictor):
        """Test that missing features raises ValueError for predict_proba."""
        df = pd.DataFrame({"f1": [1.0]})  # missing f2, f3
        with pytest.raises(ValueError, match="missing required features"):
            classifier_predictor.predict_proba(df)


class TestSaveLoad:
    """Tests for Predictor.save() and load()."""

    def test_save_load_roundtrip(self, classifier_predictor, classification_data, tmp_path):
        """Test save/load roundtrip produces identical predictions."""
        df, _ = classification_data
        save_path = UPath(tmp_path / "model_dir")

        # Save
        classifier_predictor.save(save_path)

        # Verify files exist
        assert (save_path / "model.joblib").exists()
        assert (save_path / "predictor.json").exists()

        # Load
        loaded = Predictor.load(save_path)
        assert loaded.selected_features_ == classifier_predictor.selected_features_

        # Predictions match
        orig_preds = classifier_predictor.predict(df)
        loaded_preds = loaded.predict(df)
        np.testing.assert_array_equal(orig_preds, loaded_preds)

    def test_save_load_roundtrip_regression(self, regressor_predictor, regression_data, tmp_path):
        """Test save/load roundtrip for regression models."""
        df, _ = regression_data
        save_path = UPath(tmp_path / "model_dir")

        regressor_predictor.save(save_path)
        loaded = Predictor.load(save_path)

        orig_preds = regressor_predictor.predict(df)
        loaded_preds = loaded.predict(df)
        np.testing.assert_array_almost_equal(orig_preds, loaded_preds)

    def test_load_backwards_compat_module_state(self, classifier_predictor, classification_data, tmp_path):
        """Test loading from module_state.json when predictor.json is absent."""
        df, feature_names = classification_data
        save_path = UPath(tmp_path / "model_dir")

        # Save using Predictor.save() then remove predictor.json
        classifier_predictor.save(save_path)
        (save_path / "predictor.json").unlink()

        # Create module_state.json manually (as ModuleExecution.save() would)
        state = {
            "selected_features": feature_names,
            "feature_importances": None,
        }
        with open(save_path / "module_state.json", "w") as f:
            json.dump(state, f)

        # Load should fall back to module_state.json
        loaded = Predictor.load(save_path)
        assert loaded.selected_features_ == feature_names

        orig_preds = classifier_predictor.predict(df)
        loaded_preds = loaded.predict(df)
        np.testing.assert_array_equal(orig_preds, loaded_preds)

    def test_load_missing_model_raises(self, tmp_path):
        """Test that loading from empty directory raises FileNotFoundError."""
        save_path = UPath(tmp_path / "empty_dir")
        save_path.mkdir()
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            Predictor.load(save_path)

    def test_load_missing_all_state_files_raises(self, classifier_predictor, tmp_path):
        """Test that loading fails when no state file exists."""
        save_path = UPath(tmp_path / "model_dir")
        classifier_predictor.save(save_path)

        # Remove the only state file
        (save_path / "predictor.json").unlink()

        with pytest.raises(FileNotFoundError, match=r"Neither predictor\.json nor module_state\.json"):
            Predictor.load(save_path)

    def test_predictor_json_contents(self, classifier_predictor, tmp_path):
        """Verify the JSON structure of predictor.json."""
        save_path = UPath(tmp_path / "model_dir")
        classifier_predictor.save(save_path)

        with open(save_path / "predictor.json") as f:
            data = json.load(f)

        assert "selected_features" in data
        assert data["selected_features"] == classifier_predictor.selected_features_


# =============================================================================
# ModuleExecution.save() Tests
# =============================================================================


@define
class _StubTask(Task):
    """Minimal concrete Task for testing save()."""

    module: ClassVar[str] = "stub"

    def create_module(self):
        raise NotImplementedError


@define
class _StubMLModule(MLModuleExecution[_StubTask]):
    """Minimal concrete ML execution for testing save()."""

    def fit(self, *args, **kwargs):
        raise NotImplementedError


@define
class _StubFSModule(FeatureSelectionExecution[_StubTask]):
    """Minimal concrete feature selection execution for testing save()."""

    def fit(self, *args, **kwargs):
        raise NotImplementedError


class TestModuleExecutionSave:
    """Tests for ModuleExecution.save() writing predictor.json."""

    def test_ml_module_save_creates_predictor_json(self, tmp_path):
        """Test that save() creates predictor.json when model_ is present."""
        task = _StubTask(task_id=0)
        module = _StubMLModule(config=task)

        # Simulate fitted state
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        model.fit([[1, 2], [3, 4]], [0, 1])
        module.model_ = model
        module.selected_features_ = ["f1", "f2"]
        module.feature_importances_ = {"internal": "dummy"}

        save_path = UPath(tmp_path / "module")
        module.save(save_path)

        # All three files should exist
        assert (save_path / "model.joblib").exists()
        assert (save_path / "module_state.json").exists()
        assert (save_path / "predictor.json").exists()

        # predictor.json should have selected_features
        with open(save_path / "predictor.json") as f:
            data = json.load(f)
        assert data["selected_features"] == ["f1", "f2"]

        # Predictor.load() should be able to load it
        loaded = Predictor.load(save_path)
        assert loaded.selected_features_ == ["f1", "f2"]

    def test_fs_module_save_no_predictor_json(self, tmp_path):
        """Test that save() does NOT create predictor.json for feature selection modules."""
        task = _StubTask(task_id=0)
        module = _StubFSModule(config=task)

        # Simulate fitted state (no model_)
        module.selected_features_ = ["f1", "f2"]
        module.feature_importances_ = {}

        save_path = UPath(tmp_path / "module")
        module.save(save_path)

        # module_state.json should exist, but NOT model.joblib or predictor.json
        assert (save_path / "module_state.json").exists()
        assert not (save_path / "model.joblib").exists()
        assert not (save_path / "predictor.json").exists()

        # module_state.json should have selected_features
        with open(save_path / "module_state.json") as f:
            data = json.load(f)
        assert data["selected_features"] == ["f1", "f2"]

    def test_ml_module_save_module_state_contents(self, tmp_path):
        """Test that module_state.json contains both selected_features and feature_importances."""
        task = _StubTask(task_id=0)
        module = _StubMLModule(config=task)

        model = RandomForestRegressor(n_estimators=2, random_state=42)
        model.fit([[1, 2], [3, 4]], [1.0, 2.0])
        module.model_ = model
        module.selected_features_ = ["f1", "f2"]
        module.feature_importances_ = None

        save_path = UPath(tmp_path / "module")
        module.save(save_path)

        with open(save_path / "module_state.json") as f:
            data = json.load(f)
        assert data["selected_features"] == ["f1", "f2"]
        assert data["feature_importances"] is None
