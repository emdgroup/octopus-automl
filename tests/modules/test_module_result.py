"""Tests for ModuleResult.save() and .load() roundtrip."""

import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from upath import UPath

from octopus.modules import ModuleResult
from octopus.types import ResultType
from octopus.utils import parquet_load


class TestModuleResultSaveLoad:
    """Tests for ModuleResult save/load roundtrip."""

    def test_save_load_roundtrip_with_model(self, tmp_path):
        """Test full roundtrip with all artifacts including model."""
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        model.fit([[1, 2], [3, 4]], [0, 1])

        scores = pd.DataFrame(
            {
                "result_type": ["best"],
                "metric": ["AUCROC"],
                "partition": ["dev"],
                "aggregation": ["avg"],
                "fold": [None],
                "value": [0.85],
            }
        )
        predictions = pd.DataFrame(
            {
                "row_id": [1, 2],
                "prediction": [0, 1],
                "partition": ["test", "test"],
            }
        )
        fi = pd.DataFrame(
            {
                "feature": ["f1", "f2"],
                "importance": [0.7, 0.3],
                "fi_method": ["internal", "internal"],
            }
        )

        result = ModuleResult(
            result_type=ResultType.BEST,
            module="octo",
            selected_features=["f1", "f2"],
            scores=scores,
            predictions=predictions,
            feature_importances=fi,
            model=model,
        )

        result_dir = UPath(tmp_path / "best")
        result.save(result_dir)

        # Verify files exist
        assert (result_dir / "selected_features.json").exists()
        assert (result_dir / "scores.parquet").exists()
        assert (result_dir / "predictions.parquet").exists()
        assert (result_dir / "feature_importances.parquet").exists()
        assert (result_dir / "model" / "model.joblib").exists()
        assert (result_dir / "model" / "predictor.json").exists()

        # Load and verify
        loaded = ModuleResult.load(result_dir, result_type=ResultType.BEST, module="octo")

        assert loaded.result_type == ResultType.BEST
        assert loaded.module == "octo"
        assert loaded.selected_features == ["f1", "f2"]
        assert loaded.scores is not None
        assert not loaded.scores.empty
        assert loaded.predictions is not None
        assert not loaded.predictions.empty
        assert loaded.feature_importances is not None
        assert not loaded.feature_importances.empty
        assert loaded.model is not None

    def test_save_load_roundtrip_without_model(self, tmp_path):
        """Test roundtrip without a model (feature selection module)."""
        fi = pd.DataFrame(
            {
                "feature": ["f1", "f2"],
                "importance": [0.6, 0.4],
            }
        )

        result = ModuleResult(
            result_type=ResultType.BEST,
            module="roc",
            selected_features=["f1", "f2"],
            feature_importances=fi,
        )

        result_dir = UPath(tmp_path / "best")
        result.save(result_dir)

        # model directory should not exist
        assert not (result_dir / "model").exists()

        loaded = ModuleResult.load(result_dir, result_type=ResultType.BEST, module="roc")

        assert loaded.selected_features == ["f1", "f2"]
        assert loaded.model is None
        assert loaded.feature_importances is not None and not loaded.feature_importances.empty
        assert loaded.scores is None
        assert loaded.predictions is None

    def test_save_load_with_none_dataframes(self, tmp_path):
        """Test roundtrip when all DataFrames are None (feature selection only)."""
        result = ModuleResult(
            result_type=ResultType.BEST,
            module="roc",
            selected_features=["f1"],
        )

        result_dir = UPath(tmp_path / "best")
        result.save(result_dir)

        # Only selected_features.json should exist (no parquets for empty dfs)
        assert (result_dir / "selected_features.json").exists()
        assert not (result_dir / "scores.parquet").exists()
        assert not (result_dir / "predictions.parquet").exists()
        assert not (result_dir / "feature_importances.parquet").exists()

        loaded = ModuleResult.load(result_dir, result_type=ResultType.BEST, module="roc")
        assert loaded.selected_features == ["f1"]
        assert loaded.scores is None
        assert loaded.predictions is None
        assert loaded.feature_importances is None

    def test_module_and_result_type_columns_stamped(self, tmp_path):
        """Test that module and result_type columns are stamped on saved parquets."""
        scores = pd.DataFrame(
            {
                "metric": ["AUCROC"],
                "value": [0.9],
            }
        )

        result = ModuleResult(
            result_type=ResultType.BEST,
            module="octo",
            selected_features=["f1"],
            scores=scores,
        )

        result_dir = UPath(tmp_path / "best")
        result.save(result_dir)

        loaded_df = parquet_load(result_dir / "scores.parquet")
        assert "module" in loaded_df.columns
        assert "result_type" in loaded_df.columns
        assert loaded_df.iloc[0]["module"] == "octo"
        assert loaded_df.iloc[0]["result_type"] == "best"

    def test_predictor_json_contents(self, tmp_path):
        """Test that predictor.json has correct selected_features."""
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        model.fit([[1, 2, 3], [4, 5, 6]], [0, 1])

        result = ModuleResult(
            result_type=ResultType.BEST,
            module="octo",
            selected_features=["a", "b", "c"],
            model=model,
        )

        result_dir = UPath(tmp_path / "best")
        result.save(result_dir)

        with (result_dir / "model" / "predictor.json").open() as f:
            data = json.load(f)

        assert data["selected_features"] == ["a", "b", "c"]

    def test_ensemble_selection_result_type(self, tmp_path):
        """Test save/load with ENSEMBLE_SELECTION result type."""
        result = ModuleResult(
            result_type=ResultType.ENSEMBLE_SELECTION,
            module="octo",
            selected_features=["f1"],
            scores=pd.DataFrame({"metric": ["AUCROC"], "value": [0.88]}),
        )

        result_dir = UPath(tmp_path / "ensemble_selection")
        result.save(result_dir)

        loaded = ModuleResult.load(result_dir, result_type=ResultType.ENSEMBLE_SELECTION, module="octo")
        assert loaded.result_type == ResultType.ENSEMBLE_SELECTION
        assert loaded.selected_features == ["f1"]
        assert loaded.scores is not None
        assert not loaded.scores.empty
