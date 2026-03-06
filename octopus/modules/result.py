"""Internally used result container for a single result type from a module/task."""

import json
from typing import Any

import pandas as pd
from attrs import define, field
from upath import UPath

from octopus.types import ResultType
from octopus.utils import joblib_load, joblib_save


@define
class ModuleResult:
    """Unified result container for a single result type from a module.

    Carries all 5 artifacts (selected_features, scores, predictions,
    feature_importances, model) and knows how to save/load itself.
    Each result_type gets its own directory on disk.
    """

    result_type: ResultType = field()
    module: str = field()
    selected_features: list[str] = field(factory=list)
    scores: pd.DataFrame | None = field(default=None)
    predictions: pd.DataFrame | None = field(default=None)
    feature_importances: pd.DataFrame | None = field(default=None)
    model: Any = field(default=None)

    def save(self, result_dir: UPath) -> None:
        """Save this result to a directory.

        Stamps module + result_type columns on DataFrames, saves parquets,
        selected_features.json, and model/ subdirectory if model is not None.

        Args:
            result_dir: Directory to save into (e.g. task0/best/)
        """
        result_dir.mkdir(parents=True, exist_ok=True)

        # Save selected_features.json
        with (result_dir / "selected_features.json").open("w") as f:
            json.dump(self.selected_features, f)

        # Save DataFrames with module + result_type columns stamped
        for name, df in [
            ("scores", self.scores),
            ("predictions", self.predictions),
            ("feature_importances", self.feature_importances),
        ]:
            if df is not None and not df.empty:
                out = df.copy()
                out["module"] = self.module
                out["result_type"] = self.result_type.value
                path = result_dir / f"{name}.parquet"
                out.to_parquet(str(path), storage_options=path.storage_options, engine="pyarrow")

        # Save model/ subdirectory if model exists
        if self.model is not None:
            model_dir = result_dir / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib_save(self.model, model_dir / "model.joblib")
            predictor_state = {"selected_features": self.selected_features}
            with (model_dir / "predictor.json").open("w") as f:
                json.dump(predictor_state, f, indent=2)

    @classmethod
    def load(cls, result_dir: UPath, result_type: ResultType, module: str) -> "ModuleResult":
        """Load a ModuleResult from a saved directory.

        Args:
            result_dir: Directory containing saved result files
            result_type: The ResultType for this directory
            module: Module name

        Returns:
            Reconstructed ModuleResult instance
        """
        # Load selected features
        sf_path = result_dir / "selected_features.json"
        if sf_path.exists():
            with sf_path.open() as f:
                selected_features = json.load(f)
        else:
            selected_features = []

        # Load DataFrames (None if file doesn't exist)
        scores: pd.DataFrame | None = None
        predictions: pd.DataFrame | None = None
        feature_importances: pd.DataFrame | None = None

        for name in ["scores", "predictions", "feature_importances"]:
            path = result_dir / f"{name}.parquet"
            if path.exists():
                df = pd.read_parquet(str(path), storage_options=path.storage_options, engine="pyarrow")
                if name == "scores":
                    scores = df
                elif name == "predictions":
                    predictions = df
                elif name == "feature_importances":
                    feature_importances = df

        # Load model if exists
        model = None
        model_dir = result_dir / "model"
        model_path = model_dir / "model.joblib"
        if model_path.exists():
            model = joblib_load(model_path)

        return cls(
            result_type=result_type,
            module=module,
            selected_features=selected_features,
            scores=scores,
            predictions=predictions,
            feature_importances=feature_importances,
            model=model,
        )
