"""Base task classes: config (user-facing) and execution (internal)."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

import joblib
import pandas as pd
from attrs import define, field, validators
from upath import UPath

from octopus.study.context import StudyContext


class ResultType(StrEnum):
    """Types of results produced by modules."""

    BEST = "best"
    ENSEMBLE_SELECTION = "ensemble_selection"


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
            with (model_dir / "model.joblib").open("wb") as f:
                joblib.dump(self.model, f)
            predictor_state = {"selected_features": self.selected_features}
            with (model_dir / "predictor.json").open("w") as f:
                json.dump(predictor_state, f, indent=2)

    @classmethod
    def load(cls, result_dir: UPath, result_type: ResultType, module: str) -> ModuleResult:
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
            with model_path.open("rb") as f:
                model = joblib.load(f)

        return cls(
            result_type=result_type,
            module=module,
            selected_features=selected_features,
            scores=scores,
            predictions=predictions,
            feature_importances=feature_importances,
            model=model,
        )


class FIMethod(StrEnum):
    """Feature importance computation methods."""

    INTERNAL = "internal"
    PERMUTATION = "permutation"
    SHAP = "shap"
    LOFO = "lofo"
    CONSTANT = "constant"
    COUNTS = "counts"
    COUNTS_RELATIVE = "counts_relative"


class FIDataset(StrEnum):
    """Dataset partitions for feature importance computation."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


@define
class Task(ABC):
    """Base config class for all workflow tasks."""

    task_id: int = field(validator=[validators.instance_of(int), validators.ge(0)])
    depends_on: int | None = field(default=None, validator=validators.optional(validators.instance_of(int)))
    description: str = field(default="", validator=[validators.instance_of(str)])
    categorical_encoding: bool = field(default=False, validator=[validators.instance_of(bool)])

    @property
    def module(self) -> str:
        """Module name derived from class name."""
        return type(self).__name__.lower()

    @abstractmethod
    def create_module(self) -> ModuleExecution:
        """Create an execution module from this config."""
        raise NotImplementedError("Subclasses must implement create_module()")


@define
class ModuleExecution[T: Task](ABC):
    """Base execution class. Created on worker via config.create_module()."""

    config: T = field()

    @abstractmethod
    def fit(
        self,
        data_traindev: pd.DataFrame,
        data_test: pd.DataFrame,
        feature_cols: list[str],
        study_context: StudyContext,
        outersplit_id: int,
        output_dir: UPath,
        num_assigned_cpus: int = 1,
        feature_groups: dict | None = None,
        prior_results: dict | None = None,
    ) -> dict[ResultType, ModuleResult]:
        """Fit the module. Returns dict mapping ResultType to ModuleResult."""
        raise NotImplementedError("Subclasses must implement fit()")

    def is_fitted(self) -> bool:
        """Check if module has been fitted."""
        if hasattr(self, "selected_features_"):
            return self.selected_features_ is not None
        return False


@define
class FeatureSelectionExecution[T: Task](ModuleExecution[T]):
    """Execution class for feature selection modules."""

    selected_features_: list[str] | None = field(init=False, default=None)
    feature_importances_: dict | None = field(init=False, default=None)

    def is_fitted(self) -> bool:
        """Check if module has been fitted."""
        return self.selected_features_ is not None


@define
class MLModuleExecution[T: Task](FeatureSelectionExecution[T]):
    """Execution class for ML modules that train predictive models."""

    model_: Any = field(init=False, default=None)

    def is_fitted(self) -> bool:
        """Check if module has been fitted."""
        return self.model_ is not None
