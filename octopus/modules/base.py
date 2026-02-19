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
    load_task: bool = field(default=False, validator=[validators.instance_of(bool)])
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
    ) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fit the module. Returns (selected_features, scores, predictions, feature_importances)."""
        raise NotImplementedError("Subclasses must implement fit()")

    def save(self, path: UPath) -> None:
        """Save fitted module to disk."""
        path.mkdir(parents=True, exist_ok=True)

        if hasattr(self, "model_") and self.model_ is not None:
            with (path / "model.joblib").open("wb") as f:
                joblib.dump(self.model_, f)

        state = {
            "selected_features": getattr(self, "selected_features_", None),
            "feature_importances": getattr(self, "feature_importances_", None),
        }
        with (path / "module_state.json").open("w") as f:
            json.dump(state, f, indent=2, default=str)

        if hasattr(self, "model_") and self.model_ is not None:
            predictor_state = {
                "selected_features": getattr(self, "selected_features_", []) or [],
            }
            with (path / "predictor.json").open("w") as f:
                json.dump(predictor_state, f, indent=2)

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
