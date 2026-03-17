"""Base task classes: config (user-facing) and execution (internal)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
from attrs import define, field, validators
from upath import UPath

from .context import StudyContext
from .result import ModuleResult, ResultType


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
        *,
        data_traindev: pd.DataFrame,
        data_test: pd.DataFrame,
        feature_cols: list[str],
        study_context: StudyContext,
        outersplit_id: int,
        results_dir: UPath,
        scratch_dir: UPath,
        num_assigned_cpus: int,
        feature_groups: dict[str, list[str]],
        prior_results: dict[str, pd.DataFrame],
    ) -> dict[ResultType, ModuleResult]:
        """Fit the module. Returns dict mapping ResultType to ModuleResult."""
        raise NotImplementedError("Subclasses must implement fit()")

    def is_fitted(self) -> bool:
        """Check if module has been fitted."""
        if hasattr(self, "selected_features_"):
            return self.selected_features_ is not None
        return False
