"""StudyDiagnostics — Optuna-only study-level diagnostics from saved parquet files.

Provides exploration of Optuna hyperparameter tuning results without loading
any models. All data comes from saved ``optuna_results.parquet`` artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from octopus.diagnostics._data_loader import load_optuna
from octopus.diagnostics._plots import (
    plot_optuna_hyperparameters_chart,
    plot_optuna_trial_counts_chart,
    plot_optuna_trials_chart,
)
from octopus.types import MLType


class StudyDiagnostics:
    """Optuna-only study-level diagnostics from saved parquet files.

    Loads Optuna trial results from the study directory structure.
    No model loading is performed.

    Args:
        study_path: Path to the study directory.

    Raises:
        FileNotFoundError: If the study directory does not exist.

    Example::

        from octopus.diagnostics import StudyDiagnostics

        diag = StudyDiagnostics("./studies/my_study/")
        diag.plot_optuna_trial_counts()
        diag.plot_optuna_trials()
        diag.plot_optuna_hyperparameters()
    """

    def __init__(self, study_path: str | Path) -> None:
        self._study_path = Path(study_path)
        if not self._study_path.exists():
            raise FileNotFoundError(f"Study path does not exist: {self._study_path}")

        # Load config
        config_path = self._study_path / "study_config.json"
        if config_path.exists():
            with open(config_path) as f:
                self._config: dict[str, Any] = json.load(f)
        else:
            self._config = {}

        # Lazy-loaded Optuna DataFrame
        self._optuna: pd.DataFrame | None = None

    # ── Properties ──────────────────────────────────────────────

    @property
    def study_path(self) -> Path:
        """Path to the study directory."""
        return self._study_path

    @property
    def config(self) -> dict[str, Any]:
        """Study configuration dictionary."""
        return self._config

    @property
    def ml_type(self) -> MLType:
        """Machine learning type (classification, regression, timetoevent)."""
        return MLType(self._config.get("ml_type", ""))

    @property
    def optuna_trials(self) -> pd.DataFrame:
        """All Optuna trial results across outersplits and tasks (lazy-loaded)."""
        if self._optuna is None:
            self._optuna = load_optuna(self._study_path)
        return self._optuna

    # ── Plot Methods ────────────────────────────────────────────

    def plot_optuna_trial_counts(self) -> go.Figure:
        """Plot bar chart of unique trial counts per model type.

        Returns:
            Plotly Figure.
        """
        return plot_optuna_trial_counts_chart(self.optuna_trials)

    def plot_optuna_trials(
        self,
        outersplit_id: int = 0,
        task_id: int = 0,
        direction: str = "minimize",
    ) -> go.Figure:
        """Plot Optuna trial scatter + cumulative best line.

        Args:
            outersplit_id: Outer split to filter on.
            task_id: Task to filter on.
            direction: Optimization direction ('minimize' or 'maximize').

        Returns:
            Plotly Figure.
        """
        return plot_optuna_trials_chart(
            self.optuna_trials,
            outersplit_id=outersplit_id,
            task_id=task_id,
            direction=direction,
        )

    def plot_optuna_hyperparameters(
        self,
        outersplit_id: int = 0,
        task_id: int = 0,
        model_type: str = "",
    ) -> go.Figure:
        """Plot Optuna hyperparameter scatter plots.

        Args:
            outersplit_id: Outer split to filter on.
            task_id: Task to filter on.
            model_type: Model type to filter on.

        Returns:
            Plotly Figure.
        """
        return plot_optuna_hyperparameters_chart(
            self.optuna_trials,
            outersplit_id=outersplit_id,
            task_id=task_id,
            model_type=model_type,
        )
