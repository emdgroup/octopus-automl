"""StudyDiagnostics — Optuna-focused study diagnostics from saved parquet files.

Provides exploration of Optuna hyperparameter tuning results without
loading any models.  All data comes from saved parquet artifacts on disk.

For predictions, scores, feature importances, confusion matrices, and
ROC curves, use :mod:`octopus.predict.notebook_utils` instead.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from upath import UPath

from octopus.diagnostics._data_loader import load_feature_importances, load_optuna
from octopus.diagnostics._plots import (
    plot_feature_importance_chart,
    plot_optuna_hyperparameters_chart,
    plot_optuna_trial_counts_chart,
    plot_optuna_trials_chart,
)
from octopus.types import MetricDirection, MLType


class StudyDiagnostics:
    """Optuna-focused study diagnostics from saved parquet files.

    Loads Optuna trial results from the study directory structure.
    No model loading is performed.

    For predictions, scores, feature importances, and ML-specific plots,
    use ``octopus.predict.notebook_utils`` functions instead.

    Args:
        study_path: Path to the study directory (local or cloud via UPath).

    Raises:
        FileNotFoundError: If the study directory or study_config.json
            does not exist.

    Example::

        from octopus.diagnostics import StudyDiagnostics

        diag = StudyDiagnostics("./studies/my_study/")
        diag.plot_optuna_trial_counts()
        diag.plot_optuna_trials(outersplit_id=0, task_id=0)
    """

    def __init__(self, study_path: str | UPath) -> None:
        self._study_path = UPath(study_path)
        if not self._study_path.exists():
            raise FileNotFoundError(f"Study path does not exist: {self._study_path}")

        # Load config via UPath.open() for fsspec compatibility
        config_path = self._study_path / "study_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Study config not found: {config_path}")

        import json  # noqa: PLC0415

        with config_path.open() as f:
            self._config: dict[str, Any] = json.load(f)

        # Lazy-loaded DataFrames
        self._optuna: pd.DataFrame | None = None
        self._feature_importances: pd.DataFrame | None = None

    # ── Properties ──────────────────────────────────────────────

    @property
    def study_path(self) -> UPath:
        """Path to the study directory."""
        return self._study_path

    @property
    def config(self) -> dict[str, Any]:
        """Study configuration dictionary."""
        return self._config

    @property
    def ml_type(self) -> MLType:
        """Machine learning type (binary, multiclass, regression, timetoevent)."""
        return MLType(self._config["ml_type"])

    @property
    def optuna_trials(self) -> pd.DataFrame:
        """All Optuna trial results across outersplits and tasks (lazy-loaded)."""
        if self._optuna is None:
            self._optuna = load_optuna(self._study_path)
        return self._optuna

    @property
    def feature_importances(self) -> pd.DataFrame:
        """All saved feature importances across outersplits and tasks (lazy-loaded)."""
        if self._feature_importances is None:
            self._feature_importances = load_feature_importances(self._study_path)
        return self._feature_importances

    # ── Feature Importance ──────────────────────────────────────

    def get_feature_importances(
        self,
        task_id: int | None = None,
        module: str = "",
        result_type: str = "",
    ) -> pd.DataFrame:
        """Return the raw FI table filtered by task and module across all outersplits.

        This gives access to the raw saved feature importance data for
        custom analysis. Use :meth:`plot_feature_importance` for convenient
        visualization.

        Args:
            task_id: Filter by task ID. None = all tasks.
            module: Filter by module name (e.g. ``"octo"``). Empty = all.
            result_type: Filter by result type (e.g. ``"best"``). Empty = all.

        Returns:
            DataFrame with columns: ``feature``, ``importance``,
            ``fi_method``, ``fi_dataset``, ``training_id``, ``module``,
            ``result_type``, ``outersplit_id``, ``task_id``.
        """
        df = self.feature_importances
        if df.empty:
            return df
        if task_id is not None and "task_id" in df.columns:
            df = df[df["task_id"] == int(task_id)]
        if module and "module" in df.columns:
            df = df[df["module"] == module]
        if result_type and "result_type" in df.columns:
            df = df[df["result_type"] == result_type]
        return df.reset_index(drop=True)

    def plot_feature_importance(
        self,
        fi_table: pd.DataFrame | None = None,
        *,
        outersplit_id: int | None = None,
        task_id: int | None = None,
        fi_method: str = "",
        fi_dataset: str = "",
        training_id: str = "",
        module: str = "",
        result_type: str = "",
        top_n: int | None = 20,
    ) -> None:
        """Plot feature importances as a horizontal bar chart.

        Either pass a pre-filtered ``fi_table`` (e.g. from
        :meth:`get_feature_importances`) or let this method load all FI
        data and apply filters.

        Args:
            fi_table: Pre-filtered FI DataFrame. If None, loads all FI
                data and applies the remaining filter parameters.
            outersplit_id: Filter by outer split ID. None = all.
            task_id: Filter by task ID. None = all.
            fi_method: Filter by FI method (e.g. ``"internal"``, ``"permutation"``).
            fi_dataset: Filter by dataset partition (e.g. ``"train"``, ``"dev"``).
            training_id: Filter by training ID (e.g. ``"0_0_0"``).
            module: Filter by module name (e.g. ``"octo"``).
            result_type: Filter by result type (e.g. ``"best"``).
            top_n: Number of top features to show. None = all.
        """
        df = fi_table if fi_table is not None else self.feature_importances
        if df.empty:
            print("No feature importance data found.")
            return
        fig = plot_feature_importance_chart(
            df,
            outersplit_id=outersplit_id,
            task_id=task_id,
            fi_method=fi_method,
            fi_dataset=fi_dataset,
            training_id=training_id,
            module=module,
            result_type=result_type,
            top_n=top_n,
        )
        fig.show()

    # ── Optuna Plots ────────────────────────────────────────────

    def plot_optuna_trial_counts(self) -> None:
        """Plot bar chart of unique trial counts per model type.

        Displays a subplot grid with one panel per (task, outersplit)
        combination, showing how many unique trials each model type ran.
        """
        df = self.optuna_trials
        if df.empty:
            print("No Optuna data found.")
            return
        fig = plot_optuna_trial_counts_chart(df)
        fig.show()

    def plot_optuna_trials(
        self,
        outersplit_id: int = 0,
        task_id: int = 0,
        direction: MetricDirection = MetricDirection.MINIMIZE,
    ) -> None:
        """Plot Optuna trial scatter + cumulative best line.

        Args:
            outersplit_id: Outer split to filter on.
            task_id: Task to filter on.
            direction: Optimization direction ('minimize' or 'maximize').
        """
        df = self.optuna_trials
        if df.empty:
            print("No Optuna data found.")
            return
        fig = plot_optuna_trials_chart(
            df,
            outersplit_id=outersplit_id,
            task_id=task_id,
            direction=direction,
        )
        fig.show()

    def plot_optuna_hyperparameters(
        self,
        outersplit_id: int = 0,
        task_id: int = 0,
        model_type: str = "",
    ) -> None:
        """Plot Optuna hyperparameter scatter plots.

        Args:
            outersplit_id: Outer split to filter on.
            task_id: Task to filter on.
            model_type: Model type to filter on.
        """
        df = self.optuna_trials
        if df.empty:
            print("No Optuna data found.")
            return
        fig = plot_optuna_hyperparameters_chart(
            df,
            outersplit_id=outersplit_id,
            task_id=task_id,
            model_type=model_type,
        )
        fig.show()
