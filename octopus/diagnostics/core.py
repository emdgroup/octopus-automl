"""StudyDiagnostics — interactive study-level diagnostics from saved parquet files.

Provides exploration of predictions, feature importances, scores, and Optuna
hyperparameter tuning results without loading any models. All data comes from
saved parquet artifacts on disk.

If ``ipywidgets`` is installed, plot methods offer interactive dropdown
selection. Otherwise, filter parameters must be passed explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from octopus.diagnostics._data_loader import (
    load_fi,
    load_optuna,
    load_predictions,
    load_scores,
)
from octopus.diagnostics._plots import (
    plot_confusion_matrix_chart,
    plot_fi_chart,
    plot_optuna_hyperparameters_chart,
    plot_optuna_trial_counts_chart,
    plot_optuna_trials_chart,
    plot_predictions_vs_truth_chart,
)
from octopus.types import FIResultLabel, MetricDirection, MLType


def _has_ipywidgets() -> bool:
    """Check if ipywidgets is available."""
    try:
        import ipywidgets  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def _unique_sorted(series: pd.Series) -> list[str]:
    """Get sorted unique string values from a Series."""
    return sorted(series.dropna().astype(str).unique())


class StudyDiagnostics:
    """Interactive study-level diagnostics from saved parquet files.

    Loads predictions, feature importances, scores, and Optuna results
    from the study directory structure. No model loading is performed.

    Args:
        study_path: Path to the study directory.

    Raises:
        FileNotFoundError: If the study directory or study_config.json does not exist.

    Example::

        from octopus.diagnostics import StudyDiagnostics

        diag = StudyDiagnostics("./studies/my_study/")
        diag.plot_fi()
        diag.plot_optuna_trials()
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

        # Lazy-loaded DataFrames
        self._predictions: pd.DataFrame | None = None
        self._fi: pd.DataFrame | None = None
        self._optuna: pd.DataFrame | None = None
        self._scores: pd.DataFrame | None = None

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
    def predictions(self) -> pd.DataFrame:
        """All predictions across outer splits and tasks (lazy-loaded)."""
        if self._predictions is None:
            self._predictions = load_predictions(self._study_path)
        return self._predictions

    @property
    def fi(self) -> pd.DataFrame:
        """All feature importances across outer splits and tasks (lazy-loaded)."""
        if self._fi is None:
            self._fi = load_fi(self._study_path)
        return self._fi

    @property
    def optuna_trials(self) -> pd.DataFrame:
        """All Optuna trial results across outer splits and tasks (lazy-loaded)."""
        if self._optuna is None:
            self._optuna = load_optuna(self._study_path)
        return self._optuna

    @property
    def scores(self) -> pd.DataFrame:
        """All scores across outer splits and tasks (lazy-loaded)."""
        if self._scores is None:
            self._scores = load_scores(self._study_path)
        return self._scores

    # ── Filter helpers ──────────────────────────────────────────

    def _get_filter_options(self, df: pd.DataFrame, columns: list[str]) -> dict[str, list[str]]:
        """Extract unique sorted values for each column.

        Args:
            df: DataFrame to extract from.
            columns: Column names.

        Returns:
            Dict mapping column name to sorted unique string values.
        """
        return {col: _unique_sorted(df[col]) for col in columns if col in df.columns}

    # ── Interactive Plots ───────────────────────────────────────

    def plot_fi(
        self,
        outer_split_id: int | None = None,
        task_id: int | None = None,
        training_id: str | None = None,
        fi_method: str | FIResultLabel | None = None,
    ) -> None:
        """Plot feature importance bar chart.

        If ipywidgets is available and parameters are None, shows interactive
        dropdowns. Otherwise uses provided values or defaults.

        Args:
            outer_split_id: Outer split to filter on.
            task_id: Task to filter on.
            training_id: Training ID to filter on.
            fi_method: FI method to filter on.
        """
        df = self.fi
        if df.empty:
            print("No feature importance data found.")
            return

        if _has_ipywidgets() and outer_split_id is None:
            from ipywidgets import Dropdown, interact  # noqa: PLC0415

            opts = self._get_filter_options(df, ["outer_split_id", "task_id", "training_id", "fi_method"])

            @interact(
                outer_split_id=Dropdown(options=opts.get("outer_split_id", ["0"]), description="Outer Split:"),
                task_id=Dropdown(options=opts.get("task_id", ["0"]), description="Task:"),
                training_id=Dropdown(options=opts.get("training_id", [""]), description="Training:"),
                fi_method=Dropdown(options=opts.get("fi_method", [""]), description="FI Method:"),
            )
            def _plot(outer_split_id: str, task_id: str, training_id: str, fi_method: str) -> None:
                fig = plot_fi_chart(
                    df, outer_split_id=outer_split_id, task_id=task_id, training_id=training_id, fi_method=fi_method
                )
                fig.show()
        else:
            fig = plot_fi_chart(
                df,
                outer_split_id=outer_split_id or 0,
                task_id=task_id or 0,
                training_id=training_id or "",
                fi_method=fi_method or "",
            )
            fig.show()

    def plot_confusion_matrix(
        self,
        outer_split_id: int | None = None,
        task_id: int | None = None,
        training_id: str | None = None,
    ) -> None:
        """Plot confusion matrix heatmap (classification only).

        Args:
            outer_split_id: Outer split to filter on.
            task_id: Task to filter on.
            training_id: Inner split / training ID to filter on.
        """
        df = self.predictions
        if df.empty:
            print("No prediction data found.")
            return

        if _has_ipywidgets() and outer_split_id is None:
            from ipywidgets import Dropdown, interact  # noqa: PLC0415

            opts = self._get_filter_options(df, ["outer_split_id", "task_id", "inner_split_id"])

            @interact(
                outer_split_id=Dropdown(options=opts.get("outer_split_id", ["0"]), description="Outer Split:"),
                task_id=Dropdown(options=opts.get("task_id", ["0"]), description="Task:"),
                training_id=Dropdown(options=opts.get("inner_split_id", [""]), description="Training:"),
            )
            def _plot(outer_split_id: str, task_id: str, training_id: str) -> None:
                fig = plot_confusion_matrix_chart(
                    df, outer_split_id=outer_split_id, task_id=task_id, training_id=training_id
                )
                fig.show()
        else:
            fig = plot_confusion_matrix_chart(
                df,
                outer_split_id=outer_split_id or 0,
                task_id=task_id or 0,
                training_id=training_id or "",
            )
            fig.show()

    def plot_predictions_vs_truth(
        self,
        outer_split_id: int | None = None,
        task_id: int | None = None,
        training_id: str | None = None,
    ) -> None:
        """Plot prediction vs ground truth scatter (regression only).

        Args:
            outer_split_id: Outer split to filter on.
            task_id: Task to filter on.
            training_id: Inner split / training ID to filter on.
        """
        df = self.predictions
        if df.empty:
            print("No prediction data found.")
            return

        if _has_ipywidgets() and outer_split_id is None:
            from ipywidgets import Dropdown, interact  # noqa: PLC0415

            opts = self._get_filter_options(df, ["outer_split_id", "task_id", "inner_split_id"])

            @interact(
                outer_split_id=Dropdown(options=opts.get("outer_split_id", ["0"]), description="Outer Split:"),
                task_id=Dropdown(options=opts.get("task_id", ["0"]), description="Task:"),
                training_id=Dropdown(options=opts.get("inner_split_id", [""]), description="Training:"),
            )
            def _plot(outer_split_id: str, task_id: str, training_id: str) -> None:
                fig = plot_predictions_vs_truth_chart(
                    df, outer_split_id=outer_split_id, task_id=task_id, training_id=training_id
                )
                fig.show()
        else:
            fig = plot_predictions_vs_truth_chart(
                df,
                outer_split_id=outer_split_id or 0,
                task_id=task_id or 0,
                training_id=training_id or "",
            )
            fig.show()

    def plot_optuna_trial_counts(self) -> None:
        """Plot bar chart of unique trial counts per model type."""
        df = self.optuna_trials
        if df.empty:
            print("No Optuna data found.")
            return
        fig = plot_optuna_trial_counts_chart(df)
        fig.show()

    def plot_optuna_trials(
        self,
        outer_split_id: int | None = None,
        task_id: int | None = None,
        direction: MetricDirection = MetricDirection.MINIMIZE,
    ) -> None:
        """Plot Optuna trial scatter + cumulative best line.

        Args:
            outer_split_id: Outer split to filter on.
            task_id: Task to filter on.
            direction: Optimization direction ('minimize' or 'maximize').
        """
        df = self.optuna_trials
        if df.empty:
            print("No Optuna data found.")
            return

        if _has_ipywidgets() and outer_split_id is None:
            from ipywidgets import Dropdown, interact  # noqa: PLC0415

            opts = self._get_filter_options(df, ["outer_split_id", "task_id"])

            @interact(
                outer_split_id=Dropdown(options=opts.get("outer_split_id", ["0"]), description="Outer Split:"),
                task_id=Dropdown(options=opts.get("task_id", ["0"]), description="Task:"),
            )
            def _plot(outer_split_id: str, task_id: str) -> None:
                fig = plot_optuna_trials_chart(df, outer_split_id=outer_split_id, task_id=task_id, direction=direction)
                fig.show()
        else:
            fig = plot_optuna_trials_chart(
                df,
                outer_split_id=outer_split_id or 0,
                task_id=task_id or 0,
                direction=direction,
            )
            fig.show()

    def plot_optuna_hyperparameters(
        self,
        outer_split_id: int | None = None,
        task_id: int | None = None,
        model_type: str | None = None,
    ) -> None:
        """Plot Optuna hyperparameter scatter plots.

        Args:
            outer_split_id: Outer split to filter on.
            task_id: Task to filter on.
            model_type: Model type to filter on.
        """
        df = self.optuna_trials
        if df.empty:
            print("No Optuna data found.")
            return

        if _has_ipywidgets() and outer_split_id is None:
            from ipywidgets import Dropdown, interact  # noqa: PLC0415

            opts = self._get_filter_options(df, ["outer_split_id", "task_id", "model_type"])

            @interact(
                outer_split_id=Dropdown(options=opts.get("outer_split_id", ["0"]), description="Outer Split:"),
                task_id=Dropdown(options=opts.get("task_id", ["0"]), description="Task:"),
                model_type=Dropdown(options=opts.get("model_type", [""]), description="Model:"),
            )
            def _plot(outer_split_id: str, task_id: str, model_type: str) -> None:
                fig = plot_optuna_hyperparameters_chart(
                    df, outer_split_id=outer_split_id, task_id=task_id, model_type=model_type
                )
                fig.show()
        else:
            fig = plot_optuna_hyperparameters_chart(
                df,
                outer_split_id=outer_split_id or 0,
                task_id=task_id or 0,
                model_type=model_type or "",
            )
            fig.show()
