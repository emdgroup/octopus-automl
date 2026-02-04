"""Base core class with shared functionality for all module cores."""

import pandas as pd
from attrs import define, field, validators
from upath import UPath

from octopus.experiment import OctoExperiment
from octopus.task import Task


@define
class ModuleBaseCore[TaskConfigType: Task]:
    """Base class for module cores providing common functionality.

    Provides shared properties and initialization logic for all module cores:
    - Path management (path_module, path_results)
    - Data access properties (x_traindev, y_traindev, x_test, y_test, etc.)
    - Experiment metadata (feature_cols, target_metric, ml_type, etc.)
    - Directory initialization and cleanup

    Type Parameters:
        TaskConfigType: The module configuration type (must be a Task subclass)

    Attributes:
        experiment: The OctoExperiment instance containing configuration and data
        log_dir: Directory for individual worker logs
    """

    experiment: OctoExperiment[TaskConfigType] = field(validator=[validators.instance_of(OctoExperiment)])
    log_dir: UPath = field(validator=[validators.instance_of(UPath)])

    @property
    def path_module(self) -> UPath:
        """Module directory path.

        Returns:
            Path to module results directory: {study_path}/{task_path}
        """
        return self.experiment.path_study / self.experiment.task_path

    @property
    def path_results(self) -> UPath:
        """Results directory path within module.

        Returns:
            Path to results directory: {path_module}/results
        """
        return self.path_module / "results"

    @property
    def data_traindev(self) -> pd.DataFrame:
        """Training and development dataset.

        Returns:
            Full training/development dataset from experiment
        """
        return self.experiment.data_traindev

    @property
    def data_test(self) -> pd.DataFrame:
        """Test dataset.

        Returns:
            Full test dataset from experiment
        """
        return self.experiment.data_test

    @property
    def x_traindev(self) -> pd.DataFrame:
        """Feature matrix for training/development set.

        Returns:
            Subset of data_traindev containing only feature columns
        """
        return self.experiment.x_traindev

    @property
    def y_traindev(self) -> pd.DataFrame:
        """Target values for training/development set.

        Returns:
            Subset of data_traindev containing only target columns
        """
        return self.experiment.y_traindev

    @property
    def x_test(self) -> pd.DataFrame:
        """Feature matrix for test set.

        Returns:
            Subset of data_test containing only feature columns
        """
        return self.experiment.x_test

    @property
    def y_test(self) -> pd.DataFrame:
        """Target values for test set.

        Returns:
            Subset of data_test containing only target columns
        """
        return self.experiment.y_test

    @property
    def feature_cols(self) -> list[str]:
        """Feature column names.

        Returns:
            List of feature column names used in the experiment
        """
        return self.experiment.feature_cols

    @property
    def target_assignments(self) -> dict:
        """Target column assignments.

        Returns:
            Dictionary mapping target names to column names
        """
        return self.experiment.target_assignments

    @property
    def target_metric(self) -> str:
        """Primary target metric for optimization.

        Returns:
            Name of the metric to optimize for
        """
        return self.experiment.target_metric

    @property
    def ml_type(self) -> str:
        """Machine learning problem type.

        Returns:
            One of: "classification", "regression", "timetoevent"
        """
        return self.experiment.ml_type

    @property
    def config(self) -> TaskConfigType:
        """Module-specific configuration.

        Returns:
            The ml_config from experiment (typed as TaskConfigType)
        """
        return self.experiment.ml_config

    @property
    def metrics(self) -> list[str]:
        """All metrics to track during execution.

        Returns:
            List of metric names to calculate
        """
        return self.experiment.metrics

    @property
    def stratification_column(self) -> str | None:
        """Column to use for stratified splitting.

        Returns:
            Column name for stratification, or None if not stratified
        """
        return self.experiment.stratification_column

    @property
    def row_column(self) -> str:
        """Row identifier column name.

        Returns:
            Name of the column containing row identifiers
        """
        return self.experiment.row_column

    @property
    def row_traindev(self) -> pd.Series:
        """Row identifiers for training/development set.

        Returns:
            Series containing row identifiers from data_traindev
        """
        return self.experiment.row_traindev

    @property
    def row_test(self) -> pd.Series:
        """Row identifiers for test set.

        Returns:
            Series containing row identifiers from data_test
        """
        return self.experiment.row_test

    def __attrs_post_init__(self) -> None:
        """Post-initialization hook to prepare results directory.

        Called automatically after attrs initialization. Resets the results
        directory to ensure a clean state for module execution.

        Can be overridden by subclasses that need custom initialization.
        Subclasses should call super().__attrs_post_init__() if they want
        the ModuleBaseCore default directory setup behavior.
        """
        self._reset_results_dir()

    def _reset_results_dir(self) -> None:
        """Reset and recreate the results directory.

        Deletes existing results directory if present and creates a fresh one.
        This ensures a clean state for each module run and prevents stale files
        from previous runs interfering with new results.
        """
        for directory in [self.path_results]:
            if directory.exists():
                directory.rmdir(recursive=True)
            directory.mkdir(parents=True, exist_ok=True)
