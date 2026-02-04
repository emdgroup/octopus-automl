"""Octo Experiment module."""

import gzip
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats
from attrs import Factory, define, field, validators
from upath import UPath

from octopus.task import Task

if TYPE_CHECKING:
    from octopus.results import ModuleResults


@define
class OctoExperiment[ConfigType: Task]:
    """Represents an Octopus experiment for ML pipeline execution.

    An OctoExperiment exists in two distinct states representing different stages of the
    ML workflow. The lifecycle begins with base experiments created during cross-validation
    data splitting. These base experiments serve as templates containing only the train/test
    data splits. When the pipeline executes, the manager deep copies base experiments and
    transforms them into workflow experiments by attaching ML module configurations (e.g.,
    feature selection, model training). This two-stage design separates data preparation
    from pipeline execution, allowing the same data splits to be reused across different
    pipeline configurations.
    """

    id: str = field(validator=[validators.instance_of(str)])
    """ID"""

    experiment_id: int = field(validator=[validators.instance_of(int), validators.ge(0)])
    """Identifier for the experiment."""

    task_id: int | None = field(
        validator=validators.optional(validators.and_(validators.instance_of(int), validators.ge(0)))
    )
    """Identifier for the workflow task."""

    depends_on_task: int | None = field(
        validator=validators.optional(validators.and_(validators.instance_of(int), validators.ge(-1)))
    )
    """Identifier for the input workflow task."""

    _task_path: UPath | None = field(
        validator=validators.optional(validators.instance_of(UPath)),
        converter=lambda x: UPath(x) if x is not None else None,
    )
    """Internal path storage. Use task_path property to access safely."""

    study_path: UPath = field(validator=[validators.instance_of(UPath)], converter=lambda x: UPath(x))
    """Path where the study is stored."""

    study_name: str = field(validator=[validators.instance_of(str)])
    """Name of the study."""

    ml_type: str = field(validator=[validators.instance_of(str)])
    """Type of machine learning task."""

    target_metric: str = field(validator=[validators.instance_of(str)])
    """Primary metric for model evaluation."""

    positive_class: int | None = field(validator=validators.optional(validators.instance_of(int)))
    """Positive class label for binary classification. None for regression, time-to-event, and multiclass."""

    metrics: list[str] = field(validator=[validators.instance_of(list)])
    """List of metrics to calculate."""

    imputation_method: str = field(validator=[validators.instance_of(str)])
    """Method used for imputing missing values."""

    datasplit_column: str = field(validator=[validators.instance_of(str)])
    """Column name used for data splitting."""

    row_column: str = field(validator=[validators.instance_of(str)])
    """Column name used as row identifier."""

    feature_cols: list[str] = field(validator=[validators.instance_of(list)])
    """List of column names used as features in the experiment."""

    target_assignments: dict[str, str] = field(validator=[validators.instance_of(dict)])
    """Mapping of target variables to their assignments."""

    data_traindev: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    """DataFrame containing training and development data."""

    data_test: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    """DataFrame containing test data."""

    stratification_column: str | None = field(default=None, validator=validators.optional(validators.instance_of(str)))
    """Column name used for stratification, if applicable."""

    ml_module: str = field(init=False, default="", validator=[validators.instance_of(str)])
    """Name of the machine learning module used."""

    num_assigned_cpus: int = field(init=False, default=0, validator=[validators.instance_of(int)])
    """Number of CPUs assigned to the experiment."""

    ml_config: ConfigType = field(init=False, default=None)
    """Configuration settings for the module used by the workflow task."""

    selected_features: list = field(default=Factory(list), validator=[validators.instance_of(list)])
    """List of features selected for the experiment."""

    feature_groups: dict = field(default=Factory(dict), validator=[validators.instance_of(dict)])
    """Groupings of features based on correlation analysis."""

    results: dict[str, "ModuleResults"] = field(default=Factory(dict), validator=[validators.instance_of(dict)])
    """Results of the experiment, keyed by result type."""

    prior_results: dict[str, "ModuleResults"] = field(default=Factory(dict), validator=[validators.instance_of(dict)])
    """Results of the experiment used as input, keyed by result type."""

    @property
    def path_study(self) -> UPath:
        """Get study path."""
        return UPath(self.study_path, self.study_name)

    @property
    def is_base_experiment(self) -> bool:
        """Check if this is a base experiment (no task_id)."""
        return self.task_id is None

    @property
    def is_workflow_experiment(self) -> bool:
        """Check if this is a workflow experiment (has task_id)."""
        return self.task_id is not None

    @property
    def task_path(self) -> UPath:
        """Get the workflow task path.

        Use this in modules that require a fully initialized experiment
        (not a base experiment).

        Returns:
            UPath: The workflow task path

        Raises:
            ValueError: If this is a base experiment.
            RuntimeError: If validation failed and _task_path is None for a workflow experiment.
        """
        if self.is_base_experiment:
            raise ValueError(
                "Cannot access task_path on a base experiment. "
                "This operation requires a workflow experiment with task_path set."
            )
        if self._task_path is None:
            raise RuntimeError(
                "Validation failed: workflow experiment has no task_path set. "
                f"This should not happen (task_id={self.task_id})"
            )
        return self._task_path

    def __attrs_post_init__(self):
        self._validate_experiment_state()
        self.feature_groups = self.calculate_feature_groups(self.feature_cols)

    def _validate_experiment_state(self) -> None:
        """Validate consistency between base and workflow experiment fields.

        Ensures that workflow-related fields (task_id, depends_on_task,
        _task_path) are consistent with the experiment type (base vs sequence).

        Raises:
            ValueError: If fields are inconsistent with the experiment type.
        """
        if self.task_id is None:
            if self._task_path is not None:
                raise ValueError(
                    f"Base experiments (task_id=None) cannot have _task_path set. Got _task_path={self._task_path}"
                )
            if self.depends_on_task is not None:
                raise ValueError(
                    "Base experiments (task_id=None) cannot have depends_on_task set. "
                    f"Got depends_on_task={self.depends_on_task}"
                )
        else:
            if self._task_path is None:
                raise ValueError(f"Workflow experiments (task_id={self.task_id}) must have _task_path set")
            if self.depends_on_task is None:
                raise ValueError(f"Workflow experiments (task_id={self.task_id}) must have depends_on_task set")

    def calculate_feature_groups(self, feature_cols: list[str]) -> dict[str, list[str]]:
        """Calculate feature groups based on correlation thresholds."""
        if len(feature_cols) <= 2:
            logging.warning("Not enough features to calculate correlations for feature groups.")
            return {}
        logging.info("Calculating feature groups.")

        auto_group_thresholds = [0.7, 0.8, 0.9]
        auto_groups = []

        pos_corr_matrix, _ = scipy.stats.spearmanr(np.nan_to_num(self.data_traindev[feature_cols].values))
        pos_corr_matrix = np.abs(pos_corr_matrix)

        # get groups depending on threshold
        for threshold in auto_group_thresholds:
            g: nx.Graph = nx.Graph()
            for i in range(len(feature_cols)):
                for j in range(i + 1, len(feature_cols)):
                    if pos_corr_matrix[i, j] > threshold:
                        g.add_edge(i, j)

            # Get connected components and sort them to ensure determinism
            subgraphs = [
                g.subgraph(c).copy() for c in sorted(nx.connected_components(g), key=lambda x: (len(x), sorted(x)))
            ]
            # Create groups of feature columns
            groups = []
            for sg in subgraphs:
                groups.append([feature_cols[node] for node in sorted(sg.nodes())])
            auto_groups.extend([sorted(g) for g in groups])

        # find unique groups
        auto_groups_unique = [list(t) for t in set(map(tuple, auto_groups))]

        return {f"group{i}": group for i, group in enumerate(auto_groups_unique)}

    def to_pickle(self, file_path: str | Path | UPath):
        """Save object to a compressed pickle file."""
        with UPath(file_path).open("wb") as file, gzip.GzipFile(fileobj=file, mode="wb") as gzip_file:
            pickle.dump(self, gzip_file)

    @classmethod
    def from_pickle(cls, file_path: str | Path | UPath) -> "OctoExperiment":
        """Load object from a compressed pickle file."""
        with UPath(file_path).open("rb") as file, gzip.GzipFile(fileobj=file, mode="rb") as gzip_file:
            data = pickle.load(gzip_file)

        if not isinstance(data, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}")

        return data
