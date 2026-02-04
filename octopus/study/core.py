"""Octo Study."""

import json
import os
import sys
from abc import ABC, abstractmethod

import pandas as pd
from attrs import Factory, asdict, define, field, fields, has, validators
from upath import UPath

from octopus.experiment import OctoExperiment
from octopus.logger import get_logger, set_logger_filename
from octopus.manager.core import OctoManager
from octopus.metrics import Metrics
from octopus.modules import Octo
from octopus.task import Task
from octopus.utils import DataSplit

from .data_preparator import OctoDataPreparator
from .data_validator import OctoDataValidator
from .healthChecker import HealthCheckConfig, OctoDataHealthChecker
from .prepared_data import PreparedData
from .types import DatasplitType, ImputationMethod, MLType
from .validation import validate_start_with_empty_study, validate_workflow

logger = get_logger()

_RUNNING_IN_TESTSUITE = "RUNNING_IN_TESTSUITE" in os.environ


@define
class OctoStudy(ABC):
    """Abstract base class for all Octopus studies."""

    name: str = field(validator=[validators.instance_of(str)])
    """The name of the study."""

    feature_cols: list[str] = field(validator=[validators.instance_of(list)])
    """List of all feature columns in the dataset."""

    sample_id_col: str = field(validator=validators.instance_of(str))
    """Identifier for sample instances."""

    datasplit_type: DatasplitType = field(
        default=DatasplitType.SAMPLE,
        converter=lambda x: DatasplitType(x.lower()) if isinstance(x, str) else x,
        validator=validators.instance_of(DatasplitType),
    )
    """Type of datasplit. Allowed are `sample`, `group_features` and `group_sample_and_features`."""

    row_id: str | None = field(
        default=Factory(lambda: None),
        validator=validators.optional(validators.instance_of(str)),
    )
    """Unique row identifier."""

    stratification_column: str | None = field(
        default=Factory(lambda: None),
        validator=validators.optional(validators.instance_of(str)),
    )
    """Column used for stratification during data splitting."""

    n_folds_outer: int = field(default=5 if not _RUNNING_IN_TESTSUITE else 2, validator=[validators.instance_of(int)])
    """The number of outer folds for cross-validation. Defaults to 5."""

    datasplit_seed_outer: int = field(default=0, validator=[validators.instance_of(int)])
    """The seed used for data splitting in outer cross-validation. Defaults to 0."""

    imputation_method: ImputationMethod = field(
        default=ImputationMethod.MEDIAN,
        converter=lambda x: ImputationMethod(x.lower()) if isinstance(x, str) else x,
        validator=validators.instance_of(ImputationMethod),
    )

    ignore_data_health_warning: bool = field(default=Factory(lambda: False), validator=[validators.instance_of(bool)])
    """Ignore data health checks warning and run machine learning workflow."""

    outer_parallelization: bool = field(default=Factory(lambda: True), validator=[validators.instance_of(bool)])
    """Indicates whether outer parallelization is enabled. Defaults to True."""

    run_single_experiment_num: int = field(default=Factory(lambda: -1), validator=[validators.instance_of(int)])
    """Select a single experiment to execute. Defaults to -1 to run all experiments"""

    workflow: list[Task] = field(
        default=Factory(lambda: [Octo(task_id=0)]),
        validator=[validators.instance_of(list), validate_workflow],
    )
    """A list of tasks that defines the processing workflow. Each item in the list is an instance of `Task`."""

    start_with_empty_study: bool = field(
        default=True, validator=[validators.instance_of(bool), validate_start_with_empty_study]
    )
    """If True, starts the study with an empty output directory. Defaults to True."""

    silently_overwrite_study: bool = field(default=Factory(lambda: False), validator=[validators.instance_of(bool)])
    """If False, prompts user for confirmation when overwriting existing study. Defaults to False."""

    path: UPath = field(default=UPath("./studies/"), converter=lambda x: UPath(x))
    """The path where study outputs are saved. Defaults to "./studies/"."""

    ml_type: MLType = field(init=False)
    """The type of machine learning model. Set automatically by subclass."""

    prepared: PreparedData = field(init=False)
    """Container for prepared study data and metadata after data preparation."""

    @property
    @abstractmethod
    def target_metric(self) -> str:
        """Get target metric. Must be implemented in subclasses."""
        ...

    @property
    @abstractmethod
    def metrics(self) -> list:
        """Get metrics list. Must be implemented in subclasses."""
        ...

    @property
    @abstractmethod
    def target_cols(self) -> list[str]:
        """Get target columns as a list. Must be implemented in subclasses."""
        ...

    @property
    @abstractmethod
    def target_assignments(self) -> dict[str, str]:
        """Get target assignments dict. Must be implemented in subclasses."""
        ...

    @property
    def output_path(self) -> UPath:
        """Full output path for this study (path/name)."""
        return self.path / self.name

    @property
    def log_dir(self) -> UPath:
        """Directory where logs are stored."""
        return self.output_path

    @property
    def relevant_columns(self) -> list[str]:
        """Relevant columns for the dataset (computed from prepared data)."""
        relevant_columns = list(set(self.prepared.feature_cols + self.target_cols))
        if self.sample_id_col:
            relevant_columns.append(self.sample_id_col)
        if self.prepared.row_id:
            relevant_columns.append(self.prepared.row_id)
        if self.stratification_column:
            relevant_columns.append(self.stratification_column)
        if "group_features" in self.prepared.data.columns:
            relevant_columns.append("group_features")
        if "group_sample_and_features" in self.prepared.data.columns:
            relevant_columns.append("group_sample_and_features")
        return list(set(relevant_columns))

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate the input data."""
        validator = OctoDataValidator(
            data=data,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            sample_id_col=self.sample_id_col,
            row_id=self.row_id,
            stratification_column=self.stratification_column,
            target_assignments=self.target_assignments,
            ml_type=self.ml_type.value,
            positive_class=getattr(self, "positive_class", None),
        )
        validator.validate()

    def _initialize_study_directory(self) -> None:
        """Initialize study directory."""
        if self.output_path.exists():
            if not self.silently_overwrite_study:
                confirmation = input("Study exists, do you want to continue? (yes/no): ")
                if confirmation.strip().lower() != "yes":
                    print("Exiting...")
                    sys.exit()
                print("Continuing...")

            if self.start_with_empty_study:
                print("Overwriting existing study....")
                self.output_path.rmdir(recursive=True)
            else:
                print("Resume existing study....")

        self.output_path.mkdir(parents=True, exist_ok=True)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        set_logger_filename(log_file=self.log_dir / "octo_manager.log")

    def _initialize_study_outputs(self, data: pd.DataFrame) -> None:
        """Initialize study saving config and data into study directory."""

        def serialize_value(value):
            """Convert a value to JSON-serializable format."""
            if hasattr(value, "value"):
                return value.value
            elif isinstance(value, UPath):
                return str(value)
            elif has(type(value)):
                # Convert to dict using asdict
                result = asdict(value, value_serializer=lambda _, __, v: serialize_value(v))

                # Add ClassVar 'module' field if it exists (for workflow tasks)
                if hasattr(value, "module"):
                    result["module"] = value.module

                return result
            elif isinstance(value, list):
                return [serialize_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            return value

        config = {}
        # Use fields from the actual instance class (including subclass fields)
        for attr in fields(type(self)):
            if attr.name == "prepared":
                continue
            value = getattr(self, attr.name)
            config[attr.name] = serialize_value(value)

        config["prepared"] = {
            "feature_cols": self.prepared.feature_cols,
            "row_id": self.prepared.row_id,
            "target_assignments": self.prepared.target_assignments,
        }

        config_path = self.output_path / "config.json"
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)

        data_path = self.output_path / "data.parquet"
        data.to_parquet(
            str(data_path),
            index=False,
            storage_options=data_path.storage_options,
            engine="pyarrow",
        )
        prepared_data_path = self.output_path / "data_prepared.parquet"
        self.prepared.data.to_parquet(
            str(prepared_data_path),
            index=False,
            storage_options=prepared_data_path.storage_options,
            engine="pyarrow",
        )

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare the data for training."""
        preparator = OctoDataPreparator(
            data=data,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            sample_id_col=self.sample_id_col,
            row_id=self.row_id,
            target_assignments=self.target_assignments,
        )
        prepared = preparator.prepare()
        object.__setattr__(self, "prepared", prepared)

        return prepared.data

    def _create_datasplits(self, data: pd.DataFrame) -> dict:
        """Create datasplits for outer cross-validation."""
        data_clean = data[self.relevant_columns]

        if self.datasplit_type.value == "sample":
            datasplit_col = self.sample_id_col
        else:
            datasplit_col = self.datasplit_type.value

        datasplits: dict = DataSplit(
            dataset=data_clean,
            datasplit_col=datasplit_col,
            seeds=[self.datasplit_seed_outer],
            num_folds=self.n_folds_outer,
            stratification_col=self.stratification_column,
        ).get_datasplits()

        return datasplits

    def _create_experiments(self, datasplits: dict) -> list[OctoExperiment]:
        """Create experiments from datasplits."""
        experiments = []

        # Get datasplit column based on datasplit_type
        if self.datasplit_type.value == "sample":
            datasplit_col = self.sample_id_col
        else:
            datasplit_col = self.datasplit_type.value

        for key, value in datasplits.items():
            experiment: OctoExperiment = OctoExperiment(
                id=str(key),
                experiment_id=int(key),
                task_id=None,  # indicating base experiment
                depends_on_task=None,  # indicating base experiment
                task_path=None,  # indicating base experiment
                study_path=self.path,
                study_name=self.name,
                ml_type=self.ml_type.value,
                target_metric=self.target_metric,
                positive_class=getattr(self, "positive_class", None),
                metrics=self.metrics,
                imputation_method=self.imputation_method.value,
                datasplit_column=datasplit_col,
                row_column=self.prepared.row_id,
                feature_cols=self.prepared.feature_cols,
                target_assignments=self.prepared.target_assignments,
                data_traindev=value["train"],
                data_test=value["test"],
            )
            experiments.append(experiment)

        return experiments

    def _run_health_check(self, data: pd.DataFrame, config: HealthCheckConfig | None) -> None:
        """Run data health check, save results, and check for issues."""
        checker = OctoDataHealthChecker(
            data=data,
            feature_cols=self.prepared.feature_cols,
            target_cols=self.target_cols,
            row_id=self.prepared.row_id,
            sample_id_col=self.sample_id_col,
            stratification_column=self.stratification_column,
            config=config or HealthCheckConfig(),
        )
        report = checker.generate_report()
        report_path = self.output_path / "health_check_report.csv"
        report.to_csv(
            str(report_path),
            index=False,
            storage_options=report_path.storage_options,
        )

        if report.empty:
            return

        has_critical = False
        has_warning = False

        if "severity" in report.columns:
            has_critical = (report["severity"] == "critical").any()
            has_warning = (report["severity"] == "warning").any()

        if has_critical:
            raise ValueError(f"Critical data issues detected. Please check: {report_path}")

        if has_warning and not self.ignore_data_health_warning:
            raise ValueError(
                f"Data issues detected. Please check: {report_path}\nTo proceed despite warnings, set `ignore_data_health_warning=True`."
            )

    def _flush_logger(self):
        """Flush and close all handlers of the logger."""
        for handler in logger.handlers:
            handler.flush()
            handler.close()

        set_logger_filename(log_file=None)

    def fit(
        self,
        data: pd.DataFrame,
        health_check_config: HealthCheckConfig | None = None,
    ) -> None:
        """Fit study to data.

        Args:
            data: DataFrame containing the dataset.
            health_check_config: Optional configuration for health check thresholds.
        """
        self._initialize_study_directory()
        self._validate_data(data)
        prepared_data = self._prepare_data(data)
        self._initialize_study_outputs(data)
        self._run_health_check(prepared_data, health_check_config)

        datasplits = self._create_datasplits(prepared_data)
        experiments = self._create_experiments(datasplits)
        manager = OctoManager(
            base_experiments=experiments,
            workflow=self.workflow,
            outer_parallelization=self.outer_parallelization,
            run_single_experiment_num=self.run_single_experiment_num,
            log_dir=self.log_dir,
        )
        manager.run_outer_experiments()


@define
class OctoRegression(OctoStudy):
    """Regression study."""

    target: str = field(kw_only=True, validator=validators.instance_of(str))
    """The target column to predict."""

    ml_type: MLType = field(default=MLType.REGRESSION, init=False)
    """The type of machine learning model. Automatically set to regression."""

    target_metric: str = field(
        default="RMSE",
        validator=validators.in_(Metrics.get_by_type("regression")),
    )
    """The primary metric used for model evaluation. Defaults to RMSE."""

    metrics: list = field(
        default=Factory(lambda self: [self.target_metric], takes_self=True),
        validator=[
            validators.instance_of(list),
            validators.deep_iterable(
                member_validator=validators.in_(Metrics.get_by_type("regression")),
                iterable_validator=validators.instance_of(list),
            ),
        ],
    )
    """A list of metrics to be calculated. Defaults to target_metric value."""

    @property
    def target_cols(self) -> list[str]:
        """Get target columns as a list."""
        return [self.target]

    @property
    def target_assignments(self) -> dict[str, str]:
        """Get target assignments dict. Empty for single target."""
        return {}


@define
class OctoClassification(OctoStudy):
    """Classification study (binary and multiclass)."""

    target: str = field(kw_only=True, validator=validators.instance_of(str))
    """The target column to predict."""

    ml_type: MLType = field(default=MLType.CLASSIFICATION, init=False)
    """The type of machine learning model. Automatically set to classification (binary or multiclass)."""

    target_metric: str = field(
        default="AUCROC",
        validator=validators.in_(Metrics.get_by_type("classification", "multiclass")),
    )
    """The primary metric used for model evaluation. Defaults to AUCROC."""

    metrics: list = field(
        default=Factory(lambda self: [self.target_metric], takes_self=True),
        validator=[
            validators.instance_of(list),
            validators.deep_iterable(
                member_validator=validators.in_(Metrics.get_by_type("classification", "multiclass")),
                iterable_validator=validators.instance_of(list),
            ),
        ],
    )
    """A list of metrics to be calculated. Defaults to target_metric value."""

    positive_class: int = field(default=1, validator=validators.instance_of(int))
    """The positive class label for binary classification. Defaults to 1. Not used for multiclass."""

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate the input data and determine if binary or multiclass."""
        # Detect if binary or multiclass based on unique values in target
        unique_values = data[self.target].dropna().unique()
        if len(unique_values) > 2:
            object.__setattr__(self, "ml_type", MLType.MULTICLASS)
            object.__setattr__(self, "positive_class", None)
        else:
            object.__setattr__(self, "ml_type", MLType.CLASSIFICATION)

        super()._validate_data(data)

    @property
    def target_cols(self) -> list[str]:
        """Get target columns as a list."""
        return [self.target]

    @property
    def target_assignments(self) -> dict[str, str]:
        """Get target assignments dict. Empty for single target."""
        return {}


@define
class OctoTimeToEvent(OctoStudy):
    """Time-to-event study."""

    duration_column: str = field(kw_only=True, validator=validators.instance_of(str))
    """Column containing time until event or censoring."""

    event_column: str = field(kw_only=True, validator=validators.instance_of(str))
    """Column containing event indicator (0=censored, 1=event)."""

    ml_type: MLType = field(default=MLType.TIMETOEVENT, init=False)
    """The type of machine learning model. Automatically set to time-to-event."""

    target_metric: str = field(
        default="CI",
        validator=validators.in_(Metrics.get_by_type("timetoevent")),
    )
    """The primary metric used for model evaluation. Defaults to CI (Concordance Index)."""

    metrics: list = field(
        default=Factory(lambda self: [self.target_metric], takes_self=True),
        validator=[
            validators.instance_of(list),
            validators.deep_iterable(
                member_validator=validators.in_(Metrics.get_by_type("timetoevent")),
                iterable_validator=validators.instance_of(list),
            ),
        ],
    )
    """A list of metrics to be calculated. Defaults to target_metric value."""

    @property
    def target_cols(self) -> list[str]:
        """Get target columns as a list."""
        return [self.duration_column, self.event_column]

    @property
    def target_assignments(self) -> dict[str, str]:
        """Get target assignments dict."""
        return {"duration": self.duration_column, "event": self.event_column}
