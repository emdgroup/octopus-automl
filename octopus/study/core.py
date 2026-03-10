"""Octo Study."""

import json
import os
import platform
import sys
from abc import ABC, abstractmethod
from datetime import UTC, datetime

import pandas as pd
from attrs import Factory, asdict, define, field, fields, has, validators
from upath import UPath

from octopus.datasplit import DATASPLIT_COL, DataSplit, OuterSplits
from octopus.logger import get_logger, set_logger_filename
from octopus.manager.core import OctoManager
from octopus.metrics import Metrics
from octopus.modules import Octo, Task
from octopus.types import MLType
from octopus.utils import get_package_name, get_version

from .context import StudyContext
from .data_preparator import OctoDataPreparator
from .data_validator import OctoDataValidator
from .healthChecker import HealthCheckConfig, OctoDataHealthChecker
from .prepared_data import PreparedData
from .types import ImputationMethod
from .validation import validate_workflow

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

    row_id_col: str | None = field(
        default=Factory(lambda: None),
        validator=validators.optional(validators.instance_of(str)),
    )
    """Unique row identifier."""

    stratification_col: str | None = field(
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

    run_single_outersplit_num: int = field(default=Factory(lambda: -1), validator=[validators.instance_of(int)])
    """Select a single outersplit to execute. Defaults to -1 to run all outersplits"""

    workflow: list[Task] = field(
        default=Factory(lambda: [Octo(task_id=0)]),
        validator=[validators.instance_of(list), validate_workflow],
    )
    """A list of tasks that defines the processing workflow. Each item in the list is an instance of `Task`."""

    silently_overwrite_study: bool = field(default=Factory(lambda: False), validator=[validators.instance_of(bool)])
    """If False, prompts user for confirmation when overwriting existing study. Defaults to False."""

    path: UPath = field(default=UPath("./studies/"), converter=lambda x: UPath(x))
    """The path where study outputs are saved. Defaults to "./studies/"."""

    ml_type: MLType = field(init=False)
    """The type of machine learning model. Set automatically by subclass."""

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

    def _resolve_ml_config(self, data: pd.DataFrame) -> tuple[MLType, int | None]:
        """Resolve ml_type and positive_class. Subclasses override for auto-detection."""
        return self.ml_type, getattr(self, "positive_class", None)

    def _validate_data(self, data: pd.DataFrame, ml_type: MLType, positive_class: int | None) -> None:
        """Validate the input data."""
        validator = OctoDataValidator(
            data=data,
            feature_cols=self.feature_cols,
            target_col=self.target_col if hasattr(self, "target_col") else None,
            duration_col=self.duration_col if hasattr(self, "duration_col") else None,
            event_col=self.event_col if hasattr(self, "event_col") else None,
            sample_id_col=self.sample_id_col,
            row_id_col=self.row_id_col,
            stratification_col=self.stratification_col,
            ml_type=ml_type,
            positive_class=positive_class,
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

            print("Overwriting existing study....")
            self.output_path.rmdir(recursive=True)

        self.output_path.mkdir(parents=True, exist_ok=True)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        set_logger_filename(log_file=self.log_dir / "study.log")

    def _initialize_study_outputs(
        self,
        data: pd.DataFrame,
        prepared: PreparedData,
        ml_type: MLType,
        positive_class: int | None,
    ) -> None:
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
            value = getattr(self, attr.name)
            config[attr.name] = serialize_value(value)

        # Override with resolved values (e.g. auto-detected ml_type for OctoClassification)
        config["ml_type"] = ml_type.value
        if "positive_class" in config:
            config["positive_class"] = positive_class
        config["prepared"] = {
            "feature_cols": prepared.feature_cols,
            "row_id": prepared.row_id_col,
            "target_assignments": self.target_assignments,
        }

        config_path = self.output_path / "study_config.json"
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)

        # Write study metadata (version, platform, timestamp)
        study_meta = {
            "octopus_version": get_version(),
            "package_name": get_package_name(),
            "python_version": platform.python_version(),
            "created_at": datetime.now(UTC).isoformat(),
        }
        meta_path = self.output_path / "study_meta.json"
        with meta_path.open("w") as f:
            json.dump(study_meta, f, indent=2)

        data_path = self.output_path / "data_raw.parquet"
        data.to_parquet(
            str(data_path),
            index=False,
            storage_options=data_path.storage_options,
            engine="pyarrow",
        )
        prepared_data_path = self.output_path / "data_prepared.parquet"
        prepared.data.to_parquet(
            str(prepared_data_path),
            index=False,
            storage_options=prepared_data_path.storage_options,
            engine="pyarrow",
        )

    def _prepare_data(self, data: pd.DataFrame) -> PreparedData:
        """Prepare the data for training."""
        preparator = OctoDataPreparator(
            data=data,
            feature_cols=self.feature_cols,
            sample_id_col=self.sample_id_col,
            row_id_col=self.row_id_col,
            target_col=self.target_col if hasattr(self, "target_col") else None,
            stratification_col=self.stratification_col,
            duration_col=self.duration_col if hasattr(self, "duration_col") else None,
            event_col=self.event_col if hasattr(self, "event_col") else None,
        )
        return preparator.prepare()

    def _create_datasplits(self, prepared: PreparedData) -> OuterSplits:
        """Create datasplits for outer cross-validation."""
        relevant_cols = list(prepared.feature_cols) + [
            c
            for c in (
                self.target_col if hasattr(self, "target_col") else None,
                self.duration_col if hasattr(self, "duration_col") else None,
                self.event_col if hasattr(self, "event_col") else None,
                self.sample_id_col,
                prepared.row_id_col,
                self.stratification_col,
            )
            if c is not None
        ]

        relevant_cols += [DATASPLIT_COL]

        relevant_cols = list(dict.fromkeys(relevant_cols))
        data_clean = prepared.data[relevant_cols]

        outersplits = DataSplit(
            dataset=data_clean,
            seeds=[self.datasplit_seed_outer],
            num_folds=self.n_folds_outer,
            stratification_col=self.stratification_col,
        ).get_outer_splits()

        return outersplits

    def _run_health_check(self, prepared: PreparedData, config: HealthCheckConfig | None) -> None:
        """Run data health check, save results, and check for issues."""
        checker = OctoDataHealthChecker(
            data=prepared.data,
            feature_cols=prepared.feature_cols,
            target_col=self.target_col if hasattr(self, "target_col") else None,
            duration_col=self.duration_col if hasattr(self, "duration_col") else None,
            event_col=self.event_col if hasattr(self, "event_col") else None,
            row_id_col=prepared.row_id_col,
            sample_id_col=self.sample_id_col,
            stratification_col=self.stratification_col,
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

        if "Severity" in report.columns:
            has_critical = (report["Severity"] == "Critical").any()
            has_warning = (report["Severity"] == "Warning").any()

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

    def _create_study_context(
        self, prepared: PreparedData, ml_type: MLType, positive_class: int | None
    ) -> StudyContext:
        """Create a frozen StudyContext from the current study state."""
        return StudyContext(
            ml_type=ml_type,
            target_metric=self.target_metric,
            metrics=self.metrics,
            target_assignments=self.target_assignments,
            positive_class=positive_class,
            stratification_col=self.stratification_col,
            sample_id_col=self.sample_id_col,
            feature_cols=prepared.feature_cols,
            row_id_col=prepared.row_id_col,
            output_path=self.output_path,
            log_dir=self.log_dir,
        )

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
        ml_type, positive_class = self._resolve_ml_config(data)
        self._validate_data(data, ml_type, positive_class)
        prepared = self._prepare_data(data)
        self._initialize_study_outputs(data, prepared, ml_type, positive_class)
        self._run_health_check(prepared, health_check_config)

        outersplit_data = self._create_datasplits(prepared)
        study_context = self._create_study_context(prepared, ml_type, positive_class)
        manager = OctoManager(
            outersplit_data=outersplit_data,
            study_context=study_context,
            workflow=self.workflow,
            outer_parallelization=self.outer_parallelization,
            run_single_outersplit_num=self.run_single_outersplit_num,
        )
        manager.run_outersplits()


@define
class OctoRegression(OctoStudy):
    """Regression study."""

    target_col: str = field(kw_only=True, validator=validators.instance_of(str))
    """The target column to predict."""

    ml_type: MLType = field(default=MLType.REGRESSION, init=False)
    """The type of machine learning model. Automatically set to regression."""

    target_metric: str = field(
        default="RMSE",
        validator=validators.in_(Metrics.get_by_type(MLType.REGRESSION)),
    )
    """The primary metric used for model evaluation. Defaults to RMSE."""

    metrics: list = field(
        default=Factory(lambda self: [self.target_metric], takes_self=True),
        validator=[
            validators.instance_of(list),
            validators.deep_iterable(
                member_validator=validators.in_(Metrics.get_by_type(MLType.REGRESSION)),
                iterable_validator=validators.instance_of(list),
            ),
        ],
    )
    """A list of metrics to be calculated. Defaults to target_metric value."""

    @property
    def target_assignments(self) -> dict[str, str]:
        """Get target assignments dict."""
        return {"default": self.target_col}


@define
class OctoClassification(OctoStudy):
    """Classification study (binary and multiclass)."""

    target_col: str = field(kw_only=True, validator=validators.instance_of(str))
    """The target column to predict."""

    ml_type: MLType | None = field(  # type: ignore[assignment]
        default=None,
        kw_only=True,
        validator=validators.optional(validators.in_([MLType.BINARY, MLType.MULTICLASS])),
    )
    """The type of machine learning model. Can be set explicitly or auto-detected from data (binary vs multiclass)."""

    target_metric: str = field(
        default="AUCROC",
        validator=validators.in_(Metrics.get_by_type(MLType.BINARY, MLType.MULTICLASS)),
    )
    """The primary metric used for model evaluation. Defaults to AUCROC."""

    metrics: list = field(
        default=Factory(lambda self: [self.target_metric], takes_self=True),
        validator=[
            validators.instance_of(list),
            validators.deep_iterable(
                member_validator=validators.in_(Metrics.get_by_type(MLType.BINARY, MLType.MULTICLASS)),
                iterable_validator=validators.instance_of(list),
            ),
        ],
    )
    """A list of metrics to be calculated. Defaults to target_metric value."""

    positive_class: int | None = field(default=None, validator=validators.optional(validators.instance_of(int)))
    """The positive class label for binary classification. Defaults to None. Not used for multiclass."""

    def _resolve_ml_config(self, data: pd.DataFrame) -> tuple[MLType, int | None]:
        if self.target_col not in data.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in input data.")
        ml_type = self.ml_type
        positive_class = self.positive_class
        if not ml_type:
            unique_values = data[self.target_col].dropna().unique()
            if len(unique_values) > 2:
                ml_type, positive_class = MLType.MULTICLASS, None
            else:
                ml_type, positive_class = MLType.BINARY, 1
        if ml_type == MLType.BINARY and positive_class is None:
            raise ValueError("For binary classification, `positive_class` must be specified.")
        return ml_type, positive_class

    def _validate_data(self, data: pd.DataFrame, ml_type: MLType, positive_class: int | None) -> None:
        if self.target_col not in data.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in input data.")
        super()._validate_data(data, ml_type, positive_class)

    @property
    def target_assignments(self) -> dict[str, str]:
        """Get target assignments dict."""
        return {"default": self.target_col}


@define
class OctoTimeToEvent(OctoStudy):
    """Time-to-event study."""

    duration_col: str = field(kw_only=True, validator=validators.instance_of(str))
    """Column containing time until event or censoring."""

    event_col: str = field(kw_only=True, validator=validators.instance_of(str))
    """Column containing event indicator (0=censored, 1=event)."""

    ml_type: MLType = field(default=MLType.TIMETOEVENT, init=False)
    """The type of machine learning model. Automatically set to time-to-event."""

    target_metric: str = field(
        default="CI",
        validator=validators.in_(Metrics.get_by_type(MLType.TIMETOEVENT)),
    )
    """The primary metric used for model evaluation. Defaults to CI (Concordance Index)."""

    metrics: list = field(
        default=Factory(lambda self: [self.target_metric], takes_self=True),
        validator=[
            validators.instance_of(list),
            validators.deep_iterable(
                member_validator=validators.in_(Metrics.get_by_type(MLType.TIMETOEVENT)),
                iterable_validator=validators.instance_of(list),
            ),
        ],
    )
    """A list of metrics to be calculated. Defaults to target_metric value."""

    @property
    def target_assignments(self) -> dict[str, str]:
        """Get target assignments dict."""
        return {"duration": self.duration_col, "event": self.event_col}
