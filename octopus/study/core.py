"""Octo Study."""

import datetime
import json
import os
import platform
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import UTC

import pandas as pd
from attrs import Factory, asdict, define, field, fields, has, validators
from upath import UPath

from octopus.datasplit import DATASPLIT_COL, DataSplit, OuterSplits
from octopus.logger import get_logger, set_logger_filename
from octopus.manager.core import OctoManager
from octopus.metrics import Metrics
from octopus.modules import Octo, StudyContext, Task
from octopus.types import MLType
from octopus.utils import csv_save, get_package_name, get_version, parquet_save

from .data_preparator import OctoDataPreparator
from .data_validator import OctoDataValidator
from .healthChecker import HealthCheckConfig, OctoDataHealthChecker
from .prepared_data import PreparedData
from .validation import validate_workflow

logger = get_logger()

_RUNNING_IN_TESTSUITE = "RUNNING_IN_TESTSUITE" in os.environ


@define
class OctoStudy(ABC):
    """Abstract base class for all Octopus studies."""

    feature_cols: list[str] = field(validator=[validators.instance_of(list)])
    """List of all feature columns in the dataset."""

    sample_id_col: str = field(validator=validators.instance_of(str))
    """Identifier for sample instances."""

    name: str = field(default="Octopus", validator=[validators.instance_of(str)])
    """The name of the study. Defaults to 'Octopus'."""

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

    ignore_data_health_warning: bool = field(default=Factory(lambda: False), validator=[validators.instance_of(bool)])
    """Ignore data health checks warning and run machine learning workflow."""

    outer_parallelization: bool = field(default=Factory(lambda: True), validator=[validators.instance_of(bool)])
    """Indicates whether outer parallelization is enabled. Defaults to True."""

    run_single_outersplit_num: int = field(default=Factory(lambda: -1), validator=[validators.instance_of(int)])
    """Select a single outersplit to execute. Defaults to -1 to run all outersplits"""

    workflow: Sequence[Task] = field(
        default=Factory(lambda: [Octo(task_id=0)]),
        validator=[validators.instance_of(list), validate_workflow],
    )
    """A list of tasks that defines the processing workflow. Each item in the list is an instance of `Task`."""

    path: UPath = field(default=UPath("./studies/"), converter=lambda x: UPath(x))
    """The path where study outputs are saved. Defaults to "./studies/"."""

    ml_type: MLType = field(init=False)
    """The type of machine learning model. Set automatically by subclass."""

    # Time of last fit() call (internal state)
    _fit_timestamp: str | None = field(default=None, init=False)

    @property
    @abstractmethod
    def target_metric(self) -> str:
        """Get target metric. Must be implemented in subclasses."""
        ...

    @property
    @abstractmethod
    def target_assignments(self) -> dict[str, str]:
        """Get target assignments dict. Must be implemented in subclasses."""
        ...

    @property
    def output_path(self) -> UPath:
        """Full output path for this study (path/name-timestamp)."""
        if self._fit_timestamp is None:
            raise RuntimeError("output_path is not available until fit() has been called.")
        fit_dt = datetime.datetime.fromisoformat(self._fit_timestamp)
        folder_name = f"{self.name}-{fit_dt.strftime('%Y%m%d_%H%M%S')}"
        return self.path / folder_name

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
            raise FileExistsError(f"Study output folder already exists: {self.output_path}")
        self.output_path.mkdir(parents=True, exist_ok=False)
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
        assert self._fit_timestamp is not None  # Set at start of fit()
        study_meta = {
            "octopus_version": get_version(),
            "package_name": get_package_name(),
            "python_version": platform.python_version(),
            "created_at": self._fit_timestamp,
        }
        meta_path = self.output_path / "study_meta.json"
        with meta_path.open("w") as f:
            json.dump(study_meta, f, indent=2)

        data_path = self.output_path / "data_raw.parquet"
        parquet_save(data, data_path, index=False)
        prepared_data_path = self.output_path / "data_prepared.parquet"
        parquet_save(prepared.data, prepared_data_path, index=False)

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
        csv_save(report, report_path, index=False)

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
        if self._fit_timestamp is not None:
            raise RuntimeError("fit() can only be called once per study instance.")

        # Generate single timestamp for this fit() call
        self._fit_timestamp = datetime.datetime.now(UTC).isoformat()

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
        logger.info("Study completed. Results saved to: %s", self.output_path)


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

    @property
    def target_assignments(self) -> dict[str, str]:
        """Get target assignments dict."""
        return {"duration": self.duration_col, "event": self.event_col}
