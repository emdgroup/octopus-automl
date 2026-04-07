"""Octo Training."""

import copy
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from attrs import Factory, define, field, validators
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted

from octopus.logger import get_logger
from octopus.metrics.utils import get_score_from_model
from octopus.models import Models
from octopus.types import DataPartition, FIComputeMethod, FIResultLabel, LogGroup, MLType, ModelName

logger = get_logger()

# ── Octo-only Layer 1 primitive ────────────────────────────────


def _compute_internal_fi(
    model: Any,
    feature_names: list[str],
    ml_type: MLType | None = None,
) -> pd.DataFrame:
    """Extract internal feature importance from a fitted model.

    Handles tree-based models (``feature_importances_``), linear models
    (``abs(coef_)``), and returns an empty DataFrame for unsupported models
    that expose neither attribute.  Time-to-event wrappers (e.g.
    ``CatBoostCoxSurvival``, ``XGBoostCoxSurvival``) that expose
    ``feature_importances_`` are supported and will return importances
    normally.

    This is an octo-only primitive — it is not used by the predict layer
    and therefore lives here rather than in the shared
    ``octopus.feature_importance`` module.

    Args:
        model: A fitted scikit-learn-compatible model.
        feature_names: List of feature column names matching the model.
        ml_type: The ML task type (currently unused, kept for API
            compatibility).

    Returns:
        DataFrame with columns ``["feature", "importance"]``.
    """
    empty = pd.DataFrame(columns=["feature", "importance"])

    # Tree-based models (incl. T2E wrappers like CatBoostCox, XGBoostCox)
    if hasattr(model, "feature_importances_"):
        fi = np.asarray(model.feature_importances_)
        return pd.DataFrame({"feature": feature_names, "importance": fi})

    # Linear models
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 1:
            importance = np.abs(coef)
        else:
            importance = np.mean(np.abs(coef), axis=0)

        if len(importance) != len(feature_names):
            logger.warning(
                "Length mismatch between coefficients (%d) and feature columns (%d). Skipping internal importances.",
                len(importance),
                len(feature_names),
            )
            return empty

        return pd.DataFrame({"feature": feature_names, "importance": importance})

    # Fallback
    logger.warning("Internal feature importances not available for this estimator.")
    return empty


def _compute_lofo(
    model: Any,
    x_train_processed: pd.DataFrame,
    y_train: Any,
    eval_partitions: dict[str, pd.DataFrame],
    feature_cols: list[str],
    feature_groups: dict[str, list[str]],
    target_metric: str,
    target_assignments: dict,
    positive_class: int | None,
    n_targets: int,
) -> dict[str, pd.DataFrame]:
    """Compute LOFO feature importance (octo-only primitive).

    For each feature (and feature group), retrains the model with that
    feature removed and measures the performance drop.

    Args:
        model: Fitted sklearn-compatible model to copy and retrain.
        x_train_processed: Preprocessed training features.
        y_train: Training targets.
        eval_partitions: Dict mapping partition name (e.g. ``"dev"``, ``"test"``)
            to DataFrames containing both features and target columns.
        feature_cols: List of feature column names.
        feature_groups: Dict mapping group name to list of feature names.
        target_metric: Metric name for scoring.
        target_assignments: Target column assignments.
        positive_class: Positive class label for binary classification.
        n_targets: Number of target columns (1 for single-target, >1 for
            multi-target / T2E).

    Returns:
        Dict mapping ``"lofo_{partition}"`` to DataFrame with columns
        ``["feature", "importance"]``.
    """
    # Compute baseline scores per partition
    baselines: dict[str, float] = {}
    for part_name, part_data in eval_partitions.items():
        baselines[part_name] = get_score_from_model(
            model, part_data, feature_cols, target_metric, target_assignments, positive_class=positive_class
        )

    # Build combined feature dict: individual features + groups
    lofo_features: dict[str, list[str]] = {x: [x] for x in feature_cols}
    lofo_features.update(feature_groups)

    # Per-partition importance lists
    fi_results: dict[str, list[tuple[str, float]]] = {part: [] for part in eval_partitions}

    for name, lofo_feature in lofo_features.items():
        selected_features = [x for x in feature_cols if x not in lofo_feature]
        # Single-feature datasets: cannot measure importance of the only feature
        if not selected_features:
            for _part_name, part_list in fi_results.items():
                part_list.append((name, 0.0))
            continue
        retrained_model = copy.deepcopy(model)

        # Retrain model without the left-out feature(s)
        if n_targets == 1:
            retrained_model.fit(x_train_processed[selected_features], y_train.squeeze(axis=1))
        else:
            retrained_model.fit(x_train_processed[selected_features], y_train)

        # Score on each partition
        for part_name, part_data in eval_partitions.items():
            score = get_score_from_model(
                retrained_model,
                part_data,
                selected_features,
                target_metric,
                target_assignments,
                positive_class=positive_class,
            )
            fi_results[part_name].append((name, baselines[part_name] - score))

    return {
        fi_storage_key(FIComputeMethod.LOFO, part): pd.DataFrame(rows, columns=["feature", "importance"])
        for part, rows in fi_results.items()
    }


# ── FI storage key helpers ──────────────────────────────────────


def fi_storage_key(
    method: FIComputeMethod | str,
    partition: DataPartition | str | None = None,
    stat: str | None = None,
) -> str:
    """Build a feature-importance storage key.

    Centralises the key format so that construction and lookup always
    agree.  Keys are plain strings for dict compatibility.

    Examples:
        >>> fi_storage_key(FIComputeMethod.INTERNAL)
        'internal'
        >>> fi_storage_key(FIComputeMethod.PERMUTATION, DataPartition.DEV)
        'permutation_dev'
        >>> fi_storage_key(FIComputeMethod.PERMUTATION, "dev", "mean")
        'permutation_dev_mean'

    Args:
        method: FI computation method (enum or string value).
        partition: Data partition (e.g. ``DataPartition.DEV``, ``"dev"``).
            ``None`` for partition-less methods (internal, constant).
        stat: Aggregation statistic (``"mean"``, ``"count"``).  ``None``
            for per-training keys.

    Returns:
        Underscore-joined storage key string.
    """
    parts = [str(method)]
    if partition is not None:
        parts.append(str(partition))
    if stat is not None:
        parts.append(stat)
    return "_".join(parts)


# Known stat suffixes used in aggregated FI storage keys.
_KNOWN_STAT_SUFFIXES = frozenset({"mean", "count"})


def parse_fi_storage_key(key: str) -> tuple[FIResultLabel, str]:
    """Parse a FI storage key into ``(method_label, dataset)``.

    Reverse of :func:`fi_storage_key` for the subset of keys used in
    ``get_feature_importances_df()``.  Recognises partition-aware methods
    (permutation, shap, lofo) and returns the partition suffix; for
    partition-less methods (internal, constant) returns ``"train"`` as the
    dataset label.

    Correctly handles three-part keys (e.g. ``"permutation_dev_mean"``)
    by stripping the trailing stat suffix before extracting the partition.

    Args:
        key: A storage key such as ``"permutation_dev"``,
            ``"permutation_dev_mean"``, or ``"internal"``.

    Returns:
        Tuple of ``(FIResultLabel, dataset_str)``.
    """
    known_methods = {FIComputeMethod.PERMUTATION, FIComputeMethod.SHAP, FIComputeMethod.LOFO}
    for method in known_methods:
        prefix = str(method) + "_"
        if key.startswith(prefix):
            remainder = key[len(prefix) :]
            # Strip trailing stat suffix (e.g. "_mean", "_count") if present
            for suffix in _KNOWN_STAT_SUFFIXES:
                if remainder.endswith("_" + suffix):
                    remainder = remainder[: -(len(suffix) + 1)]
                    break
            return FIResultLabel(str(method)), remainder
    # Keys without partition suffix: "internal", "constant"
    return FIResultLabel(key), "train"


# # TODO pipeline
# - implement cat encoding on module level
# - how to provide categorical info to catboost and other models?


class TrainingConfig(TypedDict, total=False):
    """Training configuration type."""

    outl_reduction: int
    n_input_features: int
    ml_model_type: ModelName
    ml_model_params: dict
    positive_class: int | None


@define
class Training:
    """Model Training Class."""

    training_id: str = field(validator=[validators.instance_of(str)])
    """Training id."""

    ml_type: MLType = field(validator=[validators.instance_of(MLType)])
    """ML-type."""

    target_assignments: dict = field(validator=[validators.instance_of(dict)])
    """Target assignments."""

    feature_cols: list[str] = field(validator=[validators.instance_of(list)])
    """Feature columns."""

    row_id_col: str = field(validator=[validators.instance_of(str)])
    """Row column name."""

    data_train: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    "Data train."

    data_dev: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    "Data dev."

    data_test: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    """Data test."""

    target_metric: str = field(validator=[validators.instance_of(str)])
    """Target metric."""

    max_features: int = field(validator=[validators.instance_of(int)])
    """Maximum number of features."""

    feature_groups: dict[str, list[str]] = field(validator=[validators.instance_of(dict)])
    """Feature Groups."""

    config_training: TrainingConfig = field()
    """Training configuration."""

    training_weight: int = field(default=1, validator=[validators.instance_of(int)])
    """Training weight for ensembling"""

    model: Any = field(default=None)
    """Model."""

    predictions: dict = field(default=Factory(dict), validator=[validators.instance_of(dict)])
    """Model predictions."""

    feature_importances: dict[str, pd.DataFrame] = field(
        default=Factory(dict), validator=[validators.instance_of(dict)]
    )
    """Feature importances, mapping from FI method to DataFrame."""

    features_used: list = field(default=Factory(list), validator=[validators.instance_of(list)])
    """Features used."""

    outlier_samples: list = field(default=Factory(list), validator=[validators.instance_of(list)])
    """Outlier samples identified."""

    is_fitted: bool = field(default=False, init=False)
    """Flag indicating whether the training has been completed."""

    preprocessing_pipeline: ColumnTransformer | Pipeline = field(init=False)
    """Preprocessing pipeline for data scaling, imputation, and categorical encoding."""

    x_train_processed: pd.DataFrame | None = field(default=None, init=False)
    """Training data after pre-processing (outlier, imputation, scaling)."""

    @property
    def outl_reduction(self) -> int:
        """Parameter outlier reduction method."""
        return self.config_training["outl_reduction"]

    @property
    def ml_model_type(self) -> ModelName:
        """ML model type."""
        return self.config_training["ml_model_type"]

    @property
    def ml_model_params(self) -> dict:
        """ML model parameters."""
        return self.config_training["ml_model_params"]

    @property
    def x_train(self):
        """x_train."""
        return self.data_train[self.feature_cols]

    @property
    def x_dev_processed(self):
        """x_dev_processed."""
        return self._transform_to_dataframe(self.data_dev[self.feature_cols], index=self.data_dev.index)

    @property
    def x_test_processed(self):
        """x_test_processed."""
        return self._transform_to_dataframe(self.data_test[self.feature_cols], index=self.data_test.index)

    @property
    def y_train(self):
        """y_train."""
        if self.ml_type == MLType.TIMETOEVENT:
            duration = self.data_train[self.target_assignments["duration"]]
            event = self.data_train[self.target_assignments["event"]]
            return np.array(
                list(zip(event, duration, strict=False)),
                dtype={"names": ("c1", "c2"), "formats": ("bool", "f8")},
            )
        else:
            return self.data_train[self.target_assignments.values()]

    @property
    def y_dev(self):
        """y_dev."""
        if self.ml_type == MLType.TIMETOEVENT:
            duration = self.data_dev[self.target_assignments["duration"]]
            event = self.data_dev[self.target_assignments["event"]]
            return np.array(
                list(zip(event, duration, strict=False)),
                dtype={"names": ("c1", "c2"), "formats": ("bool", "f8")},
            )
        else:
            return self.data_dev[self.target_assignments.values()]

    @property
    def y_test(self):
        """y_test."""
        if self.ml_type == MLType.TIMETOEVENT:
            duration = self.data_test[self.target_assignments["duration"]]
            event = self.data_test[self.target_assignments["event"]]
            return np.array(
                list(zip(event, duration, strict=False)),
                dtype={"names": ("c1", "c2"), "formats": ("bool", "f8")},
            )
        else:
            return self.data_test[self.target_assignments.values()]

    def __attrs_post_init__(self):
        # Set up preprocessing pipeline
        self._setup_preprocessing_pipeline()

    def _relabel_processed_output(
        self,
        processed_data: Any,
        index: pd.Index | None = None,
    ) -> pd.DataFrame:
        """Convert pipeline output to a correctly-labeled DataFrame in self.feature_cols order.

        Handles the ColumnTransformer column reordering issue: ColumnTransformer outputs columns
        in transformer order (numerical first, then categorical), which may differ from
        self.feature_cols order. This method uses get_feature_names_out() to correctly label
        columns, then reorders to self.feature_cols order.

        Args:
            processed_data: Raw output from preprocessing_pipeline.transform() or fit_transform().
            index: Optional index for the output DataFrame.

        Returns:
            DataFrame with columns in self.feature_cols order, correctly labeled.
        """
        # Convert sparse matrices to dense arrays
        if hasattr(processed_data, "toarray"):
            processed_data = processed_data.toarray()

        if not (hasattr(processed_data, "shape") and len(processed_data.shape) == 2):
            return pd.DataFrame(processed_data)

        try:
            output_cols = list(self.preprocessing_pipeline.get_feature_names_out())
        except AttributeError:
            # FunctionTransformer pipeline doesn't support get_feature_names_out()
            # In this case, column order is preserved (no ColumnTransformer reordering)
            output_cols = list(self.feature_cols)

        n_cols = processed_data.shape[1]
        if set(output_cols) != set(self.feature_cols):
            # If column count also mismatches, raise a clear error
            if n_cols != len(self.feature_cols):
                raise ValueError(
                    f"Pipeline output has {n_cols} columns but expected {len(self.feature_cols)}. "
                    f"Pipeline columns: {output_cols}, expected: {list(self.feature_cols)}. "
                    f"This may indicate extra/unexpected columns were passed to the transformer."
                )
            logger.warning(
                "Pipeline output columns %s do not match feature_cols %s. Falling back to positional labeling.",
                output_cols,
                self.feature_cols,
            )
            output_cols = list(self.feature_cols)

        df = pd.DataFrame(processed_data, columns=output_cols, index=index)

        # Reorder to self.feature_cols order if needed
        if output_cols != list(self.feature_cols):
            df = df[self.feature_cols]

        return df

    def _transform_to_dataframe(
        self,
        data: pd.DataFrame | np.ndarray,
        index: pd.Index | None = None,
    ) -> pd.DataFrame:
        """Transform data through preprocessing pipeline and return correctly-labeled DataFrame.

        Args:
            data: Input data to transform.
            index: Optional index for the output DataFrame.

        Returns:
            DataFrame with columns in self.feature_cols order, correctly labeled.
        """
        processed_data = self.preprocessing_pipeline.transform(data)
        return self._relabel_processed_output(processed_data, index=index)

    def _setup_preprocessing_pipeline(self):
        """Set up the preprocessing pipeline with conditional imputation and scaling.

        Pipeline handles:
        - Imputation: Only applied if model's imputation_required attribute is True
        - Scaling: Only applied if model's scaler attribute is not None

        Note: Categorical encoding (one-hot, ordinal) is handled elsewhere in the pipeline.
        """
        # Get model configuration
        model_config = Models.get_config(self.ml_model_type)

        # Identify column types
        sample_data = self.data_train[self.feature_cols]
        numerical_columns = [
            col for col in self.feature_cols if sample_data[col].dtype not in ["object", "category", "bool"]
        ]
        categorical_columns = [
            col for col in self.feature_cols if sample_data[col].dtype in ["object", "category", "bool"]
        ]

        # Build transformers
        transformers = []

        # Numerical columns transformer
        if numerical_columns:
            steps: list[tuple[str, SimpleImputer | StandardScaler]] = []
            if model_config.imputation_required:
                steps.append(("imputer", SimpleImputer(strategy="median")))
            if model_config.scaler == "StandardScaler":
                steps.append(("scaler", StandardScaler()))
            elif model_config.scaler is not None:
                raise ValueError(f"Unsupported scaler type: {model_config.scaler}")

            transformer = Pipeline(steps) if steps else "passthrough"
            transformers.append(("num", transformer, numerical_columns))

        # Categorical columns transformer
        if categorical_columns:
            transformer = (
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
                if model_config.imputation_required
                else "passthrough"
            )
            transformers.append(("cat", transformer, categorical_columns))

        # Create final pipeline
        if transformers:
            self.preprocessing_pipeline = ColumnTransformer(
                transformers=transformers, remainder="passthrough", verbose_feature_names_out=False
            )
        else:
            self.preprocessing_pipeline = Pipeline([("passthrough", FunctionTransformer())])

    # Training class functionality:
    # (1) outlier removal
    # (2) preprocessing pipeline
    # (3) model training
    # (4) model predictions
    # (5) calculate feature importance, on request

    def fit(self) -> "Training":
        """Preprocess and fit model."""
        # use copy of all train variables, as they may be change due to outlier detec.
        data_train = self.data_train.copy()
        x_train = self.x_train.copy()
        y_train = self.y_train.copy()

        # (1) outlier removal in x_train
        if self.outl_reduction > 0:
            # IsolationForest for outlier detection
            clf = IsolationForest(
                contamination=self.outl_reduction / len(x_train),
                random_state=42,
                n_jobs=1,
            )
            clf.fit(x_train)

            # Get the outlier prediction labels
            # (-1:outliers, 1:inliers)
            outlier_pred = clf.predict(x_train)
            # sometimes there seems to be a mismatch in the number of outliers
            # assert self.outl_reduction == np.sum(outlier_pred == -1)
            # print("Number of outliers specified:", self.outl_reduction)
            # print("Number of outliers found:", np.sum(outlier_pred == -1))

            # identify outlier samples
            self.outlier_samples = data_train[outlier_pred == -1][self.row_id_col].tolist()
            # print("Outlier samples:", self.outlier_samples)

            # Remove outliers from data_train, x_train, y_train
            data_train = data_train[outlier_pred == 1].copy()
            x_train = x_train[outlier_pred == 1].copy()
            y_train = y_train[outlier_pred == 1].copy()

        # (2) Imputation and scaling (after outlier removal)
        processed_data = self.preprocessing_pipeline.fit_transform(x_train)
        self.x_train_processed = self._relabel_processed_output(processed_data, index=x_train.index)

        # (3) Model training
        self.model = Models.get_instance(self.ml_model_type, self.ml_model_params)

        try:
            if len(self.target_assignments) == 1:
                # standard sklearn single target models
                self.model.fit(
                    self.x_train_processed,
                    y_train.squeeze(axis=1),
                )
            else:
                # multi target models, incl. time2event
                self.model.fit(self.x_train_processed, y_train)

            # Validate that the model actually trained by checking model attributes (cheap validation)
            self._validate_model_trained()

        except Exception as e:
            logger.error(f"Model training failed for {self.ml_model_type} in training {self.training_id}: {e}")
            raise RuntimeError(f"Model training failed for {self.ml_model_type}: {e}") from e

        # (4) Model prediction
        # Parse training_id to get split metadata: "outer_split_task_inner_split" (e.g., "1_0_2")
        parts = self.training_id.split("_")
        try:
            outer_split_id = int(parts[0])
            task_id = int(parts[1])
            inner_split_id = str(int(parts[2]))
        except (ValueError, IndexError):
            outer_split_id = 0
            task_id = 0
            inner_split_id = self.training_id

        self.predictions["train"] = pd.DataFrame()
        self.predictions["train"][self.row_id_col] = data_train[self.row_id_col]
        self.predictions["train"]["prediction"] = self.model.predict(self.x_train_processed)
        self.predictions["train"]["outer_split_id"] = outer_split_id
        self.predictions["train"]["inner_split_id"] = inner_split_id
        self.predictions["train"]["partition"] = "train"
        self.predictions["train"]["task_id"] = task_id

        self.predictions["dev"] = pd.DataFrame()
        self.predictions["dev"][self.row_id_col] = self.data_dev[self.row_id_col]
        self.predictions["dev"]["prediction"] = self.model.predict(self.x_dev_processed)
        self.predictions["dev"]["outer_split_id"] = outer_split_id
        self.predictions["dev"]["inner_split_id"] = inner_split_id
        self.predictions["dev"]["partition"] = "dev"
        self.predictions["dev"]["task_id"] = task_id

        self.predictions["test"] = pd.DataFrame()
        self.predictions["test"][self.row_id_col] = self.data_test[self.row_id_col]
        self.predictions["test"]["prediction"] = self.model.predict(self.x_test_processed)
        self.predictions["test"]["outer_split_id"] = outer_split_id
        self.predictions["test"]["inner_split_id"] = inner_split_id
        self.predictions["test"]["partition"] = "test"
        self.predictions["test"]["task_id"] = task_id

        # special treatment of targets due to sklearn
        if len(self.target_assignments) == 1:
            target_col = list(self.target_assignments.values())[0]
            self.predictions["train"][target_col] = y_train.squeeze(axis=1)
            self.predictions["dev"][target_col] = self.y_dev.squeeze(axis=1)
            self.predictions["test"][target_col] = self.y_test.squeeze(axis=1)
        else:
            for target_col in self.target_assignments.values():
                self.predictions["train"][target_col] = data_train[target_col]
                self.predictions["dev"][target_col] = self.data_dev[target_col]
                self.predictions["test"][target_col] = self.data_test[target_col]

        # add additional predictions for classifications (binary and multiclass)
        if self.ml_type in (MLType.BINARY, MLType.MULTICLASS):
            columns = [int(x) for x in self.model.classes_]  # column names --> int
            self.predictions["train"][columns] = self.model.predict_proba(self.x_train_processed)
            self.predictions["dev"][columns] = self.model.predict_proba(self.x_dev_processed)
            self.predictions["test"][columns] = self.model.predict_proba(self.x_test_processed)

        # add additional predictions for time to event predictions
        if self.ml_type == MLType.TIMETOEVENT:
            pass

        # calculate used features, but only if required for optuna max_features>0
        # (to save time, shap or permutation importances may take a lot of time)
        if self.max_features > 0:
            self.features_used = self._calculate_features_used()
        else:
            self.features_used = []

        # Set fitted flag to True
        self.is_fitted = True

        return self

    def _calculate_features_used(self):
        """Calculate used features, method based on model type."""
        feature_method = Models.get_config(self.ml_model_type).feature_method

        if feature_method == FIComputeMethod.INTERNAL:
            self._calculate_fi_internal()
            fi_df = self.feature_importances[fi_storage_key(FIComputeMethod.INTERNAL)]
        elif feature_method == FIComputeMethod.SHAP:
            self._calculate_fi_featuresused_shap(partition="dev")
            fi_df = self.feature_importances[fi_storage_key(FIComputeMethod.SHAP, "dev")]
        elif feature_method == FIComputeMethod.PERMUTATION:
            self._calculate_fi_permutation(partition="dev", n_repeats=2, use_groups=False)  # only 2 repeats!
            fi_df = self.feature_importances[fi_storage_key(FIComputeMethod.PERMUTATION, "dev")]
        elif feature_method == FIComputeMethod.CONSTANT:
            self._calculate_fi_constant()
            fi_df = self.feature_importances[fi_storage_key(FIComputeMethod.CONSTANT)]
        else:
            raise ValueError("feature method provided in model config not supported")

        # Check if feature importance calculation failed (empty DataFrame)
        if fi_df.empty:
            logger.warning(
                f"Feature importance calculation failed for model {self.ml_model_type} using method {feature_method}. Returning all features as used."
            )
            return self.feature_cols.copy()

        features_used = fi_df[fi_df["importance"] != 0]["feature"].tolist()

        # If no features have non-zero importance, return all features as a fallback
        if not features_used:
            logger.warning(
                f"All feature importances are zero for model {self.ml_model_type} using method {feature_method}. Returning all features as used."
            )
            return self.feature_cols.copy()

        return features_used

    def calculate_fi(
        self,
        fi_type: FIComputeMethod,
        partition: DataPartition | str = DataPartition.DEV,
        **kwargs: Any,
    ) -> None:
        """Calculate feature importance of the specified type.

        Single entry point for all FI calculations. Dispatches to the
        appropriate method and stores results in
        ``self.feature_importances``.

        Args:
            fi_type: The feature importance method to use.
            partition: Data partition to evaluate on (e.g. ``DataPartition.DEV``).
                Accepts both ``DataPartition`` enum and plain strings for
                backward compatibility.
                Ignored for methods that don't use partitions (internal, constant, lofo).
            **kwargs: Additional keyword arguments passed to the underlying
                method (e.g. ``n_repeats`` for permutation, ``shap_type`` for SHAP).
        """
        partition = str(partition)
        if fi_type == FIComputeMethod.INTERNAL:
            self._calculate_fi_internal()
        elif fi_type == FIComputeMethod.PERMUTATION:
            self._calculate_fi_permutation(partition=partition, **kwargs)
        elif fi_type == FIComputeMethod.SHAP:
            self._calculate_fi_shap(partition=partition, **kwargs)
        elif fi_type == FIComputeMethod.LOFO:
            self._calculate_fi_lofo()
        elif fi_type == FIComputeMethod.CONSTANT:
            self._calculate_fi_constant()
        else:
            raise ValueError(f"Feature importance method {fi_type} not supported.")

    def _calculate_fi_constant(self):
        """Provide flat feature importance table."""
        fi_df = pd.DataFrame()
        fi_df["feature"] = self.feature_cols
        fi_df["importance"] = 1
        self.feature_importances[fi_storage_key(FIComputeMethod.CONSTANT)] = fi_df

    def _calculate_fi_internal(self):
        """Sklearn-provided internal feature importance (based on train dataset).

        Delegates to ``_compute_internal_fi``, an octo-only Layer 1 primitive
        defined in this module.
        """
        fi_df = _compute_internal_fi(
            model=self.model,
            feature_names=self.feature_cols,
            ml_type=self.ml_type,
        )
        self.feature_importances[fi_storage_key(FIComputeMethod.INTERNAL)] = fi_df

    def _calculate_fi_permutation(
        self,
        partition: DataPartition | str = DataPartition.DEV,
        n_repeats: int = 10,
        use_groups: bool = True,
    ):
        """Permutation feature importance.

        Delegates to ``compute_permutation_single`` from shared computation
        functions, using the custom draw-from-pool algorithm.

        Args:
            partition: Which partition to evaluate on (e.g. ``DataPartition.DEV``).
            n_repeats: Number of permutation repeats per feature.
            use_groups: If ``True`` (default), include ``self.feature_groups`` so
                that importance is computed for both individual features and
                feature groups.  If ``False``, compute only individual feature
                importances (groups are ignored).
        """
        from octopus.feature_importance import compute_permutation_single  # noqa: PLC0415

        logger.set_log_group(LogGroup.TRAINING, f"{self.training_id}")
        logger.info(f"Calculating permutation feature importances ({partition}). This may take a while...")

        target_cols = list(self.target_assignments.values())
        if partition == DataPartition.DEV:
            eval_data = pd.concat([self.x_dev_processed, self.data_dev[target_cols]], axis=1)
        elif partition == DataPartition.TEST:
            eval_data = pd.concat([self.x_test_processed, self.data_test[target_cols]], axis=1)
        else:
            raise ValueError(f"Unsupported partition: {partition!r}")

        if not set(self.feature_cols).issubset(eval_data.columns):
            raise ValueError("Features missing in provided dataset.")

        # Build training pool for draw-from-pool permutation.
        # Use x_train_processed (larger, more representative) as the sampling pool.
        # Align targets to x_train_processed index to handle outl_reduction > 0,
        # where self.data_train retains all rows but x_train_processed has outliers removed.
        if self.x_train_processed is None:
            raise RuntimeError("Model must be fitted before computing permutation FI.")
        train_targets = self.data_train.loc[self.x_train_processed.index, target_cols]
        train_pool = pd.concat([self.x_train_processed, train_targets], axis=1)

        feature_groups = self.feature_groups if use_groups else None

        results_df = compute_permutation_single(
            model=self.model,
            X_test=eval_data,
            X_train=train_pool,
            feature_cols=self.feature_cols,
            target_metric=self.target_metric,
            target_assignments=self.target_assignments,
            positive_class=self.config_training.get("positive_class"),
            n_repeats=n_repeats,
            random_state=42,
            feature_groups=feature_groups,
        )

        results_df["n"] = n_repeats
        results_df = results_df.sort_values(by="importance", ascending=False)
        self.feature_importances[fi_storage_key(FIComputeMethod.PERMUTATION, partition)] = results_df

    def _calculate_fi_lofo(self):
        """LOFO feature importance.

        Delegates to ``_compute_lofo``, an octo-only Layer 1 primitive
        defined in this module.
        """
        logger.info("Calculating LOFO feature importance. This may take a while...")

        if self.x_train_processed is None:
            raise RuntimeError("Model must be fitted before computing LOFO FI.")

        target_cols = list(self.target_assignments.values())
        eval_partitions = {
            "dev": pd.concat([self.x_dev_processed, self.data_dev[target_cols]], axis=1),
            "test": pd.concat([self.x_test_processed, self.data_test[target_cols]], axis=1),
        }

        results = _compute_lofo(
            model=self.model,
            x_train_processed=self.x_train_processed,
            y_train=self.y_train,
            eval_partitions=eval_partitions,
            feature_cols=self.feature_cols,
            feature_groups=self.feature_groups,
            target_metric=self.target_metric,
            target_assignments=self.target_assignments,
            positive_class=self.config_training.get("positive_class"),
            n_targets=len(self.target_assignments),
        )

        for key, df in results.items():
            self.feature_importances[key] = df

    def _calculate_fi_featuresused_shap(
        self, partition: DataPartition | str = DataPartition.DEV, bg_max: int = 200
    ) -> None:
        """SHAP feature importance (for calc_features_used) with robust fallbacks.

        Uses ``"auto"`` shap_type via ``compute_shap_single`` from shared
        computation functions.  The auto mode lets SHAP pick the fastest
        explainer (Tree for tree-based, Linear for linear models, Kernel otherwise).

        Args:
            partition: Which partition to evaluate on (e.g. ``DataPartition.DEV``).
            bg_max: Maximum number of background samples for the explainer.
        """
        from octopus.feature_importance import compute_shap_single  # noqa: PLC0415

        partition = str(partition)
        if partition == DataPartition.DEV:
            X_eval_df = self.x_dev_processed
        elif partition == DataPartition.TEST:
            X_eval_df = self.x_test_processed
        else:
            raise ValueError(f"Unsupported partition: {partition!r}. Expected 'dev' or 'test'.")

        # Background from training data, sampled for speed
        X_bg_df = self.x_train_processed
        if X_bg_df is None:
            raise ValueError("Training data (x_train_processed) is required as background for SHAP.")
        if hasattr(X_bg_df, "sample") and X_bg_df.shape[0] > bg_max:
            X_bg_df = X_bg_df.sample(n=bg_max, replace=False, random_state=0)

        feature_names = list(self.feature_cols) if self.feature_cols else list(X_eval_df.columns)

        fi_df = compute_shap_single(
            model=self.model,
            X=X_eval_df,
            feature_names=feature_names,
            shap_type="auto",
            X_background=X_bg_df,
            threshold_ratio=1.0 / 1000.0,
            ml_type=self.ml_type,
        )
        self.feature_importances[fi_storage_key(FIComputeMethod.SHAP, partition)] = fi_df

    def _calculate_fi_shap(
        self, partition: DataPartition | str = DataPartition.DEV, shap_type: str = "kernel", background_size: int = 200
    ) -> None:
        """Compute SHAP feature importance with a model-agnostic explainer.

        Delegates to ``compute_shap_single`` from shared computation functions.

        Args:
            partition: Which partition to evaluate on (e.g. ``DataPartition.DEV``).
            shap_type: SHAP explainer type (``"kernel"``, ``"permutation"``, ``"exact"``).
            background_size: Maximum background dataset size for kernel explainer.
        """
        from octopus.feature_importance import compute_shap_single  # noqa: PLC0415

        logger.info(f"Calculating SHAP feature importances ({partition}, mode={shap_type})...")

        if partition == DataPartition.DEV:
            data = self.x_dev_processed
        elif partition == DataPartition.TEST:
            data = self.x_test_processed
        else:
            raise ValueError(f"Unsupported partition: {partition!r}. Expected 'dev' or 'test'.")

        feature_names = list(self.feature_cols) if self.feature_cols else list(data.columns)

        # Construct background set based on `background_size` for kernel SHAP
        if shap_type == "kernel" and background_size is not None:
            if len(data) > background_size:
                rng = np.random.default_rng(42)
                indices = rng.choice(len(data), size=background_size, replace=False)
                X_background = data.iloc[indices]
            else:
                X_background = data
        else:
            X_background = None

        fi_df = compute_shap_single(
            model=self.model,
            X=data,
            feature_names=feature_names,
            shap_type=shap_type,
            X_background=X_background,
            max_samples=None,
            threshold_ratio=1.0 / 1000.0,
            ml_type=self.ml_type,
        )
        self.feature_importances[fi_storage_key(FIComputeMethod.SHAP, partition)] = fi_df

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Predict.

        Args:
            x: Input data to make predictions on. Should have the same structure as training data.

        Returns:
            Predictions from the model.
        """
        # Ensure x is a DataFrame with proper column names for ColumnTransformer
        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x, columns=self.feature_cols)
        elif isinstance(x, pd.DataFrame):
            x = x[self.feature_cols].reset_index(drop=True)

        # Apply the same preprocessing pipeline used during training
        x_processed = self._transform_to_dataframe(x)
        return self.model.predict(x_processed)  # type: ignore

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Predict_proba.

        Args:
            x: Input data to make probability predictions on. Should have the same structure as training data.

        Returns:
            Probability predictions from the model.
        """
        # Ensure x is a DataFrame with proper column names for ColumnTransformer
        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x, columns=self.feature_cols)
        elif isinstance(x, pd.DataFrame):
            x = x[self.feature_cols].reset_index(drop=True)

        # Apply the same preprocessing pipeline used during training
        x_processed = self._transform_to_dataframe(x)
        return self.model.predict_proba(x_processed)  # type: ignore

    def _validate_model_trained(self):
        """Validate that the model actually trained using sklearn's check_is_fitted utility.

        This is a general approach that works with most sklearn-compatible models.

        Raises:
            RuntimeError: If the model appears to not have trained successfully.
        """
        if self.model is None:
            raise RuntimeError("Model is None - training failed")

        try:
            # Use sklearn's check_is_fitted utility - the most general approach
            check_is_fitted(self.model)
            logger.debug(f"Model {self.ml_model_type} validation passed - training appears successful")

        except Exception as e:
            # If check_is_fitted fails, the model is not properly fitted
            raise RuntimeError(
                f"Model {self.ml_model_type} validation failed - model appears not to be fitted: {e}"
            ) from e
