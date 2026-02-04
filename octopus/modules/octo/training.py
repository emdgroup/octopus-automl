# type: ignore

"""Octo Training."""

import copy
import gzip
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import shap
from attrs import Factory, define, field, validators
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted
from upath import UPath

from octopus.logger import LogGroup, get_logger
from octopus.metrics import Metrics
from octopus.metrics.utils import get_score_from_model
from octopus.models import Models

## TOBEDONE pipeline
# - implement cat encoding on module level
# - how to provide categorical info to catboost and other models?


logger = get_logger()


@define
class Training:
    """Model Training Class."""

    training_id: str = field(validator=[validators.instance_of(str)])
    """Training id."""

    ml_type: str = field(validator=[validators.instance_of(str)])
    """ML-type."""

    target_assignments: dict = field(validator=[validators.instance_of(dict)])
    """Target assignments."""

    feature_cols: list[str] = field(validator=[validators.instance_of(list)])
    """Feature columns."""

    row_column: str = field(validator=[validators.instance_of(str)])
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

    config_training: dict = field(validator=[validators.instance_of(dict)])
    """Training configuration."""

    training_weight: int = field(default=1, validator=[validators.instance_of(int)])
    """Training weight for ensembling"""

    model = field(default=None)
    """Model."""

    predictions: dict = field(default=Factory(dict), validator=[validators.instance_of(dict)])
    """Model predictions."""

    feature_importances: dict[str, pd.DataFrame] = field(
        default=Factory(dict), validator=[validators.instance_of(dict)]
    )
    """Feature importances."""

    features_used: list = field(default=Factory(list), validator=[validators.instance_of(list)])
    """Features used."""

    outlier_samples: list = field(default=Factory(list), validator=[validators.instance_of(list)])
    """Outlie samples identified."""

    is_fitted: bool = field(default=False, init=False)
    """Flag indicating whether the training has been completed."""

    preprocessing_pipeline = field(init=False)
    """Preprocessing pipeline for data scaling, imputation, and categorical encoding."""

    x_train_processed = field(default=None, init=False)
    """Training data after pre-processing (outlier, impuation, scaling)."""

    @property
    def outl_reduction(self) -> int:
        """Parameter outlier reduction method."""
        return self.config_training["outl_reduction"]

    @property
    def ml_model_type(self) -> str:
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
        processed_data = self.preprocessing_pipeline.transform(self.data_dev[self.feature_cols])
        # Convert back to DataFrame to preserve column names
        if hasattr(processed_data, "shape") and len(processed_data.shape) == 2:
            return pd.DataFrame(processed_data, columns=self.feature_cols, index=self.data_dev.index)
        return processed_data

    @property
    def x_test_processed(self):
        """x_test_processed."""
        processed_data = self.preprocessing_pipeline.transform(self.data_test[self.feature_cols])
        # Convert back to DataFrame to preserve column names
        if hasattr(processed_data, "shape") and len(processed_data.shape) == 2:
            return pd.DataFrame(processed_data, columns=self.feature_cols, index=self.data_test.index)
        return processed_data

    @property
    def y_train(self):
        """y_train."""
        if self.ml_type == "timetoevent":
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
        if self.ml_type == "timetoevent":
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
        """y_dev."""
        if self.ml_type == "timetoevent":
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
            steps = []
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

    def fit(self):
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
            self.outlier_samples = data_train[outlier_pred == -1][self.row_column].tolist()
            # print("Outlier samples:", self.outlier_samples)

            # Remove outliers from data_train, x_train, y_train
            data_train = data_train[outlier_pred == 1].copy()
            x_train = x_train[outlier_pred == 1].copy()
            y_train = y_train[outlier_pred == 1].copy()

        # (2) Imputation and scaling (after outlier removal)
        processed_data = self.preprocessing_pipeline.fit_transform(x_train)
        # Convert back to DataFrame to preserve column names
        if hasattr(processed_data, "shape") and len(processed_data.shape) == 2:
            self.x_train_processed = pd.DataFrame(processed_data, columns=self.feature_cols, index=x_train.index)
        else:
            self.x_train_processed = processed_data

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
        self.predictions["train"] = pd.DataFrame()
        self.predictions["train"][self.row_column] = data_train[self.row_column]
        self.predictions["train"]["prediction"] = self.model.predict(self.x_train_processed)

        self.predictions["dev"] = pd.DataFrame()
        self.predictions["dev"][self.row_column] = self.data_dev[self.row_column]
        self.predictions["dev"]["prediction"] = self.model.predict(self.x_dev_processed)

        self.predictions["test"] = pd.DataFrame()
        self.predictions["test"][self.row_column] = self.data_test[self.row_column]
        self.predictions["test"]["prediction"] = self.model.predict(self.x_test_processed)

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

        # add additional predictions for classifications
        if self.ml_type == "classification":
            columns = [int(x) for x in self.model.classes_]  # column names --> int
            self.predictions["train"][columns] = self.model.predict_proba(self.x_train_processed)
            self.predictions["dev"][columns] = self.model.predict_proba(self.x_dev_processed)
            self.predictions["test"][columns] = self.model.predict_proba(self.x_test_processed)

        # add additional predictions for multiclass classifications
        if self.ml_type == "multiclass":
            columns = [int(x) for x in self.model.classes_]  # column names --> int
            self.predictions["train"][columns] = self.model.predict_proba(self.x_train_processed)
            self.predictions["dev"][columns] = self.model.predict_proba(self.x_dev_processed)
            self.predictions["test"][columns] = self.model.predict_proba(self.x_test_processed)

        # add additional predictions for time to event predictions
        if self.ml_type == "timetoevent":
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

        if feature_method == "internal":
            self.calculate_fi_internal()
            fi_df = self.feature_importances["internal"]
        elif feature_method == "shap":
            self.calculate_fi_featuresused_shap(partition="dev")
            fi_df = self.feature_importances["shap_dev"]
        elif feature_method == "permutation":
            self.calculate_fi_permutation(partition="dev", n_repeats=2)  # only 2 repeats!
            fi_df = self.feature_importances["permutation_dev"]
        elif feature_method == "constant":
            self.calculate_fi_constant()
            fi_df = self.feature_importances["constant"]
        else:
            raise ValueError("feature method provided in model config not supported")

        # Check if feature importance calculation failed (empty DataFrame)
        if fi_df.empty:
            logger.warning(
                f"Feature importance calculation failed for model {self.ml_model_type} "
                f"using method {feature_method}. Returning all features as used."
            )
            return self.feature_cols.copy()

        features_used = fi_df[fi_df["importance"] != 0]["feature"].tolist()

        # If no features have non-zero importance, return all features as a fallback
        if not features_used:
            logger.warning(
                f"All feature importances are zero for model {self.ml_model_type} "
                f"using method {feature_method}. Returning all features as used."
            )
            return self.feature_cols.copy()

        return features_used

    def calculate_fi_constant(self):
        """Provide flat feature importance table."""
        fi_df = pd.DataFrame()
        fi_df["feature"] = self.feature_cols
        fi_df["importance"] = 1
        self.feature_importances["constant"] = fi_df

    def calculate_fi_internal(self):
        """Sklearn-provided internal feature importance (based on train dataset)."""
        # Handle unsupported "timetoevent" case as in your original code
        if getattr(self, "ml_type", None) == "timetoevent":
            fi_df = pd.DataFrame(columns=["feature", "importance"])
            logger.warning("Internal features importances not available for timetoevent.")
            self.feature_importances["internal"] = fi_df
            return

        # 1) Tree-based models exposing feature_importances_
        if hasattr(self.model, "feature_importances_"):
            fi = np.asarray(self.model.feature_importances_)
            fi_df = pd.DataFrame({"feature": self.feature_cols, "importance": fi})
            self.feature_importances["internal"] = fi_df
            return

        # 2) Linear models exposing coef_: Ridge, LinearSVC/LinearSVR, SVC/SVR with kernel='linear'
        if hasattr(self.model, "coef_"):
            coef = np.asarray(self.model.coef_)
            # coef_ can be:
            # - shape (n_features,) for single-target regression (e.g., Ridge)
            # - shape (n_targets, n_features) for multi-target regression
            # - shape (n_classes, n_features) for LinearSVC/LinearSVR (OvR)
            # - shape (n_class_pairs, n_features) for SVC with kernel='linear' (OvO)
            if coef.ndim == 1:
                importance = np.abs(coef)
            else:
                # Aggregate across classes/targets/pairs
                importance = np.mean(np.abs(coef), axis=0)

            if len(importance) != len(self.feature_cols):
                # Defensive check in case columns mismatch model coefficients
                logger.warning(
                    "Length mismatch between coefficients (%d) and feature columns (%d). "
                    "Skipping internal importances.",
                    len(importance),
                    len(self.feature_cols),
                )
                fi_df = pd.DataFrame(columns=["feature", "importance"])
            else:
                fi_df = pd.DataFrame({"feature": self.feature_cols, "importance": importance})

            self.feature_importances["internal"] = fi_df
            return

        # Fallback
        fi_df = pd.DataFrame(columns=["feature", "importance"])
        logger.warning("Internal features importances not available for this estimator.")
        self.feature_importances["internal"] = fi_df

    def calculate_fi_group_permutation(self, partition="dev", n_repeats=10):
        """Permutation feature importance, group version."""
        logger.set_log_group(LogGroup.TRAINING, f"{self.training_id}")

        logger.info(f"Calculating permutation feature importances ({partition}). This may take a while...")
        np.random.seed(42)  # reproducibility
        # fixed confidence level
        confidence_level = 0.95
        feature_cols = self.feature_cols
        target_assignments = self.target_assignments
        target_metric = self.target_metric
        model = self.model
        feature_groups = self.feature_groups

        target_cols = list(target_assignments.values())
        if partition == "dev":
            # concat processed input + target columns
            data = pd.concat([self.x_dev_processed, self.data_dev[target_cols]], axis=1)
        elif partition == "test":
            data = pd.concat([self.x_test_processed, self.data_test[target_cols]], axis=1)

        if not set(feature_cols).issubset(data.columns):
            raise ValueError("Features missing in provided dataset.")

        # keep all features and add group features
        # create features dict
        feature_cols_dict = {x: [x] for x in feature_cols}
        features_dict = {**feature_cols_dict, **feature_groups}

        # calculate baseline score
        baseline_score = get_score_from_model(
            model,
            data,
            feature_cols,
            target_metric,
            target_assignments,
            positive_class=self.config_training.get("positive_class"),
        )

        results_df = pd.DataFrame(
            columns=[
                "feature",
                "importance",
                "stddev",
                "p-value",
                "n",
                "ci_low_95",
                "ci_high_95",
            ]
        )
        # calculate pfi
        for name, feature in features_dict.items():
            data_pfi = data.copy()
            fi_lst = []

            for _ in range(n_repeats):
                # replace column with random selection from that column of data_all
                # we use data_all as the validation dataset may be small
                for feat in feature:
                    data_pfi[feat] = np.random.choice(data[feat], len(data_pfi), replace=False)
                pfi_score = get_score_from_model(
                    model,
                    data_pfi,
                    feature_cols,
                    target_metric,
                    target_assignments,
                    positive_class=self.config_training.get("positive_class"),
                )
                fi_lst.append(baseline_score - pfi_score)

            # calculate statistics
            pfi_mean = np.mean(fi_lst)
            n = len(fi_lst)
            p_value = np.nan
            stddev = np.std(fi_lst, ddof=1) if n > 1 else np.nan
            if stddev not in (np.nan, 0):
                t_stat = pfi_mean / (stddev / math.sqrt(n))
                p_value = scipy.stats.t.sf(t_stat, n - 1)
            elif stddev == 0:
                p_value = 0.5

            # calculate confidence intervals
            if any(np.isnan(val) for val in [stddev, n, pfi_mean]) or n == 1:
                ci_high = np.nan
                ci_low = np.nan
            else:
                t_val = scipy.stats.t.ppf(1 - (1 - confidence_level) / 2, n - 1)
                ci_high = pfi_mean + t_val * stddev / math.sqrt(n)
                ci_low = pfi_mean - t_val * stddev / math.sqrt(n)

            # save results
            results_df.loc[len(results_df)] = [
                name,
                pfi_mean,
                stddev,
                p_value,
                n,
                ci_low,
                ci_high,
            ]

        results_df = results_df.sort_values(by="importance", ascending=False)
        self.feature_importances["permutation" + "_" + partition] = results_df

    def calculate_fi_permutation(self, partition="dev", n_repeats=10):
        """Permutation feature importance."""
        logger.info(f"Calculating permutation feature importances ({partition}). This may take a while...")
        np.random.seed(42)  # reproducibility
        if self.ml_type == "timetoevent":
            # sksurv models only provide inbuilt scorer (CI)
            # more work needed to support other metrics
            scoring_type = None
        else:
            # Get scorer string from metrics inventory
            metric = Metrics.get_instance(self.target_metric)
            scoring_type = metric.scorer_string

        if partition == "dev":
            x = self.x_dev_processed
            y = self.y_dev
        elif partition == "test":
            x = self.x_test_processed
            y = self.y_test

        perm_importance = permutation_importance(
            self.model,
            X=x,
            y=y,
            n_repeats=n_repeats,
            random_state=0,
            scoring=scoring_type,
        )

        fi_df = pd.DataFrame()
        fi_df["feature"] = self.feature_cols
        fi_df["importance"] = perm_importance.importances_mean
        fi_df["importance_std"] = perm_importance.importances_std
        self.feature_importances["permutation" + "_" + partition] = fi_df

    def calculate_fi_lofo(self):
        """LOFO feature importance."""
        np.random.seed(42)  # reproducibility
        logger.info("Calculating LOFO feature importance. This may take a while...")
        # first, dev only
        feature_cols = self.feature_cols
        target_assignments = self.target_assignments
        # calculate dev+test baseline scores
        target_cols = list(target_assignments.values())
        data_dev = pd.concat([self.x_dev_processed, self.data_dev[target_cols]], axis=1)
        data_test = pd.concat([self.x_test_processed, self.data_test[target_cols]], axis=1)

        baseline_dev = get_score_from_model(
            self.model,
            data_dev,
            feature_cols,
            self.target_metric,
            self.target_assignments,
            positive_class=self.config_training.get("positive_class"),
        )
        baseline_test = get_score_from_model(
            self.model,
            data_test,
            feature_cols,
            self.target_metric,
            self.target_assignments,
            positive_class=self.config_training.get("positive_class"),
        )

        # create features dict
        feature_cols_dict = {x: [x] for x in feature_cols}
        lofo_features = {**feature_cols_dict, **self.feature_groups}

        # lofo
        fi_dev_df = pd.DataFrame(columns=["feature", "importance"])
        fi_test_df = pd.DataFrame(columns=["feature", "importance"])
        for name, lofo_feature in lofo_features.items():
            selected_features = copy.deepcopy(feature_cols)
            model = copy.deepcopy(self.model)
            selected_features = [x for x in selected_features if x not in lofo_feature]
            # retrain model
            if len(self.target_assignments) == 1:
                # standard sklearn single target models
                model.fit(
                    self.x_train_processed[selected_features],
                    self.y_train.squeeze(axis=1),
                )
            else:
                # multi target models, incl. time2event
                model.fit(self.x_train_processed[selected_features], self.y_train)

            # get lofo dev + test scores
            score_dev = get_score_from_model(
                model,
                data_dev,
                selected_features,
                self.target_metric,
                self.target_assignments,
                positive_class=self.config_training.get("positive_class"),
            )
            score_test = get_score_from_model(
                model,
                data_test,
                selected_features,
                self.target_metric,
                self.target_assignments,
                positive_class=self.config_training.get("positive_class"),
            )

            fi_dev_df.loc[len(fi_dev_df)] = [name, baseline_dev - score_dev]
            fi_test_df.loc[len(fi_test_df)] = [name, baseline_test - score_test]

        self.feature_importances["lofo" + "_dev"] = fi_dev_df
        self.feature_importances["lofo" + "_test"] = fi_test_df

    def calculate_fi_featuresused_shap(self, partition="dev", bg_max=200):
        """SHAP feature importance (for calc_features_used) with robust fallbacks.

        Used when model property: feature_method = "shap". The shap method used is automatically determined
        by shap. The main advantage is that for linear and tree model the feature importances are calculated
        much faster.
        """
        # Select eval data
        X_eval_df = {"dev": self.x_dev_processed, "test": self.x_test_processed}.get(partition)
        if X_eval_df is None:
            raise ValueError("dataset type not supported")

        # Background from train; sample for speed
        X_bg_df = self.x_train_processed
        if X_bg_df is None:
            raise ValueError("Training data (x_train_processed) is required as background for SHAP.")
        if hasattr(X_bg_df, "sample") and X_bg_df.shape[0] > bg_max:
            X_bg_df = X_bg_df.sample(n=bg_max, replace=False, random_state=0)

        # Convert to numpy to avoid model attribute side-effects
        X_eval = X_eval_df.to_numpy() if hasattr(X_eval_df, "to_numpy") else np.asarray(X_eval_df)
        X_bg = X_bg_df.to_numpy() if hasattr(X_bg_df, "to_numpy") else np.asarray(X_bg_df)

        n_features = X_eval.shape[1]

        # Resolve feature names
        feature_names = getattr(self, "feature_cols", None)
        if not feature_names or len(feature_names) != n_features:
            if hasattr(X_eval_df, "columns"):
                feature_names = list(X_eval_df.columns)
            else:
                feature_names = [f"f{i}" for i in range(n_features)]

        # Build explainer
        try:
            # Let SHAP auto-select the best explainer (Tree for tree models, Kernel otherwise)
            explainer = shap.Explainer(self.model, X_bg)
            sv = explainer(X_eval)
        except Exception as e1:
            logger.debug(f"SHAP auto explainer failed: {e1}. Falling back to callable + Kernel.")
            # Fallback to a plain callable; do NOT pass string link (avoids 'link needs to be callable' errors)
            if getattr(self, "ml_type", None) == "classification" and hasattr(self.model, "predict_proba"):

                def predict_fn(X):
                    return np.asarray(self.model.predict_proba(np.asarray(X)))
            else:

                def predict_fn(X):
                    return np.asarray(self.model.predict(np.asarray(X)))

            # Use the generic constructor so SHAP picks Kernel with the given background
            explainer = shap.Explainer(predict_fn, X_bg)
            sv = explainer(X_eval)

        # SHAP values and aggregation
        vals = np.asarray(sv.values)  # could be (n, f) or (n, outputs, f), etc.
        if vals.ndim == 2 and vals.shape[1] == n_features:
            importance = np.abs(vals).mean(axis=0)
        else:
            feat_axes = [i for i, d in enumerate(vals.shape) if d == n_features]
            if len(feat_axes) != 1:
                raise ValueError(f"Unexpected SHAP values shape {vals.shape}")
            feat_axis = feat_axes[0]
            reduce_axes = tuple(i for i in range(vals.ndim) if i != feat_axis)
            importance = np.mean(np.abs(vals), axis=reduce_axes)

        if importance.shape[0] != n_features:
            raise ValueError("Feature count mismatch between SHAP values and feature_cols.")

        # Build and store importance DataFrame
        fi_df = pd.DataFrame({"feature": feature_names, "importance": importance})
        if not fi_df["importance"].empty:
            fi_df = fi_df[fi_df["importance"] > fi_df["importance"].max() / 1000.0]

        self.feature_importances[f"shap_{partition}"] = fi_df

    def calculate_fi_shap(self, partition="dev", shap_type="kernel", background_size=200):
        """Compute SHAP feature importance with a model-agnostic explainer (kernel/permutation/exact)."""
        logger.info(f"Calculating SHAP feature importances ({partition}, mode={shap_type})...")

        # --- Select data
        if partition == "dev":
            data = self.x_dev_processed
        elif partition == "test":
            data = self.x_test_processed
        else:
            raise ValueError("dataset type not supported")

        # Keep feature names, but pass numpy to SHAP to avoid feature_names_in_ issues
        if hasattr(data, "columns"):
            feature_names = data.columns.tolist()
            X = data.to_numpy()
        else:
            X = np.asarray(data)
            feature_names = [f"f{i}" for i in range(X.shape[1])]

        # --- Prediction function as a plain callable (not a bound method)
        if getattr(self, "ml_type", None) == "classification" and hasattr(self.model, "predict_proba"):

            def predict_fn(X_in):
                return np.asarray(self.model.predict_proba(np.asarray(X_in)))
        else:

            def predict_fn(X_in):
                return np.asarray(self.model.predict(np.asarray(X_in)))

        # --- Build explainer (no tree option)
        if shap_type == "kernel":
            # Kernel expects a background dataset (array or DataFrame), not a masker
            # Sample a manageable background for speed
            try:
                bg = X if X.shape[0] <= background_size else shap.utils.sample(X, background_size, random_state=0)
            except Exception:
                # Fallback sampling if shap.utils.sample is unavailable
                rng = np.random.default_rng(0)
                idx = rng.choice(X.shape[0], size=min(background_size, X.shape[0]), replace=False)
                bg = X[idx]
            explainer = shap.explainers.Kernel(predict_fn, bg)

        elif shap_type == "permutation":
            explainer = shap.explainers.Permutation(predict_fn, X)

        elif shap_type == "exact":
            explainer = shap.explainers.Exact(predict_fn, X)

        else:
            raise ValueError(f"SHAP type {shap_type} not supported. Use 'kernel', 'permutation', or 'exact'.")

        # --- Compute SHAP values
        sv = explainer(X)
        vals = np.asarray(sv.values)  # shape may be (n, f) or (n, outputs, f), etc.
        n_features = X.shape[1]

        # --- Aggregate absolute SHAP to per-feature importances
        if vals.ndim == 2 and vals.shape[1] == n_features:
            importance = np.abs(vals).mean(axis=0)
        else:
            feat_axes = [i for i, d in enumerate(vals.shape) if d == n_features]
            if len(feat_axes) != 1:
                raise ValueError(f"Unexpected SHAP values shape {vals.shape}")
            feat_axis = feat_axes[0]
            reduce_axes = tuple(i for i in range(vals.ndim) if i != feat_axis)
            importance = np.mean(np.abs(vals), axis=reduce_axes)

        # --- Build importance DataFrame
        fi_df = pd.DataFrame({"feature": feature_names, "importance": importance})
        if not fi_df["importance"].empty:
            fi_df = fi_df[fi_df["importance"] > fi_df["importance"].max() / 1000.0]

        self.feature_importances[f"shap_{partition}"] = fi_df

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
            # Reset index to avoid sklearn ColumnTransformer issues
            x = x.reset_index(drop=True)

        # Apply the same preprocessing pipeline used during training
        x_processed = self.preprocessing_pipeline.transform(x)
        return self.model.predict(x_processed)

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
            # Reset index to avoid sklearn ColumnTransformer issues
            x = x.reset_index(drop=True)

        # Apply the same preprocessing pipeline used during training
        x_processed = self.preprocessing_pipeline.transform(x)
        return self.model.predict_proba(x_processed)

    def to_pickle(self, file_path: str | Path | UPath):
        """Save object to a compressed pickle file."""
        with file_path.open("wb") as file, gzip.GzipFile(fileobj=file, mode="wb") as gzip_file:
            pickle.dump(self, gzip_file)

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

    @classmethod
    def from_pickle(cls, file_path: str | Path | UPath) -> "Training":
        """Load object from a compressed pickle file."""
        with file_path.open("rb") as file, gzip.GzipFile(fileobj=file, mode="rb") as gzip_file:
            data = pickle.load(gzip_file)

        if not isinstance(data, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}")

        return data
