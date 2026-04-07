"""Boruta execution module."""

from __future__ import annotations

import copy
import json
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from attrs import define
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from octopus.metrics import Metrics
from octopus.metrics.utils import get_score_from_model
from octopus.models import Models
from octopus.modules import ModuleExecution, ModuleResult, StudyContext
from octopus.types import DataPartition, FIResultLabel, MLType, ModelName, ResultType

if TYPE_CHECKING:
    from upath import UPath

    from octopus.modules import Boruta  # noqa: F401

# Ignore all Warnings
warnings.filterwarnings("ignore")

# Tree Based models
supported_models = {
    ModelName.RandomForestClassifier,
    ModelName.RandomForestRegressor,
    ModelName.ExtraTreesClassifier,
    ModelName.ExtraTreesRegressor,
    ModelName.XGBClassifier,
    ModelName.XGBRegressor,
}


def get_param_grid(model_type: ModelName):
    """Hyperparameter grid initialization."""
    if model_type in (ModelName.XGBClassifier, ModelName.XGBRegressor):
        param_grid = {
            "learning_rate": [0.0001, 0.001, 0.01, 0.3],
            "min_child_weight": [2, 5, 10, 15],
            "subsample": [0.15, 0.3, 0.7, 1],
            "n_estimators": [30, 70, 140, 200],
            "max_depth": [3, 5, 7, 9],
        }
    else:
        # RF and ExtraTrees
        param_grid = {
            "max_depth": [3, 6, 10],
            "min_samples_split": [2, 25, 50, 100],
            "min_samples_leaf": [1, 15, 30, 50],
            "max_features": [0.1, 0.5, 1],
            "n_estimators": [500],
        }
    return param_grid


@define
class BorutaModule(ModuleExecution["Boruta"]):
    """Boruta execution module. Created by Boruta.create_module()."""

    def fit(
        self,
        *,
        data_traindev: pd.DataFrame,
        data_test: pd.DataFrame,
        feature_cols: list[str],
        study_context: StudyContext,
        results_dir: UPath,
        **kwargs,
    ) -> dict[ResultType, ModuleResult]:
        """Fit Boruta module for feature selection."""
        from octopus._optional.burota import BorutaPy  # noqa: PLC0415

        # Extract data matrices (local variables)
        x_traindev = data_traindev[feature_cols]
        y_traindev = data_traindev[list(study_context.target_assignments.values())]

        # Configuration, define default model
        if study_context.ml_type in (MLType.BINARY, MLType.MULTICLASS):
            default_model = ModelName.RandomForestClassifier
        elif study_context.ml_type == MLType.REGRESSION:
            default_model = ModelName.RandomForestRegressor
        else:
            raise ValueError(f"{study_context.ml_type} not supported")

        model_type = self.config.model if self.config.model is not None else default_model

        if model_type not in supported_models:
            raise ValueError(f"{model_type} not supported")
        print("Model used:", model_type)

        # set up model and scoring type
        model = Models.get_instance(model_type, {"random_state": 42, "verbose": False})
        # Get scorer string from metrics
        metric = Metrics.get_instance(study_context.target_metric)
        scoring_type = metric.scorer_string

        # Setup CV strategy
        target_assignments = {col: col for col in list(study_context.target_assignments.values())}

        cv: int | StratifiedKFold
        if study_context.stratification_col:
            cv = StratifiedKFold(n_splits=self.config.n_inner_splits, shuffle=True, random_state=42)
        else:
            cv = self.config.n_inner_splits

        # Hyperparameter optimization
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=get_param_grid(model_type),
            cv=cv,
            scoring=scoring_type,
            n_jobs=1,
        )
        print("Optimize base model....")
        # Perform Grid Search and Cross-Validation
        grid_search.fit(x_traindev, y_traindev.squeeze(axis=1))  # type: ignore[arg-type]
        best_model = grid_search.best_estimator_
        best_cv_score = grid_search.best_score_
        best_params = grid_search.best_params_

        # Report performance
        print(f"Dev start performance: {best_cv_score:.3f}")
        print(f"Best params: {best_params}")

        print(f"Initial number of features: {x_traindev.shape[1]}")

        boruta = BorutaPy(
            estimator=model,
            n_estimators="auto",
            perc=self.config.threshold,
            alpha=self.config.alpha,
            random_state=42,
            verbose=0,
        )

        boruta.fit(x_traindev, y_traindev.squeeze(axis=1))

        print("Feature Selection completed")
        selected_features = [feature_cols[i] for i in range(len(boruta.support_)) if boruta.support_[i]]
        n_optimal_features = len(selected_features)

        print(f"Optimal number of features: {n_optimal_features}")
        print(f"Selected features: {selected_features}")

        # Report performance on dev and test set
        best_estimator = copy.deepcopy(best_model)
        x_traindev_filtered = boruta.transform(x_traindev)
        cv_score_dev = cross_val_score(
            best_model,
            x_traindev_filtered,
            y_traindev,
            scoring=scoring_type,
            cv=cv,
        )
        dev_score_cv = np.mean(cv_score_dev)
        print(f"Dev set (cv) performance : {dev_score_cv:.3f}")

        # refit on selected features
        best_estimator.fit(x_traindev_filtered, y_traindev.squeeze(axis=1))
        test_score_refit = get_score_from_model(
            best_estimator,
            data_test,
            selected_features,
            study_context.target_metric,
            target_assignments,
            positive_class=study_context.positive_class,
        )
        print(f"Test set (refit) performance: {test_score_refit:.3f}")

        # gridsearch + retrain best model on x_traindev
        grid_search.fit(x_traindev_filtered, y_traindev.squeeze(axis=1))  # type: ignore[arg-type]
        best_gs_parameters = grid_search.best_params_
        best_gs_estimator = grid_search.best_estimator_
        # refit
        best_gs_estimator.fit(x_traindev_filtered, y_traindev.squeeze(axis=1))
        test_score_gsrefit = get_score_from_model(
            best_gs_estimator,
            data_test,
            selected_features,
            study_context.target_metric,
            target_assignments,
            positive_class=study_context.positive_class,
        )
        print(f"Test set (gridsearch+refit) performance: {test_score_gsrefit:.3f}")

        # feature importances
        fi_df = pd.DataFrame(
            {
                "feature": selected_features,
                "importance": best_gs_estimator.feature_importances_,
            }
        ).sort_values(by="importance", ascending=False)

        # Build standard scores DataFrame
        scores = pd.DataFrame(
            [
                {
                    "result_type": ResultType.BEST,
                    "metric": study_context.target_metric,
                    "partition": "dev",
                    "aggregation": "avg",
                    "split": None,
                    "value": dev_score_cv,
                },
                {
                    "result_type": ResultType.BEST,
                    "metric": study_context.target_metric,
                    "partition": "test",
                    "aggregation": "refit",
                    "split": None,
                    "value": test_score_refit,
                },
                {
                    "result_type": ResultType.BEST,
                    "metric": study_context.target_metric,
                    "partition": "test",
                    "aggregation": "gsrefit",
                    "split": None,
                    "value": test_score_gsrefit,
                },
            ]
        )

        # Build standard feature importance DataFrame
        fi_df_out = fi_df[["feature", "importance"]].copy()
        fi_df_out["fi_method"] = FIResultLabel.INTERNAL
        fi_df_out["fi_dataset"] = DataPartition.TRAIN
        fi_df_out["training_id"] = "boruta"
        fi_df_out["result_type"] = ResultType.BEST

        # Save results to JSON
        results = {
            "Dev score start": best_cv_score,
            "best_params": best_gs_parameters,
            "optimal_features": int(n_optimal_features),
            "selected_features": selected_features,
            "Dev set performance": dev_score_cv,
            "Test set (refit) performance": test_score_refit,
            "Test set (gs+refit) performance": test_score_gsrefit,
        }

        with (results_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        return {
            ResultType.BEST: ModuleResult(
                result_type=ResultType.BEST,
                module=self.config.module,
                selected_features=selected_features,
                scores=scores,
                fi=fi_df_out,
            )
        }
