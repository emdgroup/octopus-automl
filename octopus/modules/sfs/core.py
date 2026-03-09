"""SFS execution module."""

from __future__ import annotations

import copy
import json
import warnings
from typing import TYPE_CHECKING

import pandas as pd
from attrs import define
from sklearn.model_selection import BaseCrossValidator, GridSearchCV, StratifiedKFold

from octopus.metrics import Metrics
from octopus.metrics.utils import get_score_from_model
from octopus.models import Models
from octopus.modules.base import FIDataset, FIMethod, ModuleExecution, ModuleResult, ResultType
from octopus.types import MLType, SFSDirection

if TYPE_CHECKING:
    from upath import UPath

    from octopus.modules.sfs import Sfs  # noqa: F401
    from octopus.study.context import StudyContext

# Ignore all Warnings
warnings.filterwarnings("ignore")

supported_models = {
    "CatBoostClassifier",
    "CatBoostRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "XGBClassifier",
    "XGBRegressor",
}


def get_param_grid(model_type):
    """Hyperparameter grid initialization."""
    if model_type in ("CatBoostClassifier", "CatBoostRegressor"):
        param_grid = {
            "learning_rate": [0.001, 0.01, 0.1],
            "depth": [3, 6, 8, 10],
            "l2_leaf_reg": [2, 5, 7, 10],
            "iterations": [500],
        }
    elif model_type in ("XGBClassifier", "XGBRegressor"):
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
            "n_estimators": [500],
        }
    return param_grid


@define
class SfsModule(ModuleExecution["Sfs"]):
    """SFS execution module. Created by Sfs.create_module()."""

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
        """Fit SFS module for feature selection."""
        # Extract data matrices (local variables)
        x_traindev = data_traindev[feature_cols]
        y_traindev = data_traindev[list(study_context.target_assignments.values())]

        from octopus._optional.sfs import SFS  # noqa: PLC0415

        # Configuration, define default model
        if study_context.ml_type in (MLType.BINARY, MLType.MULTICLASS):
            default_model = "CatBoostClassifier"
        elif study_context.ml_type == MLType.REGRESSION:
            default_model = "CatBoostRegressor"
        else:
            raise ValueError(f"{study_context.ml_type} not supported")

        model_type = self.config.model
        if model_type == "":
            model_type = default_model

        if model_type not in supported_models:
            raise ValueError(f"{model_type} not supported")
        print("Model used:", model_type)

        # set up model and scoring type
        model = Models.get_instance(model_type, {"random_state": 42})

        # Get scorer string from metrics inventory
        metric = Metrics.get_instance(study_context.target_metric)
        scoring_type = metric.scorer_string

        cv: int | BaseCrossValidator
        if study_context.stratification_col:
            cv = StratifiedKFold(n_splits=self.config.cv, shuffle=True, random_state=42)
        else:
            cv = self.config.cv

        # Silence catboost output
        if model_type == default_model:
            model.set_params(verbose=False, allow_writing_files=False)

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

        print(f"Number of features before SFS: {x_traindev.shape[1]}")

        # Select type of SFS
        if self.config.sfs_type == SFSDirection.FORWARD:
            forward = True
            floating = False
        elif self.config.sfs_type == SFSDirection.BACKWARD:
            forward = False
            floating = False
        elif self.config.sfs_type == SFSDirection.FLOATING_FORWARD:
            forward = True
            floating = True
        elif self.config.sfs_type == SFSDirection.FLOATING_BACKWARD:
            forward = False
            floating = True
        else:
            raise ValueError(f"Unsupported SFS type: {self.config.sfs_type}")

        sfs = SFS(
            estimator=best_model,
            k_features="best",
            forward=forward,
            floating=floating,
            cv=cv,
            scoring=scoring_type,
            verbose=1,
            n_jobs=1,
        )

        sfs.fit(x_traindev, y_traindev.squeeze(axis=1))
        n_optimal_features = len(sfs.k_feature_idx_)
        selected_features = list(sfs.k_feature_names_)

        print("SFS completed")
        print(f"Optimal number of features: {n_optimal_features}")
        print(f"Selected features: {selected_features}")
        print(f"Dev set performance: {sfs.k_score_:.3f}")

        # Report performance on test set
        target_assignments = {col: col for col in list(study_context.target_assignments.values())}
        positive_class = study_context.positive_class

        best_estimator = copy.deepcopy(best_model)
        x_traindev_sfs = sfs.transform(x_traindev)
        # refit on selected features
        best_estimator.fit(x_traindev_sfs, y_traindev.squeeze(axis=1))
        test_score_refit = get_score_from_model(
            best_estimator,
            data_test,
            selected_features,
            study_context.target_metric,
            target_assignments,
            positive_class=positive_class,
        )
        print(f"Test set (refit) performance: {test_score_refit:.3f}")

        # gridsearch + retrain best model on x_traindev
        grid_search.fit(x_traindev_sfs, y_traindev.squeeze(axis=1))  # type: ignore[arg-type]
        best_gs_parameters = grid_search.best_params_
        best_gs_estimator = grid_search.best_estimator_
        best_gs_estimator.fit(x_traindev_sfs, y_traindev.squeeze(axis=1))  # refit
        test_score_gsrefit = get_score_from_model(
            best_gs_estimator,
            data_test,
            selected_features,
            study_context.target_metric,
            target_assignments,
            positive_class=positive_class,
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
                    "fold": None,
                    "value": sfs.k_score_,
                },
                {
                    "result_type": ResultType.BEST,
                    "metric": study_context.target_metric,
                    "partition": "test",
                    "aggregation": "refit",
                    "fold": None,
                    "value": test_score_refit,
                },
                {
                    "result_type": ResultType.BEST,
                    "metric": study_context.target_metric,
                    "partition": "test",
                    "aggregation": "gsrefit",
                    "fold": None,
                    "value": test_score_gsrefit,
                },
            ]
        )

        # Build standard feature_importances DataFrame
        feature_importances = fi_df[["feature", "importance"]].copy()
        feature_importances["fi_method"] = FIMethod.INTERNAL
        feature_importances["fi_dataset"] = FIDataset.TRAIN
        feature_importances["training_id"] = "sfs"
        feature_importances["result_type"] = ResultType.BEST

        # Save results to JSON
        results = {
            "Dev score start": best_cv_score,
            "best_params": best_gs_parameters,
            "optimal_features": int(n_optimal_features),
            "selected_features": selected_features,
            "Dev set performance": sfs.k_score_,
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
                feature_importances=feature_importances,
            )
        }
