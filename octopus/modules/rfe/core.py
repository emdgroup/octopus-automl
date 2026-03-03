"""RFE execution module."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING

import pandas as pd
from attrs import define
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from octopus.metrics import Metrics
from octopus.metrics.utils import get_score_from_model
from octopus.models import Models
from octopus.modules.base import FeatureSelectionExecution, FIDataset, FIMethod, ModuleResult, ResultType

if TYPE_CHECKING:
    from upath import UPath

    from octopus.modules.rfe.module import Rfe  # noqa: F401
    from octopus.study.context import StudyContext

# Supported models for RFE
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


def get_feature_importances(estimator):
    """Get feature importances from estimator or GridSearchCV."""
    if hasattr(estimator, "best_estimator_"):
        return estimator.best_estimator_.feature_importances_
    return estimator.feature_importances_


def get_param_grid(model_type: str) -> dict:
    """Get hyperparameter grid for model type."""
    if model_type in ("CatBoostClassifier", "CatBoostRegressor"):
        return {
            "learning_rate": [0.001, 0.01, 0.1],
            "depth": [3, 6, 8, 10],
            "l2_leaf_reg": [2, 5, 7, 10],
            "iterations": [500],
        }
    elif model_type in ("XGBClassifier", "XGBRegressor"):
        return {
            "learning_rate": [0.0001, 0.001, 0.01, 0.3],
            "min_child_weight": [2, 5, 10, 15],
            "subsample": [0.15, 0.3, 0.7, 1],
            "n_estimators": [30, 70, 140, 200],
            "max_depth": [3, 5, 7, 9],
        }
    else:
        # RF and ExtraTrees
        return {
            "max_depth": [3, 6, 10],
            "min_samples_split": [2, 25, 50, 100],
            "n_estimators": [500],
        }


@define
class RfeModule(FeatureSelectionExecution["Rfe"]):
    """RFE execution module. Created by Rfe.create_module()."""

    def fit(
        self,
        *,
        data_traindev: pd.DataFrame,
        data_test: pd.DataFrame,
        feature_cols: list[str],
        study_context: StudyContext,
        output_dir: UPath,
        **kwargs,
    ) -> dict[ResultType, ModuleResult]:
        """Fit RFE module by recursively eliminating features."""
        # Extract data matrices (local variables)
        x_traindev = data_traindev[feature_cols]
        y_traindev = data_traindev[list(study_context.target_assignments.values())]

        # Create results directory
        path_results = output_dir / "results"
        path_results.mkdir(parents=True, exist_ok=True)

        # Determine default model based on ml_type
        if study_context.ml_type == "classification":
            default_model = "CatBoostClassifier"
        elif study_context.ml_type == "regression":
            default_model = "CatBoostRegressor"
        else:
            raise ValueError(f"{study_context.ml_type} not supported")

        model_type = self.config.model if self.config.model else default_model

        if model_type not in supported_models:
            raise ValueError(f"{model_type} not supported")
        print("Model used:", model_type)

        # Set up model and scoring
        model = Models.get_instance(model_type, {"random_state": 42})
        metric = Metrics.get_instance(study_context.target_metric)
        scoring_type = metric.scorer_string

        # Configure cross-validation
        target_assignments = {col: col for col in list(study_context.target_assignments.values())}
        positive_class = study_context.positive_class

        cv: int | StratifiedKFold
        if study_context.stratification_col:
            cv = StratifiedKFold(n_splits=self.config.cv, shuffle=True, random_state=42)
        else:
            cv = self.config.cv

        # Silence catboost output
        if model_type in ("CatBoostClassifier", "CatBoostRegressor"):
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
        grid_search.fit(x_traindev, y_traindev.squeeze(axis=1))  # type: ignore[arg-type]
        best_model = grid_search.best_estimator_
        best_cv_score = grid_search.best_score_
        best_params = grid_search.best_params_

        print(f"Dev start performance: {best_cv_score:.3f}")
        print(f"Best params: {best_params}")

        # Select estimator based on mode
        if self.config.mode == "Mode1":
            # RFE with the trained model
            estimator = best_model
        elif self.config.mode == "Mode2":
            # RFE with hyperparameter optimization at each step
            estimator = grid_search
        else:
            raise ValueError(f"Unsupported Mode: {self.config.mode}")

        print(f"Number of features before RFE: {x_traindev.shape[1]}")

        # Run RFECV
        rfecv = RFECV(
            estimator=estimator,
            step=self.config.step,
            min_features_to_select=self.config.min_features_to_select,
            cv=cv,
            scoring=scoring_type,
            verbose=0,
            n_jobs=1,
            importance_getter=get_feature_importances,
        )

        rfecv.fit(x_traindev, y_traindev.squeeze(axis=1))  # type: ignore[arg-type]
        optimal_features = rfecv.n_features_
        selected_features = [feature_cols[i] for i in range(len(rfecv.support_)) if rfecv.support_[i]]
        selected_features = sorted(selected_features, key=lambda x: (len(x), sorted(x)))

        print("RFE completed")
        print(f"Optimal number of features: {optimal_features}")
        print(f"Selected features: {selected_features}")
        dev_score_cv = rfecv.cv_results_["mean_test_score"].max()
        print(f"Dev set performance: {dev_score_cv:.3f}")

        # Retrain best model on traindev with selected features
        best_estimator = copy.deepcopy(estimator)
        x_traindev_rfe = x_traindev[selected_features]
        best_estimator.fit(x_traindev_rfe, y_traindev.squeeze(axis=1))
        test_score_refit = get_score_from_model(
            best_estimator,
            data_test,
            selected_features,
            study_context.target_metric,
            target_assignments,
            positive_class=positive_class,
        )
        print(f"Test set (refit) performance: {test_score_refit:.3f}")

        # GridSearch + retrain on selected features
        grid_search.fit(x_traindev_rfe, y_traindev.squeeze(axis=1))  # type: ignore[arg-type]
        best_gs_parameters = grid_search.best_params_
        best_gs_estimator = grid_search.best_estimator_
        best_gs_estimator.fit(x_traindev_rfe, y_traindev.squeeze(axis=1))
        test_score_gsrefit = get_score_from_model(
            best_gs_estimator,
            data_test,
            selected_features,
            study_context.target_metric,
            target_assignments,
            positive_class=positive_class,
        )
        print(f"Test set (gridsearch+refit) performance: {test_score_gsrefit:.3f}")

        # Feature importances
        fi_df = pd.DataFrame(
            {
                "feature": selected_features,
                "importance": best_gs_estimator.feature_importances_,
            }
        ).sort_values(by="importance", ascending=False)

        # Store fitted state
        self.selected_features_ = selected_features
        self.feature_importances_ = {"internal": fi_df}

        # Build standard scores DataFrame
        scores = pd.DataFrame(
            [
                {
                    "result_type": ResultType.BEST,
                    "metric": study_context.target_metric,
                    "partition": "dev",
                    "aggregation": "avg",
                    "fold": None,
                    "value": dev_score_cv,
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
        feature_importances["training_id"] = "rfe"
        feature_importances["result_type"] = ResultType.BEST

        # Save results to JSON
        results_data = {
            "Dev score start": best_cv_score,
            "best_params": best_gs_parameters,
            "optimal_features": int(optimal_features),
            "selected_features": selected_features,
            "Dev set performance": dev_score_cv,
            "Test set (refit) performance": test_score_refit,
            "Test set (gs+refit) performance": test_score_gsrefit,
        }
        with (path_results / "results.json").open("w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=4)

        return {
            ResultType.BEST: ModuleResult(
                result_type=ResultType.BEST,
                module=self.config.module,
                selected_features=selected_features,
                scores=scores,
                feature_importances=feature_importances,
            )
        }
