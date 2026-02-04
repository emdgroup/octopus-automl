# type: ignore

"""Boruta Core."""

import copy
import json
import warnings

import numpy as np
import pandas as pd
from attrs import define
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from octopus.metrics import Metrics
from octopus.metrics.utils import get_score_from_model
from octopus.models import Models
from octopus.modules.base import ModuleBaseCore
from octopus.modules.boruta.module import Boruta
from octopus.results import ModuleResults

# Ignore all Warnings
warnings.filterwarnings("ignore")

# Tree Based models
supported_models = {
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "XGBClassifier",
    "XGBRegressor",
}


def get_param_grid(model_type):
    """Hyperparameter grid initialization."""
    if model_type in ("XGBClassifier", "XGBRegressor"):
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
            "max_depth": [3, 6, 10],  # [2, 10, 20, 32],
            "min_samples_split": [2, 25, 50, 100],
            "min_samples_leaf": [1, 15, 30, 50],
            "max_features": [0.1, 0.5, 1],
            "n_estimators": [500],  # [100, 250, 500],
        }
    return param_grid


# TOBEDONE:
# - implement Boruta using https://github.com/scikit-learn-contrib/boruta_py
# - gridsearch with estimator=boruta with params (perc,alpha,max_iter etc) optimiz ?


@define
class BorutaCore(ModuleBaseCore[Boruta]):
    """Boruta Module."""

    def run_experiment(self):
        """Run Boruta module on experiment."""
        from octopus._optional.burota import BorutaPy  # noqa: PLC0415

        # run experiment and return updated experiment object
        # Configuration, define default model
        if self.experiment.ml_type == "classification":
            default_model = "RandomForestClassifier"
        elif self.experiment.ml_type == "regression":
            default_model = "RandomForestRegressor"
        else:
            raise ValueError(f"{self.experiment.ml_type} not supported")

        model_type = self.config.model
        if model_type == "":
            model_type = default_model

        if model_type not in supported_models:
            raise ValueError(f"{model_type} not supported")
        print("Model used:", model_type)

        # set up model and scoring type
        model = Models.get_instance(model_type, {"random_state": 42, "verbose": False})
        # Get scorer string from metrics
        metric = Metrics.get_instance(self.target_metric)
        scoring_type = metric.scorer_string

        cv: int | StratifiedKFold
        stratification_column = self.experiment.stratification_column
        if stratification_column:
            cv = StratifiedKFold(n_splits=self.config.cv, shuffle=True, random_state=42)
        else:
            cv = self.config.cv

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
        grid_search.fit(self.x_traindev, self.y_traindev.squeeze(axis=1))
        best_model = grid_search.best_estimator_
        best_cv_score = grid_search.best_score_
        best_params = grid_search.best_params_

        # Report performance
        print(f"Dev start performance: {best_cv_score:.3f}")
        print(f"Best params: {best_params}")

        print(f"Initial number of features: {self.x_traindev.shape[1]}")

        boruta = BorutaPy(
            estimator=model,
            n_estimators="auto",
            perc=self.config.perc,
            alpha=self.config.alpha,
            random_state=42,
            verbose=0,
        )

        boruta.fit(self.x_traindev, self.y_traindev.squeeze(axis=1))

        print("Feature Selection completed")
        self.experiment.selected_features = [
            self.feature_cols[i] for i in range(len(boruta.support_)) if boruta.support_[i]
        ]
        n_optimal_features = len(self.experiment.selected_features)

        print(f"Optimal number of features: {n_optimal_features}")
        print(f"Selected features: {self.experiment.selected_features}")
        # print(boruta.ranking_)

        # Report performance on dev and test set
        best_estimator = copy.deepcopy(best_model)
        # x_traindev_filtered = boruta.transform(self.x_traindev, return_df=True)
        # incompatible with boruta_py version 0.3
        x_traindev_filtered = boruta.transform(self.x_traindev)
        cv_score_dev = cross_val_score(
            best_model,
            x_traindev_filtered,
            self.y_traindev,
            scoring=scoring_type,
            cv=cv,
        )
        dev_score_cv = np.mean(cv_score_dev)
        print(f"Dev set (cv) performance : {dev_score_cv:.3f}")

        # refit on selected features
        best_estimator.fit(x_traindev_filtered, self.y_traindev.squeeze(axis=1))
        test_score_refit = get_score_from_model(
            best_estimator,
            self.data_test,
            self.experiment.selected_features,
            self.target_metric,
            self.target_assignments,
            positive_class=self.experiment.positive_class,
        )
        print(f"Test set (refit) performance: {test_score_refit:.3f}")

        # gridsearch + retrain best model on x_traindev
        grid_search.fit(x_traindev_filtered, self.y_traindev.squeeze(axis=1))
        best_gs_parameters = grid_search.best_params_
        best_gs_estimator = grid_search.best_estimator_
        # refit
        best_gs_estimator.fit(x_traindev_filtered, self.y_traindev.squeeze(axis=1))
        test_score_gsrefit = get_score_from_model(
            best_gs_estimator,
            self.data_test,
            self.experiment.selected_features,
            self.target_metric,
            self.target_assignments,
            positive_class=self.experiment.positive_class,
        )
        print(f"Test set (gridsearch+refit) performance: {test_score_gsrefit:.3f}")

        # feature importances
        fi_df = pd.DataFrame(
            {
                "feature": self.experiment.selected_features,
                "importances": best_gs_estimator.feature_importances_,
            }
        ).sort_values(by="importances", ascending=False)
        # print(fi_df)

        # scores
        scores = {}
        scores["dev_avg"] = dev_score_cv
        scores["test_refit"] = test_score_refit
        scores["test_gsrefit"] = test_score_gsrefit

        # save results to experiment
        self.experiment.results["Boruta"] = ModuleResults(
            id="boruta",
            experiment_id=self.experiment.experiment_id,
            task_id=self.experiment.task_id,
            model=best_gs_estimator,
            scores=scores,
            feature_importances={
                "internal": fi_df,
            },
            selected_features=self.experiment.selected_features,
        )

        # Save results to JSON
        results = {
            "Dev score start": best_cv_score,
            "best_params": best_gs_parameters,
            "optimal_features": int(n_optimal_features),
            "selected_features": self.experiment.selected_features,
            "Dev set performance": dev_score_cv,
            "Test set (refit) performance": test_score_refit,
            "Test set (gs+refit) performance": test_score_gsrefit,
        }
        with (self.path_results / "results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        return self.experiment
