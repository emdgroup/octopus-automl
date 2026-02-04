# type: ignore

"""SFS Core (sequential feature selection)."""

import copy
import json
import warnings

import pandas as pd
from attrs import define
from sklearn.model_selection import BaseCrossValidator, GridSearchCV, StratifiedKFold

from octopus.metrics import Metrics
from octopus.metrics.utils import get_score_from_model
from octopus.models import Models
from octopus.modules.base import ModuleBaseCore
from octopus.modules.sfs.module import Sfs
from octopus.results import ModuleResults

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
            #'random_strength': [2, 5, 7, 10],
            #'rsm': [0.1, 0.5, 1],
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
            "max_depth": [3, 6, 10],  # [2, 10, 20, 32],
            "min_samples_split": [2, 25, 50, 100],
            # "min_samples_leaf": [1, 15, 30, 50],
            # "max_features": [0.1, 0.5, 1],
            "n_estimators": [500],  # [100, 250, 500],
        }
    return param_grid


# TOBEDONE/IDEAS:
# - (2) add scores to results
# - it would be nice to stop after a certain feature reduction, then
#   relearn the model parameters and the start again. k_features parameter!
# - k_features = "parsemonious", check this out
# - put scorer_string_inventory in central place
# - try verbose = 1, useful?
# - use cv object, stratifiedKfold


@define
class SfsCore(ModuleBaseCore[Sfs]):
    """SFS Module."""

    def run_experiment(self):
        """Run SFS module on experiment."""
        from octopus._optional.sfs import SFS  # noqa: PLC0415

        # run experiment and return updated experiment object
        # Configuration, define default model
        if self.experiment.ml_type == "classification":
            default_model = "CatBoostClassifier"
        elif self.experiment.ml_type == "regression":
            default_model = "CatBoostRegressor"
        else:
            raise ValueError(f"{self.experiment.ml_type} not supported")

        model_type = self.config.model
        if model_type == "":
            model_type = default_model

        if model_type not in supported_models:
            raise ValueError(f"{model_type} not supported")
        print("Model used:", model_type)

        # set up model and scoring type
        model = Models.get_instance(model_type, {"random_state": 42})

        # Get scorer string from metrics inventory
        metric = Metrics.get_instance(self.target_metric)
        scoring_type = metric.scorer_string

        cv: int | BaseCrossValidator
        stratification_col = self.experiment.stratification_col
        if stratification_col:
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
        grid_search.fit(self.x_traindev, self.y_traindev.squeeze(axis=1))
        best_model = grid_search.best_estimator_
        best_cv_score = grid_search.best_score_
        best_params = grid_search.best_params_

        # Report performance
        print(f"Dev start performance: {best_cv_score:.3f}")
        print(f"Best params: {best_params}")

        print(f"Number of features before SFS: {self.x_traindev.shape[1]}")

        # Select type of SFS
        if self.config.sfs_type == "forward":
            forward = True
            floating = False
        elif self.config.sfs_type == "backward":
            forward = False
            floating = False
        elif self.config.sfs_type == "floating_forward":
            forward = True
            floating = True
        elif self.config.sfs_type == "floating_backward":
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

        sfs.fit(self.x_traindev, self.y_traindev.squeeze(axis=1))
        n_optimal_features = len(sfs.k_feature_idx_)
        self.experiment.selected_features = list(sfs.k_feature_names_)

        print("SFS completed")
        # print(sfs.subsets_)
        print(f"Optimal number of features: {n_optimal_features}")
        print(f"Selected features: {self.experiment.selected_features}")
        print(f"Dev set performance: {sfs.k_score_:.3f}")

        # Report performance on test set
        best_estimator = copy.deepcopy(best_model)
        x_traindev_sfs = sfs.transform(self.x_traindev)
        # refit on selected features
        best_estimator.fit(x_traindev_sfs, self.y_traindev.squeeze(axis=1))
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
        grid_search.fit(x_traindev_sfs, self.y_traindev.squeeze(axis=1))
        best_gs_parameters = grid_search.best_params_
        best_gs_estimator = grid_search.best_estimator_
        best_gs_estimator.fit(x_traindev_sfs, self.y_traindev.squeeze(axis=1))  # refit
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
        scores["dev_avg"] = sfs.k_score_
        scores["test_refit"] = test_score_refit
        scores["test_gsrefit"] = test_score_gsrefit

        # save results to experiment
        self.experiment.results["Sfs"] = ModuleResults(
            id="SFS",
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
            "Dev set performance": sfs.k_score_,
            "Test set (refit) performance": test_score_refit,
            "Test set (gs+refit) performance": test_score_gsrefit,
        }
        with (self.path_results / "results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        return self.experiment
