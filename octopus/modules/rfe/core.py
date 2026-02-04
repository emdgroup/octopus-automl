# type: ignore

"""Rfe core."""

import copy
import json

import pandas as pd
from attrs import define
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from octopus.metrics import Metrics
from octopus.metrics.utils import get_score_from_model
from octopus.models import Models
from octopus.modules.base import ModuleBaseCore
from octopus.modules.rfe.module import Rfe
from octopus.results import ModuleResults

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

# for quick result
# param_grid = {
#    'iterations': [100, 200],
#    'depth': [4, 6],
#    'learning_rate': [0.01, 0.1],
#    'l2_leaf_reg': [1, 3]
# }


def get_feature_importances(estimator):
    """Set feature importance based on mode."""
    if hasattr(estimator, "best_estimator_"):
        return estimator.best_estimator_.feature_importances_
    return estimator.feature_importances_


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
# - Notes:
#   + For RFE it is important to have uncorrelated features (ROC, MRMR module)
#   + Advantage RFE over SFS - fewer retrainings, less overfitting
#   + Disadvantage RFE over SFS - need to rely on feature importances
# - General topics:
#   + (0) add scores to results
#   + (1) put scorer_string_inventory in central place
# - Question:  RFE - better test results than octo (datasplit difference?)
# - Next Steps (Improvements)
#   + separate second RFE module!
#   + better datasplit (stratification + groups)
#   + based on bag (inherits datasplit + feature importances)?
#   + model retraining after n removals
#   + find best model using stats test, smallest number of features with
#     no significant difference to max performance!!
#   + replace gridsearch with optuna or randomsearch
#   + Performance: permutation fi on dev! requires ROC, MRMR
#   + PFI: use large number of repeats, adjustable
#   + automatically remove not used features (fi == 0)
#   + Write own RFE code to have more options
#   + intelligent feature removal considering groups
#   + Efficiency option: new approach on how many features to eliminate.
#     See autogluon issue. Add random features (3-5) and remove all features below worst
#     random feature. See autogluon
#   + mode2: only one training per reduction and not for every experiment??


@define
class RfeCore(ModuleBaseCore[Rfe]):
    """RFE Module."""

    def run_experiment(self):
        """Run RFE module on experiment."""
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

        cv: int | StratifiedKFold
        stratification_column = self.experiment.stratification_column
        if stratification_column:
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

        # Mode selection
        if self.config.mode == "Mode1":
            # RFE with the trained model
            estimator = best_model
        elif self.config.mode == "Mode2":
            # RFE with hyperparameter optimization at each step
            estimator = grid_search
        else:
            raise ValueError(f"Unsupported Mode: {self.config.mode}")

        print(f"Number of features before RFE: {self.x_traindev.shape[1]}")

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

        rfecv.fit(self.x_traindev, self.y_traindev.squeeze(axis=1))
        optimal_features = rfecv.n_features_
        selected_features = [self.feature_cols[i] for i in range(len(rfecv.support_)) if rfecv.support_[i]]
        self.experiment.selected_features = sorted(selected_features, key=lambda x: (len(x), sorted(x)))

        print("RFE completed")
        # print(f"CV Results: {rfecv.cv_results_}")
        print(f"Optimal number of features: {optimal_features}")
        print(f"Selected features: {self.experiment.selected_features}")
        dev_score_cv = rfecv.cv_results_["mean_test_score"].max()
        print(f"Dev set performance: {dev_score_cv:.3f}")

        # Report performance on test set
        # test_score_cv = rfecv.score(self.x_test, self.y_test)
        # print(f"Test set (cv) performance: {test_score_cv:.3f}")

        # retrain best model on x_traindev
        best_estimator = copy.deepcopy(estimator)
        x_traindev_rfe = self.x_traindev[self.experiment.selected_features]
        # refit on selected features
        best_estimator.fit(x_traindev_rfe, self.y_traindev.squeeze(axis=1))
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
        grid_search.fit(x_traindev_rfe, self.y_traindev.squeeze(axis=1))
        best_gs_parameters = grid_search.best_params_
        best_gs_estimator = grid_search.best_estimator_
        best_gs_estimator.fit(x_traindev_rfe, self.y_traindev.squeeze(axis=1))  # refit
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
        scores["dev_avg"] = rfecv.cv_results_["mean_test_score"].max()
        # scores["test_avg"] = test_score_cv
        scores["test_refit"] = test_score_refit
        scores["test_gsrefit"] = test_score_gsrefit

        # save results to experiment
        self.experiment.results["Rfe"] = ModuleResults(
            id="rfe",
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
            "optimal_features": int(optimal_features),
            "selected_features": self.experiment.selected_features,
            "Dev set performance": dev_score_cv,
            # "Test set (cv) performance": test_score_cv,
            "Test set (refit) performance": test_score_refit,
            "Test set (gs+refit) performance": test_score_gsrefit,
        }
        with (self.path_results / "results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        return self.experiment
