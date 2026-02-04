# type: ignore

"""Efs core."""

import copy
import itertools
import json
import random
from collections import Counter

import numpy as np
import pandas as pd
from attrs import Factory, define, field, validators
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_predict,
)

from octopus.metrics import Metrics
from octopus.models import Models
from octopus.modules.base import ModuleBaseCore
from octopus.modules.efs.module import Efs
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


def get_param_grid(model_type):
    """Hyperparameter grid initialization."""
    if model_type in ("CatBoostClassifier", "CatBoostRegressor"):
        param_grid = {
            "learning_rate": [0.03],
            # "learning_rate": [0.001, 0.03, 0.1],
            # "depth": [3, 6, 8, 10],
            # "l2_leaf_reg": [2, 5, 7, 10],
            # 'random_strength': [2, 5, 7, 10],
            # 'rsm': [0.1, 0.5, 1],
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
            "max_depth": [3, 6],
            # "min_samples_split": [2, 25, 50, 100],
            # "min_samples_leaf": [1, 15, 30, 50],
            # "max_features": [0.1, 0.5, 1],
            # "n_estimators": [100, 250, 500],
        }
    return param_grid


# TOBEDONE:
# - (0) RFE(maybe EFS) - better test results than octo (datasplit difference?)
# - provide:
#   + random features 5
#   + best parameters based on traindev-size
# - results:
#   + define selected features - how exactly? random features?
# - issue: ACC and BALACC need integer pooling values! - better solution needed
# - should be done with a proper datasplit (see octo, stratification + groups)
# - should be done with optuna
# - make it work with timetoevent


@define
class EfsCore(ModuleBaseCore[Efs]):
    """EFS Module."""

    # Module-specific attributes
    model_table: pd.DataFrame = field(
        init=False,
        default=pd.DataFrame(),
        validator=[validators.instance_of(pd.DataFrame)],
    )
    scan_table: pd.DataFrame = field(
        init=False,
        default=pd.DataFrame(),
        validator=[validators.instance_of(pd.DataFrame)],
    )
    optimized_ensemble: dict = field(default=Factory(dict), validator=[validators.instance_of(dict)])

    @property
    def row_ids_traindev(self) -> pd.Series:
        """Row IDs for traindev."""
        return self.experiment.row_traindev

    @property
    def max_n_iterations(self) -> int:
        """Maximum iterations for ensemble optimization."""
        return self.config.max_n_iterations

    @property
    def max_n_models(self) -> int:
        """Maximum number of models using during optimization."""
        return self.config.max_n_models

    @property
    def metric_input(self) -> str:
        """Metric input type."""
        if self.target_metric in ["AUCROC", "LOGLOSS"]:
            return "probabilities"
        else:
            return "predictions"

    @property
    def direction(self) -> str:
        """Optuna direction."""
        return Metrics.get_direction(self.target_metric)

    def run_experiment(self):
        """Run EFS module on experiment."""
        self._create_modeltable()
        print(self.model_table)

        self._create_scantable()

        self._ensemble_optimization()
        print(self.optimized_ensemble)

        self._generate_results()

        print("EFS completed")

        # update selected features in experiment

        # Report performance on test set

        # Save results to JSON
        with (self.path_results / "results.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    # "best_cv_score": best_cv_score,
                    # "best_params": best_params,
                    # "optimal_features": int(optimal_features),
                    # "selected_features": self.experiment.selected_features,
                    # "Best Mean CV Score": max(rfecv.cv_results_["mean_test_score"]),
                    # "Dev set performance": test_score,
                },
                f,
                indent=4,
            )

        return self.experiment

    def _create_modeltable(self):
        """Create model table."""
        print("Creating model table.")
        # set seeds
        np.random.seed(0)
        random.seed(0)

        # print()
        # print("features: ", self.feature_cols)

        # Configuration, define default model
        if self.ml_type == "classification":
            default_model = "CatBoostClassifier"
        elif self.ml_type == "regression":
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

        # needs general improvements (consider groups and stratification column)
        cv: KFold | StratifiedKFold
        stratification_column = self.experiment.stratification_column
        if stratification_column:
            cv = StratifiedKFold(n_splits=self.config.cv, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=self.config.cv, shuffle=True, random_state=42)

        # Silence catboost output, stop it writing files
        if model_type == default_model:
            model.set_params(verbose=False)
            model.set_params(allow_writing_files=False)

        # Define GridSearch for HPO
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=get_param_grid(model_type),
            cv=cv,
            scoring=scoring_type,
            n_jobs=1,
        )

        # Create features subsets

        subsets = []
        for _ in range(self.config.n_subsets):
            subset = random.sample(self.feature_cols, self.config.subset_size)
            subsets.append(subset)

        # (A) create model table
        # train (gridsearch) on all subsets
        df_lst = []
        for i, subset in enumerate(subsets):
            # Perform Grid Search and Cross-Validation
            # print("subset:", subset)
            x = self.x_traindev[subset]
            y = self.y_traindev.squeeze(axis=1)
            row_ids = self.row_ids_traindev
            grid_search.fit(x, y)
            best_model = grid_search.best_estimator_
            # best_params = grid_search.best_params_
            best_cv_score = grid_search.best_score_
            print(f"Subset {i}, best_cv_score: {best_cv_score:.4f}")
            # cv_score = cross_val_score(best_model, x, y, cv=cv, scoring=scoring_type)

            # predictions
            if self.ml_type == "classification":
                cv_preds_df = pd.DataFrame()
                cv_preds_df[self.row_column] = row_ids
                cv_preds_df["predictions"] = cross_val_predict(best_model, x, y, cv=cv, method="predict")
                cv_preds_df["probabilities"] = cross_val_predict(best_model, x, y, cv=cv, method="predict_proba")[
                    :, 1
                ]  # binary only
            elif self.ml_type == "regression":
                cv_preds_df = pd.DataFrame()
                cv_preds_df[self.row_column] = row_ids
                cv_preds_df["predictions"] = cross_val_predict(best_model, x, y, cv=cv, method="predict")
                cv_preds_df["probabilities"] = np.nan

            # actual used features
            feature_importances = best_model.feature_importances_
            feature_importance_df = pd.DataFrame({"feature": subset, "importance": feature_importances})

            # ensemble metric
            metric = Metrics.get_instance(self.target_metric)
            if self.metric_input == "probabilities":
                best_ensel_performance = metric.calculate(y, cv_preds_df["probabilities"])
            else:
                best_ensel_performance = metric.calculate(y, cv_preds_df["predictions"])
            print(f"Subset {i}, best ensemble performance: {best_ensel_performance:.4f}")

            # Select features with non-zero importance
            used_features = feature_importance_df[feature_importance_df["importance"] != 0]["feature"].tolist()
            print(f"Number of input features: {len(subset)}, n_used_features: {len(used_features)}")
            print()

            # store results
            s = pd.Series()
            s["id"] = i
            s["performance"] = best_ensel_performance  # reference
            s["score"] = best_cv_score  # only for info
            s["features"] = subset
            s["used_features"] = used_features
            s["n_used_features"] = len(used_features)
            s["predict_df"] = cv_preds_df
            df_lst.append(s)

        self.model_table = pd.concat(df_lst, axis=1).T
        self.model_table["id"] = self.model_table["id"].astype(int)

        # order of table is important, depending on metric,
        # (a) direction (b) dev_pool_soft or dev_pool_hard
        ascending = self.direction != "maximize"

        self.model_table = self.model_table.sort_values(by="performance", ascending=ascending).reset_index(drop=True)

    def _ensemble_models(self, model_ids) -> float:
        """Ensemble predictions of models in model_table."""
        # Filter the model_table to include only the specified IDs
        filtered_df = self.model_table[self.model_table["id"].isin(model_ids)]
        # Extract the DataFrames from the "predict_df" column
        df_lst = filtered_df["predict_df"].tolist()
        # Concatenate all the extracted DataFrames into a single DataFrame
        groupby_df = pd.concat(df_lst, ignore_index=True).groupby(by=self.row_column).mean()

        # print("-----------------------------------")
        # print("groupby_df", groupby_df.head(20)["predictions"])

        # calculate ensemble performance

        y = self.y_traindev.squeeze(axis=1)
        # print("y", y)
        # print("-----------------------------------")

        # TODO: needs improvement!!
        model_predictions = (
            groupby_df["predictions"].round().astype(int)
            if self.target_metric in ["ACC", "ACCBAL"]
            else groupby_df["predictions"]
        )

        metric = Metrics.get_instance(self.target_metric)
        ensel_performance = (
            metric.calculate(y, model_predictions)
            if self.metric_input == "predictions"
            else metric.calculate(y, groupby_df["probabilities"])
        )

        return ensel_performance

    def _create_scantable(self):
        """Perform ensemble scan."""
        # (B) perform ensemble scan, hillclimb
        self.scan_table = pd.DataFrame(
            columns=[
                "#models",
                "performance",
            ]
        )

        for i in range(len(self.model_table)):
            model_ids = self.model_table.iloc[: (i + 1)]["id"].tolist()
            # print("model_ids:", model_ids)
            self.scan_table.loc[i] = [
                i,
                self._ensemble_models(model_ids),
            ]

        if self.direction == "maximize":
            n_best_models = self.scan_table.loc[self.scan_table["performance"].idxmax()]["#models"]
            best_performance = self.scan_table.loc[self.scan_table["performance"].idxmax()]["performance"]
        else:  # minimize
            n_best_models = self.scan_table.loc[self.scan_table["performance"].idxmin()]["#models"]
            best_performance = self.scan_table.loc[self.scan_table["performance"].idxmin()]["performance"]
        self.scan_table["#models"] = self.scan_table["#models"].astype(int)
        print("Scan table:", self.scan_table)
        print(f"Best performance: {best_performance} with base model and {n_best_models} additional models")
        print()

    def _ensemble_optimization(self):
        """Ensembling optimization with replacement."""
        # we start with an best N models example derived from self.scan_table,
        # assuming that is sorted correctly
        best_performance = (
            self.scan_table["performance"].max()
            if self.direction == "maximize"
            else self.scan_table["performance"].min()
        )
        # get the last index with best performance
        best_rows = self.scan_table[self.scan_table["performance"] == best_performance]
        last_best_index = best_rows.index[-1]
        start_n = int(last_best_index) + 1
        print("Ensemble scan, number of included best models: ", start_n)

        # startn_bags dict with path as key and repeats=1 as value
        escan_ensemble = {}
        for _, row in self.model_table.head(start_n).iterrows():
            escan_ensemble[row["id"]] = 1

        # ensemble_optimmization, reference score
        # we start with the bags found in ensemble scan
        results_df = pd.DataFrame(columns=["model", "performance", "models_lst"])
        start_model_ids = list(escan_ensemble.keys())
        print("start_model_ids", start_model_ids)
        print()
        start_perf = self._ensemble_models(start_model_ids)

        print("Ensemble optimization")
        print("Start performance:", start_perf)
        # record start performance
        results_df.loc[len(results_df)] = [
            ["ensemble scan"],
            start_perf,
            copy.deepcopy(start_model_ids),
        ]

        # optimization
        model_ensemble = copy.deepcopy(start_model_ids)
        best_global = copy.deepcopy(start_perf)

        # optimization limited to best max_n_models
        best_n_models = self.model_table["id"].tolist()[: self.max_n_models]

        for i in range(self.max_n_iterations):
            df = pd.DataFrame(columns=["model", "performance"])
            # test if any additional model improves performance
            for model in best_n_models:
                model_lst = copy.deepcopy(model_ensemble)
                model_lst.append(model)
                perf = self._ensemble_models(model_lst)
                df.loc[len(df)] = [model, perf]
            df["model"] = df["model"].astype(int)

            if self.direction == "maximize":
                best_performance = df["performance"].max()
                best_rows = df[df["performance"] == best_performance]
                random_best_row = best_rows.sample(n=1, random_state=42)
                best_model = int(random_best_row["model"].values[0])
                if best_performance < best_global:
                    break  # stop ensembling
                else:
                    best_global = best_performance
                    print(f"iteration: {i}, performance: {best_performance}, best model: {best_model}")
            else:  # minimize
                best_performance = df["performance"].min()
                best_rows = df[df["performance"] == best_performance]
                random_best_row = best_rows.sample(n=1, random_state=42)
                best_model = int(random_best_row["model"].values[0])
                if best_performance > best_global:
                    break  # stop ensembling
                else:
                    best_global = best_performance
                    print(f"iteration: {i}, performance: {best_performance}, best model: {best_model}")

            # add best model to ensemble
            model_ensemble.append(best_model)

            # record results
            results_df.loc[len(results_df)] = [
                best_model,
                best_performance,
                copy.deepcopy(model_ensemble),
            ]

        # store optimization results
        self.optimized_ensemble = dict(Counter(results_df.iloc[-1]["models_lst"]))
        print("Ensemble selection completed.")

    def _generate_results(self):
        """Generate results."""
        feature_lst = []
        for key, value in self.optimized_ensemble.items():
            row = self.model_table[self.model_table["id"] == key]
            model_features = row["used_features"].tolist()
            model_features = list(itertools.chain.from_iterable(model_features))  # flatten list
            # replicate model features with model count
            feature_lst.extend(model_features * value)

        feature_counts = Counter(feature_lst)
        feature_counts_list = list(feature_counts.items())
        feature_counts_df = pd.DataFrame(feature_counts_list, columns=["feature", "counts"])
        # absolute feature counts
        feature_counts_df = feature_counts_df.sort_values(by="counts", ascending=False)

        # relative feature counts
        n_models_used = sum(self.optimized_ensemble.values())
        feature_counts_relative_df = feature_counts_df.copy()
        feature_counts_relative_df["counts"] = feature_counts_relative_df["counts"] / n_models_used

        print(feature_counts_df.head(50))
        print(feature_counts_relative_df.head(50))

        # save results to experiment
        self.experiment.results["Efs"] = ModuleResults(
            id="efs",
            experiment_id=self.experiment.experiment_id,
            task_id=self.experiment.task_id,
            # model=None,
            # scores=scores,
            feature_importances={
                "Efs_counts": feature_counts_df,
                "Efs_counts_relative": feature_counts_relative_df,
            },
            # selected_features=selected_features,
        )
