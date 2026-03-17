"""EFS execution module."""

from __future__ import annotations

import copy
import itertools
import json
import random
from collections import Counter
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from attrs import Factory, define, field
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_predict

from octopus.metrics import Metrics
from octopus.models import Models
from octopus.modules import ModuleExecution, ModuleResult, StudyContext
from octopus.types import DataPartition, FIResultLabel, MLType, ModelName, ResultType

if TYPE_CHECKING:
    from upath import UPath

    from octopus.modules import Efs  # noqa: F401

supported_models = {
    ModelName.CatBoostClassifier,
    ModelName.CatBoostRegressor,
    ModelName.RandomForestClassifier,
    ModelName.RandomForestRegressor,
    ModelName.ExtraTreesClassifier,
    ModelName.ExtraTreesRegressor,
    ModelName.XGBClassifier,
    ModelName.XGBRegressor,
}


def get_param_grid(model_type: ModelName):
    """Hyperparameter grid initialization."""
    if model_type in (ModelName.CatBoostClassifier, ModelName.CatBoostRegressor):
        param_grid = {
            "learning_rate": [0.03],
            "iterations": [500],
        }
    elif model_type in (ModelName.XGBClassifier, ModelName.XGBRegressor):
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
        }
    return param_grid


@define
class EfsModule(ModuleExecution["Efs"]):
    """EFS execution module. Created by Efs.create_module()."""

    # Internal state (set during fit)
    model_table_: pd.DataFrame = field(init=False, default=Factory(lambda: pd.DataFrame()))
    """Table of models trained on different feature subsets."""

    scan_table_: pd.DataFrame = field(init=False, default=Factory(lambda: pd.DataFrame()))
    """Table of ensemble scan results."""

    optimized_ensemble_: dict = field(init=False, default=Factory(dict))
    """Optimized ensemble of models."""

    def fit(
        self,
        *,
        data_traindev: pd.DataFrame,
        feature_cols: list[str],
        study_context: StudyContext,
        results_dir: UPath,
        **kwargs,
    ) -> dict[ResultType, ModuleResult]:
        """Fit EFS module by creating and optimizing an ensemble of models."""
        x_traindev = data_traindev[feature_cols]
        y_traindev = data_traindev[list(study_context.target_assignments.values())]
        row_traindev = data_traindev[study_context.row_id_col]

        metric_input: Literal["predictions", "probabilities"] = (
            "probabilities" if study_context.target_metric in ["AUCROC", "LOGLOSS"] else "predictions"
        )
        direction = Metrics.get_direction(study_context.target_metric)

        self._create_modeltable(
            study_context, x_traindev, y_traindev, row_traindev, feature_cols, metric_input, direction
        )
        print(self.model_table_)

        self._create_scantable(study_context, y_traindev, metric_input, direction)

        self._ensemble_optimization(study_context, y_traindev, metric_input, direction)
        print(self.optimized_ensemble_)

        module_result = self._generate_results()

        print("EFS completed")

        # Save results to JSON
        with (results_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump({}, f, indent=4)

        # Extract selected features (computed in _generate_results)
        selected_features = sorted(
            pd.concat([self.model_table_.iloc[i]["used_features"] for i in self.optimized_ensemble_])
            .value_counts()
            .index.tolist()
        )

        # Build flat feature_importances DataFrame
        fi_dfs = []
        raw_fi = module_result["feature_importances"]
        for fi_key, fi_df in raw_fi.items():
            if isinstance(fi_df, pd.DataFrame) and not fi_df.empty:
                temp = fi_df.copy()
                # Rename 'counts' to 'importance' for standard schema
                if "counts" in temp.columns and "importance" not in temp.columns:
                    temp = temp.rename(columns={"counts": "importance"})
                temp = temp[["feature", "importance"]].copy()
                if fi_key == "Efs_counts":
                    temp["fi_method"] = FIResultLabel.COUNTS
                    temp["fi_dataset"] = DataPartition.TRAIN
                elif fi_key == "Efs_counts_relative":
                    temp["fi_method"] = FIResultLabel.COUNTS_RELATIVE
                    temp["fi_dataset"] = DataPartition.TRAIN
                temp["training_id"] = "efs"
                temp["result_type"] = ResultType.BEST
                fi_dfs.append(temp)
        feature_importances = pd.concat(fi_dfs, ignore_index=True) if fi_dfs else pd.DataFrame()

        return {
            ResultType.BEST: ModuleResult(
                result_type=ResultType.BEST,
                module=self.config.module,
                selected_features=selected_features,
                feature_importances=feature_importances,
            )
        }

    def _create_modeltable(
        self,
        study_context: StudyContext,
        x_traindev: pd.DataFrame,
        y_traindev: pd.DataFrame,
        row_traindev: pd.Series,
        feature_cols: list[str],
        metric_input: Literal["predictions", "probabilities"],
        direction: Literal["maximize", "minimize"],
    ):
        """Create model table."""
        print("Creating model table.")
        # set seeds
        np.random.seed(0)
        random.seed(0)

        # Configuration, define default model
        if study_context.ml_type in (MLType.BINARY, MLType.MULTICLASS):
            default_model = ModelName.CatBoostClassifier
        elif study_context.ml_type == MLType.REGRESSION:
            default_model = ModelName.CatBoostRegressor
        else:
            raise ValueError(f"{study_context.ml_type} not supported")

        model_type = ModelName(self.config.model) if self.config.model else default_model

        if model_type not in supported_models:
            raise ValueError(f"{model_type} not supported")
        print("Model used:", model_type)

        # set up model and scoring type
        model = Models.get_instance(model_type, {"random_state": 42})
        # Get scorer string from metrics inventory
        metric = Metrics.get_instance(study_context.target_metric)
        scoring_type = metric.scorer_string

        # needs general improvements (consider groups and stratification column)
        cv: KFold | StratifiedKFold
        if study_context.stratification_col:
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
            subset = random.sample(feature_cols, self.config.subset_size)
            subsets.append(subset)

        # (A) create model table
        # train (gridsearch) on all subsets
        df_lst = []
        for i, subset in enumerate(subsets):
            # Perform Grid Search and Cross-Validation
            x = x_traindev[subset]
            y = y_traindev.squeeze(axis=1)
            row_ids = row_traindev
            grid_search.fit(x, y)  # type: ignore[arg-type]
            best_model = grid_search.best_estimator_
            best_cv_score = grid_search.best_score_
            print(f"Subset {i}, best_cv_score: {best_cv_score:.4f}")

            # predictions
            if study_context.ml_type in (MLType.BINARY, MLType.MULTICLASS):
                cv_preds_df = pd.DataFrame()
                cv_preds_df[study_context.row_id_col] = row_ids
                cv_preds_df["predictions"] = cross_val_predict(best_model, x, y, cv=cv, method="predict")  # type: ignore[arg-type]
                cv_preds_df["probabilities"] = cross_val_predict(best_model, x, y, cv=cv, method="predict_proba")[:, 1]  # type: ignore[arg-type] # binary only
            elif study_context.ml_type == MLType.REGRESSION:
                cv_preds_df = pd.DataFrame()
                cv_preds_df[study_context.row_id_col] = row_ids
                cv_preds_df["predictions"] = cross_val_predict(best_model, x, y, cv=cv, method="predict")  # type: ignore[arg-type]
                cv_preds_df["probabilities"] = np.nan

            # actual used features
            feature_importances = best_model.feature_importances_
            feature_importance_df = pd.DataFrame({"feature": subset, "importance": feature_importances})

            # ensemble metric
            metric = Metrics.get_instance(study_context.target_metric)
            if metric_input == "probabilities":
                best_ensel_performance = metric.calculate(y, cv_preds_df["probabilities"])  # type: ignore[arg-type]
            else:
                best_ensel_performance = metric.calculate(y, cv_preds_df["predictions"])  # type: ignore[arg-type]
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

        self.model_table_ = pd.concat(df_lst, axis=1).T
        self.model_table_["id"] = self.model_table_["id"].astype(int)

        # order of table is important, depending on metric,
        # (a) direction (b) dev_pool_soft or dev_pool_hard
        ascending = direction != "maximize"

        self.model_table_ = self.model_table_.sort_values(by="performance", ascending=ascending).reset_index(drop=True)

    def _ensemble_models(
        self,
        study_context: StudyContext,
        y_traindev: pd.DataFrame,
        metric_input: Literal["predictions", "probabilities"],
        model_ids,
    ) -> float:
        """Ensemble predictions of models in model_table."""
        # Filter the model_table to include only the specified IDs
        filtered_df = self.model_table_[self.model_table_["id"].isin(model_ids)]
        # Extract the DataFrames from the "predict_df" column
        df_lst = filtered_df["predict_df"].tolist()
        # Concatenate all the extracted DataFrames into a single DataFrame
        groupby_df = pd.concat(df_lst, ignore_index=True).groupby(by=study_context.row_id_col).mean()

        # calculate ensemble performance
        # TODO: needs improvement!!
        model_predictions = (
            groupby_df["predictions"].round().astype(int)
            if study_context.target_metric in ["ACC", "ACCBAL"]
            else groupby_df["predictions"]
        )

        metric = Metrics.get_instance(study_context.target_metric)
        y = y_traindev.squeeze(axis=1)
        ensel_performance = (
            metric.calculate(y, model_predictions)  # type: ignore[arg-type]
            if metric_input == "predictions"
            else metric.calculate(y, groupby_df["probabilities"])  # type: ignore[arg-type]
        )

        return ensel_performance

    def _create_scantable(
        self,
        study_context: StudyContext,
        y_traindev: pd.DataFrame,
        metric_input: Literal["predictions", "probabilities"],
        direction: Literal["maximize", "minimize"],
    ):
        """Perform ensemble scan."""
        # (B) perform ensemble scan, hillclimb
        self.scan_table_ = pd.DataFrame(
            columns=[
                "#models",
                "performance",
            ]
        )

        for i in range(len(self.model_table_)):
            model_ids = self.model_table_.iloc[: (i + 1)]["id"].tolist()
            self.scan_table_.loc[i] = [
                i,
                self._ensemble_models(study_context, y_traindev, metric_input, model_ids),
            ]

        if direction == "maximize":
            n_best_models = self.scan_table_.loc[self.scan_table_["performance"].idxmax()]["#models"]
            best_performance = self.scan_table_.loc[self.scan_table_["performance"].idxmax()]["performance"]
        else:  # minimize
            n_best_models = self.scan_table_.loc[self.scan_table_["performance"].idxmin()]["#models"]
            best_performance = self.scan_table_.loc[self.scan_table_["performance"].idxmin()]["performance"]
        self.scan_table_["#models"] = self.scan_table_["#models"].astype(int)
        print("Scan table:", self.scan_table_)
        print(f"Best performance: {best_performance} with base model and {n_best_models} additional models")
        print()

    def _ensemble_optimization(
        self,
        study_context: StudyContext,
        y_traindev: pd.DataFrame,
        metric_input: Literal["predictions", "probabilities"],
        direction: Literal["maximize", "minimize"],
    ):
        """Ensembling optimization with replacement."""
        # we start with an best N models example derived from self.scan_table_,
        # assuming that is sorted correctly
        best_performance = (
            self.scan_table_["performance"].max() if direction == "maximize" else self.scan_table_["performance"].min()
        )
        # get the last index with best performance
        best_rows = self.scan_table_[self.scan_table_["performance"] == best_performance]
        last_best_index = best_rows.index[-1]
        start_n = int(last_best_index) + 1
        print("Ensemble scan, number of included best models: ", start_n)

        # startn_bags dict with path as key and repeats=1 as value
        escan_ensemble = {}
        for _, row in self.model_table_.head(start_n).iterrows():
            escan_ensemble[row["id"]] = 1

        # ensemble_optimization, reference score
        # we start with the bags found in ensemble scan
        results: list[tuple[str | int, float, list[int]]] = []
        start_model_ids = list(escan_ensemble.keys())
        print("start_model_ids", start_model_ids)
        print()
        start_perf = self._ensemble_models(study_context, y_traindev, metric_input, start_model_ids)

        print("Ensemble optimization")
        print("Start performance:", start_perf)
        # record start performance
        results.append(("ensemble scan", start_perf, copy.deepcopy(start_model_ids)))

        # optimization
        model_ensemble = copy.deepcopy(start_model_ids)
        best_global = copy.deepcopy(start_perf)

        # optimization limited to best max_n_models
        best_n_models = self.model_table_["id"].tolist()[: self.config.max_n_models]

        for i in range(self.config.max_n_iterations):
            df = pd.DataFrame(columns=["model", "performance"])
            # test if any additional model improves performance
            for model in best_n_models:
                model_lst = copy.deepcopy(model_ensemble)
                model_lst.append(model)
                perf = self._ensemble_models(study_context, y_traindev, metric_input, model_lst)
                df.loc[len(df)] = [model, perf]
            df["model"] = df["model"].astype(int)

            if direction == "maximize":
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
            results.append((best_model, best_performance, copy.deepcopy(model_ensemble)))

        # store optimization results
        results_df = pd.DataFrame(results, columns=["model", "performance", "models_lst"])
        self.optimized_ensemble_ = dict(Counter(results_df.iloc[-1]["models_lst"]))
        print("Ensemble selection completed.")

    def _generate_results(self) -> dict:
        """Generate results."""
        feature_lst = []
        for key, value in self.optimized_ensemble_.items():
            row = self.model_table_[self.model_table_["id"] == key]
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
        n_models_used = sum(self.optimized_ensemble_.values())
        feature_counts_relative_df = feature_counts_df.copy()
        feature_counts_relative_df["counts"] = feature_counts_relative_df["counts"] / n_models_used

        print(feature_counts_df.head(50))
        print(feature_counts_relative_df.head(50))

        # Create and return module results
        return {
            "scores": {},
            "predictions": {},
            "feature_importances": {
                "Efs_counts": feature_counts_df,
                "Efs_counts_relative": feature_counts_relative_df,
            },
        }
