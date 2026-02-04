"""Octopus prediction."""

# import itertools

import json
import warnings
from typing import Any, Literal, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from attrs import Factory, define, field, validators
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
from upath import UPath

from octopus.experiment import OctoExperiment
from octopus.modules.utils import (
    ExperimentInfo,
    get_fi_group_permutation,
    get_fi_group_shap,
    get_fi_permutation,
    get_fi_shap,
)

# Suppress specific sklearn warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names",
)

# TOBEDONE
# (1) !calculate_fi(data_df)
#     on new data we can use self.predict_proba for calculating fis.
# (2) correltly label outputs of probabilities .predict_proba()
# (3) replace metrics with score, relevant for feature importances
# (4) Permutation importance on group of features
# (5) ? create OctoML.predict(), .calculate_fi()


@define
class OctoPredict:
    """OctoPredict."""

    study_path: UPath = field(validator=[validators.instance_of(UPath)], converter=lambda x: UPath(x))
    """Path to study."""

    task_id: int = field(default=-1, validator=[validators.instance_of(int)])
    """Task id."""

    results_key: str = field(default="best", validator=[validators.instance_of(str)])
    """Results key."""

    experiments: dict[int, ExperimentInfo] = field(init=False, validator=[validators.instance_of(dict)])
    """Dictionary containing model and corresponding test_dataset."""

    results: dict = field(default=Factory(dict), validator=[validators.instance_of(dict)])
    """Results."""

    @property
    def config(self) -> dict[str, Any]:
        """Study configuration."""
        with UPath(self.study_path / "config.json").open("r") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise TypeError("Loaded object is not a dict")

        return data

    @property
    def n_experiments(self) -> int:
        """Number of experiments."""
        return int(self.config["n_folds_outer"])

    @property
    def classes_(self):
        """Get classes from the first available experiment's model.

        This is needed for sklearn compatibility when OctoPredict is used as a model.
        All experiments should have the same classes, so returning from the first is sufficient.
        """
        if self.experiments:
            first_experiment = next(iter(self.experiments.values()))
            if hasattr(first_experiment.model, "classes_"):
                return first_experiment.model.classes_
        return None

    def __attrs_post_init__(self):
        # set last workflow task as default
        if self.task_id < 0:
            self.task_id = len(self.config["workflow"]) - 1
        # get models
        self.experiments = self._get_models()

    def _get_models(self) -> dict[int, ExperimentInfo]:
        """Get all models and test data from study path."""
        print("\nLoading available experiments ......")
        experiments = {}

        for experiment_id in range(self.n_experiments):
            path_exp = (
                self.study_path
                / f"outersplit{experiment_id}"
                / f"workflowtask{self.task_id}"
                / f"exp{experiment_id}_{self.task_id}.pkl"
            )

            # extract best model. test dataset, feature columns
            if path_exp.exists():
                print(f"Outersplit{experiment_id}, task{self.task_id} found.")
                # load experiment
                experiment = OctoExperiment.from_pickle(path_exp)
                # check if results_key exists
                if self.results_key not in experiment.results:
                    raise ValueError(
                        f"Specified results key not found: {self.results_key}. Available results keys: {list(experiment.results.keys())}"
                    )

                experiments[experiment_id] = ExperimentInfo(
                    id=experiment_id,
                    model=experiment.results[self.results_key].model,
                    data_traindev=experiment.data_traindev,
                    data_test=experiment.data_test,
                    feature_cols=experiment.feature_cols,
                    row_column=experiment.row_column,
                    target_assignments=experiment.target_assignments,
                    target_metric=experiment.target_metric,
                    ml_type=experiment.ml_type,
                    feature_group_dict=experiment.feature_groups,
                    positive_class=experiment.positive_class,
                )
        print(f"{len(experiments)} experiment(s) out of {self.n_experiments} found.")
        return experiments

    @overload
    def predict(self, data: pd.DataFrame, return_df: Literal[True]) -> pd.DataFrame: ...

    @overload
    def predict(self, data: pd.DataFrame, return_df: Literal[False]) -> np.ndarray: ...

    def predict(self, data: pd.DataFrame, return_df=False):
        """Predict on new data."""
        preds_lst = []
        for experiment in self.experiments.values():
            feature_cols = experiment.feature_cols

            if set(feature_cols).issubset(data.columns):
                df = pd.DataFrame(columns=["row_id_col", "prediction"])
                df["row_id_col"] = data.index
                df["prediction"] = experiment.model.predict(data[feature_cols])
                preds_lst.append(df)
            else:
                raise ValueError("Features missing in provided dataset.")

        grouped_df = pd.concat(preds_lst, axis=0).groupby("row_id_col").mean()

        if return_df is True:
            return grouped_df
        else:
            return grouped_df.to_numpy()

    @overload
    def predict_proba(self, data: pd.DataFrame, return_df: Literal[True]) -> pd.DataFrame: ...

    @overload
    def predict_proba(self, data: pd.DataFrame, return_df: Literal[False]) -> np.ndarray: ...

    def predict_proba(self, data: pd.DataFrame, return_df=False):
        """Predict_proba on new data."""
        preds_lst = []
        for _, experiment in self.experiments.items():
            feature_cols = experiment.feature_cols
            probabilities = experiment.model.predict_proba(data[feature_cols])

            if set(feature_cols).issubset(data.columns):
                df = pd.DataFrame()
                df["row_id_col"] = data.index
                # only binary predictions are supported
                prob_columns = range(probabilities.shape[1])
                for column in prob_columns:
                    df[column] = probabilities[:, column]
                preds_lst.append(df)
            else:
                raise ValueError("Features missing in provided dataset.")

        grouped_df = pd.concat(preds_lst, axis=0).groupby("row_id_col").agg(["mean", "std", "count"])
        if return_df is True:
            return grouped_df
        else:
            # Extract mean values for all probability columns
            mean_cols = [col for col in grouped_df.columns if col[1] == "mean"]
            return grouped_df[mean_cols].to_numpy()

    def predict_test(self) -> pd.DataFrame:
        """Predict on available test data."""
        preds_lst = []
        for _, experiment in self.experiments.items():
            row_column = experiment.row_column

            df = pd.DataFrame(columns=["row_id_col", "prediction"])
            df["row_id_col"] = experiment.row_test
            df["prediction"] = experiment.model.predict(experiment.x_test)
            preds_lst.append(df)

        grouped_df = (
            pd.concat(preds_lst, axis=0)
            .groupby(row_column)["prediction"]
            .agg(["mean", "std", "count"])
            .rename(
                columns={"mean": "prediction", "std": "prediction_std", "count": "n"},
            )
            .reset_index()
        )

        return grouped_df

    def predict_proba_test(self):
        """Predict_proba on available test data."""
        preds_lst = []
        for _, experiment in self.experiments.items():
            row_column = experiment.row_column

            df = pd.DataFrame(columns=["row_id_col", "probability"])
            df["row_id_col"] = experiment.row_test
            # only binary classification!!
            df["probability"] = experiment.model.predict_proba(experiment.x_test)[:, 1]
            preds_lst.append(df)

        grouped_df = (
            pd.concat(preds_lst, axis=0)
            .groupby(row_column)["probability"]
            .agg(["mean", "std", "count"])
            .rename(
                columns={"mean": "probability", "std": "probability_std", "count": "n"},
            )
            .reset_index()
        )

        return grouped_df

    def calculate_fi(
        self,
        data: pd.DataFrame,
        n_repeat: int = 10,
        fi_type: Literal["permutation", "group_permutation", "shap", "group_shap"] = "permutation",
        shap_type: Literal["exact", "permutation", "kernel"] = "kernel",
    ):
        """Calculate feature importances on new data."""
        if shap_type not in ["exact", "permutation", "kernel"]:
            raise ValueError("Specified shap_type not supported.")

        # feature importances for every single available experiment/model
        print("Calculating feature importances for every experiment/model.")
        for experiment in self.experiments.values():
            exp_id = experiment.id
            if fi_type == "permutation":
                results_df = get_fi_permutation(experiment, n_repeat, data=data)
                self.results[f"fi_table_permutation_exp{exp_id}"] = results_df
                self._plot_permutation_fi(exp_id, results_df)
            elif fi_type == "group_permutation":
                results_df = get_fi_group_permutation(experiment, n_repeat, data=data)
                self.results[f"fi_table_group_permutation_exp{exp_id}"] = results_df
                self._plot_permutation_fi(exp_id, results_df)
            elif fi_type == "shap":
                results_df, _shap_values, _shap_data = get_fi_shap(experiment, data=data, shap_type=shap_type)
                self.results[f"fi_table_shap_exp{exp_id}"] = results_df
            elif fi_type == "group_shap":
                results_df = get_fi_group_shap(experiment, data=data, shap_type=shap_type)
                self.results[f"fi_table_group_shap_exp{exp_id}"] = results_df
            else:
                raise ValueError("Feature Importance type not supported")

        # feature importances for the combined predictions
        print("Calculating combined feature importances.")
        # create combined experiment
        feature_col_lst = []
        for experiment in self.experiments.values():
            feature_col_lst.extend(experiment.feature_cols)

        # use last experiment in for loop
        exp_combined = ExperimentInfo(
            id="_all",
            model=self,  # type: ignore[arg-type]
            data_traindev=pd.concat([experiment.data_traindev, experiment.data_test], axis=0),
            # same for all experiments
            data_test=experiment.data_test,  # not used
            feature_cols=list(set(feature_col_lst)),
            row_column=experiment.row_column,
            target_assignments=experiment.target_assignments,
            target_metric=experiment.target_metric,
            ml_type=experiment.ml_type,
            feature_group_dict=experiment.feature_group_dict,
            positive_class=experiment.positive_class,
        )

        if fi_type == "permutation":
            results_df = get_fi_permutation(exp_combined, n_repeat, data=data)
            self.results["fi_table_permutation_ensemble"] = results_df
            self._plot_permutation_fi(exp_combined.id, results_df)
        elif fi_type == "group_permutation":
            results_df = get_fi_group_permutation(exp_combined, n_repeat, data=data)
            self.results["fi_table_group_permutation_ensemble"] = results_df
            self._plot_permutation_fi(exp_combined.id, results_df)
        elif fi_type == "shap":
            results_df, _shap_values, _shap_data = get_fi_shap(exp_combined, data=data, shap_type=shap_type)
            self.results["fi_table_shap_ensemble"] = results_df
        elif fi_type == "group_shap":
            results_df = get_fi_group_shap(exp_combined, data=data, shap_type=shap_type)
            self.results["fi_table_group_shap_ensemble"] = results_df

    def calculate_fi_test(
        self,
        n_repeat: int = 10,
        fi_type: str = "group_permutation",
        experiment_id: int = -1,  # Calculate for all experiments
        shap_type: Literal["exact", "permutation", "kernel"] = "kernel",
    ):
        """Calculate feature importances on available test data."""
        if shap_type not in ["exact", "permutation", "kernel"]:
            raise ValueError("Specified shap_type not supported.")

        print("Calculating feature importances for every experiment/model.")

        # Filter experiments based on experiment_id
        experiments_to_process = [exp for exp in self.experiments.values() if experiment_id in (-1, exp.id)]

        # Define a helper function for processing an experiment
        def _process_experiment(exp):
            exp_id = exp.id
            if fi_type == "permutation":
                results_df = get_fi_permutation(exp, n_repeat, data=None)
                key = f"fi_table_permutation_exp{exp_id}"
                self._plot_permutation_fi(exp_id, results_df)
            elif fi_type == "group_permutation":
                results_df = get_fi_group_permutation(exp, n_repeat, data=None)
                key = f"fi_table_group_permutation_exp{exp_id}"
                self._plot_permutation_fi(exp_id, results_df)
            elif fi_type == "shap":
                results_df, shap_values, shap_data = get_fi_shap(exp, data=None, shap_type=shap_type)
                self._plot_shap_fi(exp_id, results_df, shap_values, shap_data)
                key = f"fi_table_shap_exp{exp_id}"
            elif fi_type == "group_shap":
                results_df = get_fi_group_shap(exp, data=None, shap_type=shap_type)
                key = f"fi_table_group_shap_exp{exp_id}"
            else:
                raise ValueError("Feature Importance type not supported")
            return key, results_df

        # Use joblib to parallelize the processing of experiments
        results = Parallel(n_jobs=-1)(delayed(_process_experiment)(exp) for exp in experiments_to_process)

        # Update shared resources in the main thread
        for key, results_df in results:
            self.results[key] = results_df

    def _plot_shap_fi(self, experiment_id, shapfi_df, shap_values, data):
        """Create plot for shape fi and save to file."""
        results_path = self.study_path / f"outersplit{experiment_id}" / f"workflowtask{self.task_id}" / "results"
        # create directories if needed, required for id="all"
        results_path.parent.mkdir(parents=True, exist_ok=True)

        # (A) Bar plot
        save_path = results_path / f"model_shap_fi_barplot_exp{experiment_id}_{self.task_id}.pdf"
        with PdfPages(save_path) as pdf:
            plt.figure(figsize=(8.27, 11.69))  # portrait orientation (A4)
            shap.summary_plot(shap_values, data, plot_type="bar", show=False)
            plt.tight_layout()
            pdf.savefig(plt.gcf(), orientation="portrait")
            plt.close()

        # (B) Beeswarm plot
        save_path = results_path / f"model_shap_fi_beeswarm_exp{experiment_id}_{self.task_id}.pdf"

        with PdfPages(save_path) as pdf:
            plt.figure(figsize=(8.27, 11.69))  # portrait orientation (A4)
            shap.summary_plot(shap_values, data, plot_type="dot", show=False)
            plt.tight_layout()
            pdf.savefig(plt.gcf(), orientation="portrait")
            plt.close()

        # (C) Save shape feature importance
        shapfi_df.reset_index()

    def _plot_permutation_fi(self, experiment_id, df):
        """Create plot for permutation fi and save to file."""
        # Calculate error bars
        lower_error = df["importance"] - df["ci_low_95"]
        upper_error = df["ci_high_95"] - df["importance"]
        error = [lower_error.values, upper_error.values]

        save_path = (
            self.study_path
            / f"outersplit{experiment_id}"
            / f"workflowtask{self.task_id}"
            / "results"
            / f"model_permutation_fi_exp{experiment_id}_{self.task_id}.pdf"
        )
        # create directories if needed, required for id="all"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # plot figure and save to pdf
        with PdfPages(save_path) as pdf:
            plt.figure(figsize=(8.27, 11.69))  # portrait orientation (A4)
            _ = plt.barh(
                df["feature"],
                df["importance"],
                xerr=error,
                capsize=5,
                color="royalblue",
                # edgecolor="black",
            )

            # Adding labels and title
            plt.ylabel("Feature")
            plt.xlabel("Importance")
            plt.title("Feature Importance with Confidence Intervals")
            plt.grid(True, axis="x")

            # Adjust layout to make room for the plot
            plt.tight_layout()

            pdf.savefig(plt.gcf(), orientation="portrait")
            plt.close()
