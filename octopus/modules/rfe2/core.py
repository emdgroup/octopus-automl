# type: ignore

"""RFE2."""

import copy

import numpy as np
import pandas as pd
from attrs import Factory, define, field, validators

from octopus.modules.octo.bag import BagBase
from octopus.modules.octo.core import OctoCoreGeneric
from octopus.modules.rfe2.module import Rfe2
from octopus.results import ModuleResults

# TOBEDONE
# - remove least important feature
#   -- deal with group features, needs to be intelligent
# - do we need the step input?
# - jason output in results, see rfe
# - how to override/disable OcoCore inputs hat are not needed?
# - model retraining after n removal, or start use module several times
# - autogluon: add 3-5 random feature and remove all feature below the lowest random
# - show removed zero features
# - pfi, set number of repeats to improve quality of pfi
# - collect fi_df for all steps for deeper understanding
# - set abs_on_fi default to False
# - Explore RFE2
#   -- retraining after n eliminations, several modules
#   -- abs() or not
#   -- use shap


@define
class Rfe2Core(OctoCoreGeneric[Rfe2]):
    """Rfe2 Core."""

    # Optional attribute (with default value)
    results: pd.DataFrame = field(
        init=False,
        default=Factory(pd.DataFrame),
        validator=[validators.instance_of(pd.DataFrame)],
    )
    """RFE results dataframe."""

    @property
    def config(self) -> Rfe2:
        """Module configuration."""
        return self.experiment.ml_config

    @property
    def feature_groups(self) -> dict:
        """Feature groups."""
        return self.experiment.feature_groups

    @property
    def fi_method(self) -> str:
        """Feature importance method."""
        return self.config.fi_method_rfe

    @property
    def step(self) -> int:
        """Number of features to be removed in RFE step."""
        return self.config.step

    @property
    def min_features_to_select(self) -> int:
        """Minimum number of features to select."""
        return self.config.min_features_to_select

    @property
    def selection_method(self) -> str:
        """Method for selection final solution (best/parsimonious)."""
        return self.config.selection_method

    @property
    def abs_on_fi(self) -> bool:
        """Convert all feature importances to positive values (abs())."""
        return self.config.abs_on_fi

    def __attrs_post_init__(self):
        # run OctoCore post_init() to create directory, etc...
        super().__attrs_post_init__()

        # Initialize results DataFrame
        self.results = pd.DataFrame(
            columns=[
                "step",
                "performance_mean",
                "performance_sem",
                "n_features",
                "features",
                "feature_importances",
                "model",
            ]
        )

    def run_experiment(self):
        """Run experiment."""
        # (1) train and optimize model
        self._optimize_splits(self.data_splits)
        # create best bag
        self._create_best_bag()
        bag_results = copy.deepcopy(self.experiment.results["best"])
        bag = bag_results.model
        bag_scores = bag_results.scores
        bag_selected_features = bag_results.selected_features

        # record baseline performance
        step = 0
        dev_lst = bag_scores["dev_lst"]
        self.results.loc[len(self.results)] = {
            "step": step,
            "performance_mean": bag_scores["dev_avg"],
            "performance_sem": np.std(dev_lst, ddof=1) / len(dev_lst),  # no np.sqrt
            "n_features": len(bag_selected_features),
            "features": bag_selected_features,
            "feature_importances": self._get_fi(bag),
            "model": copy.deepcopy(bag),
        }

        self._print_step_information()

        # (2) run RFE iterations
        while True:
            step = step + 1
            # calculate new features
            new_features = self._calculate_new_features(bag)

            if len(new_features) < self.min_features_to_select:
                break

            # retrain bag and calculate feature importances
            bag = self._retrain_and_calc_fi(bag, new_features)

            # get scores
            bag_scores = bag.get_performance()

            # record performance
            dev_lst = bag_scores["dev_lst"]
            self.results.loc[len(self.results)] = {
                "step": step,
                "performance_mean": bag_scores["dev_avg"],
                "performance_sem": np.std(dev_lst, ddof=1) / len(dev_lst),  # no np.sqrt
                "n_features": len(new_features),
                "features": new_features,
                "feature_importances": self._get_fi(bag),
                "model": copy.deepcopy(bag),
            }

            # print step results
            self._print_step_information()

        # (3) analyze results and select best model
        #    - create and save results object
        if self.selection_method == "best":
            selected_row = self.results.loc[self.results["performance_mean"].idxmax()]
        elif self.selection_method == "parsimonious":
            # best performance mean and sem
            best_performance_mean = self.results["performance_mean"].max()
            best_performance_sem = self.results.loc[
                self.results["performance_mean"] == best_performance_mean,
                "performance_sem",
            ].values[0]
            # define threshold for accepting solution with less features
            threshold = best_performance_mean - best_performance_sem
            filtered_df = self.results[self.results["performance_mean"] >= threshold]
            if not filtered_df.empty:
                selected_row = filtered_df.loc[filtered_df["n_features"].idxmin()]
            else:
                # take best value if no solution with less features can be found
                selected_row = self.results.loc[self.results["performance_mean"].idxmax()]

        # save results to experiment
        best_model = selected_row["model"]
        self.experiment.results["Rfe2"] = ModuleResults(
            id="rfe2",
            experiment_id=self.experiment.experiment_id,
            task_id=self.experiment.task_id,
            model=best_model,
            scores=best_model.get_performance(),
            feature_importances={
                "dev": selected_row["feature_importances"],
            },
            selected_features=best_model.get_selected_features(fi_methods=[self.fi_method]),
        )

        print("RFE solution:")
        print(
            f"Step: {selected_row['step']}, n_features: {selected_row['n_features']}"
            f", Perf_mean: {selected_row['performance_mean']:.4f}"
            f", Perf_sem: {selected_row['performance_sem']:.4f}"
        )
        print("Selected features:", selected_row["features"])

        return self.experiment

    def _print_step_information(self):
        """Print step performance."""
        last_row = self.results.iloc[-1]
        print(
            f"Step: {last_row['step']}, n_features: {last_row['n_features']}"
            f", Perf_mean: {last_row['performance_mean']:.4f}"
            f", Perf_sem: {last_row['performance_sem']:.4f}"
        )

    def _retrain_and_calc_fi(self, bag: BagBase, new_features: list) -> BagBase:
        """Retrain bag using new feature set and calculate feature importances."""
        bag = copy.deepcopy(bag)

        # update feature_cols and feature groups
        feature_groups = self.experiment.calculate_feature_groups(new_features)
        for training in bag.trainings:
            training.feature_cols = new_features
            training.feature_groups = feature_groups

        # update feature groups??

        # retrain bag
        bag.fit()

        # calculate feature importances
        bag.calculate_feature_importances([self.fi_method], partitions=["dev"])

        return bag

    def _get_fi(self, bag: BagBase) -> pd.DataFrame:
        """Get relevant feature importances."""
        if self.fi_method == "permutation":
            fi_df = bag.feature_importances["permutation_dev_mean"]
        elif self.fi_method == "shap":
            fi_df = bag.feature_importances["shap_dev_mean"]

        return fi_df

    def _calculate_new_features(self, bag: BagBase) -> list:
        """Perform RFE step and calculate new features."""
        bag = copy.deepcopy(bag)

        fi_df = self._get_fi(bag)

        # only keep nonzero features
        fi_df = fi_df[fi_df["importance"] != 0]

        # calculate absolute values
        fi_df["importance_abs"] = fi_df["importance"].abs()
        fi_df = fi_df.sort_values(by="importance_abs", ascending=False)

        # get group features in fi_df
        groups_df = fi_df[fi_df["feature"].str.startswith("group")].copy()
        group_features = groups_df["feature"]

        # mark group features as protected
        fi_df["protected"] = False
        for group_feature in group_features:
            # Get all features belonging to that group
            group_members = bag.feature_groups[group_feature]
            # Mark "protected" as True for all features found in those groups
            fi_df.loc[fi_df["feature"].isin(group_members), "protected"] = True

        # define column for feature elimination
        if self.abs_on_fi:
            selection_column = "importance_abs"
        else:
            selection_column = "importance"  # negative values may exist

        # only consider non-protected features
        fi_avail_df = fi_df[~fi_df["protected"]]

        # get the feature with the lowest value in the selection column
        lowest_feature = fi_avail_df.loc[fi_avail_df[selection_column].idxmin(), "feature"]

        # get all feature to be removed, including group members
        if lowest_feature in bag.feature_groups:
            drop_features = bag.feature_groups[lowest_feature]
            drop_features.append(lowest_feature)
        else:
            drop_features = [lowest_feature]

        # remove drop features in fi_df
        fi_df = fi_df[~fi_df["feature"].isin(drop_features)]

        # remove all group features -> single features
        fi_df = fi_df[~fi_df["feature"].str.startswith("group")]

        print("Features removed in this step:", drop_features)
        print("Number of removed features:", len(drop_features))
        print("Number of remaining features:", len(fi_df))

        return sorted(fi_df["feature"], key=lambda x: (len(x), sorted(x)))
