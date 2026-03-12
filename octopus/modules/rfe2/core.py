"""Rfe2 execution module."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from attrs import Factory, define, field

from octopus.modules import ModuleResult, Rfe2
from octopus.modules.octo.core import OctoModuleTemplate
from octopus.utils import calculate_feature_groups

if TYPE_CHECKING:
    from upath import UPath

from octopus.modules import StudyContext
from octopus.modules.octo.bag import BagBase
from octopus.types import FIComputeMethod, FIDataset, FIResultLabel, ResultType


@define
class Rfe2Module(OctoModuleTemplate[Rfe2]):
    """Rfe2 execution module. Created by Rfe2.create_module()."""

    # Internal state for RFE
    rfe_results_: pd.DataFrame = field(init=False, default=Factory(lambda: pd.DataFrame()))
    """RFE results dataframe tracking performance at each step."""

    def fit(
        self,
        *,
        data_traindev: pd.DataFrame,
        data_test: pd.DataFrame,
        feature_cols: list[str],
        study_context: StudyContext,
        outersplit_id: int,
        results_dir: UPath,
        scratch_dir: UPath,
        num_assigned_cpus: int,
        feature_groups: dict[str, list[str]],
        **kwargs,
    ) -> dict[ResultType, ModuleResult]:
        """Fit Rfe2 module by running Octo optimization followed by RFE."""
        x_traindev = data_traindev[feature_cols]
        y_traindev = data_traindev[list(study_context.target_assignments.values())]

        # Initialize RFE results DataFrame
        self.rfe_results_ = pd.DataFrame(
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

        # Initialize Octo-specific setup
        self._initialize_octo(
            study_context,
            data_traindev,
            x_traindev,
            y_traindev,
            feature_cols,
            outersplit_id,
            num_assigned_cpus,
            scratch_dir,
            results_dir,
        )

        # (1) Run Octo optimization to get best bag

        # Initialize local results collection
        results: dict[str, dict] = {}

        # (1) model training and optimization
        self._run_globalhp_optimization(
            study_context,
            data_test,
            feature_cols,
            feature_groups,
            outersplit_id,
            results_dir,
            results,
        )

        # (2) Get best bag from Octo results
        if "best" not in results:
            raise ValueError("RFE2 requires 'best' results from Octo optimization")

        # Get bag from module state (stored by OctoModule._run_globalhp_optimization)
        bag = copy.deepcopy(results["best"]["_bag"])
        bag_scores = results["best"]["scores"]
        bag_selected_features = bag.get_selected_features(fi_methods=[self.config.fi_method_rfe])

        # (3) Run RFE iterations
        # record baseline performance
        step = 0
        dev_lst = bag_scores["dev_lst"]
        result = {
            "step": step,
            "performance_mean": bag_scores["dev_avg"],
            "performance_sem": np.std(dev_lst, ddof=1) / len(dev_lst),  # no np.sqrt
            "n_features": len(bag_selected_features),
            "features": bag_selected_features,
            "feature_importances": self._get_fi(bag),
            "model": copy.deepcopy(bag),
        }
        self.rfe_results_.loc[len(self.rfe_results_)] = result

        self._print_step_information()

        # Run RFE iterations
        while True:
            step = step + 1
            # calculate new features
            new_features = self._calculate_new_features(bag)

            if len(new_features) < self.config.min_features_to_select:
                break

            # retrain bag and calculate feature importances
            bag = self._retrain_and_calc_fi(bag, data_traindev, new_features)

            # get scores
            bag_scores = bag.get_performance()

            # record performance
            dev_lst = bag_scores["dev_lst"]
            result = {
                "step": step,
                "performance_mean": bag_scores["dev_avg"],
                "performance_sem": np.std(dev_lst, ddof=1) / len(dev_lst),  # no np.sqrt
                "n_features": len(new_features),
                "features": new_features,
                "feature_importances": self._get_fi(bag),
                "model": copy.deepcopy(bag),
            }
            self.rfe_results_.loc[len(self.rfe_results_)] = result

            # print step results
            self._print_step_information()

        # (4) Analyze results and select best model
        if self.config.selection_method == "best":
            selected_row = self.rfe_results_.loc[self.rfe_results_["performance_mean"].idxmax()]
        elif self.config.selection_method == "parsimonious":
            # best performance mean and sem
            best_performance_mean = self.rfe_results_["performance_mean"].max()
            best_performance_sem = self.rfe_results_.loc[
                self.rfe_results_["performance_mean"] == best_performance_mean,
                "performance_sem",
            ].values[0]
            # define threshold for accepting solution with less features
            threshold = best_performance_mean - best_performance_sem
            filtered_df = self.rfe_results_[self.rfe_results_["performance_mean"] >= threshold]
            if not filtered_df.empty:
                selected_row = filtered_df.loc[filtered_df["n_features"].idxmin()]
            else:
                # take best value if no solution with less features can be found
                selected_row = self.rfe_results_.loc[self.rfe_results_["performance_mean"].idxmax()]

        # save results
        best_model = selected_row["model"]
        selected_features = best_model.get_selected_features(fi_methods=[self.config.fi_method_rfe])

        print("RFE solution:")
        print(
            f"Step: {selected_row['step']}, n_features: {selected_row['n_features']}"
            f", Perf_mean: {selected_row['performance_mean']:.4f}"
            f", Perf_sem: {selected_row['performance_sem']:.4f}"
        )
        print("Selected features:", selected_row["features"])

        # Build flat scores DataFrame from best_model
        scores = best_model.get_performance_df(metric=study_context.target_metric)
        scores["result_type"] = ResultType.BEST

        # Build flat predictions DataFrame
        predictions = best_model.get_predictions_df()
        if not predictions.empty:
            predictions["result_type"] = ResultType.BEST

        # Build flat feature_importances DataFrame
        fi_df = selected_row["feature_importances"]
        feature_importances = fi_df[["feature", "importance"]].copy()
        fi_method = self.config.fi_method_rfe
        if fi_method == FIComputeMethod.PERMUTATION:
            feature_importances["fi_method"] = FIResultLabel.PERMUTATION
            feature_importances["fi_dataset"] = FIDataset.DEV
        elif fi_method == FIComputeMethod.SHAP:
            feature_importances["fi_method"] = FIResultLabel.SHAP
            feature_importances["fi_dataset"] = FIDataset.DEV
        else:
            feature_importances["fi_method"] = fi_method
            feature_importances["fi_dataset"] = FIDataset.DEV
        feature_importances["training_id"] = "rfe2"
        feature_importances["result_type"] = ResultType.BEST

        return {
            ResultType.BEST: ModuleResult(
                result_type=ResultType.BEST,
                module=self.config.module,
                selected_features=selected_features,
                scores=scores,
                predictions=predictions,
                feature_importances=feature_importances.to_frame(),
                model=best_model,
            )
        }

    def _print_step_information(self):
        """Print step performance."""
        last_row = self.rfe_results_.iloc[-1]
        print(
            f"Step: {last_row['step']}, n_features: {last_row['n_features']}"
            f", Perf_mean: {last_row['performance_mean']:.4f}"
            f", Perf_sem: {last_row['performance_sem']:.4f}"
        )

    def _retrain_and_calc_fi(self, bag: BagBase, data_traindev: pd.DataFrame, new_features: list) -> BagBase:
        """Retrain bag using new feature set and calculate feature importances."""
        bag = copy.deepcopy(bag)

        # update feature_cols and feature groups
        feature_groups = calculate_feature_groups(data_traindev, new_features)
        for training in bag.trainings:
            training.feature_cols = new_features
            training.feature_groups = feature_groups

        # retrain bag
        bag.fit()

        # calculate feature importances
        bag.calculate_feature_importances([self.config.fi_method_rfe], partitions=["dev"])

        return bag

    def _get_fi(self, bag: BagBase) -> pd.DataFrame:
        """Get relevant feature importances."""
        if self.config.fi_method_rfe == FIComputeMethod.PERMUTATION:
            fi_df = bag.feature_importances["permutation_dev_mean"]
        elif self.config.fi_method_rfe == FIComputeMethod.SHAP:
            fi_df = bag.feature_importances["shap_dev_mean"]

        return fi_df  # type: ignore[return-value]

    def _calculate_new_features(self, bag: BagBase) -> list[str]:
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
        if self.config.abs_on_fi:
            selection_column = "importance_abs"
        else:
            selection_column = "importance"  # negative values may exist

        # only consider non-protected features
        fi_avail_df = fi_df[~fi_df["protected"]]

        # get the feature with the lowest value in the selection column
        lowest_feature: str = fi_avail_df.loc[fi_avail_df[selection_column].idxmin(), "feature"]  # type: ignore[assignment]

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
