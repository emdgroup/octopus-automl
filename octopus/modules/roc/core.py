# type: ignore

"""ROC core (removal of correlated features)."""

import random

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats
from attrs import define, field, validators
from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)

from octopus.logger import get_logger
from octopus.modules.base import ModuleBaseCore
from octopus.modules.roc.module import Roc
from octopus.modules.utils import rdc_correlation_matrix

logger = get_logger()

# TOBEDONE
# - add hierarchical clustering
#   https://scikit-learn.org/stable/auto_examples/inspection/
#   plot_permutation_importance_multicollinear.html
# - add pearson
# - maybe, a function that create as table showing the selected feature and
# the removed group features
# - How to select best feature im group
#   (a) classification and regression --> mutual information
#   (b) timetoevent --> first feature
# - Use univariate model to assess association with target, should work for all ml_types


filter_inventory = {
    "mutual_info": {
        "classification": mutual_info_classif,
        "regression": mutual_info_regression,
    },
    "f_statistics": {
        "classification": f_classif,
        "regression": f_regression,
    },
}


@define
class RocCore(ModuleBaseCore[Roc]):
    """Roc Module (Removal of Correlated features).

    Inherits log_dir from ModuleBaseCore.
    """

    feature_groups: list = field(init=False, validator=[validators.instance_of(list)])

    @property
    def config(self) -> Roc:
        """Module configuration."""
        return self.experiment.ml_config

    @property
    def filter_type(self) -> str:
        """Filter Type."""
        return self.experiment.ml_config.filter_type

    def __attrs_post_init__(self):
        """Initialize feature groups and prepare directories."""
        self.feature_groups = []
        super().__attrs_post_init__()  # Clean/create results directory

    def run_experiment(self):
        """Run ROC module on experiment."""
        # run experiment and return updated experiment object

        # set seeds for reproducibility
        random.seed(0)
        np.random.seed(0)

        correlation_type = self.config.correlation_type
        threshold = self.config.threshold

        logger.info(f"Correlation type: {correlation_type}")
        logger.info(f"Threshold: {threshold}")
        logger.info(f"Filter type: {self.filter_type}")

        logger.info("Calculating dependency to target")
        # Note, timetoevent is treated differently
        if self.ml_type == "timetoevent":
            logger.info("Time2Event: Note, that the first group element is selected.")
        elif self.filter_type == "mutual_info":
            # set random state
            values = filter_inventory[self.filter_type][self.ml_type](
                self.x_traindev, self.y_traindev.to_numpy().ravel(), random_state=0
            )
            dependency = pd.Series(values, index=self.feature_cols)
        elif self.filter_type == "f_statistics":
            # ignoring p-values
            values, _ = filter_inventory[self.filter_type][self.ml_type](
                self.x_traindev, self.y_traindev.to_numpy().ravel()
            )
            dependency = pd.Series(values, index=self.feature_cols)

        logger.info("Calculating feature groups.")
        # correlation matrix
        if correlation_type == "spearmanr":
            # (A) spearmamr correlation matrix
            pos_corr_matrix, _ = scipy.stats.spearmanr(np.nan_to_num(self.x_traindev.values))
            pos_corr_matrix = np.abs(pos_corr_matrix)
        elif correlation_type == "rdc":
            # (B) RDC correlation matrix
            pos_corr_matrix = np.abs(rdc_correlation_matrix(self.x_traindev))
        else:
            raise ValueError(f"Correlation type {correlation_type} not supported")

        g = nx.Graph()

        # Add edges to the graph based on the correlation matrix
        for i in range(len(self.feature_cols)):
            for j in range(i + 1, len(self.feature_cols)):
                if pos_corr_matrix[i, j] > threshold:
                    g.add_edge(i, j)

        # Get connected components and sort them to ensure determinism
        subgraphs = [
            g.subgraph(c).copy() for c in sorted(nx.connected_components(g), key=lambda x: (len(x), sorted(x)))
        ]

        # Create groups of feature columns
        groups = []
        for sg in subgraphs:
            groups.append([self.feature_cols[node] for node in sorted(sg.nodes())])

        # Sort each group to ensure determinism
        self.feature_groups = [sorted(g) for g in groups]

        # g = nx.Graph()
        #
        # for i in range(len(self.feature_cols)):
        #    for j in range(i + 1, len(self.feature_cols)):
        #        if pos_corr_matrix[i, j] > threshold:
        #            g.add_edge(i, j)
        #
        # subgraphs = [g.subgraph(c) for c in nx.connected_components(g)]
        #
        # groups = []
        # for sg in subgraphs:
        #    groups.append([self.feature_cols[node] for node in sg.nodes()])
        # self.feature_groups = [sorted(g) for g in groups]

        # select features to keep and to remove
        keep_list = []
        remove_list = []

        # Process each group
        for group in self.feature_groups:
            if group:
                if self.ml_type == "timetoevent":
                    # timetovent: keep first features
                    keep_feature = group[0]
                else:
                    # regression, classification: use mutual information
                    # to find group element with maximum mutual information
                    keep_feature = dependency[group].idxmax()

                keep_list.append(keep_feature)
                # Add the remaining features to the remove list
                remove_list.extend([x for x in group if x != keep_feature])

        # get features after filtering
        # remaining_features = sorted(set(self.feature_cols) - set(remove_list))
        remaining_features = sorted(set(self.feature_cols) - set(remove_list))

        logger.info(f"Remaining features: {remaining_features}")

        logger.info(f"Number of features before correlation removal: {len(self.feature_cols)}")
        logger.info(f"Number of features after correlation removal: {len(remaining_features)}")

        # save features selected by ROC
        self.experiment.selected_features = sorted(remaining_features, key=lambda x: (len(x), sorted(x)))

        logger.info("ROC completed")

        return self.experiment
