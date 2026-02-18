"""ROC module (removal of correlated features)."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats
from attrs import Factory, define, field
from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)

from octopus.logger import get_logger
from octopus.modules.base import FeatureSelectionExecution
from octopus.modules.utils import rdc_correlation_matrix

if TYPE_CHECKING:
    from upath import UPath

    from octopus.modules.roc.module import Roc
    from octopus.study.context import StudyContext

logger = get_logger()

# Filter functions for feature selection
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
class RocModule(FeatureSelectionExecution["Roc"]):
    """ROC execution module. Created by Roc.create_module()."""

    feature_groups_: list = field(init=False, default=Factory(list))
    """Feature groups discovered during fit (stored for inspection)."""

    def fit(
        self,
        data_traindev: pd.DataFrame,
        data_test: pd.DataFrame,
        feature_cols: list[str],
        study: StudyContext,
        outersplit_id: int,
        output_dir: UPath,
        num_assigned_cpus: int = 1,
        feature_groups: dict | None = None,
        prior_results: dict | None = None,
    ) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fit ROC module by identifying and filtering correlated features."""
        # Set seeds for reproducibility
        random.seed(0)
        np.random.seed(0)

        logger.info(f"Correlation type: {self.config.correlation_type}")
        logger.info(f"Threshold: {self.config.threshold}")
        logger.info(f"Filter type: {self.config.filter_type}")

        # Extract feature matrices (local variables)
        x_traindev = data_traindev[feature_cols]
        y_traindev = data_traindev[list(study.target_assignments.values())]

        # Calculate dependency to target
        logger.info("Calculating dependency to target")
        if study.ml_type == "timetoevent":
            logger.info("Time2Event: Note, that the first group element is selected.")
            dependency = None  # Not used for timetoevent
        elif self.config.filter_type == "mutual_info":
            # Set random state
            values = filter_inventory[self.config.filter_type][study.ml_type](
                x_traindev, y_traindev.to_numpy().ravel(), random_state=0
            )
            dependency = pd.Series(values, index=feature_cols)
        elif self.config.filter_type == "f_statistics":
            # Ignoring p-values
            values, _ = filter_inventory[self.config.filter_type][study.ml_type](
                x_traindev, y_traindev.to_numpy().ravel()
            )
            dependency = pd.Series(values, index=feature_cols)

        # Calculate correlation matrix
        logger.info("Calculating feature groups.")
        if self.config.correlation_type == "spearmanr":
            pos_corr_matrix, _ = scipy.stats.spearmanr(np.nan_to_num(x_traindev.values))
            pos_corr_matrix = np.abs(pos_corr_matrix)
        elif self.config.correlation_type == "rdc":
            pos_corr_matrix = np.abs(rdc_correlation_matrix(x_traindev))
        else:
            raise ValueError(f"Correlation type {self.config.correlation_type} not supported")

        # Build graph of correlated features
        g = nx.Graph()
        for i in range(len(feature_cols)):
            for j in range(i + 1, len(feature_cols)):
                if pos_corr_matrix[i, j] > self.config.threshold:
                    g.add_edge(i, j)

        # Get connected components and sort for determinism
        subgraphs = [
            g.subgraph(c).copy() for c in sorted(nx.connected_components(g), key=lambda x: (len(x), sorted(x)))
        ]

        # Create groups of feature columns
        groups = []
        for sg in subgraphs:
            groups.append([feature_cols[node] for node in sorted(sg.nodes())])

        # Sort each group for determinism
        self.feature_groups_ = [sorted(g) for g in groups]

        # Select features to keep and to remove
        keep_list = []
        remove_list = []

        for group in self.feature_groups_:
            if group:
                if study.ml_type == "timetoevent":
                    # timetoevent: keep first feature
                    keep_feature = group[0]
                else:
                    # regression, classification: use filter to find best feature
                    keep_feature = dependency[group].idxmax()

                keep_list.append(keep_feature)
                remove_list.extend([x for x in group if x != keep_feature])

        # Get features after filtering
        remaining_features = sorted(set(feature_cols) - set(remove_list))

        logger.info(f"Remaining features: {remaining_features}")
        logger.info(f"Number of features before correlation removal: {len(feature_cols)}")
        logger.info(f"Number of features after correlation removal: {len(remaining_features)}")

        # Store selected features (fitted state only)
        selected_features = sorted(remaining_features, key=lambda x: (len(x), sorted(x)))
        self.selected_features_ = selected_features

        logger.info("ROC completed")

        # Store fitted state
        self.feature_importances_ = {}

        return (selected_features, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
