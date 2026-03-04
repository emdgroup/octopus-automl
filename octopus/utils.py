"""Utils."""

import logging
from importlib.metadata import version
from typing import Any

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats
from upath import UPath


def get_package_name() -> str:
    """Return the package name."""
    return "octopus-automl"


def get_version() -> str:
    """Return the installed version of octopus-automl."""
    return version(get_package_name())


def joblib_save(obj: Any, path: UPath) -> None:
    """Save an object with joblib through a file handle (fsspec-compatible)."""
    with path.open("wb") as f:
        joblib.dump(obj, f)


def joblib_load(path: UPath) -> Any:
    """Load an object with joblib through a file handle (fsspec-compatible)."""
    with path.open("rb") as f:
        return joblib.load(f)


def calculate_feature_groups(data_traindev: pd.DataFrame, feature_cols: list[str]) -> dict[str, list[str]]:
    """Calculate feature groups based on correlation thresholds.

    Args:
        data_traindev: DataFrame containing the training data
        feature_cols: List of feature column names to group

    Returns:
        Dictionary mapping group names to lists of feature names
    """
    if len(feature_cols) <= 2:
        logging.warning("Not enough features to calculate correlations for feature groups.")
        return {}
    logging.info("Calculating feature groups.")

    auto_group_thresholds = [0.7, 0.8, 0.9]
    auto_groups = []

    pos_corr_matrix, _ = scipy.stats.spearmanr(np.nan_to_num(data_traindev[feature_cols].values))
    pos_corr_matrix = np.abs(pos_corr_matrix)

    # get groups depending on threshold
    for threshold in auto_group_thresholds:
        g: nx.Graph = nx.Graph()
        for i in range(len(feature_cols)):
            for j in range(i + 1, len(feature_cols)):
                if pos_corr_matrix[i, j] > threshold:
                    g.add_edge(i, j)

        # Get connected components and sort them to ensure determinism
        subgraphs = [
            g.subgraph(c).copy() for c in sorted(nx.connected_components(g), key=lambda x: (len(x), sorted(x)))
        ]
        # Create groups of feature columns
        groups = []
        for sg in subgraphs:
            groups.append([feature_cols[node] for node in sorted(sg.nodes())])
        auto_groups.extend([sorted(g) for g in groups])

    # find unique groups
    auto_groups_unique = [list(t) for t in sorted(set(map(tuple, auto_groups)))]

    return {f"group{i}": group for i, group in enumerate(auto_groups_unique)}
