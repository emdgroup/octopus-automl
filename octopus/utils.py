"""Utils."""

import logging
import random

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats
from attrs import Factory, define, field, validators
from sklearn.model_selection import KFold, StratifiedKFold

from .logger import LogGroup, get_logger

logger = get_logger()


@define
class DataSplit:
    """Data Split.

    We don't use groupKFold as it does not offer the shuffle option.
    The StratifiedGroupKfold might work as an alternative (check examples).
    StratifiedGroupKfold is not available for sklearn 0.24.3
    which is required for Auto-Sklearn 0.15.
    stratification_col: contains the group info used for stratification
    datasplit_col: contains group info on samples. Each group goes either
    into the training or the test dataset.
    """

    datasplit_col: str = field(validator=[validators.instance_of(str)])
    seeds: list = field(
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(int),
            iterable_validator=validators.instance_of(list),
        )
    )
    num_folds: int = field(validator=[validators.instance_of(int)])
    dataset: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    stratification_col: str | None = field(
        default=Factory(lambda: None),
        validator=validators.optional(validators.instance_of(str)),
    )
    process_id: str = field(default="Outer", validator=[validators.instance_of(str)])

    def __attrs_post_init__(self):
        logger.set_log_group(LogGroup.CREATING_DATASPLITS, self.process_id)
        logger.info("Initializing data split creation")
        logger.info("Starting data split process...")
        self.dataset.reset_index(drop=True, inplace=True)

    def get_datasplits(self):
        """Get combined datasplits from a list of datasplit seeds."""
        logger.info("Generating splits for multiple seeds...")
        combined_values = []

        for seed in self.seeds:
            logger.info(f"Processing seed: {seed}")
            result = self._single_seed_datasplits(seed)
            for key in sorted(result.keys()):
                combined_values.append(result[key])

        combined_datasplits = dict(enumerate(combined_values))
        logger.info("Combined data splits created successfully")
        return combined_datasplits

    def _single_seed_datasplits(self, datasplit_seed):
        """Get datasplits for single seed."""
        logger.info(f"Generating splits for seed: {datasplit_seed}")
        random.seed(datasplit_seed)
        np.random.seed(datasplit_seed)

        dataset_unique = self.dataset.drop_duplicates(subset=self.datasplit_col, keep="first", inplace=False)
        dataset_unique.reset_index(drop=True, inplace=True)

        logger.info("Analyzing dataset structure")
        logger.info(f"Number of unique groups (as in column: {self.datasplit_col}): {len(dataset_unique)}")
        logger.info(f"Number of rows in dataset: {len(self.dataset)}")

        kf: KFold | StratifiedKFold
        if self.stratification_col:
            logger.info("Determining split method: Stratified KFold")
            kf = StratifiedKFold(
                n_splits=self.num_folds,
                shuffle=True,
                random_state=datasplit_seed,
            )

            if dataset_unique[self.stratification_col].dtype.kind not in "iub":
                logger.error("Stratification column is of wrong type (expected: bool, int)")
                raise ValueError("Stratification column is of wrong type (expected: bool, int)")

            stratification_target = dataset_unique[self.stratification_col].astype(int)
        else:
            logger.info("Determining split method: KFold (unstratified)")
            kf = KFold(
                n_splits=self.num_folds,
                shuffle=True,
                random_state=datasplit_seed,
            )
            stratification_target = None

        data_splits = {}
        all_test_indices = []
        all_test_groups = []
        logger.info(f"Setting number of splits: {self.num_folds}")
        logger.info("Generating splits...")

        for num_split, (train_ind, test_ind) in enumerate(kf.split(dataset_unique, stratification_target)):  # type: ignore
            groups_train = set(dataset_unique.iloc[train_ind][self.datasplit_col])
            groups_test = set(dataset_unique.iloc[test_ind][self.datasplit_col])
            assert groups_train.intersection(groups_test) == set()
            all_test_groups.extend(list(groups_test))

            partition_train = self.dataset[self.dataset[self.datasplit_col].isin(groups_train)]
            partition_test = self.dataset[self.dataset[self.datasplit_col].isin(groups_test)]
            assert set(partition_train.index).intersection(partition_test.index) == set()
            all_test_indices.extend(partition_test.index.tolist())

            partition_train.reset_index(drop=True, inplace=True)
            partition_test.reset_index(drop=True, inplace=True)

            if self.process_id == "Outer":
                info_name = "EXP"
                dataset_name_1 = "Train/Dev"
                dataset_name_2 = "Test"
            else:
                info_name = "SPLIT"
                dataset_name_1 = "Train"
                dataset_name_2 = "Dev"

            logger.info(
                f"{info_name} {num_split} created: "
                f"{dataset_name_1} - {len(partition_train)} rows, "
                f"{len(set(groups_train))} groups | "
                f"{dataset_name_2} - {len(partition_test)} rows, "
                f"{len(set(groups_test))} groups"
            )

            data_splits[num_split] = {
                "test": partition_test,
                "train": partition_train,
            }

        assert len(all_test_groups) == len(set(all_test_groups))
        assert len(set(self.dataset[self.datasplit_col]).symmetric_difference(set(all_test_groups))) == 0
        assert len(all_test_indices) == len(set(all_test_indices))
        assert len(self.dataset) == len(all_test_indices)

        logger.info("Data splits creation completed successfully")
        return data_splits


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
