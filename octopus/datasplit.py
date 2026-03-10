"""Data splitting utilities for nested cross-validation."""

import random

import numpy as np
import pandas as pd
from attrs import Factory, define, field, frozen, validators
from sklearn.model_selection import KFold, StratifiedKFold

from .logger import LogGroup, get_logger

logger = get_logger()


@frozen
class OuterSplit:
    """One fold of outer cross-validation: traindev + held-out test."""

    traindev: pd.DataFrame
    test: pd.DataFrame


@frozen
class InnerSplit:
    """One fold of inner cross-validation: train + dev/validation."""

    train: pd.DataFrame
    dev: pd.DataFrame


OuterSplits = dict[int, OuterSplit]
InnerSplits = dict[int, InnerSplit]

DATASPLIT_COL = "datasplit_group"


@define
class DataSplit:
    """Data Split.

    We don't use groupKFold as it does not offer the shuffle option.
    The StratifiedGroupKfold might work as an alternative (check examples).
    StratifiedGroupKfold is not available for sklearn 0.24.3
    which is required for Auto-Sklearn 0.15.
    stratification_col: contains the group info used for stratification
    """

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
        self.dataset.reset_index(drop=True, inplace=True)

    def get_outer_splits(self) -> OuterSplits:
        """Get outer cross-validation splits (traindev / test)."""
        combined_values = self._get_combined_values(name_a="Train/Dev", name_b="Test")
        return {i: OuterSplit(traindev=train, test=test) for i, (train, test) in enumerate(combined_values)}

    def get_inner_splits(self) -> InnerSplits:
        """Get inner cross-validation splits (train / dev)."""
        combined_values = self._get_combined_values(name_a="Train", name_b="Dev")
        return {i: InnerSplit(train=train, dev=test) for i, (train, test) in enumerate(combined_values)}

    def _get_combined_values(self, name_a: str, name_b: str) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Get combined raw split values from all datasplit seeds."""
        combined_values: list[tuple[pd.DataFrame, pd.DataFrame]] = []

        for seed in self.seeds:
            result = self._single_seed_datasplits(seed, name_a, name_b)
            for key in sorted(result.keys()):
                combined_values.append(result[key])

        return combined_values

    def _single_seed_datasplits(
        self, datasplit_seed, name_a: str, name_b: str
    ) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
        """Get datasplits for single seed."""
        random.seed(datasplit_seed)
        np.random.seed(datasplit_seed)

        dataset_unique = self.dataset.drop_duplicates(subset=DATASPLIT_COL, keep="first", inplace=False)
        dataset_unique.reset_index(drop=True, inplace=True)

        kf: KFold | StratifiedKFold
        split_method: str
        if self.stratification_col:
            kf = StratifiedKFold(
                n_splits=self.num_folds,
                shuffle=True,
                random_state=datasplit_seed,
            )

            stratification_target = dataset_unique[self.stratification_col]
            split_method = "StratifiedKFold"
        else:
            kf = KFold(
                n_splits=self.num_folds,
                shuffle=True,
                random_state=datasplit_seed,
            )
            stratification_target = None
            split_method = "KFold"

        logger.info(
            f"{len(self.dataset)} rows, {len(dataset_unique)} groups (column: {DATASPLIT_COL}), "
            f"{split_method}, {self.num_folds} folds, seed {datasplit_seed}"
        )

        raw_splits: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
        all_test_indices = []
        all_test_groups = []

        for num_split, (train_ind, test_ind) in enumerate(kf.split(dataset_unique, stratification_target)):  # type: ignore
            groups_train = set(dataset_unique.iloc[train_ind][DATASPLIT_COL])
            groups_test = set(dataset_unique.iloc[test_ind][DATASPLIT_COL])
            assert groups_train.intersection(groups_test) == set()
            all_test_groups.extend(list(groups_test))

            partition_train = self.dataset[self.dataset[DATASPLIT_COL].isin(groups_train)]
            partition_test = self.dataset[self.dataset[DATASPLIT_COL].isin(groups_test)]
            assert set(partition_train.index).intersection(partition_test.index) == set()
            all_test_indices.extend(partition_test.index.tolist())

            partition_train.reset_index(drop=True, inplace=True)
            partition_test.reset_index(drop=True, inplace=True)

            logger.info(
                f"{self.process_id} {num_split} created: "
                f"{name_a} - {len(partition_train)} rows, "
                f"{len(set(groups_train))} groups | "
                f"{name_b} - {len(partition_test)} rows, "
                f"{len(set(groups_test))} groups"
            )

            raw_splits[num_split] = (partition_train, partition_test)

        assert len(all_test_groups) == len(set(all_test_groups))
        assert len(set(self.dataset[DATASPLIT_COL]).symmetric_difference(set(all_test_groups))) == 0
        assert len(all_test_indices) == len(set(all_test_indices))
        assert len(self.dataset) == len(all_test_indices)

        return raw_splits
