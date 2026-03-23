"""Data splitting utilities for nested cross-validation."""

<<<<<<< HEAD
=======
from typing import Any

>>>>>>> 1de9f14 (Replace custom datasplit logic by scikit learns built in functions, fixes #84, #384)
import pandas as pd
from attrs import Factory, define, field, frozen, validators
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from .logger import get_logger
from .types import LogGroup

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

    Creates group-aware cross-validation splits using sklearn's built-in splitters.

    - GroupKFold is used for non-stratified splitting.
    - StratifiedGroupKFold is used when ``stratification_col`` is provided.

    Splits are always created on ``datasplit_group`` to prevent group leakage.
    ``stratification_col`` (if provided) controls class-balance stratification.
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
<<<<<<< HEAD
        dataset_unique = self.dataset.drop_duplicates(subset=DATASPLIT_COL, keep="first", inplace=False)
        dataset_unique.reset_index(drop=True, inplace=True)

        kf: KFold | StratifiedKFold
=======
        groups = self.dataset[DATASPLIT_COL]
        num_groups = groups.nunique()

        splitter: GroupKFold | StratifiedGroupKFold
>>>>>>> 1de9f14 (Replace custom datasplit logic by scikit learns built in functions, fixes #84, #384)
        split_method: str
        if self.stratification_col:
            splitter = StratifiedGroupKFold(
                n_splits=self.num_folds,
                shuffle=True,
                random_state=datasplit_seed,
            )
            stratification_target = self.dataset[self.stratification_col]
            split_method = "StratifiedGroupKFold"
            split_iterator = splitter.split(self.dataset, stratification_target, groups=groups)
        else:
            # Runtime sklearn (>=1.6) supports shuffle/random_state on GroupKFold,
            # but currently available sklearn stubs lag behind this signature and
            # can cause false-positive mypy errors in pre-commit.
            group_kfold_kwargs: dict[str, Any] = {
                "n_splits": self.num_folds,
                "shuffle": True,
                "random_state": datasplit_seed,
            }
            splitter = GroupKFold(**group_kfold_kwargs)
            split_method = "GroupKFold"
            split_iterator = splitter.split(self.dataset, groups=groups)

        logger.info(
            f"{len(self.dataset)} rows, {num_groups} groups (column: {DATASPLIT_COL}), "
            f"{split_method}, {self.num_folds} folds, seed {datasplit_seed}"
        )

        raw_splits: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
        all_test_indices = []
        all_test_groups = []

        for num_split, (train_ind, test_ind) in enumerate(split_iterator):
            groups_train = set(self.dataset.iloc[train_ind][DATASPLIT_COL])
            groups_test = set(self.dataset.iloc[test_ind][DATASPLIT_COL])
            assert groups_train.intersection(groups_test) == set()
            all_test_groups.extend(list(groups_test))

            partition_train = self.dataset.iloc[train_ind]
            partition_test = self.dataset.iloc[test_ind]
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
