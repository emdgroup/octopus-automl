"""Data splitting utilities for nested cross-validation."""

import pandas as pd
from attrs import Factory, define, field, frozen, validators
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from .exceptions import SingleClassSplitError
from .logger import get_logger
from .types import LogGroup

logger = get_logger()


@frozen
class OuterSplit:
    """One split of outer cross-validation: traindev + held-out test."""

    traindev: pd.DataFrame
    test: pd.DataFrame


@frozen
class InnerSplit:
    """One split of inner cross-validation: train + dev/validation."""

    train: pd.DataFrame
    dev: pd.DataFrame


OuterSplits = dict[int, OuterSplit]
InnerSplits = dict[int, InnerSplit]

DATASPLIT_COL = "datasplit_group"


def validate_class_coverage(
    splits: OuterSplits | InnerSplits,
    target_col: str,
) -> None:
    """Verify no split partition contains only a single class.

    Raises SingleClassSplitError if any partition has just one unique class,
    which would cause degenerate model training or undefined metrics.
    """
    for split_id, split in splits.items():
        if isinstance(split, OuterSplit):
            partitions = {"traindev": split.traindev, "test": split.test}
        else:
            partitions = {"train": split.train, "dev": split.dev}

        for part_name, part_df in partitions.items():
            unique_classes = part_df[target_col].unique()
            if len(unique_classes) <= 1:
                raise SingleClassSplitError(
                    f"Split {split_id} {part_name} partition contains only class(es) "
                    f"{sorted(unique_classes)}. "
                    "Try: changing `inner_split_seeds` or `outer_split_seed`, "
                    "reducing `n_inner_splits` or `n_outer_splits`, "
                    "or setting `stratification_col` to the target column "
                    "for balanced splits."
                )


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
    n_splits: int = field(validator=[validators.instance_of(int)])
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
        groups = self.dataset[DATASPLIT_COL]
        n_groups = groups.nunique()

        splitter: GroupKFold | StratifiedGroupKFold
        if self.stratification_col is not None:
            splitter = StratifiedGroupKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=datasplit_seed,
            )
            stratification_target = self.dataset[self.stratification_col]
            split_iterator = splitter.split(self.dataset, stratification_target, groups=groups)
        else:
            # Runtime sklearn (>=1.6) supports shuffle/random_state on GroupKFold,
            # but currently available sklearn stubs lag behind this signature.
            splitter = GroupKFold(
                n_splits=self.n_splits,
                shuffle=True,  # type: ignore[call-arg]  # sklearn stubs lag behind sklearn version (>=1.6)
                random_state=datasplit_seed,
            )
            split_iterator = splitter.split(self.dataset, groups=groups)

        logger.info(
            f"{len(self.dataset)} rows, {n_groups} groups (column: {DATASPLIT_COL}), "
            f"{type(splitter).__name__}, {self.n_splits} splits, seed {datasplit_seed}"
        )

        raw_splits: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
        all_test_indices = []
        all_test_groups = []

        for split_idx, (train_ind, test_ind) in enumerate(split_iterator):
            partition_train = self.dataset.iloc[train_ind]
            partition_test = self.dataset.iloc[test_ind]

            groups_train = set(partition_train[DATASPLIT_COL])
            groups_test = set(partition_test[DATASPLIT_COL])
            if groups_train & groups_test:
                raise RuntimeError(
                    f"Group leakage detected: groups {groups_train & groups_test} appear in both train and test"
                )
            all_test_groups.extend(list(groups_test))

            if set(partition_train.index) & set(partition_test.index):
                raise RuntimeError("Index overlap between train and test partitions")
            all_test_indices.extend(partition_test.index.tolist())

            partition_train.reset_index(drop=True, inplace=True)
            partition_test.reset_index(drop=True, inplace=True)

            logger.info(
                f"{self.process_id} {split_idx} created: "
                f"{name_a} - {len(partition_train)} rows, "
                f"{len(set(groups_train))} groups | "
                f"{name_b} - {len(partition_test)} rows, "
                f"{len(set(groups_test))} groups"
            )

            raw_splits[split_idx] = (partition_train, partition_test)

        if len(all_test_groups) != len(set(all_test_groups)):
            raise RuntimeError("Duplicate groups across test splits")
        if set(self.dataset[DATASPLIT_COL]).symmetric_difference(set(all_test_groups)):
            raise RuntimeError("Not all groups covered by test splits")
        if len(all_test_indices) != len(set(all_test_indices)):
            raise RuntimeError("Duplicate row indices across test splits")
        if len(self.dataset) != len(all_test_indices):
            raise RuntimeError(f"Test splits contain {len(all_test_indices)} rows, expected {len(self.dataset)}")

        return raw_splits
