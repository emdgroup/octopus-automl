"""Tests for the data splitting logic."""

import random

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from octopus.datasplit import (
    DATASPLIT_COL,
    DataSplit,
    InnerSplit,
    InnerSplits,
    OuterSplit,
    OuterSplits,
    validate_class_coverage,
)
from octopus.exceptions import SingleClassSplitError


def _grouped_df() -> pd.DataFrame:
    """Return a tiny grouped dataset with stable row IDs."""
    return pd.DataFrame(
        {
            "row_id": [10, 11, 20, 30, 31, 40],
            "feature": [1.0, 1.1, 2.0, 3.0, 3.1, 4.0],
            "target": [0, 0, 1, 0, 0, 1],
            DATASPLIT_COL: [0, 0, 1, 2, 2, 3],
        }
    )


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by row_id and reset index for reliable frame comparisons."""
    return df.sort_values("row_id").reset_index(drop=True)


def test_get_outer_splits_keeps_groups_together_and_covers_all_rows_once():
    """Outer splits keep each group intact and cover each row once in test."""
    df = _grouped_df()

    splits = DataSplit(
        dataset=df,
        seeds=[7],
        n_splits=2,
    ).get_outer_splits()

    assert len(splits) == 2
    assert all(isinstance(split, OuterSplit) for split in splits.values())

    all_test_rows = []
    test_split_by_row = {}

    for split_id, split in splits.items():
        train_groups = set(split.traindev[DATASPLIT_COL])
        test_groups = set(split.test[DATASPLIT_COL])

        assert train_groups.isdisjoint(test_groups)

        for row_id in split.test["row_id"]:
            all_test_rows.append(row_id)
            test_split_by_row[row_id] = split_id

    assert sorted(all_test_rows) == [10, 11, 20, 30, 31, 40]
    assert test_split_by_row[10] == test_split_by_row[11]
    assert test_split_by_row[30] == test_split_by_row[31]


def test_get_inner_splits_keeps_groups_together_and_covers_all_rows_once():
    """Inner splits follow the same rules as outer splits, just train/dev named."""
    df = _grouped_df()

    splits = DataSplit(
        dataset=df,
        seeds=[7],
        n_splits=2,
    ).get_inner_splits()

    assert len(splits) == 2
    assert all(isinstance(split, InnerSplit) for split in splits.values())

    all_dev_rows = []

    for split in splits.values():
        train_groups = set(split.train[DATASPLIT_COL])
        dev_groups = set(split.dev[DATASPLIT_COL])

        assert train_groups.isdisjoint(dev_groups)
        all_dev_rows.extend(split.dev["row_id"].tolist())

    assert sorted(all_dev_rows) == [10, 11, 20, 30, 31, 40]


def test_multiple_seeds_concatenate_seed_results_in_seed_then_split_order():
    """With multiple seeds, results are returned seed by seed, then split by split."""
    df = _grouped_df()

    seed_11 = DataSplit(dataset=df.copy(), seeds=[11], n_splits=2).get_outer_splits()
    seed_22 = DataSplit(dataset=df.copy(), seeds=[22], n_splits=2).get_outer_splits()

    combined = DataSplit(
        dataset=df.copy(),
        seeds=[11, 22],
        n_splits=2,
    ).get_outer_splits()

    assert len(combined) == 4

    pdt.assert_frame_equal(_norm(combined[0].test), _norm(seed_11[0].test))
    pdt.assert_frame_equal(_norm(combined[1].test), _norm(seed_11[1].test))
    pdt.assert_frame_equal(_norm(combined[2].test), _norm(seed_22[0].test))
    pdt.assert_frame_equal(_norm(combined[3].test), _norm(seed_22[1].test))


def test_same_seed_is_deterministic_for_outer_splits():
    """Running the same seed twice should give the exact same partitions."""
    df = _grouped_df()

    first = DataSplit(dataset=df.copy(), seeds=[123], n_splits=2).get_outer_splits()
    second = DataSplit(dataset=df.copy(), seeds=[123], n_splits=2).get_outer_splits()

    for i in range(2):
        pdt.assert_frame_equal(_norm(first[i].traindev), _norm(second[i].traindev))
        pdt.assert_frame_equal(_norm(first[i].test), _norm(second[i].test))


def test_stratified_splitting_preserves_class_presence_across_splits():
    """In this balanced setup, each stratified test split should include both classes."""
    df = pd.DataFrame(
        {
            "row_id": list(range(8)),
            "feature": list(range(8)),
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
            DATASPLIT_COL: list(range(8)),
        }
    )

    splits = DataSplit(
        dataset=df,
        seeds=[0],
        n_splits=4,
        stratification_col="target",
    ).get_outer_splits()

    assert len(splits) == 4

    for split in splits.values():
        assert len(split.test) == 2
        assert set(split.test["target"]) == {0, 1}


def test_stratified_group_split_with_mixed_label_group_has_expected_target_counts():
    """Regression test: seed 0 should yield the known stratified target counts per split."""
    df = pd.DataFrame(
        {
            "row_id": list(range(32)),
            "feature": [1.0] * 32,
            # group 0 is mixed (8x class 0, 2x class 1)
            "target": ([0] * 8) + ([1] * 2) + ([1] * 10) + ([0] * 10) + ([0] * 2),
            DATASPLIT_COL: ([0] * 10) + ([1] * 10) + ([2] * 10) + ([3] * 2),
        }
    )

    splits = DataSplit(
        dataset=df,
        seeds=[0],
        n_splits=2,
        stratification_col="target",
    ).get_outer_splits()

    assert len(splits) == 2
    assert splits[0].test["target"].value_counts().to_dict() == {0: 8, 1: 2}
    assert splits[1].test["target"].value_counts().to_dict() == {0: 12, 1: 10}


def test_datasplit_resets_input_index_in_place():
    """The splitter resets the input index in place during initialization."""
    df = _grouped_df()
    df.index = [100, 101, 102, 103, 104, 105]

    splitter = DataSplit(
        dataset=df,
        seeds=[0],
        n_splits=2,
    )

    assert splitter.dataset.index.tolist() == [0, 1, 2, 3, 4, 5]
    assert df.index.tolist() == [0, 1, 2, 3, 4, 5]


def test_missing_datasplit_group_column_raises_key_error():
    """Missing datasplit_group should fail immediately."""
    df = pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "feature": [1.0, 2.0, 3.0],
            "target": [0, 1, 0],
        }
    )

    with pytest.raises(KeyError):
        DataSplit(
            dataset=df,
            seeds=[0],
            n_splits=2,
        ).get_outer_splits()


def test_num_splits_greater_than_number_of_groups_raises_value_error():
    """KFold should reject a split count larger than the number of groups."""
    df = pd.DataFrame(
        {
            "row_id": [0, 1],
            "feature": [1.0, 2.0],
            "target": [0, 1],
            DATASPLIT_COL: [0, 1],
        }
    )

    with pytest.raises(ValueError, match="n_splits=3"):
        DataSplit(
            dataset=df,
            seeds=[0],
            n_splits=3,
        ).get_outer_splits()


def test_stratified_split_warns_when_class_has_too_few_groups():
    """StratifiedKFold emits a warning when one class is too small."""
    df = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "feature": [1.0, 2.0, 3.0, 4.0],
            "target": [0, 0, 0, 1],
            DATASPLIT_COL: [0, 1, 2, 3],
        }
    )

    with pytest.warns(UserWarning, match="least populated class"):
        splits = DataSplit(
            dataset=df,
            seeds=[0],
            n_splits=2,
            stratification_col="target",
        ).get_outer_splits()

    assert len(splits) == 2


def test_each_split_traindev_plus_test_equals_all_rows():
    """Within each split, traindev + test must equal the full dataset."""
    df = _grouped_df()

    splits = DataSplit(
        dataset=df,
        seeds=[7],
        n_splits=2,
    ).get_outer_splits()

    all_row_ids = sorted(df["row_id"].tolist())

    for split in splits.values():
        split_rows = sorted(split.traindev["row_id"].tolist() + split.test["row_id"].tolist())
        assert split_rows == all_row_ids


def test_each_split_train_plus_dev_equals_all_rows_inner():
    """Within each inner split, train + dev must equal the full dataset."""
    df = _grouped_df()

    splits = DataSplit(
        dataset=df,
        seeds=[7],
        n_splits=2,
    ).get_inner_splits()

    all_row_ids = sorted(df["row_id"].tolist())

    for split in splits.values():
        split_rows = sorted(split.train["row_id"].tolist() + split.dev["row_id"].tolist())
        assert split_rows == all_row_ids


def test_datasplit_does_not_corrupt_global_random_state():
    """DataSplit must not leave numpy/random global state altered for the caller."""
    random.seed(999)
    np.random.seed(999)

    random_before = random.random()
    numpy_before = np.random.random()

    # Reset to the same state and run DataSplit in between
    random.seed(999)
    np.random.seed(999)

    DataSplit(
        dataset=_grouped_df(),
        seeds=[42],
        n_splits=2,
    ).get_outer_splits()

    random_after = random.random()
    numpy_after = np.random.random()

    assert random_before == random_after
    assert numpy_before == numpy_after


def test_inner_splits_with_stratification_preserves_class_presence():
    """Stratified inner splits should include both classes in each dev split."""
    df = pd.DataFrame(
        {
            "row_id": list(range(8)),
            "feature": list(range(8)),
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
            DATASPLIT_COL: list(range(8)),
        }
    )

    splits = DataSplit(
        dataset=df,
        seeds=[0],
        n_splits=4,
        stratification_col="target",
    ).get_inner_splits()

    assert len(splits) == 4

    for split in splits.values():
        assert len(split.dev) == 2
        assert set(split.dev["target"]) == {0, 1}


def test_single_split_raises_value_error():
    """n_splits=1 is rejected by sklearn KFold."""
    df = _grouped_df()

    with pytest.raises(ValueError, match=r"k-fold.*n_splits=2 or more"):
        DataSplit(
            dataset=df,
            seeds=[0],
            n_splits=1,
        ).get_outer_splits()


def test_three_seeds_produce_correct_index_numbering():
    """Three seeds x 2 splits -> 6 entries keyed 0..5 matching individual runs."""
    df = _grouped_df()

    individual = {}
    for seed in [10, 20, 30]:
        individual[seed] = DataSplit(dataset=df.copy(), seeds=[seed], n_splits=2).get_outer_splits()

    combined = DataSplit(
        dataset=df.copy(),
        seeds=[10, 20, 30],
        n_splits=2,
    ).get_outer_splits()

    assert list(combined.keys()) == [0, 1, 2, 3, 4, 5]

    pdt.assert_frame_equal(_norm(combined[0].test), _norm(individual[10][0].test))
    pdt.assert_frame_equal(_norm(combined[1].test), _norm(individual[10][1].test))
    pdt.assert_frame_equal(_norm(combined[2].test), _norm(individual[20][0].test))
    pdt.assert_frame_equal(_norm(combined[3].test), _norm(individual[20][1].test))
    pdt.assert_frame_equal(_norm(combined[4].test), _norm(individual[30][0].test))
    pdt.assert_frame_equal(_norm(combined[5].test), _norm(individual[30][1].test))


def test_different_seeds_produce_different_splits():
    """Two different seeds must produce at least one different partition."""
    df = pd.DataFrame(
        {
            "row_id": list(range(20)),
            "feature": list(range(20)),
            "target": [0, 1] * 10,
            DATASPLIT_COL: list(range(20)),
        }
    )

    splits_a = DataSplit(dataset=df.copy(), seeds=[0], n_splits=5).get_outer_splits()
    splits_b = DataSplit(dataset=df.copy(), seeds=[999], n_splits=5).get_outer_splits()

    some_differ = False
    for i in range(5):
        test_a = set(splits_a[i].test["row_id"])
        test_b = set(splits_b[i].test["row_id"])
        if test_a != test_b:
            some_differ = True
            break

    assert some_differ, "Different seeds should produce at least one different split"


def test_single_group_with_two_splits_raises_value_error():
    """A dataset with only one group cannot be split into 2 splits."""
    df = pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "feature": [1.0, 2.0, 3.0],
            "target": [0, 1, 0],
            DATASPLIT_COL: [0, 0, 0],
        }
    )

    with pytest.raises(ValueError, match="n_splits=2"):
        DataSplit(
            dataset=df,
            seeds=[0],
            n_splits=2,
        ).get_outer_splits()


def _unequal_groups_df() -> pd.DataFrame:
    """Return a dataset with 4 groups of very different sizes."""
    return pd.DataFrame(
        {
            "row_id": list(range(104)),
            "feature": [1.0] * 104,
            "target": [0] * 104,
            # group 0: 50 rows, group 1: 50 rows, group 2: 2 rows, group 3: 2 rows
            DATASPLIT_COL: ([0] * 50) + ([1] * 50) + ([2] * 2) + ([3] * 2),
        }
    )


def test_unequal_group_sizes_balances_group_count_across_splits():
    """With unequal group sizes, each split gets the same number of groups."""
    df = _unequal_groups_df()

    splits = DataSplit(
        dataset=df,
        seeds=[0],
        n_splits=2,
    ).get_outer_splits()

    for split in splits.values():
        train_groups = set(split.traindev[DATASPLIT_COL])
        test_groups = set(split.test[DATASPLIT_COL])
        assert train_groups.isdisjoint(test_groups)

    test_group_counts = [len(set(s.test[DATASPLIT_COL])) for s in splits.values()]
    assert test_group_counts == [2, 2]


def test_validate_class_coverage_raises_on_single_class_outer_split():
    """validate_class_coverage raises SingleClassSplitError when an outer split has one class."""
    splits: OuterSplits = {
        0: OuterSplit(
            traindev=pd.DataFrame({"target": [0, 0, 1, 1], DATASPLIT_COL: [0, 0, 1, 1]}),
            test=pd.DataFrame({"target": [1, 1], DATASPLIT_COL: [2, 2]}),
        ),
    }

    with pytest.raises(SingleClassSplitError, match=r"Split 0 test.*only class"):
        validate_class_coverage(splits, "target")


def test_validate_class_coverage_raises_on_single_class_inner_split():
    """validate_class_coverage raises SingleClassSplitError for inner splits too."""
    splits: InnerSplits = {
        0: InnerSplit(
            train=pd.DataFrame({"target": [0, 0, 1, 1]}),
            dev=pd.DataFrame({"target": [0, 1]}),
        ),
        1: InnerSplit(
            train=pd.DataFrame({"target": [0, 0, 0, 0]}),
            dev=pd.DataFrame({"target": [0, 1]}),
        ),
    }

    with pytest.raises(SingleClassSplitError, match=r"Split 1 train.*only class"):
        validate_class_coverage(splits, "target")


def test_validate_class_coverage_passes_when_all_classes_present():
    """No error when all classes appear in every partition."""
    splits: OuterSplits = {
        0: OuterSplit(
            traindev=pd.DataFrame({"target": [0, 1], DATASPLIT_COL: [0, 1]}),
            test=pd.DataFrame({"target": [0, 1], DATASPLIT_COL: [2, 3]}),
        ),
        1: OuterSplit(
            traindev=pd.DataFrame({"target": [0, 1], DATASPLIT_COL: [2, 3]}),
            test=pd.DataFrame({"target": [0, 1], DATASPLIT_COL: [0, 1]}),
        ),
    }

    validate_class_coverage(splits, "target")


def test_validate_class_coverage_with_expected_classes_missing_raises():
    """Multiclass split missing a class raises when expected_classes is passed."""
    splits: OuterSplits = {
        0: OuterSplit(
            traindev=pd.DataFrame({"target": [0, 1, 2, 0, 1, 2], DATASPLIT_COL: [0, 0, 0, 1, 1, 1]}),
            test=pd.DataFrame({"target": [0, 1, 0, 1], DATASPLIT_COL: [2, 2, 3, 3]}),
        ),
    }

    with pytest.raises(SingleClassSplitError, match="missing classes"):
        validate_class_coverage(splits, "target", expected_classes={0, 1, 2})


def test_validate_class_coverage_with_expected_classes_all_present_passes():
    """Multiclass split with all expected classes passes."""
    splits: OuterSplits = {
        0: OuterSplit(
            traindev=pd.DataFrame({"target": [0, 1, 2, 0, 1, 2], DATASPLIT_COL: [0, 0, 0, 1, 1, 1]}),
            test=pd.DataFrame({"target": [0, 1, 2], DATASPLIT_COL: [2, 2, 2]}),
        ),
    }

    validate_class_coverage(splits, "target", expected_classes={0, 1, 2})


def test_validate_class_coverage_without_expected_classes_allows_subset():
    """Without expected_classes, a partition with 2 of 3 classes passes (existing behavior)."""
    splits: OuterSplits = {
        0: OuterSplit(
            traindev=pd.DataFrame({"target": [0, 1, 2, 0, 1, 2], DATASPLIT_COL: [0, 0, 0, 1, 1, 1]}),
            test=pd.DataFrame({"target": [0, 1, 0, 1], DATASPLIT_COL: [2, 2, 3, 3]}),
        ),
    }

    validate_class_coverage(splits, "target")
