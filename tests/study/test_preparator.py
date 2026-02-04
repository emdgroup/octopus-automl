"""Test OctoData preparator."""

import numpy as np
import pandas as pd
import pytest

from octopus.study.data_preparator import OctoDataPreparator


@pytest.fixture
def sample_data():
    """Create sample data."""
    return pd.DataFrame(
        {
            "feature1": [1, 2, 1, 2, 3, 3, 4, 4],
            "feature2": ["a", "b", "a", "b", "c", "c", "d", "e"],
            "target": [0, 1, 0, 1, 1, 0, 1, 1],
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s5", "s6", "s6"],
            "bool_col": [True, False, True, False, True, True, False, True],
            "null_col": ["none", "null", "nan", "na", "", "\x00", "\x00\x00", "n/a"],
            "inf_col": ["inf", "-infinity", "inf", "-inf", "∞", "-infinity", 5, 6],
        }
    )


@pytest.fixture
def octo_preparator(sample_data):
    """Create OctoDataPreparator instance from sample data."""
    return OctoDataPreparator(
        data=sample_data,
        feature_cols=["feature1", "feature2", "bool_col", "null_col", "inf_col"],
        target_cols=["target"],
        sample_id="sample_id",
        row_id=None,
        target_assignments={},
    )


def test_prepare(octo_preparator):
    """Test prepare function."""
    prepared = octo_preparator.prepare()

    assert isinstance(prepared.data, pd.DataFrame)
    assert isinstance(prepared.feature_cols, list)
    assert isinstance(prepared.row_id, str)
    assert isinstance(prepared.target_assignments, dict)


def test_sort_features(octo_preparator):
    """Test sort features function."""
    octo_preparator.feature_cols = [
        "feature1",
        "feature2",
        "bool_col",
        "null_col",
        "inf_col",
        "a",
        "aa",
        "aaa",
        "b",
    ]
    octo_preparator._sort_features()
    assert octo_preparator.feature_cols == [
        "a",
        "b",
        "aa",
        "aaa",
        "inf_col",
        "bool_col",
        "feature1",
        "feature2",
        "null_col",
    ]


def test_set_target_assignments(octo_preparator):
    """Test set target assignments function."""
    octo_preparator._set_target_assignments()
    assert octo_preparator.target_assignments == {"default": "target"}


def test_remove_singlevalue_features(octo_preparator):
    """Test remove single value features function."""
    octo_preparator.data["single_value"] = [1, 1, 1, 1, 1, 1, 1, 1]
    octo_preparator.feature_cols.append("single_value")
    octo_preparator._remove_singlevalue_features()
    assert "single_value" not in octo_preparator.feature_cols


def test_transform_bool_to_int(octo_preparator):
    """Test transform bool to int function."""
    octo_preparator._transform_bool_to_int()
    assert octo_preparator.data["bool_col"].dtype == int
    assert octo_preparator.data["bool_col"].tolist() == [1, 0, 1, 0, 1, 1, 0, 1]


def test_create_row_id(octo_preparator):
    """Test create row id function."""
    octo_preparator._create_row_id()
    assert "row_id" in octo_preparator.data.columns
    assert octo_preparator.row_id == "row_id"
    assert octo_preparator.data["row_id"].tolist() == list(range(8))


def test_add_group_features(octo_preparator):
    """Test add group features function."""
    octo_preparator._standardize_null_values()
    octo_preparator._standardize_inf_values()
    octo_preparator._transform_bool_to_int()
    octo_preparator._add_group_features()

    assert "group_features" in octo_preparator.data.columns
    assert "group_sample_and_features" in octo_preparator.data.columns
    assert octo_preparator.data.loc[0, "group_features"] == octo_preparator.data.loc[2, "group_features"]
    assert octo_preparator.data.loc[1, "group_features"] == octo_preparator.data.loc[3, "group_features"]
    assert octo_preparator.data.loc[4, "group_features"] != octo_preparator.data.loc[5, "group_features"]
    assert octo_preparator.data.loc[6, "group_features"] != octo_preparator.data.loc[7, "group_features"]
    assert octo_preparator.data["group_features"].nunique() == 6
    assert (
        octo_preparator.data.loc[0, "group_sample_and_features"]
        == octo_preparator.data.loc[2, "group_sample_and_features"]
    )
    assert (
        octo_preparator.data.loc[4, "group_sample_and_features"]
        == octo_preparator.data.loc[5, "group_sample_and_features"]
    )
    assert (
        octo_preparator.data.loc[6, "group_sample_and_features"]
        == octo_preparator.data.loc[7, "group_sample_and_features"]
    )
    assert octo_preparator.data["group_sample_and_features"].nunique() == 4
    assert octo_preparator.data.index.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert (
        octo_preparator.data.loc[6, "group_sample_and_features"]
        == octo_preparator.data.loc[7, "group_sample_and_features"]
    )
    assert octo_preparator.data.loc[6, "group_features"] != octo_preparator.data.loc[7, "group_features"]
    assert octo_preparator.data.loc[0, "group_features"] == octo_preparator.data.loc[2, "group_features"]
    assert (
        octo_preparator.data.loc[0, "group_sample_and_features"]
        == octo_preparator.data.loc[2, "group_sample_and_features"]
    )


def test_standardize_null_values(octo_preparator):
    """Test standardize null values function."""
    octo_preparator._standardize_null_values()
    assert octo_preparator.data["null_col"].isna().all()


def test_standardize_inf_values(octo_preparator):
    """Test standardize inf values function."""
    octo_preparator._standardize_inf_values()
    assert np.isinf(octo_preparator.data["inf_col"].iloc[0])
    assert np.isinf(octo_preparator.data["inf_col"].iloc[1])
    assert np.isinf(octo_preparator.data["inf_col"].iloc[4])
    assert np.isinf(octo_preparator.data["inf_col"].iloc[5])


def test_prepare_full_process(octo_preparator):
    """Test preparation function."""
    prepared = octo_preparator.prepare()

    assert "row_id" in prepared.data.columns
    assert "group_features" in prepared.data.columns
    assert "group_sample_and_features" in prepared.data.columns
    assert prepared.data["bool_col"].dtype == int
    assert prepared.data["null_col"].isna().all()
    assert np.isinf(prepared.data["inf_col"].iloc[0])
    assert "single_value" not in prepared.feature_cols
    assert prepared.target_assignments == {"default": "target"}


def test_add_group_features_with_categorical_and_nan():
    """Test add group features with categorical columns containing NaN values.

    This test specifically addresses the issue where categorical columns with
    NaN values would raise a TypeError when trying to fill with a placeholder
    that wasn't in the category list.
    """
    data = pd.DataFrame(
        {
            "cat_feature": pd.Categorical(["a", "b", "a", None, "c", None]),
            "num_feature": [1, 2, 1, 3, 4, 3],
            "target": [0, 1, 0, 1, 1, 0],
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["cat_feature", "num_feature"],
        target_cols=["target"],
        sample_id="sample_id",
        row_id=None,
        target_assignments={},
    )

    prep._add_group_features()

    assert "group_features" in prep.data.columns
    assert "group_sample_and_features" in prep.data.columns
    assert prep.data.loc[0, "group_features"] == prep.data.loc[2, "group_features"]
    assert prep.data.loc[3, "group_features"] == prep.data.loc[5, "group_features"]
    assert prep.data["group_features"].nunique() == 4


def test_add_group_features_with_mixed_types_and_nan():
    """Test add group features with mixed column types (categorical, numeric, string) and NaN values."""
    data = pd.DataFrame(
        {
            "cat_col": pd.Categorical(["x", "y", None, "x", None]),
            "num_col": [1.5, 2.5, np.nan, 1.5, np.nan],
            "str_col": ["alpha", "beta", "alpha", "alpha", "beta"],
            "inf_col": [np.inf, -np.inf, 1.0, np.inf, -np.inf],
            "target": [0, 1, 0, 1, 1],
            "sample_id": ["s1", "s2", "s3", "s4", "s5"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["cat_col", "num_col", "str_col", "inf_col"],
        target_cols=["target"],
        sample_id="sample_id",
        row_id=None,
        target_assignments={},
    )

    prep._add_group_features()

    assert "group_features" in prep.data.columns
    assert "group_sample_and_features" in prep.data.columns
    assert prep.data.loc[0, "group_features"] == prep.data.loc[3, "group_features"]
    assert prep.data["group_features"].nunique() == 4


def test_add_group_features_with_all_nan_column():
    """Test grouping when one column is entirely NaN.

    This edge case ensures that columns with all NaN values are handled properly
    and rows are still grouped correctly based on other features.
    """
    data = pd.DataFrame(
        {
            "all_nan": [np.nan, np.nan, np.nan, np.nan],
            "feat2": [1, 2, 1, 2],
            "target": [0, 1, 0, 1],
            "sample_id": ["s1", "s2", "s3", "s4"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["all_nan", "feat2"],
        target_cols=["target"],
        sample_id="sample_id",
        row_id=None,
        target_assignments={},
    )

    prep._add_group_features()

    assert "group_features" in prep.data.columns
    assert prep.data.loc[0, "group_features"] == prep.data.loc[2, "group_features"]
    assert prep.data.loc[1, "group_features"] == prep.data.loc[3, "group_features"]
    assert prep.data["group_features"].nunique() == 2


def test_add_group_features_same_sample_different_features():
    """Test grouping when rows have same sample_id but different features.

    This ensures that group_features groups by features only (ignores sample_id),
    while group_sample_and_features correctly groups rows with same sample_id.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4],
            "feat2": ["a", "b", "c", "d"],
            "target": [0, 1, 0, 1],
            "sample_id": ["s1", "s1", "s2", "s2"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        target_cols=["target"],
        sample_id="sample_id",
        row_id=None,
        target_assignments={},
    )

    prep._add_group_features()

    assert prep.data["group_features"].nunique() == 4
    assert prep.data.loc[0, "group_sample_and_features"] == prep.data.loc[1, "group_sample_and_features"]
    assert prep.data.loc[2, "group_sample_and_features"] == prep.data.loc[3, "group_sample_and_features"]


def test_add_group_features_same_features_different_samples():
    """Test grouping when rows have same features but different sample_ids.

    This ensures that rows with identical features are grouped together
    in both group_features and group_sample_and_features.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 1, 2, 2],
            "feat2": ["a", "a", "b", "b"],
            "target": [0, 1, 0, 1],
            "sample_id": ["s1", "s2", "s3", "s4"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        target_cols=["target"],
        sample_id="sample_id",
        row_id=None,
        target_assignments={},
    )

    prep._add_group_features()

    assert prep.data.loc[0, "group_features"] == prep.data.loc[1, "group_features"]
    assert prep.data.loc[2, "group_features"] == prep.data.loc[3, "group_features"]
    assert prep.data["group_features"].nunique() == 2
    assert prep.data.loc[0, "group_sample_and_features"] == prep.data.loc[1, "group_sample_and_features"]
    assert prep.data.loc[2, "group_sample_and_features"] == prep.data.loc[3, "group_sample_and_features"]


def test_add_group_features_large_dataset_with_duplicates():
    """Test grouping on a larger dataset with many duplicate feature combinations.

    This ensures the grouping logic scales properly and handles duplicates correctly.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 3] * 27,
            "feat2": ["a", "b", "c"] * 27,
            "target": [0, 1] * 40 + [0],
            "sample_id": [f"s{i}" for i in range(81)],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        target_cols=["target"],
        sample_id="sample_id",
        row_id=None,
        target_assignments={},
    )

    prep._add_group_features()

    assert prep.data["group_features"].nunique() == 3
    group_sizes = prep.data["group_features"].value_counts()
    assert sorted(group_sizes.tolist()) == [27, 27, 27]
    mask_1a = (prep.data["feat1"] == 1) & (prep.data["feat2"] == "a")
    groups_1a = prep.data.loc[mask_1a, "group_features"].unique()
    assert len(groups_1a) == 1, "All (1, 'a') rows should be in the same group"


def test_add_group_features_transitive_closure():
    """Test transitive closure in group_sample_and_features.

    Critical scenario: Row 0 and 1 share sample_id, Row 1 and 2 share features.
    Therefore, rows 0, 1, and 2 must ALL be in the same group through transitive
    closure, even though rows 0 and 2 don't directly share sample_id or features.

    Example:
        Row 0: sample_id="s1", features=[1, "a"]
        Row 1: sample_id="s1", features=[2, "b"]  <- shares sample_id with row 0
        Row 2: sample_id="s2", features=[2, "b"]  <- shares features with row 1
        Row 3: sample_id="s3", features=[3, "c"]  <- independent

    Expected result: Rows 0, 1, 2 should all have the same group_sample_and_features
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 2, 3],
            "feat2": ["a", "b", "b", "c"],
            "target": [0, 1, 0, 1],
            "sample_id": ["s1", "s1", "s2", "s3"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        target_cols=["target"],
        sample_id="sample_id",
        row_id=None,
        target_assignments={},
    )

    prep._add_group_features()

    # Row 1 and 2 should have same group_features (identical feature values)
    assert prep.data.loc[1, "group_features"] == prep.data.loc[2, "group_features"]

    # All rows 0, 1, 2 should be in the same group_sample_and_features due to transitive closure
    assert prep.data.loc[0, "group_sample_and_features"] == prep.data.loc[1, "group_sample_and_features"]
    assert prep.data.loc[1, "group_sample_and_features"] == prep.data.loc[2, "group_sample_and_features"]
    assert prep.data.loc[0, "group_sample_and_features"] == prep.data.loc[2, "group_sample_and_features"]

    # Row 3 should be in a different group (no connection to others)
    assert prep.data.loc[3, "group_sample_and_features"] != prep.data.loc[0, "group_sample_and_features"]

    # Should have exactly 2 unique groups: {0,1,2} and {3}
    assert prep.data["group_sample_and_features"].nunique() == 2


def test_add_group_features_transitive_closure_long_chain():
    """Test transitive closure with a longer chain of connections.

    This tests a more complex transitive closure scenario:
        Row 0-1: same sample_id
        Row 1-2: same features
        Row 2-3: same sample_id
        Row 3-4: same features

    All rows 0-4 should end up in the same group.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 2, 3, 3, 9],
            "feat2": ["a", "b", "b", "c", "c", "z"],
            "target": [0, 1, 0, 1, 0, 1],
            "sample_id": ["s1", "s1", "s2", "s2", "s3", "s4"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        target_cols=["target"],
        sample_id="sample_id",
        row_id=None,
        target_assignments={},
    )

    prep._add_group_features()

    # All rows 0-4 should be in the same group through the chain
    for i in range(4):
        assert prep.data.loc[i, "group_sample_and_features"] == prep.data.loc[i + 1, "group_sample_and_features"], (
            f"Row {i} and {i + 1} should be in the same group"
        )

    # Row 5 should be independent
    assert prep.data.loc[5, "group_sample_and_features"] != prep.data.loc[0, "group_sample_and_features"]

    # Should have exactly 2 unique groups
    assert prep.data["group_sample_and_features"].nunique() == 2


def test_add_group_features_same_sample_consecutive():
    """Test (a): Multiple rows with same sample_id - consecutive in table.

    Rows 0, 1, 2 have same sample_id and appear consecutively.
    All should be grouped together.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4, 5],
            "feat2": ["a", "b", "c", "d", "e"],
            "target": [0, 1, 0, 1, 0],
            "sample_id": ["s1", "s1", "s1", "s2", "s3"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        target_cols=["target"],
        sample_id="sample_id",
        row_id=None,
        target_assignments={},
    )

    prep._add_group_features()

    # Rows 0, 1, 2 should be in same group (same sample_id)
    assert prep.data.loc[0, "group_sample_and_features"] == prep.data.loc[1, "group_sample_and_features"]
    assert prep.data.loc[1, "group_sample_and_features"] == prep.data.loc[2, "group_sample_and_features"]

    # Rows 3 and 4 should be in different groups
    assert prep.data.loc[3, "group_sample_and_features"] != prep.data.loc[0, "group_sample_and_features"]
    assert prep.data.loc[4, "group_sample_and_features"] != prep.data.loc[0, "group_sample_and_features"]
    assert prep.data.loc[3, "group_sample_and_features"] != prep.data.loc[4, "group_sample_and_features"]


def test_add_group_features_same_sample_with_gaps():
    """Test (a): Multiple rows with same sample_id - with gaps in table.

    Rows 0, 3, 5 have same sample_id but are not consecutive.
    All should still be grouped together.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4, 5, 6],
            "feat2": ["a", "b", "c", "d", "e", "f"],
            "target": [0, 1, 0, 1, 0, 1],
            "sample_id": ["s1", "s2", "s3", "s1", "s4", "s1"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        target_cols=["target"],
        sample_id="sample_id",
        row_id=None,
        target_assignments={},
    )

    prep._add_group_features()

    # Rows 0, 3, 5 should be in same group (same sample_id "s1")
    assert prep.data.loc[0, "group_sample_and_features"] == prep.data.loc[3, "group_sample_and_features"]
    assert prep.data.loc[3, "group_sample_and_features"] == prep.data.loc[5, "group_sample_and_features"]

    # Other rows should be in different groups
    assert prep.data.loc[1, "group_sample_and_features"] != prep.data.loc[0, "group_sample_and_features"]
    assert prep.data.loc[2, "group_sample_and_features"] != prep.data.loc[0, "group_sample_and_features"]
    assert prep.data.loc[4, "group_sample_and_features"] != prep.data.loc[0, "group_sample_and_features"]


def test_add_group_features_same_features_consecutive():
    """Test (b): Multiple rows with same features - consecutive in table.

    Rows 1, 2, 3 have identical features and appear consecutively.
    All should be grouped together.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 2, 2, 5],
            "feat2": ["a", "b", "b", "b", "e"],
            "target": [0, 1, 0, 1, 0],
            "sample_id": ["s1", "s2", "s3", "s4", "s5"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        target_cols=["target"],
        sample_id="sample_id",
        row_id=None,
        target_assignments={},
    )

    prep._add_group_features()

    # Rows 1, 2, 3 should be in same group (same features)
    assert prep.data.loc[1, "group_sample_and_features"] == prep.data.loc[2, "group_sample_and_features"]
    assert prep.data.loc[2, "group_sample_and_features"] == prep.data.loc[3, "group_sample_and_features"]

    # Rows 0 and 4 should be in different groups
    assert prep.data.loc[0, "group_sample_and_features"] != prep.data.loc[1, "group_sample_and_features"]
    assert prep.data.loc[4, "group_sample_and_features"] != prep.data.loc[1, "group_sample_and_features"]


def test_add_group_features_same_features_with_gaps():
    """Test (b): Multiple rows with same features - with gaps in table.

    Rows 1, 4, 7 have identical features but are not consecutive.
    All should still be grouped together.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4, 2, 5, 6, 2],
            "feat2": ["a", "b", "c", "d", "b", "e", "f", "b"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        target_cols=["target"],
        sample_id="sample_id",
        row_id=None,
        target_assignments={},
    )

    prep._add_group_features()

    # Rows 1, 4, 7 should be in same group (same features [2, "b"])
    assert prep.data.loc[1, "group_sample_and_features"] == prep.data.loc[4, "group_sample_and_features"]
    assert prep.data.loc[4, "group_sample_and_features"] == prep.data.loc[7, "group_sample_and_features"]

    # Other rows should be in different groups
    assert prep.data.loc[0, "group_sample_and_features"] != prep.data.loc[1, "group_sample_and_features"]
    assert prep.data.loc[2, "group_sample_and_features"] != prep.data.loc[1, "group_sample_and_features"]
    assert prep.data.loc[3, "group_sample_and_features"] != prep.data.loc[1, "group_sample_and_features"]


def test_add_group_features_complex_with_gaps():
    """Test (c) + (d): Complex scenario with sample_id and feature connections with gaps.

    Specific test case:
        - Rows 1, 3, 5 have same sample_id "s1"
        - Rows 5, 9, 20 have same features [100, "x"]

    Expected: All rows 1, 3, 5, 9, 20 should be in the same group due to transitive closure:
        1-3-5 connected by sample_id
        5-9-20 connected by features
        Therefore: 1-3-5-9-20 all connected
    """
    # Create a dataframe with 25 rows to accommodate row 20
    data = pd.DataFrame(
        {
            "feat1": [
                10,
                100,
                20,
                100,
                30,
                100,
                40,
                50,
                60,
                100,
                70,
                80,
                90,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                100,
                18,
                19,
                20,
                21,
            ],
            "feat2": [
                "a",
                "x",
                "c",
                "x",
                "e",
                "x",
                "g",
                "h",
                "i",
                "x",
                "k",
                "l",
                "m",
                "n",
                "o",
                "p",
                "q",
                "r",
                "s",
                "t",
                "x",
                "v",
                "w",
                "y",
                "z",
            ],
            "target": [0] * 25,
            "sample_id": [
                "s0",
                "s1",
                "s2",
                "s1",
                "s4",
                "s1",
                "s6",
                "s7",
                "s8",
                "s9",
                "s10",
                "s11",
                "s12",
                "s13",
                "s14",
                "s15",
                "s16",
                "s17",
                "s18",
                "s19",
                "s20",
                "s21",
                "s22",
                "s23",
                "s24",
            ],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        target_cols=["target"],
        sample_id="sample_id",
        row_id=None,
        target_assignments={},
    )

    prep._add_group_features()

    # Rows 1, 3, 5 should be connected by sample_id "s1"
    assert prep.data.loc[1, "group_sample_and_features"] == prep.data.loc[3, "group_sample_and_features"]
    assert prep.data.loc[3, "group_sample_and_features"] == prep.data.loc[5, "group_sample_and_features"]

    # Rows 5, 9, 20 should be connected by features [100, "x"]
    assert prep.data.loc[5, "group_sample_and_features"] == prep.data.loc[9, "group_sample_and_features"]
    assert prep.data.loc[9, "group_sample_and_features"] == prep.data.loc[20, "group_sample_and_features"]

    # All rows 1, 3, 5, 9, 20 should be in the same group (transitive closure)
    assert prep.data.loc[1, "group_sample_and_features"] == prep.data.loc[5, "group_sample_and_features"]
    assert prep.data.loc[1, "group_sample_and_features"] == prep.data.loc[9, "group_sample_and_features"]
    assert prep.data.loc[1, "group_sample_and_features"] == prep.data.loc[20, "group_sample_and_features"]

    # Row 0 should be independent
    assert prep.data.loc[0, "group_sample_and_features"] != prep.data.loc[1, "group_sample_and_features"]

    # Row 2 should be independent
    assert prep.data.loc[2, "group_sample_and_features"] != prep.data.loc[1, "group_sample_and_features"]


def test_add_group_features_single_row():
    """Test edge case: DataFrame with only one row.

    Should create valid group columns without errors.
    """
    data = pd.DataFrame(
        {
            "feat1": [1],
            "feat2": ["a"],
            "target": [0],
            "sample_id": ["s1"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        target_cols=["target"],
        sample_id="sample_id",
        row_id=None,
        target_assignments={},
    )

    prep._add_group_features()

    # Should have exactly 1 group
    assert prep.data["group_features"].nunique() == 1
    assert prep.data["group_sample_and_features"].nunique() == 1
    # Group should be numbered 0
    assert prep.data.loc[0, "group_features"] == 0
    assert prep.data.loc[0, "group_sample_and_features"] == 0


def test_add_group_features_all_same_sample():
    """Test: All rows have the same sample_id.

    All rows should be in one group for group_sample_and_features,
    even if they have different features.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4, 5],
            "feat2": ["a", "b", "c", "d", "e"],
            "target": [0, 1, 0, 1, 0],
            "sample_id": ["s1", "s1", "s1", "s1", "s1"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        target_cols=["target"],
        sample_id="sample_id",
        row_id=None,
        target_assignments={},
    )

    prep._add_group_features()

    # All rows should be in the same group_sample_and_features
    assert prep.data["group_sample_and_features"].nunique() == 1
    group_id = prep.data.loc[0, "group_sample_and_features"]
    assert all(prep.data["group_sample_and_features"] == group_id)

    # But group_features should have 5 different groups (all features are unique)
    assert prep.data["group_features"].nunique() == 5


def test_add_group_features_all_different():
    """Test: All rows have unique sample_id and unique features.

    Each row should be in its own separate group.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4, 5],
            "feat2": ["a", "b", "c", "d", "e"],
            "target": [0, 1, 0, 1, 0],
            "sample_id": ["s1", "s2", "s3", "s4", "s5"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        target_cols=["target"],
        sample_id="sample_id",
        row_id=None,
        target_assignments={},
    )

    prep._add_group_features()

    # All rows should be in separate groups
    assert prep.data["group_features"].nunique() == 5
    assert prep.data["group_sample_and_features"].nunique() == 5

    # Groups should be numbered 0, 1, 2, 3, 4
    assert set(prep.data["group_sample_and_features"]) == {0, 1, 2, 3, 4}


def test_add_group_features_multiple_independent_groups():
    """Test: Multiple independent groups with no connections between them.

    Should correctly identify the exact number of independent groups.
    Group A: rows 0, 1, 2 (same sample_id)
    Group B: rows 3, 4 (same features)
    Group C: row 5 (alone)
    Group D: row 6 (alone)
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 10, 10, 20, 30],
            "feat2": ["a", "b", "c", "x", "x", "z", "w"],
            "target": [0, 1, 0, 1, 0, 1, 0],
            "sample_id": ["s1", "s1", "s1", "s2", "s3", "s4", "s5"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        target_cols=["target"],
        sample_id="sample_id",
        row_id=None,
        target_assignments={},
    )

    prep._add_group_features()

    # Should have exactly 4 groups
    assert prep.data["group_sample_and_features"].nunique() == 4

    # Group A: rows 0, 1, 2
    assert prep.data.loc[0, "group_sample_and_features"] == prep.data.loc[1, "group_sample_and_features"]
    assert prep.data.loc[1, "group_sample_and_features"] == prep.data.loc[2, "group_sample_and_features"]

    # Group B: rows 3, 4
    assert prep.data.loc[3, "group_sample_and_features"] == prep.data.loc[4, "group_sample_and_features"]

    # Groups should be independent
    group_a = prep.data.loc[0, "group_sample_and_features"]
    group_b = prep.data.loc[3, "group_sample_and_features"]
    group_c = prep.data.loc[5, "group_sample_and_features"]
    group_d = prep.data.loc[6, "group_sample_and_features"]

    assert len({group_a, group_b, group_c, group_d}) == 4


def test_add_group_features_data_integrity():
    """Test: Ensure original data columns are not modified.

    The function should only add new columns, not modify existing ones.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4],
            "feat2": ["a", "b", "c", "d"],
            "target": [0, 1, 0, 1],
            "sample_id": ["s1", "s2", "s1", "s2"],
        }
    )

    # Store original values
    original_feat1 = data["feat1"].copy()
    original_feat2 = data["feat2"].copy()
    original_target = data["target"].copy()
    original_sample_id = data["sample_id"].copy()

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        target_cols=["target"],
        sample_id="sample_id",
        row_id=None,
        target_assignments={},
    )

    prep._add_group_features()

    # Verify original columns unchanged
    assert prep.data["feat1"].equals(original_feat1)
    assert prep.data["feat2"].equals(original_feat2)
    assert prep.data["target"].equals(original_target)
    assert prep.data["sample_id"].equals(original_sample_id)

    # Verify new columns exist
    assert "group_features" in prep.data.columns
    assert "group_sample_and_features" in prep.data.columns


def test_add_group_features_sequential_numbering():
    """Test: Verify group numbers are sequential starting from 0.

    Groups should be numbered 0, 1, 2, ... not arbitrary numbers.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4, 5],
            "feat2": ["a", "b", "c", "d", "e"],
            "target": [0, 1, 0, 1, 0],
            "sample_id": ["s1", "s2", "s3", "s4", "s5"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        target_cols=["target"],
        sample_id="sample_id",
        row_id=None,
        target_assignments={},
    )

    prep._add_group_features()

    # Verify groups are sequential 0, 1, 2, 3, 4
    group_values = sorted(prep.data["group_sample_and_features"].unique())
    assert group_values == list(range(len(group_values)))
    assert group_values[0] == 0

    feature_group_values = sorted(prep.data["group_features"].unique())
    assert feature_group_values == list(range(len(feature_group_values)))
    assert feature_group_values[0] == 0


def test_add_group_features_nan_in_sample_id():
    """Test: Handle NaN values in sample_id column.

    Rows with NaN sample_id should NOT be grouped together unless they share features.
    Each NaN is treated as a unique value by pandas groupby with dropna=False.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4, 5],
            "feat2": ["a", "b", "c", "d", "e"],
            "target": [0, 1, 0, 1, 0],
            "sample_id": [np.nan, "s1", np.nan, "s1", np.nan],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        target_cols=["target"],
        sample_id="sample_id",
        row_id=None,
        target_assignments={},
    )

    prep._add_group_features()

    # Rows 1 and 3 should be grouped (same sample_id "s1")
    assert prep.data.loc[1, "group_sample_and_features"] == prep.data.loc[3, "group_sample_and_features"]

    # Rows 0, 2, 4 have NaN sample_id and different features, should be separate
    # (pandas groupby treats each NaN as separate group)
    assert prep.data.loc[0, "group_sample_and_features"] != prep.data.loc[1, "group_sample_and_features"]
    assert prep.data.loc[2, "group_sample_and_features"] != prep.data.loc[1, "group_sample_and_features"]
    assert prep.data.loc[4, "group_sample_and_features"] != prep.data.loc[1, "group_sample_and_features"]
