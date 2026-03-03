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
            "sample_id_col": ["s1", "s2", "s3", "s4", "s5", "s5", "s6", "s6"],
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
        sample_id_col="sample_id_col",
        row_id_col=None,
    )


def test_prepare(octo_preparator):
    """Test prepare function."""
    prepared = octo_preparator.prepare()

    assert isinstance(prepared.data, pd.DataFrame)
    assert isinstance(prepared.feature_cols, list)
    assert isinstance(prepared.row_id_col, str)


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


def test_create_row_id_col(octo_preparator):
    """Test create row id function."""
    octo_preparator._create_row_id_col()
    assert "row_id" in octo_preparator.data.columns
    assert octo_preparator.row_id_col == "row_id"
    assert octo_preparator.data["row_id"].tolist() == list(range(8))


def test_add_datasplit_group(octo_preparator):
    """Test add datasplit group function."""
    octo_preparator._standardize_null_values()
    octo_preparator._standardize_inf_values()
    octo_preparator._transform_bool_to_int()
    octo_preparator._add_datasplit_group()

    assert "datasplit_group" in octo_preparator.data.columns
    assert octo_preparator.data.loc[0, "datasplit_group"] == octo_preparator.data.loc[2, "datasplit_group"]
    assert octo_preparator.data.loc[4, "datasplit_group"] == octo_preparator.data.loc[5, "datasplit_group"]
    assert octo_preparator.data.loc[6, "datasplit_group"] == octo_preparator.data.loc[7, "datasplit_group"]
    assert octo_preparator.data["datasplit_group"].nunique() == 4
    assert octo_preparator.data.index.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]


def test_standardize_null_values(octo_preparator):
    """Test standardize null values function."""
    octo_preparator._standardize_null_values()
    assert octo_preparator.data["null_col"].isna().all()


def test_standardize_only_pipeline_columns():
    """Test that null/inf standardization only applies to explicit pipeline columns.

    Standardization must:
    - Convert null/inf-like strings in feature and target columns
    - NOT modify sample_id_col (to prevent data leakage via groupby merging NaN sample IDs)
    - NOT modify columns outside the pipeline
    """
    data = pd.DataFrame(
        {
            "feature1": ["none", "inf", 3],
            "target": ["null", 1, 0],
            "sample_id_col": ["None", "inf", "s3"],
            "extra_col": ["NA", "infinity", "valid"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feature1"],
        sample_id_col="sample_id_col",
        row_id_col=None,
        target_col="target",
    )

    prep._standardize_null_values()
    prep._standardize_inf_values()

    # Pipeline columns should be standardized
    assert pd.isna(prep.data["feature1"].iloc[0])
    assert np.isinf(prep.data["feature1"].iloc[1])
    assert pd.isna(prep.data["target"].iloc[0])

    # sample_id_col should remain unchanged
    assert prep.data["sample_id_col"].tolist() == ["None", "inf", "s3"]

    # Non-pipeline columns should remain unchanged
    assert prep.data["extra_col"].tolist() == ["NA", "infinity", "valid"]


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

    assert prepared.row_id_col in prepared.data.columns
    assert "datasplit_group" in prepared.data.columns
    assert "group_features" not in prepared.data.columns
    assert "group_sample_and_features" not in prepared.data.columns
    assert prepared.data["bool_col"].dtype == int
    assert np.isinf(prepared.data["inf_col"].iloc[0])
    assert "null_col" not in prepared.feature_cols


def test_add_datasplit_group_with_categorical_and_nan():
    """Test add datasplit group with categorical columns containing NaN values.

    This test specifically addresses the issue where categorical columns with
    NaN values would raise a TypeError when trying to fill with a placeholder
    that wasn't in the category list.
    """
    data = pd.DataFrame(
        {
            "cat_feature": pd.Categorical(["a", "b", "a", None, "c", None]),
            "num_feature": [1, 2, 1, 3, 4, 3],
            "target": [0, 1, 0, 1, 1, 0],
            "sample_id_col": ["s1", "s2", "s3", "s4", "s5", "s6"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["cat_feature", "num_feature"],
        sample_id_col="sample_id_col",
        row_id_col=None,
    )

    prep._add_datasplit_group()

    assert "datasplit_group" in prep.data.columns
    # Rows 0 and 2 share features (a, 1), rows 3 and 5 share features (None, 3)
    assert prep.data.loc[0, "datasplit_group"] == prep.data.loc[2, "datasplit_group"]
    assert prep.data.loc[3, "datasplit_group"] == prep.data.loc[5, "datasplit_group"]
    assert prep.data["datasplit_group"].nunique() == 4


def test_add_datasplit_group_with_mixed_types_and_nan():
    """Test add datasplit group with mixed column types (categorical, numeric, string) and NaN values."""
    data = pd.DataFrame(
        {
            "cat_col": pd.Categorical(["x", "y", None, "x", None]),
            "num_col": [1.5, 2.5, np.nan, 1.5, np.nan],
            "str_col": ["alpha", "beta", "alpha", "alpha", "beta"],
            "inf_col": [np.inf, -np.inf, 1.0, np.inf, -np.inf],
            "target": [0, 1, 0, 1, 1],
            "sample_id_col": ["s1", "s2", "s3", "s4", "s5"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["cat_col", "num_col", "str_col", "inf_col"],
        sample_id_col="sample_id_col",
        row_id_col=None,
    )

    prep._add_datasplit_group()

    assert "datasplit_group" in prep.data.columns
    # Rows 0 and 3 share features (x, 1.5, alpha, inf)
    assert prep.data.loc[0, "datasplit_group"] == prep.data.loc[3, "datasplit_group"]
    assert prep.data["datasplit_group"].nunique() == 4


def test_add_datasplit_group_with_all_nan_column():
    """Test grouping when one column is entirely NaN.

    This edge case ensures that columns with all NaN values are handled properly
    and rows are still grouped correctly based on other features.
    """
    data = pd.DataFrame(
        {
            "all_nan": [np.nan, np.nan, np.nan, np.nan],
            "feat2": [1, 2, 1, 2],
            "target": [0, 1, 0, 1],
            "sample_id_col": ["s1", "s2", "s3", "s4"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["all_nan", "feat2"],
        sample_id_col="sample_id_col",
        row_id_col=None,
    )

    prep._add_datasplit_group()

    assert "datasplit_group" in prep.data.columns
    # Rows 0 and 2 share features (nan, 1), rows 1 and 3 share features (nan, 2)
    assert prep.data.loc[0, "datasplit_group"] == prep.data.loc[2, "datasplit_group"]
    assert prep.data.loc[1, "datasplit_group"] == prep.data.loc[3, "datasplit_group"]
    assert prep.data["datasplit_group"].nunique() == 2


def test_add_datasplit_group_same_sample_different_features():
    """Test grouping when rows have same sample_id_col but different features.

    This ensures that datasplit_group correctly groups rows with same sample_id_col.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4],
            "feat2": ["a", "b", "c", "d"],
            "target": [0, 1, 0, 1],
            "sample_id_col": ["s1", "s1", "s2", "s2"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        sample_id_col="sample_id_col",
        row_id_col=None,
    )

    prep._add_datasplit_group()

    assert prep.data.loc[0, "datasplit_group"] == prep.data.loc[1, "datasplit_group"]
    assert prep.data.loc[2, "datasplit_group"] == prep.data.loc[3, "datasplit_group"]


def test_add_datasplit_group_same_features_different_samples():
    """Test grouping when rows have same features but different sample_id_cols.

    This ensures that rows with identical features are grouped together.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 1, 2, 2],
            "feat2": ["a", "a", "b", "b"],
            "target": [0, 1, 0, 1],
            "sample_id_col": ["s1", "s2", "s3", "s4"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        sample_id_col="sample_id_col",
        row_id_col=None,
    )

    prep._add_datasplit_group()

    assert prep.data.loc[0, "datasplit_group"] == prep.data.loc[1, "datasplit_group"]
    assert prep.data.loc[2, "datasplit_group"] == prep.data.loc[3, "datasplit_group"]
    assert prep.data["datasplit_group"].nunique() == 2


def test_add_datasplit_group_large_dataset_with_duplicates():
    """Test grouping on a larger dataset with many duplicate feature combinations.

    This ensures the grouping logic scales properly and handles duplicates correctly.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 3] * 27,
            "feat2": ["a", "b", "c"] * 27,
            "target": [0, 1] * 40 + [0],
            "sample_id_col": [f"s{i}" for i in range(81)],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        sample_id_col="sample_id_col",
        row_id_col=None,
    )

    prep._add_datasplit_group()

    # 3 unique feature combos, each sample unique -> 3 groups
    assert prep.data["datasplit_group"].nunique() == 3
    group_sizes = prep.data["datasplit_group"].value_counts()
    assert sorted(group_sizes.tolist()) == [27, 27, 27]
    mask_1a = (prep.data["feat1"] == 1) & (prep.data["feat2"] == "a")
    groups_1a = prep.data.loc[mask_1a, "datasplit_group"].unique()
    assert len(groups_1a) == 1, "All (1, 'a') rows should be in the same group"


def test_add_datasplit_group_transitive_closure():
    """Test transitive closure in datasplit_group.

    Critical scenario: Row 0 and 1 share sample_id_col, Row 1 and 2 share features.
    Therefore, rows 0, 1, and 2 must ALL be in the same group through transitive
    closure, even though rows 0 and 2 don't directly share sample_id_col or features.

    Example:
        Row 0: sample_id_col="s1", features=[1, "a"]
        Row 1: sample_id_col="s1", features=[2, "b"]  <- shares sample_id_col with row 0
        Row 2: sample_id_col="s2", features=[2, "b"]  <- shares features with row 1
        Row 3: sample_id_col="s3", features=[3, "c"]  <- independent

    Expected result: Rows 0, 1, 2 should all have the same datasplit_group
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 2, 3],
            "feat2": ["a", "b", "b", "c"],
            "target": [0, 1, 0, 1],
            "sample_id_col": ["s1", "s1", "s2", "s3"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        sample_id_col="sample_id_col",
        row_id_col=None,
    )

    prep._add_datasplit_group()

    # All rows 0, 1, 2 should be in the same datasplit_group due to transitive closure
    assert prep.data.loc[0, "datasplit_group"] == prep.data.loc[1, "datasplit_group"]
    assert prep.data.loc[1, "datasplit_group"] == prep.data.loc[2, "datasplit_group"]
    assert prep.data.loc[0, "datasplit_group"] == prep.data.loc[2, "datasplit_group"]

    # Row 3 should be in a different group (no connection to others)
    assert prep.data.loc[3, "datasplit_group"] != prep.data.loc[0, "datasplit_group"]

    # Should have exactly 2 unique groups: {0,1,2} and {3}
    assert prep.data["datasplit_group"].nunique() == 2


def test_add_datasplit_group_transitive_closure_long_chain():
    """Test transitive closure with a longer chain of connections.

    This tests a more complex transitive closure scenario:
        Row 0-1: same sample_id_col
        Row 1-2: same features
        Row 2-3: same sample_id_col
        Row 3-4: same features

    All rows 0-4 should end up in the same group.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 2, 3, 3, 9],
            "feat2": ["a", "b", "b", "c", "c", "z"],
            "target": [0, 1, 0, 1, 0, 1],
            "sample_id_col": ["s1", "s1", "s2", "s2", "s3", "s4"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        sample_id_col="sample_id_col",
        row_id_col=None,
    )

    prep._add_datasplit_group()

    # All rows 0-4 should be in the same group through the chain
    for i in range(4):
        assert prep.data.loc[i, "datasplit_group"] == prep.data.loc[i + 1, "datasplit_group"], (
            f"Row {i} and {i + 1} should be in the same group"
        )

    # Row 5 should be independent
    assert prep.data.loc[5, "datasplit_group"] != prep.data.loc[0, "datasplit_group"]

    # Should have exactly 2 unique groups
    assert prep.data["datasplit_group"].nunique() == 2


def test_add_datasplit_group_same_sample_consecutive():
    """Test (a): Multiple rows with same sample_id_col - consecutive in table.

    Rows 0, 1, 2 have same sample_id_col and appear consecutively.
    All should be grouped together.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4, 5],
            "feat2": ["a", "b", "c", "d", "e"],
            "target": [0, 1, 0, 1, 0],
            "sample_id_col": ["s1", "s1", "s1", "s2", "s3"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        sample_id_col="sample_id_col",
        row_id_col=None,
    )

    prep._add_datasplit_group()

    # Rows 0, 1, 2 should be in same group (same sample_id_col)
    assert prep.data.loc[0, "datasplit_group"] == prep.data.loc[1, "datasplit_group"]
    assert prep.data.loc[1, "datasplit_group"] == prep.data.loc[2, "datasplit_group"]

    # Rows 3 and 4 should be in different groups
    assert prep.data.loc[3, "datasplit_group"] != prep.data.loc[0, "datasplit_group"]
    assert prep.data.loc[4, "datasplit_group"] != prep.data.loc[0, "datasplit_group"]
    assert prep.data.loc[3, "datasplit_group"] != prep.data.loc[4, "datasplit_group"]


def test_add_datasplit_group_same_sample_with_gaps():
    """Test (a): Multiple rows with same sample_id_col - with gaps in table.

    Rows 0, 3, 5 have same sample_id_col but are not consecutive.
    All should still be grouped together.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4, 5, 6],
            "feat2": ["a", "b", "c", "d", "e", "f"],
            "target": [0, 1, 0, 1, 0, 1],
            "sample_id_col": ["s1", "s2", "s3", "s1", "s4", "s1"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        sample_id_col="sample_id_col",
        row_id_col=None,
    )

    prep._add_datasplit_group()

    # Rows 0, 3, 5 should be in same group (same sample_id_col "s1")
    assert prep.data.loc[0, "datasplit_group"] == prep.data.loc[3, "datasplit_group"]
    assert prep.data.loc[3, "datasplit_group"] == prep.data.loc[5, "datasplit_group"]

    # Other rows should be in different groups
    assert prep.data.loc[1, "datasplit_group"] != prep.data.loc[0, "datasplit_group"]
    assert prep.data.loc[2, "datasplit_group"] != prep.data.loc[0, "datasplit_group"]
    assert prep.data.loc[4, "datasplit_group"] != prep.data.loc[0, "datasplit_group"]


def test_add_datasplit_group_same_features_consecutive():
    """Test (b): Multiple rows with same features - consecutive in table.

    Rows 1, 2, 3 have identical features and appear consecutively.
    All should be grouped together.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 2, 2, 5],
            "feat2": ["a", "b", "b", "b", "e"],
            "target": [0, 1, 0, 1, 0],
            "sample_id_col": ["s1", "s2", "s3", "s4", "s5"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        sample_id_col="sample_id_col",
        row_id_col=None,
    )

    prep._add_datasplit_group()

    # Rows 1, 2, 3 should be in same group (same features)
    assert prep.data.loc[1, "datasplit_group"] == prep.data.loc[2, "datasplit_group"]
    assert prep.data.loc[2, "datasplit_group"] == prep.data.loc[3, "datasplit_group"]

    # Rows 0 and 4 should be in different groups
    assert prep.data.loc[0, "datasplit_group"] != prep.data.loc[1, "datasplit_group"]
    assert prep.data.loc[4, "datasplit_group"] != prep.data.loc[1, "datasplit_group"]


def test_add_datasplit_group_same_features_with_gaps():
    """Test (b): Multiple rows with same features - with gaps in table.

    Rows 1, 4, 7 have identical features but are not consecutive.
    All should still be grouped together.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4, 2, 5, 6, 2],
            "feat2": ["a", "b", "c", "d", "b", "e", "f", "b"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
            "sample_id_col": ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        sample_id_col="sample_id_col",
        row_id_col=None,
    )

    prep._add_datasplit_group()

    # Rows 1, 4, 7 should be in same group (same features [2, "b"])
    assert prep.data.loc[1, "datasplit_group"] == prep.data.loc[4, "datasplit_group"]
    assert prep.data.loc[4, "datasplit_group"] == prep.data.loc[7, "datasplit_group"]

    # Other rows should be in different groups
    assert prep.data.loc[0, "datasplit_group"] != prep.data.loc[1, "datasplit_group"]
    assert prep.data.loc[2, "datasplit_group"] != prep.data.loc[1, "datasplit_group"]
    assert prep.data.loc[3, "datasplit_group"] != prep.data.loc[1, "datasplit_group"]


def test_add_datasplit_group_complex_with_gaps():
    """Test (c) + (d): Complex scenario with sample_id_col and feature connections with gaps.

    Specific test case:
        - Rows 1, 3, 5 have same sample_id_col "s1"
        - Rows 5, 9, 20 have same features [100, "x"]

    Expected: All rows 1, 3, 5, 9, 20 should be in the same group due to transitive closure:
        1-3-5 connected by sample_id_col
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
            "sample_id_col": [
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
        sample_id_col="sample_id_col",
        row_id_col=None,
    )

    prep._add_datasplit_group()

    # Rows 1, 3, 5 should be connected by sample_id_col "s1"
    assert prep.data.loc[1, "datasplit_group"] == prep.data.loc[3, "datasplit_group"]
    assert prep.data.loc[3, "datasplit_group"] == prep.data.loc[5, "datasplit_group"]

    # Rows 5, 9, 20 should be connected by features [100, "x"]
    assert prep.data.loc[5, "datasplit_group"] == prep.data.loc[9, "datasplit_group"]
    assert prep.data.loc[9, "datasplit_group"] == prep.data.loc[20, "datasplit_group"]

    # All rows 1, 3, 5, 9, 20 should be in the same group (transitive closure)
    assert prep.data.loc[1, "datasplit_group"] == prep.data.loc[5, "datasplit_group"]
    assert prep.data.loc[1, "datasplit_group"] == prep.data.loc[9, "datasplit_group"]
    assert prep.data.loc[1, "datasplit_group"] == prep.data.loc[20, "datasplit_group"]

    # Row 0 should be independent
    assert prep.data.loc[0, "datasplit_group"] != prep.data.loc[1, "datasplit_group"]

    # Row 2 should be independent
    assert prep.data.loc[2, "datasplit_group"] != prep.data.loc[1, "datasplit_group"]


def test_add_datasplit_group_single_row():
    """Test edge case: DataFrame with only one row.

    Should create valid group columns without errors.
    """
    data = pd.DataFrame(
        {
            "feat1": [1],
            "feat2": ["a"],
            "target": [0],
            "sample_id_col": ["s1"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        sample_id_col="sample_id_col",
        row_id_col=None,
    )

    prep._add_datasplit_group()

    # Should have exactly 1 group
    assert prep.data["datasplit_group"].nunique() == 1
    # Group should be numbered 0
    assert prep.data.loc[0, "datasplit_group"] == 0


def test_add_datasplit_group_all_same_sample():
    """Test: All rows have the same sample_id_col.

    All rows should be in one datasplit_group, even if they have different features.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4, 5],
            "feat2": ["a", "b", "c", "d", "e"],
            "target": [0, 1, 0, 1, 0],
            "sample_id_col": ["s1", "s1", "s1", "s1", "s1"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        sample_id_col="sample_id_col",
        row_id_col=None,
    )

    prep._add_datasplit_group()

    # All rows should be in the same datasplit_group
    assert prep.data["datasplit_group"].nunique() == 1
    group_id = prep.data.loc[0, "datasplit_group"]
    assert all(prep.data["datasplit_group"] == group_id)


def test_add_datasplit_group_all_different():
    """Test: All rows have unique sample_id_col and unique features.

    Each row should be in its own separate group.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4, 5],
            "feat2": ["a", "b", "c", "d", "e"],
            "target": [0, 1, 0, 1, 0],
            "sample_id_col": ["s1", "s2", "s3", "s4", "s5"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        sample_id_col="sample_id_col",
        row_id_col=None,
    )

    prep._add_datasplit_group()

    # All rows should be in separate groups
    assert prep.data["datasplit_group"].nunique() == 5

    # Groups should be numbered 0, 1, 2, 3, 4
    assert set(prep.data["datasplit_group"]) == {0, 1, 2, 3, 4}


def test_add_datasplit_group_multiple_independent_groups():
    """Test: Multiple independent groups with no connections between them.

    Should correctly identify the exact number of independent groups.
    Group A: rows 0, 1, 2 (same sample_id_col)
    Group B: rows 3, 4 (same features)
    Group C: row 5 (alone)
    Group D: row 6 (alone)
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 10, 10, 20, 30],
            "feat2": ["a", "b", "c", "x", "x", "z", "w"],
            "target": [0, 1, 0, 1, 0, 1, 0],
            "sample_id_col": ["s1", "s1", "s1", "s2", "s3", "s4", "s5"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        sample_id_col="sample_id_col",
        row_id_col=None,
    )

    prep._add_datasplit_group()

    # Should have exactly 4 groups
    assert prep.data["datasplit_group"].nunique() == 4

    # Group A: rows 0, 1, 2
    assert prep.data.loc[0, "datasplit_group"] == prep.data.loc[1, "datasplit_group"]
    assert prep.data.loc[1, "datasplit_group"] == prep.data.loc[2, "datasplit_group"]

    # Group B: rows 3, 4
    assert prep.data.loc[3, "datasplit_group"] == prep.data.loc[4, "datasplit_group"]

    # Groups should be independent
    group_a = prep.data.loc[0, "datasplit_group"]
    group_b = prep.data.loc[3, "datasplit_group"]
    group_c = prep.data.loc[5, "datasplit_group"]
    group_d = prep.data.loc[6, "datasplit_group"]

    assert len({group_a, group_b, group_c, group_d}) == 4


def test_add_datasplit_group_data_integrity():
    """Test: Ensure original data columns are not modified.

    The function should only add new columns, not modify existing ones.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4],
            "feat2": ["a", "b", "c", "d"],
            "target": [0, 1, 0, 1],
            "sample_id_col": ["s1", "s2", "s1", "s2"],
        }
    )

    # Store original values
    original_feat1 = data["feat1"].copy()
    original_feat2 = data["feat2"].copy()
    original_target = data["target"].copy()
    original_sample_id_col = data["sample_id_col"].copy()

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        sample_id_col="sample_id_col",
        row_id_col=None,
    )

    prep._add_datasplit_group()

    # Verify original columns unchanged
    assert prep.data["feat1"].equals(original_feat1)
    assert prep.data["feat2"].equals(original_feat2)
    assert prep.data["target"].equals(original_target)
    assert prep.data["sample_id_col"].equals(original_sample_id_col)

    # Verify new column exists
    assert "datasplit_group" in prep.data.columns


def test_add_datasplit_group_sequential_numbering():
    """Test: Verify group numbers are sequential starting from 0.

    Groups should be numbered 0, 1, 2, ... not arbitrary numbers.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4, 5],
            "feat2": ["a", "b", "c", "d", "e"],
            "target": [0, 1, 0, 1, 0],
            "sample_id_col": ["s1", "s2", "s3", "s4", "s5"],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        sample_id_col="sample_id_col",
        row_id_col=None,
    )

    prep._add_datasplit_group()

    # Verify groups are sequential 0, 1, 2, 3, 4
    group_values = sorted(prep.data["datasplit_group"].unique())
    assert group_values == list(range(len(group_values)))
    assert group_values[0] == 0


def test_add_datasplit_group_nan_in_sample_id_col():
    """Test: Handle NaN values in sample_id_col column.

    With dropna=False, pandas groupby groups all NaN values together into a
    single group. This means rows 0, 2, 4 (all NaN sample_id) are grouped
    together, which is the conservative behavior — preventing potential leakage
    by keeping all unknown-identity samples in the same split.
    """
    data = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4, 5],
            "feat2": ["a", "b", "c", "d", "e"],
            "target": [0, 1, 0, 1, 0],
            "sample_id_col": [np.nan, "s1", np.nan, "s1", np.nan],
        }
    )

    prep = OctoDataPreparator(
        data=data,
        feature_cols=["feat1", "feat2"],
        sample_id_col="sample_id_col",
        row_id_col=None,
    )

    prep._add_datasplit_group()

    # Rows 1 and 3 should be grouped (same sample_id_col "s1")
    assert prep.data.loc[1, "datasplit_group"] == prep.data.loc[3, "datasplit_group"]

    # All NaN sample_id rows are grouped together by pandas groupby with dropna=False
    assert prep.data.loc[0, "datasplit_group"] == prep.data.loc[2, "datasplit_group"]
    assert prep.data.loc[2, "datasplit_group"] == prep.data.loc[4, "datasplit_group"]

    # NaN group is separate from "s1" group
    assert prep.data.loc[0, "datasplit_group"] != prep.data.loc[1, "datasplit_group"]

    # 2 groups total: {0, 2, 4} (NaN) and {1, 3} (s1)
    assert prep.data["datasplit_group"].nunique() == 2
