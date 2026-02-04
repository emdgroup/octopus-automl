"""Test PreparedData class."""

import numpy as np
import pandas as pd
import pytest

from octopus.study.prepared_data import PreparedData


@pytest.fixture
def sample_data_with_types():
    """Create sample data with all allowed column types."""
    df = pd.DataFrame(
        {
            "id": range(1, 101),
            "sample_id_col": [f"S{i}" for i in range(1, 101)],
            "int_feature": np.random.randint(0, 100, 100),
            "float_feature": np.random.rand(100),
            "bool_feature": np.random.choice([True, False], 100),
            "nominal_feature": pd.Categorical(["A", "B", "C"] * 33 + ["A"], ordered=False),
            "ordinal_feature": pd.Categorical(
                ["Low", "Medium", "High"] * 33 + ["Low"],
                categories=["Low", "Medium", "High"],
                ordered=True,
            ),
            "target": np.random.randint(0, 2, 100),
        }
    )
    return df


@pytest.fixture
def prepared_data(sample_data_with_types):
    """Create PreparedData instance."""
    return PreparedData(
        data=sample_data_with_types,
        feature_cols=[
            "int_feature",
            "float_feature",
            "bool_feature",
            "nominal_feature",
            "ordinal_feature",
        ],
        row_id_col="id",
        target_assignments={"target": "target"},
    )


def test_numerical_columns(prepared_data):
    """Test num_features property with all allowed numeric types (int, float, bool)."""
    numerical_cols = prepared_data.num_features
    assert isinstance(numerical_cols, list)
    assert "int_feature" in numerical_cols
    assert "float_feature" in numerical_cols
    assert "bool_feature" in numerical_cols
    assert "nominal_feature" not in numerical_cols
    assert "ordinal_feature" not in numerical_cols
    assert len(numerical_cols) == 3


def test_categorical_nominal_columns(prepared_data):
    """Test cat_nominal_features property."""
    nominal_cols = prepared_data.cat_nominal_features
    assert isinstance(nominal_cols, list)
    assert "nominal_feature" in nominal_cols
    assert "ordinal_feature" not in nominal_cols
    assert "int_feature" not in nominal_cols
    assert "float_feature" not in nominal_cols
    assert "bool_feature" not in nominal_cols
    assert len(nominal_cols) == 1


def test_categorical_ordinal_columns(prepared_data):
    """Test cat_ordinal_features property."""
    ordinal_cols = prepared_data.cat_ordinal_features
    assert isinstance(ordinal_cols, list)
    assert "ordinal_feature" in ordinal_cols
    assert "nominal_feature" not in ordinal_cols
    assert "int_feature" not in ordinal_cols
    assert "float_feature" not in ordinal_cols
    assert "bool_feature" not in ordinal_cols
    assert len(ordinal_cols) == 1


def test_all_columns_covered(prepared_data):
    """Test that all feature columns are covered by the three property types."""
    all_typed_columns = (
        prepared_data.num_features + prepared_data.cat_nominal_features + prepared_data.cat_ordinal_features
    )
    assert set(all_typed_columns) == set(prepared_data.feature_cols)


def test_no_overlap_between_column_types(prepared_data):
    """Test that column types don't overlap and sum equals total features."""
    numerical = set(prepared_data.num_features)
    nominal = set(prepared_data.cat_nominal_features)
    ordinal = set(prepared_data.cat_ordinal_features)

    # Check no overlap
    assert len(numerical & nominal) == 0
    assert len(numerical & ordinal) == 0
    assert len(nominal & ordinal) == 0

    # Check sum equals total features
    total_by_type = (
        len(prepared_data.num_features)
        + len(prepared_data.cat_nominal_features)
        + len(prepared_data.cat_ordinal_features)
    )
    assert total_by_type == len(prepared_data.feature_cols)
