import numpy as np
import pandas as pd
import pytest

from octopus.utils import parquet_load, parquet_save

_CASES = [
    # bool
    ([True, False, False], "bool"),
    # int
    ([True, False, False], "int32"),
    ([True, False, False], "int64"),
    ([1, 2, 3], "int64"),
    ([1, 2, 3], "int32"),
    # float
    ([1, 2, 3], "float64"),
    ([1, 2, 3], "float32"),
    ([1.2, 2.3, 3.4], "float64"),
    ([1.2, 2.3, 3.4], "float32"),
    # object
    ([True, False, False], "object"),
    ([1, 2, 3], "object"),
    ([1.2, 2.3, 3.4], "object"),
    (["a", "b", "c"], "object"),
    pytest.param(["a", 1, 1.2, True], "object", marks=pytest.mark.xfail),
    # string
    ([True, False, False], "StringDtype"),
    ([1, 2, 3], "StringDtype"),
    ([1.2, 2.3, 3.4], "StringDtype"),
    (["a", "b", "c"], "StringDtype"),
    pytest.param(["a", 1, 1.2, True], "StringDtype", marks=pytest.mark.xfail),
    # category
    ([True, False, False], "category"),
    ([1, 2, 3], "category"),
    ([1.2, 2.3, 3.4], "category"),
    (["a", "b", "c"], "category"),
    pytest.param(["a", 1, 1.2, True], "category", marks=pytest.mark.xfail),
    # CategoricalDtype
    ([True, False, False], "CategoricalDtype"),
    ([1, 2, 3], "CategoricalDtype"),
    ([1.2, 2.3, 3.4], "CategoricalDtype"),
    (["a", "b", "c"], "CategoricalDtype"),
    pytest.param(["a", 1, 1.2, True], "CategoricalDtype", marks=pytest.mark.xfail),
]


@pytest.mark.parametrize("data, dtype", _CASES, ids=[f"{d[1]}, {d[0]}" for d in _CASES])
def test_parquet_dtype_roundtrip(tmp_path, data, dtype):
    """Test that saving and loading a DataFrame with parquet_save and parquet_load works correctly."""
    if dtype == "CategoricalDtype":
        dtype = pd.CategoricalDtype(set(data))
    elif dtype == "StringDtype":
        dtype = pd.StringDtype()

    df = pd.DataFrame({"col": data}, dtype=dtype)

    path = tmp_path / "test.parquet"
    parquet_save(df, path)
    loaded_df = parquet_load(path)

    pd.testing.assert_frame_equal(df, loaded_df)


@pytest.mark.parametrize(
    "column_names",
    [
        (0, 1),
        ("a", "b"),
        (0, "a"),
        ("a", 0),
        (True, 12),
        (np.int64(1), 12),
    ],
    ids=str,
)
def test_parquet_column_name_roundtrip(tmp_path, column_names):
    """Test that saving and loading a DataFrame with parquet_save and parquet_load works correctly."""
    df = pd.DataFrame(
        {
            column_names[0]: [1, 2, 3],
            column_names[1]: [4, 5, 6],
        }
    ).reset_index()

    path = tmp_path / "test.parquet"
    parquet_save(df, path)
    loaded_df = parquet_load(path)

    pd.testing.assert_frame_equal(df, loaded_df)
