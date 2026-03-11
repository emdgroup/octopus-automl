# FAQ

## When loading `.parquet` files, categorical columns seem to be returned as `int`, losing the information that they were categorical.

This is a known issue with parquet file support in Python.
Both existing libraries, `pyarrow` as well as `fastparquet` do not exactly reproduce original input data types when it comes to categorical columns.
See e.g. [Issue 29017](https://github.com/apache/arrow/issues/29017) and [Issue 27067](https://github.com/apache/arrow/issues/27067).

To ensure proper data type roundtrip, the module `octopus.utils` provides the functions `parquet_load()` and `parquet_save()` to store and reconstruct precise dtype information in the parquet metadata.
Files written with `parquet_save()` are expected to be readable with every parquet-compatible code.
Still, proper dtypes are only guaranteed to be reconstructed using `parquet_load()`.

For details on which dtypes are tested and supported, see [tests/infrastructure/test_file_io.py](https://github.com/emdgroup/octopus/blob/main/tests/infrastructure/test_file_io.py).
