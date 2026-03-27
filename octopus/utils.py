"""Utils."""

import contextlib
import json
import logging
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from upath import UPath


def rmtree(path: UPath) -> None:
    """Recursively remove a directory tree (fsspec-compatible).

    Unlike ``UPath.rmdir(recursive=True)``, this function does not call
    ``is_dir()`` before deletion.  On object stores such as S3 an empty
    "directory" created via ``mkdir()`` may not be recognised as a
    directory by ``is_dir()``, which causes ``rmdir`` to raise
    ``NotADirectoryError``.  Bypassing that check and calling the
    filesystem's ``rm`` directly avoids the problem.

    Args:
        path: Directory to remove.  If it does not exist the call is a
            no-op.
    """
    with contextlib.suppress(FileNotFoundError):
        path.fs.rm(path.path, recursive=True)


def get_package_name() -> str:
    """Return the package name."""
    return "octopus-automl"


def get_version() -> str:
    """Return the installed version of octopus-automl."""
    return version(get_package_name())


def joblib_save(obj: Any, path: UPath) -> None:
    """Save an object with joblib through a file handle (fsspec-compatible)."""
    with path.open("wb") as f:
        joblib.dump(obj, f)


def joblib_load(path: UPath) -> Any:
    """Load an object with joblib through a file handle (fsspec-compatible)."""
    with path.open("rb") as f:
        return joblib.load(f)


_PARQUET_METADATA_KEY = f"{get_package_name()}.dtype_fidelity.v1".encode()


def _generate_dtype_fidelity_metadata(col, dtype) -> dict[str, str | bool | list]:
    """Generate metadata for a given dtype to preserve fidelity when saving to Parquet."""
    result: dict[str, str | bool | list] = {}

    if not isinstance(col, str):
        result["name"] = col

    if isinstance(dtype, pd.CategoricalDtype):
        result["kind"] = "category"
        result["ordered"] = bool(dtype.ordered)
        result["categories"] = dtype.categories.tolist()
        result["cat_dtype"] = str(dtype.categories.dtype)
    elif dtype.name == "object":
        result["kind"] = "object"
    elif isinstance(dtype, pd.StringDtype):
        result["kind"] = "string"
    elif isinstance(dtype, (pd.Float64Dtype, pd.Float32Dtype)):
        result["kind"] = str(dtype)

    return result


def parquet_save(df: pd.DataFrame, path: str | Path | UPath, index: bool = True) -> None:
    """Save a DataFrame to Parquet with octopus-specific dtype fidelity metadata."""
    path = UPath(path)

    # Build fidelity metadata: only what pyarrow can't roundtrip
    fidelity: dict[str, dict] = {}
    for col in df.columns:
        if metadata := _generate_dtype_fidelity_metadata(col, df[col].dtype):
            fidelity[str(col)] = metadata

    table = pa.Table.from_pandas(df, preserve_index=index)

    if fidelity:
        existing_md = table.schema.metadata or {}
        new_md = {**existing_md, _PARQUET_METADATA_KEY: json.dumps(fidelity).encode()}
        table = table.replace_schema_metadata(new_md)

    with path.open("wb") as f:
        pq.write_table(table, f)


def parquet_load(path: str | Path | UPath) -> pd.DataFrame:
    """Load a DataFrame from Parquet, restoring dtype fidelity if metadata exists."""
    path = UPath(path)

    with path.open("rb") as f:
        table = pq.read_table(f)

    md = table.schema.metadata or {}
    if _PARQUET_METADATA_KEY not in md:
        return table.to_pandas()  # type: ignore[no-any-return]

    fidelity = json.loads(md[_PARQUET_METADATA_KEY].decode("utf-8"))

    cat_cols = [col for col, spec in fidelity.items() if spec.get("kind") == "category" and col in table.column_names]

    df: pd.DataFrame = table.to_pandas(categories=cat_cols if cat_cols else None)

    col_name_mapper = {}
    for col, spec in fidelity.items():
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' specified in parquet metadata not found in table columns {table.column_names}"
            )

        if (name := spec.get("name", None)) is not None:
            col_name_mapper[col] = name

        kind = spec.get("kind")
        if kind == "category":
            if "categories" in spec:
                cat_dtype = spec.get("cat_dtype", "object")
                full_cats = pd.Index(spec["categories"], dtype=cat_dtype)
                df[col] = df[col].cat.set_categories(full_cats)
            if spec.get("ordered", False):
                df[col] = df[col].cat.as_ordered()
            else:
                df[col] = df[col].cat.as_unordered()
        elif kind == "object":
            df[col] = df[col].astype("object")
        elif kind == "string":
            df[col] = df[col].astype("string")
        elif kind in ("Float64", "Float32"):
            df[col] = df[col].astype(kind)

    if col_name_mapper:
        df.rename(columns=col_name_mapper, inplace=True)

    return df


def csv_save(df: pd.DataFrame, path: str | Path | UPath, **kwargs) -> None:
    """Save a DataFrame to CSV format."""
    path = UPath(path)
    df.to_csv(str(path), storage_options=dict(path.storage_options), **kwargs)


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

    import networkx as nx  # noqa: PLC0415
    import scipy.stats  # noqa: PLC0415

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
