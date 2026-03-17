"""Test MRMR."""

import time
from typing import Literal

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from octopus.modules.mrmr.core import _maxrminr as maxrminr


def generate_sample_data(n_samples, n_features, random_state):
    """Generate sample data."""
    np.random.seed(42)
    X, _y, _ = make_regression(n_samples=n_samples, n_features=n_features, random_state=random_state, coef=True)
    df_features = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df_feature_importances = pd.DataFrame(
        {
            "feature": df_features.columns,
            "importance": np.random.rand(df_features.shape[1]),
        }
    )
    return df_features, df_feature_importances


@pytest.fixture(
    params=[
        (500, 20, 0, "sample_data_1"),
        (500, 10, 0, "sample_data_2"),
        (500, 15, 2, "sample_data_3"),
    ]
)
def sample_data(request):
    """Create sample data."""
    n_samples, n_features, random_state, name = request.param
    return generate_sample_data(n_samples, n_features, random_state), name


def test_mrmr_feature_selection_order(sample_data):
    """Test MRMR feature selection for different datasets."""
    (df_features, df_feature_importances), data_name = sample_data

    results = {
        "pearson": maxrminr(df_features, df_feature_importances, [5], correlation_type="pearson"),
        "spearman": maxrminr(df_features, df_feature_importances, [5], correlation_type="spearman"),
        "rdc": maxrminr(df_features, df_feature_importances, [5], correlation_type="rdc"),
    }

    if data_name == "sample_data_1":
        assert results["pearson"][5][0] == "feature_11"
        assert results["pearson"][5][1] == "feature_2"
        assert results["pearson"][5][2] == "feature_17"
        assert results["pearson"][5][-1] == "feature_3"
        assert results["pearson"][20][0] == "feature_11"
        assert results["pearson"][20][-1] == "feature_10"

        assert results["spearman"][5][0] == "feature_11"
        assert results["spearman"][5][1] == "feature_1"
        assert results["spearman"][5][2] == "feature_7"
        assert results["spearman"][5][-1] == "feature_12"
        assert results["spearman"][20][0] == "feature_11"
        assert results["spearman"][20][-1] == "feature_10"

        assert results["rdc"][5][0] == "feature_11"
        assert results["rdc"][5][1] == "feature_1"
        assert results["rdc"][5][2] == "feature_7"
        assert results["rdc"][5][-1] == "feature_2"
        assert results["rdc"][20][0] == "feature_11"
        assert results["rdc"][20][-1] == "feature_10"

    elif data_name == "sample_data_2":
        assert results["pearson"][5][0] == "feature_1"
        assert results["pearson"][5][1] == "feature_9"
        assert results["pearson"][5][2] == "feature_7"
        assert results["pearson"][5][-1] == "feature_2"
        assert results["pearson"][10][0] == "feature_1"
        assert results["pearson"][10][-1] == "feature_6"

    elif data_name == "sample_data_3":
        assert results["spearman"][5][0] == "feature_11"
        assert results["spearman"][5][1] == "feature_0"
        assert results["spearman"][5][2] == "feature_9"
        assert results["spearman"][5][-1] == "feature_7"
        assert results["spearman"][15][0] == "feature_11"
        assert results["spearman"][15][-1] == "feature_10"

        assert results["rdc"][5][0] == "feature_11"
        assert results["rdc"][5][1] == "feature_7"
        assert results["rdc"][5][2] == "feature_1"
        assert results["rdc"][5][-1] == "feature_9"
        assert results["rdc"][15][0] == "feature_11"
        assert results["rdc"][15][-1] == "feature_10"


@pytest.fixture(params=[(200, 1000, 0, "many_features")])
def scalability_data(request):
    """Create sample data for scalability testing."""
    n_samples, n_features, random_state, name = request.param
    return generate_sample_data(n_samples, n_features, random_state), name


def test_mrmr_scalability(scalability_data):
    """Test MRMR algorithm scalability with large feature sets."""
    (df_features, df_feature_importances), data_name = scalability_data

    # Dictionary to store execution times
    execution_times = {}

    # Test different correlation types: pearson only
    types: list[Literal["pearson", "spearman", "rdc"]] = ["pearson"]
    for corr_type in types:
        start_time = time.time()

        # Select top 20 features
        results = maxrminr(df_features, df_feature_importances, [20], correlation_type=corr_type)

        end_time = time.time()
        execution_times[corr_type] = end_time - start_time

        # Basic assertions to ensure the function works
        assert len(results[20]) == 20
        assert isinstance(results[20][0], str)

    # Print execution times
    print(f"MRMR Scalability Test ({data_name}):")
    for corr_type, exec_time in execution_times.items():
        print(f"  {corr_type}: {exec_time:.4f} seconds")
