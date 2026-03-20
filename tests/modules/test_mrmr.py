"""Test MRMR."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from octopus.modules.mrmr.core import _maxrminr as maxrminr
from octopus.modules.mrmr.core import _relevance_from_dependency, _relevance_fstats
from octopus.modules.result import ModuleResult
from octopus.types import CorrelationType, FIResultLabel, MLType, ResultType


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
        "pearson": maxrminr(df_features, df_feature_importances, [5], correlation_type=CorrelationType.PEARSON),
        "spearman": maxrminr(df_features, df_feature_importances, [5], correlation_type=CorrelationType.SPEARMAN),
        "rdc": maxrminr(df_features, df_feature_importances, [5], correlation_type=CorrelationType.RDC),
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


def test_mrmr_suppresses_redundant_features():
    """Test that MRMR prefers a less relevant but non-redundant feature over a redundant one.

    Setup: 3 features where A and B are highly correlated (redundant).
    - A: importance=0.9, B: importance=0.8, C: importance=0.3
    - corr(A, B)~0.95, corr(A, C)~0, corr(B, C)~0

    Expected: MRMR selects A first (highest relevance), then C (low redundancy
    with A beats B's high redundancy), then B.
    Without redundancy penalization, the order would be A, B, C.
    """
    np.random.seed(0)
    n = 500
    a = np.random.randn(n)
    b = 0.95 * a + 0.05 * np.random.randn(n)  # highly correlated with A
    c = np.random.randn(n)  # independent

    df_features = pd.DataFrame({"A": a, "B": b, "C": c})
    df_relevance = pd.DataFrame({"feature": ["A", "B", "C"], "importance": [0.9, 0.8, 0.3]})

    result = maxrminr(df_features, df_relevance, [3], correlation_type=CorrelationType.PEARSON)

    assert result[3][0] == "A", "Most relevant feature should be selected first"
    assert result[3][1] == "C", "Non-redundant feature C should be preferred over redundant B"
    assert result[3][2] == "B", "Redundant feature B should be selected last"


def test_mrmr_drops_negative_importance():
    """Test that features with non-positive importance are excluded from selection."""
    np.random.seed(0)
    n = 100
    df_features = pd.DataFrame(
        {
            "good": np.random.randn(n),
            "neutral": np.random.randn(n),
            "bad": np.random.randn(n),
        }
    )
    df_relevance = pd.DataFrame(
        {
            "feature": ["good", "neutral", "bad"],
            "importance": [0.5, 0.0, -0.3],
        }
    )

    result = maxrminr(df_features, df_relevance, [1], correlation_type=CorrelationType.PEARSON)

    assert result[1] == ["good"]
    assert max(result.keys()) == 1, "Only 1 feature with positive importance should be selectable"


def test_mrmr_raises_on_no_positive_relevance():
    """Test that MRMR raises when all features have non-positive importance."""
    df_features = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df_relevance = pd.DataFrame({"feature": ["A", "B"], "importance": [0.0, -1.0]})

    with pytest.raises(ValueError, match="No features with positive relevance"):
        maxrminr(df_features, df_relevance, [1], correlation_type=CorrelationType.PEARSON)


def test_relevance_fstats_ranks_informative_feature_highest():
    """Test that f-statistics relevance correctly ranks a predictive feature above noise."""
    np.random.seed(42)
    n = 500
    target = np.random.choice([0, 1], size=n)
    informative = target + 0.1 * np.random.randn(n)  # strongly predictive
    noise = np.random.randn(n)  # random noise

    df_features = pd.DataFrame({"informative": informative, "noise": noise})
    df_target = pd.DataFrame({"target": target})

    result = _relevance_fstats(df_features, df_target, ["informative", "noise"], MLType.BINARY)

    informative_importance = result.loc[result["feature"] == "informative", "importance"].iloc[0]
    noise_importance = result.loc[result["feature"] == "noise", "importance"].iloc[0]
    assert informative_importance > noise_importance


def test_relevance_from_dependency_averages_across_splits():
    """Test that relevance from dependency correctly averages feature importances across CV splits."""
    df_fi = pd.DataFrame(
        {
            "feature": ["A", "A", "B", "B", "C", "C"],
            "importance": [0.8, 0.6, 0.3, 0.1, -0.1, -0.2],
            "fi_method": ["permutation"] * 6,
            "training_id": [0, 1, 0, 1, 0, 1],
        }
    )
    dependency_results = {
        ResultType.BEST: ModuleResult(
            result_type=ResultType.BEST,
            module="octo",
            feature_importances=df_fi,
        )
    }

    result = _relevance_from_dependency(["A", "B", "C"], dependency_results, FIResultLabel.PERMUTATION)

    # A: mean(0.8, 0.6) = 0.7, B: mean(0.3, 0.1) = 0.2, C: mean(-0.1, -0.2) = -0.15
    # Positive filtering happens in _maxrminr, not here — all features returned
    assert list(result["feature"]) == ["A", "B", "C"]
    assert result.loc[result["feature"] == "A", "importance"].iloc[0] == pytest.approx(0.7)
    assert result.loc[result["feature"] == "B", "importance"].iloc[0] == pytest.approx(0.2)
    assert result.loc[result["feature"] == "C", "importance"].iloc[0] == pytest.approx(-0.15)
