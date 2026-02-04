"""Test creation of correlation groups."""

import numpy as np
import pandas as pd
import pytest

from octopus.experiment import OctoExperiment


@pytest.fixture
def correlated_data():
    """Create dataset for correlation grouping test."""
    np.random.seed(42)
    n_samples = 1000

    # Base features
    x1 = np.random.randn(n_samples)
    x2 = np.random.randn(n_samples)

    # Highly correlated with x1 (correlation > 0.9)
    x3 = x1 * 0.9 + np.random.randn(n_samples) * 0.1

    # Moderately correlated with x1 (0.9 > correlation > 0.8)
    x4 = x1 * 0.7 + np.random.randn(n_samples) * 0.35

    # Weakly correlated with x1 (0.8 > correlation > 0.7)
    x5 = x1 * 0.5 + np.random.randn(n_samples) * 0.4

    # Correlated with x2 (correlation > 0.8)
    x6 = x2 * 0.7 + np.random.randn(n_samples) * 0.35

    # Uncorrelated feature
    x7 = np.random.randn(n_samples)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "x5": x5,
            "x6": x6,
            "x7": x7,
            "target": np.random.randint(0, 2, n_samples),
        }
    )


@pytest.fixture
def mock_octo_experiment(correlated_data):
    """Create octo experiment for correlation grouping test."""
    return OctoExperiment(
        id="test_id",
        experiment_id=1,
        task_id=2,
        depends_on_task=3,
        task_path="/mock/path",
        study_path="./studies/",
        study_name="test",
        ml_type="regression",
        target_metric="R2",
        positive_class=1,
        metrics=["R2"],
        imputation_method="median",
        datasplit_column="split",
        row_column="row_id",
        feature_cols=["x1", "x2", "x3", "x4", "x5", "x6", "x7"],
        target_assignments={"target": "value"},
        data_traindev=correlated_data,
        data_test=correlated_data.sample(n=200, random_state=42),
    )


def test_feature_groups_structure(mock_octo_experiment):
    """Test structure of groups."""
    feature_groups = mock_octo_experiment.feature_groups
    assert isinstance(feature_groups, dict), "feature_groups should be a dictionary"
    assert len(feature_groups) >= 3, f"Expected at least 3 groups, but got {len(feature_groups)}"


def test_correlated_features_grouping(mock_octo_experiment):
    """Test for correct thresholds groups."""
    feature_groups = mock_octo_experiment.feature_groups

    def check_features_grouped(features, threshold):
        for group_name, group in feature_groups.items():
            if set(features).issubset(set(group)):
                print(
                    f"""Features {features} found in group {group_name}
                    with threshold > {threshold}"""
                )
                return
        pytest.fail(
            f"""Features {features} should be grouped together
            (correlation > {threshold})"""
        )

    # Check for high correlation group (> 0.9)
    check_features_grouped(["x1", "x3"], 0.9)

    # Check for moderate correlation group (> 0.8)
    check_features_grouped(["x1", "x3", "x4"], 0.8)
    check_features_grouped(["x2", "x6"], 0.8)

    # Check for weak correlation group (> 0.7)
    check_features_grouped(["x1", "x3", "x4", "x5"], 0.7)


def test_uncorrelated_features(mock_octo_experiment):
    """Test uncorrelated features not in any group."""
    feature_groups = mock_octo_experiment.feature_groups
    uncorrelated_feature = "x7"
    for group in feature_groups.values():
        if len(group) > 1:
            assert uncorrelated_feature not in group, f"""Uncorrelated feature {uncorrelated_feature}
                should not be in group {group}"""


def test_correlation_thresholds(mock_octo_experiment):
    """Test correlation thresholds."""
    feature_groups = mock_octo_experiment.feature_groups
    correlation_matrix = (
        mock_octo_experiment.data_traindev[mock_octo_experiment.feature_cols].corr(method="spearman").abs()
    )

    for group_name, group in feature_groups.items():
        if len(group) > 1:
            group_corr = correlation_matrix.loc[group, group]
            min_corr = group_corr.values[np.triu_indices_from(group_corr, k=1)].min()

            if "0.9" in group_name:
                assert min_corr >= 0.9, f"""Minimum correlation in group {group_name} is {min_corr},
                    which is below the threshold of 0.9"""
            elif "0.8" in group_name:
                assert min_corr >= 0.8, f"""Minimum correlation in group {group_name} is {min_corr},
                    which is below the threshold of 0.8"""
            elif "0.7" in group_name:
                assert min_corr >= 0.7, f"""Minimum correlation in group {group_name} is {min_corr},
                      which is below the threshold of 0.7"""


def test_features_in_multiple_groups(mock_octo_experiment):
    """Test features are in multiple groups."""
    feature_groups = mock_octo_experiment.feature_groups
    feature_to_groups = {}
    for group_name, features in feature_groups.items():
        for feature in features:
            if feature not in feature_to_groups:
                feature_to_groups[feature] = []
            feature_to_groups[feature].append(group_name)

    # Check that x1, x3, and x4 are in multiple groups
    assert len(feature_to_groups["x1"]) > 1, "x1 should be in multiple groups"
    assert len(feature_to_groups["x3"]) > 1, "x3 should be in multiple groups"
    assert len(feature_to_groups["x4"]) > 1, "x4 should be in multiple groups"
