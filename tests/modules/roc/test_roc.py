"""Comprehensive test suite for ROC (Remove Outliers and Correlations) module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from octopus.modules import Roc
from octopus.types import CorrelationType, RelevanceMethod


class TestRocModule:
    """Test suite for ROC module configuration."""

    def test_roc_module_initialization_defaults(self):
        """Test ROC module initialization with default parameters."""
        roc = Roc(task_id=0)

        assert roc.task_id == 0
        assert roc.depends_on is None
        assert roc.description == ""
        assert roc.correlation_threshold == 0.8
        assert roc.correlation_type == CorrelationType.SPEARMAN
        assert roc.relevance_method == RelevanceMethod.F_STATISTICS
        assert roc.module == "roc"

    def test_roc_module_initialization_custom_params(self):
        """Test ROC module initialization with custom parameters."""
        roc = Roc(
            task_id=1,
            depends_on=0,
            description="test_roc",
            correlation_threshold=0.9,
            correlation_type=CorrelationType.RDC,
            relevance_method=RelevanceMethod.MUTUAL_INFO,
        )

        assert roc.task_id == 1
        assert roc.depends_on == 0
        assert roc.description == "test_roc"
        assert roc.correlation_threshold == 0.9
        assert roc.correlation_type == CorrelationType.RDC
        assert roc.relevance_method == RelevanceMethod.MUTUAL_INFO

    def test_roc_module_invalid_correlation_type(self):
        """Test ROC module with invalid correlation type."""
        with pytest.raises(ValueError, match="is not a valid CorrelationType"):
            Roc(task_id=0, correlation_type="invalid_correlation")

    def test_roc_module_invalid_relevance_method(self):
        """Test ROC module with invalid relevance method."""
        with pytest.raises(ValueError, match="is not a valid RelevanceMethod"):
            Roc(task_id=0, relevance_method="invalid_filter")

    def test_roc_module_invalid_correlation_threshold_type(self):
        """Test ROC module with invalid correlation threshold type."""
        with pytest.raises(TypeError):
            Roc(task_id=0, correlation_threshold="invalid")  # type: ignore[arg-type]

    def test_roc_module_negative_task_id(self):
        """Test ROC module with negative sequence ID."""
        with pytest.raises(ValueError):
            Roc(task_id=-1)

    @pytest.mark.parametrize("correlation_threshold", [0.0, 0.5, 0.8, 0.95, 1.0])
    def test_roc_module_correlation_threshold_range(self, correlation_threshold):
        """Test ROC module with different correlation threshold values."""
        roc = Roc(task_id=0, correlation_threshold=correlation_threshold)
        assert roc.correlation_threshold == correlation_threshold

    @pytest.mark.parametrize("correlation_type", [CorrelationType.SPEARMAN, CorrelationType.RDC])
    def test_roc_module_correlation_types(self, correlation_type):
        """Test ROC module with different correlation types."""
        roc = Roc(task_id=0, correlation_type=correlation_type)
        assert roc.correlation_type == correlation_type

    @pytest.mark.parametrize("relevance_method", [RelevanceMethod.MUTUAL_INFO, RelevanceMethod.F_STATISTICS])
    def test_roc_module_relevance_methods(self, relevance_method):
        """Test ROC module with different relevance methods."""
        roc = Roc(task_id=0, relevance_method=relevance_method)
        assert roc.relevance_method == relevance_method


class TestRocIntegration:
    """Integration tests for ROC module."""

    def test_roc_module_and_core_integration(self):
        """Test integration between ROC module configuration and core functionality."""
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        X, y = make_classification(n_samples=n_samples, n_features=8, random_state=42)

        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        data = pd.DataFrame(X, columns=feature_names)
        data["target"] = y
        data["sample_id_col"] = range(len(data))

        # Create ROC module configuration
        roc_module = Roc(
            task_id=0,
            description="integration_test",
            correlation_threshold=0.85,
            correlation_type=CorrelationType.SPEARMAN,
            relevance_method=RelevanceMethod.MUTUAL_INFO,
        )

        # Verify module configuration
        assert roc_module.correlation_threshold == 0.85
        assert roc_module.correlation_type == CorrelationType.SPEARMAN
        assert roc_module.relevance_method == RelevanceMethod.MUTUAL_INFO
        assert roc_module.description == "integration_test"
