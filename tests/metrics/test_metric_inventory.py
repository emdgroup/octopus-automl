"""Test metric inventory."""

import pytest

from octopus.metrics import Metrics


class TestMetricInventory:
    """Test that all registered metrics can be instantiated."""

    @pytest.mark.parametrize("name", sorted(Metrics._config_factories.keys()))
    def test_metric_instance_instantiates(self, name):
        """Test that the factory for each registered metric produces a valid Metric."""
        metric = Metrics.get_instance(name)
        assert metric is not None
        assert metric.ml_types  # non-empty
