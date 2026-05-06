"""Test metrics coverage between octopus metrics and autogluon metrics inventory."""

import pytest

from octopus.metrics import Metrics
from octopus.modules.autogluon.core import _AG_METRIC_MAP
from octopus.types import MLType


class TestAutogluonMetricsCoverage:
    """Test that all octopus classification, multiclass, and regression metrics are in AG."""

    @pytest.mark.parametrize("ml_type", [MLType.BINARY, MLType.MULTICLASS, MLType.REGRESSION])
    def test_all_metrics_covered(self, ml_type: MLType) -> None:
        """Every octopus metric for the given ML type must have an AG mapping."""
        octopus_metrics = set(Metrics.get_by_type(ml_type))
        ag_metrics = set(_AG_METRIC_MAP.keys())
        missing = sorted(octopus_metrics - ag_metrics)
        assert not missing, (
            f"Octopus {ml_type.value} metrics missing from autogluon inventory: {missing}. "
            f"Octopus: {sorted(octopus_metrics)}. AG: {sorted(ag_metrics)}"
        )

    def test_t2e_metrics_excluded(self) -> None:
        """Time-to-event metrics should NOT be in the AG inventory."""
        t2e_metrics = set(Metrics.get_by_type(MLType.TIMETOEVENT))
        ag_metrics = set(_AG_METRIC_MAP.keys())
        assert t2e_metrics, "Expected at least one T2E metric to exist"
        overlap = t2e_metrics & ag_metrics
        assert not overlap, f"T2E metrics should not be in AG inventory: {overlap}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
