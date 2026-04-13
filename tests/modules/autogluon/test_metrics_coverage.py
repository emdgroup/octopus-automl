"""Test metrics coverage between octopus metrics and autogluon metrics inventory."""

from unittest.mock import MagicMock

import pytest

from octopus.metrics import Metrics
from octopus.modules.autogluon.core import AutoGluonModule, metrics_inventory_autogluon
from octopus.types import MLType


def _get_octopus_metrics_for_type(ml_type: MLType) -> list[str]:
    """Get all octopus metric names that support a given ML type."""
    result = []
    for metric_name in Metrics.get_all_metrics():
        config = Metrics.get_instance(metric_name)
        if config.supports_ml_type(ml_type):
            result.append(metric_name)
    return result


def _assert_all_covered(ml_type: MLType) -> None:
    """Assert every octopus metric for ml_type is in the AG inventory."""
    octopus_metrics = _get_octopus_metrics_for_type(ml_type)
    ag_metrics = set(metrics_inventory_autogluon.keys())
    missing = sorted(set(octopus_metrics) - ag_metrics)
    assert not missing, (
        f"Octopus {ml_type.value} metrics missing from autogluon inventory: {missing}. "
        f"Octopus: {sorted(octopus_metrics)}. AG: {sorted(ag_metrics)}"
    )


class TestAutogluonMetricsCoverage:
    """Test that all octopus classification, multiclass, and regression metrics are in AG."""

    def test_all_binary_metrics_covered(self):
        """Every octopus BINARY metric must have an AG mapping."""
        _assert_all_covered(MLType.BINARY)

    def test_all_multiclass_metrics_covered(self):
        """Every octopus MULTICLASS metric must have an AG mapping."""
        _assert_all_covered(MLType.MULTICLASS)

    def test_all_regression_metrics_covered(self):
        """Every octopus REGRESSION metric must have an AG mapping."""
        _assert_all_covered(MLType.REGRESSION)

    def test_mse_maps_to_mse_not_rmse(self):
        """MSE and RMSE must map to different AG scorers."""
        mse_scorer = metrics_inventory_autogluon["MSE"]
        rmse_scorer = metrics_inventory_autogluon["RMSE"]
        assert mse_scorer is not rmse_scorer, (
            f"MSE and RMSE must map to different AG scorers, but both map to {mse_scorer}"
        )

    def test_t2e_metrics_excluded(self):
        """Time-to-event metrics should NOT be in the AG inventory."""
        t2e_metrics = _get_octopus_metrics_for_type(MLType.TIMETOEVENT)
        ag_metrics = set(metrics_inventory_autogluon.keys())
        assert t2e_metrics, "Expected at least one T2E metric to exist"
        overlap = set(t2e_metrics) & ag_metrics
        assert not overlap, f"T2E metrics should not be in AG inventory: {overlap}"

    def test_full_coverage(self):
        """100% coverage across binary + multiclass + regression."""
        all_relevant = set()
        for ml_type in (MLType.BINARY, MLType.MULTICLASS, MLType.REGRESSION):
            all_relevant.update(_get_octopus_metrics_for_type(ml_type))

        ag_metrics = set(metrics_inventory_autogluon.keys())
        missing = sorted(all_relevant - ag_metrics)
        assert not missing, f"Missing metrics: {missing}"


class TestAutogluonT2EGuard:
    """Test that AutoGluon rejects time-to-event tasks."""

    def test_fit_raises_on_timetoevent(self):
        """fit() must raise ValueError for T2E tasks."""
        module = AutoGluonModule(config=MagicMock())
        study_context = MagicMock()
        study_context.ml_type = MLType.TIMETOEVENT

        with pytest.raises(ValueError, match="time-to-event"):
            module.fit(
                data_traindev=MagicMock(),
                data_test=MagicMock(),
                feature_cols=[],
                study_context=study_context,
                outer_split_id=0,
                results_dir=MagicMock(),
                scratch_dir=MagicMock(),
                n_assigned_cpus=1,
                feature_groups={},
                prior_results={},
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
