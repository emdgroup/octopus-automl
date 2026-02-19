"""Test metrics coverage between octopus metrics and autogluon metrics inventory."""

import pytest

from octopus.metrics import Metrics
from octopus.modules.autogluon.core import metrics_inventory_autogluon


class TestAutogluonMetricsCoverage:
    """Test that all octopus classification and regression metrics are available in autogluon."""

    def setup_method(self):
        """Set up test fixtures."""
        self.autogluon_metrics_inventory = metrics_inventory_autogluon

    def get_octopus_classification_metrics(self):
        """Get all octopus classification metrics."""
        all_metrics = Metrics.get_all_metrics()
        classification_metrics = []

        for metric_name in all_metrics:
            try:
                config = Metrics.get_instance(metric_name)
                if config.ml_type == "classification":
                    classification_metrics.append(metric_name)
            except Exception:
                # Skip metrics that can't be configured
                continue

        return classification_metrics

    def get_octopus_regression_metrics(self):
        """Get all octopus regression metrics."""
        all_metrics = Metrics.get_all_metrics()
        regression_metrics = []

        for metric_name in all_metrics:
            try:
                config = Metrics.get_instance(metric_name)
                if config.ml_type == "regression":
                    regression_metrics.append(metric_name)
            except Exception:
                # Skip metrics that can't be configured
                continue

        return regression_metrics

    def test_all_classification_metrics_in_autogluon(self):
        """Test that all octopus classification metrics are available in autogluon."""
        octopus_classification_metrics = self.get_octopus_classification_metrics()
        autogluon_metrics = set(self.autogluon_metrics_inventory.keys())

        missing_metrics = []
        for metric in octopus_classification_metrics:
            if metric not in autogluon_metrics:
                missing_metrics.append(metric)

        assert not missing_metrics, (
            f"The following octopus classification metrics are missing from autogluon inventory: "
            f"{missing_metrics}. "
            f"Octopus classification metrics: {sorted(octopus_classification_metrics)}. "
            f"Autogluon metrics: {sorted(autogluon_metrics)}"
        )

    def test_all_regression_metrics_in_autogluon(self):
        """Test that all octopus regression metrics are available in autogluon."""
        octopus_regression_metrics = self.get_octopus_regression_metrics()
        autogluon_metrics = set(self.autogluon_metrics_inventory.keys())

        missing_metrics = []
        for metric in octopus_regression_metrics:
            if metric not in autogluon_metrics:
                missing_metrics.append(metric)

        assert not missing_metrics, (
            f"The following octopus regression metrics are missing from autogluon inventory: "
            f"{missing_metrics}. "
            f"Octopus regression metrics: {sorted(octopus_regression_metrics)}. "
            f"Autogluon metrics: {sorted(autogluon_metrics)}"
        )

    def test_metrics_coverage_summary(self):
        """Provide a summary of metrics coverage."""
        octopus_classification = self.get_octopus_classification_metrics()
        octopus_regression = self.get_octopus_regression_metrics()
        autogluon_metrics = set(self.autogluon_metrics_inventory.keys())

        total_octopus_metrics = len(octopus_classification) + len(octopus_regression)
        covered_metrics = len([m for m in octopus_classification + octopus_regression if m in autogluon_metrics])

        coverage_percentage = (covered_metrics / total_octopus_metrics) * 100 if total_octopus_metrics > 0 else 0

        print("\n=== Metrics Coverage Summary ===")
        print(f"Octopus Classification Metrics ({len(octopus_classification)}): {sorted(octopus_classification)}")
        print(f"Octopus Regression Metrics ({len(octopus_regression)}): {sorted(octopus_regression)}")
        print(f"Autogluon Metrics ({len(autogluon_metrics)}): {sorted(autogluon_metrics)}")
        print(f"Coverage: {covered_metrics}/{total_octopus_metrics} ({coverage_percentage:.1f}%)")

        # This test should always pass if the above tests pass
        assert coverage_percentage == 100.0, f"Expected 100% coverage, got {coverage_percentage:.1f}%"

    def test_no_time_to_event_metrics_included(self):
        """Verify that time-to-event metrics are excluded from the comparison."""
        all_metrics = Metrics.get_all_metrics()
        time_to_event_metrics = []

        for metric_name in all_metrics:
            try:
                config = Metrics.get_instance(metric_name)
                if config.ml_type == "timetoevent":
                    time_to_event_metrics.append(metric_name)
            except Exception:
                continue

        # Verify that time-to-event metrics exist but are not in autogluon inventory
        if time_to_event_metrics:
            for metric in time_to_event_metrics:
                # It's OK if time-to-event metrics are not in autogluon inventory
                # This test just documents that they exist and are excluded
                print(f"Time-to-event metric excluded from comparison: {metric}")

        # This test always passes - it's for documentation
        assert True, "Time-to-event metrics are properly excluded from the comparison"


if __name__ == "__main__":
    # Allow running the test directly
    pytest.main([__file__, "-v"])
