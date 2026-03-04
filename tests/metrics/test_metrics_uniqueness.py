"""Test metrics uniqueness and consistency.

This test is critical because functions in modules/utils deduce the ml_type from metrics,
so any inconsistencies or duplicates could break the system.
"""

from collections import Counter, defaultdict

import pytest

from octopus.metrics import Metrics
from octopus.types import ML_TYPES, MLType


class TestMetricsUniqueness:
    """Test that all available metrics are unique in their name and registry."""

    def setup_method(self):
        """Set up test fixtures."""
        self.all_metrics = Metrics.get_all_metrics()

    def test_registry_keys_are_unique(self):
        """Test that all registry keys are unique.

        This should always pass due to dict nature, but documents the expectation.
        """
        registry_keys = list(self.all_metrics.keys())
        unique_keys = set(registry_keys)

        assert len(registry_keys) == len(unique_keys), (
            f"Registry keys are not unique. Found {len(registry_keys)} keys but only "
            f"{len(unique_keys)} unique keys. Keys: {sorted(registry_keys)}"
        )

    def test_metric_config_names_are_unique(self):
        """Test that all metric config names are unique.

        This is critical for the utils functions that deduce ml_type from metrics.
        """
        config_names = []
        config_name_to_registry_key = {}

        for registry_key in self.all_metrics:
            try:
                config = Metrics.get_instance(registry_key)
                config_name = config.name
                config_names.append(config_name)

                if config_name in config_name_to_registry_key:
                    config_name_to_registry_key[config_name].append(registry_key)
                else:
                    config_name_to_registry_key[config_name] = [registry_key]

            except Exception as e:
                pytest.fail(f"Failed to get config for metric '{registry_key}': {e}")

        # Check for duplicates
        name_counts = Counter(config_names)
        duplicates = {name: count for name, count in name_counts.items() if count > 1}

        if duplicates:
            duplicate_details = []
            for name, count in duplicates.items():
                registry_keys = config_name_to_registry_key[name]
                duplicate_details.append(f"'{name}' appears {count} times in registry keys: {registry_keys}")

            pytest.fail(
                f"Found {len(duplicates)} duplicate metric config names:\n"
                + "\n".join(duplicate_details)
                + f"\nAll config names: {sorted(config_names)}"
            )

    def test_registry_key_matches_config_name(self):
        """Test that registry keys match their corresponding config names.

        Ensures consistency between @Metrics.register("KEY") and config.name.
        """
        mismatches = []

        for registry_key in self.all_metrics:
            try:
                config = Metrics.get_instance(registry_key)
                config_name = config.name

                if registry_key != config_name:
                    mismatches.append(f"Registry key '{registry_key}' != config name '{config_name}'")

            except Exception as e:
                pytest.fail(f"Failed to get config for metric '{registry_key}': {e}")

        assert not mismatches, f"Found {len(mismatches)} registry key/config name mismatches:\n" + "\n".join(mismatches)

    def test_all_metrics_have_valid_ml_types(self):
        """Test that all metrics have valid ml_types values.

        Ensures utils functions can properly deduce ML types.
        """
        valid_ml_types = set(ML_TYPES)
        invalid_metrics = []
        ml_type_distribution = defaultdict(list)

        for registry_key in self.all_metrics:
            try:
                config = Metrics.get_instance(registry_key)

                for ml_type in config.ml_types:
                    if ml_type.value not in valid_ml_types:
                        invalid_metrics.append(f"'{registry_key}' has invalid ml_type: '{ml_type}'")
                    else:
                        ml_type_distribution[ml_type.value].append(registry_key)

            except Exception as e:
                pytest.fail(f"Failed to get config for metric '{registry_key}': {e}")

        assert not invalid_metrics, (
            f"Found {len(invalid_metrics)} metrics with invalid ml_types:\n"
            + "\n".join(invalid_metrics)
            + f"\nValid ml_types are: {sorted(valid_ml_types)}"
        )

        # Print distribution for documentation
        print("\n=== ML Type Distribution ===")
        for ml_type in sorted(ml_type_distribution):
            metrics = sorted(ml_type_distribution[ml_type])
            print(f"{ml_type} ({len(metrics)}): {metrics}")

    def test_all_metrics_have_valid_prediction_types(self):
        """Test that all metrics have valid prediction_type values."""
        valid_prediction_types = {"predict", "predict_proba"}
        invalid_metrics = []

        for registry_key in self.all_metrics:
            try:
                config = Metrics.get_instance(registry_key)
                prediction_type = config.prediction_type

                if prediction_type not in valid_prediction_types:
                    invalid_metrics.append(f"'{registry_key}' has invalid prediction_type: '{prediction_type}'")

            except Exception as e:
                pytest.fail(f"Failed to get config for metric '{registry_key}': {e}")

        assert not invalid_metrics, (
            f"Found {len(invalid_metrics)} metrics with invalid prediction_types:\n"
            + "\n".join(invalid_metrics)
            + f"\nValid prediction_types are: {sorted(valid_prediction_types)}"
        )

    def test_metrics_loaded_dynamically(self):
        """Test that all metric modules are properly imported and registry is populated."""
        assert len(self.all_metrics) > 0, "No metrics found in registry. Check imports in metrics/__init__.py"

        # Verify we have metrics from different categories
        ml_types = set()
        for registry_key in self.all_metrics:
            try:
                config = Metrics.get_instance(registry_key)
                for t in config.ml_types:
                    ml_types.add(t.value)
            except Exception:
                continue

        expected_ml_types = {MLType.BINARY.value, MLType.REGRESSION.value}  # At minimum
        missing_types = expected_ml_types - ml_types

        assert not missing_types, (
            f"Missing expected ML types: {missing_types}. "
            f"Found ML types: {sorted(ml_types)}. "
            f"This suggests some metric modules may not be imported properly."
        )

    def test_no_metric_config_attribute_conflicts(self):
        """Test that metric configs don't have conflicting attributes for same names."""
        configs_by_name = {}
        conflicts = []

        for registry_key in self.all_metrics:
            try:
                config = Metrics.get_instance(registry_key)
                config_name = config.name

                if config_name in configs_by_name:
                    # Compare all attributes
                    existing_config = configs_by_name[config_name]

                    if existing_config.ml_types != config.ml_types:
                        conflicts.append(
                            f"'{config_name}': ml_type conflict - '{existing_config.ml_types}' vs '{config.ml_types}'"
                        )

                    if existing_config.prediction_type != config.prediction_type:
                        conflicts.append(
                            f"'{config_name}': prediction_type conflict - "
                            f"'{existing_config.prediction_type}' vs '{config.prediction_type}'"
                        )

                    if existing_config.higher_is_better != config.higher_is_better:
                        conflicts.append(
                            f"'{config_name}': higher_is_better conflict - "
                            f"'{existing_config.higher_is_better}' vs '{config.higher_is_better}'"
                        )
                else:
                    configs_by_name[config_name] = config

            except Exception as e:
                pytest.fail(f"Failed to get config for metric '{registry_key}': {e}")

        assert not conflicts, f"Found {len(conflicts)} metric config attribute conflicts:\n" + "\n".join(conflicts)

    def test_comprehensive_metrics_summary(self):
        """Provide a comprehensive summary of all metrics for documentation."""
        metrics_by_ml_type = defaultdict(list)
        metrics_by_prediction_type = defaultdict(list)
        total_metrics = len(self.all_metrics)

        for registry_key in self.all_metrics:
            try:
                config = Metrics.get_instance(registry_key)
                for t in config.ml_types:
                    metrics_by_ml_type[t.value].append(registry_key)
                metrics_by_prediction_type[config.prediction_type].append(registry_key)
            except Exception:
                continue

        print(f"\n=== Comprehensive Metrics Summary ({total_metrics} total) ===")

        print("\nBy ML Type:")
        for ml_type in sorted(metrics_by_ml_type.keys()):
            metrics = sorted(metrics_by_ml_type[ml_type])
            print(f"  {ml_type} ({len(metrics)}): {metrics}")

        print("\nBy Prediction Type:")
        for pred_type in sorted(metrics_by_prediction_type.keys()):
            metrics = sorted(metrics_by_prediction_type[pred_type])
            print(f"  {pred_type} ({len(metrics)}): {metrics}")

        # This test always passes - it's for documentation
        assert total_metrics > 0, f"Expected metrics to be loaded, found {total_metrics}"


if __name__ == "__main__":
    # Allow running the test directly
    pytest.main([__file__, "-v", "-s"])
