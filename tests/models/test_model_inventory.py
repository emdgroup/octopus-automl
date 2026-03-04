"""Test model inventory."""

import pytest

from octopus.models import Models


class TestModelInventory:
    """Test that all registered models can be instantiated."""

    @pytest.mark.parametrize("name", sorted(Models._config_factories.keys()))
    def test_model_config_instantiates(self, name):
        """Test that the factory for each registered model produces a valid ModelConfig."""
        config = Models.get_config(name)
        assert config is not None
        assert config.ml_types  # non-empty
