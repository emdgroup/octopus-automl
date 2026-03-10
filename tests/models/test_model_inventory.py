"""Test model inventory."""

import pytest

from octopus.models import ModelName, Models
from octopus.types import MLType


class TestModelInventory:
    """Test that all registered models can be instantiated."""

    @pytest.mark.parametrize("name", sorted(Models._config_factories.keys()))
    def test_model_config_instantiates(self, name):
        """Test that the factory for each registered model produces a valid ModelConfig."""
        config = Models.get_config(name)
        assert config is not None
        assert config.ml_types  # non-empty

    def test_model_name_enum_matches_registry(self):
        """Ensure ModelName enum stays in sync with the Models registry."""
        enum_names = {member.value for member in ModelName}
        registry_names = set(Models._config_factories.keys())
        assert enum_names == registry_names, (
            f"ModelName enum and Models registry are out of sync.\n"
            f"  In enum but not registry: {enum_names - registry_names}\n"
            f"  In registry but not enum: {registry_names - enum_names}"
        )


class TestGetDefaults:
    """Test Models.get_defaults()."""

    @pytest.mark.parametrize("ml_type", [MLType.BINARY, MLType.MULTICLASS, MLType.REGRESSION])
    def test_get_defaults(self, ml_type):
        """Test that get_defaults returns non-empty list of models marked as default."""
        defaults = Models.get_defaults(ml_type)
        assert len(defaults) > 0
        assert all(Models.get_config(name).default for name in defaults)
