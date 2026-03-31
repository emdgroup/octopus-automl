"""Test that octopus core package imports work correctly.

These tests run early in the test suite to catch import issues quickly.
They verify that all core modules can be imported without errors.
"""

import pytest


@pytest.mark.windows
def test_core_imports() -> None:
    """Test all core package imports work without optional dependencies."""
    import octopus  # noqa: F401, PLC0415
    from octopus.manager.core import OctoManager  # noqa: F401, PLC0415
    from octopus.metrics import Metrics  # noqa: F401, PLC0415
    from octopus.models import Models  # noqa: F401, PLC0415
    from octopus.modules import Task  # noqa: F401, PLC0415
    from octopus.study import OctoStudy  # noqa: F401, PLC0415


@pytest.mark.windows
def test_core_functionality() -> None:
    """Test that core functionality can be instantiated."""
    from octopus.metrics import Metrics  # noqa: PLC0415
    from octopus.models import Models  # noqa: PLC0415

    # Verify model registry is accessible
    num_models = len(Models._config_factories)
    assert num_models > 0, "Models registry should have at least one model"

    # Verify metrics inventory can be instantiated
    metrics = Metrics
    assert len(metrics.get_all_metrics()) > 0, "Metrics should have at least one available metric"
