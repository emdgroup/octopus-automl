import os
from datetime import UTC, datetime
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_datetime_for_studies():
    """Mock datetime.now() to return a predictable timestamp per test."""
    fixed_datetime = datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)

    with patch("octopus.study.core.datetime") as mock_dt:
        mock_dt.datetime.now.return_value = fixed_datetime
        mock_dt.datetime.fromisoformat.side_effect = datetime.fromisoformat
        mock_dt.UTC = UTC
        yield mock_dt


def pytest_configure(config):
    """Called after command line options have been parsed and all plugins loaded."""
    os.environ["RUNNING_IN_TESTSUITE"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
