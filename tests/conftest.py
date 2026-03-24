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

    # Pin AWS_DEFAULT_REGION to us-east-1 for the entire test session.
    # This prevents IllegalLocationConstraintException errors in moto when
    # the host environment has AWS_DEFAULT_REGION set to a non-us-east-1
    # region (e.g. eu-central-1).  The S3 CreateBucket API only allows
    # omitting LocationConstraint when the region is us-east-1.
    # Uses setdefault so explicit overrides for real-S3 tests are respected.
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


def pytest_addoption(parser):
    parser.addoption(
        "--run-examples",
        action="store_true",
        default=False,
        help="Run example tests",
    )
