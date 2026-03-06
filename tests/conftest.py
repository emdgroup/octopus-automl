import os


def pytest_configure(config):
    """Called after command line options have been parsed and all plugins loaded.

    Set environment variables for all tests.
    """
    os.environ["RUNNING_IN_TESTSUITE"] = "1"
    # Use non-interactive matplotlib backend to prevent GUI-related issues
    os.environ["MPLBACKEND"] = "Agg"


def pytest_addoption(parser):
    parser.addoption(
        "--run-examples",
        action="store_true",
        default=False,
        help="Run example tests",
    )
