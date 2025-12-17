"""Octopus - AutoML for small Datasets."""

import sys

from octopus.study import OctoClassification, OctoRegression, OctoStudy, OctoTimeToEvent  # noqa: F401

if not sys.version_info >= (3, 12):
    raise ValueError("Minimum required Python version is 3.12")
