"""Exceptions."""


class OptionalImportError(ImportError):
    """An attempt was made to import an optional but uninstalled dependency."""


class UnknownModelError(Exception):
    """An attempt was made to use an model that does not exists."""


class UnknownMetricError(Exception):
    """An attempt was made to use an metric that does not exists."""


class SingleClassFoldError(ValueError):
    """A cross-validation fold contains only a single class label."""
