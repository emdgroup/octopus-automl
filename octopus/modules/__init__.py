"""Init modules."""

try:
    from .autogluon import AutoGluon
except ImportError:

    class AutoGluon:  # type: ignore[no-redef]
        """AutoGluon module placeholder when AutoGluon is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "AutoGluon is not installed. Please install it with `pip install octopus[autogluon]` to use this module."
            )


from octopus.types import DataPartition, FIResultLabel

from .base import ModuleExecution, Task
from .boruta import Boruta
from .context import StudyContext
from .mrmr import Mrmr
from .result import ModuleResult, ResultType
from .roc import Roc
from .tako import Tako
from .utils import rdc_correlation_matrix

__all__ = [
    "AutoGluon",
    "Boruta",
    "DataPartition",
    "FIResultLabel",
    "ModuleExecution",
    "ModuleResult",
    "Mrmr",
    "ResultType",
    "Roc",
    "StudyContext",
    "Tako",
    "Task",
    "rdc_correlation_matrix",
]
