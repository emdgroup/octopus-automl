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
from .efs import Efs
from .mrmr import Mrmr
from .octo import Octo
from .result import ModuleResult, ResultType
from .rfe import Rfe
from .rfe2 import Rfe2
from .roc import Roc
from .sfs import Sfs
from .utils import rdc_correlation_matrix

__all__ = [
    "AutoGluon",
    "Boruta",
    "DataPartition",
    "Efs",
    "FIResultLabel",
    "ModuleExecution",
    "ModuleResult",
    "Mrmr",
    "Octo",
    "ResultType",
    "Rfe",
    "Rfe2",
    "Roc",
    "Sfs",
    "StudyContext",
    "Task",
    "rdc_correlation_matrix",
]
