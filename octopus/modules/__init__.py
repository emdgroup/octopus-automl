"""Init modules."""

import os
import platform

import threadpoolctl

try:
    from .autogluon import AutoGluon
except ImportError:

    class AutoGluon:  # type: ignore[no-redef]
        """AutoGluon module placeholder when AutoGluon is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "AutoGluon is not installed. Please install it with `pip install octopus[autogluon]` to use this module."
            )


from octopus.types import FIDataset, FIResultLabel

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
    "Efs",
    "FIDataset",
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

_PARALLELIZATION_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

for env_var in _PARALLELIZATION_ENV_VARS:
    if (num_threads := os.environ.setdefault(env_var, "1")) != "1":
        if platform.system() == "Darwin":
            print(
                f"Warning: {env_var} is set to {num_threads} on macOS. "
                "This may lead to issues/crashes in some libraries. "
                f"Consider setting {env_var}=1 for better stability."
            )
        else:
            print(
                f"Warning: {env_var} is set to {num_threads}. "
                "This may lead to resource oversubscription and slow execution. "
                f"Consider setting {env_var}=1 or at least perform "
                "a thorough threading performance evaluation."
            )

_THREADPOOL_LIMIT = threadpoolctl.threadpool_limits(limits=1)

del os
del platform
del threadpoolctl
