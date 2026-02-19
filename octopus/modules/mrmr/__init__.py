"""Init Mrmr."""

from .core import _maxrminr as maxrminr
from .module import Mrmr

# Legacy Core class (deprecated - use Mrmr directly)
# from .core import MrmrCore

__all__ = ["Mrmr", "maxrminr"]
