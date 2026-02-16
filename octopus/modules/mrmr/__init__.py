"""Init Mrmr."""

from .module import Mrmr
from .core import _maxrminr as maxrminr

# Legacy Core class (deprecated - use Mrmr directly)
# from .core import MrmrCore

__all__ = ["Mrmr", "maxrminr"]
