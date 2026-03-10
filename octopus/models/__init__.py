"""Init."""

# Import model definitions so that @Models.register decorators run on import
from ..types import ModelName
from .classification_models import *  # noqa: F403
from .core import Models
from .regression_models import *  # noqa: F403
from .time_to_event_models import *  # noqa: F403

__all__ = ["ModelName", "Models"]
