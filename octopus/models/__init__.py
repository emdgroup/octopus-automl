"""Init."""

# Import model definitions so that @Models.register decorators run on import
from .classification_models import *  # noqa: F403
from .core import Models
from .model_name import ModelName
from .regression_models import *  # noqa: F403
from .time_to_event_models import *  # noqa: F403

__all__ = ["ModelName", "Models"]
