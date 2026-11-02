"""Machine learning models config."""

from typing import Any, Protocol, runtime_checkable

from attrs import Attribute, define, field, validators

from octopus.models.hyperparameter import Hyperparameter
from octopus.types import (
    FIComputeMethod,
    MLType,
    to_ml_types_frozenset,
    validate_ml_types,
)


def validate_hyperparameters(instance: "ModelConfig", attribute: Attribute, value: list[Hyperparameter]) -> None:
    """Validate hyperparameters.

    Make sure that the hyperparameters do not contain names
    that match the instance's n_jobs or model_seed.

    Args:
        instance: The instance of ModelConfig being validated.
        attribute: The name of the attribute being validated.
        value: The list of hyperparameters to validate.

    Raises:
        ValueError: If any hyperparameter's name matches n_jobs or model_seed.
    """
    forbidden_names = {"n_jobs", "model_seed"}

    for hyperparameter in value:
        if hyperparameter.name in forbidden_names:
            raise ValueError(f"""Hyperparameter '{hyperparameter.name}' is not allowed in 'hyperparameters'.""")


class EstimatorLike(Protocol):
    """Interface of the model classes registered in octopus (scikit-learn estimator API).

    Model classes do not need to inherit from this class; any class providing these
    methods matches, e.g. scikit-learn, XGBoost and CatBoost estimators.
    """

    def fit(self, X: Any, y: Any, /) -> Any:
        """Fit the model."""
        ...

    def predict(self, X: Any, /) -> Any:
        """Predict."""
        ...

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get the model parameters."""
        ...

    def set_params(self, **params: Any) -> Any:
        """Set the model parameters."""
        ...


@runtime_checkable
class ClassifierLike(EstimatorLike, Protocol):
    """Interface of the model classes registered for classification (binary or multiclass).

    In addition to the estimator interface, classifiers must provide class probabilities.
    Fitted classifiers must also expose the ``classes_`` attribute (scikit-learn convention),
    which cannot be part of the runtime-checkable protocol because it only exists after fitting.
    """

    def predict_proba(self, X: Any, /) -> Any:
        """Predict class probabilities."""
        ...


_CLASSIFICATION_ML_TYPES = frozenset({MLType.BINARY, MLType.MULTICLASS})


def validate_classifier_interface(instance: "ModelConfig", attribute: Attribute, value: frozenset[MLType]) -> None:
    """Validate that models registered for classification provide the classifier interface.

    Args:
        instance: The instance of ModelConfig being validated.
        attribute: The name of the attribute being validated.
        value: The ml_types of the model.

    Raises:
        TypeError: If the model supports a classification ml_type but its class does not
            implement ``predict_proba``.
    """
    if not value & _CLASSIFICATION_ML_TYPES:
        return
    model_class = instance.model_class
    if not (isinstance(model_class, type) and issubclass(model_class, ClassifierLike)):
        classification_types = ", ".join(sorted(t.value for t in value & _CLASSIFICATION_ML_TYPES))
        raise TypeError(
            f"Model class {getattr(model_class, '__name__', model_class)!r} is registered for "
            f"{classification_types} but does not implement predict_proba()."
        )


@define(slots=False)
class ModelConfig:
    """Create model config."""

    model_class: type[EstimatorLike]
    fi_method: FIComputeMethod = field(converter=FIComputeMethod)
    ml_types: frozenset[MLType] = field(
        converter=to_ml_types_frozenset, validator=[validate_ml_types, validate_classifier_interface]
    )
    hyperparameters: list[Hyperparameter] = field(validator=validate_hyperparameters)
    n_repeats: None | int = field(factory=lambda: None)
    n_jobs: None | str = field(factory=lambda: "n_jobs")
    model_seed: None | str = field(factory=lambda: "model_seed")
    chpo_compatible: bool = field(default=False)
    scaler: None | str = field(default=None, validator=validators.in_([None, "StandardScaler"]))
    imputation_required: bool = field(default=True)
    categorical_enabled: bool = field(default=False)
    default: bool = field(default=False)

    def supports_ml_type(self, ml_type: MLType) -> bool:
        """Check if this model supports the given ml_type."""
        return ml_type in self.ml_types
