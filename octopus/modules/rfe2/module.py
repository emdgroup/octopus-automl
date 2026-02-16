"""Rfe2 module configuration (RFE with Octo optimization)."""

from __future__ import annotations

from attrs import define, field, validators

from octopus.modules.octo.module import Octo

from .core import Rfe2Module


@define
class Rfe2(Octo):
    """Rfe2 module for recursive feature elimination with Octo optimization.

    Extends Octo to add RFE functionality. First runs Octo optimization to get
    a best bag, then iteratively removes features based on feature importances.

    Configuration:
        (inherits all Octo configuration)
        min_features_to_select: Minimum number of features to keep
        fi_method_rfe: Feature importance method for RFE
        selection_method: Method to select best solution (best or parsimonious)
        abs_on_fi: Convert negative feature importances to positive
    """

    min_features_to_select: int = field(validator=[validators.instance_of(int)], default=1)
    """Minimum number of features to be selected."""

    fi_method_rfe: str = field(validator=[validators.in_(["permutation", "shap"])], default="permutation")
    """Feature importance method for RFE."""

    selection_method: str = field(validator=[validators.in_(["best", "parsimonious"])], default="best")
    """Method to select best solution. Parsimonious: smallest solutions within sem."""

    abs_on_fi: bool = field(validator=[validators.instance_of(bool)], default=False)
    """Convert negative feature importances to positive (abs())."""

    def __attrs_post_init__(self):
        # Call parent post_init
        super().__attrs_post_init__()

        # overwrite fi_methods_bestbag for Octo
        self.fi_methods_bestbag = [self.fi_method_rfe]

    def create_module(self) -> Rfe2Module:
        """Create Rfe2Module execution instance."""
        return Rfe2Module(config=self)
