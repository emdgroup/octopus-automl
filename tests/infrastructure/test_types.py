"""Tests for octopus.types enums."""

from octopus.types import FIResultLabel


class TestFIResultLabel:
    """Tests for FIResultLabel enum."""

    def test_is_strenum(self) -> None:
        assert isinstance(FIResultLabel.INTERNAL, str)

    def test_string_values(self) -> None:
        assert FIResultLabel.INTERNAL == "internal"
        assert FIResultLabel.PERMUTATION == "permutation"
        assert FIResultLabel.SHAP == "shap"
        assert FIResultLabel.LOFO == "lofo"
        assert FIResultLabel.CONSTANT == "constant"
        assert FIResultLabel.COUNTS == "counts"
        assert FIResultLabel.COUNTS_RELATIVE == "counts_relative"

    def test_member_count(self) -> None:
        assert len(FIResultLabel) == 7

    def test_string_comparison(self) -> None:
        """FIResultLabel members compare equal to their string values."""
        assert FIResultLabel.PERMUTATION == "permutation"
        assert "shap" == FIResultLabel.SHAP
