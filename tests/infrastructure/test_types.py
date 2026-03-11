"""Tests for octopus.types enums."""

import pytest

from octopus.types import FIComputeMethod, FIResultLabel, FIType, ShapExplainerType


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


class TestFIComputeMethod:
    """Tests for FIComputeMethod enum."""

    def test_is_strenum(self) -> None:
        assert isinstance(FIComputeMethod.INTERNAL, str)

    def test_string_values(self) -> None:
        assert FIComputeMethod.INTERNAL == "internal"
        assert FIComputeMethod.PERMUTATION == "permutation"
        assert FIComputeMethod.SHAP == "shap"
        assert FIComputeMethod.LOFO == "lofo"
        assert FIComputeMethod.CONSTANT == "constant"

    def test_member_count(self) -> None:
        assert len(FIComputeMethod) == 5

    def test_string_comparison(self) -> None:
        """FIComputeMethod members compare equal to their string values."""
        assert FIComputeMethod.PERMUTATION == "permutation"
        assert "shap" == FIComputeMethod.SHAP

    def test_construction_from_string(self) -> None:
        """FIComputeMethod can be constructed from a plain string."""
        assert FIComputeMethod("internal") == FIComputeMethod.INTERNAL
        assert FIComputeMethod("permutation") == FIComputeMethod.PERMUTATION

    def test_is_subset_of_fi_result_label(self) -> None:
        """Every FIComputeMethod value exists in FIResultLabel."""
        result_label_values = {m.value for m in FIResultLabel}
        for member in FIComputeMethod:
            assert member.value in result_label_values


class TestFIType:
    """Tests for FIType enum."""

    def test_is_strenum(self) -> None:
        assert isinstance(FIType.PERMUTATION, str)

    def test_string_values(self) -> None:
        assert FIType.PERMUTATION == "permutation"
        assert FIType.GROUP_PERMUTATION == "group_permutation"
        assert FIType.SHAP == "shap"

    def test_member_count(self) -> None:
        assert len(FIType) == 3

    def test_construction_from_string(self) -> None:
        """FIType can be constructed from a plain string."""
        assert FIType("permutation") == FIType.PERMUTATION
        assert FIType("group_permutation") == FIType.GROUP_PERMUTATION
        assert FIType("shap") == FIType.SHAP

    def test_invalid_string_raises(self) -> None:
        """FIType rejects unknown strings."""
        with pytest.raises(ValueError):
            FIType("invalid")


class TestShapExplainerType:
    """Tests for ShapExplainerType enum."""

    def test_is_strenum(self) -> None:
        assert isinstance(ShapExplainerType.KERNEL, str)

    def test_string_values(self) -> None:
        assert ShapExplainerType.KERNEL == "kernel"
        assert ShapExplainerType.PERMUTATION == "permutation"
        assert ShapExplainerType.EXACT == "exact"

    def test_member_count(self) -> None:
        assert len(ShapExplainerType) == 3

    def test_construction_from_string(self) -> None:
        """ShapExplainerType can be constructed from a plain string."""
        assert ShapExplainerType("kernel") == ShapExplainerType.KERNEL
        assert ShapExplainerType("permutation") == ShapExplainerType.PERMUTATION
        assert ShapExplainerType("exact") == ShapExplainerType.EXACT

    def test_invalid_string_raises(self) -> None:
        """ShapExplainerType rejects unknown strings."""
        with pytest.raises(ValueError):
            ShapExplainerType("invalid")
