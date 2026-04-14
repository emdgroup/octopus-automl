"""Tests for study output path behavior."""

import tempfile

import pytest

from tests.poststudy.test_predict import _create_classification_study


def test_output_path_raises_before_fit():
    """Test that accessing output_path before fit() raises RuntimeError."""
    with tempfile.TemporaryDirectory() as tmp:
        study, _ = _create_classification_study(tmp)
        with pytest.raises(RuntimeError, match="not available until fit"):
            _ = study.output_path


def test_output_path_has_correct_timestamp_format():
    """Test that output_path uses the timestamp from fit()."""
    with tempfile.TemporaryDirectory() as tmp:
        study, df = _create_classification_study(tmp)
        study.fit(data=df)

        # The mocked datetime returns 2024-01-15 10:30:45
        expected_name = f"{study.study_name}-20240115_103045"
        assert study.output_path.name == expected_name


def test_fit_raises_on_second_call():
    """Test that calling fit() twice raises RuntimeError."""
    with tempfile.TemporaryDirectory() as tmp:
        study, df = _create_classification_study(tmp)

        study.fit(data=df)

        with pytest.raises(RuntimeError, match="fit\\(\\) can only be called once"):
            study.fit(data=df)
