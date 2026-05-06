"""Unit tests for octopus.modules.autogluon.scoring.

These tests hand-build prediction frames in the canonical schema and verify
`compute_scores` produces a long-format scores DataFrame whose values agree
with `get_performance_from_predictions` called directly. No AG fit is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from upath import UPath

from octopus.metrics import Metrics
from octopus.metrics.utils import get_performance_from_predictions
from octopus.modules import StudyContext
from octopus.modules.autogluon.core import _compute_scores as compute_scores
from octopus.types import DataPartition, MLType


def _study_context(ml_type: MLType, *, target_metric: str, positive_class: int | None = None) -> StudyContext:
    return StudyContext(
        ml_type=ml_type,
        target_metric=target_metric,
        target_assignments={"target": "y"},
        positive_class=positive_class,
        stratification_col=None,
        sample_id_col="id",
        feature_cols=[],
        row_id_col="id",
        output_path=UPath("/tmp"),
        log_dir=UPath("/tmp"),
    )


def _binary_predictions() -> dict[DataPartition, pd.DataFrame]:
    rng = np.random.default_rng(0)
    rows = 40
    y_dev = rng.integers(0, 2, size=rows)
    y_test = rng.integers(0, 2, size=rows)
    proba_dev = rng.uniform(0.0, 1.0, size=rows)
    proba_test = rng.uniform(0.0, 1.0, size=rows)
    dev = pd.DataFrame(
        {
            "id": np.arange(rows),
            "y": y_dev,
            "prediction": (proba_dev >= 0.5).astype(int),
            0: 1.0 - proba_dev,
            1: proba_dev,
            "outer_split_id": 0,
            "inner_split_id": "autogluon",
            "partition": DataPartition.DEV,
            "task_id": 0,
        }
    )
    test = pd.DataFrame(
        {
            "id": np.arange(rows, 2 * rows),
            "y": y_test,
            "prediction": (proba_test >= 0.5).astype(int),
            0: 1.0 - proba_test,
            1: proba_test,
            "outer_split_id": 0,
            "inner_split_id": "autogluon",
            "partition": DataPartition.TEST,
            "task_id": 0,
        }
    )
    return {DataPartition.DEV: dev, DataPartition.TEST: test}


def _multiclass_predictions() -> dict[DataPartition, pd.DataFrame]:
    rng = np.random.default_rng(1)
    rows = 30
    y_dev = rng.integers(0, 3, size=rows)
    y_test = rng.integers(0, 3, size=rows)
    proba_dev = rng.dirichlet(alpha=(1.0, 1.0, 1.0), size=rows)
    proba_test = rng.dirichlet(alpha=(1.0, 1.0, 1.0), size=rows)
    dev = pd.DataFrame(
        {
            "id": np.arange(rows),
            "y": y_dev,
            "prediction": proba_dev.argmax(axis=1),
            0: proba_dev[:, 0],
            1: proba_dev[:, 1],
            2: proba_dev[:, 2],
            "outer_split_id": 0,
            "inner_split_id": "autogluon",
            "partition": DataPartition.DEV,
            "task_id": 0,
        }
    )
    test = pd.DataFrame(
        {
            "id": np.arange(rows, 2 * rows),
            "y": y_test,
            "prediction": proba_test.argmax(axis=1),
            0: proba_test[:, 0],
            1: proba_test[:, 1],
            2: proba_test[:, 2],
            "outer_split_id": 0,
            "inner_split_id": "autogluon",
            "partition": DataPartition.TEST,
            "task_id": 0,
        }
    )
    return {DataPartition.DEV: dev, DataPartition.TEST: test}


def _regression_predictions() -> dict[DataPartition, pd.DataFrame]:
    rng = np.random.default_rng(2)
    rows = 30
    y_dev = rng.normal(size=rows)
    y_test = rng.normal(size=rows)
    dev = pd.DataFrame(
        {
            "id": np.arange(rows),
            "y": y_dev,
            "prediction": y_dev + rng.normal(scale=0.1, size=rows),
            "outer_split_id": 0,
            "inner_split_id": "autogluon",
            "partition": DataPartition.DEV,
            "task_id": 0,
        }
    )
    test = pd.DataFrame(
        {
            "id": np.arange(rows, 2 * rows),
            "y": y_test,
            "prediction": y_test + rng.normal(scale=0.1, size=rows),
            "outer_split_id": 0,
            "inner_split_id": "autogluon",
            "partition": DataPartition.TEST,
            "task_id": 0,
        }
    )
    return {DataPartition.DEV: dev, DataPartition.TEST: test}


class TestComputeScores:
    """Verify compute_scores delegates correctly to get_performance_from_predictions."""

    def test_binary_score_shape(self) -> None:
        """Binary scores cover all binary metrics for both partitions in long format."""
        ctx = _study_context(MLType.BINARY, target_metric="ACCBAL", positive_class=1)
        scores = compute_scores(_binary_predictions(), study_context=ctx)
        expected_metrics = set(Metrics.get_by_type(MLType.BINARY))
        assert set(scores["metric"]) == expected_metrics
        assert set(scores["partition"]) == {DataPartition.DEV, DataPartition.TEST}
        assert (scores["aggregation"] == "ensemble").all()
        assert scores["split"].isna().all()
        assert len(scores) == 2 * len(expected_metrics)

    def test_multiclass_score_shape(self) -> None:
        """Multiclass scores cover all multiclass metrics for both partitions."""
        ctx = _study_context(MLType.MULTICLASS, target_metric="ACCBAL_MC")
        scores = compute_scores(_multiclass_predictions(), study_context=ctx)
        expected_metrics = set(Metrics.get_by_type(MLType.MULTICLASS))
        assert set(scores["metric"]) == expected_metrics
        assert len(scores) == 2 * len(expected_metrics)

    def test_regression_score_shape(self) -> None:
        """Regression scores cover all regression metrics for both partitions."""
        ctx = _study_context(MLType.REGRESSION, target_metric="R2")
        scores = compute_scores(_regression_predictions(), study_context=ctx)
        expected_metrics = set(Metrics.get_by_type(MLType.REGRESSION))
        assert set(scores["metric"]) == expected_metrics
        assert len(scores) == 2 * len(expected_metrics)

    @pytest.mark.parametrize(
        ("ml_type", "metric_name", "positive_class", "factory"),
        [
            (MLType.BINARY, "AUCROC", 1, _binary_predictions),
            (MLType.BINARY, "ACCBAL", 1, _binary_predictions),
            (MLType.MULTICLASS, "ACCBAL_MC", None, _multiclass_predictions),
            (MLType.MULTICLASS, "AUCROC_MACRO", None, _multiclass_predictions),
            (MLType.REGRESSION, "R2", None, _regression_predictions),
            (MLType.REGRESSION, "RMSE", None, _regression_predictions),
        ],
    )
    def test_values_match_canonical_helper(
        self,
        ml_type: MLType,
        metric_name: str,
        positive_class: int | None,
        factory,
    ) -> None:
        """Each value in the scores frame must equal the helper's direct result."""
        ctx = _study_context(ml_type, target_metric=metric_name, positive_class=positive_class)
        predictions = factory()
        scores = compute_scores(predictions, study_context=ctx)

        reference = get_performance_from_predictions(
            {"ensemble": dict(predictions.items())},
            target_metric=metric_name,
            target_assignments=ctx.target_assignments,
            positive_class=positive_class,
        )

        for partition in (DataPartition.DEV, DataPartition.TEST):
            row = scores[(scores["metric"] == metric_name) & (scores["partition"] == partition)]
            assert len(row) == 1
            assert float(row["value"].iloc[0]) == pytest.approx(reference["ensemble"][partition], rel=1e-9)
