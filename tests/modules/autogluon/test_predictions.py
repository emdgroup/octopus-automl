"""Unit tests for octopus.modules.autogluon.predictions.

These tests fit small AutoGluon predictors directly (bypassing the octopus
workflow) and verify that `build_predictions` produces frames matching Tako's
canonical schema: columns, dtypes, target attachment, length, label space.

Each predictor fixture is module-scoped so the fits happen once per module run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import ray
from sklearn.datasets import make_classification, make_regression
from upath import UPath

from octopus._optional.autogluon import TabularPredictor
from octopus.modules import StudyContext
from octopus.modules.autogluon.predictions import build_predictions
from octopus.types import DataPartition, MLType


def _study_context(ml_type: MLType, *, positive_class: int | None = None) -> StudyContext:
    return StudyContext(
        ml_type=ml_type,
        target_metric="ACCBAL" if ml_type == MLType.BINARY else "R2",
        target_assignments={"target": "target"},
        positive_class=positive_class,
        stratification_col=None,
        sample_id_col="row_id",
        feature_cols=[f"feat_{i}" for i in range(5)],
        row_id_col="row_id",
        output_path=UPath("/tmp"),
        log_dir=UPath("/tmp"),
    )


def _binary_data(n: int = 60) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n,
        n_features=5,
        n_informative=3,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = y
    df["row_id"] = np.arange(n)
    return df


def _multiclass_data(n: int = 90) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n,
        n_features=5,
        n_informative=4,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = y
    df["row_id"] = np.arange(n)
    return df


def _regression_data(n: int = 60) -> pd.DataFrame:
    X, y = make_regression(n_samples=n, n_features=5, noise=0.1, random_state=42, coef=False)  # type: ignore[misc]
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = y
    df["row_id"] = np.arange(n)
    return df


def _split(df: pd.DataFrame, n_test: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df.iloc[:-n_test].reset_index(drop=True), df.iloc[-n_test:].reset_index(drop=True)


def _fit(traindev: pd.DataFrame, *, eval_metric: str, tmp_path) -> TabularPredictor:
    feature_cols = [f"feat_{i}" for i in range(5)]
    train_data = traindev[[*feature_cols, "target"]]
    predictor = TabularPredictor(
        label="target",
        eval_metric=eval_metric,
        verbosity=0,
        log_to_file=False,
        path=str(tmp_path),
        learner_kwargs={"cache_data": True},
    )
    predictor.fit(train_data, time_limit=30, num_bag_folds=2, num_bag_sets=1)
    if ray.is_initialized():
        ray.shutdown()
    return predictor


@pytest.fixture(scope="module")
def binary_setup(tmp_path_factory):
    df = _binary_data()
    traindev, test = _split(df)
    predictor = _fit(traindev, eval_metric="balanced_accuracy", tmp_path=tmp_path_factory.mktemp("ag_binary"))
    ctx = _study_context(MLType.BINARY, positive_class=1)
    preds = build_predictions(
        predictor,
        study_context=ctx,
        data_traindev=traindev,
        data_test=test,
        outer_split_id=0,
        task_id=0,
    )
    yield {"traindev": traindev, "test": test, "predictions": preds, "class_labels": [0, 1]}
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture(scope="module")
def multiclass_setup(tmp_path_factory):
    df = _multiclass_data()
    traindev, test = _split(df, n_test=30)
    predictor = _fit(traindev, eval_metric="balanced_accuracy", tmp_path=tmp_path_factory.mktemp("ag_multi"))
    ctx = _study_context(MLType.MULTICLASS)
    preds = build_predictions(
        predictor,
        study_context=ctx,
        data_traindev=traindev,
        data_test=test,
        outer_split_id=0,
        task_id=0,
    )
    yield {"traindev": traindev, "test": test, "predictions": preds, "class_labels": [0, 1, 2]}
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture(scope="module")
def regression_setup(tmp_path_factory):
    df = _regression_data()
    traindev, test = _split(df)
    predictor = _fit(traindev, eval_metric="r2", tmp_path=tmp_path_factory.mktemp("ag_reg"))
    ctx = _study_context(MLType.REGRESSION)
    preds = build_predictions(
        predictor,
        study_context=ctx,
        data_traindev=traindev,
        data_test=test,
        outer_split_id=0,
        task_id=0,
    )
    yield {"traindev": traindev, "test": test, "predictions": preds}
    if ray.is_initialized():
        ray.shutdown()


_COMMON_COLS = {"row_id", "target", "prediction", "outer_split_id", "inner_split_id", "partition", "task_id"}


class TestBinarySchema:
    """Binary predictions must include the target, hard prediction, and proba columns 0/1."""

    def test_partitions_present(self, binary_setup) -> None:
        """Both DEV and TEST partitions are returned."""
        preds = binary_setup["predictions"]
        assert set(preds.keys()) == {DataPartition.DEV, DataPartition.TEST}

    @pytest.mark.parametrize("partition", [DataPartition.DEV, DataPartition.TEST])
    def test_columns(self, binary_setup, partition: DataPartition) -> None:
        """Frame carries the canonical schema columns plus int probability columns 0 and 1."""
        df = binary_setup["predictions"][partition]
        assert _COMMON_COLS.issubset(df.columns)
        assert 0 in df.columns and 1 in df.columns
        assert all(isinstance(c, int) for c in (0, 1))

    @pytest.mark.parametrize("partition", [DataPartition.DEV, DataPartition.TEST])
    def test_length_matches_source(self, binary_setup, partition: DataPartition) -> None:
        """Row count equals the source data length."""
        df = binary_setup["predictions"][partition]
        source = binary_setup["traindev"] if partition == DataPartition.DEV else binary_setup["test"]
        assert len(df) == len(source)

    @pytest.mark.parametrize("partition", [DataPartition.DEV, DataPartition.TEST])
    def test_target_attached(self, binary_setup, partition: DataPartition) -> None:
        """Target column values match the source frame in order."""
        df = binary_setup["predictions"][partition]
        source = binary_setup["traindev"] if partition == DataPartition.DEV else binary_setup["test"]
        assert df["target"].tolist() == source["target"].tolist()

    @pytest.mark.parametrize("partition", [DataPartition.DEV, DataPartition.TEST])
    def test_label_space(self, binary_setup, partition: DataPartition) -> None:
        """Hard predictions stay within the binary label set."""
        df = binary_setup["predictions"][partition]
        assert set(df["prediction"].unique()).issubset({0, 1})

    @pytest.mark.parametrize("partition", [DataPartition.DEV, DataPartition.TEST])
    def test_probabilities_sum_to_one(self, binary_setup, partition: DataPartition) -> None:
        """Probabilities for the two classes sum to one per row."""
        df = binary_setup["predictions"][partition]
        sums = df[[0, 1]].sum(axis=1)
        np.testing.assert_allclose(sums.to_numpy(), 1.0, rtol=1e-5)

    def test_inner_split_id(self, binary_setup) -> None:
        """All rows carry the AG-specific inner_split_id sentinel."""
        for df in binary_setup["predictions"].values():
            assert (df["inner_split_id"] == "autogluon").all()


class TestMulticlassSchema:
    """Multiclass predictions must include columns 0/1/2 and labels in [0, 1, 2]."""

    @pytest.mark.parametrize("partition", [DataPartition.DEV, DataPartition.TEST])
    def test_columns(self, multiclass_setup, partition: DataPartition) -> None:
        """Frame carries the canonical schema columns plus int probability columns 0, 1, 2."""
        df = multiclass_setup["predictions"][partition]
        assert _COMMON_COLS.issubset(df.columns)
        for label in (0, 1, 2):
            assert label in df.columns

    @pytest.mark.parametrize("partition", [DataPartition.DEV, DataPartition.TEST])
    def test_label_space(self, multiclass_setup, partition: DataPartition) -> None:
        """Hard predictions stay within the multiclass label set."""
        df = multiclass_setup["predictions"][partition]
        assert set(df["prediction"].unique()).issubset({0, 1, 2})

    @pytest.mark.parametrize("partition", [DataPartition.DEV, DataPartition.TEST])
    def test_target_attached(self, multiclass_setup, partition: DataPartition) -> None:
        """Target column values match the source frame in order."""
        df = multiclass_setup["predictions"][partition]
        source = multiclass_setup["traindev"] if partition == DataPartition.DEV else multiclass_setup["test"]
        assert df["target"].tolist() == source["target"].tolist()

    @pytest.mark.parametrize("partition", [DataPartition.DEV, DataPartition.TEST])
    def test_probabilities_sum_to_one(self, multiclass_setup, partition: DataPartition) -> None:
        """Probabilities for the three classes sum to one per row."""
        df = multiclass_setup["predictions"][partition]
        sums = df[[0, 1, 2]].sum(axis=1)
        np.testing.assert_allclose(sums.to_numpy(), 1.0, rtol=1e-5)


class TestRegressionSchema:
    """Regression predictions must NOT carry probability columns."""

    @pytest.mark.parametrize("partition", [DataPartition.DEV, DataPartition.TEST])
    def test_columns(self, regression_setup, partition: DataPartition) -> None:
        """Frame contains exactly the canonical columns - no probability columns."""
        df = regression_setup["predictions"][partition]
        assert set(df.columns) == _COMMON_COLS

    @pytest.mark.parametrize("partition", [DataPartition.DEV, DataPartition.TEST])
    def test_prediction_dtype_numeric(self, regression_setup, partition: DataPartition) -> None:
        """Regression predictions are numeric."""
        df = regression_setup["predictions"][partition]
        assert pd.api.types.is_numeric_dtype(df["prediction"])

    @pytest.mark.parametrize("partition", [DataPartition.DEV, DataPartition.TEST])
    def test_length_matches_source(self, regression_setup, partition: DataPartition) -> None:
        """Row count equals the source data length."""
        df = regression_setup["predictions"][partition]
        source = regression_setup["traindev"] if partition == DataPartition.DEV else regression_setup["test"]
        assert len(df) == len(source)


@pytest.fixture(scope="module")
def shuffled_binary_setup(tmp_path_factory):
    """Binary fit on a traindev frame whose index is randomly permuted.

    Locks in the contract that OOF predictions are aligned to the caller's
    row order via `reindex(source_index)`, not to AG's internal order.
    """
    df = _binary_data()
    traindev_unshuffled, test = _split(df)
    rng = np.random.default_rng(123)
    shuffled_index = rng.permutation(traindev_unshuffled.index.to_numpy())
    traindev = traindev_unshuffled.reindex(shuffled_index)
    predictor = _fit(
        traindev,
        eval_metric="balanced_accuracy",
        tmp_path=tmp_path_factory.mktemp("ag_binary_shuffled"),
    )
    ctx = _study_context(MLType.BINARY, positive_class=1)
    preds = build_predictions(
        predictor,
        study_context=ctx,
        data_traindev=traindev,
        data_test=test,
        outer_split_id=0,
        task_id=0,
    )
    yield {"predictor": predictor, "traindev": traindev, "predictions": preds}
    if ray.is_initialized():
        ray.shutdown()


class TestShuffledIndexAlignment:
    """OOF predictions must align to the caller's source-frame index, not AG's internal order."""

    def test_dev_prediction_matches_manual_reindex(self, shuffled_binary_setup) -> None:
        """`prediction` column equals `predict_oof` reindexed to the shuffled traindev index."""
        predictor = shuffled_binary_setup["predictor"]
        traindev = shuffled_binary_setup["traindev"]
        dev = shuffled_binary_setup["predictions"][DataPartition.DEV]
        expected = predictor.predict_oof(model=predictor.model_best).reindex(traindev.index).to_numpy()
        np.testing.assert_array_equal(dev["prediction"].to_numpy(), expected)

    def test_dev_target_matches_source_in_shuffled_order(self, shuffled_binary_setup) -> None:
        """`target` column reflects the shuffled source order, not the unshuffled original."""
        traindev = shuffled_binary_setup["traindev"]
        dev = shuffled_binary_setup["predictions"][DataPartition.DEV]
        assert dev["target"].tolist() == traindev["target"].tolist()
