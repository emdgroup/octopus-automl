"""Tests for octopus.poststudy module.

Self-contained: runs a minimal classification study in a session-scoped
fixture so that tests do not depend on pre-existing study directories.

Covers:
- study_io.py (StudyInfo, TaskOutersplitLoader, load_study_config, load_study_information)
- task_predictor.py (OctoPredictor)
- task_evaluator_test.py (OctoTestEvaluator)
- feature_importance.py (permutation FI)
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from upath import UPath

from octopus.example_data import load_breast_cancer_data
from octopus.modules import Octo
from octopus.poststudy.analysis.evaluator import OctoTestEvaluator
from octopus.poststudy.predict.predictor import OctoPredictor
from octopus.poststudy.study_io import StudyInfo, load_study_config, load_study_information
from octopus.study import OctoClassification
from octopus.types import FIComputeMethod, FIType, MLType, ModelName
from octopus.utils import parquet_load


@pytest.fixture(autouse=True)
def _no_plotly_show(monkeypatch):
    """Patch plotly Figure.show() to prevent opening browser windows."""
    monkeypatch.setattr(go.Figure, "show", lambda *_args, **_kwargs: None)


_STUDY_TMPDIR: tempfile.TemporaryDirectory | None = None


def _create_classification_study(tmp_path: str) -> tuple[OctoClassification, pd.DataFrame]:
    """Create a minimal classification study and data without calling fit().

    Parameters:
        tmp_path: Path to temporary directory for study output.

    Returns:
        Tuple of (study, data) for use in tests.
    """
    # Use breast cancer dataset (enough samples for nonzero FI)
    df, features, _targets = load_breast_cancer_data()
    features = features[:5]  # Use only first 5 features for faster testing

    study = OctoClassification(
        study_name="predict_test_study",
        target_metric="ACCBAL",
        feature_cols=features,
        target_col="target",
        sample_id_col="index",
        stratification_col="target",
        outer_split_seed=1234,
        n_outer_splits=2,
        studies_directory=tmp_path,
        workflow=[
            Octo(
                description="step_1_octo",
                task_id=0,
                depends_on=None,
                n_inner_splits=3,
                models=[ModelName.ExtraTreesClassifier],
                max_outliers=0,
                fi_methods=[FIComputeMethod.PERMUTATION],
                n_startup_trials=3,
                n_trials=5,
                max_features=5,
                penalty_factor=1.0,
                ensemble_selection=True,
                n_ensemble_candidates=5,
            ),
        ],
    )
    return study, df


def _run_classification_study() -> str:
    """Run a minimal classification study and return the study path."""
    global _STUDY_TMPDIR  # noqa: PLW0603
    _STUDY_TMPDIR = tempfile.TemporaryDirectory()
    tmp = _STUDY_TMPDIR.name

    study, df = _create_classification_study(tmp)
    study.fit(data=df)
    return str(study.output_path)


@pytest.fixture(scope="module")
def study_path():
    """Run a study once per module and yield its path."""
    path = _run_classification_study()
    yield path
    if _STUDY_TMPDIR is not None:
        _STUDY_TMPDIR.cleanup()


@pytest.fixture(scope="module")
def study_info(study_path):
    """Module-scoped StudyInfo."""
    return load_study_information(study_path)


@pytest.fixture(scope="module")
def tpt(study_info):
    """Module-scoped OctoTestEvaluator."""
    return OctoTestEvaluator(study_info=study_info, task_id=0)


@pytest.fixture(scope="module")
def tp(study_info):
    """Module-scoped OctoPredictor."""
    return OctoPredictor(study_info=study_info, task_id=0)


class TestStudyIO:
    """Tests for study_io module functions and StudyInfo."""

    def test_load_config(self, study_path):
        """Verify load_study_config returns config with correct ml_type and folds."""
        cfg = load_study_config(study_path)
        assert cfg["ml_type"] == MLType.BINARY
        assert cfg["n_outer_splits"] == 2

    def test_load_study_information(self, study_path):
        """Verify load_study_information returns StudyInfo with correct fields."""
        info = load_study_information(study_path)
        assert isinstance(info, StudyInfo)
        assert info.ml_type == MLType.BINARY
        assert info.target_metric == "ACCBAL"
        assert len(info.feature_cols) == 5
        assert info.n_outer_splits == 2
        assert info.target_col == "target"
        assert info.row_id_col is not None

    def test_load_study_information_missing_row_id_col(self, study_path):
        """Verify load_study_information returns StudyInfo with row_id_col=None for old config."""
        import json as _json  # noqa: PLC0415

        cfg = load_study_config(study_path)
        cfg["prepared"].pop("row_id_col", None)
        # Write modified config to a temp copy
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            import shutil  # noqa: PLC0415

            shutil.copytree(study_path, Path(tmp) / "study", dirs_exist_ok=True)
            with (Path(tmp) / "study" / "study_config.json").open("w") as f:
                _json.dump(cfg, f)
            info = load_study_information(str(Path(tmp) / "study"))
            assert info.row_id_col is None

    def test_predictor_raises_on_none_row_id_col(self, study_path):
        """Verify predictor raises ValueError when row_id_col is None."""
        import json as _json  # noqa: PLC0415

        cfg = load_study_config(study_path)
        cfg["prepared"].pop("row_id_col", None)
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            import shutil  # noqa: PLC0415

            shutil.copytree(study_path, Path(tmp) / "study", dirs_exist_ok=True)
            with (Path(tmp) / "study" / "study_config.json").open("w") as f:
                _json.dump(cfg, f)
            info = load_study_information(str(Path(tmp) / "study"))
            with pytest.raises(ValueError, match="row_id_col"):
                OctoPredictor(study_info=info, task_id=0)

    def test_predictor_accepts_study_info(self, study_path):
        """Verify predictor accepts StudyInfo."""
        info = load_study_information(study_path)
        predictor = OctoPredictor(study_info=info, task_id=0)
        assert predictor.study_info.ml_type == MLType.BINARY

    def test_test_predictor_accepts_study_info(self, study_path):
        """Verify OctoTestEvaluator accepts StudyInfo."""
        info = load_study_information(study_path)
        tpt = OctoTestEvaluator(study_info=info, task_id=0)
        assert tpt.study_info.ml_type == MLType.BINARY


class TestLoadPartitionGuards:
    """Test duplicate row ID guards in load_partition."""

    def test_duplicate_row_ids_in_json(self, study_path):
        """Verify ValueError on duplicate row IDs in split_row_ids.json."""
        import json as _json  # noqa: PLC0415

        from octopus.poststudy.study_io import load_partition as _load_partition  # noqa: PLC0415

        split_ids_path = UPath(study_path) / "outersplit0" / "split_row_ids.json"
        with split_ids_path.open() as f:
            split_info = _json.load(f)

        original_ids = split_info["traindev_row_ids"]
        split_info["traindev_row_ids"] = [*original_ids, original_ids[0]]

        with split_ids_path.open("w") as f:
            _json.dump(split_info, f)

        try:
            with pytest.raises(ValueError, match="duplicate row IDs"):
                _load_partition(UPath(study_path), 0, "traindev")
        finally:
            split_info["traindev_row_ids"] = original_ids
            with split_ids_path.open("w") as f:
                _json.dump(split_info, f)

    def test_duplicate_row_ids_in_parquet(self, study_path):
        """Verify ValueError on duplicate values in data_prepared.parquet ID column."""
        from octopus.poststudy.study_io import load_partition as _load_partition  # noqa: PLC0415

        prepared_path = UPath(study_path) / "data_prepared.parquet"
        original_data = parquet_load(prepared_path)

        duped = pd.concat([original_data, original_data.iloc[:1]], ignore_index=True)
        duped.to_parquet(prepared_path)

        try:
            with pytest.raises(ValueError, match="duplicate values"):
                _load_partition(UPath(study_path), 0, "traindev")
        finally:
            original_data.to_parquet(prepared_path)

    def test_valid_row_ids(self, study_path):
        """Verify load_partition succeeds with valid unique IDs."""
        from octopus.poststudy.study_io import load_partition as _load_partition  # noqa: PLC0415

        result = _load_partition(UPath(study_path), 0, "traindev")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestOctoTestEvaluatorProperties:
    """Test OctoTestEvaluator properties."""

    def test_ml_type(self, tpt):
        """Verify ml_type is classification."""
        assert tpt.study_info.ml_type == MLType.BINARY

    def test_n_outersplits(self, tpt):
        """Verify n_outersplits matches study configuration."""
        assert tpt.study_info.n_outersplits == 2

    def test_outersplits(self, tpt):
        """Verify outersplits returns correct split indices."""
        assert tpt.study_info.outersplits == [0, 1]

    def test_feature_cols(self, tpt):
        """Verify feature_cols is non-empty."""
        assert len(tpt.feature_cols) > 0

    def test_classes(self, tpt):
        """Verify classes_ contains two classes for binary classification."""
        assert len(tpt.classes_) == 2


class TestOctoTestEvaluatorPredict:
    """Test OctoTestEvaluator.predict()."""

    def test_predict_returns_dataframe(self, tpt):
        """Verify predict always returns a DataFrame with expected columns."""
        result = tpt.predict()
        assert isinstance(result, pd.DataFrame)
        for col in ("outersplit", "row_id", "prediction", "target"):
            assert col in result.columns
        assert result["outersplit"].nunique() == 2
        assert len(result) > 0


class TestOctoTestEvaluatorPredictProba:
    """Test OctoTestEvaluator.predict_proba()."""

    def test_predict_proba_returns_dataframe(self, tpt):
        """Verify predict_proba always returns DataFrame with outersplit and target."""
        result = tpt.predict_proba()
        assert isinstance(result, pd.DataFrame)
        assert "outersplit" in result.columns
        assert "target" in result.columns
        assert result["outersplit"].nunique() == 2

    def test_predict_proba_sums_to_one(self, tpt):
        """Verify predicted probabilities sum to 1 for each sample."""
        result = tpt.predict_proba()
        class_cols = [c for c in result.columns if c not in ("outersplit", "row_id", "target")]
        np.testing.assert_allclose(result[class_cols].sum(axis=1), 1.0, atol=1e-6)


class TestOctoTestEvaluatorPerformance:
    """Test OctoTestEvaluator.performance()."""

    def test_performance_default(self, tpt):
        """Verify default performance returns wide DataFrame with Mean and Merged."""
        result = tpt.performance()
        assert isinstance(result, pd.DataFrame)
        assert "Mean" in result.index
        assert "Merged" in result.index
        n_splits = tpt.study_info.n_outersplits
        assert len(result) == n_splits + 2

    def test_performance_multiple_metrics(self, tpt):
        """Verify performance with multiple metrics returns correct columns."""
        result = tpt.performance(metrics=["ACCBAL", "AUCROC"])
        assert list(result.columns) == ["ACCBAL", "AUCROC"]
        assert "Mean" in result.index
        assert "Merged" in result.index


class TestOctoTestEvaluatorFI:
    """Test OctoTestEvaluator.calculate_fi()."""

    def test_fi_permutation(self, tpt):
        """Verify permutation FI returns DataFrame with per-split and ensemble rows."""
        fi = tpt.calculate_fi(fi_type=FIType.PERMUTATION, n_repeats=2)
        assert isinstance(fi, pd.DataFrame)
        assert "feature" in fi.columns
        assert "importance_mean" in fi.columns
        assert "fi_source" in fi.columns
        # Should have per-split + ensemble rows
        sources = fi["fi_source"].unique()
        assert "ensemble" in sources

    def test_fi_invalid_type(self, tpt):
        """Verify invalid fi_type raises ValueError."""
        with pytest.raises(ValueError, match="is not a valid FIType"):
            tpt.calculate_fi(fi_type="invalid_method")


class TestGetTargetColumns:
    """Test OctoTestEvaluator._get_target_columns helper."""

    def test_single_target_returns_target_key(self, tpt):
        """Verify single-target tasks produce {'target': ...} dict."""
        # Binary classification has a single target assignment
        test_df = pd.DataFrame({"target": [0, 1, 0], "other": [1, 2, 3]})
        # Override target_assignments to a known single-target mapping
        original = tpt._study_info.target_assignments
        try:
            tpt._study_info = StudyInfo(
                path=tpt._study_info.path,
                n_outer_splits=tpt._study_info.n_outer_splits,
                workflow_tasks=tpt._study_info.workflow_tasks,
                outersplit_dirs=tpt._study_info.outersplit_dirs,
                ml_type=tpt._study_info.ml_type,
                target_metric=tpt._study_info.target_metric,
                target_col=tpt._study_info.target_col,
                target_assignments={"default": "target"},
                positive_class=tpt._study_info.positive_class,
                row_id_col=tpt._study_info.row_id_col,
                feature_cols=tpt._study_info.feature_cols,
            )
            result = tpt._get_target_columns(test_df)
            assert list(result.keys()) == ["target"]
            np.testing.assert_array_equal(result["target"], [0, 1, 0])
        finally:
            tpt._study_info = StudyInfo(
                path=tpt._study_info.path,
                n_outer_splits=tpt._study_info.n_outer_splits,
                workflow_tasks=tpt._study_info.workflow_tasks,
                outersplit_dirs=tpt._study_info.outersplit_dirs,
                ml_type=tpt._study_info.ml_type,
                target_metric=tpt._study_info.target_metric,
                target_col=tpt._study_info.target_col,
                target_assignments=original,
                positive_class=tpt._study_info.positive_class,
                row_id_col=tpt._study_info.row_id_col,
                feature_cols=tpt._study_info.feature_cols,
            )

    def test_multi_target_returns_prefixed_keys(self, tpt):
        """Verify multi-target tasks produce {'target_role': ...} dict for each role."""
        test_df = pd.DataFrame(
            {
                "time_col": [10.0, 20.0, 30.0],
                "event_col": [1, 0, 1],
            }
        )
        original_ml_type = tpt._study_info.ml_type
        original_target_assignments = tpt._study_info.target_assignments
        try:
            tpt._study_info = StudyInfo(
                path=tpt._study_info.path,
                n_outer_splits=tpt._study_info.n_outer_splits,
                workflow_tasks=tpt._study_info.workflow_tasks,
                outersplit_dirs=tpt._study_info.outersplit_dirs,
                ml_type=MLType.TIMETOEVENT,
                target_metric=tpt._study_info.target_metric,
                target_col=tpt._study_info.target_col,
                target_assignments={"duration": "time_col", "event": "event_col"},
                positive_class=tpt._study_info.positive_class,
                row_id_col=tpt._study_info.row_id_col,
                feature_cols=tpt._study_info.feature_cols,
            )
            result = tpt._get_target_columns(test_df)
            assert "target_duration" in result
            assert "target_event" in result
            assert "target" not in result
            np.testing.assert_array_equal(result["target_duration"], [10.0, 20.0, 30.0])
            np.testing.assert_array_equal(result["target_event"], [1, 0, 1])
        finally:
            tpt._study_info = StudyInfo(
                path=tpt._study_info.path,
                n_outer_splits=tpt._study_info.n_outer_splits,
                workflow_tasks=tpt._study_info.workflow_tasks,
                outersplit_dirs=tpt._study_info.outersplit_dirs,
                ml_type=original_ml_type,
                target_metric=tpt._study_info.target_metric,
                target_col=tpt._study_info.target_col,
                target_assignments=original_target_assignments,
                positive_class=tpt._study_info.positive_class,
                row_id_col=tpt._study_info.row_id_col,
                feature_cols=tpt._study_info.feature_cols,
            )

    def test_predict_single_target_has_target_column(self, tpt):
        """Verify predict includes 'target' column for single-target tasks."""
        result = tpt.predict()
        assert "target" in result.columns

    def test_predict_proba_single_target_has_target_column(self, tpt):
        """Verify predict_proba includes 'target' column for single-target tasks."""
        result = tpt.predict_proba()
        assert "target" in result.columns


class TestBuildPoolData:
    """Test OctoPredictor._build_pool_data fallback behavior."""

    def test_keyerror_fallback(self, tp, study_path, monkeypatch):
        """Verify _build_pool_data falls back to data on KeyError."""
        import octopus.poststudy.predict.predictor as _pred_mod  # noqa: PLC0415

        monkeypatch.setattr(
            _pred_mod, "load_partition", lambda *a, **kw: (_ for _ in ()).throw(KeyError("missing row"))
        )
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        pool = tp._build_pool_data(data)
        for split_id in tp.study_info.outersplits:
            assert pool[split_id] is data

    def test_filenotfounderror_fallback(self, tp, study_path, monkeypatch):
        """Verify _build_pool_data falls back to data on FileNotFoundError."""
        import octopus.poststudy.predict.predictor as _pred_mod  # noqa: PLC0415

        monkeypatch.setattr(_pred_mod, "load_partition", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()))
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        pool = tp._build_pool_data(data)
        for split_id in tp.study_info.outersplits:
            assert pool[split_id] is data


class TestOctoPredictorPredict:
    """Test OctoPredictor predict and predict_proba."""

    def test_predict_returns_dataframe(self, tp, study_path):
        """Verify OctoPredictor predict returns wide-format DataFrame."""
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        result = tp.predict(data)
        assert isinstance(result, pd.DataFrame)
        assert "row_id" in result.columns
        assert "ensemble" in result.columns
        assert len(result) == len(data)
        split_cols = [c for c in result.columns if c.startswith("split_")]
        assert len(split_cols) >= 1

    def test_predict_proba_returns_dataframe(self, tp, study_path):
        """Verify OctoPredictor predict_proba returns wide-format DataFrame."""
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        result = tp.predict_proba(data)
        assert isinstance(result, pd.DataFrame)
        assert "row_id" in result.columns
        assert len(result) == len(data)
        non_row_id_cols = [c for c in result.columns if c != "row_id"]
        assert len(non_row_id_cols) >= 2


class TestOctoPredictorPerformance:
    """Test OctoPredictor.performance()."""

    def test_performance_returns_dataframe(self, tp, study_path):
        """Verify performance returns wide DataFrame with Mean and Ensemble."""
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        result = tp.performance(data, metrics=["ACCBAL", "ACC"])
        assert isinstance(result, pd.DataFrame)
        assert "Mean" in result.index
        assert "Ensemble" in result.index
        n_splits = tp.study_info.n_outersplits
        assert len(result) == n_splits + 2

    def test_performance_columns(self, tp, study_path):
        """Verify performance columns match requested metrics."""
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        result = tp.performance(data, metrics=["ACCBAL", "AUCROC"])
        assert list(result.columns) == ["ACCBAL", "AUCROC"]

    def test_performance_values_in_range(self, tp, study_path):
        """Verify classification scores are in [0, 1]."""
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        result = tp.performance(data, metrics=["ACCBAL"])
        assert (result["ACCBAL"] >= 0).all()
        assert (result["ACCBAL"] <= 1).all()


class TestOctoPredictorFI:
    """Test OctoPredictor.calculate_fi()."""

    def test_fi_permutation(self, tp, study_path):
        """Verify permutation FI returns DataFrame with per-split and ensemble rows."""
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        fi = tp.calculate_fi(data, fi_type=FIType.PERMUTATION, n_repeats=2)
        assert isinstance(fi, pd.DataFrame)
        assert "feature" in fi.columns
        assert "importance_mean" in fi.columns
        assert "fi_source" in fi.columns
        sources = fi["fi_source"].unique()
        assert "ensemble" in sources

    def test_fi_invalid_type(self, tp, study_path):
        """Verify invalid fi_type raises ValueError."""
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        with pytest.raises(ValueError, match="is not a valid FIType"):
            tp.calculate_fi(data, fi_type="invalid_method")


class TestOctoPredictorSaveLoad:
    """Test OctoPredictor save/load round-trip."""

    def test_roundtrip(self, tp, tmp_path):
        """Verify save/load preserves ml_type, n_outersplits, and feature_cols."""
        tp.save(tmp_path / "saved")
        loaded = OctoPredictor.load(tmp_path / "saved")
        assert loaded.study_info.ml_type == tp.study_info.ml_type
        assert loaded.study_info.n_outersplits == tp.study_info.n_outersplits
        assert loaded.feature_cols == tp.feature_cols

    def test_loaded_predicts(self, tp, study_path, tmp_path):
        """Verify a loaded OctoPredictor can still produce predictions."""
        tp.save(tmp_path / "saved2")
        loaded = OctoPredictor.load(tmp_path / "saved2")
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        result = loaded.predict(data)
        assert isinstance(result, pd.DataFrame)
        assert "ensemble" in result.columns
        assert len(result) == len(data)


class TestPredictProbaMLTypeGuard:
    """Test predict_proba raises TypeError for non-classification."""

    def test_mock_regression_guard(self, tp):
        """Temporarily override ml_type to test the guard."""
        original = tp._study_info.ml_type
        try:
            tp._study_info = StudyInfo(
                path=tp._study_info.path,
                n_outer_splits=tp._study_info.n_outer_splits,
                workflow_tasks=tp._study_info.workflow_tasks,
                outersplit_dirs=tp._study_info.outersplit_dirs,
                ml_type=MLType.REGRESSION,
                target_metric=tp._study_info.target_metric,
                target_col=tp._study_info.target_col,
                target_assignments=tp._study_info.target_assignments,
                positive_class=tp._study_info.positive_class,
                row_id_col=tp._study_info.row_id_col,
                feature_cols=tp._study_info.feature_cols,
            )
            data = pd.DataFrame({"f0": [1], "f1": [2], "f2": [3], "f3": [4], "f4": [5]})
            with pytest.raises(TypeError, match=r"predict_proba.*only available"):
                tp.predict_proba(data)
        finally:
            tp._study_info = StudyInfo(
                path=tp._study_info.path,
                n_outer_splits=tp._study_info.n_outer_splits,
                workflow_tasks=tp._study_info.workflow_tasks,
                outersplit_dirs=tp._study_info.outersplit_dirs,
                ml_type=original,
                target_metric=tp._study_info.target_metric,
                target_col=tp._study_info.target_col,
                target_assignments=tp._study_info.target_assignments,
                positive_class=tp._study_info.positive_class,
                row_id_col=tp._study_info.row_id_col,
                feature_cols=tp._study_info.feature_cols,
            )


class TestEnsembleRow:
    """Test that OctoPredictor.performance() Ensemble row differs from Mean."""

    def test_ensemble_row_present(self, tp, study_path):
        """Verify Ensemble row exists and has numeric values."""
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        result = tp.performance(data, metrics=["ACCBAL"])
        assert "Ensemble" in result.index
        assert result.loc["Ensemble", "ACCBAL"] > 0

    def test_ensemble_differs_from_mean(self, tp, study_path):
        """Verify Ensemble and Mean are computed differently."""
        data = parquet_load(f"{study_path}/data_prepared.parquet")
        result = tp.performance(data, metrics=["AUCROC"])
        assert "Mean" in result.index
        assert "Ensemble" in result.index


class TestLoadedPredictorAttributes:
    """R1: Verify loaded predictor has all expected attributes."""

    def test_all_base_attributes_present(self, tp, tmp_path):
        """Verify a loaded predictor has all _PredictorBase fields."""
        tp.save(tmp_path / "saved_r1")
        loaded = OctoPredictor.load(tmp_path / "saved_r1")
        expected_attrs = [
            "_study_info",
            "_task_id",
            "_result_type",
            "_models",
            "_feature_cols_per_split",
            "_feature_groups_per_split",
            "_feature_cols",
        ]
        for attr in expected_attrs:
            assert hasattr(loaded, attr), f"Loaded predictor missing attribute: {attr}"

    def test_loaded_properties_accessible(self, tp, tmp_path):
        """Verify public properties work on a loaded predictor."""
        tp.save(tmp_path / "saved_r1b")
        loaded = OctoPredictor.load(tmp_path / "saved_r1b")
        assert loaded.task_id == tp.task_id
        assert loaded.result_type == tp.result_type
        assert loaded.feature_cols == tp.feature_cols
        assert loaded.study_info.ml_type == tp.study_info.ml_type
