import os
import stat
import tempfile

import pandas as pd
import pytest
from botocore.session import Session
from moto.moto_server.threaded_moto_server import ThreadedMotoServer
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from upath import UPath

from octopus.modules import Octo
from octopus.study import OctoClassification
from octopus.types import FIComputeMethod, ModelName
from octopus.utils import joblib_load, joblib_save


class TestFSSpecIntegration:
    """Test for ensuring that all file IO goes through fsspec."""

    def test_joblib_roundtrip_memory_filesystem(self):
        """Verify joblib_save / joblib_load roundtrip on memory:// filesystem."""
        model = LinearRegression()
        model.fit([[1, 2], [3, 4]], [1, 2])

        path = UPath("memory://test_joblib/model.joblib")
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib_save(model, path)
        loaded = joblib_load(path)

        assert type(loaded) is type(model)
        assert loaded.coef_.tolist() == model.coef_.tolist()

    @pytest.fixture
    def breast_cancer_dataset(self):
        """Create synthetic binary classification dataset for testing (faster than breast cancer dataset)."""
        # Create synthetic binary classification dataset with reduced size for faster testing
        X, y = make_classification(
            n_samples=30,
            n_features=5,
            n_informative=3,
            n_redundant=2,
            n_classes=2,
            random_state=42,
        )

        # Create DataFrame similar to breast cancer dataset structure
        feature_names = [f"feature_{i}" for i in range(5)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        df = df.reset_index()

        return df, feature_names

    @pytest.fixture(scope="module")
    def s3_base(self):
        """Fixture for a local S3 server using moto."""
        port = 5555
        endpoint_uri = f"http://127.0.0.1:{port}/"
        bucket_name = "test_bucket"

        server = ThreadedMotoServer(ip_address="127.0.0.1", port=port)
        server.start()
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "foo")
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "foo")
        os.environ.setdefault("FSSPEC_S3_ENDPOINT_URL", endpoint_uri)
        os.environ.setdefault("FSSPEC_S3_KEY", "foo")
        os.environ.setdefault("FSSPEC_S3_SECRET", "foo")
        os.environ.setdefault("FSSPEC_S3_USE_SSL", "False")

        # NB: we use the sync botocore client for setup.
        # Region MUST be us-east-1 so that CreateBucket works without
        # LocationConstraint (any other region requires it).
        session = Session()
        client = session.create_client("s3", region_name="us-east-1", endpoint_url=endpoint_uri)
        client.create_bucket(Bucket=bucket_name, ACL="public-read-write")

        print("server up")
        yield endpoint_uri, bucket_name
        print("moto done")
        server.stop()

    def test_joblib_roundtrip_s3_filesystem(self, s3_base):
        """Verify joblib_save / joblib_load roundtrip on moto-backed s3:// filesystem.

        Fast smoke test (~2 s) that validates the S3 I/O pathway works
        without running a full ML workflow.
        """
        endpoint_uri, bucket_name = s3_base

        model = LinearRegression()
        model.fit([[1, 2], [3, 4]], [1, 2])

        path = UPath(
            f"s3://{bucket_name}/test_joblib_s3/model.joblib",
            endpoint_url=endpoint_uri,
        )

        joblib_save(model, path)
        loaded = joblib_load(path)

        assert type(loaded) is type(model)
        assert loaded.coef_.tolist() == model.coef_.tolist()

    @pytest.mark.slow
    def test_fsspec_mocked_s3_support(self, breast_cancer_dataset, s3_base):
        """Test Amazon S3 support using a local S3 servervia the s3:// protocol of fsspec."""
        endpoint_uri, bucket_name = s3_base

        self.run_experiment(
            breast_cancer_dataset,
            root_dir=UPath(
                f"s3://{bucket_name}/",
                endpoint_url=endpoint_uri,
            ),
        )

    @pytest.mark.skipif(
        any(
            key not in os.environ
            for key in (
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN",
                "OCTOPUS_TEST_S3_BUCKET_NAME",
            )
        ),
        reason="Testing against a real S3 bucket needs AWS credentials.",
    )
    @pytest.mark.slow
    def test_fsspec_s3_support(self, breast_cancer_dataset):
        """Test Amazon S3 support via the s3:// protocol of fsspec.

        This test requires real AWS credentials and a real, existing S3 bucket.
        It will only run if the following environment variables are set:
        ```
        export AWS_ACCESS_KEY_ID="ASI...FXL"
        export AWS_SECRET_ACCESS_KEY="3N9...tM8"
        export AWS_SESSION_TOKEN="IQo...y3Z"
        export OCTOPUS_TEST_S3_BUCKET_NAME="mathias-mtpilot-dev"
        ```
        Credentials can be generated in the AWS console
        and should have write access to the specified bucket.

        The test will overwrite content inside the bucket and will
        not clean up after itself.
        """
        bucket_name = os.environ["OCTOPUS_TEST_S3_BUCKET_NAME"]

        self.run_experiment(
            breast_cancer_dataset,
            root_dir=UPath(
                f"s3://{bucket_name}/",
                # endpoint_url=endpoint_uri,
            ),
        )

    @pytest.mark.slow
    def test_fsspec_in_memory_support(self, breast_cancer_dataset):
        """Test for working memory file system support via the memory:// protocol of fsspec."""
        self.run_experiment(breast_cancer_dataset, root_dir=UPath("memory://test_dir/"))

    def run_experiment(self, breast_cancer_dataset, root_dir: UPath):
        """Run an example experiment with a specified study directory.

        Test that the Octopus intro classification workflow actually runs end-to-end, using a
        specific UPath / fsspec compatible directory.
        This is more or less a copy of test_octo_intro_classification_actual_execution().
        """
        df, features = breast_cancer_dataset

        with tempfile.TemporaryDirectory(delete=True) as tmpdir:
            old_dir = os.getcwd()
            os.chdir(tmpdir)
            # Disable all read/write access to the temp dir to ensure that
            # all file IO goes through fsspec and the specified root_dir.
            # Execute permission is needed to enter the dir.
            os.chmod(tmpdir, mode=stat.S_IXUSR | stat.S_IRUSR)

            try:
                study = OctoClassification(
                    study_name="test_octo_intro_execution",
                    target_metric="ACCBAL",
                    feature_cols=features,
                    target_col="target",
                    sample_id_col="index",
                    stratification_col="target",
                    outer_split_seed=1,
                    n_outer_splits=2,
                    studies_directory=root_dir,
                    single_outer_split=0,
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
                            n_trials=2,
                            ensemble_selection=False,
                        )
                    ],
                )

                study.fit(data=df)

                study_path = study.output_path

                # Verify that the study was created and files exist
                assert study_path.exists(), "Study directory should be created"

                assert (study_path / "study.log").exists(), "Study log file should exist"

                # Check for expected files (new architecture uses files, not directories)
                assert (study_path / "data_raw.parquet").exists(), "Data parquet file should exist"
                assert (study_path / "data_prepared.parquet").exists(), "Prepared data parquet file should exist"

                assert (study_path / "study_config.json").exists(), "Config JSON file should exist"
                assert (study_path / "study_meta.json").exists(), "Study meta JSON file should exist"
                assert (study_path / "outersplit0").exists(), "Experiment directory should exist"

                # Verify that the Octo step was executed by checking for workflow directories
                experiment_path = study_path / "outersplit0"
                workflow_dirs = [d for d in experiment_path.iterdir() if d.is_dir() and d.name.startswith("task")]

                assert len(workflow_dirs) >= 1, (
                    f"Should have at least 1 workflow directory, found: {[d.name for d in workflow_dirs]}"
                )

            finally:
                os.chdir(old_dir)
