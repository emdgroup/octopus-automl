import os
import stat
import tempfile

import pandas as pd
import pytest
from botocore.session import Session
from moto.moto_server.threaded_moto_server import ThreadedMotoServer
from sklearn.datasets import make_classification
from upath import UPath

from octopus import OctoClassification
from octopus.modules import Octo


class TestFSSpecIntegration:
    """Test for ensuring that all file IO goes through fsspec."""

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

        # NB: we use the sync botocore client for setup
        session = Session()
        client = session.create_client("s3", endpoint_url=endpoint_uri)
        client.create_bucket(Bucket=bucket_name, ACL="public-read-write")

        print("server up")
        yield endpoint_uri, bucket_name
        print("moto done")
        server.stop()

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
                    name="test_octo_intro_execution",
                    target_metric="ACCBAL",
                    feature_cols=features,
                    target="target",
                    sample_id="index",
                    stratification_column="target",
                    metrics=["AUCROC", "ACCBAL", "ACC", "LOGLOSS"],
                    datasplit_seed_outer=1234,
                    n_folds_outer=2,
                    path=root_dir,
                    ignore_data_health_warning=True,
                    outer_parallelization=False,
                    run_single_experiment_num=0,
                    workflow=[
                        Octo(
                            description="step_1_octo",
                            task_id=0,
                            depends_on_task=-1,
                            load_task=False,
                            n_folds_inner=3,
                            models=["ExtraTreesClassifier"],
                            model_seed=0,
                            n_jobs=1,
                            max_outl=0,
                            fi_methods_bestbag=["permutation"],
                            inner_parallelization=True,
                            n_workers=2,
                            optuna_seed=0,
                            n_optuna_startup_trials=3,
                            resume_optimization=False,
                            n_trials=5,
                            max_features=5,
                            penalty_factor=1.0,
                            ensemble_selection=True,
                            ensel_n_save_trials=5,
                        )
                    ],
                )

                study.fit(data=df)

                study_path = root_dir / study.name

                # Verify that the study was created and files exist
                assert study_path.exists(), "Study directory should be created"

                assert (study_path / "octo_manager.log").exists(), "Octo Manager log file should exist"

                # Check for expected files (new architecture uses files, not directories)
                assert (study_path / "data.parquet").exists(), "Data parquet file should exist"
                assert (study_path / "data_prepared.parquet").exists(), "Prepared data parquet file should exist"

                assert (study_path / "config.json").exists(), "Config JSON file should exist"
                assert (study_path / "outersplit0").exists(), "Experiment directory should exist"

                # Verify that the Octo step was executed by checking for workflow directories
                experiment_path = study_path / "outersplit0"
                workflow_dirs = [
                    d for d in experiment_path.iterdir() if d.is_dir() and d.name.startswith("workflowtask")
                ]

                assert len(workflow_dirs) >= 1, (
                    f"Should have at least 1 workflow directory, found: {[d.name for d in workflow_dirs]}"
                )

            finally:
                os.chdir(old_dir)
