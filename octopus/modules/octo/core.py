# type: ignore

"""Octo execution module."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

import optuna
import optuna.storages.journal
import pandas as pd
from attrs import Factory, define, field
from optuna.trial import TrialState
from upath import UPath

from octopus.datasplit import DataSplit, InnerSplits
from octopus.logger import LogGroup, get_logger
from octopus.modules.base import MLModuleExecution, ResultType
from octopus.modules.mrmr.core import _maxrminr, _relevance_fstats
from octopus.modules.octo.bag import Bag
from octopus.modules.octo.enssel import EnSel
from octopus.modules.octo.objective_optuna import ObjectiveOptuna
from octopus.modules.octo.training import Training

from .optuna_storage_backend import JournalFsspecFileBackend

if TYPE_CHECKING:
    from octopus.modules.octo.module import Octo  # noqa: F401
    from octopus.study.context import StudyContext

logger = get_logger()


@define
class OctoModule(MLModuleExecution["Octo"]):
    """Octo execution module. Created by Octo.create_module()."""

    # Internal state (set during fit)
    data_splits_: InnerSplits = field(init=False, default=Factory(dict))
    """Data splits for inner CV."""

    paths_optuna_db_: dict[str, UPath] = field(init=False, default=Factory(dict))
    """Paths to Optuna databases."""

    top_trials_: list = field(init=False, default=Factory(list))
    """Top performing trials."""

    mrmr_features_: dict = field(init=False, default=Factory(dict))
    """MRMR feature sets for different feature counts."""

    # Temporary execution state (available during fit)
    _study: StudyContext | None = field(init=False, default=None)
    """StudyContext reference (temporary state during fit)."""

    _output_dir: Any = field(init=False, default=None)
    """Output directory (temporary state during fit)."""

    _feature_groups: dict = field(init=False, default=Factory(dict))
    """Feature groups (temporary state during fit)."""

    _data_traindev: Any = field(init=False, default=None)
    """Training/development data (temporary state during fit)."""

    _data_test: Any = field(init=False, default=None)
    """Test data (temporary state during fit)."""

    _feature_cols: list[str] | None = field(init=False, default=None)
    """Feature columns (temporary state during fit)."""

    _outersplit_id: int | None = field(init=False, default=None)
    """Fold ID (temporary state during fit)."""

    _num_assigned_cpus: int = field(init=False, default=1)
    """Number of assigned CPUs (temporary state during fit)."""

    _prior_results: dict = field(init=False, default=Factory(dict))
    """Prior results (temporary state during fit)."""

    def fit(
        self,
        data_traindev: pd.DataFrame,
        data_test: pd.DataFrame,
        feature_cols: list[str],
        study: StudyContext,
        outersplit_id: int,
        output_dir: UPath,
        num_assigned_cpus: int = 1,
        feature_groups: dict | None = None,
        prior_results: dict | None = None,
    ) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fit Octo module by running hyperparameter optimization with Optuna."""
        # Store execution state temporarily for internal methods
        self._study = study
        self._outersplit_id = outersplit_id
        self._output_dir = output_dir
        self._num_assigned_cpus = num_assigned_cpus
        self._data_traindev = data_traindev
        self._data_test = data_test
        self._feature_cols = feature_cols
        self._feature_groups = feature_groups or {}
        self._prior_results = prior_results or {}

        # Initialize Octo-specific setup
        self._initialize_octo()

        # Initialize local results collection
        results = {}
        selected_features = []

        # (1) model training and optimization
        best_selected_features = self._run_globalhp_optimization(results)
        selected_features = best_selected_features

        # (2) ensemble selection
        if self.config.ensemble_selection:
            ensel_selected_features = self._run_ensemble_selection(results)
            if ensel_selected_features:
                selected_features = ensel_selected_features

        # Store fitted state (permanent)
        self.selected_features_ = selected_features
        primary_key = "ensel" if "ensel" in results else "best"
        self.feature_importances_ = results[primary_key]["feature_importances"]

        # Build flat DataFrames from results
        best_bag = results["best"]["_bag"]
        scores_best = best_bag.get_performance_df(metric=self.target_metric)
        scores_best["result_type"] = ResultType.BEST

        predictions_best = best_bag.get_predictions_df()
        predictions_best["result_type"] = ResultType.BEST

        fi_best = best_bag.get_feature_importances_df()
        fi_best["result_type"] = ResultType.BEST

        if self.config.ensemble_selection and "ensel" in results:
            ensel_bag = results["ensel"]["_bag"]
            scores_ensel = ensel_bag.get_performance_df(metric=self.target_metric)
            scores_ensel["result_type"] = ResultType.ENSEMBLE_SELECTION

            predictions_ensel = ensel_bag.get_predictions_df()
            predictions_ensel["result_type"] = ResultType.ENSEMBLE_SELECTION

            fi_ensel = ensel_bag.get_feature_importances_df()
            fi_ensel["result_type"] = ResultType.ENSEMBLE_SELECTION

            scores = pd.concat([scores_best, scores_ensel], ignore_index=True)
            predictions = pd.concat([predictions_best, predictions_ensel], ignore_index=True)
            feature_importances = pd.concat([fi_best, fi_ensel], ignore_index=True)
        else:
            scores = scores_best
            predictions = predictions_best
            feature_importances = fi_best

        return (selected_features, scores, predictions, feature_importances)

    @property
    def path_module(self) -> UPath:
        """Module directory path (available during fit)."""
        return self._output_dir

    @property
    def path_trials(self) -> UPath:
        """Trials path (available during fit)."""
        return self.path_module / "trials"

    @property
    def path_results(self) -> UPath:
        """Results path (available during fit)."""
        return self.path_module / "results"

    @property
    def datasplit_column(self) -> str:
        """Column used for data splitting (available during fit)."""
        if self._study.datasplit_type == "sample":
            return self._study.sample_id_col
        return self._study.datasplit_type

    @property
    def stratification_col(self) -> str | None:
        """Stratification column (available during fit)."""
        return self._study.stratification_col

    @property
    def ml_type(self) -> str:
        """ML type (available during fit)."""
        return self._study.ml_type

    @property
    def target_metric(self) -> str:
        """Target metric (available during fit)."""
        return self._study.target_metric

    @property
    def target_assignments(self) -> dict:
        """Target column assignments (available during fit)."""
        return self._study.target_assignments

    @property
    def row_column(self) -> str:
        """Row ID column (available during fit)."""
        return self._study.row_id_col

    @property
    def positive_class(self):
        """Positive class (available during fit). None for regression."""
        return self._study.positive_class

    @property
    def x_traindev(self) -> pd.DataFrame:
        """Feature matrix (available during fit)."""
        return self._data_traindev[self._feature_cols]

    @property
    def y_traindev(self) -> pd.DataFrame:
        """Target values (available during fit)."""
        return self._data_traindev[list(self._study.target_assignments.values())]

    def _initialize_octo(self):
        """Initialize Octo-specific data structures and directories."""
        # create datasplit during init
        self.data_splits_ = DataSplit(
            dataset=self._data_traindev,
            datasplit_col=self.datasplit_column,
            seeds=self.config.datasplit_seeds_inner,
            num_folds=self.config.n_folds_inner,
            stratification_col=self.stratification_col,
            process_id=f"OUTER {self._outersplit_id} SEQ TBD",
        ).get_inner_splits()

        # if we don't want to resume optimization:
        # delete directories /trials /results to ensure clean state
        # of module when restarted, required for parallel optuna runs
        # as optuna.create(...,load_if_exists=True)
        for directory in [self.path_trials, self.path_results]:
            if not self.config.resume_optimization and directory.exists():
                directory.rmdir(recursive=True)
            directory.mkdir(parents=True, exist_ok=True)

        # check if there is a mismatch between configured resources
        # and resources assigned to the outersplit
        self._check_resources()

        # Create MRMR feature lists
        self._create_mrmr_features()

    def _create_mrmr_features(self):
        """Calculate feature lists for all provided features numbers."""
        logger.info("Calculating MRMR feature sets...")
        # remove duplicates and cap max number
        feature_numbers = list(set(self.config.mrmr_feature_numbers))
        feature_numbers = [x for x in feature_numbers if isinstance(x, int) and x <= len(self._feature_cols)]
        # if no mrmr features are requested, only add original features
        if not feature_numbers:
            # add original features
            self.mrmr_features_[len(self._feature_cols)] = self._feature_cols
            return

        # prepare inputs
        feature_cols = self._feature_cols
        features = self.x_traindev
        target = self.y_traindev

        # create relevance information
        re_df = _relevance_fstats(
            features=features,
            target=target,
            feature_cols=feature_cols,
            ml_type=self.ml_type,
        )

        # calculate MRMR features for all feature_numbers
        self.mrmr_features_ = _maxrminr(
            features=features,
            relevance=re_df,
            requested_feature_counts=feature_numbers,
            correlation_type="spearman",
        )
        # add original features
        self.mrmr_features_[len(self._feature_cols)] = self._feature_cols

    def _check_resources(self):
        """Check resources, assigned vs requested."""
        logger.set_log_group(LogGroup.PREPARE_EXECUTION, f"OUTER {self._outersplit_id} SQE TBD")

        if self.config.inner_parallelization is True:
            num_requested_cpus = self.config.n_workers * self.config.n_jobs
        else:
            num_requested_cpus = self.config.n_jobs
        logger.info(
            f"""CPU Resources | \
        Available: {self._num_assigned_cpus} | \
        Requested: {num_requested_cpus} | """
        )

    def _run_ensemble_selection(self, results: dict) -> list[str]:
        """Run ensemble selection."""
        ensel = EnSel(
            target_metric=self.target_metric,
            path_trials=self.path_trials,
            max_n_iterations=100,
            row_id_col=self.row_column,
            target_assignments=self.target_assignments,
            positive_class=self.positive_class,
        )
        ensemble_paths_dict = ensel.start_ensemble
        return self._create_ensemble_bag(ensemble_paths_dict, results)

    def _create_ensemble_bag(self, ensemble_paths_dict: dict, results: dict) -> list[str]:
        """Create ensemble bag from a ensemble path dict."""
        if len(ensemble_paths_dict) == 0:
            raise ValueError("Valid ensemble information need to be provided")

        # Compute outersplit_task_id from outersplit_id and task_id
        training_id = f"{self._outersplit_id}_{self.config.task_id}"

        # extract trainings
        trainings = []
        train_id = 0
        for path, weight in ensemble_paths_dict.items():
            bag = Bag.from_pickle(path)
            for training in bag.trainings:
                for _ in range(int(weight)):
                    train_cp = copy.deepcopy(training)
                    train_cp.training_id = training_id + "_" + str(train_id)
                    train_cp.training_weight = 1
                    train_id += 1
                    trainings.append(train_cp)

        # create ensemble bag
        ensel_bag = Bag(
            bag_id=training_id + "_ensel",
            trainings=trainings,
            train_status=True,
            target_assignments=self.target_assignments,
            parallel_execution=self.config.inner_parallelization,
            num_workers=self.config.n_workers,
            target_metric=self.target_metric,
            row_id_col=self.row_column,
            ml_type=self.ml_type,
            log_dir=self._study.log_dir,
        )
        # save ensel bag
        ensel_bag.to_pickle(self.path_results / "ensel_bag.pkl")

        # save performance values of best bag
        ensel_scores = ensel_bag.get_performance()
        target_metric = self.target_metric
        # show and save test results
        logger.set_log_group(LogGroup.RESULTS)
        logger.info("Ensemble selection performance")
        logger.info(
            f"Training: {training_id} "
            f"{target_metric} "
            f"(ensemble selection): "
            f"Dev {ensel_scores['dev_pool']:.3f}, "
            f"Test {ensel_scores['test_pool']:.3f}"
        )

        with (self.path_results / "ensel_scores_scores.json").open("w", encoding="utf-8") as f:
            json.dump(ensel_scores, f)

        # calculate feature importances of best bag
        fi_methods = []  # disable calculation of pfi for ensel_bag
        ensel_bag_fi = ensel_bag.calculate_feature_importances(fi_methods, partitions=["dev"])

        # calculate selected features
        selected_features = ensel_bag.get_selected_features(fi_methods)

        # save best bag and results to local dict
        results["ensel"] = {
            "scores": ensel_scores,
            "predictions": ensel_bag.get_predictions(),
            "feature_importances": ensel_bag_fi,
            "_bag": ensel_bag,
        }

        return selected_features

    def _run_globalhp_optimization(self, results: dict) -> list[str]:
        """Optimization run with a global HP set over all inner folds."""
        logger.info("Running Optuna Optimization with a global HP set")

        # Optimize splits.
        splits = self.data_splits_
        study_name = f"optuna_{self._outersplit_id}_{self.config.task_id}"
        outersplit_task_id = f"{self._outersplit_id}_{self.config.task_id}"
        task_path = f"outersplit{self._outersplit_id}/task{self.config.task_id}"

        # set up Optuna study
        objective = ObjectiveOptuna(
            outersplit_task_id=outersplit_task_id,
            outersplit_id=self._outersplit_id,
            ml_type=self.ml_type,
            target_assignments=self.target_assignments,
            feature_cols=self._feature_cols,
            row_column=self.row_column,
            data_test=self._data_test,
            target_metric=self.target_metric,
            feature_groups=self._feature_groups,
            positive_class=self.positive_class,
            config=self.config,
            path_study=self._study.output_path,
            task_path=task_path,
            data_splits=splits,
            study_name=study_name,
            top_trials=self.top_trials_,
            mrmr_features=self.mrmr_features_,
            log_dir=self._study.log_dir,
        )

        # multivariate sampler with group option
        sampler = optuna.samplers.TPESampler(
            multivariate=True,
            group=True,
            constant_liar=True,
            seed=self.config.optuna_seed,
            n_startup_trials=self.config.n_optuna_startup_trials,
        )

        # create study with unique name and database
        db_path = self.path_module / (study_name + "_optuna.log")
        storage = optuna.storages.JournalStorage(JournalFsspecFileBackend(db_path))
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",  # metric adjustment in optuna objective
            sampler=sampler,
            storage=storage,
            load_if_exists=True,
        )
        # store optuna db path
        self.paths_optuna_db_[study_name] = db_path

        def logging_callback(study, trial):
            logger.set_log_group(LogGroup.OPTUNA)
            if trial.state == TrialState.COMPLETE:
                logger.info(f"Trial {trial.number} finished with value: {trial.value} and parameters: {trial.params}")
            elif trial.state == TrialState.PRUNED:
                logger.info(f"Trial {trial.number} pruned.")
            elif trial.state == TrialState.FAIL:
                logger.error(f"Trial {trial.number} failed.")

        study.optimize(
            objective,
            n_jobs=1,
            n_trials=self.config.n_trials,
            callbacks=[logging_callback],
        )

        # save optuna results as parquet file
        dict_optuna = []
        for trial in study.get_trials():
            for name, _ in trial.distributions.items():
                if name == "ml_model_type":
                    continue
                if "ml_model_type" in trial.params:
                    model_type = trial.params["ml_model_type"]
                else:
                    model_type = trial.user_attrs["config_training"]["ml_model_type"]

                dict_optuna.append(
                    {
                        "outersplit_id": self._outersplit_id,
                        "task_id": self.config.task_id,
                        "trial": trial.number,
                        "value": trial.value,
                        "model_type": model_type,
                        "hyper_param": name.split(f"_{model_type}")[0],
                        "param_value": str(trial.params[name]),
                    }
                )
        dict_optuna_path = self.path_module / f"{study_name}_optuna_results.parquet"
        pd.DataFrame(dict_optuna).to_parquet(
            str(dict_optuna_path),
            storage_options=dict_optuna_path.storage_options,
            engine="pyarrow",
        )

        # display results
        logger.set_log_group(LogGroup.SCORES, f"OUTER {self._outersplit_id} SQE TBD")
        logger.info("Optimization results: ")
        logger.info(
            f"Best trial number {study.best_trial.number}",
        )
        logger.info(f"Best target value {study.best_value}")
        user_attrs = study.best_trial.user_attrs
        performance_info = {key: v for key, v in user_attrs.items() if key not in ["config_training"]}
        logger.info(f"Best parameters {user_attrs['config_training']}")
        logger.info(f"Performance Info: {performance_info}")

        logger.info("Create best bag.....")
        n_input_features = user_attrs["config_training"]["n_input_features"]
        best_bag_feature_cols = self.mrmr_features_[n_input_features]

        # Compute outersplit_task_id from outersplit_id and task_id
        training_id = f"{self._outersplit_id}_{self.config.task_id}"

        # create best bag from optuna info
        best_trainings = []
        for key, split in splits.items():
            best_trainings.append(
                Training(
                    training_id=training_id + "_" + str(key),
                    ml_type=self.ml_type,
                    target_assignments=self.target_assignments,
                    feature_cols=best_bag_feature_cols,
                    row_id_col=self.row_column,
                    data_train=split.train,  # inner datasplit, train
                    data_dev=split.dev,  # inner datasplit, dev
                    data_test=self._data_test,
                    config_training=user_attrs["config_training"],
                    target_metric=self.target_metric,
                    max_features=self.config.max_features,
                    feature_groups=self._feature_groups,
                )
            )
        # create bag with all provided trainings
        best_bag = Bag(
            bag_id=training_id + "_best",
            trainings=best_trainings,
            target_assignments=self.target_assignments,
            parallel_execution=self.config.inner_parallelization,
            num_workers=self.config.n_workers,
            target_metric=self.target_metric,
            row_id_col=self.row_column,
            ml_type=self.ml_type,
            log_dir=self._study.log_dir,
        )

        # train all models in best_bag
        best_bag.fit()

        # save best bag
        best_bag.to_pickle(self.path_results / "best_bag.pkl")

        # save performance values of best bag
        best_bag_performance = best_bag.get_performance()
        logger.info(f"Best bag performance {best_bag_performance}")
        target_metric = self.target_metric

        # show and save test results
        logger.set_log_group(LogGroup.RESULTS)
        logger.info(
            f"Training: {training_id} "
            f"{target_metric} "
            f"(best bag - ensembled): "
            f"Dev {best_bag_performance['dev_pool']:.3f}, "
            f"Test {best_bag_performance['test_pool']:.3f}"
        )

        # save best bag performance
        with (self.path_results / "best_bag_performance.json").open("w", encoding="utf-8") as f:
            json.dump(best_bag_performance, f)

        # calculate feature importances of best bag
        fi_methods = self.config.fi_methods_bestbag
        best_bag_fi = best_bag.calculate_feature_importances(fi_methods, partitions=["dev"])

        # calculate selected features
        selected_features = best_bag.get_selected_features(fi_methods)

        # save best bag and results to local dict
        results["best"] = {
            "scores": best_bag_performance,
            "predictions": best_bag.get_predictions(),
            "feature_importances": best_bag_fi,
            "_bag": best_bag,
        }

        # Store the best bag as the module's fitted model
        self.model_ = best_bag

        # log selected features info
        logger.set_log_group(LogGroup.RESULTS)
        logger.info("---")
        logger.info(f"Number of original features: {len(self._feature_cols)}")
        logger.info(f"Number of selected features: {len(selected_features)}")
        if len(selected_features) == 0:
            logger.warning("Best bag - all feature importances values are zero. This hints at a model related problem.")

        return selected_features
