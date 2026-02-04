# type: ignore

"""OctoFull core function."""

import copy
import json

import optuna
import optuna.storages.journal
import pandas as pd
from attrs import Factory, define, field, validators
from optuna.trial import TrialState
from upath import UPath

from octopus.logger import LogGroup, get_logger
from octopus.modules.base import ModuleBaseCore
from octopus.modules.mrmr.core import maxrminr, relevance_fstats
from octopus.modules.octo.bag import Bag
from octopus.modules.octo.enssel import EnSel
from octopus.modules.octo.module import Octo
from octopus.modules.octo.objective_optuna import ObjectiveOptuna
from octopus.modules.octo.training import Training
from octopus.results import ModuleResults
from octopus.utils import DataSplit

from .optuna_storage_backend import JournalFsspecFileBackend

logger = get_logger()


@define
class OctoCoreGeneric[TaskConfigType: Octo](ModuleBaseCore[TaskConfigType]):
    """Manages and executes machine learning experiments.

    This class integrates all components necessary for conducting
    experiments using OctoExperiment configurations.
    It supports operations such as data splitting, path management,
    model optimization with Optuna, and results handling.
    The class is designed to work seamlessly with the defined experiment
    configurations and ensures robust handling of experiment resources,
    directories, and optimization processes.

    Attributes:
        data_splits: Stores training and validation data splits.
        paths_optuna_db: File paths to Optuna databases for each experiment.
        top_trials: Keeps track of the best performing trials.
        mrmr_features: Feature lists created by MRMR for different feature counts.

    Inherits from ModuleBaseCore:
        experiment: The OctoExperiment instance (from ModuleBaseCore).
        log_dir: Directory for individual worker logs (from ModuleBaseCore).
        All common properties from ModuleBaseCore (paths, data, metadata).

    Raises:
        ValueError: When encountering invalid operations or unsupported
            configurations during experiment execution.

    Usage:
        Initialize with an OctoExperiment object and call run_experiment()
        to conduct comprehensive machine learning optimization with Optuna.
    """

    # model = field(default=None)
    data_splits: dict = field(default=Factory(dict), validator=[validators.instance_of(dict)])

    paths_optuna_db: dict[str, UPath] = field(default=Factory(dict), validator=[validators.instance_of(dict)])

    top_trials: list = field(default=Factory(list), validator=[validators.instance_of(list)])

    mrmr_features: dict = field(default=Factory(dict), validator=[validators.instance_of(dict)])

    @property
    def path_trials(self) -> UPath:
        """Trials path."""
        return self.path_module / "trials"

    def __attrs_post_init__(self):
        """Initialize OctoCoreGeneric with data splits and configuration.

        Note: Does NOT call super().__attrs_post_init__() because this has
        custom directory management (handles path_trials) that differs from
        the base class.
        """
        # create datasplit during init
        self.data_splits = DataSplit(
            dataset=self.experiment.data_traindev,
            datasplit_col=self.experiment.datasplit_column,
            seeds=self.experiment.ml_config.datasplit_seeds_inner,
            num_folds=self.experiment.ml_config.n_folds_inner,
            stratification_col=self.experiment.stratification_col,
            process_id=f"EXP {self.experiment.experiment_id} SEQ TBD",
        ).get_datasplits()
        # if we don't want to resume optimization:
        # delete directories /trials /optuna /results to ensure clean state
        # of module when restarted, required for parallel optuna runs
        # as optuna.create(...,load_if_exists=True)
        # create directory if it does not exist
        for directory in [self.path_trials, self.path_results]:
            if not self.experiment.ml_config.resume_optimization and directory.exists():
                directory.rmdir(recursive=True)
            directory.mkdir(parents=True, exist_ok=True)

        # check if there is a mismatch between configured resources
        # and resources assigned to the experiment
        self._check_resources()

        # Create MRMR feature lists
        self._create_mrmr_features()

    def _create_mrmr_features(self):
        """Calculate feature lists for all provided features numbers."""
        logger.info("Calculating MRMR feature sets...")
        # remove duplicates and cap max number
        feature_numbers = list(set(self.experiment.ml_config.mrmr_feature_numbers))
        feature_numbers = [x for x in feature_numbers if isinstance(x, int) and x <= len(self.experiment.feature_cols)]
        # if no mrmr features are requested, only add original features
        if not feature_numbers:
            # add original features
            self.mrmr_features[len(self.experiment.feature_cols)] = self.experiment.feature_cols
            return

        # prepare inputs
        feature_cols = self.experiment.feature_cols
        features = self.experiment.x_traindev
        target = self.experiment.y_traindev

        # create relevance information
        re_df = relevance_fstats(
            features=features,
            target=target,
            feature_cols=feature_cols,
            ml_type=self.experiment.ml_type,
        )

        # calculate MRMR features for all feature_numbers
        self.mrmr_features = maxrminr(
            features=features,
            relevance=re_df,
            requested_feature_counts=feature_numbers,
            correlation_type="spearman",
        )
        # add original features
        self.mrmr_features[len(self.experiment.feature_cols)] = self.experiment.feature_cols

    def _check_resources(self):
        """Check resources, assigned vs requested."""
        logger.set_log_group(LogGroup.PREPARE_EXECUTION, f"EXP {self.experiment.experiment_id} SQE TBD")

        if self.experiment.ml_config.inner_parallelization is True:
            num_requested_cpus = self.experiment.ml_config.n_workers * self.experiment.ml_config.n_jobs
        else:
            num_requested_cpus = self.experiment.ml_config.n_jobs
        logger.info(
            f"""CPU Resources | \
        Available: {self.experiment.num_assigned_cpus} | \
        Requested: {num_requested_cpus} | """
        )

    def run_experiment(self):
        """Run experiment."""
        # (1) model training and optimization
        self._run_globalhp_optimization()

        # (2) ensemble selection
        if self.experiment.ml_config.ensemble_selection:
            self._run_ensemble_selection()

        return self.experiment

    def _run_ensemble_selection(self):
        """Run ensemble selection."""
        ensel = EnSel(
            target_metric=self.experiment.target_metric,
            path_trials=self.path_trials,
            max_n_iterations=100,
            row_column=self.experiment.row_column,
            target_assignments=self.experiment.target_assignments,
            positive_class=self.experiment.positive_class,
        )
        ensemble_paths_dict = ensel.start_ensemble
        # ensemble_paths_dict = ensel.optimized_ensemble
        self._create_ensemble_bag(ensemble_paths_dict)

    def _create_ensemble_bag(self, ensemble_paths_dict):
        """Create ensemble bag from a ensemble path dict."""
        if len(ensemble_paths_dict) == 0:
            raise ValueError("Valid ensemble information need to be provided")

        # extract trainings
        # here, we don't use the weight info
        # this requires more work for scores and feature importances
        trainings = []
        train_id = 0
        for path, weight in ensemble_paths_dict.items():
            bag = Bag.from_pickle(path)
            for training in bag.trainings:
                # training.training_weight - tobedone
                for _ in range(int(weight)):
                    train_cp = copy.deepcopy(training)
                    train_cp.training_id = self.experiment.id + "_" + str(train_id)
                    train_cp.training_weight = 1
                    train_id += 1
                    trainings.append(train_cp)

        # create ensemble bag
        ensel_bag = Bag(
            bag_id=self.experiment.id + "_ensel",
            trainings=trainings,
            train_status=True,
            target_assignments=self.experiment.target_assignments,
            parallel_execution=self.experiment.ml_config.inner_parallelization,
            num_workers=self.experiment.ml_config.n_workers,
            target_metric=self.experiment.target_metric,
            row_column=self.experiment.row_column,
            ml_type=self.experiment.ml_type,
            log_dir=self.log_dir,
        )
        # save ensel bag
        ensel_bag.to_pickle(self.path_results / "ensel_bag.pkl")

        # save performance values of best bag
        ensel_scores = ensel_bag.get_performance()
        target_metric = self.experiment.target_metric
        # show and save test results
        logger.set_log_group(LogGroup.RESULTS)
        logger.info("Ensemble selection performance")
        logger.info(
            f"Experiment: {self.experiment.id} "
            f"{target_metric} "
            f"(ensemble selection): "
            f"Dev {ensel_scores['dev_pool']:.3f}, "
            f"Test {ensel_scores['test_pool']:.3f}"
        )

        with (self.path_results / "ensel_scores_scores.json").open("w", encoding="utf-8") as f:
            json.dump(ensel_scores, f)

        # calculate feature importances of best bag
        # fi_methods = self.experiment.ml_config.fi_methods_bestbag
        fi_methods = []  # disable calculation of pfi for ensel_bag
        ensel_bag_fi = ensel_bag.calculate_feature_importances(fi_methods, partitions=["dev"])

        # calculate selected features
        selected_features = ensel_bag.get_selected_features(fi_methods)

        # save best bag and results to experiment
        self.experiment.results["ensel"] = ModuleResults(
            id="ensel",
            experiment_id=self.experiment.experiment_id,
            task_id=self.experiment.task_id,
            model=ensel_bag,
            scores=ensel_scores,
            feature_importances=ensel_bag_fi,
            predictions=ensel_bag.get_predictions(),
            selected_features=selected_features,
        )

    def _run_globalhp_optimization(self):
        """Optimization run with a global HP set over all inner folds."""
        logger.info("Running Optuna Optimization with a global HP set")

        # Optimize splits.
        splits = self.data_splits
        study_name = f"optuna_{self.experiment.experiment_id}_{self.experiment.task_id}"

        # set up Optuna study
        objective = ObjectiveOptuna(
            experiment=self.experiment,
            data_splits=splits,
            study_name=study_name,
            top_trials=self.top_trials,
            mrmr_features=self.mrmr_features,
            log_dir=self.log_dir,
        )

        # multivariate sampler with group option
        sampler = optuna.samplers.TPESampler(
            multivariate=True,
            group=True,
            constant_liar=True,
            seed=self.experiment.ml_config.optuna_seed,
            n_startup_trials=self.experiment.ml_config.n_optuna_startup_trials,
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
        self.paths_optuna_db[study_name] = db_path

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
            n_trials=self.experiment.ml_config.n_trials,
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
                        "experiment_id": self.experiment.experiment_id,
                        "task_id": self.experiment.task_id,
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
        logger.set_log_group(LogGroup.SCORES, f"EXP {self.experiment.experiment_id} SQE TBD")
        logger.info("Optimization results: ")
        # print("Best trial:", study.best_trial) #full info
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
        best_bag_feature_cols = self.mrmr_features[n_input_features]

        # create best bag from optuna info
        best_trainings = []
        for key, split in splits.items():
            best_trainings.append(
                Training(
                    training_id=self.experiment.id + "_" + str(key),
                    ml_type=self.experiment.ml_type,
                    target_assignments=self.experiment.target_assignments,
                    feature_cols=best_bag_feature_cols,
                    row_column=self.experiment.row_column,
                    data_train=split["train"],  # inner datasplit, train
                    data_dev=split["test"],  # inner datasplit, dev
                    data_test=self.experiment.data_test,
                    config_training=user_attrs["config_training"],
                    target_metric=self.experiment.target_metric,
                    max_features=self.experiment.ml_config.max_features,
                    feature_groups=self.experiment.feature_groups,
                )
            )
        # create bag with all provided trainings
        best_bag = Bag(
            bag_id=self.experiment.id + "_best",
            trainings=best_trainings,
            target_assignments=self.experiment.target_assignments,
            parallel_execution=self.experiment.ml_config.inner_parallelization,
            num_workers=self.experiment.ml_config.n_workers,
            target_metric=self.experiment.target_metric,
            row_column=self.experiment.row_column,
            ml_type=self.experiment.ml_type,
            log_dir=self.log_dir,
            # path?
        )

        # train all models in best_bag
        best_bag.fit()

        # save best bag
        best_bag.to_pickle(self.path_results / "best_bag.pkl")

        # save performance values of best bag
        best_bag_performance = best_bag.get_performance()
        logger.info(f"Best bag performance {best_bag_performance}")
        target_metric = self.experiment.target_metric

        # show and save test results
        logger.set_log_group(LogGroup.RESULTS)
        logger.info(
            f"Experiment: {self.experiment.id} "
            f"{target_metric} "
            f"(best bag - ensembled): "
            f"Dev {best_bag_performance['dev_pool']:.3f}, "
            f"Test {best_bag_performance['test_pool']:.3f}"
        )

        # save best bag performance
        with (self.path_results / "best_bag_performance.json").open("w", encoding="utf-8") as f:
            json.dump(best_bag_performance, f)

        # calculate feature importances of best bag
        fi_methods = self.experiment.ml_config.fi_methods_bestbag
        best_bag_fi = best_bag.calculate_feature_importances(fi_methods, partitions=["dev"])

        # calculate selected features
        selected_features = best_bag.get_selected_features(fi_methods)

        # save best bag and results to experiment
        self.experiment.results["best"] = ModuleResults(
            id="best",
            experiment_id=self.experiment.experiment_id,
            task_id=self.experiment.task_id,
            model=best_bag,
            scores=best_bag_performance,
            feature_importances=best_bag_fi,
            selected_features=selected_features,
            predictions=best_bag.get_predictions(),
        )

        # save selected features to experiment
        logger.set_log_group(LogGroup.RESULTS)
        logger.info("---")
        logger.info(f"Number of original features: {len(self.experiment.feature_cols)}")
        self.experiment.selected_features = selected_features
        logger.info(f"Number of selected features: {len(self.experiment.selected_features)}")
        if len(self.experiment.selected_features) == 0:
            logger.warning("Best bag - all feature importances values are zero. This hints at a model related problem.")

        return True


class OctoCore(OctoCoreGeneric[Octo]):
    """Octo Core."""

    pass
