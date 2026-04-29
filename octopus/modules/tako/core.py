"""Tako execution module."""

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING

import optuna
import pandas as pd
from attrs import Factory, define, field
from optuna.trial import TrialState
from upath import UPath

from octopus.datasplit import DataSplit, InnerSplits, validate_class_coverage
from octopus.logger import get_logger
from octopus.metrics import Metrics
from octopus.models import Models
from octopus.modules import ModuleExecution, ModuleResult
from octopus.modules.mrmr.core import _maxrminr, _relevance_fstats
from octopus.types import (
    CorrelationType,
    DataPartition,
    FIComputeMethod,
    LogGroup,
    MetricDirection,
    MLType,
    PerformanceKey,
    ResultType,
)
from octopus.utils import joblib_load, parquet_save

from .bag import Bag, BagBase
from .enssel import EnSel
from .objective_optuna import ObjectiveOptuna
from .training import Training

if TYPE_CHECKING:
    from octopus.modules import StudyContext, Tako

logger = get_logger()


@define
class TakoModuleTemplate[T: Tako](ModuleExecution[T]):
    """Tako execution module. Created by Tako.create_module()."""

    # Internal state (set during fit)
    data_splits_: InnerSplits = field(init=False, default=Factory(dict))
    """Data splits for inner CV."""

    top_trials_: list = field(init=False, default=Factory(list))
    """Top performing trials."""

    mrmr_features_: dict = field(init=False, default=Factory(dict))
    """MRMR feature sets for different feature counts."""

    def fit(
        self,
        *,
        data_traindev: pd.DataFrame,
        data_test: pd.DataFrame,
        feature_cols: list[str],
        study_context: StudyContext,
        outer_split_id: int,
        results_dir: UPath,
        scratch_dir: UPath,
        n_assigned_cpus: int,
        feature_groups: dict[str, list[str]],
        **kwargs,
    ) -> dict[ResultType, ModuleResult]:
        """Fit Tako module by running hyperparameter optimization with Optuna."""
        x_traindev = data_traindev[feature_cols]
        y_traindev = data_traindev[list(study_context.target_assignments.values())]

        # Initialize Tako-specific setup
        self._initialize_tako(
            study_context,
            data_traindev,
            x_traindev,
            y_traindev,
            feature_cols,
            outer_split_id,
            scratch_dir,
            results_dir,
        )

        # Initialize local results collection
        results: dict[str, dict] = {}

        # (1) model training and optimization
        best_selected_features = self._run_globalhp_optimization(
            study_context=study_context,
            data_test=data_test,
            feature_cols=feature_cols,
            feature_groups=feature_groups,
            outer_split_id=outer_split_id,
            n_assigned_cpus=n_assigned_cpus,
            results_dir=results_dir,
            results=results,
            scratch_dir=scratch_dir,
        )

        # Build best ModuleResult
        best_bag: BagBase = results["best"]["_bag"]
        all_metrics = Metrics.get_by_type(study_context.ml_type)
        best_scores = pd.concat(
            [best_bag.get_performance_df(metric=m, n_assigned_cpus=n_assigned_cpus) for m in all_metrics],
            ignore_index=True,
        )
        best_result = ModuleResult(
            result_type=ResultType.BEST,
            module=self.config.module,
            selected_features=best_selected_features,
            scores=best_scores,
            predictions=best_bag.get_predictions_df(n_assigned_cpus=n_assigned_cpus),
            fi=best_bag.get_fi_df(),
            model=best_bag,
        )

        module_results: dict[ResultType, ModuleResult] = {ResultType.BEST: best_result}

        # (2) ensemble selection
        if self.config.ensemble_selection:
            ensel_selected_features = self._run_ensemble_selection(
                study_context=study_context,
                outer_split_id=outer_split_id,
                n_assigned_cpus=n_assigned_cpus,
                scratch_dir=scratch_dir,
                results=results,
            )

            # Always save ensemble result if it was produced
            if "ensel" in results:
                ensel_bag: BagBase = results["ensel"]["_bag"]
                ensel_scores = pd.concat(
                    [ensel_bag.get_performance_df(metric=m, n_assigned_cpus=n_assigned_cpus) for m in all_metrics],
                    ignore_index=True,
                )
                ensel_result = ModuleResult(
                    result_type=ResultType.ENSEMBLE_SELECTION,
                    module=self.config.module,
                    selected_features=ensel_selected_features or best_selected_features,
                    scores=ensel_scores,
                    predictions=ensel_bag.get_predictions_df(n_assigned_cpus=n_assigned_cpus),
                    fi=ensel_bag.get_fi_df(),
                    model=ensel_bag,
                )
                module_results[ResultType.ENSEMBLE_SELECTION] = ensel_result

        return module_results

    def _initialize_tako(
        self,
        study_context: StudyContext,
        data_traindev: pd.DataFrame,
        x_traindev: pd.DataFrame,
        y_traindev: pd.DataFrame,
        feature_cols: list[str],
        outer_split_id: int,
        scratch_dir: UPath,
        results_dir: UPath,
    ):
        """Initialize Tako-specific data structures and directories."""
        # Resolve default models if none were specified
        if self.config.models is None:
            self.config.models = Models.get_defaults(study_context.ml_type)
            logger.debug(f"Using default models for {study_context.ml_type.value}: {self.config.models}")

            # Filter out chpo-incompatible models when constrained HPO is active
            if self.config.max_features > 0:
                self.config.models = [m for m in self.config.models if Models.get_config(m).chpo_compatible]
                if not self.config.models:
                    raise ValueError(
                        "No default models are compatible with constrained HPO (max_features > 0). Specify models explicitly."
                    )

        # Validate model-task compatibility
        for model_name in self.config.models:
            Models.validate_model_compatibility(model_name, study_context.ml_type)

        # create datasplit during init
        self.data_splits_ = DataSplit(
            dataset=data_traindev,
            seeds=self.config.inner_split_seeds,
            n_splits=self.config.n_inner_splits,
            stratification_col=study_context.stratification_col,
            process_id=f"OUTER {outer_split_id} SEQ TBD",
        ).get_inner_splits()

        if study_context.ml_type in (MLType.BINARY, MLType.MULTICLASS):
            validate_class_coverage(self.data_splits_, list(study_context.target_assignments.values())[0])
        elif study_context.ml_type == MLType.TIMETOEVENT:
            validate_class_coverage(self.data_splits_, study_context.target_assignments["event"])

        logger.set_log_group(LogGroup.PREPARE_EXECUTION, f"OUTER {outer_split_id} SQE TBD")

        # Create MRMR feature lists
        self._create_mrmr_features(study_context, x_traindev, y_traindev, feature_cols)

    def _create_mrmr_features(
        self,
        study_context: StudyContext,
        x_traindev: pd.DataFrame,
        y_traindev: pd.DataFrame,
        feature_cols: list[str],
    ):
        """Calculate feature lists for all provided features numbers."""
        logger.info("Calculating MRMR feature sets...")
        # remove duplicates and cap max number
        feature_numbers = list(set(self.config.n_mrmr_features))
        feature_numbers = [x for x in feature_numbers if isinstance(x, int) and x <= len(feature_cols)]
        # if no mrmr features are requested, only add original features
        if not feature_numbers:
            # add original features
            self.mrmr_features_[len(feature_cols)] = feature_cols
            return

        # create relevance information
        re_df = _relevance_fstats(
            features=x_traindev,
            target=y_traindev,
            feature_cols=feature_cols,
            ml_type=study_context.ml_type,
        )

        # calculate MRMR features for all feature_numbers
        self.mrmr_features_ = _maxrminr(
            df_features=x_traindev,
            df_relevance=re_df,
            n_features_list=feature_numbers,
            correlation_type=CorrelationType.SPEARMAN,
        )
        # add original features
        self.mrmr_features_[len(feature_cols)] = feature_cols

    def _run_ensemble_selection(
        self, study_context: StudyContext, outer_split_id: int, n_assigned_cpus: int, scratch_dir: UPath, results: dict
    ) -> list[str]:
        """Run ensemble selection."""
        ensel = EnSel(
            target_metric=study_context.target_metric,
            path_trials=scratch_dir,
            max_n_iterations=25,
            row_id_col=study_context.row_id_col,
            target_assignments=study_context.target_assignments,
            positive_class=study_context.positive_class,
            n_assigned_cpus=n_assigned_cpus,
            ml_type=study_context.ml_type,
        )
        ensemble_paths_dict = ensel.optimized_ensemble
        return self._create_ensemble_bag(study_context, outer_split_id, n_assigned_cpus, ensemble_paths_dict, results)

    def _create_ensemble_bag(
        self,
        study_context: StudyContext,
        outer_split_id: int,
        n_assigned_cpus: int,
        ensemble_paths_dict: dict,
        results: dict,
    ) -> list[str]:
        """Create ensemble bag from a ensemble path dict."""
        if len(ensemble_paths_dict) == 0:
            raise ValueError("Valid ensemble information need to be provided")

        # Compute outer_split_task_id from outer_split_id and task_id
        training_id = f"{outer_split_id}_{self.config.task_id}"

        # extract trainings
        trainings = []
        train_id = 0
        for path, weight in ensemble_paths_dict.items():
            bag = joblib_load(path)
            for training in bag.trainings:
                for _ in range(int(weight)):
                    train_cp = copy.deepcopy(training)
                    train_cp.training_id = training_id + "_" + str(train_id)
                    train_cp.training_weight = 1
                    # Update inner_split_id in predictions to match new training position
                    for part in train_cp.predictions:
                        if isinstance(train_cp.predictions[part], pd.DataFrame):
                            train_cp.predictions[part] = train_cp.predictions[part].copy()
                            train_cp.predictions[part]["inner_split_id"] = str(train_id)
                    train_id += 1
                    trainings.append(train_cp)

        # create ensemble bag
        ensel_bag = Bag(
            bag_id=training_id + "_ensel",
            trainings=trainings,
            train_status=True,
            target_assignments=study_context.target_assignments,
            target_metric=study_context.target_metric,
            row_id_col=study_context.row_id_col,
            ml_type=study_context.ml_type,
            log_dir=study_context.log_dir,
        )
        # save performance values of best bag
        ensel_scores = ensel_bag.get_performance(n_assigned_cpus=n_assigned_cpus)
        target_metric = study_context.target_metric
        # show and save test results
        logger.set_log_group(LogGroup.RESULTS)
        logger.info("Ensemble selection performance")
        logger.info(
            f"Training: {training_id} {target_metric} (ensemble selection): Dev {ensel_scores[PerformanceKey.DEV_ENSEMBLE]:.3f}, Test {ensel_scores[PerformanceKey.TEST_ENSEMBLE]:.3f}"
        )

        # calculate feature importances of best bag
        fi_methods: list[FIComputeMethod] = []  # disable calculation of pfi for ensel_bag
        ensel_bag_fi = ensel_bag.calculate_fi(
            fi_methods=fi_methods, partitions=[DataPartition.DEV], n_assigned_cpus=n_assigned_cpus
        )

        # calculate selected features
        selected_features = ensel_bag.get_selected_features(fi_methods=fi_methods)

        # save best bag and results to local dict
        results["ensel"] = {
            "scores": ensel_scores,
            "predictions": ensel_bag.get_predictions(n_assigned_cpus=n_assigned_cpus),
            "fi": ensel_bag_fi,
            "_bag": ensel_bag,
        }

        return selected_features

    def _run_globalhp_optimization(
        self,
        study_context: StudyContext,
        data_test: pd.DataFrame,
        feature_cols: list[str],
        feature_groups: dict,
        outer_split_id: int,
        n_assigned_cpus: int,
        results_dir: UPath,
        results: dict[str, dict],
        scratch_dir: UPath,
    ) -> list[str]:
        """Optimization run with a global HP set over all inner splits."""
        logger.info("Running Optuna Optimization with a global HP set")

        # Optimize splits.
        splits = self.data_splits_
        study_name = f"optuna_{outer_split_id}_{self.config.task_id}"
        outer_split_task_id = f"{outer_split_id}_{self.config.task_id}"

        # Warn when the penalty_factor may not match the metric's scale
        if self.config.max_features > 0 and study_context.target_metric in {"MAE", "MSE", "RMSE"}:
            logger.warning(
                "penalty_factor=%.2f is used with metric '%s' whose values are not "
                "in the 0..1 range. Make sure penalty_factor is scaled to the "
                "metric's magnitude for the feature constraint to be effective.",
                self.config.penalty_factor,
                study_context.target_metric,
            )

        # set up Optuna study
        objective = ObjectiveOptuna(
            outer_split_task_id=outer_split_task_id,
            outer_split_id=outer_split_id,
            ml_type=study_context.ml_type,
            target_assignments=study_context.target_assignments,
            feature_cols=feature_cols,
            row_column=study_context.row_id_col,
            data_test=data_test,
            target_metric=study_context.target_metric,
            feature_groups=feature_groups,
            positive_class=study_context.positive_class,
            config=self.config,
            scratch_dir=scratch_dir,
            data_splits=splits,
            study_name=study_name,
            top_trials=self.top_trials_,
            mrmr_features=self.mrmr_features_,
            log_dir=study_context.log_dir,
            n_assigned_cpus=n_assigned_cpus,
        )

        # multivariate sampler with group option
        # Suppress ExperimentalWarning — these parameters have been stable for 3+ years
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
            sampler = optuna.samplers.TPESampler(
                multivariate=True,
                group=True,
                constant_liar=True,
                seed=0,
                n_startup_trials=self.config.n_startup_trials,
            )

        # create study with in-memory storage
        study = optuna.create_study(
            study_name=study_name,
            direction=MetricDirection.MAXIMIZE,  # metric adjustment in optuna objective
            sampler=sampler,
        )

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
                        "outer_split_id": outer_split_id,
                        "task_id": self.config.task_id,
                        "trial": trial.number,
                        "value": trial.value,
                        "model_type": model_type,
                        "hyper_param": name.split(f"_{model_type}")[0],
                        "param_value": str(trial.params[name]),
                    }
                )
        dict_optuna_path = results_dir / "optuna_results.parquet"
        parquet_save(pd.DataFrame(dict_optuna), dict_optuna_path)

        # display results
        logger.set_log_group(LogGroup.SCORES, f"OUTER {outer_split_id} SQE TBD")
        logger.info("Optimization results: ")
        logger.info(f"Best trial number {study.best_trial.number}")
        logger.info(f"Best target value {study.best_value}")
        user_attrs = study.best_trial.user_attrs
        performance_info = {key: v for key, v in user_attrs.items() if key not in ["config_training"]}
        logger.info(f"Best parameters {user_attrs['config_training']}")
        logger.info(f"Performance Info: {performance_info}")

        logger.info("Create best bag.....")
        n_input_features = user_attrs["config_training"]["n_input_features"]
        best_bag_feature_cols = self.mrmr_features_[n_input_features]

        # Compute outer_split_task_id from outer_split_id and task_id
        training_id = f"{outer_split_id}_{self.config.task_id}"

        # create best bag from optuna info
        best_trainings = []
        for key, split in splits.items():
            best_trainings.append(
                Training(
                    training_id=training_id + "_" + str(key),
                    ml_type=study_context.ml_type,
                    target_assignments=study_context.target_assignments,
                    feature_cols=best_bag_feature_cols,
                    row_id_col=study_context.row_id_col,
                    data_train=split.train,  # inner datasplit, train
                    data_dev=split.dev,  # inner datasplit, dev
                    data_test=data_test,
                    config_training=user_attrs["config_training"],
                    target_metric=study_context.target_metric,
                    max_features=self.config.max_features,
                    feature_groups=feature_groups,
                )
            )
        # create bag with all provided trainings
        best_bag = Bag(
            bag_id=training_id + "_best",
            trainings=best_trainings,
            target_assignments=study_context.target_assignments,
            target_metric=study_context.target_metric,
            row_id_col=study_context.row_id_col,
            ml_type=study_context.ml_type,
            log_dir=study_context.log_dir,
        )

        # train all models in best_bag
        best_bag.fit(n_assigned_cpus=n_assigned_cpus)

        # save performance values of best bag
        best_bag_performance = best_bag.get_performance(n_assigned_cpus=n_assigned_cpus)
        logger.info(f"Best bag performance {best_bag_performance}")
        target_metric = study_context.target_metric

        # show and save test results
        logger.set_log_group(LogGroup.RESULTS)
        logger.info(
            f"Training: {training_id} "
            f"{target_metric} "
            f"(best bag - ensembled): "
            f"Dev {best_bag_performance[PerformanceKey.DEV_ENSEMBLE]:.3f}, "
            f"Test {best_bag_performance[PerformanceKey.TEST_ENSEMBLE]:.3f}"
        )

        # calculate feature importances of best bag
        fi_methods = self.config.fi_methods
        best_bag_fi = best_bag.calculate_fi(fi_methods, partitions=[DataPartition.DEV], n_assigned_cpus=n_assigned_cpus)

        # calculate selected features
        selected_features = best_bag.get_selected_features(fi_methods)

        # save best bag and results to local dict
        results["best"] = {
            "scores": best_bag_performance,
            "predictions": best_bag.get_predictions(n_assigned_cpus=n_assigned_cpus),
            "fi": best_bag_fi,
            "_bag": best_bag,
        }

        # log selected features info
        logger.set_log_group(LogGroup.RESULTS)
        logger.info("---")
        logger.info(f"Number of original features: {len(feature_cols)}")
        logger.info(f"Number of selected features: {len(selected_features)}")
        if len(selected_features) == 0:
            logger.warning("Best bag - all feature importances values are zero. This hints at a model related problem.")

        return selected_features


type TakoModule = TakoModuleTemplate["Tako"]
