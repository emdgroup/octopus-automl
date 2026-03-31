"""Objective function for optuna optimization."""

import heapq

from optuna.trial import Trial
from upath import UPath

from octopus.datasplit import InnerSplits
from octopus.logger import get_logger
from octopus.manager import ParallelResources
from octopus.metrics import Metrics
from octopus.models import Models
from octopus.types import LogGroup, MetricDirection, MLType, ModelName, OptunaReturnType
from octopus.utils import joblib_save

from .bag import Bag, BagClassifier, BagRegressor
from .training import Training, TrainingConfig

logger = get_logger()


class ObjectiveOptuna:
    """Callable optuna objective.

    A single solution for global and individual HP optimizations.
    """

    def __init__(
        self,
        outersplit_task_id: str,
        outersplit_id: int,
        ml_type: MLType,
        target_assignments: dict,
        feature_cols: list[str],
        row_column: str,
        data_test,
        target_metric: str,
        feature_groups: dict,
        positive_class,
        config,
        path_study: UPath,
        task_path: str,
        data_splits: InnerSplits,
        study_name,
        top_trials,
        mrmr_features,
        log_dir: UPath,
        resources: ParallelResources,
    ):
        self.outersplit_task_id = outersplit_task_id
        self.outersplit_id = outersplit_id
        self.ml_type = ml_type
        self.target_assignments = target_assignments
        self.feature_cols = feature_cols
        self.row_id_col = row_column  # Store as row_id_col for compatibility with Training
        self.data_test = data_test
        self.target_metric = target_metric
        self.feature_groups = feature_groups
        self.positive_class = positive_class
        self.config = config
        self.path_study = path_study
        self.task_path = task_path
        self.data_splits = data_splits
        self.study_name = study_name
        self.top_trials = top_trials
        self.mrmr_features = mrmr_features
        # saving trials
        self.ensel = self.config.ensemble_selection
        self.n_save_trials = self.config.ensel_n_save_trials
        # parameters potentially used for optimizations
        self.ml_model_types = self.config.models
        self.max_outl = self.config.max_outl
        self.max_features = self.config.max_features
        self.penalty_factor = self.config.penalty_factor
        self.hyper_parameters = self.config.hyperparameters
        # fixed parameters
        self.ml_seed = self.config.model_seed
        # training parameters
        self.log_dir = log_dir
        self.resources = resources

    def __call__(self, trial: Trial) -> float:
        """Call.

        We have different types of parameters:
        (a) non-model parameters that are needed in
            the training
        (b) model parameters that are varied by optuna
            (defined by default or optuna_model_settings)
        (c) global parameters that have to be translated
            in fixed model specific parameters
        """
        # get non-model parameters
        # (1) ml_model_type
        ml_model_type = ModelName(
            trial.suggest_categorical(name="ml_model_type", choices=self.ml_model_types)
            if len(self.ml_model_types) > 1
            else self.ml_model_types[0]
        )

        # (3) number of outliers to be detected
        if self.max_outl > 0:
            num_outl = trial.suggest_int(name="num_outl", low=0, high=self.max_outl)
        else:
            num_outl = 0

        # (4) selected mrmr features
        if self.mrmr_features:
            feat_id = trial.suggest_categorical(name="n_mrmr_features", choices=list(self.mrmr_features.keys()))
            feature_cols = self.mrmr_features[feat_id]
        else:
            feature_cols = self.feature_cols

        # get hyper parameter space for selected model
        model_params = Models.create_trial_parameters(
            trial,
            ml_model_type,
            self.hyper_parameters,
            n_jobs=1,  # inner parallelization happens over inner splits, so we do not allow any further parallelization here.  # TODO: how about setting parallelization over inner splits and then compute n_jobs accordingly?
            model_seed=self.ml_seed,
        )

        config_training: TrainingConfig = {
            "outl_reduction": num_outl,
            "n_input_features": len(feature_cols),
            "ml_model_type": ml_model_type,
            "ml_model_params": model_params,
            "positive_class": self.positive_class,
        }

        # create trainings
        trainings = []
        for key, split in self.data_splits.items():
            trainings.append(
                Training(
                    training_id=self.outersplit_task_id + "_" + str(key),
                    ml_type=self.ml_type,
                    target_assignments=self.target_assignments,
                    feature_cols=feature_cols,
                    row_id_col=self.row_id_col,
                    data_train=split.train,  # inner datasplit, train
                    data_dev=split.dev,  # inner datasplit, dev
                    data_test=self.data_test,
                    config_training=config_training,
                    target_metric=self.target_metric,
                    max_features=self.config.max_features,
                    feature_groups=self.feature_groups,
                )
            )

        # create bag with all provided trainings
        bag_trainings = Bag(
            bag_id=self.outersplit_task_id + "_" + str(trial.number),
            trainings=trainings,
            target_assignments=self.target_assignments,
            target_metric=self.target_metric,
            row_id_col=self.row_id_col,
            ml_type=self.ml_type,
            log_dir=self.log_dir,
            # path?
        )

        # train all models in bag
        bag_trainings.fit(resources=self.resources)

        # evaluate trainings using target metric
        bag_performance = bag_trainings.get_performance(resources=self.resources)

        # get number of features used in bag
        n_features_mean = bag_trainings.n_features_used_mean

        # add scores info to the optuna trial
        for key, value in bag_performance.items():
            trial.set_user_attr(key, value)

        # add config_training to user attributes
        trial.set_user_attr("config_training", config_training)

        # log results
        self._log_trial_scores(bag_performance)

        # define optuna target
        if self.config.optuna_return == OptunaReturnType.POOL:
            optuna_target: float = bag_performance["dev_pool"]
        else:
            optuna_target = bag_performance["dev_avg"]

        # adjust direction, optuna in octofull always minimizes
        target_metric = self.target_metric
        if Metrics.get_direction(target_metric) == MetricDirection.MINIMIZE:
            optuna_target = -optuna_target

        # add penaltiy for n_features > max_features if configured
        if self.max_features > 0:
            diff_nfeatures = n_features_mean - self.max_features
            # only consider if n_features_mean > max_features
            diff_nfeatures = max(diff_nfeatures, 0)
            n_features = len(self.feature_cols)
            optuna_target = optuna_target - self.penalty_factor * diff_nfeatures / n_features

        # save bag if we plan to run ensemble selection
        if self.ensel:
            self._save_topn_trials(bag_trainings, optuna_target, trial.number)

        logger.info(f"Otarget: {optuna_target}")
        logger.info(f"Number of features used: {int(n_features_mean)}")

        return optuna_target

    def _save_topn_trials(self, bag: BagClassifier | BagRegressor, target_value, n_trial):
        max_n_trials = self.config.ensel_n_save_trials
        path_save = self.path_study / self.task_path / "scratch" / f"trial_{n_trial}_bag.joblib"

        # saving top n_trials to disk
        # the optuna target_value will always be minimized. Heappop removes the lowest
        # value, therefore target_value needs to be negated.
        heapq.heappush(self.top_trials, (-1 * target_value, path_save))
        joblib_save(bag, path_save)
        if len(self.top_trials) > max_n_trials:
            # delete trial with lowest perfomrmance in n_trials
            _, path_delete = heapq.heappop(self.top_trials)
            if path_delete.is_file():
                path_delete.unlink()
            else:
                raise FileNotFoundError("Problem deleting trial-pkl file")

    def _log_trial_scores(self, scores):
        logger.set_log_group(LogGroup.SCORES, f"OUTER {self.outersplit_id} SQE TBD")
        # Log the target metric
        logger.info(f"Trial scores for metric: {self.target_metric}")

        # Separate list and non-list values
        list_items = {}
        non_list_items = {}

        for key, value in scores.items():
            if isinstance(value, list):
                list_items[key] = value
            else:
                non_list_items[key] = value

        # Log non-list items in one message with | as a divider
        non_list_message = " | ".join([f"{key}: {value:.3f}" for key, value in non_list_items.items()])
        logger.info(f"Scores: {non_list_message}")

        # Log list items individually
        for key, value in list_items.items():
            logger.info(f"{key}: {value}")
