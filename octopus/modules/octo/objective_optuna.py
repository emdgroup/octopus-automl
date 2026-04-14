"""Objective function for optuna optimization."""

import heapq

from optuna.trial import Trial
from upath import UPath

from octopus.datasplit import InnerSplits
from octopus.logger import get_logger
from octopus.metrics import Metrics
from octopus.models import Models
from octopus.types import DataPartition, LogGroup, MetricDirection, MLType, ModelName, ScoringMethod
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
        outer_split_task_id: str,
        outer_split_id: int,
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
        n_assigned_cpus: int,
    ):
        self.outer_split_task_id = outer_split_task_id
        self.outer_split_id = outer_split_id
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
        self.n_save_trials = self.config.n_ensemble_candidates
        # parameters potentially used for optimizations
        self.ml_model_types = self.config.models
        self.max_outl = self.config.max_outliers
        self.max_features = self.config.max_features
        self.hyper_parameters = self.config.hyperparameters
        # training parameters
        self.log_dir = log_dir
        self.n_assigned_cpus = n_assigned_cpus

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
            n_outliers = trial.suggest_int(name="n_outliers", low=0, high=self.max_outl)
        else:
            n_outliers = 0

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
            model_seed=0,
        )

        config_training: TrainingConfig = {
            "outl_reduction": n_outliers,
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
                    training_id=self.outer_split_task_id + "_" + str(key),
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
            bag_id=self.outer_split_task_id + "_" + str(trial.number),
            trainings=trainings,
            target_assignments=self.target_assignments,
            target_metric=self.target_metric,
            row_id_col=self.row_id_col,
            ml_type=self.ml_type,
            log_dir=self.log_dir,
            # path?
        )

        # train all models in bag
        bag_trainings.fit(n_assigned_cpus=self.n_assigned_cpus)

        # evaluate trainings using target metric
        bag_performance = bag_trainings.get_performance(n_assigned_cpus=self.n_assigned_cpus)

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
        if self.config.scoring_method == ScoringMethod.COMBINED:
            optuna_target: float = bag_performance["dev_ensemble"]
        else:
            optuna_target = bag_performance["dev_avg"]

        # adjust direction, optuna always maximizes (higher = better)
        target_metric = self.target_metric
        if Metrics.get_direction(target_metric) == MetricDirection.MINIMIZE:
            optuna_target = -optuna_target

        # add penaltiy for n_features > max_features if configured
        if self.max_features > 0:
            diff_nfeatures = n_features_mean - self.max_features
            # only consider if n_features_mean > max_features
            diff_nfeatures = max(diff_nfeatures, 0)
            n_features = len(self.feature_cols)
            optuna_target = optuna_target - self.config.penalty_factor * diff_nfeatures / n_features

        # save bag if we plan to run ensemble selection
        if self.ensel:
            self._save_topn_trials(bag_trainings, optuna_target, trial.number)

        logger.info(f"Otarget: {optuna_target}")
        logger.info(f"Number of features used: {int(n_features_mean)}")

        return optuna_target

    def _save_topn_trials(self, bag: BagClassifier | BagRegressor, target_value, n_trial):
        max_n_trials = self.n_save_trials
        path_bag = self.path_study / self.task_path / "scratch" / f"trial_{n_trial}_bag.joblib"
        path_preds = self.path_study / self.task_path / "scratch" / f"trial_{n_trial}_preds.joblib"

        # saving top n_trials to disk
        # target_value is always "higher = better" (optuna maximizes).
        # Min-heap: heappop removes the lowest value (worst trial).
        heapq.heappush(self.top_trials, (target_value, path_bag))
        joblib_save(bag, path_bag)

        # Save lightweight predictions for fast EnSel loading
        full_predictions = bag.get_predictions(n_assigned_cpus=self.n_assigned_cpus)
        predictions_ensel = {}
        for key, partitions in full_predictions.items():
            predictions_ensel[key] = {p: df for p, df in partitions.items() if p != DataPartition.TRAIN}
        predictions_data = {
            "predictions": predictions_ensel,
            "bag_id": bag.bag_id,
            "n_features_used_mean": bag.n_features_used_mean,
            "target_dtypes": {
                col: bag.trainings[0].data_train[col].dtype
                for col in self.target_assignments.values()
                if col in bag.trainings[0].data_train.columns
            },
        }
        joblib_save(predictions_data, path_preds)

        if len(self.top_trials) > max_n_trials:
            _, path_delete = heapq.heappop(self.top_trials)
            if path_delete.is_file():
                path_delete.unlink()
            else:
                raise FileNotFoundError("Problem deleting trial-pkl file")
            path_preds_delete = path_delete.with_name(path_delete.name.replace("_bag.joblib", "_preds.joblib"))
            if path_preds_delete.is_file():
                path_preds_delete.unlink()

    def _log_trial_scores(self, scores):
        logger.set_log_group(LogGroup.SCORES, f"OUTER {self.outer_split_id} SQE TBD")
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
