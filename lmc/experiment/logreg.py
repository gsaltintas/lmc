from copy import deepcopy
from dataclasses import dataclass, field, fields
import json
from sklearn.linear_model import LogisticRegression
import numpy as np
import wandb

from lmc.config import Config
from lmc.experiment_config import PerturbedTrainer, dataclass_from_dict
from lmc.experiment.base import ExperimentManager
from lmc.utils.run import run_command, command_result_is_error
from lmc.utils.setup_training import setup_model_dir, setup_wandb


@dataclass
class LogregConfig(Config):
    _name_prefix: str = "Logistic regression search hparams"
    _description: str = "Hyperparameters for active learning logistic regression."

    logreg_x: str = "perturb_scale"
    logreg_y: str = "lmc-0-1/lmc/loss/weighted/barrier_train"
    logreg_threshold: float = 0.0
    logreg_n: int = 20
    logreg_max_step_ratio: float = 10
    logreg_reseed_every_run: bool = True
    # logreg_early_stop_runs: bool = False
    # logreg_early_stop_n_points: int = 3
    # logreg_early_stop_n_epochs: int = 3
    # logreg_early_stop_threshold: float = -1

    _logreg_x: str = (
        "config key for the independent variable in the logistic regression"
    )
    _logreg_y: str = "wandb key for the response variable in the logistic regression"
    _logreg_threshold: str = "if y < threshold after training, give a response of 0, else 1, for logistic regression"
    _logreg_n: str = "how many perturbation runs to do, including the initial guess at config.[logreg_x]"
    _logreg_max_step_ratio: str = "maximum multiplicative factor M > 1 to choose next x by, so that x_1 / M <= x_2 <= M x_1"
    _logreg_reseed_every_run: str = "If true, use a new seed for each run that is deterministically computed from the seeds given as hyperparameters. Otherwise, use the same seeds for every run. Defaults to true."
    # _logreg_early_stop_runs: str = "if True, stop run early if following condition holds: at each epoch, check if logreg_early_stop_threshold has been exceeded for logreg_early_stop_n_epochs times. If this condition holds, stop the run early and assign a response of 0"
    # _logreg_early_stop_n_points: str = "if logreg_early_stop_runs, how many n_points of LMCConfig to evaluate along interpolated path at every epoch"
    # _logreg_early_stop_n_epochs: str = "if logreg_early_stop_runs, how many epochs that logreg_y must exceed logreg_early_stop_threshold before stopping the run, must be > 0"
    # _logreg_early_stop_threshold: str = "if logreg_early_stop_runs, test loreg_value against this threshold. If less than 0, use logreg_threshold. Cannot be smaller than logreg_threshold."


@dataclass(init=False)
class LogisticRegressionSearch(PerturbedTrainer):
    logreg: LogregConfig = None

    def __init__(self, *args, **kwargs):
        self.logreg = kwargs.get("logreg") or dataclass_from_dict(LogregConfig, kwargs)
        super().__init__(*args, **kwargs)


@dataclass
class ActiveLearningLogisticRegressionSearch(ExperimentManager):
    config: LogisticRegressionSearch = field(
        init=True, default=LogisticRegressionSearch
    )

    _name: str = "logreg"

    def __post_init__(self):
        #TODO require wandb for now since we have no other way of accessing logged metrics, however wandb_offline can be True
        assert self.config.logger.use_wandb, "Need to use wandb in order to track experiment values"
        self.regression_model = LogisticRegression()
        return super().__post_init__()

    def setup(self) -> None:
        self.exp_config = deepcopy(self.config)
        setup_model_dir(self.config)
        setup_wandb(self.config)
        self.wandb_entity = wandb.run.entity
        self.wandb_project = wandb.run.project
        self.wandb_run_id = wandb.run.id

    @staticmethod
    def description():
        return "Repeat PerturbedTrainingRunner experiment to produce data for active learning logistic regression, finding x at which y is below a threshold 1/2 of the time. Assumes that x and y are monotonically and positively correlated!"

    def run(self):
        # set initial guess based on config value, e.g. if config.logreg_x = "perturb_step", use config.perturb_step
        x_predicted = getattr(self.config, self.config.logreg.logreg_x)
        x = []
        y = []
        # choose next data point based on where model predicts probability of 0.5
        # for binary classification this is equivalent to max entropy method, which is simple and effective for active learning
        # see Yang, Y., & Loog, M. (2018). A benchmark and comparison of active learning for logistic regression. Pattern Recognition, 83, 401-415.
        for i in range(self.config.logreg.logreg_n):
            # update seeds and run name
            label, log_dct = self.perturb_with_early_stopping(x_predicted, i)

            # fit logistic regression
            x.append(x_predicted)
            y.append(label)
            x_predicted, slope, intercept = max_entropy_x(
                x,
                y,
                step_ratio=self.config.logreg.logreg_max_step_ratio,
                regression_model=self.regression_model,
            )

            # log results
            log_dct = {
                **log_dct,
                "logreg/run/slope": slope,
                "logreg/run/intercept": intercept,
                "logreg/run/max_entropy_x": x_predicted,
            }
            if self.config.logger.use_wandb:
                wandb.log(log_dct)

        # estimate one final time without min/max limits on x
        x_predicted, slope, intercept = max_entropy_x(
            x, y, regression_model=self.regression_model
        )
        log_dct = {
            "logreg/slope": slope,
            "logreg/intercept": intercept,
            "logreg/max_entropy_x": x_predicted,
        }
        if self.config.logger.use_wandb:
            wandb.log(log_dct)

    def perturb_with_early_stopping(self, x: float, iter: int):
        # write hparams to file so that subprocess can access them easily
        log_dct = self.update_perturbed_trainer_config(x, iter)
        config_dir = self.config.model_dir / f"run-{iter}"
        self.exp_config.save(config_dir, zip_code_base=False)
        # run the experiment as subprocess, this guarantees the runs are identical to individual experiments
        command = f"python main.py perturb --config_file {config_dir}/config.yaml"
        result = run_command(
            command,
            print_output=True,
        )
        if command_result_is_error(result):
            raise ValueError("Error when running {command}")

        # as part of ExperimentManager.finish(), the wandb summary is written out to a json file
        with open(self.exp_config.model_dir / "wandb_summary.json", "r") as f:
            summary = json.load(f)
        value = summary[self.config.logreg.logreg_y]
        label = 0 if value < self.config.logreg.logreg_threshold else 1
        log_dct = {
            **log_dct,
            "logreg/run/" + self.config.logreg.logreg_y: value,
            "logreg/run/threshold_label": label,
        }
        return label, log_dct

    def update_perturbed_trainer_config(self, x, iter):
        setattr(self.exp_config, self.config.logreg.logreg_x, x)
        self.exp_config.model_dir = self.config.model_dir.parent / f"{self.config.model_dir.name}-run{iter}"
        self.exp_config.logger.entity = self.wandb_entity
        self.exp_config.logger.project = self.wandb_project
        self.exp_config.logger.run_id = f"{self.wandb_run_id}-run{iter}"
        self.exp_config.logger.run_name = f"{self.config.logger.run_name}-run{iter}"
        log_dct = {
            "logreg/run/iter": iter,
            f"logreg/run/{self.config.logreg.logreg_x}": x,
            "logreg/run/id": self.exp_config.logger.run_id,
            **reseed(self.config.seeds, self.exp_config.seeds, iter),
            **reseed(self.config.perturb_seeds, self.exp_config.perturb_seeds, iter),
        }
        return log_dct


def reseed(source, dest, iter):
    log_dct = {}
    for field_ in fields(source):
        value = getattr(source, field_.name)
        if "seed" in field_.name and isinstance(value, int):
            # numpy random seed must be between 0 and 2**32 - 1
            new_seed = hash((value, iter)) % 2**32
            setattr(dest, field_.name, new_seed)
        log_dct[f"logreg/run/{field_.name}"] = getattr(dest, field_.name)
    return log_dct


def max_entropy_x(x, y, step_ratio=None, regression_model=None):
    if step_ratio is None:
        min_val = None
        max_val = None
    else:
        min_val = np.min(x) / step_ratio
        max_val = np.max(x) * step_ratio

    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    # if y are all 0 or all 1, return a step in the appropriate direction by step_ratio
    if np.all(y == 0):  # we can't find a big enough value, so return inf
        next_x = np.inf if step_ratio is None else max_val
        return next_x, np.nan, np.nan
    elif np.all(y == 1):  # we can't find a small enough value, so return 0
        next_x = 0 if step_ratio is None else min_val
        return next_x, np.nan, np.nan

    regression_model.fit(x, y)
    next_x = find_midpoint(regression_model, min_val=min_val, max_val=max_val)
    return next_x, regression_model.coef_.item(), regression_model.intercept_.item()


def find_midpoint(
    regression_model: LogisticRegression, min_val=None, max_val=None, probability=0.5
):
    # y = ax + b, solve for x when y=probability
    a = regression_model.coef_.item()
    b = regression_model.intercept_.item()
    # x = (y - b) / a
    if (
        a <= 0
    ):  # if slope is 0, return min or max value depending on whether midpoint is below/above probability
        return min_val if b < probability else max_val
    midpoint = (probability - b) / a

    # clip output
    if min_val is not None:
        midpoint = max(midpoint, min_val)
    if max_val is not None:
        midpoint = min(midpoint, max_val)
    return midpoint
