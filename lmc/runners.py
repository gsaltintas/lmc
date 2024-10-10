# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import argparse
import logging
import math
from dataclasses import dataclass, field
from typing import Dict

import torch
import wandb
from rich.logging import RichHandler
from torch import nn

import train
from lmc.butterfly.butterfly import get_gaussian_noise, perturb_model
from lmc.config import Step
from lmc.data.data_stats import SAMPLE_DICT
from lmc.experiment_config import Experiment, PerturbedTrainer, Trainer
from lmc.utils.metrics import report_results
from lmc.utils.setup_training import TrainingElements, setup_experiment

FORMAT = "%(name)s - %(levelname)s: %(message)s"

@dataclass
class Runner(abc.ABC):
    config: Experiment = field(init=True, default=Experiment)
    _name: str = ""
    logger: logging.Logger = None

    def __post_init__(self):
        logging.basicConfig(level=self.config.logger.level.upper(), format=FORMAT, handlers=[RichHandler(show_time=False)])
        self.logger = logging.getLogger(self._name)
        self.loss_fn, self.test_loss_fn = nn.CrossEntropyLoss(label_smoothing=self.config.trainer.label_smoothing), nn.CrossEntropyLoss()
    
    """An instance of a training run of some kind."""

    @staticmethod
    @abc.abstractmethod
    def description() -> str:
        """A description of this runner."""

        pass

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add all command line flags necessary for this runner."""
        cls.config.add_args(parser)
        pass

    # @staticmethod
    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> 'Runner':
        """Create a runner from command line arguments."""
        config = cls.config.create_from_args(args)
        return cls(config)
        pass

    @abc.abstractmethod
    def run(self) -> None:
        """Run the job."""

        pass

    @abc.abstractmethod
    def setup(self) -> None:
        pass

@dataclass
class TrainingRunner(Runner):
    config: Trainer = field(init=True, default=Trainer)
    _name: str = "trainer"

    training_elements: TrainingElements = None
    device: torch.device = None
    steps_per_epoch: int = None
    max_epochs: int = None
    global_step: int = 0

    @staticmethod
    def description():
        return "Train n model(s)."
    
    def setup(self) -> None:
        self.training_elements: TrainingElements
        self.device: torch.device 
        self.training_elements, self.device = setup_experiment(self.config)
        self.steps_per_epoch = math.ceil(SAMPLE_DICT[self.config.data.dataset] / self.config.data.batch_size)
        self.max_epochs = self.training_elements.max_steps.get_epoch(self.steps_per_epoch)

    def on_epoch_start(self):
        pass

    def run(self):
        self.setup()
        print(self.config.display)
        global_step: int = 0
        early_iter_ckpt_steps = train.get_early_iter_ckpt_steps(self.steps_per_epoch, n_ckpts=10)
        for ep in range(1, self.max_epochs+1):
            ### train epoch
            log_dct = dict(epoch=ep)
            self.on_epoch_start()
            self.training_elements.on_epoch_start()
            train_loaders = [iter(el.train_loader) for el in self.training_elements]
            for ind, batches in enumerate(zip(*train_loaders)):
                if global_step >= self.training_elements.max_steps.get_step(self.steps_per_epoch):
                    break
                global_step += 1
                for i, (x, y) in enumerate(batches):
                    element = self.training_elements[i]
                    if element.curr_step >= element.max_steps.get_step(self.steps_per_epoch):
                        break
                    train.step_element(self.config, element, x, y, self.device, self.loss_fn, ep, self.steps_per_epoch, early_iter_ckpt_steps, i=i)
            
            self.on_epoch_end(ep, log_dct)

    def on_epoch_end(self, ep: int, log_dct: dict):
        train.eval_epoch(self.config, self.training_elements, self.device, self.steps_per_epoch, self.test_loss_fn, ep, log_dct)
        if self.config.logger.use_wandb:
            wandb.log(log_dct)
        if self.config.logger.print_summary and log_dct:
            report_results(log_dct, ep, self.config.n_models)

@dataclass
class PerturbedTrainingRunner(TrainingRunner):
    config: PerturbedTrainer = field(init=True, default=PerturbedTrainer)
    noise_dct: Dict[int, Dict[str, torch.Tensor]] = None
    _name: str = "perturbed-trainer"

    def __post_init__(self):
        self.noise_dct = dict()
        return super().__post_init__()
    @staticmethod
    def description():
        return "Train n model(s) with perturbations."
    
    def setup(self) -> None:
        super().setup()
        for ind, el in enumerate(self.training_elements, start=1):
            if ind in self.config.perturb_inds:
                if self.config.perturb_mode == "batch":
                    raise NotImplementedError("Not implemented")
                elif self.config.perturb_mode == "gaussian":
                    self.noise_dct[ind] = get_gaussian_noise(el.model)
                steps = el.max_steps.get_step(self.steps_per_epoch) + self.config.perturb_step
                el.max_steps = Step(steps, self.steps_per_epoch)
    
    def perturb_model(self):
        for ind, el in enumerate(self.training_elements, start=1):
            if ind in self.config.perturb_inds:
                if self.config.perturb_mode == "batch":
                    raise NotImplementedError("Not implemented")
                elif self.config.perturb_mode == "gaussian":
                    perturb_model(el.model, self.noise_dct[ind], self.config.perturb_scale)

                    self.logger.info("Model %d perturbed with %f scaling.", ind, self.config.perturb_scale)
        
    def run(self):
        self.setup()
        #TPDP: make training step as Step
        print(self.config.display)
        global_step: int = 0
        early_iter_ckpt_steps = train.get_early_iter_ckpt_steps(self.steps_per_epoch, n_ckpts=10)
        for ep in range(1, self.max_epochs+1):
            ### train epoch
            log_dct = dict(epoch=ep)
            self.on_epoch_start()
            self.training_elements.on_epoch_start()
            train_loaders = [iter(el.train_loader) for el in self.training_elements]
            for ind, batches in enumerate(zip(*train_loaders)):
                if global_step >= self.training_elements.max_steps.get_step(self.steps_per_epoch):
                    break
                if global_step == self.config.perturb_step:
                    self.perturb_model()
                global_step += 1
                for i, (x, y) in enumerate(batches):
                    element = self.training_elements[i]
                    if element.curr_step >= element.max_steps.get_step(self.steps_per_epoch):
                        break
                    train.step_element(self.config, element, x, y, self.device, self.loss_fn, ep, self.steps_per_epoch, early_iter_ckpt_steps, i=i+1)
            
            self.on_epoch_end(ep, log_dct)




runners = dict(train=TrainingRunner, perturb=PerturbedTrainingRunner)


def get(runner_name: str) -> Runner:
    if runner_name not in runners:
        raise ValueError('No such runner: {}'.format(runner_name))
    else:
        return runners[runner_name]
