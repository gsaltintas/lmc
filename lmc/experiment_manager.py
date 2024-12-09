# 

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
from lmc.butterfly.butterfly import (get_batch_noise, get_gaussian_noise,
                                     perturb_model)
from lmc.config import Step
from lmc.data.data_stats import SAMPLE_DICT
from lmc.experiment_config import Experiment, PerturbedTrainer, Trainer
from lmc.utils.metrics import report_results
from lmc.utils.setup_training import TrainingElements, setup_experiment

FORMAT = "%(name)s - %(levelname)s: %(message)s"

@dataclass
class ExperimentManager(abc.ABC):
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
        """A description of this manager."""

        pass

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add all command line flags necessary for this manager."""
        cls.config.add_args(parser)

    # @staticmethod
    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> 'ExperimentManager':
        """Create a manager from command line arguments."""
        config = cls.config.create_from_args(args)
        return cls(config)

    @abc.abstractmethod
    def run(self) -> None:
        """Run the job."""

        pass

    @abc.abstractmethod
    def setup(self) -> None:
        pass

@dataclass
class TrainingRunner(ExperimentManager):
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
        # import code; code.interact(local=locals()|globals())
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
                for element_ind, (x, y) in enumerate(batches):
                    element = self.training_elements[element_ind]
                    if element.curr_step >= element.max_steps.get_step(self.steps_per_epoch):
                        break
                    train.step_element(self.config, element, x, y, self.device, self.loss_fn, ep, self.steps_per_epoch, early_iter_ckpt_steps, i=element_ind+1)
            
            self.on_epoch_end(ep, log_dct)

    def on_epoch_end(self, ep: int, log_dct: dict):
        train.eval_epoch(self.config, self.training_elements, self.device, self.steps_per_epoch, self.test_loss_fn, ep, log_dct)
        if self.config.logger.use_wandb:
            wandb.log(log_dct)
        if self.config.logger.print_summary and log_dct:
            report_results(log_dct, ep, self.config.n_models)

def is_same_model(training_elements):
    same_models = True
    for (n1, p1), (n2, p2) in zip(training_elements[0].model.named_parameters(), training_elements[1].model.named_parameters()):
        same_models = same_models and torch.allclose(p1, p2)
        if not same_models:
            return False
        
    return same_models

# is_same_model(self.training_elements)

@dataclass
class PerturbedTrainingRunner(TrainingRunner):
    config: PerturbedTrainer = field(init=True, default=PerturbedTrainer)
    noise_dct: Dict[int, Dict[str, torch.Tensor]] = None
    _name: str = "perturbed-trainer"
    _noise_created: bool = False

    def __post_init__(self):
        self.noise_dct = dict()
        return super().__post_init__()
    @staticmethod
    def description():
        return "Train n model(s) with perturbations."
    
    def create_noise_dicts(self):
        self._noise_created = True
        for ind, el in enumerate(self.training_elements, start=1):
            if ind in self.config.perturb_inds:
                if self.config.perturb_mode == "batch":\
                    # TODO: here double check if the seed messes up somethings
                    self.noise_dct[ind] = get_batch_noise(el.model, dataloader=el.train_loader, noise_seed=el.perturb_seed, loss_fn=el.loss_fn)
                elif self.config.perturb_mode == "gaussian":
                    self.noise_dct[ind] = get_gaussian_noise(el.model, noise_seed=el.perturb_seed)

    def setup(self) -> None:
        super().setup()
        # TODO: create noise dicts at noise creation step
        if self.config.sample_noise_at == "init":
            self.create_noise_dicts()
            self.logger.info("Noise created for models %s at initialization.", self.config.perturb_inds)
    
    def perturb_model(self):
        for ind, el in enumerate(self.training_elements, start=1):
            if ind in self.config.perturb_inds:
                if not self._noise_created:
                    self.create_noise_dicts()
                    self.logger.info("Noise created for models %s at perturbance time.", self.config.perturb_inds)

                if self.config.perturb_mode == "batch":
                    perturb_model(el.model, self.noise_dct[ind], self.config.perturb_scale)
                elif self.config.perturb_mode == "gaussian":
                    perturb_model(el.model, self.noise_dct[ind], self.config.perturb_scale)

                self.logger.info("Model %d perturbed with %f scaling.", ind, self.config.perturb_scale)
                steps = el.max_steps.get_step(self.steps_per_epoch) + self.config.perturb_step
                el.max_steps = Step(steps, self.steps_per_epoch)

    def run(self):
        self.setup()
        #TPDP: make training step as Step
        print(self.config.display)
        print("Running perturbed training.")
        global_step: int = 0
        early_iter_ckpt_steps = train.get_early_iter_ckpt_steps(self.steps_per_epoch, n_ckpts=10)
        # [el.model.to(torch.float64) for el in self.training_elements]
        for ep in range(1, self.max_epochs+1):
            ### train epoch
            log_dct = dict(epoch=ep)
            self.on_epoch_start()
            self.training_elements.on_epoch_start()
            # import code; code.interact(local=locals()|globals())
            if is_same_model(self.training_elements):
                self.logger.info("Models are the same at initialization.")
            train_loaders = [iter(el.train_loader) for el in self.training_elements]
            for batch_ind, batches in enumerate(zip(*train_loaders)):
                if global_step >= self.training_elements.max_steps.get_step(self.steps_per_epoch):
                    break
                if global_step == self.config.perturb_step:
                    self.perturb_model()
                global_step += 1
                # for element_ind, element in enumerate(self.training_elements):
                for element_ind, (x, y) in enumerate(batches):
                    element = self.training_elements[element_ind]
                    if element.curr_step >= element.max_steps.get_step(self.steps_per_epoch):
                        break

                    #something wrong here, train errors come the same but test diff, models change why
                    loss = train.step_element(self.config, element, x, y, self.device, self.loss_fn, ep, self.steps_per_epoch, early_iter_ckpt_steps, i=element_ind+1)
                    # print(loss)
                # print(is_same_model(self.training_elements))
            # import code; code.interact(local=locals()|globals())
            self.on_epoch_end(ep, log_dct)

# # check the loaders return the same batch
# for batch_ind, batches in enumerate(zip(*train_loaders)):
#     prev_x, prev_y = None, None
#     for element_ind, (x, y) in enumerate(batches):
#         if prev_x is not None:
#             print(batch_ind, torch.allclose(prev_x, x))
#             # print(torch.allclose(prev_y, y))
#         else:
#             prev_x, prev_y = x, y

def on_epoch_end(self, ep: int, log_dct: dict):
        # TODO: add lmc stats

        super().on_epoch_end(ep, log_dct)

def same_models(training_elements):
    for (n1, p1), (n2, p2) in zip(training_elements[0].model.named_parameters(), training_elements[1].model.named_parameters()):
        print(n1, torch.allclose(p1, p2))

# same_models(self.training_elements)

managers = dict(train=TrainingRunner, perturb=PerturbedTrainingRunner)


def get(manager_name: str) -> ExperimentManager:
    if manager_name not in managers:
        raise ValueError('No such runner: {}'.format(manager_name))
    else:
        return managers[manager_name]
