import math
from dataclasses import dataclass, field

import torch
import wandb

import train
from lmc.data.data_stats import SAMPLE_DICT
from lmc.experiment_config import Trainer
from lmc.utils.lmc_utils import check_lmc
from lmc.utils.metrics import report_results
from lmc.utils.setup_training import TrainingElements, setup_experiment
from lmc.experiment.base import ExperimentManager


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
        self.steps_per_epoch = math.ceil(
            SAMPLE_DICT[self.config.data.dataset] / self.config.data.batch_size
        )
        self.max_epochs = self.training_elements.max_steps.get_epoch(
            self.steps_per_epoch
        )

    def on_train_start(self):
        pass

    def on_epoch_start(self):
        pass

    def run(self):
        self.setup()
        print(self.config.display)
        early_iter_ckpt_steps = train.get_early_iter_ckpt_steps(
            self.steps_per_epoch, n_ckpts=10
        )
        for ep in range(1, self.max_epochs + 1):
            ### train epoch
            log_dct = dict(epoch=ep)
            self.on_epoch_start()
            self.training_elements.on_epoch_start()
            train_loaders = [iter(el.train_loader) for el in self.training_elements]
            for ind, batches in enumerate(zip(*train_loaders)):
                if self.global_step >= self.training_elements.max_steps.get_step(
                    self.steps_per_epoch
                ):
                    break
                self.global_step += 1
                for element_ind, (x, y) in enumerate(batches):
                    element = self.training_elements[element_ind]
                    if element.curr_step >= element.max_steps.get_step(
                        self.steps_per_epoch
                    ):
                        break
                    train.step_element(
                        self.config,
                        element,
                        x,
                        y,
                        self.device,
                        self.loss_fn,
                        ep,
                        self.steps_per_epoch,
                        early_iter_ckpt_steps,
                        i=element_ind + 1,
                    )

            self.on_epoch_end(ep, log_dct)

    def on_epoch_end(self, ep: int, log_dct: dict):
        train.eval_epoch(
            self.config,
            self.training_elements,
            self.device,
            self.steps_per_epoch,
            self.test_loss_fn,
            ep,
            log_dct,
            self.steps_per_epoch * ep,
        )
        if self.config.n_models > 1 and self.config.lmc.lmc_on_epoch_end:
            check_lmc(
                self.training_elements,
                self.config,
                ep,
                log_dct,
                check_perms=self.config.lmc.lmc_check_perms,
            )

        if self.config.logger.use_wandb:
            wandb.log(log_dct)
        if self.config.logger.print_summary and log_dct:
            report_results(log_dct, ep, self.config.n_models)

    def training_finished(self, training_elements: TrainingElements) -> bool:
        return all(
            el.curr_step >= el.max_steps.get_step(self.steps_per_epoch)
            for el in training_elements
        )
