from dataclasses import dataclass, field
from typing import Dict

import torch
import wandb
from torch.nn.utils import parameters_to_vector

import train
from lmc.butterfly.butterfly import (
    get_batch_noise,
    get_gaussian_noise,
    get_noise_l2,
    perturb_model,
)
from lmc.config import Step
from lmc.experiment_config import PerturbedTrainer
from lmc.utils.lmc_utils import check_lmc
from lmc.utils.opt import get_lr, reset_base_lrs
from lmc.utils.setup_training import (
    TrainingElement,
    configure_lr_scheduler,
    setup_loader,
)
from lmc.experiment.train import TrainingRunner


def is_same_model(training_elements):
    same_models = True
    for (n1, p1), (n2, p2) in zip(
        training_elements[0].model.named_parameters(),
        training_elements[1].model.named_parameters(),
    ):
        same_models = same_models and torch.allclose(p1, p2)
        if not same_models:
            return False

    return same_models


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
                if (
                    self.config.perturb_mode == "batch"
                ):  # TODO: here double check if the seed messes up somethings
                    dl = setup_loader(
                        self.config.data,
                        train=True,
                        evaluate=False,
                        loader_seed=el.perturb_seed,
                    )
                    self.noise_dct[ind] = get_batch_noise(
                        el.model,
                        dataloader=dl,
                        noise_seed=el.perturb_seed,
                        loss_fn=el.loss_fn,
                    )
                elif self.config.perturb_mode == "gaussian":
                    self.noise_dct[ind] = get_gaussian_noise(
                        el.model, noise_seed=el.perturb_seed
                    )

    def setup(self) -> None:
        super().setup()
        if self.config.sample_noise_at == "init":
            self.create_noise_dicts()
            self.logger.info(
                "Noise created for models %s at initialization.",
                self.config.perturb_inds,
            )

    def on_train_start(self):
        super().on_train_start()
        self.models_at_init = [
            parameters_to_vector(el.model.parameters()) for el in self.training_elements
        ]
        log_dct = dict()
        log_dct.update(
            {
                f"static/l2_at_init/{i}": torch.norm(v).item()
                for i, v in enumerate(self.models_at_init, start=1)
            }
        )
        log_dct.update(
            {
                f"static/l2_dist_at_init/{i}-{i+1}": torch.norm(v1 - v2).item()
                for i, (v1, v2) in enumerate(
                    zip(self.models_at_init, self.models_at_init[1:]), start=1
                )
            }
        )
        if self.config.logger.use_wandb:
            wandb.log(log_dct)

    def reset_lr_schedule(
        self, element: TrainingElement, prev_max_steps: int = None
    ) -> None:
        current_lr = get_lr(element.opt)
        self.logger.info("Lr scheduler will continue from this point (%s).", current_lr)
        for g in element.opt.param_groups:
            assert (
                g["lr"] == current_lr
            ), f"Lr of the parameter group {g} is not configured properly."
        steps_per_epoch = len(element.train_loader)

        if prev_max_steps is None:
            prev_max_steps = element.max_steps.get_step(steps_per_epoch)
        warmup_remaining = max(
            0, self.config.trainer.opt.warmup_ratio - self.global_step / prev_max_steps
        )
        warmup_ratio = self.config.trainer.opt.warmup_ratio
        warmup_steps = warmup_ratio * prev_max_steps
        # Log the warmup state
        if warmup_remaining > 0:
            self.logger.info(
                "Warmup period detected. Remaining warmup ratio: %.4f", warmup_remaining
            )
        # start from 0
        element.scheduler = configure_lr_scheduler(
            element.opt,
            element.max_steps.get_step(steps_per_epoch),
            self.config.trainer.opt.lr_scheduler,
            warmup_ratio,
            {},
            global_step=self.global_step,  # to restart lr from 0, set this to 0
            warmup_steps=warmup_steps,
        )
        # if warmup_ratio == 0:
        if warmup_remaining == 0:
            reset_base_lrs(element.opt, current_lr, element.scheduler)
            # reset_base_lrs(element.opt, current_lr, element.scheduler)

    def perturb_model(self, log_dct: dict):
        for ind, el in enumerate(self.training_elements, start=1):
            if ind not in self.config.perturb_inds:
                continue
            if not self._noise_created:
                self.create_noise_dicts()
                self.logger.info(
                    "Noise created for models %s at perturbance time.",
                    self.config.perturb_inds,
                )

            perturb_model(el.model, self.noise_dct[ind], self.config.perturb_scale)

            noise_l2 = get_noise_l2(self.noise_dct[ind])
            self.logger.info(
                "Model %d perturbed with %f scaling, absolute l2 %f.",
                ind,
                self.config.perturb_scale,
                noise_l2,
            )
            log_dct[f"static/noise/{ind}-l2"] = noise_l2
            log_dct[f"static/noise/{ind}-l2-scaled"] = noise_l2 * (
                self.config.perturb_scale**2
            )
            if self.config.same_steps_pperturb:
                if self.global_step < 1:
                    continue
                prev_max_steps = el.max_steps.get_step(self.steps_per_epoch)
                steps = prev_max_steps + self.config.perturb_step.get_step(
                    self.steps_per_epoch
                )
                el.max_steps = Step(steps, self.steps_per_epoch)
                self.logger.info(
                    "Model %d steps set to %d.",
                    ind,
                    steps,
                )
                self.reset_lr_schedule(el, prev_max_steps=prev_max_steps)
                self.logger.info("Model %d lr schedule reset.", ind)

    def run(self):
        self.setup()
        # TPDP: make training step as Step
        print(self.config.display)
        print("Running perturbed training.")
        early_iter_ckpt_steps = train.get_early_iter_ckpt_steps(
            self.steps_per_epoch, n_ckpts=10
        )
        ep: int = 1
        if is_same_model(self.training_elements):
            self.logger.info("Models are the same at initialization.")
        self.on_train_start()
        while not self.training_finished(self.training_elements):
            ### train epoch
            log_dct = dict(epoch=ep)
            self.on_epoch_start()
            self.training_elements.on_epoch_start()
            train_loaders = [iter(el.train_loader) for el in self.training_elements]
            for batch_ind, batches in enumerate(zip(*train_loaders)):
                if self.global_step >= self.training_elements.max_steps.get_step(
                    self.steps_per_epoch
                ):
                    break
                if self.global_step == self.config.perturb_step.get_step(
                    self.steps_per_epoch
                ):
                    self.perturb_model(log_dct=log_dct)
                self.global_step += 1
                for element_ind, (x, y) in enumerate(batches):
                    element = self.training_elements[element_ind]
                    if element.curr_step >= element.max_steps.get_step(
                        self.steps_per_epoch
                    ):
                        continue
                    element.train_iterator.update()

                    loss = train.step_element(
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
            ep += 1
        self.on_train_end(ep)

    def on_epoch_end(self, ep: int, log_dct: dict):
        super().on_epoch_end(ep, log_dct)

    def on_train_end(self, ep: int):
        log_dct = dict(epoch=ep)
        if (
            self.config.n_models > 1
            and self.config.lmc.lmc_on_train_end
            and not self.config.lmc.lmc_on_epoch_end
        ):
            check_lmc(
                self.training_elements,
                self.config,
                ep,
                log_dct,
                check_perms=self.config.lmc.lmc_check_perms,
            )
        if self.config.logger.use_wandb:
            wandb.log(log_dct)
