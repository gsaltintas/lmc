from dataclasses import dataclass, field

import torch
from torch.nn.utils import parameters_to_vector
from transformers import AutoTokenizer

import wandb
from lmc.butterfly.butterfly import get_average_grad_norm, sample_noise_and_perturb
from lmc.experiment.train import TrainingRunner
from lmc.experiment_config import PerturbedTrainer
from lmc.models.layers import has_batch_norm
from lmc.utils.lmc_utils import repair
from lmc.utils.opt import get_lr, reset_base_lrs
from lmc.utils.setup_training import configure_lr_scheduler, setup_loader
from lmc.utils.step import Step
from lmc.utils.training_element import TrainingElement


@dataclass
class PerturbedTrainingRunner(TrainingRunner):
    config: PerturbedTrainer = field(init=True, default=PerturbedTrainer)
    _name: str = "perturbed-trainer"

    @staticmethod
    def description():
        return "Train n model(s) with perturbations."

    def on_train_start(self):
        super().on_train_start()
        print("Running perturbed training.")
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
                f"static/l2_dist_at_init/{i}-{i + 1}": torch.norm(v1 - v2).item()
                for i, (v1, v2) in enumerate(
                    zip(self.models_at_init, self.models_at_init[1:]), start=1
                )
            }
        )
        # save per-layer L2s
        for i, el in enumerate(self.training_elements, start=1):
            log_dct.update(
                {
                    f"static/l2_at_init/layers/{i}/{k}": torch.norm(v.flatten()).item()
                    for k, v in el.model.named_parameters()
                    if v.requires_grad
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
            assert g["lr"] == current_lr, (
                f"Lr of the parameter group {g} is not configured properly."
            )
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

    def get_train_loader(self, seed: int = None, tokenizer: AutoTokenizer = None):
        return setup_loader(
            self.config.data,
            train=True,
            evaluate=False,
            loader_seed=seed,
            tokenizer=tokenizer,
        )

    def perturb_model(self):
        log_dct = {"step/global": self.global_step}
        for ind, el in enumerate(self.training_elements, start=1):
            if ind not in self.config.perturb_inds:
                continue
            noise_stats = sample_noise_and_perturb(
                self.config, el.model, el.perturb_seed, el.loss_fn
            )
            noise_stats = {f"static/noise/{ind}-{k}": v for k, v in noise_stats.items()}
            log_dct.update(noise_stats)
            log_dct[f"step/model{ind}"] = el.curr_step
            if has_batch_norm(el.model):
                dl = setup_loader(
                    self.config.data,
                    train=True,
                    evaluate=False,
                    loader_seed=el.loader_seed,
                )
                repair(el.model, dl)
                self.logger.info(
                    "Model has batch norm, passing training data once to eliminate variance collapse."
                )
            for num_data_points in [1, 5, -1]:
                dl = self.get_train_loader(el.loader_seed, tokenizer=el.tokenizer)
                avg_grad_norm, grad_count = get_average_grad_norm(
                    el.model, dl, num_datapoints=num_data_points
                )
                log_dct[
                    self.wandb_registry.get_metric(
                        f"grad_norm_{ind}_on_{num_data_points}"
                    ).log_name
                ] = avg_grad_norm
                log_dct[
                    self.wandb_registry.get_metric(f"grad_count_{ind}").log_name
                ] = grad_count
                del dl
            self.logger.info(
                "Model %d perturbed at %i with %f scaling, absolute l2 %f.",
                ind,
                el.curr_step,
                self.config.perturb_scale,
                log_dct[f"static/noise/{ind}-l2"],
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
                if el.scheduler is not None:
                    self.reset_lr_schedule(el, prev_max_steps=prev_max_steps)
                    self.logger.info("Model %d lr schedule reset.", ind)
        if self.config.logger.use_wandb:
            wandb.log(log_dct)

    def run(self):
        if self.config.perturb_debug_dummy_run:
            return self.dummy_run()
        return super().run()

    def step_all_training_elements(self, batches):
        if self.global_step == self.config.perturb_step.get_step(self.steps_per_epoch):
            self.perturb_model()
        return super().step_all_training_elements(batches)

    def dummy_run(self):
        # for testing and debugging logreg, this avoids having to do multiple training runs
        if self.config.logger.use_wandb:
            # randomly draw from uniform [0, perturb_step)
            value = torch.rand(1).item() * self.config.perturb_scale
            wandb.log({"test/dummyvalue": value})
            wandb.log({"test/dummyvalue": value})
