import math
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from tqdm import tqdm
import wandb

from lmc.data.data_stats import SAMPLE_DICT
from lmc.experiment_config import Trainer
from lmc.utils.lmc_utils import check_lmc
from lmc.utils.metrics import AverageMeter, mixup_topk_accuracy, report_results
from lmc.utils.setup_training import TrainingElements, save_model_opt, setup_experiment
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
        print(self.config.display)
        early_iter_ckpt_steps = self.get_early_iter_ckpt_steps(
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
                    self.step_element(
                        element,
                        x,
                        y,
                        ep,
                        early_iter_ckpt_steps,
                        i=element_ind + 1,
                    )

            self.on_epoch_end(ep, log_dct)

    def on_epoch_end(self, ep: int, log_dct: dict):
        log_dct["lr/global_step"] = self.global_step
        self.eval_epoch(
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

    def get_early_iter_ckpt_steps(self, steps_per_epoch: int, n_ckpts: int = 10):
        """schedule for checkpoints"""
        first_epoch = np.concatenate(
            ([1, 2, 3, 4, 5], np.linspace(6, steps_per_epoch, n_ckpts))
        )
        later_epochs = np.concatenate(
            [
                np.linspace(ep * steps_per_epoch, (ep + 1) * steps_per_epoch, n_ckpts)
                for ep in range(
                    1,
                    10,
                )
            ]
        )
        ckpts = np.concatenate((first_epoch, later_epochs)).astype(int)
        return ckpts

    def step_element(self, element, x, y, ep, ckpt_steps: List[int] = [], i: int = 1):
        element.curr_step += 1
        # stop runs if nan
        if element.scheduler is None:
            lr = element.opt.param_groups[0]["lr"]
        else:
            lr = element.scheduler.get_last_lr()[-1]
        if self.config.logger.use_wandb:
            wandb.log({f"lr/model{i}": lr})
            wandb.log({f"lr/step/model{i}": element.curr_step})

        element.opt.zero_grad()
        x = x.to(self.device)
        y = y.to(self.device)
        out = element.model(x)
        loss = self.loss_fn(out, y)
        targs_perm = None  # depreceated, when using mixup/cutmix
        loss.backward()

        element.opt.step()
        if element.scheduler is not None:
            element.scheduler.step()

        # update metrics
        acc, topk = mixup_topk_accuracy(
            out.detach(), y.detach(), targs_perm, k=3, avg=True
        )
        element.metrics.update(acc.item(), topk.item(), None, loss.item(), x.shape[0])
        # TODO: sosmething wrong here, saved ckpts are not in this list?
        save: bool = (
            element.save_freq_step
            and element.save_freq_step.modulo(
                element.curr_step, mode="st", steps_per_epoch=self.steps_per_epoch
            )
            == 0
        )
        save = save or (
            self.config.trainer.save_early_iters and element.curr_step in ckpt_steps
        )
        if save:
            ckpt_name = f"checkpoints/ep-{ep}-st-{element.curr_step}.ckpt"
            save_model_opt(
                element.model,
                element.opt,
                element.model_dir.joinpath(ckpt_name),
                step=element.curr_step,
                epoch=ep,
                scheduler=element.scheduler,
            )

        return loss.detach()

    def eval_epoch(self, ep, log_dct, curr_step):
        for i, element in enumerate(self.training_elements, start=1):
            if curr_step > element.max_steps.get_step(self.steps_per_epoch):
                continue
            # save the end of epoch results
            ckpt_name = f"checkpoints/ep-{ep}.ckpt"
            save_model_opt(
                element.model,
                element.opt,
                element.model_dir.joinpath(ckpt_name),
                step=element.curr_step,
                epoch=ep,
                scheduler=element.scheduler,
            )
            element.model.eval()
            # logging
            log_dct.update(
                {
                    f"model{i}/train/cross_entropy": element.metrics.cross_entropy.get_avg(
                        percentage=False
                    ),
                    f"model{i}/train/accuracy": element.metrics.total_acc.get_avg(
                        percentage=False
                    ),
                    f"model{i}/train/top_3_accuracy": element.metrics.total_topk.get_avg(
                        percentage=False
                    ),
                }
            )
            # post epoch processing
            # element.train_iterator.set_postfix(double_res)
            # element.train_iterator.refresh()

            ### test
            with torch.no_grad():
                test_res = self.test(
                    element.model, element.test_loader, element.test_iterator
                )
                log_dct.update(
                    {
                        f"model{i}/test/accuracy": test_res["accuracy"],
                        f"model{i}/test/top_3_accuracy": test_res["top_3_accuracy"],
                        f"model{i}/test/cross_entropy": test_res["cross_entropy"],
                    }
                )

                if (test_acc := test_res["accuracy"]) > element.optimal_acc:
                    element.optimal_acc = test_acc
                    if self.config.trainer.save_best:
                        self.logger.info(f"Saving best params at epoch {ep}")
                        if element.optimal_path is not None:
                            element.optimal_path.unlink()
                        element.optimal_path = element.model_dir.joinpath(
                            "checkpoints", f"optimal_params-epoch-{ep}.ckpt"
                        )
                        save_model_opt(
                            element.model,
                            element.opt,
                            element.optimal_path,
                            epoch=ep,
                            scheduler=element.scheduler,
                        )

    @torch.no_grad()
    def test(self, model, loader, iterator: tqdm):
        model.eval()
        total_acc, total_topk, cross_entropy = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )

        iterator.reset()
        for ims, targs in loader:
            ims = ims.to(self.device)
            targs = targs.to(self.device)
            iterator.update()
            preds = model(ims)
            loss = self.test_loss_fn(preds, targs)

            cross_entropy.update(loss.item(), ims.shape[0])
            acc, topk = mixup_topk_accuracy(preds, targs, k=3, avg=True)
            total_acc.update(acc.item(), ims.shape[0])
            total_topk.update(topk.item(), ims.shape[0])

        res = {
            "cross_entropy": cross_entropy.get_avg(percentage=False),
            "accuracy": total_acc.get_avg(percentage=False),
            "top_3_accuracy": total_topk.get_avg(percentage=False),
        }
        iterator.set_postfix(res)
        iterator.refresh()
        return res
