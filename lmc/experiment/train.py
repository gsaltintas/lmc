import os
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import torch
from torchmetrics import SQuAD
from tqdm import tqdm

import wandb
from lmc.data.data_stats import TaskType
from lmc.experiment.base import ExperimentManager
from lmc.experiment_config import Trainer
from lmc.logging.wandb_registry import WandbMetricsRegistry
from lmc.utils.lmc_utils import check_lmc, evaluate_merge
from lmc.utils.metrics import (
    AverageMeter,
    Metrics,
    compute_metrics,
    mixup_topk_accuracy,
    report_results,
)
from lmc.utils.setup_training import setup_experiment
from lmc.utils.step import Step
from lmc.utils.training_element import TrainingElement, TrainingElements

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class TrainingRunner(ExperimentManager):
    config: Trainer = field(init=True, default=Trainer)
    _name: str = "trainer"

    training_elements: TrainingElements = None
    device: torch.device = None
    steps_per_epoch: int = None
    global_step: int = 0
    ep: int = 0
    start_step: int = 0

    @staticmethod
    def description():
        return "Train n model(s)."

    def setup(self) -> None:
        self.training_elements, self.device, self.start_step = setup_experiment(
            self.config
        )
        self.global_step = 0
        self.ep = 0
        self.steps_per_epoch = self.config.data.get_steps_per_epoch()
        self.max_steps = self.training_elements.max_steps.get_step(self.steps_per_epoch)

        self.eval_steps = self.get_steps(
            self.config.trainer.eval_freq, self.config.trainer.eval_specific_steps
        )
        self.save_steps = self.get_steps(
            self.config.trainer.save_freq, self.config.trainer.save_specific_steps
        )
        if self.config.trainer.save_early_iters:
            self.save_steps = self.save_steps.union(self.get_early_iter_ckpt_steps())
        # lmc_on_epoch_end is deprecated
        lmc_freq = (
            "1ep" if self.config.lmc.lmc_on_epoch_end else self.config.lmc.lmc_freq
        )
        self.lmc_steps = self.get_steps(lmc_freq, self.config.lmc.lmc_specific_steps)
        self.wandb_registry = WandbMetricsRegistry(self.config.n_models)

    def get_steps(self, freq, step_list):
        steps = set()
        if (
            freq is not None
            and freq != ""
            and freq.lower() != "none"
            and freq.lower() != "false"
        ):
            skip = Step.from_short_string(freq, self.steps_per_epoch).get_step()
            steps = set(range(0, self.max_steps, skip))
        for step in step_list.split(","):
            if step != "":
                steps.add(Step.from_short_string(step, self.steps_per_epoch).get_step())
        return steps

    def get_early_iter_ckpt_steps(self, n_ckpts: int = 10):
        """schedule for checkpoints"""
        first_epoch = np.concatenate(
            ([1, 2, 3, 4, 5], np.linspace(6, self.steps_per_epoch, n_ckpts))
        )
        later_epochs = np.concatenate(
            [
                np.linspace(
                    ep * self.steps_per_epoch, (ep + 1) * self.steps_per_epoch, n_ckpts
                )
                for ep in range(
                    1,
                    10,
                )
            ]
        )
        ckpts = np.concatenate((first_epoch, later_epochs)).astype(int)
        return ckpts

    def on_train_start(self):
        print(self.config.display)
        if self.training_elements.is_same_model():
            self.logger.info("Models are the same at initialization.")

    def on_epoch_start(self):
        self.training_elements.on_epoch_start()

    def on_epoch_end(self):
        self.ep += 1
        self.training_elements.on_epoch_end()

    def run(self):
        self.on_train_start()
        while self.global_step < self.max_steps:
            ### train epoch
            self.on_epoch_start()
            train_loaders = [iter(el.train_loader) for el in self.training_elements]
            for batch_ind, batches in enumerate(zip(*train_loaders)):
                if not (self.global_step < self.max_steps):
                    break
                if self.start_step > self.global_step:
                    # advance batches to bring dataloader state to start_step
                    self.advance_step_without_training()
                else:
                    self.save_all_training_elements()
                    self.step_all_training_elements(batches)
            self.on_epoch_end()
        self.on_train_end()

    def evaluate_element(self, element: TrainingElement, i):
        element.model.eval()
        log_dct = {f"step/model{i}": element.curr_step}
        log_dct[self.wandb_registry.get_metric(f"l2_dist_from_init_{i}").log_name] = (
            element.dist_from_init()
        )
        if (
            i < self.config.n_models
        ):  # i is index of element + 1, in range [1, n_models], so next element is at i in self.training_elements
            next_el = self.training_elements[i]
            log_dct[self.wandb_registry.get_metric(f"l2_dist_{i}-{i + 1}").log_name] = (
                element.dist_from_element(next_el)
            )
        # Choose evaluation function based on task
        if self.config.data.is_language_dataset():
            log_dct.update(self._eval_language(element, i))
        else:
            log_dct.update(self._eval_vision(element, i))
        return log_dct

    def evaluate_lmc(self):
        log_dct = {}
        if self.config.n_models > 1:
            check_lmc(
                self.training_elements,
                self.config,
                self.ep,
                log_dct,
                check_perms=self.config.lmc.lmc_check_perms,
            )
        if self.config.n_models > 2:
            evaluate_merge(
                self.training_elements,
                self.config,
                log_dct,
            )
        return log_dct

    def save_all_training_elements(self):
        # save checkpoints before doing anything to get a precise snapshot of this iteration
        for element in self.training_elements:
            if element.curr_step in self.save_steps:
                element.save(self.steps_per_epoch)

    def on_train_end(self):
        # eval always happens on the last step
        log_dct = {"step/epoch": self.ep, "step/global": self.global_step}
        for i, element in enumerate(self.training_elements, start=1):
            log_dct.update(self.evaluate_element(element, i))
            element.save(self.steps_per_epoch)
        if self.config.lmc.lmc_on_train_end:
            log_dct.update(self.evaluate_lmc())
        if self.config.logger.use_wandb:
            wandb.log(log_dct)

    def advance_step_without_training(self):
        self.global_step += 1
        for i, element in enumerate(self.training_elements, start=1):
            if element.curr_step >= element.max_steps.get_step(self.steps_per_epoch):
                continue
            element.curr_step += 1

    def step_all_training_elements(self, batches):
        log_dct = {}
        # Compute LMC stats
        if self.global_step in self.lmc_steps:
            log_dct.update(self.evaluate_lmc())
        # go through each training element
        step_dct = {"step/global": self.global_step}
        self.global_step += 1
        for i, (batch, element) in enumerate(
            zip(batches, self.training_elements), start=1
        ):
            if element.curr_step >= element.max_steps.get_step(self.steps_per_epoch):
                continue
            # evaluate
            if element.curr_step in self.eval_steps:
                log_dct.update(self.evaluate_element(element, i))
            # train
            step_dct.update(
                self.step_element(
                    element,
                    batch,
                    i=i,
                )
            )
            # evaluate
            if element.curr_step in self.eval_steps:
                log_dct.update(self.evaluate_element(element, i))
            # save checkpoint
            if element.curr_step in self.save_steps:
                element.save(self.steps_per_epoch)
        # print summary if log_dct is not empty
        if self.config.logger.print_summary and log_dct:
            log_dct["step/epoch"] = self.ep
            report_results(log_dct, self.ep, self.config.n_models)
        # log all of the info together at once
        log_dct.update(step_dct)
        if self.config.logger.use_wandb:
            wandb.log(log_dct)

    def step_element(self, element, batch, i: int = 1) -> Dict[str, Any]:
        element.model.train()
        log_dct = {f"step/model{i}": element.curr_step}
        element.train_iterator.update()
        element.curr_step += 1
        # Get learning rate
        if element.scheduler is None:
            lr = element.opt.param_groups[0]["lr"]
        else:
            lr = element.scheduler.get_last_lr()[-1]
        log_dct[f"lr/model{i}"] = lr

        if element.curr_step % self.config.trainer.gradient_accumulation_steps == 0:
            element.opt.zero_grad()
        if self.config.data.is_language_dataset():
            self._step_element_language(element, batch)
        else:
            self._step_element_vision(element, batch)

        return log_dct

    def _step_element_language(self, element, batch):
        # Pre-fetch next batch while computing current one
        batch = {
            k: v.to(self.device, non_blocking=True)
            if isinstance(v, torch.Tensor)
            else v
            for k, v in batch.items()
        }

        # Forward pass depends on task type
        if self.config.data.task_type == TaskType.GENERATION:
            # Language modeling
            outputs = element.model(**batch)
            loss = outputs.loss

        elif self.config.data.task_type in [
            TaskType.CLASSIFICATION,
            TaskType.NATURAL_LANGUAGE_INFERENCE,
            TaskType.SEQUENCE_PAIR,
        ]:
            # Classification tasks
            outputs = element.model(**batch)
            loss = outputs.loss
            logits = outputs.logits
        elif self.config.data.task_type == TaskType.QUESTION_ANSWERING:
            # Question answering
            outputs = element.model(**batch)
            loss = outputs.loss
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
        elif self.config.data.task_type == TaskType.REGRESSION:
            # Regression tasks (like STS-B)
            outputs = element.model(**batch)
            loss = outputs.loss
            logits = outputs.logits.squeeze(
                -1
            )  # Remove last dimension since it's regression
        else:
            raise ValueError(f"Unsupported task type: {self.config.data.task_type}")

        if element.curr_step % self.config.trainer.gradient_accumulation_steps == 0:
            loss.backward()
            if clip_val := self.config.trainer.opt.gradient_clip_val:
                torch.nn.utils.clip_grad_norm_(element.model.parameters(), clip_val)
            element.opt.step()
            if element.scheduler is not None:
                element.scheduler.step()

        # Update metrics based on task
        with torch.no_grad():
            metrics_kwargs = {"cross_entropy": loss.item(), "n": len(batch)}
            dataset = self.config.data.dataset_info
            if dataset.metrics:
                if self.config.data.task_type == TaskType.REGRESSION:
                    predictions = outputs.logits
                else:
                    predictions = outputs.logits.argmax(1)
                d = compute_metrics(
                    dataset.metrics, predictions.detach(), batch["labels"].detach()
                )
                metrics_kwargs.update(d)
            else:
                if self.config.data.task_type == TaskType.GENERATION:
                    perplexity = torch.exp(loss)
                    metrics_kwargs["perplexity"] = perplexity.item()

                elif self.config.data.task_type in [
                    TaskType.CLASSIFICATION,
                    TaskType.NATURAL_LANGUAGE_INFERENCE,
                ]:
                    acc, topk = mixup_topk_accuracy(
                        logits.detach(), batch["labels"].detach(), None, k=3, avg=True
                    )
                    metrics_kwargs["total_acc"] = acc.item()
                    metrics_kwargs["total_topk"] = topk.item()

                elif self.config.data.task_type == TaskType.QUESTION_ANSWERING:
                    # For QA, we typically track EM (Exact Match) and F1 score
                    # This would require post-processing with the tokenizer
                    start_pred = torch.argmax(start_logits, dim=-1)
                    end_pred = torch.argmax(end_logits, dim=-1)
                    # Simplified metric for training (just position accuracy)
                    start_correct = (
                        (start_pred == batch["start_positions"]).float().mean()
                    )
                    end_correct = (end_pred == batch["end_positions"]).float().mean()
                    avg_correct = (start_correct + end_correct) / 2
                    metrics_kwargs["accuracy"] = avg_correct.item()

        element.metrics.update(**metrics_kwargs)
        return loss.detach()

    def _step_element_vision(self, element, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        out = element.model(x)
        loss = self.loss_fn(out, y)
        targs_perm = None  # depreceated, when using mixup/cutmix
        loss.backward()
        # TODO: gradclipping

        element.opt.step()
        if element.scheduler is not None:
            element.scheduler.step()

        # update metrics
        acc, topk = mixup_topk_accuracy(
            out.detach(), y.detach(), targs_perm, k=3, avg=True
        )
        element.metrics.update(acc.item(), topk.item(), None, loss.item(), n=x.shape[0])
        return loss.detach()

    def _eval_vision(self, element, model_idx: int) -> Dict[str, float]:
        """Vision-specific evaluation logging"""

        log_dct = {
            f"model{model_idx}/train/{key}": val
            for (key, val) in element.metrics.get_metrics(percentage=False).items()
        }

        # Run test evaluation
        with torch.no_grad():
            test_res = self._test_vision(
                element.model, element.test_loader, element.test_iterator
            )
            log_dct.update(
                {
                    f"model{model_idx}/test/accuracy": test_res["accuracy"],
                    f"model{model_idx}/test/top_3_accuracy": test_res["top_3_accuracy"],
                    f"model{model_idx}/test/cross_entropy": test_res["cross_entropy"],
                }
            )

            self._handle_best_checkpoint(element, test_res["accuracy"])
        return log_dct

    def _eval_language(
        self, element: TrainingElement, model_idx: int
    ) -> Dict[str, float]:
        """Language-specific evaluation logging"""
        # Base metrics all tasks have
        ## log train metrics
        log_dct = {
            f"model{model_idx}/train/{key}": val
            for (key, val) in element.metrics.get_metrics(percentage=False).items()
        }
        # Run test evaluation
        with torch.no_grad():
            test_res = self._test_language(
                element.model, element.test_loader, element.test_iterator
            )
            log_dct.update(
                {f"model{model_idx}/test/{k}": v for k, v in test_res.items()}
            )

            # Use appropriate metric for best checkpoint
            best_metric = (
                test_res.get("accuracy")
                or test_res.get("f1")
                or -test_res.get("loss", float("inf"))
            )
            self._handle_best_checkpoint(element, best_metric)
        return log_dct

    def _handle_best_checkpoint(self, element, metric_value: float):
        """Handle saving of best checkpoint"""
        if metric_value > element.optimal_acc:
            element.optimal_acc = metric_value
            if self.config.trainer.save_best:
                self.logger.info("Saving best params at epoch %d", self.ep)
                if element.optimal_path is not None:
                    element.optimal_path.unlink()
                element.save(self.steps_per_epoch, save_name="best.ckpt")

    @torch.no_grad()
    def _test_vision(self, model, loader, iterator: tqdm):
        """Vision-specific testing"""
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

    @torch.no_grad()
    def _test_language(self, model, loader, iterator: tqdm):
        """Language-specific testing"""
        dataset = self.config.data.dataset_info

        # Initialize metrics dictionary with cross_entropy
        metrics = Metrics()

        iterator.reset()
        for batch in loader:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            iterator.update()

            outputs = model(**batch)
            n = batch["input_ids"].shape[0]
            metrics_kwargs = {"n": n}
            # Always track loss
            metrics_kwargs["cross_entropy"] = outputs.loss.item()

            # Update dataset-specific metrics
            if dataset.metrics:
                if self.config.data.task_type == TaskType.REGRESSION:
                    predictions = outputs.logits
                else:
                    predictions = outputs.logits.argmax(1)

                metric_results = compute_metrics(
                    dataset.metrics,
                    predictions.detach(),
                    batch["labels"].detach(),
                )
                metrics_kwargs.update(metric_results)
            metrics.update(**metrics_kwargs)
        # Prepare results dict - only include metrics that were updated
        res = metrics.get_metrics(percentage=False)

        iterator.set_postfix(res)
        iterator.refresh()
        return res

    @torch.no_grad()
    def _test_language_old(self, model, loader, iterator: tqdm):
        """Language-specific testing"""
        model.eval()
        task_type = self.config.data.task_type
        metrics = {
            "cross_entropy": AverageMeter(),
            "accuracy": AverageMeter(),
            "f1": AverageMeter(),
            "em": AverageMeter(),
            "perplexity": AverageMeter(),
        }

        iterator.reset()
        for batch in loader:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            iterator.update()

            outputs = model(**batch)
            n = batch["input_ids"].shape[0]

            # Update task-specific metrics
            if task_type in [
                TaskType.CLASSIFICATION,
                TaskType.NATURAL_LANGUAGE_INFERENCE,
            ]:
                preds = torch.argmax(outputs.logits, dim=-1)
                acc = (preds == batch["labels"]).float().mean()
                metrics["accuracy"].update(acc.item(), n)

            elif task_type == TaskType.QUESTION_ANSWERING:
                # Calculate EM and F1
                squad = SQuAD()
                # squad_res = squad(outputs, batch[])
                # metrics["em"].update(squad_res["exact_match"].item())
                # metrics["f1"].update(squad_res["exact_match"].item())

            elif task_type == TaskType.GENERATION:
                loss = outputs.loss
                perplexity = torch.exp(loss)
                metrics["perplexity"].update(perplexity.item(), n)

            # Always track loss
            metrics["cross_entropy"].update(outputs.loss.item(), n)

        # Prepare results dict
        res = {
            k: v.get_avg(percentage=False) for k, v in metrics.items() if v.count > 0
        }  # Only include metrics that were updated

        iterator.set_postfix(res)
        iterator.refresh()
        return res

    @torch.no_grad()
    def test(self, model, loader, iterator: tqdm):
        model.eval()
        # Choose evaluation function based on task
        if self.config.data.is_language_dataset():
            return self._test_language(model, loader, iterator)
        else:
            return self._test_vision(model, loader, iterator)
