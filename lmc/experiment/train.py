import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch
from torchmetrics import SQuAD
from tqdm import tqdm

import wandb
from lmc.data.data_stats import TaskType
from lmc.data.math_datasets import MathMetricsEvaluator
from lmc.experiment.base import ExperimentManager
from lmc.experiment_config import Trainer
from lmc.logging.wandb_registry import WandbMetricsRegistry
from lmc.utils.cka import evaluate_cka, evaluate_ensemble
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
        self.max_steps = self.training_elements.max_steps

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
        if (
            self.config.data.task_type == TaskType.GENERATION
            and self.config.data.dataset in ["gsm8k", "math", "mathqa", "asdiv"]
        ):
            self.math_evaluator = MathMetricsEvaluator(self.config.data.dataset)

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
        self.training_elements.on_epoch_start()
        if self.training_elements.is_same_model():
            self.logger.info("Models are the same at initialization.")
        # don't eval/save if advancing to start_step
        if self.start_step == 0:
            self.eval_and_save()

    def on_epoch_start(self):
        self.training_elements.on_epoch_start()

    def on_epoch_end(self):
        self.ep += 1

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
                    self.step_all_training_elements(batches)
            self.on_epoch_end()
        self.on_train_end()

    def evaluate_element(self, element: TrainingElement, i):
        log_dct = {f"step/model{i}": element.curr_step}
        log_dct.update(element.dist_from_init())
        for next_el_ind in range(i, self.config.n_models):
            next_el = self.training_elements[next_el_ind]
            log_dct.update(element.dist_from_element(next_el))
            if self.config.cka_n_train and self.config.n_models > 1:
                log_dct.update(
                    evaluate_cka(
                        element, next_el, train=True, n_examples=self.config.cka_n_train
                    )
                )
            if self.config.cka_n_test and self.config.n_models > 1:
                log_dct.update(
                    evaluate_cka(
                        element, next_el, train=False, n_examples=self.config.cka_n_test
                    )
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
            if not self.config.data.is_language_dataset():
                log_dct.update(evaluate_ensemble(self.training_elements, train=True))
                log_dct.update(evaluate_ensemble(self.training_elements, train=False))
        if self.config.n_models > 2:
            evaluate_merge(
                self.training_elements,
                self.config,
                log_dct,
            )
        return log_dct

    def eval_and_save(self):
        log_dct = {}
        for i, element in enumerate(self.training_elements, start=1):
            if element.curr_step >= element.max_steps:
                continue
            if element.curr_step in self.eval_steps:
                log_dct.update(self.evaluate_element(element, i))
                # log lr, batch hashes of last step
                log_dct.update(element.get_step_snapshot())
            if element.curr_step in self.save_steps:
                element.save(self.steps_per_epoch)
        if self.global_step in self.lmc_steps:
            log_dct.update(self.evaluate_lmc())
        # print summary if log_dct is not empty
        if self.config.logger.print_summary and log_dct:
            report_results(log_dct, self.ep, self.config.n_models)
        return log_dct

    def on_train_end(self):
        # eval always happens on the last step
        log_dct = {"step/epoch": self.ep, "step/global": self.global_step}
        for i, element in enumerate(self.training_elements, start=1):
            # log any training metrics that haven't been logged since the end of the last epoch
            if self.global_step % self.steps_per_epoch != 0:
                log_dct.update(element.log_train_metrics())
            log_dct.update(self.evaluate_element(element, i))
            element.save(self.steps_per_epoch)
        if self.config.lmc.lmc_on_train_end:
            log_dct.update(self.evaluate_lmc())
        if self.config.logger.use_wandb:
            wandb.log(log_dct)

    def advance_step_without_training(self):
        self.global_step += 1
        for i, element in enumerate(self.training_elements, start=1):
            if element.curr_step >= element.max_steps:
                continue
            element.curr_step += 1

    def step_all_training_elements(self, batches):
        # go through each training element
        self.global_step += 1
        log_dct = {}
        first_batch = None
        for i, (batch, element) in enumerate(
            zip(batches, self.training_elements), start=1
        ):
            if element.curr_step >= element.max_steps:
                continue

            # tie all batches together until perturb_use_dataloader1_to_step - this allows implementing of parent-child spawning experiment (Frankle et al. 2020)
            first_batch = batch if first_batch is None else first_batch
            # allow negative indices (i.e. -1 means to last training step)
            if element.curr_step < (
                self.config.perturb_use_dataloader1_to_step % (element.max_steps + 1)
            ):
                batch = [
                    x.detach().clone() if isinstance(x, torch.Tensor) else deepcopy(x)
                    for x in first_batch
                ]

            # train
            element.step(batch)
            # if at end of batch, log lr, batch hashes, training metrics
            if self.global_step % self.steps_per_epoch == 0:
                log_dct.update(element.log_train_metrics())
                # log lr, batch hashes of last step
                log_dct.update(element.get_step_snapshot())
            # log lr, batch hashes, every step of first epoch (useful for sanity checking/debugging)
            if element.curr_step < self.steps_per_epoch:
                log_dct.update(element.get_step_snapshot())
        # log all of the info together at once
        log_dct.update(self.eval_and_save())
        if self.config.logger.use_wandb and log_dct:
            log_dct.update({"step/epoch": self.ep, "step/global": self.global_step})
            wandb.log(log_dct)

    def _eval_vision(self, element, model_idx: int) -> Dict[str, float]:
        """Vision-specific evaluation logging"""
        element.model.eval()

        # Run test evaluation
        with torch.no_grad():
            test_res = self._test_vision(
                element.model, element.test_loader, element.test_iterator
            )
            log_dct = {
                f"model{model_idx}/test/accuracy": test_res["accuracy"],
                f"model{model_idx}/test/top_3_accuracy": test_res["top_3_accuracy"],
                f"model{model_idx}/test/cross_entropy": test_res["cross_entropy"],
            }

            self._handle_best_checkpoint(element, test_res["accuracy"])
        return log_dct

    def _eval_language(
        self, element: TrainingElement, model_idx: int
    ) -> Dict[str, float]:
        element.model.eval()
        """Language-specific evaluation logging"""
        # Base metrics all tasks have
        # Run test evaluation
        ##TODO:  math
        with torch.no_grad():
            test_res = self._test_language(
                element.model, element.test_loader, element.test_iterator
            )
            log_dct = {f"model{model_idx}/test/{k}": v for k, v in test_res.items()}

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
                element.save(self.steps_per_epoch, save_name="best.ckpt")

    @torch.no_grad()
    def _test_segmentation(self, model, loader, iterator: tqdm):
        """Segmentation-specific testing"""
        # Metrics for segmentation: pixel accuracy, mean IoU, and cross-entropy loss
        pixel_acc, mean_iou, cross_entropy = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )

        # For per-class IoU calculation
        num_classes = getattr(model, "num_classes", 150)
        intersection_sum = torch.zeros(num_classes, device=self.device)
        union_sum = torch.zeros(num_classes, device=self.device)
        target_sum = torch.zeros(num_classes, device=self.device)

        iterator.reset()
        for images, masks in loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            iterator.update()

            # Forward pass
            outputs = model(pixel_values=images)

            # Handle different output formats
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
            # import code

            # code.interact(local=locals() | globals())
            # Resize logits to match mask size if needed
            if logits.shape[2:] != masks.shape[1:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[1:], mode="bilinear", align_corners=False
                )

            # Compute loss
            loss = self.test_loss_fn(logits, masks)
            cross_entropy.update(loss.item(), images.shape[0])

            # Get predictions
            preds = torch.argmax(logits, dim=1)

            # Calculate pixel accuracy
            valid_mask = masks != 255  # Ignore index for unlabeled pixels
            correct = ((preds == masks) & valid_mask).sum().item()
            total = valid_mask.sum().item()
            pixel_acc.update(correct / max(total, 1), images.shape[0])

            # Calculate IoU metrics
            for cls in range(num_classes):
                pred_inds = preds == cls
                target_inds = masks == cls

                intersection = (pred_inds & target_inds).sum()
                union = (pred_inds | target_inds).sum()
                target_area = target_inds.sum()

                intersection_sum[cls] += intersection
                union_sum[cls] += union
                target_sum[cls] += target_area

        # Calculate mean IoU from accumulated values
        valid_classes = target_sum > 0
        if valid_classes.sum() > 0:
            class_iou = intersection_sum[valid_classes] / (
                union_sum[valid_classes] + 1e-10
            )
            mean_iou_value = class_iou.mean().item()
        else:
            mean_iou_value = 0.0

        mean_iou.update(mean_iou_value, 1)

        # Build results dictionary
        res = {
            "cross_entropy": cross_entropy.get_avg(percentage=False),
            "pixel_accuracy": pixel_acc.get_avg(percentage=False),
            "mean_iou": mean_iou.get_avg(percentage=False),
        }

        # Add per-class IoU if needed
        if hasattr(self, "log_per_class_iou") and self.log_per_class_iou:
            for cls in range(num_classes):
                if target_sum[cls] > 0:
                    iou = (intersection_sum[cls] / (union_sum[cls] + 1e-10)).item()
                    res[f"iou_class_{cls}"] = iou

        iterator.set_postfix(res)
        iterator.refresh()
        return res

    @torch.no_grad()
    def _test_vision(self, model, loader, iterator: tqdm):
        """Vision-specific testing"""
        total_acc, total_topk, cross_entropy = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )
        if self.config.data.task_type == TaskType.SEGMENTATION:
            return self._test_segmentation(model, loader, iterator)
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
        for batch_idx, batch in enumerate(loader):
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
            metrics_kwargs["perplexity"] = torch.exp(outputs.loss).item()

            if self.config.data.task_type == TaskType.GENERATION:
                # Generate text for math reasoning tasks
                generated_outputs = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    # max_new_tokens=256,
                    # max_new_tokens=512,
                    # num_beams=5,  # Add beam search
                    temperature=0.7,  # Add some temperature
                    no_repeat_ngram_size=3,  # Prevent repetition
                    max_new_tokens=self.config.data.max_gen_seq_length,
                    pad_token_id=model.tokenizer.pad_token_id,
                    # eos_token_id=model.tokenizer.eos_token_id,
                    # do_sample=False,
                    do_sample=True,
                )
                # Clean the outputs during decoding
                predictions = [
                    self.math_evaluator.extract_answer(
                        self.math_evaluator.normalize_answer(pred)
                    )
                    for pred in model.generation_tokenizer.batch_decode(
                        generated_outputs,
                        skip_special_tokens=True,  # We'll clean manually
                    )
                ]

                # Clean the references
                labels = batch["labels"].clone()
                labels[labels == -100] = model.generation_tokenizer.pad_token_id

                references = [
                    self.math_evaluator.extract_answer(
                        self.math_evaluator.normalize_answer(ref)
                    )
                    for ref in model.generation_tokenizer.batch_decode(
                        labels,
                        skip_special_tokens=True,  # We'll clean manually
                    )
                ]
                math_metrics = self.math_evaluator.compute_metrics(
                    predictions,
                    references,
                    # generated_outputs[""]
                )

                predictions = outputs.logits

                metrics_kwargs.update({"accuracy": math_metrics["exact_match"]})

            # Update dataset-specific metrics
            elif dataset.metrics:
                if self.config.data.task_type == TaskType.REGRESSION:
                    predictions = outputs.logits
                else:
                    predictions = outputs.logits.argmax(1)

                metric_results = compute_metrics(
                    dataset.metrics,
                    predictions.detach(),
                    batch.get("labels").detach(),
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
