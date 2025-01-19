import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
from torchmetrics import SQuAD
from tqdm import tqdm

import wandb
from lmc.data.data_stats import TaskType
from lmc.experiment.base import ExperimentManager
from lmc.experiment_config import Trainer
from lmc.logging.wandb_registry import WandbMetricsRegistry
from lmc.utils.lmc_utils import check_lmc
from lmc.utils.metrics import AverageMeter, mixup_topk_accuracy, report_results
from lmc.utils.setup_training import (TrainingElements, save_model_opt,
                                      setup_experiment)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        self.steps_per_epoch = self.config.data.get_steps_per_epoch()
        self.max_epochs = self.training_elements.max_steps.get_epoch(
            self.steps_per_epoch
        )
        self.wandb_registry = WandbMetricsRegistry(self.config.n_models)

    def on_train_start(self):
        pass

    def on_epoch_start(self):
        pass
    
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

    def run(self):
        print(self.config.display)
        early_iter_ckpt_steps = self.get_early_iter_ckpt_steps(
            self.steps_per_epoch, n_ckpts=10
        )
        ep: int = 1
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
                self.global_step += 1
                for element_ind, batch in enumerate(batches):
                    element = self.training_elements[element_ind]
                    if element.curr_step >= element.max_steps.get_step(
                        self.steps_per_epoch
                    ):
                        break
                    element.train_iterator.update()
                        
                    self.step_element(
                        element,
                        batch,
                        ep,
                        early_iter_ckpt_steps,
                        i=element_ind + 1,
                    )

            self.on_epoch_end(ep, log_dct)
            ep += 1
        self.on_train_end(ep)

    def on_epoch_end(self, ep: int, log_dct: dict):
        log_dct["lr/global_step"] = self.global_step
        self.eval_epoch(
            ep,
            log_dct,
            self.global_step
            # self.steps_per_epoch * ep,
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


    def step_element(self, element, batch, ep, ckpt_steps: List[int] = [], i: int = 1):
        element.curr_step += 1
        # Get learning rate
        if element.scheduler is None:
            lr = element.opt.param_groups[0]["lr"]
        else:
            lr = element.scheduler.get_last_lr()[-1]
        
        if self.config.logger.use_wandb:
            wandb.log({f"lr/model{i}": lr, f"lr/step/model{i}": element.curr_step})
        if element.curr_step % self.config.trainer.gradient_accumulation_steps == 0:
            element.opt.zero_grad()
        if self.config.data.is_language_dataset():
            loss = self._step_element_language(element, batch)
        else:
            loss = self._step_element_vision(element, batch)
        
        # Save checkpoint if needed
        save = (
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
        return loss
    
    def _step_element_language(self, element, batch):
        # Pre-fetch next batch while computing current one
        batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass depends on task type
        if self.config.data.task_type == TaskType.GENERATION:
            # Language modeling
            outputs = element.model(**batch)
            loss = outputs.loss
            
        elif self.config.data.task_type in [TaskType.CLASSIFICATION, TaskType.NATURAL_LANGUAGE_INFERENCE]:
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
            
        else:
            raise ValueError(f"Unsupported task type: {self.config.data.task_type}")
    
        if element.curr_step % self.config.trainer.gradient_accumulation_steps == 0:
            loss.backward()
            if (clip_val := self.config.trainer.opt.gradient_clip_val):
                torch.nn.utils.clip_grad_norm_(
                element.model.parameters(), 
                clip_val
            )
            element.opt.step()
            if element.scheduler is not None:
                element.scheduler.step()

        # Update metrics based on task
        with torch.no_grad():
            metrics_kwargs = {"cross_entropy": loss.item(), "n": len(batch)}
            if self.config.data.task_type == TaskType.GENERATION:
                perplexity = torch.exp(loss)
                metrics_kwargs["perplexity"] = perplexity.item()
                    
            elif self.config.data.task_type in [TaskType.CLASSIFICATION, TaskType.NATURAL_LANGUAGE_INFERENCE]:
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
                start_correct = (start_pred == batch['start_positions']).float().mean()
                end_correct = (end_pred == batch['end_positions']).float().mean()
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
        #TODO: gradclipping

        element.opt.step()
        if element.scheduler is not None:
            element.scheduler.step()

        # update metrics
        acc, topk = mixup_topk_accuracy(
            out.detach(), y.detach(), targs_perm, k=3, avg=True
        )
        element.metrics.update(acc.item(), topk.item(), None, loss.item(), n=x.shape[0])
        # TODO: sosmething wrong here, saved ckpts are not in this list?
        save: bool = (
            element.save_freq_step
            and element.save_freq_step.modulo(
                element.curr_step, mode="st", steps_per_epoch=self.steps_per_epoch
            )
            == 0
        )
        return loss.detach()

    def eval_epoch(self, ep, log_dct, curr_step):
        for i, element in enumerate(self.training_elements, start=1):
            if curr_step > element.max_steps.get_step(self.steps_per_epoch):
                continue
                
            # Save checkpoint
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

            # Choose evaluation function based on task
            if self.config.data.is_language_dataset():
                log_dct.update(self._eval_language(element, i, ep))
            else:
                log_dct.update(self._eval_vision(element, i, ep))

    def _eval_vision(self, element, model_idx: int, ep: int) -> Dict[str, float]:
        """Vision-specific evaluation logging"""
        log_dct = {
            f"model{model_idx}/train/cross_entropy": element.metrics.cross_entropy.get_avg(
                percentage=False
            ),
            f"model{model_idx}/train/accuracy": element.metrics.total_acc.get_avg(
                percentage=False
            ),
            f"model{model_idx}/train/top_3_accuracy": element.metrics.total_topk.get_avg(
                percentage=False
            ),
        }

        # Run test evaluation
        with torch.no_grad():
            test_res = self._test_vision(
                element.model, element.test_loader, element.test_iterator
            )
            log_dct.update({
                f"model{model_idx}/test/accuracy": test_res["accuracy"],
                f"model{model_idx}/test/top_3_accuracy": test_res["top_3_accuracy"],
                f"model{model_idx}/test/cross_entropy": test_res["cross_entropy"],
            })

            self._handle_best_checkpoint(element, test_res["accuracy"], ep)
            
        return log_dct

    def _eval_language(self, element, model_idx: int, ep: int) -> Dict[str, float]:
        """Language-specific evaluation logging"""
        task_type = self.config.data.task_type
        # import code; code.interact(local=locals()|globals())
        # Base metrics all tasks have
        log_dct = {
            f"model{model_idx}/train/cross_entropy": element.metrics.cross_entropy.get_avg(
                percentage=False
            )
        }
        
        # Add task-specific metrics
        if task_type in [TaskType.CLASSIFICATION, TaskType.NATURAL_LANGUAGE_INFERENCE]:
            log_dct.update({
                f"model{model_idx}/train/accuracy": element.metrics.total_acc.get_avg(
                    percentage=False
                )
            })
        elif task_type == TaskType.QUESTION_ANSWERING:
            log_dct.update({
                f"model{model_idx}/train/em": element.metrics.exact_match.get_avg(
                    percentage=False
                ),
                # todo: f1
                f"model{model_idx}/train/f1": element.metrics.f1_score.get_avg(
                    percentage=False
                ),
                f"model{model_idx}/train/top_3_accuracy": element.metrics.total_topk.get_avg(percentage=False)
            })
        elif task_type == TaskType.GENERATION:
            log_dct.update({
                f"model{model_idx}/train/perplexity": element.metrics.perplexity.get_avg(
                    percentage=False
                )
            })

        # Run test evaluation
        with torch.no_grad():
            test_res = self._test_language(
                element.model, element.test_loader, element.test_iterator
            )
            log_dct.update({
                f"model{model_idx}/test/{k}": v for k, v in test_res.items()
            })

            # Use appropriate metric for best checkpoint
            best_metric = (
                test_res.get("accuracy") or 
                test_res.get("f1") or 
                -test_res.get("loss", float('inf'))
            )
            self._handle_best_checkpoint(element, best_metric, ep)
            
        return log_dct

    def _handle_best_checkpoint(self, element, metric_value: float, ep: int):
        """Handle saving of best checkpoint"""
        if metric_value > element.optimal_acc:
            element.optimal_acc = metric_value
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
    def _test_vision(self, model, loader, iterator: tqdm):
        """Vision-specific testing"""
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

    @torch.no_grad()
    def _test_language(self, model, loader, iterator: tqdm):
        """Language-specific testing"""
        model.eval()
        task_type = self.config.data.task_type
        metrics = {
            "cross_entropy": AverageMeter(),
            "accuracy": AverageMeter(),
            "f1": AverageMeter(),
            "em": AverageMeter(),
            "perplexity": AverageMeter()
        }

        iterator.reset()
        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            iterator.update()
            
            outputs = model(**batch)
            n = batch['input_ids'].shape[0]
            
            # Update task-specific metrics
            if task_type in [TaskType.CLASSIFICATION, TaskType.NATURAL_LANGUAGE_INFERENCE]:
                preds = torch.argmax(outputs.logits, dim=-1)
                acc = (preds == batch['labels']).float().mean()
                metrics['accuracy'].update(acc.item(), n)
                
            elif task_type == TaskType.QUESTION_ANSWERING:
                # Calculate EM and F1
                squad = SQuAD()
                # squad_res = squad(outputs, batch[])
                # metrics["em"].update(squad_res["exact_match"].item())
                # metrics["f1"].update(squad_res["exact_match"].item())
                
            elif task_type == TaskType.GENERATION:
                loss = outputs.loss
                perplexity = torch.exp(loss)
                metrics['perplexity'].update(perplexity.item(), n)
            
            # Always track loss
            metrics["cross_entropy"].update(outputs.loss.item(), n)

        # Prepare results dict
        res = {k: v.get_avg(percentage=False) 
            for k, v in metrics.items() 
            if v.count > 0}  # Only include metrics that were updated
            
        iterator.set_postfix(res)
        iterator.refresh()
        return res

    @torch.no_grad()
    def test(self, model, loader, iterator: tqdm):
        
        # Choose evaluation function based on task
        if self.config.data.is_language_dataset():
            return self._test_language(model, loader, iterator)
        else:
            return self._test_vision(model, loader, iterator)
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
