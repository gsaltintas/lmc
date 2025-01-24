from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from lmc.data.data_stats import TaskType
from lmc.experiment_config import Trainer
from lmc.utils.metrics import Metrics, compute_metrics, mixup_topk_accuracy
from lmc.utils.step import Step


class Iterator(tqdm):
    """dummy iterator class to use when tqdm is disabled"""

    def set_description_str(self, s):
        pass

    def update(self, n: float | None = 1) -> bool | None:
        pass

    def reset(self, total: float | None = None) -> None:
        pass

    def set_postfix(
        self,
        ordered_dict: Mapping[str, object] | None = None,
        refresh: bool | None = True,
        **kwargs,
    ) -> None:
        pass

    def refresh(
        self,
        nolock: bool = False,
        lock_args: tuple[bool | None, float | None] | tuple[bool | None] | None = None,
    ) -> None:
        pass


@dataclass
class TrainingElement(ABC):
    """dataclass holding everything pertaining to the training elements, models, loaders, optimizers steps, etc."""

    config: Trainer
    model: nn.Module
    opt: optim.Optimizer
    train_loader: DataLoader
    train_eval_loader: DataLoader
    test_loader: DataLoader
    element_ind: int = None
    scheduler: optim.lr_scheduler.LRScheduler = None
    seed: int = 42
    loader_seed: int = 42
    aug_seed: int = 42
    perturb_seed: int = None
    optimal_acc: float = -1
    optimal_path: Path = None
    permutation = None
    prev_perm_wm = None
    prev_perm_am = None
    max_steps: Step = None
    curr_step: int = 0  # not sure if this is the best way?
    train_iterator: tqdm = Iterator()
    test_iterator: tqdm = Iterator()
    train_eval_iterator: tqdm = Iterator()
    extra_iterator: tqdm = Iterator()
    metrics: Metrics = field(init=True, default_factory=Metrics)
    # TODO: later add the loss func for nlp models
    loss_fn: callable = nn.CrossEntropyLoss()
    device: torch.device = None
    tokenizer: AutoTokenizer = None
    init_model_vector: torch.Tensor = None

    def on_epoch_start(self):
        """call on epoch start to prepare for training the epoch"""
        self.opt.zero_grad()
        self.metrics.reset()

    def on_epoch_end(self):
        """call on epoch end to prepare for the evaluations"""
        self.metrics.reset()

    def params_equal(self, other: "TrainingElement"):
        for (n1, p1), (n2, p2) in zip(
            self.model.named_parameters(),
            other.model.named_parameters(),
        ):
            if not torch.allclose(p1, p2):
                return False
        return True

    def dist_from_init(self) -> float:
        current_vector = torch.nn.utils.parameters_to_vector(self.model.parameters())
        dist = torch.linalg.norm(current_vector.detach().cpu() - self.init_model_vector)
        return dist.item()

    def dist_from_element(self, el: "TrainingElement") -> float:
        current_vector = torch.nn.utils.parameters_to_vector(self.model.parameters())
        other_vector = torch.nn.utils.parameters_to_vector(el.model.parameters())
        dist = torch.linalg.norm(
            current_vector.detach().cpu() - other_vector.detach().cpu()
        )
        return dist.item()

    def save(self, steps_per_epoch, save_name=None):
        """Saves the model state, optimizer and scheduler state along with epoch."""
        step = Step(self.curr_step, steps_per_epoch)
        ep, st = step.get_epoch_step_pair()
        if save_name is None:
            save_name = f"{step.to_short_string()}.ckpt"
        self.model.eval()
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict()
                if self.scheduler is not None
                else None,
                "epoch": ep,
                "step": st,
            },
            self.config.model_dir / f"model{self.element_ind}" / "checkpoints" / save_name,
            pickle_protocol=4,
        )

    @abstractmethod
    def step(self, batch) -> Dict[str, Any]:
        self.model.train()
        log_dct = {f"step/model{self.element_ind}": self.curr_step}
        self.train_iterator.update()
        self.curr_step += 1
        # Get learning rate
        if self.scheduler is None:
            lr = self.opt.param_groups[0]["lr"]
        else:
            lr = self.scheduler.get_last_lr()[-1]
        log_dct[f"lr/model{self.element_ind}"] = lr

        if self.curr_step % self.config.trainer.gradient_accumulation_steps == 0:
            self.opt.zero_grad()
        return log_dct


class VisionTrainingElement(TrainingElement):
    def step(self, batch):
        log_dct = super().step(batch)
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        out = self.model(x)
        loss = self.loss_fn(out, y)
        targs_perm = None  # depreceated, when using mixup/cutmix
        loss.backward()
        # TODO: gradclipping

        self.opt.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # update metrics
        acc, topk = mixup_topk_accuracy(
            out.detach(), y.detach(), targs_perm, k=3, avg=True
        )
        self.metrics.update(acc.item(), topk.item(), None, loss.item(), n=x.shape[0])
        return log_dct


class NLPTrainingElement(TrainingElement):
    def step(self, batch):
        log_dct = super().step(batch)
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
            outputs = self.model(**batch)
            loss = outputs.loss

        elif self.config.data.task_type in [
            TaskType.CLASSIFICATION,
            TaskType.NATURAL_LANGUAGE_INFERENCE,
            TaskType.SEQUENCE_PAIR,
        ]:
            # Classification tasks
            outputs = self.model(**batch)
            loss = outputs.loss
            logits = outputs.logits
        elif self.config.data.task_type == TaskType.QUESTION_ANSWERING:
            # Question answering
            outputs = self.model(**batch)
            loss = outputs.loss
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
        elif self.config.data.task_type == TaskType.REGRESSION:
            # Regression tasks (like STS-B)
            outputs = self.model(**batch)
            loss = outputs.loss
            logits = outputs.logits.squeeze(
                -1
            )  # Remove last dimension since it's regression
        else:
            raise ValueError(f"Unsupported task type: {self.config.data.task_type}")

        if self.curr_step % self.config.trainer.gradient_accumulation_steps == 0:
            loss.backward()
            if clip_val := self.config.trainer.opt.gradient_clip_val:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)
            self.opt.step()
            if self.scheduler is not None:
                self.scheduler.step()

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

        self.metrics.update(**metrics_kwargs)
        return log_dct


class TrainingElements(object):
    """container for training elements"""

    _elements: List[TrainingElement]

    def __init__(self, *elements: TrainingElement):
        self._elements = []
        cnt = 0
        for i, el in enumerate(elements):
            self._elements.append(el)
            setattr(self, str(i), el)
            cnt += 0

        self.n_elements = cnt

    @property
    def count(self):
        return len(self._elements)

    def add_element(self, element: TrainingElement):
        self.n_elements += 1
        setattr(self, str(self.n_elements), element)
        self._elements.append(element)

    def __dict__(self):
        return {i: getattr(self, i) for i in range(self.n_elements)}

    @property
    def max_steps(self) -> Step:
        max_step = None
        for el in self._elements:
            if max_step is None:
                max_step = el.max_steps
                continue
            if max_step.get_step() < el.max_steps.get_step():
                max_step = el.max_steps
        return max_step

    def on_epoch_start(self):
        for el in self._elements:
            el.on_epoch_start()
            el.train_iterator.reset()
            el.train_iterator.set_description_str(
                f"Training model {el.element_ind} - epoch: "
            )

    def on_epoch_end(self):
        for el in self._elements:
            el.on_epoch_end()

    def __iter__(self):
        for el in self._elements:
            yield (el)

    def __getitem__(self, i: int):
        return self._elements[i]

    def __len__(self):
        return len(self._elements)

    def is_same_model(self):
        for element in self[1:]:
            if not self[0].params_equal(element):
                return False
        return True
