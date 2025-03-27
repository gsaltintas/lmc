import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from lmc.data.data_stats import TaskType
from lmc.experiment_config import Trainer
from lmc.utils.metrics import (
    MathMetricsEvaluator,
    Metrics,
    compute_metrics,
    mixup_topk_accuracy,
)
from lmc.utils.step import Step

logger = logging.getLogger("setup")


def get_ckpts_by_step(ckpt_dir: Path, steps_per_epoch: int) -> OrderedDict[int, Path]:
    ckpts = []
    for ckpt in ckpt_dir.glob("*.ckpt"):
        if ckpt.stem != "best":
            step = Step.from_short_string(ckpt.stem, steps_per_epoch).get_step()
            ckpts.append((step, ckpt))
    sorted_ckpt_dict = OrderedDict()
    for k, v in sorted(ckpts, key=lambda x: x[0]):
        sorted_ckpt_dict[k] = v
    return sorted_ckpt_dict


def get_last_ckpt(ckpt_dir: Path, steps_per_epoch: int) -> Path:
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt
    ckpts = get_ckpts_by_step(ckpt_dir, steps_per_epoch)
    last_step = list(ckpts.keys())[-1]
    return ckpts[last_step]


def load_model_from_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    ignore_mismatched_sizes: bool = True,
) -> None:
    """
    Load weights from a checkpoint into a model, gracefully handling missing or mismatched keys.
    Similar to HuggingFace's approach, where missing keys retain their random initialization.

    Args:
        model: PyTorch model to load weights into
        path: Path to checkpoint file
        ignore_mismatched_sizes: Whether to skip loading weights with mismatched sizes
    """
    # Load checkpoint
    model_state_dict = model.state_dict()

    # Track different types of keys
    missing_keys = []
    unexpected_keys = []
    mismatched_keys = []
    to_be_loaded = OrderedDict()

    # Load matching keys
    for k, v in state_dict.items():
        if k not in model_state_dict:
            unexpected_keys.append(k)
            continue

        if v.shape != model_state_dict[k].shape:
            if ignore_mismatched_sizes:
                print(k)
                mismatched_keys.append(k)
                continue
            else:
                raise ValueError(
                    f"Size mismatch for {k}: checkpoint has {v.shape}, model has {model_state_dict[k].shape}"
                )

        to_be_loaded[k] = v

    # Identify missing keys
    missing_keys = [k for k in model_state_dict.keys() if k not in to_be_loaded]

    # Load the matching weights
    model.load_state_dict(to_be_loaded, strict=False)

    # Log informative messages
    if unexpected_keys:
        logger.info("Keys in checkpoint but not in model: %s", unexpected_keys)

    if mismatched_keys:
        logger.info("Keys skipped due to size mismatch: %s", mismatched_keys)

    if missing_keys:
        keys_str = ", ".join(missing_keys)
        logger.warning(
            "Some weights are newly initialized and should be trained: %s\n"
            "You should TRAIN these layers to ensure good performance!",
            keys_str,
        )


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
    element_ind: int
    device: torch.device
    max_steps: int
    train_loader: DataLoader
    train_eval_loader: DataLoader
    test_loader: DataLoader
    model: nn.Module
    opt: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler = None
    tokenizer: AutoTokenizer = None
    perturb_seed: int = None
    metrics: Metrics = field(init=True, default_factory=Metrics)
    loss_fn: callable = (
        nn.CrossEntropyLoss()
    )  # TODO: later add the loss func for nlp models

    ITERATOR_COLORS: Tuple[str] = ("#75507b", "#4f42b5", "#808080")

    def __post_init__(self):
        self.curr_step = 0
        self.optimal_acc: float = -1
        self.setup_iterators()
        # TODO if resume_from starts at nonzero step, init_model_vector is from current step, not 0
        params = [p.detach().cpu().contiguous() for p in self.model.parameters()]
        self.init_model_vector = nn.utils.parameters_to_vector(params)

    def setup_iterators(self):
        if self.config.logger.use_tqdm:
            color = self.ITERATOR_COLORS[self.element_ind % len(self.ITERATOR_COLORS)]
            self.train_iterator = tqdm(
                total=len(self.train_loader),
                desc=f"Training model {self.element_ind} - epoch: ",
                position=2 * self.element_ind,
                leave=True,
                # leave=False, disable=None,
                colour=color,
            )
            self.train_eval_iterator = tqdm(
                total=len(self.train_loader),
                desc=f"Evaluating model {self.element_ind} on train - epoch: ",
                position=2 + 2 * self.element_ind,
                leave=True,
                # leave=False, disable=None,
                colour=color,
            )
            self.test_iterator = tqdm(
                total=len(self.test_loader),
                desc=f"Evaluating model {self.element_ind} - epoch: ",
                position=1 + 2 * self.element_ind,
                leave=True,
                # leave=False, disable=None,
                colour=color,
            )
            self.extra_iterator = tqdm(
                position=2 + 2 * self.element_ind,
                desc="Extra iterator used for anything",
                colour="white",
            )
        else:
            self.train_iterator = Iterator()
            self.test_iterator = Iterator()
            self.train_eval_iterator = Iterator()
            self.extra_iterator = Iterator()

    def on_epoch_start(self):
        """call on epoch start to prepare for training the epoch"""
        self.opt.zero_grad()
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
        current_params = [
            p.detach().cpu().contiguous() for p in self.model.parameters()
        ]
        current_vector = torch.nn.utils.parameters_to_vector(current_params)
        dist = torch.linalg.norm(current_vector.detach().cpu() - self.init_model_vector)
        return dist.item()

    def dist_from_element(self, el: "TrainingElement") -> float:
        current_params = [
            p.detach().cpu().contiguous() for p in self.model.parameters()
        ]
        current_vector = torch.nn.utils.parameters_to_vector(current_params)
        other_params = [p.detach().cpu().contiguous() for p in el.model.parameters()]
        other_vector = torch.nn.utils.parameters_to_vector(other_params)
        dist = torch.linalg.norm(
            current_vector.detach().cpu() - other_vector.detach().cpu()
        )
        return dist.item()

    def log_train_metrics(self):
        log_dct = {
            f"model{self.element_ind}/train/{key}": val
            for (key, val) in self.metrics.get_metrics(percentage=False).items()
        }
        return log_dct

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
            self.config.model_dir
            / f"model{self.element_ind}"
            / "checkpoints"
            / save_name,
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

        # if self.curr_step % self.config.trainer.gradient_accumulation_steps == 0:
        #     self.opt.zero_grad()
        return log_dct


class VisionTrainingElement(TrainingElement):
    def step(self, batch):
        log_dct = super().step(batch)
        self.opt.zero_grad()
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


class SegmentationTrainingElement(VisionTrainingElement):
    def __init__(self, *args, ignore_index=255, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index

    def step(self, batch):
        log_dct = super(VisionTrainingElement, self).step(
            batch
        )  # Skip VisionTrainingElement's step
        self.opt.zero_grad()

        # Handle batch data for segmentation
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # Forward pass
        outputs = self.model(x)

        # Handle different output formats (some models return dict, others return logits directly)
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs
        import code

        code.interact(local=locals() | globals())

        # Resize logits to match mask size if needed
        if logits.shape[-2:] != y.shape[-2:]:
            logits = torch.nn.functional.interpolate(
                logits, size=y.shape[-2:], mode="bilinear", align_corners=False
            )

        # Calculate loss
        loss = self.loss_fn(logits, y)

        # Backward and optimize
        loss.backward()
        # TODO: grad clipping

        self.opt.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # Calculate segmentation metrics (instead of classification metrics)
        with torch.no_grad():
            # Get predictions
            preds = torch.argmax(logits, dim=1)

            # Calculate IoU
            intersection, union = self._calculate_iou(preds, y)
            miou = intersection.sum() / (union.sum() + 1e-10)

            # Calculate pixel accuracy
            mask = y != self.ignore_index
            correct = ((preds == y) & mask).sum()
            total = mask.sum()
            pixel_acc = correct / (total + 1e-10)

        # Update metrics with segmentation-specific values
        self.metrics.update(
            pixel_acc.item(), miou.item(), None, loss.item(), n=x.shape[0]
        )

        return log_dct

    def _calculate_iou(self, preds, targets):
        """Calculate intersection and union for IoU metric"""
        # Create mask to ignore certain pixels (typically 255)
        mask = targets != self.ignore_index

        # Get valid predictions and targets
        preds = preds[mask]
        targets = targets[mask]

        # Count unique classes in this batch
        num_classes = max(
            self.model.num_classes if hasattr(self.model, "num_classes") else 150,
            preds.max().item() + 1,
            targets.max().item() + 1,
        )

        # Calculate IoU for each class
        intersection = torch.zeros(1, device=preds.device)
        union = torch.zeros(1, device=preds.device)

        for cls in range(num_classes):
            pred_inds = preds == cls
            target_inds = targets == cls

            intersection += (pred_inds & target_inds).sum()
            union += (pred_inds | target_inds).sum()

        return intersection, union


class NLPTrainingElement(TrainingElement):
    def __post_init__(self):
        super().__post_init__()

        if (
            self.config.data.task_type == TaskType.GENERATION
            and self.config.data.dataset in ["gsm8k", "math", "mathqa", "asdiv"]
        ):
            self.math_evaluator = MathMetricsEvaluator()

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
        loss = loss / self.config.trainer.gradient_accumulation_steps
        loss.backward()

        if (self.curr_step + 1) % self.config.trainer.gradient_accumulation_steps == 0:
            # After accumulating gradients for config.trainer.gradient_accumulation_steps steps:
            if clip_val := self.config.trainer.opt.gradient_clip_val:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)
            self.opt.step()

            if self.scheduler is not None:
                self.scheduler.step()
            self.opt.zero_grad()  # Important: Clear gradients after optimizer step

        # Update metrics based on task
        with torch.no_grad():
            metrics_kwargs = {
                "cross_entropy": loss.item(),
                "n": len(batch),
                "perplexity": torch.exp(loss).item(),
            }
            dataset = self.config.data.dataset_info
            if self.config.data.task_type == TaskType.GENERATION:
                perplexity = torch.exp(loss)
                metrics_kwargs["perplexity"] = perplexity.item()
                #  Optionally track token-level accuracy if needed
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                token_acc = (shift_logits.argmax(-1) == shift_labels).float().mean()
                metrics_kwargs["accuracy"] = token_acc.item()
                # metrics_kwargs["token_accuracy"] = token_acc.item()

            elif dataset.metrics:
                if self.config.data.task_type == TaskType.REGRESSION:
                    predictions = outputs.logits
                else:
                    predictions = outputs.logits.argmax(1)
                d = compute_metrics(
                    dataset.metrics, predictions.detach(), batch["labels"].detach()
                )
                metrics_kwargs.update(d)
            else:
                if self.config.data.task_type in [
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


@dataclass(init=False)
class CheckpointEvaluationElement(TrainingElement):
    class DummyMetrics(Metrics):
        def get_metrics(self, percentage=False, task_type=TaskType.CLASSIFICATION):
            return {}  # checkpoints save no metrics during training

    def __init__(
        self,
        config: Trainer,
        element_ind: int,
        device: torch.device,
        max_steps: int,
        train_loader: DataLoader,
        train_eval_loader: DataLoader,
        test_loader: DataLoader,
        model: nn.Module,
        opt: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler = None,
        tokenizer: AutoTokenizer = None,
        perturb_seed: int = None,
        metrics: Metrics = field(init=True, default_factory=Metrics),
        loss_fn: callable = nn.CrossEntropyLoss(),  # TODO: later add the loss func for nlp models,
    ):
        self.config = config
        self.element_ind = element_ind
        self.device = device
        self.max_steps = max_steps
        self.train_loader = train_loader
        self.train_eval_loader = train_eval_loader
        self.test_loader = test_loader
        self._model = model  # save this way to access through getter
        self.scheduler = None
        self.tokenizer = tokenizer
        self.perturb_seed = None
        self.metrics = self.DummyMetrics()
        self.loss_fn = loss_fn

        self.loaded_model_step = None
        self.ckpt_dir = Path(getattr(config, f"evaluate_ckpt{self.element_ind}"))
        if (self.ckpt_dir / "checkpoints").exists():
            self.ckpt_dir = self.ckpt_dir / "checkpoints"
        steps_per_epoch = self.config.data.get_steps_per_epoch()
        self.ckpts = get_ckpts_by_step(self.ckpt_dir, steps_per_epoch)
        logger.info(
            f"model{self.element_ind}: using checkpoints for evaluation from {self.ckpt_dir} with steps {list(self.ckpts.keys())}"
        )
        self.__post_init__()

    def step(self, batch):
        self.train_iterator.update()
        self.curr_step += 1
        return {}

    def on_epoch_start(self):
        pass  # do nothing

    @property
    def model(self):
        if self.loaded_model_step != self.curr_step:
            if self.curr_step in self.ckpts:
                ckpt_path = self.ckpts[self.curr_step]
                ckpt = torch.load(ckpt_path, map_location=self.device)
                logger.info(
                    f"model{self.element_ind} loaded from checkpoint {ckpt_path}"
                )
                load_model_from_state_dict(self._model, ckpt["state_dict"])
                self.loaded_model_step = self.curr_step
            else:
                raise ValueError(
                    f"Checkpoint at step {self.curr_step} not found in {self.ckpt_dir}"
                )
        return self._model

    def save(self, steps_per_epoch, save_name=None):
        pass  # do nothing


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
        return max(el.max_steps for el in self._elements)

    def on_epoch_start(self):
        for el in self._elements:
            el.on_epoch_start()
            el.train_iterator.reset()
            el.train_iterator.set_description_str(
                f"Training model {el.element_ind} - epoch: "
            )

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

    def is_same_model(self):
        for element in self[1:]:
            if not self[0].params_equal(element):
                return False
        return True
