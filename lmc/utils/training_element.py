from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from lmc.utils.metrics import Metrics
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
class TrainingElement(object):
    """dataclass holding everything pertaining to the training elements, models, loaders, optimizers steps, etc."""

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
    model_dir: Path = None
    train_iterator: tqdm = Iterator()
    test_iterator: tqdm = Iterator()
    train_eval_iterator: tqdm = Iterator()
    extra_iterator: tqdm = Iterator()
    metrics: Metrics = field(init=True, default_factory=Metrics)
    # TODO: later add the loss func for nlp models
    loss_fn: callable = nn.CrossEntropyLoss()
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
            self.model_dir / "checkpoints" / save_name,
            pickle_protocol=4,
        )


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
