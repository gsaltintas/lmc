import logging
import warnings
from collections import OrderedDict, defaultdict
from typing import Dict, Generator, Iterable, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from repsim.metrics import AngularCKA

from lmc.data.data_stats import TaskType
from lmc.utils.lmc_utils import evaluate_model_language, evaluate_model_vision
from lmc.utils.training_element import TrainingElement

logger = logging.getLogger("cka")


def evaluate_cka(
    el_1: TrainingElement,
    el_2: TrainingElement,
    train=False,
    n_examples=-1,
    is_language: bool = False,
):
    log_dct = {}
    config = el_1.config
    include = [x for x in config.cka_include.split(",") if len(x) > 0]
    exclude = [x for x in config.cka_exclude.split(",") if len(x) > 0]
    print(include, exclude)
    dataloader = el_1.train_eval_loader if train else el_1.test_loader

    device = el_1.device
    split = "train" if train else "test"

    # get intermediate activations
    intermediates_1, outputs_1 = evaluate_model(
        el_1.model,
        dataloader,
        include=include,
        exclude=exclude,
        device=device,
        n_examples=n_examples,
        is_language=is_language,
    )
    intermediates_2, outputs_2 = evaluate_model(
        el_2.model,
        dataloader,
        include=include,
        exclude=exclude,
        device=device,
        n_examples=n_examples,
        is_language=is_language,
    )
    n_examples = outputs_1.shape[0]

    # compute CKA
    for k in intermediates_1.keys():
        v_1 = intermediates_1[k].reshape(n_examples, -1).to(torch.float32)
        v_2 = intermediates_2[k].reshape(n_examples, -1).to(torch.float32)
        metric = AngularCKA(m=n_examples)
        dist = metric.length(
            metric.neural_data_to_point(v_1), metric.neural_data_to_point(v_2)
        )
        log_dct[f"cka/{el_1.element_ind}-{el_2.element_ind}/{split}/{k}"] = dist

    # compute difference in output in magnitude of error vector and classification disagreement
    log_dct[f"disagree/{split}/class"] = torch.sum(
        torch.not_equal(torch.argmax(outputs_1, dim=1), torch.argmax(outputs_2, dim=1))
    )
    log_dct[f"disagree/{split}/margin"] = torch.linalg.norm(
        torch.softmax(outputs_1, dim=1) - torch.softmax(outputs_2, dim=1)
    )

    return log_dct


def evaluate_ensemble(elements: List[TrainingElement], train=False):
    config = elements[0].config
    split = "train" if train else "test"
    dataloader = elements[0].train_eval_loader if train else elements[0].test_loader
    if elements[0].config.data.task_type == TaskType.GENERATION:
        # if elements[0].config.data.is_language_dataset():
        # TODO: later do this with ensembling over the vocab
        raise NotImplementedError(
            "model ensembling not implemented for generation tasks"
        )
    device = elements[0].device
    loss_fn = elements[0].loss_fn

    ensemble = EnsembleModel([el.model for el in elements], device)
    ensemble.eval()
    if elements[0].config.data.is_language_dataset():
        results = evaluate_model_language(
            ensemble, dataloader, elements[0].config.data, device=device
        )
    else:
        num_classes = config.data.get_num_labels()
        results = evaluate_model_vision(
            ensemble,
            dataloader,
            num_classes=num_classes,
            device=device,
            criterion=loss_fn,
        )
    log_dct = {
        f"ensemble/{split}/accuracy": results["acc"],
        f"ensemble/{split}/cross_entropy": results["ce"],
    }
    return log_dct


class EnsembleModel(nn.Module):
    def __init__(self, models: List[nn.Module], device):
        super().__init__()
        self.models = models
        self.device = device

    def eval(self):
        for model in self.models:
            model.eval()

    def __call__(self, x):
        # sequential processing
        logits = []
        for model in self.models:
            with torch.no_grad():
                logit = model(x)
                logits.append(logit.detach().cpu())  # Move to CPU immediately
                torch.cuda.empty_cache()  # Clear GPU cache

        # Stack and average on CPU, then move result back to GPU
        ensemble_logits = torch.mean(torch.stack(logits, dim=0), dim=0)
        return ensemble_logits.to(self.device)


"""
from fork of open_lth
"""


def is_identity(x: torch.tensor, y: torch.tensor):
    return len(x.flatten()) == len(y.flatten()) and torch.all(
        x.flatten() == y.flatten()
    )


def match_key(
    key: str, include: Optional[List[str]] = None, exclude: Optional[List[str]] = None
):
    if not key:
        return False
    if include:
        if not any(k in key for k in include):
            return False
    if exclude:
        if any(k in key for k in exclude):
            return False
    return True


class SaveIntermediateHook:
    def __init__(
        self,
        named_modules: Iterable[Tuple[str, nn.Module]],
        include: List[str] = None,
        exclude: List[str] = None,
        device="cpu",
        verbose=False,
        cka_strategy: Literal["last_element", "all", "pool"] = "last_element",
    ):
        """Get intermediate values (output of every module/layer)
        from a forward() pass, by hooking into nn.Module.

        This is a context manager which resets and removes all hooks on exit.
        Layers with identical intermediate values are ignored.

        Example usage:
            ```
            intermediates = SaveIntermediateHook(model.named_modules())
            for x in data:
                with intermediates as hidden:
                    model(x)
                    yield hidden  # do not place this outside of context manager, or else hidden will be reset!
            ```

        Args:
            named_modules (Iterable[Tuple[str, nn.Module]]): modules to add hook to.
            include (List[str], optional): If set, only include modules with names
                containing at least one of these patterns. Defaults to None.
            exclude (List[str], optional): If set, exclude any modules with names
                containing any of these patterns. Defaults to None.
            device (str, optional): Device to move intermediates to. Defaults to 'cpu'.
            verbose (bool, optional): Warn when two intermediates are identical and one is discarded.
                This occurs often when modules are nested. Defaults to False.
            cka_strategy(Literal["last_element", "all", "pool"], optional): Determines how to handle sequence dimension, last_element: evaluates cka on the last element of the sequence, all: flattens over the sequence dim., pool: averages over the sequence dim.
        """
        self.named_modules = list(named_modules)
        self.device = device
        self.include = include
        self.exclude = exclude
        self.verbose = verbose
        self.intermediates = OrderedDict()
        self.cka_strategy = cka_strategy
        logging.info("Registered hooks")

    def __enter__(self):
        self.module_names = OrderedDict()
        self.handles = []
        for name, module in self.named_modules:
            self.module_names[module] = name
            self.handles.append(module.register_forward_hook(self))
        return self.intermediates

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for handle in self.handles:
            handle.remove()
        self.intermediates = OrderedDict()

    def __call__(self, module, args, return_val):
        layer_name = self.module_names[module]
        for arg in args:
            if arg is None or arg is False:  # for transformers models
                continue
            self._add_if_missing(layer_name + ".in", arg)
        if isinstance(return_val, tuple):  # for transformers models
            return_val = return_val[0]

        from transformers.modeling_outputs import (
            BaseModelOutput,
            BaseModelOutputWithPooling,
            ImageClassifierOutput,
            ModelOutput,
            SequenceClassifierOutput,
            TokenClassifierOutput,
        )

        # Handle different output types
        if isinstance(return_val, ImageClassifierOutput):
            return_val = return_val.logits
        elif isinstance(return_val, (SequenceClassifierOutput, TokenClassifierOutput)):
            # For classification outputs, use logits
            return_val = return_val.logits
        elif isinstance(return_val, (BaseModelOutput, BaseModelOutputWithPooling)):
            # For base model outputs, use last_hidden_state
            return_val = return_val.last_hidden_state
        elif hasattr(return_val, "last_hidden_state"):
            # Fallback for other outputs with last_hidden_state
            return_val = return_val.last_hidden_state
        elif hasattr(return_val, "logits"):
            # Fallback for outputs with logits
            return_val = return_val.logits
        elif (
            hasattr(return_val, "hidden_states")
            and return_val.hidden_states is not None
        ):
            # Use the last hidden state from the tuple
            return_val = return_val.hidden_states[-1]
        self._add_if_missing(layer_name + ".out", return_val)

    def _add_if_missing(self, key, value):
        # copy to prevent value from changing in later operations
        if match_key(key, self.include, self.exclude):
            if value.ndim == 3 and self.cka_strategy == "last_element":
                # TODO: properly handle this
                value = value[:, -1, :].detach().clone().to(device=self.device)
            elif value.ndim == 3 and self.cka_strategy == "pool":
                value = value.mean(1).detach().clone().to(device=self.device)
            else:
                value = value.detach().clone().to(device=self.device)
            for k, v in self.intermediates.items():
                if is_identity(v, value):
                    if self.verbose:
                        warnings.warn(f"{key} and {k} are equal, omitting {key}")
                    return
            assert key not in self.intermediates, key
            self.intermediates[key] = value


@torch.no_grad()
def _evaluate_intermediates_language(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    intermediates,
    device: str = "cuda",
    verbose=False,
    n_examples=-1,
) -> Generator:
    total_processed = 0
    for i, batch in enumerate(dataloader):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        original_batch_size = batch["input_ids"].shape[0]
        if n_examples > 0 and total_processed >= n_examples:
            return

        with intermediates as hidden:
            outputs = model(**batch)
            yield None, hidden, outputs, None

        total_processed += original_batch_size


@torch.no_grad()
def _evaluate_intermediates_vision(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    intermediates,
    device: str = "cuda",
    verbose=False,
    n_examples=-1,
) -> Generator:
    total_processed = 0
    for i, (batch_examples, labels) in enumerate(dataloader):
        original_batch_size = len(labels)
        if n_examples > 0 and total_processed >= n_examples:
            return

        # Process full batch (original behavior)
        batch_size = len(labels)

        with intermediates as hidden:
            if verbose:
                print(f"...batch {i}, full batch size {batch_size}")

            batch_examples = batch_examples.to(device=device)
            labels = labels.to(device=device)

            output = model(batch_examples)

            yield batch_examples, hidden, output, labels

        total_processed += batch_size


@torch.no_grad()
def evaluate_intermediates(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    named_modules: Iterable[Tuple[str, nn.Module]] = None,
    include: List[str] = None,
    exclude: List[str] = None,
    verbose=False,
    n_examples=-1,
    is_language: bool = False,
) -> Generator:
    """Evaluate a model on a dataloader, returning inputs, intermediate values, outputs, and labels

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (torch.utils.data.DataLoader): Dataloader containing data to evaluate on.
        device (str, optional): Device to evaluate on. Defaults to "cuda".
        named_modules (Iterable[Tuple[str, nn.Module]], optional): If set,
            only get intermediates values from these modules,
            otherwise include all intermediates from model.named_modules(). Defaults to None.
        include (List[str], optional): If set, only include modules with names
            containing at least one of these patterns. Defaults to None.
        exclude (List[str], optional): If set, exclude any modules with names
            containing any of these patterns. Defaults to None.
        device (str, optional): Device to move intermediates to. Defaults to 'cpu'.
        verbose (bool, optional): Warn when two intermediates are identical and one is discarded.
            This occurs often when modules are nested. Defaults to False.
            If None, processes full batches. Defaults to None.

    Yields:
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
            a tuple for every batch containing (inputs, intermediate values, outputs, true labels)
    """

    if named_modules is None:
        named_modules = list(model.named_modules())
    if verbose:
        print(model, "MODULES", *[k for k, v in named_modules], sep="\n")
    model.to(device=device)
    model.eval()
    intermediates = SaveIntermediateHook(
        named_modules, include=include, exclude=exclude, device=device
    )
    if is_language:
        yield from _evaluate_intermediates_language(
            model,
            dataloader,
            intermediates,
            device,
            verbose,
            n_examples,
        )
    else:
        yield from _evaluate_intermediates_vision(
            model,
            dataloader,
            intermediates,
            device,
            verbose,
            n_examples,
        )


@torch.no_grad()
def combine_batches(
    eval_intermediates_generator: Generator, return_inputs: bool = False
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Combines batches from evaluate_intermediates().

    Args:
        eval_intermediates_generator: the generator returned by evaluate_intermediates().

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
            tuple containing (inputs, intermediate values, outputs, true labels)
    """
    inputs, outputs, labels = [], [], []
    hiddens = defaultdict(list)
    for input, hidden, output, label in eval_intermediates_generator:
        if return_inputs:
            inputs.append(input)
            labels.append(label)
        if hasattr(output, "logits"):
            output = output.logits
        elif isinstance(output, dict) and "logits" in output:
            output = output["logits"]
        outputs.append(output)
        for k, v in hidden.items():
            hiddens[k].append(v)
    for k, v in hiddens.items():
        hiddens[k] = torch.cat(v, dim=0)

    if return_inputs:
        return (
            torch.cat(inputs, dim=0),
            hiddens,
            torch.cat(outputs, dim=0),
            torch.cat(labels, dim=0),
        )
    else:
        return (None, hiddens, torch.cat(outputs, dim=0), None)


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    include,
    exclude,
    device: str = "cuda",
    verbose: bool = False,
    n_examples: int = -1,
    is_language: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Evaluate model and return outputs, and optionally labels, accuracy, and loss.
    This does not return intermediate values.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (torch.utils.data.DataLoader): Dataloader containing data to evaluate on.
        state_dict (Dict, optional): If set, load these model parameters before evaluating. Defaults to None.
        device (str, optional): Device to evaluate on. Defaults to "cuda".
        return_accuracy (bool, optional): If True, include
            `torch.argmax(outputs) == labels` in return. Defaults to False.
        loss_fn (nn.Module, optional): If set, include
            `loss_fn(outputs, labels)` in return. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            a tuple of (outputs, true labels, accuracy, loss), with first dimension over examples in batch order.
            If return_accuracy or loss_fn are not set, accuracy or loss are None respectively.
    """
    eval_iterator = evaluate_intermediates(
        model,
        dataloader,
        device,
        include=include,
        exclude=exclude,
        verbose=verbose,
        n_examples=n_examples,
        is_language=is_language,
    )
    _, intermediates, outputs, _ = combine_batches(eval_iterator)
    return intermediates, outputs
