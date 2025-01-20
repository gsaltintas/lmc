import gc
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchmetrics import Accuracy
from tqdm import tqdm

from lmc.config import Config
from lmc.data.data_stats import DatasetRegistry
from lmc.permutations.activation_alignment import activation_matching
from lmc.permutations.weight_alignment import weight_matching
from lmc.utils.setup_training import Iterator


@torch.no_grad()
def interpolate_models(
    model1: Union[nn.Module], model2: Union[nn.Module], alpha: float
):
    """Given two models, linearly interpolates them with alpha.

    .. math::

        W \gets (1 - \alpha) W_1 + \alpha W_2
    """
    d = OrderedDict()
    model_interpolated = deepcopy(model1)
    for (name, param), (name1, param1), (name2, param2) in zip(
        model_interpolated.named_parameters(),
        model1.named_parameters(),
        model2.named_parameters(),
    ):
        assert name1 == name2, f"Model parameters do not have matching names at {name}."
        assert (
            param1.shape == param2.shape
        ), f"Model parameters do not have matching shapes at {name}."
        if not param.requires_grad:
            d[name] = param
            assert torch.allclose(
                param1, param2
            ), f"Parameter ({name}) doesn't require grad, hence we are not interpolating, ensure that these parameters are the same."
            continue
        with torch.no_grad():
            new_param = (1.0 - alpha) * param1 + alpha * param2
            d[name] = new_param
            ## the following also work
            # param.copy_(new_param)
            # param.data.copy_((1 - alpha) * param1.data + alpha * param2.data)
            # param.data = ((1 - alpha) * param1.data + alpha * param2.data)
    # make sure to copy over buffers as well, or maybe better to put None's idk
    for (name, param), (name1, param1), (name2, param2) in zip(
        model_interpolated.named_buffers(),
        model1.named_buffers(),
        model2.named_buffers(),
    ):
        d[name] = param

    model_interpolated.load_state_dict(d, strict=False)
    return model_interpolated


def evaluate_model_vision(
    model, loader, num_classes: int = 10, device=None, criterion=None
) -> dict:
    start_time = time.time()
    if device is None:
        device = model.device
    model.eval()

    acc = Accuracy("multiclass", num_classes=num_classes).to(device)
    # , device=device).to
    if criterion is None:
        criterion = nn.CrossEntropyLoss().to(device)
    ce = 0.0
    cnt = 0
    correct = 0
    for i, (x, y) in enumerate(loader):
        cnt += x.size(0)
        # doesn't work for some reason
        with torch.no_grad() and torch.autocast(
            device_type=model.device.type
        ):  # , dtype=model.dtype):
            if x.device != model.device:
                x = x.to(model.device)
                y = y.to(model.device)
            out = model(x)
            ce += criterion(out, y).item() * x.size(0)
        acc.update(out, y)
    acc_ = acc.compute().item()
    # acc_ = (correct/cnt)
    ce = ce / cnt
    end_time = time.time()
    # print(f"Time spent {end_time-start_time} seconds")
    return {"ce": ce, "acc": acc_, "err": 100 - 100 * acc_}


def get_empty_df() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [["train", "test"], ["loss", "ce", "acc", "err", "ppl", "em", "f1"]]
    )
    r = pd.DataFrame(columns=index)
    r["epoch"] = None
    r["alpha"] = None
    r.set_index(
        [
            "epoch",
            "alpha",
        ],
        inplace=True,
    )
    return r


def barrier_from_df(
        results: pd.DataFrame, ep: int, split: str, metric: str, prefix: str
    ) -> dict[str, float]:
    alpha = results.loc[ep, (split, metric)].idxmax()
    minalpha = results.loc[ep, (split, metric)].idxmin()

    scale = 1 / 100 if metric == "err" else 1

    max_interpolated = results.loc[ep, (split, metric)].max() * scale
    min_interpolated = results.loc[ep, (split, metric)].min() * scale
    endpoint_0 = results.loc[(ep, 0)][(split, metric)] * scale
    endpoint_1 = results.loc[(ep, 1)][(split, metric)] * scale

    linear_path = (1.0 - alpha) * endpoint_0 + alpha * endpoint_1
    barrier = max_interpolated - linear_path
    return {
        prefix + f"weighted/barrier_{split}": barrier,
        prefix + f"weighted/maxint_{split}": max_interpolated,
        prefix + f"weighted/minint_{split}": min_interpolated,
        prefix + f"weighted/maxalpha_{split}": alpha,
        prefix + f"weighted/minalpha_{split}": minalpha,
        prefix
        + f"weighted/increase_{split}": max_interpolated
        - min(endpoint_0, endpoint_1),
        prefix + f"weighted/increase_end0_{split}": max_interpolated - endpoint_0,
        prefix + f"weighted/increase_end1_{split}": max_interpolated - endpoint_1,
        prefix
        + f"weighted/decrease_{split}": min_interpolated
        - endpoint_0
        - min(endpoint_0, endpoint_1),
        prefix + f"weighted/decrease_end0_{split}": min_interpolated - endpoint_0,
        prefix + f"weighted/decrease_end1_{split}": min_interpolated - endpoint_1,
    }

def extract_barrier_vision(results: pd.DataFrame, ep: int) -> Dict[str, float]:
    """utility function to extract loss & error barriers corresponding to a vision task from a dataframe"""

    
    return {
        **barrier_from_df(results, ep, "train", "err", "lmc/"),
        **barrier_from_df(results, ep, "test", "err", "lmc/"),
        **barrier_from_df(results, ep, "train", "ce", "lmc/loss/"),
        **barrier_from_df(results, ep, "test", "ce", "lmc/loss/"),
    }
    

def extract_barrier_language(results: pd.DataFrame, ep: int) -> Dict[str, float]:
    """utility function to extract loss & error barriers corresponding to a language task from a dataframe"""
    
    return {
        **barrier_from_df(results, ep, "train", "err", "lmc/"),
        **barrier_from_df(results, ep, "test", "err", "lmc/"),
        **barrier_from_df(results, ep, "train", "ppl", "lmc/perplexity/"),
        **barrier_from_df(results, ep, "test", "ppl", "lmc/perplexity/"),
        **barrier_from_df(results, ep, "train", "ce", "lmc/loss/"),
        **barrier_from_df(results, ep, "test", "ce", "lmc/loss/"),
    }


def extract_barrier(results: pd.DataFrame, ep: int, is_language_task: bool = False) -> Dict[str, float]:
    """utility function to extract loss & error barriers from a dataframe"""
    if is_language_task:
        return extract_barrier_language(results, ep)
    return extract_barrier_vision(results, ep)


@torch.no_grad()
def check_lmc(
    training_elements,
    config: Config,
    ep,
    log_dct,
    results: Union[pd.DataFrame, None] = None,
    results_perm_wm: Union[pd.DataFrame, None] = None,
    results_perm_act_aligned: Union[pd.DataFrame, None] = None,
    check_perms: bool = False,
) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
    """
    checks lmc between the models, also accounts for the initial perm if any

     Args:
        training_elements: List of elements representing training states or models.
        config: Configuration object with relevant settings.
        ep: Current epoch.
        log_dct: Dictionary to log intermediate results.
        results: DataFrame to store results; created if None.
        results_perm_wm: DataFrame for matched permutation results; created if None.
        results_perm_act_aligned: DataFrame for activation-aligned results; created if None.
        check_perms: Boolean flag to indicate if permutations should be checked.

    Returns:
        Tuple containing updated results, results_perm_wm,
        and results_perm_act_aligned.


        TODO: missing stats
    """
    n_models = len(training_elements)
    if results is None:
        results = get_empty_df()

    # Initialize results DataFrames if not provided
    if results is None:
        results = get_empty_df()
    if check_perms and results_perm_wm is None:
        results_perm_wm = get_empty_df()
        results_perm_act_aligned = get_empty_df()

    is_language_model = config.data.is_language_dataset()

    # Iterate over previous models to evaluate LMC
    for model_ind, el in enumerate(training_elements):
        model = el.model
        for other_ind in range(max(0, model_ind - n_models), model_ind):
            prev_model = training_elements[other_ind].model
            if prev_model is None:
                continue

            # Evaluate LMC
            # todo: maybe add this to basemodel or trainingelement
            num_classes = (
                config.data.get_num_labels()
            )  # This will already raise an appropriate error if dataset not found
            results_ = interpolate_evaluate(
                ep,
                model,
                prev_model,
                None,
                model.device,
                el.train_eval_loader,
                el.test_loader,
                n_points=config.lmc.n_points,
                inner_tqdm=el.extra_iterator,
                num_classes=num_classes,
                is_language_model=is_language_model
            )
            results_["model1_ind"] = model_ind
            results_["model2_ind"] = other_ind
            if results is None:
                results = results_
            else:
                results = pd.concat([results, results_])
            lmc_res = extract_barrier(results, ep, is_language_task=is_language_model)
            log_dct.update(
                {f"lmc-{other_ind}-{model_ind}/{k}": v for k, v in lmc_res.items()}
            )

            # Check permutations if required
            if check_perms:
                ps = model.permutation_spec()
                for perm_method in ["wm", "am"]:
                    # weight_matching
                    if perm_method == "wm":
                        perm = weight_matching(
                            ps,
                            model.model.state_dict(),
                            prev_model.model.state_dict(),
                            init_perm=None,
                            verbose=False,
                        )
                        res_df = results_perm_wm
                    # Activation matching
                    else:
                        if ps.acts_to_perms is None:
                            continue
                        # todo: rename this activation_matching_samples
                        perm = activation_matching(
                            ps,
                            model,
                            prev_model,
                            dataloader=el.train_loader,
                            verbose=False,
                            num_samples=config.lmc.activation_matching_samples,
                        )
                        res_df = results_perm_act_aligned
                    permuted_ = prev_model._permute(perm, inplace=False)
                    results_perm = interpolate_evaluate(
                        ep,
                        permuted_,
                        model,
                        None,
                        model.device,
                        el.train_eval_loader,
                        el.test_loader,
                        n_points=config.lmc.n_points,
                        inner_tqdm=el.extra_iterator,
                        num_classes=num_classes,
                        is_language_model=is_language_model,
                    )
                    results_perm["model1_ind"] = model_ind
                    results_perm["model2_ind"] = other_ind
                    if res_df is None:
                        res_df = results_perm
                    else:
                        res_df = pd.concat([res_df, results_perm])
                    log_dct.update(
                        {
                            f"perm/{perm_method}-{other_ind}-{model_ind}/{k}": v
                            for k, v in extract_barrier(results_perm, ep, is_language_task=is_language_model).items()
                        }
                    )

    return results, results_perm_wm, results_perm_act_aligned

def evaluate_model_language(model, loader, device=None, criterion=None) -> dict:
    """
    Evaluate language model using HuggingFace model's native functionality.
    The model is expected to have a self.model attribute containing a HuggingFace transformer model.
    """
    if device is None:
        device = model.device
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    cnt = 0
    for batch in loader:
        cnt += 1
        if cnt == 3:
            break
        # Ensure all tensors are on the right device
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        with torch.no_grad():
            # Use the model's forward pass directly with the batch
            # This will handle loss computation if labels are provided
            outputs = model(**batch)

            # HuggingFace models return loss directly if labels are provided
            if hasattr(outputs, "loss"):
                loss = outputs.loss
                total_loss += loss.item() * (batch["labels"].numel() if "labels" in batch else batch["input_ids"].size(0))
                total_tokens += (batch["labels"].numel() if "labels" in batch else batch["input_ids"].size(0))
                
            # For models that don't return loss directly
            elif hasattr(outputs, "logits") and "labels" in batch:
                logits = outputs.logits
                labels = batch["labels"]
                if criterion is None:
                    # Use the model's default loss function if available
                    if hasattr(model.model, "compute_loss"):
                        loss = model.model.compute_loss(outputs, labels)
                    else:
                        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
                else:
                    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    
                total_loss += loss.item() * labels.numel()
                total_tokens += labels.numel()

                # Calculate accuracy if applicable
                if len(logits.shape) == len(labels.shape) + 1:  # Classification scenario
                    pred = logits.argmax(dim=-1)
                    total_correct += (pred == labels).sum().item()

    # Compute averages
    metrics = {}
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        metrics["ce"] = avg_loss
        metrics["ppl"] = torch.exp(torch.tensor(avg_loss)).item()
        
        if total_correct > 0:  # Only include accuracy metrics if we computed them
            accuracy = total_correct / total_tokens
            metrics["acc"] = accuracy
            metrics["err"] = 100 - 100 * accuracy
        # TODO: add glue metrics
    return metrics

def interpolate_evaluate(
    ai,
    model1,
    model2,
    results,
    device,
    train_loader,
    test_loader,
    suffix="",
    n_points=20,
    inner_tqdm=None,
    num_classes=None,
    criterion=None,
    interpolation_func=None,
    is_language_model=False,
) -> pd.DataFrame:
    """_summary_

    Args:
        ai (_type_): _description_
        model1 (_type_): _description_
        model2 (_type_): _description_
        results (_type_): _description_
        device (_type_): _description_
        train_loader (_type_): _description_
        test_loader (_type_): _description_
        interpolation_func (callable): If not None must be a valid function with the signature (modela, modelb, alpha) -> nn.Module
        suffix (str, optional): _description_. Defaults to "".
        n_points (int, optional): _description_. Defaults to 20.
        inner_tqdm (tqdm, optional): _description_. Defaults to None.
        num_classes (int, optional): _description_. Defaults to 10.
        criterion (_type_, optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    if interpolation_func is None:
        interpolation_func = interpolate_models

    ts = torch.linspace(0.0, 1.0, n_points)
    if inner_tqdm is None:
        inner_tqdm = tqdm(enumerate(ts), leave=True, desc="Interpolation")
    else:
        inner_tqdm.iterable = enumerate(ts)
        inner_tqdm.total = len(ts)
        inner_tqdm.set_description_str("Interpolation")

    inner_tqdm.reset()
    res_dict = dict()

    if results is None:
        results = get_empty_df()

    for i, t in inner_tqdm.iterable:
        gc.collect()
        inner_tqdm.set_description_str(f"Interpolation: {i}")
        model = interpolation_func(model1, model2, t)
        res = dict()

        for name, loader in [("train", train_loader), ("test", test_loader)]:
            if loader is None:
                continue
            if is_language_model:
                res[name] = evaluate_model_language(
                    model, loader, device=device
                )
            else:
                res[name] = evaluate_model_vision(
                    model,
                    loader,
                    num_classes=num_classes,
                    device=device,
                    criterion=criterion,
                )

        res = {
            (outerKey, innerKey): values
            for outerKey, innerDict in res.items()
            for innerKey, values in innerDict.items()
        }

        res_dict[i] = {
            f"{outer}/{inner}/{suffix}": it for (outer, inner), it in res.items()
        }
        names = results.index.names
        columns = results.columns

        if len(results) == 0:
            res["epoch"] = ai
            res["alpha"] = t.item()
            index = pd.MultiIndex.from_tuples(
                [(ai, t.item())], names=["epoch", "alpha"]
            )
            results = pd.DataFrame(res, index=index, columns=columns)
        else:
            results = pd.concat([results, pd.DataFrame(res, index=[(ai, t.item())])])

        results.index.names = names
        inner_tqdm.set_postfix({f"{k[0]}/{k[1]}": v for k, v in res.items()})
        inner_tqdm.update()

    inner_tqdm.reset()
    return results
