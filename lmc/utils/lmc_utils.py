

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
from lmc.data.data_stats import CLASS_DICT
from lmc.permutations.activation_alignment import activation_matching
from lmc.permutations.alignment_methods import weight_matching
from lmc.utils.setup_training import Iterator


@torch.no_grad()
def interpolate_models(model1: Union[nn.Module], model2: Union[nn.Module], alpha: float):
    """Given two models, linearly interpolates them with alpha.
    
    .. math::

        W \gets (1 - \alpha) W_1 + \alpha W_2
    """
    d = OrderedDict()
    model_interpolated = deepcopy(model1)
    for (name, param), (name1, param1), (name2, param2) in zip(model_interpolated.named_parameters(), model1.named_parameters(), model2.named_parameters()):
        assert name1 == name2, f"Model parameters do not have matching names at {name}."
        assert param1.shape == param2.shape, f"Model parameters do not have matching shapes at {name}."
        if not param.requires_grad:
            d[name] = param
            assert torch.allclose(param1, param2), f"Parameter ({name}) doesn't require grad, hence we are not interpolating, ensure that these parameters are the same."
            continue
        with torch.no_grad():
            new_param = ((1. - alpha) * param1 + alpha * param2)
            d[name] = new_param
            ## the following also work
            # param.copy_(new_param)
            # param.data.copy_((1 - alpha) * param1.data + alpha * param2.data)
            # param.data = ((1 - alpha) * param1.data + alpha * param2.data)
    # make sure to copy over buffers as well, or maybe better to put None's idk
    for (name, param), (name1, param1), (name2, param2) in zip(model_interpolated.named_buffers(), model1.named_buffers(), model2.named_buffers()):
        d[name] = param

    model_interpolated.load_state_dict(d, strict=True)
    return model_interpolated



def evaluate_model(model, loader, num_classes:int=10, device=None, criterion=None) -> dict:
    start_time = time.time()
    if device is None:
        device = model.device
    model.eval()
    
    acc = Accuracy("multiclass", num_classes=num_classes).to(device)
    # , device=device).to
    if criterion is None:
        criterion = nn.CrossEntropyLoss().to(device)
    ce = 0.
    cnt = 0
    correct = 0
    for i, (x, y) in enumerate(loader):
        cnt += x.size(0)
        # doesn't work for some reason
        with torch.no_grad() and torch.autocast(device_type=model.device.type): #, dtype=model.dtype):
            if x.device != model.device:
                x = x.to(model.device)
                y = y.to(model.device)
            out = model(x)
            ce += criterion(out, y).item() * x.size(0)
        acc.update(out, y)
    acc_ = acc.compute().item()
    # acc_ = (correct/cnt)
    ce = (ce / cnt)
    end_time = time.time()
    # print(f"Time spent {end_time-start_time} seconds")
    return {"ce": ce, "acc": acc_, "err": 100 - 100 * acc_}


def get_empty_df() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [["train", "test"], ["loss", "ce", "acc", "err"]]
    )
    r = pd.DataFrame(columns=index)
    r["epoch"] = None
    r["alpha"] = None
    r.set_index(["epoch", "alpha", ], inplace=True)
    return r
    r["model1_ind"] = None
    r["model2_ind"] = None
    r.set_index(["epoch", "alpha", "model1_ind", "model2_ind"], inplace=True)
    return r

def interpolate_evaluate(ai, model1, model2, results, device, train_loader, test_loader, suffix="", n_points: int=20, inner_tqdm: tqdm=Iterator(), num_classes=10, criterion=None, interpolation_func: callable = None) -> pd.DataFrame:
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
            res[name] = evaluate_model(model, loader, device=device, num_classes=num_classes, criterion=criterion)
        res = {
            (outerKey, innerKey): values
            for outerKey, innerDict in res.items()
            for innerKey, values in innerDict.items()
        }
        # todo make this not index but column names
        res_dict[i] = {f"{outer}/{inner}/{suffix}": it for (outer, inner), it in res.items()}
        names = results.index.names
        columns = results.columns
        if len(results) == 0:
            # pandas warning otherwise
            res["epoch"] = ai
            res["alpha"] = t.item()
            index = pd.MultiIndex.from_tuples([(ai, t.item())], names=["epoch", "alpha"] )
            results = pd.DataFrame(res, index=index, columns=columns)
            # results.set_index(["epoch", "alpha"], inplace=True)
            # print(results)
        else:
            results = pd.concat([results, pd.DataFrame(res, index=[(ai, t.item())])])
        results.index.names = names
        inner_tqdm.set_postfix({f"{k[0]}/{k[1]}": v for k, v in res.items()})
        inner_tqdm.update()
    inner_tqdm.reset()
    return results


def extract_barrier(results: pd.DataFrame, ep: int) -> Dict[str, float]:
    """ utility function to extract loss & error barriers from a dataframe """
    train_alpha = results.loc[ep, ("train", "err")].idxmax()
    test_alpha = results.loc[ep, ("train", "err")].idxmax()
    train_loss_alpha = results.loc[ep, ("train", "ce")].idxmax()
    test_loss_alpha = results.loc[ep, ("train", "ce")].idxmax()

    train_barr = results.loc[ep, ("train", "err")].max() -  (
        (1.-train_alpha) * results.loc[(ep, 0)][("train", "err")] + train_alpha * results.loc[(ep, 1)][("train", "err")]
    )
    test_barr = results.loc[ep, ("test", "err")].max() - (
        (1.-test_alpha) * results.loc[(ep, 0)][("test", "err")] + test_alpha * results.loc[(ep, 1)][("test", "err")]
    )
    train_loss_barr = results.loc[ep, ("train", "ce")].max() - (
        (1.-train_loss_alpha) * results.loc[(ep, 0)][("train", "ce")] + train_loss_alpha * results.loc[(ep, 1)][("train", "ce")]
    )
    test_loss_barr = results.loc[ep, ("test", "ce")].max() - (
        (1.-test_loss_alpha) * results.loc[(ep, 0)][("test", "ce")] + test_loss_alpha * results.loc[(ep, 1)][("test", "ce")]
    )
    
    d = {
        "lmc/weighted/barrier_train": train_barr / 100,
        "lmc/weighted/barrier_test": test_barr / 100,
        "lmc/loss/weighted/barrier_train": train_loss_barr,
        "lmc/loss/weighted/barrier_test": test_loss_barr,
        }
    return d



@torch.no_grad()
def check_lmc(training_elements, config: Config, ep, log_dct, results: Union[pd.DataFrame, None] = None, results_perm_wm: Union[pd.DataFrame, None] = None, results_perm_act_aligned: Union[pd.DataFrame, None] = None, check_perms: bool = False) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
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
    

    # Iterate over previous models to evaluate LMC
    for model_ind, el in enumerate(training_elements):
        model = el.model
        for other_ind in range(max(0, model_ind - n_models), model_ind):
            prev_model = training_elements[other_ind].model
            if prev_model is None:
                continue
            
            # Evaluate LMC
            # todo: maybe add this to basemodel or trainingelement
            num_classes = CLASS_DICT[config.data.dataset]
            results_ = interpolate_evaluate(ep, model, prev_model, None, model.device, el.train_eval_loader, el.test_loader, n_points=config.lmc.n_points, inner_tqdm=el.extra_iterator, num_classes=num_classes)
            results_["model1_ind"] = model_ind
            results_["model2_ind"] = other_ind
            if results is None:
                results = results_
            else:
                results = pd.concat([results, results_])
            lmc_res = extract_barrier(results, ep)
            log_dct.update({f"lmc-{other_ind}-{model_ind}/{k}": v for k, v in lmc_res.items()})
            
            # Check permutations if required
            if check_perms:
                ps = model.permutation_spec()
                for perm_method in ["wm", "am"]:
                    # weight_matching
                    if perm_method == "wm":
                        perm = weight_matching(ps, model.model.state_dict(), prev_model.model.state_dict(), init_perm=None, verbose=False)
                        res_df = results_perm_wm
                    # Activation matching
                    else:
                        # todo: rename this activation_matching_samples
                        perm = activation_matching(ps, model, prev_model, dataloader=el.train_loader, verbose=False, num_samples=config.lmc.activation_matching_samples)
                        res_df = results_perm_act_aligned
                    permuted_ = prev_model._permute(perm, inplace=False)
                    results_perm = interpolate_evaluate(ep, permuted_, model, None, model.device, el.train_eval_loader, el.test_loader, n_points=config.lmc.n_points, inner_tqdm=el.extra_iterator, num_classes=num_classes)
                    results_perm["model1_ind"] = model_ind
                    results_perm["model2_ind"] = other_ind
                    if res_df is None:
                        res_df = results_perm
                    else:
                        res_df = pd.concat([res_df, results_perm])
                    log_dct.update({f"perm/{perm_method}-{other_ind}-{model_ind}/{k}": v for k, v in extract_barrier(results_perm, ep).items()})
        
    return results, results_perm_wm, results_perm_act_aligned
   