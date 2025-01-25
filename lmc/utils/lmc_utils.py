import gc
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, Tuple, Union

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm

from lmc.config import Config, DataConfig
from lmc.data.data_stats import TaskType
from lmc.models.base_model import BaseModel
from lmc.permutations import get_cost
from lmc.permutations.activation_alignment import activation_matching
from lmc.permutations.perm_stability import sinkhorn_kl
from lmc.permutations.perm_stats import get_fixed_points_count, get_fixed_points_ratio
from lmc.permutations.weight_alignment import weight_matching
from lmc.utils.metrics import Metrics, compute_metrics
from lmc.utils.setup_training import Iterator
from lmc.utils.training_element import TrainingElement


@torch.no_grad()
def repair(model: BaseModel, loader: DataLoader) -> nn.Module:
    cnt = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # m.momentum = None  # use simple average
            m.reset_running_stats()
            cnt += 1
    if cnt == 0:
        return model
    model.train()
    with torch.no_grad():
        for batch in loader:
            if model.is_language_model:
                batch = {
                    k: v.to(model.device, non_blocking=True)
                    if isinstance(v, torch.Tensor)
                    else v
                    for k, v in batch.items()
                }
                output = model(**batch)
            else:
                images, _ = batch
                output = model(images.to(model.device, non_blocking=True))
    return model


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
        assert param1.shape == param2.shape, (
            f"Model parameters do not have matching shapes at {name}."
        )
        if not param.requires_grad:
            d[name] = param
            assert torch.allclose(param1, param2), (
                f"Parameter ({name}) doesn't require grad, hence we are not interpolating, ensure that these parameters are the same."
            )
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
        [
            ["train", "test"],
            [
                "loss",
                "ce",
                "acc",
                "err",
                "ppl",
                "em",
                "f1",
                "cross_entropy",
                "accuracy",
                "perplexity",
                "matthews_correlation",
                "pearson_correlation",
            ],
        ]
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
    results_ = results.dropna(axis=1, how="all")

    if (split, metric) not in results_.columns:
        return {}
    alpha = results_.loc[ep, (split, metric)].idxmax()
    minalpha = results_.loc[ep, (split, metric)].idxmin()

    scale = 1 / 100 if metric == "err" else 1

    max_interpolated = results_.loc[ep, (split, metric)].max() * scale
    endpoint_0 = results_.loc[(ep, 0)][(split, metric)] * scale
    endpoint_1 = results_.loc[(ep, 1)][(split, metric)] * scale

    linear_path = (1.0 - alpha) * endpoint_0 + alpha * endpoint_1
    barrier = max_interpolated - linear_path
    return {
        prefix + f"weighted/barrier_{split}": barrier,
        prefix + f"weighted/alpha_{split}": alpha,
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
    # todo: better way to handle these
    d = {
        **barrier_from_df(results, ep, "train", "cross_entropy", "lmc/loss/"),
        **barrier_from_df(results, ep, "test", "cross_entropy", "lmc/loss/"),
    }
    cols = results.columns.get_level_values(1)
    if "accuracy" in cols:
        d.update(
            {
                **barrier_from_df(results, ep, "train", "accuracy", "lmc/"),
                **barrier_from_df(results, ep, "test", "accuracy", "lmc/"),
            }
        )
    if "perplexity" in cols:
        d.update(
            {
                **barrier_from_df(
                    results, ep, "train", "perplexity", "lmc/perplexity/"
                ),
                **barrier_from_df(results, ep, "test", "perplexity", "lmc/perplexity/"),
            }
        )
    if "matthews_correlation" in cols:
        d.update(
            {
                **barrier_from_df(
                    results, ep, "train", "matthews_correlation", "lmc/loss/"
                ),
                **barrier_from_df(
                    results, ep, "test", "matthews_correlation", "lmc/loss/"
                ),
            }
        )
    if "pearson_correlation" in cols:
        d.update(
            {
                **barrier_from_df(
                    results, ep, "train", "pearson_correlation", "lmc/loss/"
                ),
                **barrier_from_df(
                    results, ep, "test", "pearson_correlation", "lmc/loss/"
                ),
            }
        )
    return d


def extract_barrier(
    results: pd.DataFrame, ep: int, is_language_task: bool = False
) -> Dict[str, float]:
    """utility function to extract loss & error barriers from a dataframe"""
    if is_language_task:
        return extract_barrier_language(results, ep)
    return extract_barrier_vision(results, ep)


@torch.no_grad()
def evaluate_merge(
    training_elements,
    config: Config,
    log_dct,
) -> None:
    """
    Merge models using different methods and evaluate their performance

    Args:
        training_elements: List of elements representing training states or models.
        config: Configuration object with relevant settings.
        log_dct: Dictionary to log intermediate results.
        results: DataFrame to store original merge results; created if None.
        results_merged_wm: DataFrame for weight-matched merge results; created if None.
        results_merged_act_aligned: DataFrame for activation-aligned merge results; created if None.

    """

    is_language_model = config.data.is_language_dataset()
    reference_el: TrainingElement = training_elements[0]
    reference_model = reference_el.model
    model_dct = {
        key: val / config.n_models
        for (key, val) in reference_model.state_dict().items()
    }
    after_perms_model_dct = {
        key: val / config.n_models
        for (key, val) in reference_model.state_dict().items()
    }
    for model_ind, el in enumerate(training_elements):
        if model_ind == 0:
            continue

        model = el.model
        if model is None:
            continue

        # Vanilla average merging
        for n, p in model.state_dict().items():
            model_dct[n] += 1.0 / config.n_models * p.clone().detach()

        ps = model.permutation_spec()
        perm = weight_matching(
            ps,
            model.model.state_dict(),
            reference_model.model.state_dict(),
            init_perm=None,
            verbose=False,
        )
        permuted_ = model._permute(perm, inplace=False)
        for n, p in permuted_.state_dict().items():
            after_perms_model_dct[n] += 1.0 / config.n_models * p.clone().detach()
    merged_model = deepcopy(model)
    merged_model.load_state_dict(model_dct)
    permuted_.load_state_dict(after_perms_model_dct)
    vanilla_results = {}
    perm_results = {}
    for name, loader in [
        ("train", reference_el.train_eval_loader),
        ("test", reference_el.test_loader),
    ]:
        if loader is None:
            continue
        if is_language_model:
            vanilla_results = evaluate_model_language(
                merged_model,
                reference_el.train_eval_loader,
                config.data,
                device=reference_model.device,
                criterion=reference_el.loss_fn,
            )
            perm_results = evaluate_model_language(
                permuted_,
                reference_el.train_eval_loader,
                config.data,
                device=reference_model.device,
                criterion=reference_el.loss_fn,
            )
        else:
            vanilla_results = evaluate_model_vision(
                merged_model,
                reference_el.train_eval_loader,
                num_classes=config.data.get_num_labels(),
                criterion=reference_el.loss_fn,
            )
            perm_results = evaluate_model_vision(
                permuted_,
                reference_el.train_eval_loader,
                num_classes=config.data.get_num_labels(),
                criterion=reference_el.loss_fn,
            )

        log_dct.update(
            {f"merge/{name}/{key}": val for key, val in vanilla_results.items()}
        )
        log_dct.update(
            {f"merge/wm/{name}/{key}": val for key, val in perm_results.items()}
        )


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
                is_language_model=is_language_model,
                data_config=config.data,
            )
            results_["model1_ind"] = model_ind
            results_["model2_ind"] = other_ind
            if results is None:
                results = results_
            else:
                results = pd.concat([results, results_])
            lmc_res = extract_barrier(results_, ep, is_language_task=is_language_model)
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

                    if config.logger.report_permutation_stats:
                        ## TODO: log costs
                        d = {
                            "fixed_points_ratio": get_fixed_points_ratio(perm),
                            "fixed_points_count": get_fixed_points_count(perm),
                            "sinkhorn_kl": sum(list(sinkhorn_kl(perm).values())),
                        }
                        log_dct.update(
                            {
                                f"perm/{perm_method}-{other_ind}-{model_ind}/{key}": val
                                for key, val in d.items()
                            }
                        )
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
                        data_config=config.data,
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
                            for k, v in extract_barrier(
                                results_perm, ep, is_language_task=is_language_model
                            ).items()
                        }
                    )
    print("=" * 25, " LMC Results ", "=" * 25)
    print(results.dropna(axis=1, how="all"))
    print("=" * 22, " LMC Results (WM) ", "=" * 23)
    print(results_perm_wm.dropna(axis=1, how="all"))
    print("=" * 22, " LMC Results (AM) ", "=" * 23)
    print(results_perm_act_aligned.dropna(axis=1, how="all"))
    print(log_dct)
    return results, results_perm_wm, results_perm_act_aligned


def evaluate_model_language(
    model, loader, data_config: DataConfig, device=None, criterion=None
) -> dict:
    """
    Evaluate language model using HuggingFace model's native functionality.
    The model is expected to have a self.model attribute containing a HuggingFace transformer model.
    """
    if device is None:
        device = model.device
    model.eval()
    dataset = data_config.dataset_info

    metrics = Metrics()
    metrics_kwargs = {}
    for batch in loader:
        batch = {
            k: v.to(model.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        outputs = model(**batch)
        n = batch["input_ids"].shape[0]
        metrics_kwargs = {"n": n}
        # Always track loss
        metrics_kwargs["cross_entropy"] = outputs.loss.item()

        # Update dataset-specific metrics
        if dataset.metrics:
            if data_config.task_type == TaskType.REGRESSION:
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
    res = metrics.get_metrics(percentage=False)
    return res


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
    data_config: DataConfig = None,
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
        repair(model, train_loader)

        for name, loader in [("train", train_loader), ("test", test_loader)]:
            if loader is None:
                continue
            if is_language_model:
                res[name] = evaluate_model_language(
                    model, loader, data_config, device=device
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
