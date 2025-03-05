import gc
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, Tuple, Union

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lmc.experiment_config import Experiment
from lmc.models.base_model import BaseModel
from lmc.permutations.activation_alignment import activation_matching
from lmc.permutations.perm_stability import sinkhorn_kl
from lmc.permutations.perm_stats import get_fixed_points_count, get_fixed_points_ratio
from lmc.permutations.weight_alignment import weight_matching
from lmc.utils.training_element import VisionTrainingElement, NLPTrainingElement


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


def get_empty_df() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [
            ["train", "test"],
            [
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
    r["step"] = None
    r["alpha"] = None
    r.set_index(
        [
            "epoch",
            "step",
            "alpha",
        ],
        inplace=True,
    )
    return r


def barrier_from_df(
    results: pd.DataFrame, split: str, metric: str, prefix: str
) -> dict[str, float]:
    results_ = results.dropna(axis=1, how="all")

    if (split, metric) not in results_.columns:
        return {}
    rows = results[(split, metric)]
    alpha = rows.idxmax()
    scale = 1 / 100 if metric == "err" else 1

    max_interpolated = rows.max() * scale
    endpoint_0 = rows[0] * scale
    endpoint_1 = rows[1] * scale

    linear_path = (1.0 - alpha) * endpoint_0 + alpha * endpoint_1
    barrier = max_interpolated - linear_path
    return {
        prefix + f"weighted/barrier_{split}": barrier,
        prefix + f"weighted/alpha_{split}": alpha,
    }


def extract_barrier_vision(results: pd.DataFrame) -> Dict[str, float]:
    """utility function to extract loss & error barriers corresponding to a vision task from a dataframe"""

    return {
        **barrier_from_df(results, "train", "err", "lmc/"),
        **barrier_from_df(results, "test", "err", "lmc/"),
        **barrier_from_df(results, "train", "cross_entropy", "lmc/loss/"),
        **barrier_from_df(results, "test", "cross_entropy", "lmc/loss/"),
    }


def extract_barrier_language(results: pd.DataFrame) -> Dict[str, float]:
    """utility function to extract loss & error barriers corresponding to a language task from a dataframe"""
    # todo: better way to handle these
    d = {
        **barrier_from_df(results, "train", "cross_entropy", "lmc/loss/"),
        **barrier_from_df(results, "test", "cross_entropy", "lmc/loss/"),
    }
    cols = results.columns.get_level_values(1)
    if "accuracy" in cols:
        d.update(
            {
                **barrier_from_df(results, "train", "accuracy", "lmc/"),
                **barrier_from_df(results, "test", "accuracy", "lmc/"),
            }
        )
    if "perplexity" in cols:
        d.update(
            {
                **barrier_from_df(results, "train", "perplexity", "lmc/perplexity/"),
                **barrier_from_df(results, "test", "perplexity", "lmc/perplexity/"),
            }
        )
    if "matthews_correlation" in cols:
        d.update(
            {
                **barrier_from_df(
                    results, "train", "matthews_correlation", "lmc/loss/"
                ),
                **barrier_from_df(results, "test", "matthews_correlation", "lmc/loss/"),
            }
        )
    if "pearson_correlation" in cols:
        d.update(
            {
                **barrier_from_df(results, "train", "pearson_correlation", "lmc/loss/"),
                **barrier_from_df(results, "test", "pearson_correlation", "lmc/loss/"),
            }
        )
    return d


def extract_barrier(
    results: pd.DataFrame, is_language_task: bool = False
) -> Dict[str, float]:
    """utility function to extract loss & error barriers from a dataframe"""
    if is_language_task:
        return extract_barrier_language(results)
    return extract_barrier_vision(results)


@torch.no_grad()
def evaluate_merge(
    training_elements,
    config: Experiment,
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

    reference_el = copy_training_element(training_elements[0])
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
            reference_model.state_dict(),
            init_perm=None,
            verbose=False,
        )
        permuted_model = model._permute(perm, inplace=False)
        for n, p in permuted_model.state_dict().items():
            after_perms_model_dct[n] += 1.0 / config.n_models * p.clone().detach()
    merged_model = deepcopy(model)
    merged_model.load_state_dict(model_dct)
    permuted_model.load_state_dict(after_perms_model_dct)
    for split in ["train", "test"]:
        reference_el.model = merged_model
        vanilla_results = reference_el.evaluate(split=split)
        reference_el.model = permuted_model
        perm_results = reference_el.evaluate(split=split)

        log_dct.update(
            {f"merge/{split}/{key}": val for key, val in vanilla_results.items()}
        )
        log_dct.update(
            {f"merge/wm/{split}/{key}": val for key, val in perm_results.items()}
        )


def get_lmc_pairs(config):
    lmc_pairs = config.lmc.lmc_pairs
    # include all pairs if not set
    if not lmc_pairs:
        n_models = config.n_models
        return [(i, j) for i in range(0, n_models) for j in range(i + 1, n_models)]
    # parse string into comma separated pairs of form {i}-{j}
    pair_str = lmc_pairs.split(",")
    pairs = []
    for pair in pair_str:
        i, j = pair.split("-")
        pairs.append((int(i), int(j)))
    return pairs


@torch.no_grad()
def check_lmc(
    training_elements,
    config: Experiment,
    log_dct,
) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
    """
    checks lmc between the models, also accounts for the initial perm if any

     Args:
        training_elements: List of elements representing training states or models.
        config: Configuration object with relevant settings.
        log_dct: Dictionary to log intermediate results.
        check_perms: Boolean flag to indicate if permutations should be checked.

    Returns:
        Tuple containing updated results, results_perm_wm,
        and results_perm_act_aligned.
    """
    # Initialize results DataFrames if not provided
    results = get_empty_df()
    results_perm_wm = get_empty_df()
    results_perm_act_aligned = get_empty_df()

    is_language_model = config.data.is_language_dataset()

    # Iterate over previous models to evaluate LMC
    for model_ind, other_ind in get_lmc_pairs(config):
        el = training_elements[model_ind]
        prev_el = training_elements[other_ind]

        # Evaluate LMC
        # todo: maybe add this to basemodel or trainingelement
        results_ = interpolate_evaluate(el, prev_el)
        lmc_res = extract_barrier(results_, is_language_task=is_language_model)
        log_dct.update(
            {f"lmc-{model_ind}-{other_ind}/{k}": v for k, v in lmc_res.items()}
        )
        results = results_ if results is None else pd.concat([results, results_])

        # Check permutations if required
        if config.lmc.lmc_check_perms:
            ps = el.model.permutation_spec()
            for perm_method in ["wm", "am"]:
                # weight_matching
                if perm_method == "wm":
                    perm = weight_matching(
                        ps,
                        el.model.model.state_dict(),
                        prev_el.model.model.state_dict(),
                        init_perm=None,
                        verbose=False,
                    )
                    results_perm = results_perm_wm

                # Activation matching
                else:
                    if ps.acts_to_perms is None:
                        continue
                    # todo: rename this activation_matching_samples
                    perm = activation_matching(
                        ps,
                        el.model,
                        prev_el.model,
                        dataloader=el.train_eval_loader,
                        verbose=False,
                        num_samples=config.lmc.activation_matching_samples,
                    )
                    results_perm = results_perm_act_aligned

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
                results_perm_ = interpolate_evaluate(el, prev_el, perm)
                perm_res = extract_barrier(
                    results_perm_, is_language_task=is_language_model
                )
                log_dct.update(
                    {
                        f"perm/{perm_method}-{other_ind}-{model_ind}/{k}": v
                        for k, v in perm_res.items()
                    }
                )
                results_perm = (
                    results_perm_
                    if results_perm is None
                    else pd.concat([results_perm, results_perm_])
                )
    print("=" * 25, " LMC Results ", "=" * 25)
    print(results)
    print("=" * 22, " LMC Results (WM) ", "=" * 23)
    print(results_perm_wm)
    print("=" * 22, " LMC Results (AM) ", "=" * 23)
    print(results_perm_act_aligned)
    return results, results_perm_wm, results_perm_act_aligned


def copy_training_element(element):
    # create a copy of TrainingElement so model can be replaced by interpolated models
    training_element_class = (
        NLPTrainingElement if element.model.is_language_model else VisionTrainingElement
    )
    new_element = training_element_class(
        config=element.config,
        element_ind=0,
        device=element.device,
        max_steps=element.max_steps,
        train_loader=element.train_loader,
        train_eval_loader=element.train_eval_loader,
        test_loader=element.test_loader,
        model=element.model,
        opt=None,
        scheduler=None,
        tokenizer=element.tokenizer,
        perturb_seed=None,
    )
    return new_element


def interpolate_evaluate(
    el1,
    el2,
    perm=None,
) -> pd.DataFrame:
    model1 = el1.model
    model2 = el2.model
    if perm is not None:
        model2 = model2._permute(perm, inplace=False)
    results = get_empty_df()

    interpolate_el = copy_training_element(el1)
    config = interpolate_el.config
    inner_tqdm = interpolate_el.extra_iterator

    ts = torch.linspace(0.0, 1.0, config.lmc.n_points)
    if inner_tqdm is None:
        inner_tqdm = tqdm(enumerate(ts), leave=True, desc="Interpolation")
    else:
        inner_tqdm.iterable = enumerate(ts)
        inner_tqdm.total = len(ts)
        inner_tqdm.set_description_str("Interpolation")
    inner_tqdm.reset()

    for i, t in inner_tqdm.iterable:
        inner_tqdm.set_description_str(f"Interpolation: {i}")
        if config.lmc.lmc_use_saved_endpoint_evaluations and t == 0:
            element = el1
        elif config.lmc.lmc_use_saved_endpoint_evaluations and t == 1 and perm is None:
            element = el2
        else:
            gc.collect()
            model = interpolate_models(model1, model2, t)
            repair(model, el1.train_eval_loader)
            interpolate_el.model = model
            element = interpolate_el

        res = {}
        for split in ["train", "test"]:
            row = element.evaluate(split=split)
            row["err"] = 100 - 100 * row["accuracy"]
            res.update({(split, k): v for k, v in row.items()})

        res["alpha"] = t.item()
        results = pd.concat([results, pd.DataFrame(res, index=[t.item()])])

        inner_tqdm.set_postfix({f"{k[0]}/{k[1]}": v for k, v in res.items()})
        inner_tqdm.update()

    inner_tqdm.reset()
    results["model1_ind"] = el1.element_ind
    results["model2_ind"] = el2.element_ind
    return results
