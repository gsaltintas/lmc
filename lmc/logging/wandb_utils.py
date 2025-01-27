"""Data processing utilities for wandb experiment analysis.

This module provides functions for fetching, processing, and analyzing machine learning
experiment data from Weights & Biases (wandb). It handles various data manipulation tasks
including:

- Merging run configurations with metrics
- Handling missing values and history data
- Processing timeseries metrics
- Converting between different metric formats
- Standardizing experiment configurations
- Scaling and normalizing metrics

Key Functions:
   get_merged_df: Merge run configs with metrics and process into standard format
   get_missings_from_hist: Fetch missing metric values from run history
   get_timeseries_metrics: Extract timeseries data for specified metrics
   parse_steps: Convert step information to proper format

Example Usage:
   runs = wandb.Api().runs("project_name")
   merged_df, registry = get_merged_df(runs, return_registry=True)
   timeseries = get_timeseries_metrics_from_wandb(run_id, metrics=["loss", "accuracy"])
"""

import json
import logging
from typing import Collection, List, Optional, Tuple

import pandas as pd

import wandb
from lmc.logging.wandb_registry import MetricCategory, Split, WandbMetricsRegistry
from lmc.utils.step import Step
from lmc.utils.utils import unflatten_dict

logger = logging.getLogger(__file__)

from typing import Collection

import pandas as pd

import wandb

must_have_cols = ["trainer.opt.warmup_ratio"]


def parse_steps(row):
    """
    Parse step information from a row of data and convert any matching values to Step instances.

    Args:
        row (dict): A dictionary containing configuration values,
                    some of which may be in the form of Step(value=...).

    Returns:
        dict: A new dictionary with Step instances where applicable.
    """
    for key, value in row.items():
        # Check if the value is a string and matches the Step format
        if Step.is_step(value):
            row[key] = Step.from_string(value)
            continue

    return row


def get_merged_df(
    runs,
    performance_aware: bool = False,
    scale_barriers: bool = True,
    find_missing: bool = False,
    return_registry: bool = False,
) -> Tuple[pd.DataFrame, Optional[WandbMetricsRegistry]]:
    """Merge wandb runs into a single DataFrame with optional metric processing.

    Args:
        runs: List/WandbRuns of wandb runs to process
        performance_aware: Whether to normalize LMC accuracy metrics by models test performance
        scale_barriers: Whether to scale barrier metrics to percentages
        find_missing: Whether to search run histories for missing metrics
        return_registry: Whether to return the metrics registry alongside DataFrame

    Returns:
        DataFrame with merged runs data and optionally the metrics registry

    The function:
    - Combines run configs and metrics into single DataFrame
    - Handles missing values and metric scaling
    - Processes model architecture info
    - Optionally normalizes LMC metrics by model performance
    """
    results_df = pd.DataFrame(
        [
            {"project": run.project, "run_id": run.id} | run.summary._json_dict
            for run in runs
        ]
    )
    n_models = max([run.config.get("n_models", 1) for run in runs])
    wandb_keys = WandbMetricsRegistry(n_models)

    if find_missing:
        logger.info(
            f"Searching for {len(results_df[results_df['epoch'].isna()])} entries."
        )
        results_df[results_df["epoch"].isna()] = results_df[
            results_df["epoch"].isna()
        ].apply(get_missings_from_hist, keys=wandb_keys.get_log_names(), axis=1)

    config_df = pd.DataFrame(
        [
            {
                "run_id": run.id,
                "run.group": run.group,
                "run.name": run.name,
                "run_full_path": "/".join(run.path),
            }
            | run.config
            | unflatten_dict(run.config)
            for run in runs
        ]
    )
    # parse any Step logged as a string
    config_df = config_df.apply(parse_steps, axis=1)
    merged_df = pd.merge(config_df, results_df, on="run_id")
    merged_df.columns = [c.replace("/", ".") for c in merged_df.columns]
    # correct_config_changes(merged_df)
    if performance_aware:
        # Get the lists of column names
        acc_cols = wandb_keys.get_metrics_by_category(
            MetricCategory.ACCURACY, Split.TEST
        ).get_flat_names()
        ce_cols = wandb_keys.get_metrics_by_category(
            MetricCategory.CROSS_ENTROPY, Split.TEST
        ).get_flat_names()
        lmc_acc_cols = wandb_keys.get_lmc_metrics(
            split=Split.TEST, metric_type=MetricCategory.LMC_ACCURACY
        ).get_flat_names()
        lmc_err_cols = wandb_keys.get_lmc_metrics(
            split=Split.TEST, metric_type=MetricCategory.LMC_ERROR
        ).get_flat_names()
        lmc_ce_cols = wandb_keys.get_lmc_metrics(
            split=Split.TEST, metric_type=MetricCategory.LMC_LOSS
        ).get_flat_names()

        # Calculate averages using the column lists
        avg_acc = merged_df[acc_cols].mean(axis=1)
        avg_loss = merged_df[ce_cols].mean(axis=1)

        # Perform division for each column
        merged_df[lmc_acc_cols] = merged_df[lmc_acc_cols].div(avg_acc, axis=0)
        merged_df[lmc_err_cols] = merged_df[lmc_err_cols].div(avg_acc * 100, axis=0)
        merged_df[lmc_ce_cols] = merged_df[lmc_ce_cols].div(avg_loss, axis=0)

    if scale_barriers:
        lmc_acc_cols = [
            c
            for c in wandb_keys.get_metrics_by_category(
                category=MetricCategory.LMC_ACCURACY
            ).get_flat_names()
            if c in merged_df.columns
        ]
        merged_df[lmc_acc_cols] *= 100
        print(lmc_acc_cols)
        assert (merged_df[lmc_acc_cols] <= 100).all().all()

    # will depreceate
    if "trainer.opt.warmup_ratio" in merged_df.columns:
        merged_df["trainer.opt.warmup_ratio"].fillna(0, inplace=True)
    merged_df.fillna("null", inplace=True)
    if "model.widths" in merged_df.columns:
        merged_df["model.widths"] = merged_df["model.widths"].apply(json.loads)
    merged_df["model.model_name"] = merged_df[["model.model_name", "model.act"]].apply(
        lambda x: "linear" if x["model.act"] == "linear" else x["model.model_name"],
        axis=1,
    )
    for col in must_have_cols:
        if col not in merged_df.columns:
            merged_df.insert(len(merged_df.columns), col, value=None)
    if return_registry:
        return merged_df, wandb_keys
    return merged_df


def get_missings_from_hist(row, keys: Optional[List] = None):
    """Fetch missing values from wandb history, handling different logging frequencies.
    Each key's last available value is fetched independently."""
    if not keys or len(keys) == 0:
        return row

    try:
        project, run_id = row[["project", "run_id"]]
        api = wandb.Api()
        run = api.run(f"{project}/{run_id}")

        # Process each key individually
        for key in keys:
            if pd.isna(row[key]):
                try:
                    # Get history just for this key
                    values = run.history(keys=[key])
                    if not values.empty and key in values:
                        # Find last non-null value for this key
                        last_value = values[key].dropna().iloc[-1]
                        row[key] = last_value
                except Exception as e:
                    logger.error(f"Error fetching {key} for run {run_id}: {str(e)}")
                    continue

    except Exception as e:
        print(f"Error accessing run {run_id}: {str(e)}")

    return row


def get_timeseries_metrics_from_wandb(
    run_id: str,
    metrics: Collection[str],
    config_vars: Collection[str],
    max_steps: int = 1e6,
    train_acc_threshold: float = 0,
    report_epochs: bool = True,
) -> pd.DataFrame:
    """
    Fetches timeseries metrics and configuration variables from a Weights and Biases (wandb) run.

    Parameters:
        run_id (str): The ID of the wandb run.
        metrics (Collection[str]): List of metric names to fetch.
        config_vars (Collection[str]): List of configuration variables to fetch.
        max_steps (int, optional): Maximum number of steps to consider. Defaults to 1e6.
        train_acc_threshold (float, optional): Minimum train accuracy to include metrics. Defaults to 0.
        report_epochs (bool, optional): Whether to report epochs instead of steps. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the requested metrics and configuration variables.
    """
    # Initialize wandb API client
    api = wandb.Api()

    # Fetch the run by its ID
    run = api.run(run_id)

    # Retrieve configuration variables
    config_data = {var: run.config.get(var, None) for var in config_vars}

    # Initialize a dictionary to store metrics data
    data = {"step": []}
    for metric in metrics:
        data[metric] = []

    # Retrieve history (metrics) from the run
    history = run.history(keys=list(metrics), pandas=False, max_step=max_steps)

    # Filter and process the history
    for record in history:
        step = record["_step"]
        if (
            "train/accuracy" in record
            and record["train/accuracy"] < train_acc_threshold
        ):
            continue
        data["step"].append(step)
        for metric in metrics:
            data[metric].append(record.get(metric, None))

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Add configuration variables to the DataFrame
    for var, value in config_data.items():
        df[var] = value

    # Optionally convert steps to epochs if applicable
    if report_epochs and "epoch" in run.history(stream="keys"):
        epochs = [record.get("epoch", None) for record in history]
        df["epoch"] = epochs
        df.drop(columns="step", inplace=True)

    return df
