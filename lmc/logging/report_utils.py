"""Utilities for filtering and processing Weights & Biases experiment data.

This module provides functions for data cleaning, filtering experiments based on performance
metrics, and processing experiment labels. It handles common data preprocessing tasks when
working with wandb experiment logs, particularly for neural network training experiments
with metrics like accuracy, loss, and LMC metrics.

The module includes functions for:
- Checking and handling null values across multiple columns
- Filtering experiments based on performance thresholds
- Processing and formatting experiment labels for grouping and visualization
- Converting wandb metric names and handling hierarchical data structures

Typical usage:
   filtered_df = filter_bad_runs(merged_df, registry, min_avg_test_acc=0.7)
   labels = get_labels(filtered_df, group_indices=["trainer.opt.lr"])
"""

from typing import List, Union

import pandas as pd

from lmc.logging.wandb_registry import MetricCategory, Split
from lmc.utils.step import Step


def check_nulls(df: pd.DataFrame, columns: Union[str, List[str]]) -> pd.Series:
    """Helper to check various forms of null values in one or more columns.
    
    Args:
        df: DataFrame to check
        columns: Single column name or list of column names
    
    Returns:
        pd.Series: Boolean series where True means no null values in any form
    """
    # Handle single column case
    if isinstance(columns, str):
        return (df[columns] != "null") & (df[columns] != "NaN") & (~df[columns].isna())
    
    # Handle multiple columns
    conditions = []
    for col in columns:
        conditions.append(
            (df[col] != "null") & 
            (df[col] != "NaN") & 
            (~df[col].isna())
        )
    
    # Combine all conditions with logical AND
    return pd.concat(conditions, axis=1).all(axis=1)


def filter_bad_runs(
    merged_df: pd.DataFrame, 
    registry: "WandbMetricsRegistry",
    min_avg_test_acc: float = 0.7,
    min_model_acc: float = 0.8,
    model_idx: int = 1,
) -> pd.DataFrame:
    """Filter out bad runs from the dataset.
    
    Args:
        merged_df: DataFrame containing experiment data
        registry: Metrics registry
        perturb_method: Method of perturbation to filter for
        min_avg_test_acc: Minimum average test accuracy required
        min_model_acc: Minimum model train accuracy required
        model_idx: Index of the model to check accuracy for
    
    Returns:
        pd.DataFrame: Filtered dataset
    """
    # Get relevant metrics
    test_accs = registry.get_metrics_by_category(split=Split.TEST, category=MetricCategory.ACCURACY)
    train_accs = registry.get_metrics_by_category(split=Split.TRAIN, category=MetricCategory.ACCURACY)
    lmc_loss = registry.get_metrics_by_category(split=Split.TRAIN, category=MetricCategory.LMC_LOSS)
    lmc_barr = registry.get_metrics_by_category(split=Split.TRAIN, category=MetricCategory.LMC_ACCURACY)
    
    # Calculate minimum epoch from training steps
    min_epoch = merged_df["trainer.training_steps"].apply(lambda x: Step(x).get_epoch()).min()
    
    # Apply filters
    filtered = merged_df.copy()
    # filtered.loc[filtered["perturb_step"] == -1, "perturb_step"] = 0
    filtered = filtered[
        check_nulls(filtered, lmc_loss.get_flat_names()) &
        check_nulls(filtered, lmc_barr.get_flat_names()) &
        (filtered["average_test_acc"] >= min_avg_test_acc) &
        (filtered[train_accs.get_flat_names()].apply(lambda x: all(x >= min_model_acc), axis=1)) &
        (filtered["epoch"] >= min_epoch) 
    ]
    return filtered

def get_labels(data, group_indices: list = ["trainer.opt.optimizer", "trainer.opt.lr"], rename_opt_lr: bool = False, format_labels: bool = True):
    """Create masks for grouping data by specified columns.

    Args:
        data: DataFrame to group
        group_indices: Columns to group by
        rename_opt_lr: Flag to rename optimizer and learning rate (not implemented)
        format_labels: Whether to format group names with dashes between values

    Returns:
        dict: Mapping of formatted group names to row indices
    """
    if rename_opt_lr:
        raise NotImplementedError()
        data = data.apply(rename_opt_lr, axis=1)
    grouped = data.groupby(group_indices)
    unique_groups = grouped.groups.keys()
    
    masks = {}
    for g in unique_groups:
        name = g
        if format_labels:
            name = str(g) if type(g) in [str, float, int] or len(g) == 1 else "-".join([str(x) for x in g])
        masks[name] = grouped.get_group(g).index
    return masks
