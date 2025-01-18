from pathlib import Path
from textwrap import wrap
from typing import List, Literal, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lmc.logging.plot_utils import (MARKERSIZE, NON_METRIC_LABEL_MAP,
                                    NON_METRIC_LEGEND_MAP, extract_legend,
                                    get_hues, modify_confidence_alphas,
                                    setup_styles)
from lmc.logging.report_utils import filter_bad_runs as filter_bad_runs_func
from lmc.logging.report_utils import get_labels
from lmc.logging.wandb_registry import (MetricCategory, PermMethod, Split,
                                        WandbMetric)


def plot_perturb_barrier(
    merged_df: pd.DataFrame,
    registry: "WandbMetricsRegistry",
    perturb_method: str,
    metric_name: str,
    x_metric: Union[str, WandbMetric],
    labels: Optional[List[str]] = None,
    zoom: Optional[Literal["first", "last"]] = None,
    out_dir: Path = Path("outputs"),
    filter_bad_runs: bool = True,
    zoom_first_step: int = 10,
    zoom_last_step: int = 380,
    separate_legend: bool = True,
    save_fig: bool = True,
    x_label: str = None,
    plot_type: Literal["scatter", "line"] = "line",
    title: str = "",
    legend_template: str = ""
) -> Path:
    """Plot barrier metrics comparing train and test performance across different perturbation scales.

    Creates a plot showing train (solid lines) and test (dashed lines) metrics for different perturbation
    scales (lambda values). The plot can be zoomed to focus on early or late training stages.

    Args:
        merged_df: DataFrame containing metrics from wandb runs
        registry: Registry containing metric definitions and transformations
        perturb_method: Method used for perturbation (e.g., "batch")
        metric_name: Name of the metric to plot from the registry
        x_metric: Metric or column name to use for x-axis values
        labels: List of label names to group by (defaults to ["perturb_scale"])
        zoom: Whether to focus on specific training range:
            None: full range
            "first": early training (≤ zoom_first_step)
            "last": late training (> zoom_last_step)
        out_dir: Directory to save plot files
        filter_bad_runs: Whether to filter out poor performing runs
        zoom_first_step: Step threshold when zoom="first"
        zoom_last_step: Step threshold when zoom="last"
        separate_legend: Whether to save legend as separate figure
        save_fig: Whether to save the generated plots
        x_label: Custom x-axis label (defaults to metric's ylabel or mapped label)
        plot_type: Type of plot - "scatter" or "line"
        title: Plot title (wrapped automatically)

    Returns:
        Path to the saved figure file
    """
    plt_kwargs = {}
    if plot_type == "scatter":
        plot_fn = sns.scatterplot
    elif plot_type == "line":
        plot_fn = sns.lineplot
        plt_kwargs.update(dict(
                   markersize=MARKERSIZE))
    else:
        raise ValueError(f"{plot_type} currently not supported")
    # Setup
    setup_styles(2)
    if labels is None:
        labels = ["perturb_scale"]
    # Get metrics
    base_name = metric_name
    base_metric = registry.get_metric(base_name)
    x = x_metric
    if (isinstance(x_metric, str) and registry.has_metric(x_metric)):
        x_metric = registry.get_metric(x_metric)
        x = x_metric.flat_name
    if isinstance(x_metric, WandbMetric) and x_label is None:
        x_label = x_metric.ylabel
    if x_label is None:
        x_label = NON_METRIC_LABEL_MAP.get(x_metric)

    tmp = merged_df.copy()
    if filter_bad_runs:
        tmp = filter_bad_runs_func(merged_df, registry)
    # Apply zoom filters
    tmp = tmp.reset_index()
    tmp = tmp[tmp["perturb_mode"] == perturb_method]
    path = out_dir.joinpath(f"{perturb_method}-{base_metric.prefix}").with_suffix(".pdf")
    if zoom == "first":
        tmp = tmp[tmp[x] <= zoom_first_step]
        path = path.with_stem(f"{path.stem}-zoom-first")
    elif zoom == "last":
        tmp = tmp[tmp[x] > zoom_last_step]
        path = path.with_stem(f"{path.stem}-zoom-last")

    # Plot
    fig, ax = plt.subplots()
    tmp = tmp.sort_values(x, ascending=True)
    masks = get_labels(tmp, labels, format_labels=False)
    hues = get_hues(1, len(masks), "plasma")

    for i, (label_, mask) in enumerate(masks.items()):
        df = tmp.loc[mask]
        color = hues[i % len(hues)]
        label = legend_template.format(label_)

        for mode in [Split.TRAIN, Split.TEST]:
            orig_split = base_metric.split
            metric_name_ = metric_name.replace(orig_split.value, mode.value)
            metric = registry.get_metric(metric_name_)

            plot_fn(df, x=x, y=metric.flat_name, ax=ax,
                   marker="o", color=color, 
                   label=label if mode==Split.TRAIN else None,
                   linestyle="-" if mode==Split.TRAIN else "--",
                   **plt_kwargs)

    plt.xlabel(x_label)
    plt.ylabel(base_metric.general_ylabel)
    # plt.title("\n".join(wrap(f"ResNet 20 - {perturb_method.title()} Perturbance", 30)))
    plt.title("\n".join(wrap(title)))
    if plot_type == "line":
        modify_confidence_alphas(ax, 0.1)

    if separate_legend:
        fig_leg = extract_legend(ax, ncol=2, add_tr_te=True, title=" ".join(labels))
    else:
        fig_leg = None
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if save_fig:
        path.parent.mkdir(exist_ok=True, parents=True)
        
        fig.savefig(path)
        if fig_leg is not None:
            fig_leg.savefig(path.as_posix().replace(".pdf", "-legend.pdf"), dpi=300, bbox_inches="tight")
        
    plt.show()
    return path
