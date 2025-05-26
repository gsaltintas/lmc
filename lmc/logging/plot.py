from pathlib import Path
from textwrap import wrap
from typing import List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker, transforms
from matplotlib.scale import ScaleBase

from lmc.logging.plot_utils import (
    MARKERSIZE,
    NON_METRIC_LABEL_MAP,
    NON_METRIC_LEGEND_MAP,
    extract_legend,
    get_hues,
    modify_confidence_alphas,
    setup_styles,
)
from lmc.logging.report_utils import filter_bad_runs as filter_bad_runs_func
from lmc.logging.report_utils import get_labels
from lmc.logging.wandb_registry import MetricCategory, PermMethod, Split, WandbMetric


class SafeLogScale(ScaleBase):
    """
    Custom log scale that treats log(0) as 0 instead of -∞.
    """

    name = "safe_log"

    def __init__(self, axis, **kwargs):
        super().__init__(axis)

    def get_transform(self):
        return self.SafeLogTransform()

    def set_default_locators_and_formatters(self, axis):
        """
        Define default locators and formatters for the custom scale.
        """
        axis.set_major_locator(ticker.LogLocator(base=10.0, subs="auto"))
        axis.set_major_formatter(
            ticker.LogFormatterMathtext()
        )  # <-- Uses 10^x notation
        axis.set_minor_locator(
            ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)
        )

    class SafeLogTransform(transforms.Transform):
        input_dims = output_dims = 1

        def transform_non_affine(self, values):
            values = np.asarray(values)
            return np.where(
                values > 0, np.log10(values), 0
            )  # log10(x) for x > 0, else 0

        def inverted(self):
            return SafeLogScale.InvertedSafeLogTransform()

    class InvertedSafeLogTransform(transforms.Transform):
        input_dims = output_dims = 1

        def transform_non_affine(self, values):
            values = np.asarray(values)
            return np.where(values > 0, 10**values, 0)  # 10^x for x > 0, else 0

        def inverted(self):
            return SafeLogScale.SafeLogTransform()


from matplotlib import scale as mscale

# Register the scale so it can be used in plots
mscale.register_scale(SafeLogScale)


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
    y_label: str = None,
    plot_type: Literal["scatter", "line"] = "line",
    title: str = "",
    legend_template: str = "",
    legend_title: str = None,
    xscale: Literal["log", "linear", "symlog"] = "linear",
    yscale: Literal["log", "linear", "symlog"] = "linear",
    hue_cnt: Optional[int] = None,
    ax=None,
    markers: bool = True,
    ncols: int = 2,
    save_prefix: str = "",
    uncertainty: bool = False,
    inset_fig: bool = False,
    inset_pos: Literal["left", "right", "bottom-left"] = "left",
    zoom_y_last_step: int = None,
    add_test_line: bool = False,
    file_extension: Literal["pdf", "png", "jpg"] = "pdf",
    style_by: str = None,
    palette: str = "viridis",
    marker="o",
    markersize=MARKERSIZE,
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
    ### change above after rebuttal

    if inset_fig and ax:
        raise NotImplementedError(f"Inset fig and ax are mutually exclusive for now")
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    if inset_fig:
        if inset_pos == "left":
            inset_ax = plt.axes([0.2, 0.6, 0.2, 0.2], facecolor="lightgray")
        elif inset_pos == "right":
            inset_ax = plt.axes([0.7, 0.6, 0.2, 0.2], facecolor="lightgray")
        elif inset_pos == "bottom-left":
            inset_ax = plt.axes([0.2, 0.2, 0.2, 0.2], facecolor="lightgray")

        else:
            raise ValueError(f"{inset_pos} unkonw")
    plt_kwargs = {}
    if plot_type == "scatter":
        plot_fn = sns.scatterplot
        # plt_kwargs.update(markersize=markersize)
    elif plot_type == "line":
        plot_fn = sns.lineplot
        markersize = markersize if markers else 0
        plt_kwargs.update(dict(markersize=markersize, errorbar="ci"))
        # plt_kwargs.update(dict(markersize=markersize, errorbar="ci", style=style_by))
    else:
        raise ValueError(f"{plot_type} currently not supported")
    # Setup
    setup_styles(2)
    if labels is None:
        labels = ["perturb_scale"]
    # Get metrics
    base_name = metric_name
    if registry.has_metric(base_name):
        base_metric = registry.get_metric(base_name)
    else:
        base_metric = base_name
    x = x_metric
    print("hello", x, x_metric, registry.has_metric(x_metric))
    if isinstance(x_metric, str) and registry.has_metric(x_metric):
        x_metric = registry.get_metric(x_metric)
        x = x_metric.flat_name
    print(x_metric, x, x_label)

    if isinstance(x_metric, WandbMetric) and x_label is None:
        x_label = x_metric.ylabel
    if x_label is None:
        x_label = NON_METRIC_LABEL_MAP.get(x)
    print(x_label)
    tmp = merged_df.copy()
    # if yscale == "log":
    #     print("uuu log")
    #     tmp[x] = tmp[x] + 5e-3
    #     print(tmp[x].min())
    if filter_bad_runs:
        tmp = filter_bad_runs_func(merged_df, registry)
    # Apply zoom filters
    tmp = tmp.reset_index()
    tmp = tmp[tmp["perturb_mode"] == perturb_method]
    save_prefix += "-" if save_prefix else ""
    suffix = f".{file_extension}"
    if not isinstance(base_metric, str):
        suffix = f"-{base_metric.prefix}{suffix}"
    path = out_dir.joinpath(f"{save_prefix}{perturb_method}{suffix}")
    if yscale == "log":
        path = path.with_stem(f"{path.stem}-log")

    if zoom == "first":
        print("zoooom")
        tmp = tmp[tmp[x] <= zoom_first_step]
        path = path.with_stem(f"{path.stem}-zoom-first")
    elif zoom == "last":
        tmp = tmp[tmp[x] > zoom_last_step]
        path = path.with_stem(f"{path.stem}-zoom-last")

    # Plot
    tmp = tmp.sort_values(x, ascending=True)
    masks = get_labels(tmp, labels, format_labels=False)
    if hue_cnt is None:
        hue_cnt = len(masks)
    hues = get_hues(1, hue_cnt + 1, palette, equidistant=True)[:-1]
    # hues = get_hues(1, hue_cnt + 1, "plasma", equidistant=True)[1:]
    print(len(hues), len(masks))

    for i, (label_, mask) in enumerate(masks.items()):
        # print(label_)
        df = tmp.loc[mask]
        color = hues[i % len(hues)]
        # If label_ is not already a list, convert it to one without splitting strings
        if not isinstance(label_, (list, tuple)):
            label_ = [label_]  # This will keep strings intact

        label = legend_template.format(*label_)
        if isinstance(base_metric, str):
            splits = [Split.TRAIN]
        else:
            splits = [base_metric.split]
        if add_test_line:
            splits.append(Split.TEST)
        for mode in splits:
            axs_to_plot = [ax]
            if mode == Split.TRAIN and inset_fig:
                axs_to_plot.append(inset_ax)
            orig_split = splits[0]
            metric_name_ = metric_name
            if orig_split is not None:
                metric_name_ = metric_name.replace(orig_split.value, mode.value)
            if registry.has_metric(metric_name_):
                metric = registry.get_metric(metric_name_)
                metric_name_ = metric.flat_name
            for ax_to_plot in axs_to_plot:
                put_label = mode == Split.TRAIN or len(splits) == 1
                plot_fn(
                    df,
                    x=x,
                    y=metric_name_,
                    ax=ax_to_plot,
                    marker=marker if markers else "",
                    color=color,
                    label=label if put_label else None,
                    # style=style_by,
                    # linestyle="-" if put_label else "--",
                    **plt_kwargs,
                )

    ax.set_xlabel(x_label)
    if not y_label:
        if isinstance(base_metric, str):
            raise Warning(
                f"Metric not found in the registry, pass y_label for proper formatting."
            )
        else:
            y_label = base_metric.general_ylabel
    ax.set_ylabel(y_label)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    # ax.set_title("\n".join(wrap(f"ResNet 20 - {perturb_method.title()} Perturbance", 30)))
    ax.set_title("\n".join(wrap(title, 30)))
    if inset_fig:
        inset_ax.set_xlabel("")
        inset_ax.set_ylabel("")
        inset_ax.set_xscale(xscale)
        inset_ax.set_xlim(zoom_first_step, zoom_last_step)
        inset_ax.set_yscale(yscale)
        # Adjust font size
        inset_ax.tick_params(axis="both", labelsize=8)  # Smaller font for tick labels
        # inset_ax.set_title("Inset", fontsize=10)  # Smaller title font
        inset_ax.get_legend().remove()
        if zoom_y_last_step is not None:
            inset_ax.set_ylim(bottom=None, top=zoom_y_last_step)
    if plot_type == "line":
        trans = 0.1 if uncertainty else 0
        modify_confidence_alphas(ax, trans)

    if legend_title is None:
        legend_title = " ".join(labels)
    if separate_legend:
        fig_leg = extract_legend(
            ax, ncol=ncols, add_tr_te=add_test_line, title=legend_title
        )
    else:
        fig_leg = None
        ax.legend(title=legend_title)
        #   , bbox_to_anchor=(1, 0.5))
        # ax.legend(title=legend_title, bbox_to_anchor=(1, 0.5))
    if save_fig:
        path.parent.mkdir(exist_ok=True, parents=True)
        # sns.despine()
        if separate_legend:
            plt.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        if fig_leg is not None:
            leg_path = path.as_posix().replace(
                f".{file_extension}", f"-legend.{file_extension}"
            )
            fig_leg.savefig(
                leg_path,
                dpi=300,
                bbox_inches="tight",
            )
            print(f"Saved legend to {leg_path}")
        print(f"Saved figure to {path}")
        plt.show()
    return path
