from pathlib import Path
from textwrap import wrap
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lmc.logging.plot_utils import (MARKERSIZE, extract_legend, get_hues,
                                    modify_confidence_alphas, setup_styles)
from lmc.logging.report_utils import check_nulls
from lmc.logging.report_utils import filter_bad_runs as filter_bad_runs_func
from lmc.logging.report_utils import get_labels
from lmc.logging.wandb_registry import MetricCategory, PermMethod, Split


def plot_perturb_barrier(
    merged_df: pd.DataFrame,
    registry: "WandbMetricsRegistry",
    perturb_method: str,
    metric_name: str,
    perm: bool = False,
    labels: Optional[List[str]] = None,
    zoom: Optional[Literal["first", "last"]] = None,  # None, "first", or "last"
    out_dir: Path = Path("outputs"),
    filter_bad_runs: bool = True,
    zoom_first_step: int = 10,
    zoom_last_step: int = 380,
    max_scale: float = 0.5,
    perm_method: Optional[PermMethod] = None,  # "am" or "wm" if metric_type is "perm"
    separate_legend: bool = True,
    save_fig: bool = True
) -> Path:
    """Plot LMC or Permutation barrier metrics.
    
    Args:
        merged_df: DataFrame with merged metrics
        registry: WandbMetricsRegistry instance
        perturb_method: Perturbation method ("batch", etc.)
        metric_type: Type of metric to plot ("lmc" or "perm")
        zoom: Zoom type (None, "first", or "last")
        out_dir: Output directory for plots
        min_test_acc: Minimum average test accuracy filter
        min_model_acc: Minimum model accuracy filter
        min_epoch: Minimum epoch filter
        zoom_first_step: Step cutoff for "first" zoom
        zoom_last_step: Step cutoff for "last" zoom
        max_scale: Maximum perturbation scale for no zoom
        perm_method: Permutation method if metric_type is "perm"
    """
    # Setup
    setup_styles(2)
    hues = get_hues(1, 5, "plasma")
    if labels is None:
        labels = ["perturb_scale"]
    # Get metrics
    base_name = metric_name
    base_metric = registry.get_metric(base_name)

    tmp = merged_df.copy()
    if filter_bad_runs:
        tmp = filter_bad_runs_func(merged_df, registry)
    # Apply zoom filters
    tmp = tmp.reset_index()
    tmp = tmp[tmp["perturb_mode"] == perturb_method]
    path = out_dir.joinpath(f"{perturb_method}-{base_metric.prefix}").with_suffix(".pdf")
    if zoom == "first":
        tmp = tmp[tmp["perturb_step"] <= zoom_first_step]
        path = path.with_stem(f"{path.stem}-zoom-first")
    elif zoom == "last":
        tmp = tmp[tmp["perturb_step"] > zoom_last_step]
        path = path.with_stem(f"{path.stem}-zoom-last")

    # Plot
    fig, ax = plt.subplots()
    tmp = tmp.sort_values("perturb_step")
    masks = get_labels(tmp, labels, format_labels=False)

    for i, (label_, mask) in enumerate(masks.items()):
        df = tmp.loc[mask]
        color = hues[i % len(hues)]
        label = rf"$\lambda={label_}$"
        
        for mode in [Split.TRAIN, Split.TEST]:
            orig_split = base_metric.split
            metric_name_ = metric_name.replace(orig_split.value, mode.value)
            metric = registry.get_metric(metric_name_)

            sns.lineplot(df, x="perturb_step", y=metric.flat_name, ax=ax,
                   marker="o", color=color, 
                   label=label if mode==Split.TRAIN else None,
                   linestyle="-" if mode==Split.TRAIN else "--",
                   markersize=MARKERSIZE)

    plt.xlabel("Perturbed at Step")
    plt.ylabel(base_metric.general_ylabel)
    plt.title("\n".join(wrap(f"ResNet 20 - {perturb_method.title()} Perturbance", 30)))
    modify_confidence_alphas(ax, 0.1)

    if separate_legend:
        fig_leg = extract_legend(ax, ncol=2, add_tr_te=True)
    else:
    # if zoom == "first":
        fig_leg = None
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if save_fig:
        path.parent.mkdir(exist_ok=True, parents=True)
        
        fig.savefig(path)
        if fig_leg is not None:
            fig_leg.savefig(path.as_posix().replace(".pdf", "-legend.pdf"), dpi=300, bbox_inches="tight")
        
    plt.show()
    return path