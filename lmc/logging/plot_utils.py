"""Plotting utilities for creating and customizing matplotlib figures.

This module provides functions for setting up plotting styles, creating color schemes,
handling legends, and other matplotlib-related utilities. It includes support for
LaTeX rendering and custom figure sizing.
"""

from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from matplotlib import lines
from matplotlib import pyplot as plt
from seaborn.distributions import math

MARKERSIZE = 8
FONTSIZE = 13
SCALE = 1.15
AX_SCALE = 1.1
TITLE_SCALE = 1.2
MILA_PURPLE = (109 / 255, 0 / 255, 109 / 255)  # Normalize the RGB values to [0, 1]


NON_METRIC_LABEL_MAP = {
    "perturb_step": "Perturbed at step",
    "perturb_scale": "$\sigma$",
    # "perturb_scale": "Perturb scale",
}

NON_METRIC_LEGEND_MAP = {"perturb_step": "$t={{}}$", "perturb_scale": "$\sigma={{}}$"}

TABLE_MAP = {
    "run.group": "",
    "code_name": "",
    "trainer.opt.optimizer": "Optimizer",
    "trainer.opt.lr": "LR",
    #              "model.lr_schedule.pct_start": "Warmup",
    "trainer.opt.warmup_ratio": "W-up",
    "trainer.opt.lr_scheduler": "LR Scheduler",
    "data.dataset": "Dataset",
    "data.batch_size": "BS",
    "data.mixup": "Aug.",
    "data.use_hflip": "Aug.",
    "trainer.gradient_clip_val": "GC",
    "trainer.opt.weight_decay": "WD",
    "lmc.barrier_test": r"$\mathcal{B}_{\mathrm{te}}$",
    "lmc.barrier_train": r"$\mathcal{B}_{\mathrm{tr}}$",
    "lmc.weighted.barrier_test": r"$\mathcal{B}_{\mathrm{te}}$",
    "lmc.weighted.barrier_train": r"$\mathcal{B}_{\mathrm{tr}}$",
    "model1.test.accuracy": r"$\mathrm{Acc}^1_{\mathrm{te}}$",
    "model1/test/accuracy": r"$\mathrm{Acc}^1_{\mathrm{te}}$",
    "model2.test.accuracy": r"$\mathrm{Acc}^2_{\mathrm{te}}$",
    "model1/test/cross_entropy": r"$\mathrm{CE}^1_{\mathrm{te}}$",
    "model1.train.cross_entropy": r"$\mathrm{CE}^1_{\mathrm{tr}}$",
    "model2.train.cross_entropy": r"$\mathrm{CE}^2_{\mathrm{tr}}$",
    "model1.test.cross_entropy": r"$\mathrm{CE}^1_{\mathrm{te}}$",
    "model2.test.cross_entropy": r"$\mathrm{CE}^2_{\mathrm{te}}$",
    "model1.train.accuracy": r"$\mathrm{Acc}^1_{\mathrm{tr}}$",
    "model2.train.accuracy": r"$\mathrm{Acc}^2_{\mathrm{tr}}$",
    "average_test_acc": r"$\Bar{\mathrm{Acc}}_{\mathrm{te}}$",
    "average_train_acc": r"$\Bar{\mathrm{Acc}}_{\mathrm{tr}}$",
    "reverse_spawning": "RSpawn",
    "data_spawn_steps": "\#Spawn",
    "pretty_group": "Setting",
    "model.model_name": "Model",
    "trainer.training_steps": "Steps",
    "bert_ckpt": "Starting Checkpoint",
}


def get_hues(nalphas=2, ncolors=4, palette="magma", equidistant: bool = False):
    """Generate a list of colors with varying alpha values.

    Args:
        nalphas: int, number of alpha variations for each color
        ncolors: int, number of distinct base colors to generate
        palette: str, name of the seaborn color palette to use
        equidistant: bool, whether to generate equidistant colors from the palette

    Returns:
        list: List of RGBA colors with varying alpha values
    """
    if equidistant:
        # Generate equally spaced values between 0 and 1
        equi_space = np.linspace(0, 1, ncolors)

        # Use these values to get equidistant colors from the palette
        palette = sns.color_palette(palette, as_cmap=True)(equi_space)
    else:
        palette = sns.color_palette(palette, ncolors)
    # palette = plt.cm.tab10(np.linspace(0, 1, ncolors))

    # Pre-generate hues by varying the alpha channel
    alpha_values = np.linspace(1, 0.5, nalphas)
    hue_colors = []

    for base_color in palette:
        rgba_color = mcolors.to_rgba(base_color)
        for alpha in alpha_values:
            hue_color = list(rgba_color[:3]) + [alpha]
            hue_colors.append(hue_color)

    return hue_colors


def modify_confidence_alphas(ax, alpha: float = 0.05):
    """Modify the alpha (transparency) of confidence interval regions in a plot.

    Args:
        ax: matplotlib axis object containing the plot
        alpha: float, desired alpha value for confidence intervals (0-1)
    """
    for shaded_region in ax.collections:
        # # Check if the collection is not a scatterplot
        # if not isinstance(shaded_region, plt.collections.PathCollection):
        shaded_region.set_alpha(alpha)  # Set desired alpha for the shaded region


def setup_styles(
    n_figs: int = 1,
    document_font_size: int = FONTSIZE,
    legend_scale: float = 1,
    use_pgf: bool = False,
):
    """Sets up the styles for the figures, note that this script uses pdflatex to render the plots so that different custom LaTeX commands can be used in figure titles/labels/ etc.

    Args:
        n_figs (int, optional): Number of figures that will be stacked to. Defaults to 1.
    """

    from pathlib import Path

    import matplotlib
    import seaborn as sns

    plt.set_loglevel("error")
    # matplotlib.use("pdf")
    usepackage = []
    if use_pgf:
        # matplotlib.use("pgf")
        from matplotlib.backends.backend_pgf import FigureCanvasPgf

        matplotlib.backend_bases.register_backend("pdf", FigureCanvasPgf)
        # Extract only the filename without extension for usepackage
        usepackage = [
            r"\usepackage{" + pack.with_suffix("").as_posix() + "}"
            for pack in Path("../assets/styles").resolve().rglob("*.sty")
        ]
    fig_width_pt = 246.0 * (n_figs)  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inches
    golden_mean = (math.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    rc = {
        "lines.linewidth": 2.5 * SCALE,
        "lines.markersize": MARKERSIZE * SCALE,
        "font.size": document_font_size * SCALE * n_figs,
        "axes.labelsize": document_font_size * SCALE * n_figs,
        "axes.titlesize": document_font_size * SCALE * n_figs,
        "axes.titleweight": "bold",
        "xtick.labelsize": document_font_size * AX_SCALE * n_figs,
        "ytick.labelsize": document_font_size * AX_SCALE * n_figs,
        "legend.fontsize": document_font_size * SCALE * legend_scale,
        "axes.titley": 1.0 + 0.02 * n_figs,
        "figure.titleweight": "bold",
        "figure.titlesize": document_font_size * TITLE_SCALE * n_figs,
        "savefig.bbox": "tight",
        "figure.figsize": fig_size,
        "font.family": "sans-serif",
    }
    sns.set(
        style="whitegrid",
        rc=rc,
    )

    preamble = "\n".join(
        [
            # r"\usepackage[utf8]{inputenc}",
            # r"\usepackage[T1]{fontenc}",
            # r"\usepackage{url}",
            # r"\usepackage{unicode-math}",
            r"\renewcommand{\rmdefault}{\sfdefault}",
            *usepackage,
            r"\renewcommand*\familydefault{\sfdefault}",
            r"\renewcommand*\familydefault{\sfdefault}",
            r"\usepackage{sansmath}",  # load up the sansmath so that math -> helvet
            r"\sansmath",  # <- tricky! -- gotta actually tell tex to use!
        ]
    )
    plt.rcParams.update({"text.usetex": False, "pgf.rcfonts": True} | rc)

    if use_pgf:
        plt.rcParams.update(
            {
                "text.usetex": False,
                # "pgf.texsystem": "lualatex",  # or "xelatex"
                "pgf.texsystem": "xelatex",  # or "xelatex"
                "pgf.rcfonts": False,
                "pgf.preamble": preamble,
                "text.latex.preamble": preamble,
                "axes.formatter.use_mathtext": False,
            }
            | rc
        )

    return rc


def extract_legend(
    ax, ncol: int = None, add_tr_te: bool = True, title: str = "", **legend_kwargs
):
    """Extract and create a separate figure containing only the legend from a matplotlib plot.

    This function takes a matplotlib axis and creates a new figure containing only the legend,
    with options to customize its appearance and add train/test line style indicators.

    Args:
        ax: matplotlib axis object containing the plot
        ncol: Optional[int], number of columns in the legend. If None, calculates based on number of entries
        add_tr_te: bool, whether to add train/test line style indicators to the legend
        title: str, title for the legend
        **legend_kwargs: Additional keyword arguments passed to matplotlib.pyplot.legend()

    Returns:
        matplotlib.figure.Figure: A new figure containing only the formatted legend

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1,2,3], label='data1')
        >>> ax.plot([2,3,4], label='data2')
        >>> legend_fig = extract_legend(ax, ncol=2, add_tr_te=True)
        >>> legend_fig.savefig('legend.png')
    """
    if add_tr_te:
        # Custom legend for line styles
        solid_line = lines.Line2D(
            [], [], color="black", linestyle="solid", label="Training"
        )
        dashed_line = lines.Line2D(
            [], [], color="black", linestyle="dashed", label="Test"
        )

    ax.legend(
        loc="upper center",
    )
    ax.get_legend().remove()

    fig_leg = plt.figure()

    handles, labels = ax.get_legend_handles_labels()
    if add_tr_te:
        labels.extend(["Train", "Test"])
        handles.extend([solid_line, dashed_line])
    if ncol is None:
        ncol = math.ceil(len(handles) / 4)

    fig_leg = plt.figure()
    ax_leg = fig_leg.add_subplot(111)
    legend = ax_leg.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol,
        frameon=False,
        title=title,
        **legend_kwargs,
    )
    ax_leg.axis("off")

    # Draw the legend
    fig_leg.canvas.draw()

    # Calculate the bounding box of the legend
    bbox = legend.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height

    # Set figure size based on legend
    fig_leg.set_size_inches(width + 0.2, height + 0.2)
    return fig_leg


def turn_off_tex():
    import matplotlib.pyplot as plt

    # Disable TeX rendering for all text
    plt.rcParams["text.usetex"] = False


# useful for combining legend and fig
def combine_and_save_figures(fig1, fig2, save_path):
    # Create a new figure
    fig_combined, ax_combined = plt.subplots()

    # Add the contents of fig1 to the combined figure
    ax_combined.axis("off")  # Hide axes if necessary
    ax_combined.add_axes(fig1.axes[0].get_position())
    ax_combined.plot(
        fig1.axes[0].lines[0].get_xdata(), fig1.axes[0].lines[0].get_ydata()
    )
    ax_combined.legend()

    # Add the contents of fig2 to the combined figure
    ax_combined.add_axes(fig2.axes[0].get_position())
    ax_combined.plot(
        fig2.axes[0].lines[0].get_xdata(), fig2.axes[0].lines[0].get_ydata()
    )

    # Save the combined figure
    fig_combined.savefig(save_path)
    plt.close(fig_combined)  # Close the combined figure to avoid displaying it


def plot_timeseries_from_wandb():
    pass
