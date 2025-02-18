from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import chain
from typing import Dict, List, Optional, Tuple


class LMCMetricType(Enum):
    ACCURACY = "barrier"
    # TODO: not sure how these are logged
    ERROR = "err"
    LOSS = "loss/weighted/barrier"
    MAXINT = "maxint"
    LOSS_MAXINT = "loss/weighted/maxint"


class MetricCategory(Enum):
    TOP_3_ACCURACY = "top_3_accuracy"
    ACCURACY = "accuracy"
    CROSS_ENTROPY = "cross_entropy"
    LEARNING_RATE = "lr"
    L2_DISTANCE = "l2_distance"
    COUNT = "count"
    NOISE = "noise"
    LMC_ACCURACY = "barrier"
    # TODO: not sure how these are logged
    LMC_ERROR = "err"
    LMC_LOSS = "loss/weighted/barrier"
    LMC_MAXINT = "maxint"
    LMC_LOSS_MAXINT = "loss/weighted/maxint"
    ## LANGUAGE
    PERPLEXITY = "perplexity"
    LOSS = "loss"
    CORRELATION = "correlation"
    NLP_OTHER = "nlp_other"


class Split(Enum):
    TRAIN = "train"
    TEST = "test"


class PermMethod(Enum):
    AM = "am"
    WM = "wm"


@dataclass
class WandbMetric:
    """Single Wandb metric configuration with LaTeX support."""

    log_name: str
    ylabel: str
    prefix: str
    category: Optional[MetricCategory] = None
    split: Optional[Split] = None
    general_ylabel: str = None
    flat_name: str = field(init=False)

    def __post_init__(self):
        self.flat_name = self.log_name.replace("/", ".")

        # Auto-detect category if not provided
        if self.category is None:
            for cat in MetricCategory:
                if f"/{cat.value}" in self.log_name:
                    self.category = cat
                    break

        # Auto-detect split if not provided
        if self.split is None:
            for split in Split:
                if split.value in self.log_name:
                    self.split = split
                    break

        if self.general_ylabel is None:
            replaces = ["train", "Tr", "Te", "test"]
            for sub in replaces:
                self.general_ylabel = self.ylabel.replace(sub, "")


@dataclass
class MetricTemplate:
    """Template for generating metrics for multiple models."""

    base_path: str
    ylabel_template: str
    prefix_template: str
    category: MetricCategory


@dataclass
class MetricsRegistry:
    """Base class for metrics registries providing common access patterns."""

    metrics: Dict[str, WandbMetric] = field(default_factory=dict)

    def get_log_names(self) -> List[str]:
        """Get all wandb logging names."""
        return [m.log_name for m in self.metrics.values()]

    def get_names(self) -> List[str]:
        """Get all metric keys."""
        return list(self.metrics.keys())

    def get_flat_names(self) -> List[str]:
        """Get all flattened metric names."""
        return [m.flat_name for m in self.metrics.values()]

    def get_metrics_by_category(
        self,
        category: MetricCategory,
        split: Optional[Split] = None,
        pattern: Optional[str] = None,
    ) -> "MetricsRegistry":
        """Get all metrics of a specific category and optionally of a specific split."""
        filtered_metrics = dict(
            (name, metric)
            for (name, metric) in self.metrics.items()
            if metric.category == category  # Direct comparison
            and (split is None or metric.split == split)  # Direct comparison
            and (pattern is None or pattern in metric.log_name)
        )
        return MetricsRegistry(filtered_metrics)

    def add_metric(self, key: str, metric: WandbMetric):
        """Add a new metric to the registry."""
        self.metrics[key] = metric

    def get_metric(self, key: str) -> WandbMetric:
        """Get a metric by its key."""
        return self.metrics[key]

    def __iter__(self):
        """Iterate through (key, metric) pairs in the registry."""
        return iter(self.metrics.items())

    # Optionally add len support too
    def __len__(self):
        return len(self.metrics)

    def update(self, d: "MetricsRegistry") -> None:
        for name, metric in d.metrics.items():  # Access the metrics dict directly
            self.metrics[name] = metric


@dataclass(init=False)
class WandbMetricsRegistry(MetricsRegistry):
    n_models: int  # = 1
    # metrics: Dict[str, WandbMetric] = field(default_factory=dict)
    _templates: Dict[str, MetricTemplate] = field(default_factory=dict)
    _lmc_templates: Dict[str, MetricTemplate] = field(default_factory=dict)

    def __init__(self, n_models, metrics=None, templates=None, lmc_templates=None):
        self.n_models = n_models
        if metrics is None:
            metrics = {}
        if templates is None:
            templates = {}
        if lmc_templates is None:
            lmc_templates = {}
        super().__init__(metrics)
        self._templates = templates
        self._lmc_templates = lmc_templates
        self.__post_init__()

    def __post_init__(self):
        self._initialize_templates()
        self._initialize_base_metrics()
        self._initialize_model_metrics()
        self._initialize_lmc_metrics()

    def _initialize_templates(self):
        """Initialize metric templates for different categories."""
        self._templates.update(
            {
                "accuracy": MetricTemplate(
                    "model{}/{}/accuracy",
                    "$\\mathrm{{Acc}}^{{{}}}_{{\\mathrm{{{}}}}}$",
                    "model{}-acc",
                    MetricCategory.ACCURACY,
                ),
                "top3_accuracy": MetricTemplate(
                    "model{}/{}/top_3_accuracy",
                    "$\\mathrm{{Top\\ 3\\ Acc}}^{{{}}}_{{\\mathrm{{{}}}}}$",
                    "model{}-top3-acc",
                    MetricCategory.TOP_3_ACCURACY,
                ),
                "cross_entropy": MetricTemplate(
                    "model{}/{}/cross_entropy",
                    "$\\mathrm{{CE}}^{{{}}}_{{\\mathrm{{{}}}}}$",
                    "model{}-ce",
                    MetricCategory.CROSS_ENTROPY,
                ),
            }
        )
        ## language
        self._templates.update(
            {
                "perplexity": MetricTemplate(
                    "model{}/{}/perplexity",
                    "$\\mathrm{{Perplexity}}^{{{}}}_{{\\mathrm{{{}}}}}$",
                    "model{}-perplexity",
                    MetricCategory.PERPLEXITY,
                ),
                "top3_accuracy": MetricTemplate(
                    "model{}/{}/top_3_accuracy",
                    "$\\mathrm{{Top\\ 3\\ Acc}}^{{{}}}_{{\\mathrm{{{}}}}}$",
                    "model{}-top3-acc",
                    MetricCategory.TOP_3_ACCURACY,
                ),
                "cross_entropy": MetricTemplate(
                    "model{}/{}/cross_entropy",
                    "$\\mathrm{{CE}}^{{{}}}_{{\\mathrm{{{}}}}}$",
                    "model{}-ce",
                    MetricCategory.CROSS_ENTROPY,
                ),
                "exact_match": MetricTemplate(
                    "model{}/{}/exact_match",
                    "Exact Match",
                    "model{}-em",
                    MetricCategory.NLP_OTHER,
                ),
                "f1": MetricTemplate(
                    "model{}/{}/f1",
                    "F1",
                    "model{}-f1",
                    MetricCategory.NLP_OTHER,
                ),
                "pearson_correlation": MetricTemplate(
                    "model{}/{}/pearson_correlation",
                    "Pearson Correlation",
                    "model{}-pearson",
                    MetricCategory.CORRELATION,
                ),
                "spearman_correlation": MetricTemplate(
                    "model{}/{}/spearman_correlation",
                    "Spearman Correlationmodel{}-spearman",
                    "model{}-spearman",
                    MetricCategory.CORRELATION,
                ),
            }
        )
        self._lmc_templates.update(
            {
                "barrier": MetricTemplate(
                    # "lmc_{}_{}_{}", # split, ind0, ind1
                    "{}/lmc/weighted/barrier_{}",  # basekey, split
                    "$\\mathcal{{B}}^{{{}}}_{{\\mathrm{{{}}}}}$",  # split abb
                    "{}-weighted-barrier",  # base_key
                    MetricCategory.LMC_ACCURACY,
                ),
                "loss": MetricTemplate(
                    # f"lmc_loss_{split.value}_{ind0}_{ind1}",
                    "{}/lmc/loss/weighted/barrier_{}",
                    "$\\mathcal{{B}}^{{{}}}_{{CE_\\mathrm{{{}}}}}$",  # split abb
                    "{}-loss-weighted-barrier",
                    MetricCategory.LMC_LOSS,
                ),
                "loss_maxint": MetricTemplate(
                    # f"lmc_loss_maxint_{}_{ind0}_{ind1}",
                    "{}/lmc/loss/weighted/maxint_{}",
                    "CE^{{{}}}_{{\\mathrm{{{}}}}} Max int",
                    "{}-loss-weighted-maxint",
                    MetricCategory.LMC_LOSS_MAXINT,
                ),
                "maxint": MetricTemplate(
                    # f"lmc_maxint_{}_{ind0}_{ind1}",
                    "{}/lmc/weighted/maxint_{}",
                    "Acc^{{{}}}_{{\\mathrm{{{}}}}} Max int",
                    "{}-weighted-maxint",
                    MetricCategory.LMC_MAXINT,
                ),
            }
        )

    def _initialize_lmc_metrics(self):
        """Initialize LMC and permutation metrics."""
        for ind0 in range(self.n_models - 1):
            ind1 = ind0 + 1

            for split in Split:
                split_abbrev = "Te" if split == Split.TEST else "Tr"

                # LMC metrics
                for template_name, template in self._lmc_templates.items():
                    # metric_key format: "lmc_{split.value}_{ind0}_{ind1}"
                    template_name = (
                        "" if template_name == "barrier" else f"_{template_name}"
                    )
                    metric_key = f"lmc{template_name}_{split.value}_{ind0}_{ind1}"
                    base_key = f"lmc-{ind0}-{ind1}"
                    ind_key = f"({ind0}-{ind1})" if self.n_models > 2 else ""

                    self.add_metric(
                        metric_key,
                        WandbMetric(
                            template.base_path.format(base_key, split.value),
                            template.ylabel_template.format(ind_key, split_abbrev),
                            template.prefix_template.format(base_key, split.value),
                            category=template.category,
                            split=split,
                            general_ylabel=template.ylabel_template.format(ind_key, ""),
                        ),
                    )

                    # Permutation metrics
                    for method in PermMethod:
                        metric_key = f"perm_{method.value}{template_name}_{split.value}_{ind0}_{ind1}"
                        base_key = f"perm/{method.value}-{ind0}-{ind1}"

                        self.add_metric(
                            metric_key,
                            WandbMetric(
                                template.base_path.format(base_key, split.value),
                                template.ylabel_template.format(ind_key, split_abbrev),
                                template.prefix_template.format(base_key, split.value),
                                category=template.category,
                                split=split,
                                general_ylabel=template.ylabel_template.format(
                                    ind_key, ""
                                ),
                            ),
                        )

    def _initialize_base_metrics(self):
        """Initialize non-model-specific metrics."""
        self.add_metric("epoch", WandbMetric("epoch", "Epoch", ""))

    def _initialize_model_metrics(self):
        """Initialize metrics for each model."""
        for model_idx in range(1, self.n_models + 1):
            # Learning rate
            self.add_metric(
                f"lr_{model_idx}",
                WandbMetric(f"lr/model{model_idx}", f"$\\eta_{{{model_idx}}}$", "lr"),
            )

            # Model metrics for both train and test
            for split in Split:
                split_abbrev = "Tr" if split == Split.TRAIN else "Te"

                for template_name, template in self._templates.items():
                    metric_key = f"{split.value}_{template_name}_{model_idx}"
                    self.add_metric(
                        metric_key,
                        WandbMetric(
                            template.base_path.format(model_idx, split.value),
                            template.ylabel_template.format(model_idx, split_abbrev),
                            template.prefix_template.format(model_idx),
                            category=template.category,
                            split=split,
                            general_ylabel=template.ylabel_template.format(
                                model_idx, ""
                            ),
                        ),
                    )

            # L2 and noise metrics
            self.add_metric(
                f"l2_at_init_{model_idx}",
                WandbMetric(
                    f"static/l2_at_init/{model_idx}",
                    f"L2 at Init Model {model_idx}",
                    f"l2-{model_idx}",
                    category=MetricCategory.L2_DISTANCE,  # Using L2_DISTANCE category since it's a norm
                ),
            )

            self.add_metric(
                f"noise_l2_{model_idx}",
                WandbMetric(
                    f"static/noise/{model_idx}-l2",
                    f"Noise L2 for Model {model_idx}",
                    f"noise{model_idx}",
                    category=MetricCategory.L2_DISTANCE,  # Using L2_DISTANCE category since it's a norm
                ),
            )
            self.add_metric(
                f"noise_l2_scaled_{model_idx}",
                WandbMetric(
                    f"static/noise/{model_idx}-l2-scaled",
                    f"Effective Noise L2 for Model {model_idx}",
                    f"noise_scaled{model_idx}",
                    category=MetricCategory.L2_DISTANCE,  # Using L2_DISTANCE category since it's a norm
                ),
            )
            self.add_metric(
                f"grad_count_{model_idx}",  # Using model_idx to track per model
                WandbMetric(
                    f"static/avg_grad_count/{model_idx}",  # log_name following your convention
                    f"Total number of parameters with gradient for Model {model_idx}",  # ylabel
                    f"grad_count_{model_idx}",  # prefix
                    category=MetricCategory.COUNT,  # Using L2_DISTANCE category since it's a norm
                ),
            )
            for datapoints in [1, 5, -1]:
                self.add_metric(
                    f"grad_norm_{model_idx}_on_{datapoints}",  # Using model_idx to track per model
                    WandbMetric(
                        f"static/avg_grad_norm/{model_idx}/on_{datapoints}",  # log_name following your convention
                        f"Average Gradient L2 Norm for Model {model_idx} on {datapoints} Datapoints",  # ylabel
                        f"avg_grad_norm_{model_idx}/on_{datapoints}",  # prefix
                        category=MetricCategory.L2_DISTANCE,  # Using L2_DISTANCE category since it's a norm
                    ),
                )
            self.add_metric(
                f"grad_count_{model_idx}",  # Using model_idx to track per model
                WandbMetric(
                    f"static/grad_count/{model_idx}",  # log_name following your convention
                    f"Parameter Count for Model {model_idx}",  # ylabel
                    f"grad_count_{model_idx}",  # prefix
                    category=MetricCategory.L2_DISTANCE,  # Using L2_DISTANCE category since it's a norm
                ),
            )

            self.add_metric(
                f"l2_dist_from_init_{model_idx}",
                WandbMetric(
                    f"l2/dist_from_init_{model_idx}",
                    rf"$\lVert\theta_{{t_{{{model_idx}}}}} - \theta_{{{0}_{{{model_idx}}}}} \rVert_F$",
                    f"l2_dist_from_init_{model_idx}",
                    category=MetricCategory.L2_DISTANCE,
                ),
            )

            self.add_metric(
                f"cos_dist_from_init_{model_idx}",
                WandbMetric(
                    f"cos/dist_from_init_{model_idx}",
                    rf"$\cos \left ( \theta_{{t_{{{model_idx}}}}}, \theta_{{{0}_{{{model_idx}}}}} \right )$",
                    f"cos_dist_from_init_{model_idx}",
                    category=MetricCategory.L2_DISTANCE,
                ),
            )

            for next_el_ind in range(model_idx + 1, self.n_models + 1):
                # if model_idx < self.n_models:
                self.add_metric(
                    f"l2_dist_{model_idx}-{next_el_ind}",
                    WandbMetric(
                        f"l2/dist_{model_idx}-{next_el_ind}",
                        rf"$\lVert\theta_{{t_{{{model_idx}}}}} - \theta_{{t_{{{next_el_ind}}}}} \rVert_F$",
                        f"l2_dist_{model_idx}-{next_el_ind}",
                        category=MetricCategory.L2_DISTANCE,
                    ),
                )
                self.add_metric(
                    f"cos_dist_{model_idx}-{next_el_ind}",
                    WandbMetric(
                        f"cos/dist_{model_idx}-{next_el_ind}",
                        rf"\cos \left ( \theta_{{t_{{{model_idx}}}}} - \theta_{{t_{{{next_el_ind}}}}} \right )$",
                        f"cos_dist_{model_idx}-{next_el_ind}",
                        category=MetricCategory.L2_DISTANCE,
                    ),
                )

    def has_metric(self, metric_name: str) -> bool:
        if metric_name in self.metrics.keys():
            return True
        return False

    def get_model_metrics(self, model_idx: int) -> MetricsRegistry:
        """Get all metrics associated with a specific model."""
        return MetricsRegistry(
            dict(
                (name, metric)
                for (name, metric) in self.metrics.items()
                if f"model{model_idx}" in metric.log_name
            )
        )

    def get_lmc_metrics(
        self,
        split: Optional[Split] = None,
        metric_type: Optional[LMCMetricType] = None,
        perm_method: Optional[PermMethod] = None,
    ) -> MetricsRegistry:
        """Get LMC metrics with optional filtering.

        Args:
            split: Optional Split to filter metrics
            metric_type: Optional LMCMetricType to filter specific type of LMC metrics
        """
        filtered_metrics = dict(
            (name, metric)
            for (name, metric) in self.metrics.items()
            if name.startswith("lmc_")
            and (split is None or split == metric.split)
            and (metric_type is None or metric_type == metric.category)
            and (perm_method is None or perm_method.value in metric.log_name)
        )
        return MetricsRegistry(filtered_metrics)

    def get_perm_metrics(
        self,
        method: Optional[PermMethod] = None,
        split: Optional[Split] = None,
        metric_type: Optional[LMCMetricType] = None,
    ) -> MetricsRegistry:
        """Get permutation metrics with optional filtering.

        Args:
            method: Optional PermMethod to filter specific permutation method
            split: Optional Split to filter metrics
            metric_type: Optional LMCMetricType to filter specific type of metrics
        """
        filtered_metrics = dict(
            (name, metric)
            for (name, metric) in self.metrics.items()
            if name.startswith("perm_")
            and (method is None or f"_{method.value}_" in name)
            and (split is None or split == metric.split)
            and (metric_type is None or metric_type == metric.category)
        )
        return MetricsRegistry(filtered_metrics)


# Example usage:
if __name__ == "__main__":
    registry = WandbMetricsRegistry(n_models=2)

    # Get all metrics for model 1
    model1_metrics = registry.get_model_metrics(1)

    # Get all accuracy metrics
    accuracy_metrics = registry.get_metrics_by_category(MetricCategory.ACCURACY)

    # Get a specific metric
    train_acc_1 = registry.get_metric("train_accuracy_1")

    lmc_loss_metrics = registry.get_metrics_by_category(MetricCategory.LMC_LOSS)
    lmc_acc_metrics = registry.get_metrics_by_category(MetricCategory.LMC_ACCURACY)

    print(accuracy_metrics.get_log_names(), "\n")
    print(model1_metrics.get_log_names(), "\n")
    print(lmc_loss_metrics.get_log_names(), "\n")
    print(lmc_acc_metrics.get_log_names(), "\n")
