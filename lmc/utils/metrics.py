from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from tabulate import tabulate

from lmc.data.data_stats import TaskType


def mixup_topk_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    targets_shuffled: Optional[torch.Tensor] = None,
    k: int = 3,
    avg: bool = False,
):
    # TODO: may need to change this in the future, to control for how mixup is handled in torch loader
    """computes the top-1 and top-k accuracy with or without mixup"""
    num = preds.size(0) if avg else 1
    k = min(preds.size(1), k)
    _, top_k_inds = torch.topk(preds, k)
    topk = (
        1.0
        / num
        * torch.sum(torch.any(top_k_inds == targets.unsqueeze(dim=1), dim=1), dim=0)
    )
    acc = 1.0 / num * torch.sum(top_k_inds[:, 0].eq(targets), dim=0)

    if targets_shuffled is not None:
        topk_shuffled = (
            1.0
            / num
            * torch.sum(
                torch.any(
                    torch.logical_or(
                        top_k_inds == targets.unsqueeze(1),
                        top_k_inds == targets_shuffled.unsqueeze(1),
                    ),
                    dim=1,
                ),
                dim=0,
            )
        )
        acc_shuffled = (
            1.0
            / num
            * torch.sum(
                torch.logical_or(
                    top_k_inds[:, 0].eq(targets), top_k_inds[:, 0].eq(targets_shuffled)
                ),
                dim=0,
            )
        )

        return acc_shuffled, topk_shuffled
    return acc, topk


class AverageMeter(object):
    """Computes and stores the average and current value
    Script by Gregor Bachmann (Original: https://github.com/gregorbachmann/scaling_mlps/blob/af88bdbfd99c1bfb089af273995ffbf88e8fece8/utils/metrics.py#L5)
    Date: August 1, 2023

    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val is None:
            return
        self.val = val
        self.sum += val * n
        self.count += n

    def get_avg(self, percentage=False):
        cnt = self.count if self.count != 0 else 1
        return self.sum / cnt if not percentage else self.sum * 100 / cnt


### Language metrics
def compute_qa_metrics(predictions, references, tokenizer):
    """Compute QA metrics (Exact Match and F1)"""
    metric = evaluate.load("squad")
    results = metric.compute(predictions=predictions, references=references)
    return results["exact_match"], results["f1"]


def compute_classification_metrics(predictions, references):
    """Compute classification metrics"""
    assert predictions.shape == references.shape
    n = predictions.size(0)
    return 1.0 / n * (predictions == references).sum().item()
    return float(predictions == references).mean().item()
    return float((predictions == references).mean())


def compute_f1_metrics(predictions, references):
    """Compute F1 score (used for MRPC, QQP)"""
    return f1_score(y_true=references, y_pred=predictions)
    metric = evaluate.load("f1")
    results = metric.compute(
        predictions=predictions,
        references=references,
    )
    return results["f1"]


def compute_matthews_correlation(predictions, references):
    """Compute Matthews Correlation (used for CoLA)"""
    return matthews_corrcoef(references, predictions)


def compute_pearson_spearman_corr(predictions, references):
    """Compute Pearson & Spearman correlation (used for STS-B)"""
    pearson_corr, _ = pearsonr(references, predictions)
    spearman_corr, _ = spearmanr(references, predictions)
    return {
        "pearson_correlation": pearson_corr,
        "spearman_correlation": spearman_corr,
    }


def compute_matthews_correlation_hf(predictions, references):
    """Compute Matthews Correlation (used for CoLA)"""
    metric = evaluate.load("matthews_correlation")
    results = metric.compute(predictions=predictions, references=references)
    return results["matthews_correlation"]


def compute_pearson_spearman_corr_hf(predictions, references):
    """Compute Pearson & Spearman correlation (used for STS-B)"""
    metric = evaluate.load("glue", "stsb")
    results = metric.compute(predictions=predictions, references=references)
    return {
        "pearson_correlation": results["pearson"],
        "spearman_correlation": results["spearmanr"],
    }


METRIC_TO_FUNCTION = {
    "matthews_correlation": compute_matthews_correlation,
    "pearson_correlation": compute_pearson_spearman_corr,
    "spearman_correlation": compute_pearson_spearman_corr,
    "f1": compute_f1_metrics,
    "squad": compute_qa_metrics,
    "qa": compute_qa_metrics,
    "accuracy": compute_classification_metrics,
    "acc": compute_classification_metrics,
}


def compute_metrics(metrics: List[str], predictions, references):
    d = dict()
    for metric_name in metrics:
        if metric_name not in METRIC_TO_FUNCTION:
            raise ValueError(f"Unkown metric name {metric_name}.")
        func = METRIC_TO_FUNCTION[metric_name]
        res = func(predictions, references)
        if isinstance(res, dict):
            d.update(res)
        else:
            d[metric_name] = res
    return d


@dataclass
class Metrics:
    total_acc: AverageMeter = field(default_factory=AverageMeter)
    total_topk: AverageMeter = field(default_factory=AverageMeter)
    total_loss: AverageMeter = field(default_factory=AverageMeter)
    cross_entropy: AverageMeter = field(default_factory=AverageMeter)

    # New NLP-specific metrics
    perplexity: AverageMeter = field(default_factory=AverageMeter)
    exact_match: AverageMeter = field(default_factory=AverageMeter)
    f1_score: AverageMeter = field(default_factory=AverageMeter)
    ## additional
    matthews_correlation: AverageMeter = field(default_factory=AverageMeter)
    pearson_correlation: AverageMeter = field(default_factory=AverageMeter)
    spearman_correlation: AverageMeter = field(default_factory=AverageMeter)

    def reset(self):
        self.total_acc.reset()
        self.total_topk.reset()
        self.total_loss.reset()
        self.cross_entropy.reset()
        self.perplexity.reset()
        self.exact_match.reset()
        self.matthews_correlation.reset()
        self.pearson_correlation.reset()
        self.spearman_correlation.reset()
        self.f1_score.reset()

    def update(
        self,
        accuracy: Optional[float] = None,
        topk_accuracy: Optional[float] = None,
        total_loss: Optional[float] = None,
        cross_entropy: Optional[float] = None,
        perplexity: Optional[float] = None,
        exact_match: Optional[float] = None,
        f1: Optional[float] = None,
        n: int = 1,
        matthews_correlation: Optional[float] = None,
        pearson_correlation: Optional[float] = None,
        spearman_correlation: Optional[float] = None,
    ):
        """
        Update metrics based on task type:
        - Classification/NLI: accuracy, cross_entropy
        - Generation: perplexity, loss
        - QA: exact_match, f1_score
        """
        if accuracy is not None:
            self.total_acc.update(accuracy, n)
        if topk_accuracy is not None:
            self.total_topk.update(topk_accuracy, n)
        if total_loss is not None:
            self.total_loss.update(total_loss, n)
        if cross_entropy is not None:
            self.cross_entropy.update(cross_entropy, n)
        if perplexity is not None:
            self.perplexity.update(perplexity, n)
        if exact_match is not None:
            self.exact_match.update(exact_match, n)
        if matthews_correlation is not None:
            self.matthews_correlation.update(matthews_correlation, n)
        if pearson_correlation is not None:
            self.pearson_correlation.update(pearson_correlation, n)
        if spearman_correlation is not None:
            self.spearman_correlation.update(spearman_correlation, n)
        if f1 is not None:
            self.f1_score.update(f1, n)

    def get_metrics(
        self, percentage: bool = False, task_type: TaskType = TaskType.CLASSIFICATION
    ) -> Dict[str, float]:
        """Get relevant metrics based on task type"""
        metrics = {}

        def update_return_dct(metric_name, attribute_name: str = None):
            attribute_name = metric_name if attribute_name is None else attribute_name
            if getattr(self, attribute_name).count > 0:
                metrics[metric_name] = getattr(self, attribute_name).get_avg(percentage)

        update_return_dct("cross_entropy")
        update_return_dct("accuracy", "total_acc")
        update_return_dct("top_3_accuracy", "total_topk")

        update_return_dct("perplexity")
        update_return_dct("exact_match")
        update_return_dct("f1", "f1_score")
        update_return_dct("matthews_correlation")
        update_return_dct("pearson_correlation")
        update_return_dct("spearman_correlation")
        return metrics
        metrics = {"loss": self.total_loss.avg}

        if task_type == TaskType.GENERATION:
            metrics.update(
                {
                    "perplexity": self.perplexity.avg,
                    "cross_entropy": self.cross_entropy.avg,
                }
            )

        elif task_type in [
            TaskType.CLASSIFICATION,
            TaskType.NATURAL_LANGUAGE_INFERENCE,
        ]:
            metrics.update(
                {
                    "accuracy": self.total_acc.avg,
                    "cross_entropy": self.cross_entropy.avg,
                }
            )
            if self.total_topk.count > 0:
                metrics["top_k_accuracy"] = self.total_topk.avg

        elif task_type == TaskType.QUESTION_ANSWERING:
            metrics.update(
                {"exact_match": self.exact_match.avg, "f1_score": self.f1_score.avg}
            )

        return metrics


def report_results(log_dct, epoch: int, n_models: int = 1):
    """given the log_dct, generates and prints a nice looking summary of the training results for the given epoch"""
    # TODO: modify these with registry
    for prefix, key_prefix in [
        ("", ""),
        ("Permuted - ", "matched-perm-1-2/"),
        ("Permuted (Act.) - ", "activation-matching1-2/"),
    ]:
        contains_lmc = f"{key_prefix}lmc/barrier_train" in log_dct.keys()
        if contains_lmc:
            # Specified keys for cross-entropy and accuracy
            ce_keys = [
                "lmc/loss/barrier_train",
                "lmc/loss/barrier_test",
                "lmc/loss/max_err_alpha_train",
                "lmc/loss/max_err_alpha_test",
            ]
            ce_keys_weighted = [
                "lmc/loss/weighted/barrier_train",
                "lmc/loss/weighted/barrier_test",
            ]
            accuracy_keys = [
                "lmc/barrier_train",
                "lmc/barrier_test",
                "lmc/max_econfirr_alpha_train",
                "lmc/max_err_alpha_test",
            ]
            accuracy_keys_weighted = [
                "lmc/weighted/barrier_train",
                "lmc/weighted/barrier_test",
            ]

            # Custom headers
            headers = ["barr/train", "barr/test", "alpha/train", "alpha/test"]

            # Extracting values
            ce_values = [log_dct.get(f"{key_prefix}{key}", None) for key in ce_keys]
            ce_values_weighted = [
                log_dct.get(f"{key_prefix}{key}", None) for key in ce_keys_weighted
            ]
            accuracy_values = [
                log_dct.get(f"{key_prefix}{key}", None) for key in accuracy_keys
            ]
            accuracy_values_weighted = [
                log_dct.get(f"{key_prefix}{key}", None)
                for key in accuracy_keys_weighted
            ]

            # Combining rows with labels
            rows = [
                accuracy_values,
                ce_values,
                accuracy_values_weighted,
                ce_values_weighted,
            ]
            start_columns = ["", ""]
            row_labels = [
                [f"{prefix}Accuracy", ""],
                [f"{prefix}Cross-Entropy", ""],
                [f"{prefix}Accuracy w. alpha scaling", ""],
                [f"{prefix}CE with alpha scaling", ""],
            ]
            title = f"LMC - Epoch {epoch}"
            print_to_rich(title, headers, rows, start_columns, row_labels)
            print(
                tabulate(rows, headers=headers, tablefmt="grid", showindex=row_labels)
            )

    # Specified keys
    modes = ["test", "train"]
    acc_keys = [
        f"model{i}/{mod}/accuracy" for mod in modes for i in range(1, n_models + 1)
    ]
    ce_keys = [
        f"model{i}/{mod}/cross_entropy" for mod in modes for i in range(1, n_models + 1)
    ]

    # Custom headers
    headers = [f"{mod}/{i}" for mod in modes for i in range(1, n_models + 1)]

    start_columns = ["", ""]
    row_labels = [["Accuracy", ""], ["Cross-Entropy", ""]]
    # Extracting values
    acc_values = [log_dct.get(key, None) for key in acc_keys]
    ce_values = [log_dct.get(key, None) for key in ce_keys]
    rows = [acc_values, ce_values]
    title = f"Epoch {epoch}"
    print_to_rich(title, headers, rows, start_columns, row_labels)


def print_to_rich(
    title: str,
    headers: List[str],
    rows: List[List[Any]],
    start_columns: List[str],
    row_labels: List[List[Any]],
) -> None:
    """
    Displays a formatted table in the console using the rich library.

    Args:
        title (str): The title of the table.
        headers (List[str]): A list of header names for the table columns.
        rows (List[List[Any]]): Each inner list represents a row of data to display.
        start_columns (List[str]): A list of column names to be displayed at the beginning of the table.
        row_labels (List[List[Any]]): Each inner list represents the data labels for the corresponding row.
    """
    console = Console()
    console.print("\n" * 2)

    # Create a rich Table
    table = Table(title=title)

    # Add columns for the table
    for key in start_columns:
        table.add_column(key, justify="center", style="bold navy_blue")

    for i, key in enumerate(headers):
        justify = "right" if i % 2 == 0 else "left"
        table.add_column(key, justify=justify, style="deep_sky_blue4")

    def _format_str(s: Union[str, float, int]) -> str:
        if isinstance(s, float):
            s = f"{s:.2f}"
        return s

    # Add rows for each non-default field
    for label, row in zip(row_labels, rows):
        row_content = [_format_str(l) for l in label] + [_format_str(r) for r in row]
        table.add_row(*row_content)

    console.print(table)
    console.print("\n" * 2)
    console.print("\n" * 2)
    console.print("\n" * 2)
    console.print("\n" * 2)
