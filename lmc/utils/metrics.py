
from dataclasses import dataclass
from typing import Any, List, Union

import torch
from rich.console import Console
from rich.table import Table
from tabulate import tabulate


def mixup_topk_accuracy(
    preds, targets, targets_shuffled=None, k: int = 3, avg: bool = False
):
    num = preds.size(0) if avg else 1
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
        return torch.maximum(acc, acc_shuffled), torch.maximum(topk, topk_shuffled)
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
        if val is None: return
        self.val = val
        self.sum += val * n
        self.count += n

    def get_avg(self, percentage=False):
        return self.sum / self.count if not percentage else self.sum * 100 / self.count



@dataclass
class Metrics:
    total_acc = AverageMeter()
    total_topk = AverageMeter()
    total_loss = AverageMeter()
    cross_entropy = AverageMeter()

    def reset(self):
        self.total_acc.reset()
        self.total_topk.reset()
        self.total_loss.reset()
        self.cross_entropy.reset()

    def update(self, total_acc: float = None, total_topk: float = None, total_loss: float = None, cross_entropy: float = None, n: int = 1):
        self.total_acc.update(total_acc, n)
        self.total_topk.update(total_topk, n)
        self.total_loss.update(total_loss, n)
        self.cross_entropy.update(cross_entropy, n)


def report_results(log_dct, epoch: int, n_models: int = 1):
    """ given the log_dct, generates and prints a nice looking summary of the training results for the given epoch """
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
            ce_keys_weighted = ["lmc/loss/weighted/barrier_train", "lmc/loss/weighted/barrier_test"]
            accuracy_keys = [
                "lmc/barrier_train",
                "lmc/barrier_test",
                "lmc/max_err_alpha_train",
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
            ce_values_weighted = [log_dct.get(f"{key_prefix}{key}", None) for key in ce_keys_weighted]
            accuracy_values = [log_dct.get(f"{key_prefix}{key}", None) for key in accuracy_keys]
            accuracy_values_weighted = [log_dct.get(f"{key_prefix}{key}", None) for key in accuracy_keys_weighted]

            # Combining rows with labels
            rows = [accuracy_values, ce_values, accuracy_values_weighted, ce_values_weighted]
            start_columns = ["", ""]
            row_labels = [
                [f"{prefix}Accuracy", ""],
                [f"{prefix}Cross-Entropy", ""],
                [f"{prefix}Accuracy w. alpha scaling", ""],
                [f"{prefix}CE with alpha scaling", ""],
            ]
            title = f"LMC - Epoch {epoch}"
            print_to_rich(title, headers, rows, start_columns, row_labels)
            print(tabulate(rows, headers=headers, tablefmt="grid", showindex=row_labels))

    # Specified keys
    modes = ["test", "train"]
    acc_keys = [f"model{i}/{mod}/accuracy" for mod in modes for i in range(1, n_models+1)]
    ce_keys = [f"model{i}/{mod}/cross_entropy" for mod in modes for i in range(1, n_models+1)]

    # Custom headers
    headers = [f"{mod}/{i}" for mod in modes for i in range(1, n_models+1)]

    start_columns = ["", ""]
    row_labels = [["Accuracy", ""], ["Cross-Entropy", ""]]
    # Extracting values
    acc_values = [log_dct.get(key, None) for key in acc_keys]
    ce_values = [log_dct.get(key, None) for key in ce_keys]
    rows = [acc_values, ce_values]
    title = f"Epoch {epoch}"
    print_to_rich(title, headers, rows, start_columns, row_labels)

def print_to_rich(title: str, headers: List[str], rows: List[List[Any]], start_columns: List[str], row_labels: List[List[Any]]) -> None:
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
        justify = "right" if i%2 == 0 else "left"
        table.add_column(key, justify=justify, style="deep_sky_blue4")

    def _format_str(s: Union[str, float, int]) -> str:
        if isinstance(s, float):
            s = f"{s:.2f}"
        return s
            
    # Add rows for each non-default field
    for (label, row) in zip(row_labels, rows):
        row_content = [_format_str(l) for l in label] + [_format_str(r) for r in row]
        table.add_row(*row_content)

    console.print(table)
    console.print("\n" * 2)

