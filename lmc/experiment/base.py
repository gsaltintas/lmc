import abc
import argparse
import json
import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import torch
import torch.autograd.profiler as profiler
from rich.logging import RichHandler
from torch import nn

import wandb
from lmc.experiment_config import Experiment
from lmc.utils.setup_training import cleanup

FORMAT = "%(name)s - %(levelname)s: %(message)s"


@dataclass
class ExperimentManager(abc.ABC):
    config: Experiment = field(init=True, default=Experiment)
    _name: str = ""
    logger: logging.Logger = None

    def __post_init__(self):
        logging.basicConfig(
            level=self.config.logger.level.upper(),
            format=FORMAT,
            handlers=[RichHandler(show_time=False)],
        )
        self.logger = logging.getLogger(self._name)
        self.loss_fn, self.test_loss_fn = (
            nn.CrossEntropyLoss(label_smoothing=self.config.trainer.label_smoothing),
            nn.CrossEntropyLoss(),
        )

    """An instance of a training run of some kind."""

    @staticmethod
    @abc.abstractmethod
    def description() -> str:
        """A description of this manager."""

        pass

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, defaults=None) -> None:
        """Add all command line flags necessary for this manager."""
        cls.config.add_args(parser, defaults=defaults)

    # @staticmethod
    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> "ExperimentManager":
        """Create a manager from command line arguments."""
        config = cls.config.create_from_args(args)
        return cls(config)

    @classmethod
    def create_from_file(cls, file: Union[Path, str]) -> "ExperimentManager":
        """Create a manager from config file."""
        config = cls.config.load_from_file(file)
        return cls(config)

    @abc.abstractmethod
    def run(self) -> None:
        """Run the job."""
        pass

    @abc.abstractmethod
    def setup(self) -> None:
        pass

    def finish(self) -> None:
        # save wandb run summary values to file
        if self.config.logger.use_wandb:
            with open(self.config.model_dir / "wandb_summary.json", "w") as f:
                json.dump(dict(wandb.run.summary), f)

    def run_experiment(self):
        try:
            self.setup()
            if self.config.logger.profile:
                with profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
                    self.run()
                print(prof.key_averages().table(sort_by="cuda_time_total"))
            else:
                self.run()
            self.finish()
            return True
        except Exception:
            traceback.print_exc()
            return False
        finally:
            cleanup(self.config)
