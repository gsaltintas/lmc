import abc
import argparse
import logging
from dataclasses import dataclass, field

from rich.logging import RichHandler
from torch import nn
from lmc.experiment_config import Experiment


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
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add all command line flags necessary for this manager."""
        cls.config.add_args(parser)

    # @staticmethod
    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> "ExperimentManager":
        """Create a manager from command line arguments."""
        config = cls.config.create_from_args(args)
        return cls(config)

    @abc.abstractmethod
    def run(self) -> None:
        """Run the job."""

        pass

    @abc.abstractmethod
    def setup(self) -> None:
        pass
