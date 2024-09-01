import argparse
import logging

from ml_collections import config_flags
from torch import nn, optim

from lmc.config import Config
from lmc.experiment_config import Trainer
from lmc.utils.setup_training import TrainingElement, cleanup, setup_experiment

logger = logging.getLogger("trainer")

def train(config):
    # import code; code.interact(local=locals()|globals())
    training_elements = setup_experiment(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument("subcommand")
    # import code; code.interact(local=locals()|globals())
    # parser.add_argument(
    #     "--platform", default="local", help="The platform on which to run the job."
    # )
    # Add arguments for the various runners.
    Trainer.add_args(parser)

    args = parser.parse_args()
    platform = Trainer.create_from_args(args)
    # platform.display()
    print(platform.display)
    train(platform)
    platform.save("hello")
    p = Trainer.load_from_file("config.yaml")
    cleanup(platform)
    # print(p.display)
    # print(platform.hflip)