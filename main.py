import argparse
import logging
import sys
from typing import Type

from lmc.config import maybe_get_arg
from lmc.experiment.base import ExperimentManager
from lmc.experiment.logreg import ActiveLearningLogisticRegressionSearch
from lmc.experiment.perturb import PerturbedTrainingRunner
from lmc.experiment.train import TrainingRunner

logger = logging.getLogger("")


managers = dict(
    train=TrainingRunner,
    perturb=PerturbedTrainingRunner,
    logreg=ActiveLearningLogisticRegressionSearch,
)


def get_experiment(manager_name: str) -> Type[ExperimentManager]:
    if manager_name not in managers:
        raise ValueError("No such runner: {}".format(manager_name))
    else:
        return managers[manager_name]


if __name__ == "__main__":
    helptext = "Choose a command to run:"
    for name, manager in managers.items():
        helptext += "\n    * {} {} [...] => {}".format(
            sys.argv[0], name, manager.description
        )
    helptext += "\n" + "=" * 82

    manager_name = maybe_get_arg("subcommand", positional=True, position=0)
    config_file = maybe_get_arg("config_file")
    if manager_name not in managers:
        print(helptext)
        sys.exit(1)

    # Add the arguments for that command.
    experiment_class = get_experiment(manager_name)

    if config_file is None:
        usage = "main.py {} [...] => {}".format(
            manager_name, experiment_class.description
        )
        usage += "\n" + "=" * 82 + "\n"
        parser = argparse.ArgumentParser(usage=usage, conflict_handler="resolve")
        parser.add_argument("subcommand")
        # Add arguments for the various managers.
        experiment_class.add_args(parser)

        args = parser.parse_args()
        experiment_manager = experiment_class.create_from_args(args)
    else:
        if len(sys.argv) > 4:
            ## check if there are additional arguments passed to
            usage = "main.py {} [...] => {}".format(
                manager_name, experiment_class.description
            )
            usage += "\n" + "=" * 82 + "\n"
            parser = argparse.ArgumentParser(usage=usage, conflict_handler="resolve")

            parser.add_argument("subcommand")
            parser.add_argument("--config_file")
            # Load defaults from config file if provided
            defaults = experiment_class.create_from_file(config_file)
            experiment_class.add_args(parser, defaults=defaults)
            args = parser.parse_args()
            experiment_manager = experiment_class.create_from_args(args)
        else:
            experiment_manager = experiment_class.create_from_file(config_file)

    experiment_manager.run_experiment()
