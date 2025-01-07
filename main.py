import argparse
import logging
import sys
import traceback

from lmc.config import maybe_get_arg
from lmc.utils.setup_training import cleanup
from lmc.experiment.base import ExperimentManager
from lmc.experiment.train import TrainingRunner
from lmc.experiment.perturb import PerturbedTrainingRunner


logger = logging.getLogger("")


managers = dict(
    train=TrainingRunner,
    perturb=PerturbedTrainingRunner,
)


def get_experiment(manager_name: str) -> ExperimentManager:
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
    if manager_name not in managers:
        print(helptext)
        sys.exit(1)

    # Add the arguments for that command.
    experiment = get_experiment(manager_name)
    usage = "main.py {} [...] => {}".format(manager_name, experiment.description)
    usage += "\n" + "=" * 82 + "\n"

    parser = argparse.ArgumentParser(usage=usage, conflict_handler="resolve")
    parser.add_argument("subcommand")

    # Add arguments for the various managers.
    experiment.add_args(parser)

    args = parser.parse_args()
    experiment_manager = experiment.create_from_args(args)
    try:
        experiment_manager.run()
    except Exception:
        traceback.print_exc()
        pass
    cleanup(experiment_manager.config)
