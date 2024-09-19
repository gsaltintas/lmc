
import argparse
import logging
import sys
import traceback

from lmc.config import maybe_get_arg
from lmc.experiment_manager import managers
from lmc.utils.setup_training import cleanup

logger = logging.getLogger("")

if __name__ == '__main__':
    helptext = "Choose a command to run:"
    for name, manager in managers.items():
        helptext += "\n    * {} {} [...] => {}".format(sys.argv[0], name, manager.description)
    helptext += '\n' + '='*82

    manager_name = maybe_get_arg('subcommand', positional=True, position=0)
    if manager_name not in managers:
        print(helptext)
        sys.exit(1)

    # Add the arguments for that command.
    usage = 'main.py {} [...] => {}'.format(manager_name, managers.get(manager_name).description)
    usage += '\n' + '='*82 + '\n'

    parser = argparse.ArgumentParser(usage=usage, conflict_handler='resolve')
    parser.add_argument('subcommand')

    # Add arguments for the various managers.
    managers.get(manager_name).add_args(parser)

    args = parser.parse_args()
    manager = managers.get(manager_name).create_from_args(args)
    try:
        manager.run()
    except Exception as e:
        traceback.print_exc()
        pass
    cleanup(manager.config)