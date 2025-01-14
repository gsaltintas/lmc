import shlex
import subprocess
import traceback
from typing import List, Union


def run_command(command: Union[str, List[str]], print_output=False):
    # split command by spaces, remove excess spaces and line continuation symbols ("\"), replace commas with spaces to allow lists
    if isinstance(command, str):
        command = shlex.split(command)
        # command = [x for x in "".join(command.split("\\")).split(" ") if len(x) > 0]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        if print_output:
            print("Error during run: ", e, sep="\n")
            traceback.print_exc()
            print(e.stdout)
            print(e.stderr)
        return e
    # check that output doesn't contain traceback
    idx = result.stderr.find("Traceback (")
    if idx >= 0:
        print(result.stderr[idx:])
    if print_output:
        print(result.stdout)
    return result


def command_result_is_error(result):
    if isinstance(result, subprocess.CalledProcessError):
        # print("subprocess ERROR: CalledProcessError", result)
        return True
    idx = result.stderr.find("Traceback (")
    if idx >= 0:
        # print("subprocess ERROR: Traceback", result)
        return True
    # print("subprocess finished successfully", result)
    return False
