"""
step.py
This module provides the Step class and a utility function for handling step values
in a training process. The Step class allows for the representation of steps in terms
of epochs or specific step counts, including methods for conversion and validation.
Functions:
format_step(value): Converts a numeric value to a formatted string representing steps.
Classes:
Step: A dataclass that encapsulates a step value, its associated properties, and methods
for manipulating and retrieving step-related information.
"""
import math
import re
from dataclasses import dataclass
from typing import Literal, Optional, Union


def format_step(value):
    value = str(value)
    if value.isnumeric():
        value = f"{value}st"
    value = value.lower()
    return value

@dataclass
class Step:
    value: Union[str, int, None] = None #field(init=True, default_factory=format_step)
    steps_per_epoch: Optional[int] = 1
    _name: str = None

    def __post_init__(self):
        if str(self.value).isnumeric():
            self.value = f"{self.value}st"
        self.value = str(self.value).lower()
        if not self.value.endswith("ep") and not self.value.endswith(
            "st"
        ):
            raise ValueError(f"Please specify {self._name} steps as either X | Xst or Xep.")

    @property
    def suffix(self):
        if str(self.value).isnumeric():
            return "st"
        return self.value[-2:]
    
    def add(self, value) -> None:
        #TODO: implement this
        if str(value).isnumeric():
            value = f"{value}st"
        # total_steps = self.get_step()
    
    def get_step(self, steps_per_epoch: Optional[int] = None) -> int:
        val = int(self.value[:-2])
        if self.value.endswith("st"):
            return val
        steps_per_epoch = steps_per_epoch if steps_per_epoch is not None else self.steps_per_epoch
        return val * steps_per_epoch

    def get_epoch(self, steps_per_epoch: Optional[int] = None) -> int:
        val = int(self.value[:-2])
        if self.value.endswith("ep"):
            return val
        steps_per_epoch = steps_per_epoch if steps_per_epoch is not None else self.steps_per_epoch
        return math.ceil(val / steps_per_epoch)
    
    def modulo(self, val: int, mode: Literal["st", "ep"], steps_per_epoch: Optional[int] = None) -> int:
        if mode == "st":
            return val % self.get_step(steps_per_epoch)
        return val % self.get_epoch(steps_per_epoch)
    
    
    @classmethod
    def from_string(cls, step_str: str):
        """Parse a string representation of a Step and return a Step instance."""
        # Use regex to extract the value and steps_per_epoch
        match = re.match(r"Step\(value='([^']+)', steps_per_epoch=(\d+)\)", step_str)
        if not match:
            raise ValueError(f"Invalid step string format: {step_str}")
        
        value = match.group(1)
        steps_per_epoch = int(match.group(2))
        
        return cls(value=value, steps_per_epoch=steps_per_epoch)
    
    def wandb_dct(self):
        # TODO: not sure about this design choice
        # logging string format preserves all info but hard for plots
        return {"value": self.value, "steps_per_epoch": self.steps_per_epoch, "step": self.get_step(), "epoch": self.get_epoch()}