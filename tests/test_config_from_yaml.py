import argparse
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn

from lmc.config import DataConfig, LoggerConfig, ModelConfig, TrainerConfig
from lmc.experiment_config import PerturbedTrainer, Trainer
from lmc.permutations import (PermSpec, PermType, get_permutation_sizes,
                              get_random_permutation_with_fixed_points,
                              permute_param)

config_yaml = Path(__file__).parent.joinpath("perturb.yaml").resolve().absolute()
class TestConfig(unittest.TestCase):
    def test_build(self):
        conf = PerturbedTrainer.load_from_file(config_yaml)
        assert type(conf) is PerturbedTrainer
        print(conf.model)




if __name__ == "__main__":
    unittest.main()
