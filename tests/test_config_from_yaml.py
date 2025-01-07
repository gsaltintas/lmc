import argparse
import os
import unittest
from pathlib import Path
from shutil import rmtree
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

    def setUp(self):
        test_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        self.log_dir = test_dir / "tmp"
        self.log_dir.mkdir(exist_ok=True)

    # def tearDown(self):
    #     rmtree(self.log_dir)

    def test_build(self):
        conf = PerturbedTrainer.load_from_file(config_yaml)
        self.assertIsInstance(conf, PerturbedTrainer)
        print(conf.model)
        conf.save(self.log_dir, zip_code_base=False)
        new_yaml = self.log_dir / "config.yaml"
        with open(config_yaml, 'r') as f1:
            with open(new_yaml , 'r') as f2:
                for l1, l2 in zip(f1, f2):
                    self.assertEqual(l1, l2)
                    print(l1, l2)
        conf2 = PerturbedTrainer.load_from_file(new_yaml)



if __name__ == "__main__":
    unittest.main()
