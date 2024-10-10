import argparse
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn

from lmc.config import DataConfig, LoggerConfig, ModelConfig, TrainerConfig
from lmc.experiment_config import Trainer
from lmc.permutations import (PermSpec, PermType, get_permutation_sizes,
                              get_random_permutation_with_fixed_points,
                              permute_param)


class TestConfig(unittest.TestCase):

    def setUp(self):
        self.trainer_config = TrainerConfig(training_steps="10ep")
        self.model_config = ModelConfig(model_name="resnet20")
        self.data_config = DataConfig(dataset="cifar10")
        self.logger_config = LoggerConfig()

        self.trainer = Trainer(
            trainer=self.trainer_config,
            model=self.model_config,
            data=self.data_config,
            logger=self.logger_config,
        )

    def test_instantiation(self):
        # Test that the Trainer class can be instantiated and contains the correct default fields.
        self.assertIsInstance(self.trainer, Trainer)
        self.assertEqual(self.trainer.trainer, self.trainer_config)
        self.assertEqual(self.trainer.model, self.model_config)
        self.assertEqual(self.trainer.data, self.data_config)
        self.assertEqual(self.trainer.logger, self.logger_config)

    def test_hashname(self):
        return
        # Test that the hashname is generated correctly.
        hashname = self.trainer.hashname
        self.assertTrue(hashname.startswith(self.trainer.name_prefix()))

    def test_add_args(self):
        # Test the add_args method adds arguments to the parser correctly.
        parser = argparse.ArgumentParser()
        Trainer.add_args(parser)
        args = parser.parse_args(
            [
                "--training_steps=10ep",
                "--lr=0.1",
                "--optimizer=sgd",
                "--model_name=resnet",
                "--dataset=mnist",
            ]
        )

        # Ensure all subconfigs have been added.
        self.assertIn("training_steps", args)
        self.assertIn("lr", args)
        self.assertIn("norm", args)
        self.assertIn("batch_size", args)
        self.assertIn("use_wandb", args)

    def test_create_from_args(self):
        # return
        # Mocking the argparse Namespace
        parser = argparse.ArgumentParser()
        Trainer.add_args(parser)
        args = parser.parse_args(
            [
                "--training_steps=10ep",
                "--lr=0.1",
                "--optimizer=sgd",
                "--model_name=resnet",
                "--dataset=mnist",
            ]
        )

        experiment = Trainer.create_from_args(args)

        self.assertIsInstance(experiment, Trainer)
        self.assertEqual(experiment.trainer.training_steps, args.training_steps)
        self.assertEqual(experiment.trainer.lr, args.lr)
        self.assertEqual(experiment.trainer.optimizer, args.optimizer)

    def test_display(self):
        # Test the display property returns the correct string.
        display_str = self.trainer.display
        for subconfig_ in self.trainer._subconfigs:
            subconfig = getattr(self.trainer, subconfig_)
            self.assertIn(subconfig.display, display_str)


if __name__ == "__main__":
    unittest.main()
