import torchvision

torchvision.disable_beta_transforms_warning()
import argparse
import unittest

from lmc.config import DataConfig, LoggerConfig, ModelConfig, TrainerConfig, _SGDConfig
from lmc.experiment_config import PerturbedTrainer, Trainer, LMCConfig, make_seeds_class


class TestConfig(unittest.TestCase):
    PerturbedTrainingKwargs = dict(
        training_steps="2ep",
        norm="layernorm",
        hflip=True,
        random_rotation=10,
        random_crop=False,
        lr_scheduler="triangle",
        lr=0.001,
        warmup_ratio=0.02,
        optimizer="sgd",
        momentum=0.8,
        save_early_iters=True,
        cleanup_after=False,
        use_wandb=False,
        run_name="test_butterfly_deterministic",
        project="tests",
        n_models=2,
        perturb_inds=[1],
        perturb_mode="gaussian",
        n_points=3,
        lmc_check_perms=False,
        batch_size=128,
        perturb_seed1=4,
        perturb_seed2=5,
        seed1=1,
        seed2=2
    )

    def setUp(self):
        self.opt_config = _SGDConfig()
        self.trainer_config = TrainerConfig(training_steps="10ep", opt=self.opt_config)
        self.model_config = ModelConfig(model_name="resnet20-16")
        self.data_config = DataConfig(dataset="cifar10")
        self.logger_config = LoggerConfig()
        self.lmc_config = LMCConfig()
        self.seeds_config = make_seeds_class(2)

        self.trainer = Trainer(
            trainer=self.trainer_config,
            model=self.model_config,
            data=self.data_config,
            logger=self.logger_config,
            lmc=self.lmc_config,
        )

    def test_instantiation(self):
        # Test that the Trainer class can be instantiated and contains the correct default fields.
        self.assertIsInstance(self.trainer, Trainer)
        self.assertEqual(self.trainer.trainer, self.trainer_config)
        self.assertEqual(self.trainer.model, self.model_config)
        self.assertEqual(self.trainer.data, self.data_config)
        self.assertEqual(self.trainer.logger, self.logger_config)
        self.assertEqual(self.trainer.lmc, self.lmc_config)
        self.assertEqual(self.trainer.trainer.opt, self.opt_config)

    def test_hashname(self):
        # Test that the hashname is generated correctly.
        hashname = self.trainer.hashname
        self.assertTrue(hashname.startswith(self.trainer._name_prefix))

    def test_add_args(self):
        # Test the add_args method adds arguments to the parser correctly.
        parser = argparse.ArgumentParser()
        Trainer.add_args(parser)
        args = parser.parse_args(
            [
                "--training_steps=10ep",
                "--lr=0.1",
                "--optimizer=sgd",
                "--model_name=resnet20-16",
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
                "--model_name=resnet20-16",
                "--dataset=mnist",
            ]
        )

        experiment = Trainer.create_from_args(args)

        self.assertIsInstance(experiment, Trainer)
        self.assertEqual(f"{experiment.trainer.training_steps.get_epoch()}ep", args.training_steps)
        self.assertEqual(experiment.trainer.opt.lr, args.lr)
        self.assertEqual(experiment.trainer.opt.optimizer, args.optimizer)

    def test_display(self):
        # Test the display property returns the correct string.
        display_str = self.trainer.display
        for subconfig_ in self.trainer._subconfigs:
            subconfig = getattr(self.trainer, subconfig_)
            self.assertIn(subconfig.display, display_str)

    def test_init(self):
        config = PerturbedTrainer(
                **self.PerturbedTrainingKwargs,
                perturb_step=1,
                deterministic=False,
                model_name="mlp/128x3",
                dataset="mnist",
        )
        self.common(config)
    
    def common(self, config):
        self.assertEqual(f"{config.trainer.training_steps.get_epoch()}ep", self.PerturbedTrainingKwargs["training_steps"])
        self.assertEqual(config.trainer.opt.lr, self.PerturbedTrainingKwargs["lr"])
        self.assertEqual(config.trainer.opt.momentum, self.PerturbedTrainingKwargs["momentum"])
        self.assertEqual(config.trainer.opt.optimizer, self.PerturbedTrainingKwargs["optimizer"])
        self.assertEqual(config.trainer.opt.lr_scheduler, self.PerturbedTrainingKwargs["lr_scheduler"])
        self.assertEqual(config.trainer.opt.optimizer, self.PerturbedTrainingKwargs["optimizer"])
        self.assertEqual(config.seeds.seed1, self.PerturbedTrainingKwargs["seed1"])
        self.assertEqual(config.seeds.seed2, self.PerturbedTrainingKwargs["seed2"])
        self.assertEqual(config.perturb_seeds.perturb_seed1, self.PerturbedTrainingKwargs["perturb_seed1"])
        self.assertEqual(config.perturb_seeds.perturb_seed2, self.PerturbedTrainingKwargs["perturb_seed2"])
        self.assertEqual(config.model.initialization_strategy, "kaiming_normal", "Default argument not set correctly")
        
    def test_from_dict(self):
        config = PerturbedTrainer.from_dict(
            dict(
                **self.PerturbedTrainingKwargs,
                perturb_step=1,
                deterministic=False,
                model_name="mlp/128x3",
                dataset="mnist",
            )
        )
        self.common(config)
        
if __name__ == "__main__":
    unittest.main()
