import unittest
from glob import glob

from lmc.experiment.perturb import PerturbedTrainingRunner
from lmc.experiment_config import PerturbedTrainer
from lmc.utils.run import command_result_is_error, run_command
from tests.base import BaseTest


class TestFFCV(BaseTest):

    def test_ffcv(self):
        self.data_dir = self.data_dir / "ffcv"
        config = self.get_test_config(
            n_models=2,
            perturb_step=5,
            perturb_scale=0.1,
            perturb_mode="batch",
            perturb_seed2=99,
            perturb_inds=[2],
            lmc_on_epoch_end=True,
            n_points=3,
            use_ffcv=True,
            model_name="resnet8-4",
            dataset="cifar10",
            hflip=True,
            random_rotation=10,
            random_crop="false",
            random_translate=4,
            cutout=4,
            )
        exp_manager = PerturbedTrainingRunner(config)
        exp_manager.run_experiment()
        steps_per_epoch = exp_manager.steps_per_epoch
        steps = config.trainer.training_steps.get_step(
            steps_per_epoch
        ) + config.perturb_step.get_step(steps_per_epoch)
        self.assertEqual(
            exp_manager.global_step,
            steps,
            f"Expected {steps} steps, got {exp_manager.global_step}",
        )
        perturbed_steps = exp_manager.training_elements[1].curr_step
        unperturbed_steps = exp_manager.training_elements[0].curr_step
        self.assertEqual(
            perturbed_steps,
            steps,
            f"Expected {steps} steps for the perturbed model, got {perturbed_steps}",
        )
        orig_steps = config.trainer.training_steps.get_step(steps_per_epoch)
        self.assertEqual(
            unperturbed_steps,
            orig_steps,
            f"Expected {orig_steps} steps for the non perturbed model, got {unperturbed_steps}",
        )


if __name__ == "__main__":
    unittest.main()
