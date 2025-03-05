import unittest
from glob import glob

from lmc.experiment.perturb import PerturbedTrainingRunner
from lmc.experiment_config import PerturbedTrainer
from lmc.utils.run import command_result_is_error, run_command
from tests.base import BaseTest


class TestTrainingRunner(BaseTest):

    def test_batch_perturb_seed(self):
        # check batch perturb seed doesn't mess up other dataloaders
        # do this by setting perturb_scale = 0 and comparing gaussian vs batch, they should be the same
        batch = self.run_command_and_return_result(
            f"test-perturb-seed-batch",
            "model1/train/cross_entropy",
            perturb_mode="batch",
            perturb_scale=0,
            perturb_seed1=99,
            perturb_seed2=98,
        )
        gaussian = self.run_command_and_return_result(
            f"test-perturb-seed-gaussian",
            "model1/train/cross_entropy",
            perturb_mode="gaussian",
            perturb_scale=0,
            perturb_seed1=99,
            perturb_seed2=98,
        )
        self.assertEqual(batch, gaussian)

    def test_butterfly_deterministic(self):
        # check whether model pairs are same or not

        # should fail
        with self.subTest("bad args"):
            self.assertEqual(
                "error", self.run_butterfly_deterministic(perturb_step=None)
            )

        # should not be identical due to extra training time
        with self.subTest("extra training"):
            self.assertEqual(
                "different",
                self.run_butterfly_deterministic(
                    perturb_step=1, same_steps_pperturb="true"
                ),
            )

        # should not be identical
        with self.subTest("perturbed"):
            self.assertEqual(
                "different", self.run_butterfly_deterministic(perturb_scale=0.00000001)
            )

        # should not be identical
        with self.subTest("different batch order"):
            self.assertEqual("different", self.run_butterfly_deterministic(seed2=999))

        # should not be identical (needs larger model for GPU non-determinism to cause differences)
        with self.subTest("non-deterministic"):
            self.assertEqual(
                "different",
                self.run_butterfly_deterministic(
                    deterministic=False,
                    model_name="resnet20-8",
                    dataset="cifar10",
                    lmc_on_train_end="true",
                ),
            )

        # should be identical
        with self.subTest("perturb both same noise"):
            self.assertEqual(
                "same",
                self.run_butterfly_deterministic(perturb_scale=1, perturb_inds=[1, 2]),
            )

        # should not be identical
        with self.subTest("different lr"):
            self.assertEqual("same", self.run_butterfly_deterministic(rewind_lr="true"))

    def run_butterfly_deterministic(self, seed1=42, seed2=None, **kwargs):
        seed2 = seed1 if seed2 is None else seed2
        command = self.get_test_command(seed1=seed1, seed2=seed2, **kwargs)
        result = run_command(command)
        if command_result_is_error(result):
            return "error"

        # check if training models 1 and 2 gives the same result
        last_run = self.get_last_created_in_dir(self.log_dir / "*")
        ckpt_1, ckpt_2 = self.get_last_ckpts(last_run)
        # check there is more than 1 saved ckpt per model
        self.assertGreater(len(glob(str(ckpt_1.parent / "*.ckpt"))), 1)
        self.assertGreater(len(glob(str(ckpt_2.parent / "*.ckpt"))), 1)
        # check if last ckpts are exactly identical
        return "same" if self.ckpts_match(ckpt_1, ckpt_2) else "different"

    def test_post_perturb_steps(self):
        perturb_ind = 1
        non_perturbed = 0
        config = self.get_test_config(
            n_models=2,
            perturb_step=5,
            perturb_scale=0.1,
            perturb_mode="gaussian",
            perturb_seed2=99,
            perturb_inds=[perturb_ind + 1],
            lmc_on_epoch_end=True,
            n_points=3,
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
        perturbed_steps = exp_manager.training_elements[perturb_ind].curr_step
        unperturbed_steps = exp_manager.training_elements[non_perturbed].curr_step
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
