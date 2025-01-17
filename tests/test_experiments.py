import unittest
from glob import glob

from lmc.experiment_config import PerturbedTrainer
from lmc.experiment.perturb import PerturbedTrainingRunner
from lmc.utils.run import command_result_is_error, run_command
from tests.base import BaseTest



class TestTrainingRunner(BaseTest):
    def test_training_steps(self):
        # check if an arbitrary number of training steps can run
        for step in [9, 392]:
            with self.subTest(f"training steps: {step}"):
                model_dir = self.log_dir / f"test-training-steps-{step}"
                command = self.get_test_command(model_name="resnet20-8", dataset="cifar10", training_steps=f"{step}st", lmc_on_train_end="true", use_wandb="true", model_dir=model_dir)
                result = run_command(command, print_output=True)
                self.assertFalse(command_result_is_error(result))
                self.assertEqual(self.get_summary_value(model_dir, "lr/global_step"), step)

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

    PerturbedTrainingKwargs = dict(
        training_steps="2ep",
        norm="layernorm",
        hflip=True,
        random_rotation=10,
        lr_scheduler="triangle",
        lr=0.1,
        warmup_ratio=0.02,
        optimizer="sgd",
        momentum=0.9,
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
    )

    def test_post_perturb_steps(
        self,
        seed1=42,
        seed2=None,
        perturb_step: int = 1,
        perturb_scale=0.1,
        deterministic=True,
        model_name="mlp/128x3",
        dataset="mnist",
    ):
        if seed2 is None:
            seed2 = seed1
        perturb_ind = 1
        non_perturbed = 0
        config = PerturbedTrainer.from_dict(
            dict(
                **self.PerturbedTrainingKwargs,
                seed1=seed1,
                seed2=seed2,
                perturb_step=perturb_step,
                perturb_scale=perturb_scale,
                deterministic=deterministic,
                log_dir=self.log_dir,
                path=self.data_dir / dataset,
                model_name=model_name,
                dataset=dataset,
            )
            | {"perturb_inds": [perturb_ind + 1]}
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
