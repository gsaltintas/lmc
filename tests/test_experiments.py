import torchvision

torchvision.disable_beta_transforms_warning()
import glob
import os
import subprocess
import unittest
from argparse import ArgumentParser
from pathlib import Path
from shutil import rmtree

import torch

from lmc.experiment_config import PerturbedTrainer
from lmc.experiment_manager import ExperimentManager, PerturbedTrainingRunner


class TestTrainingRunner(unittest.TestCase):
    PerturbedTrainingKwargs = dict(
        training_steps="2ep",
        norm="layernorm",
        hflip=True,
        random_rotation=10,
        random_crop=False,
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
        download=True
    )

    PerturbedTrainingRunnerCommand = """python main.py perturb  \
            --training_steps 2ep  \
            --model_name {model}  \
            --norm layernorm  \
            --path {data_dir}/{dataset}  \
            --dataset {dataset}  \
            --hflip true  \
            --random_rotation 10  \
            --random_crop false  \
            --lr_scheduler triangle  \
            --lr 0.1  \
            --warmup_ratio 0.02  \
            --optimizer sgd  \
            --momentum 0.9  \
            --save_early_iters true  \
            --cleanup_after false  \
            --use_wandb false  \
            --run_name test_butterfly_deterministic  \
            --project tests  \
            --n_models 2  \
            --seed1 {seed1}  \
            --seed2 {seed2}  \
            --loader_seed1 {seed1}  \
            --loader_seed2 {seed2}  \
            --perturb_seed1 {seed1}  \
            --perturb_seed2 {seed2}  \
            --perturb_step {perturb_step}  \
            --perturb_inds {perturb_inds}  \
            --perturb_mode gaussian  \
            --perturb_scale {perturb_scale}  \
            --deterministic {deterministic}  \
            --n_points 3  \
            --lmc_check_perms false  \
            --rewind_lr {rewind_lr}  \
            --log_dir {log_dir}  \
    """

    def setUp(self):
        test_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        self.log_dir = test_dir / "tmp"
        self.log_dir.mkdir(exist_ok=True)
        DATA_PATH_ENV_VAR = "DATASET_DIR"
        self.data_dir = os.environ.get(DATA_PATH_ENV_VAR)
        if self.data_dir is None:
            raise ValueError(
                f"Need to set the environment variable {DATA_PATH_ENV_VAR}"
            )

    def tearDown(self):
        rmtree(self.log_dir)

    def test_butterfly_deterministic(self):
        # should fail
        with self.subTest("bad args"):
            self.assertEqual("error", self.run_butterfly_deterministic(perturb_step=None))

        # should be identical
        with self.subTest("deterministic"):
            self.assertEqual("same", self.run_butterfly_deterministic())

        # should not be identical due to extra training time
        with self.subTest("extra training"):
            self.assertEqual("different", self.run_butterfly_deterministic(perturb_step=1))

        # should not be identical
        with self.subTest("perturbed"):
            self.assertEqual("different", self.run_butterfly_deterministic(perturb_scale=0.00000001))

        # should not be identical
        with self.subTest("different batch order"):
            self.assertEqual("different", self.run_butterfly_deterministic(seed2=999))

        # should not be identical (needs larger model for GPU non-determinism to cause differences)
        with self.subTest("non-deterministic"):
            self.assertEqual("different", self.run_butterfly_deterministic(deterministic=False, model="resnet20-8", dataset="cifar10"))

        # should be identical
        with self.subTest("perturb both same noise"):
            self.assertEqual("same", self.run_butterfly_deterministic(perturb_scale=1, perturb_inds=[1, 2]))

        # should not be identical
        with self.subTest("different lr"):
            self.assertEqual("same", self.run_butterfly_deterministic(rewind_lr="true"))

    def run_butterfly_deterministic(
        self, seed1=42, seed2=None, perturb_step=0, perturb_scale=0, deterministic=True, model="mlp/128x3", dataset="mnist", perturb_inds=[1], rewind_lr="false"
    ):
        if seed2 is None:
            seed2 = seed1
        command = str.format(
            self.PerturbedTrainingRunnerCommand,
            seed1=seed1,
            seed2=seed2,
            perturb_step=perturb_step,
            perturb_scale=perturb_scale,
            deterministic=deterministic,
            log_dir=self.log_dir,
            data_dir=self.data_dir,
            model=model,
            dataset=dataset,
            perturb_inds=" ".join(str(x) for x in perturb_inds),
            rewind_lr=rewind_lr,
        )
        # split command by spaces, remove excess spaces and line continuation symbols ("\"), replace commas with spaces to allow lists
        one_line = [x for x in "".join(command.split("\\")).split(" ") if len(x) > 0]
        try:
            results = subprocess.run(
                one_line, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            print("Error during run: ", e, e.stderr, sep="\n")
            return "error"
        # check that output doesn't contain traceback
        idx = results.stderr.find("Traceback (")
        if idx >= 0:
            print(results.stderr[idx:])
            return "error"

        return self.compare_last_ckpts(seed1, seed2)

    def compare_last_ckpts(self, seed1, seed2):
        last_run = self.get_last_created_in_dir(self.log_dir / "*")
        ckpt_1 = self.get_last_created_in_dir(
            last_run / f"model1-seed_{seed1}-ls_{seed1}" / "checkpoints" / "ep-*.ckpt"
        )
        ckpt_2 = self.get_last_created_in_dir(
            last_run / f"model2-seed_{seed2}-ls_{seed2}" / "checkpoints" / "ep-*.ckpt"
        )
        # check there is more than 1 saved ckpt per model
        self.assertGreater(len(self.glob(ckpt_1.parent / "*.ckpt")), 1)
        self.assertGreater(len(self.glob(ckpt_2.parent / "*.ckpt")), 1)
        # check if last ckpts are exactly identical
        return self.ckpts_match(ckpt_1, ckpt_2)

    def glob(self, path):
        return glob.glob(str(path))

    def get_last_created_in_dir(self, path):
        children = self.glob(path)
        if len(children):
            return Path(max(children, key=os.path.getctime))
        return None

    def ckpts_match(self, ckpt_1, ckpt_2):
        sd_1 = torch.load(ckpt_1)["state_dict"]
        sd_2 = torch.load(ckpt_2)["state_dict"]
        self.assertSetEqual(set(sd_1.keys()), set(sd_2.keys()))
        for k, v in sd_1.items():
            if not torch.equal(v, sd_2[k]):
                return "different"
        return "same"

    def test_post_perturb_steps(
        self,
        seed1=42,
        seed2=None,
        perturb_step: int = 1,
        perturb_scale=0,
        deterministic=True,
        model="mlp/128x3",
        dataset="mnist",
    ):
        if seed2 is None:
            seed2 = seed1
        config = PerturbedTrainer.from_dict(
            dict(
                **self.PerturbedTrainingKwargs,
                seed1=seed1,
                seed2=seed2,
                perturb_step=perturb_step,
                perturb_scale=perturb_scale,
                deterministic=deterministic,
                log_dir=self.log_dir,
                data_dir=self.data_dir,
                model_name=model,
                dataset=dataset,
            )
        )
        exp_manager = PerturbedTrainingRunner(config)
        exp_manager.setup()
        exp_manager.run()
        steps_per_epoch = exp_manager.steps_per_epoch
        steps = config.trainer.training_steps.get_step(steps_per_epoch) + config.perturb_step.get_step(steps_per_epoch)
        self.assertEqual(
            exp_manager.global_step, steps, f"Expected {steps} steps, got {exp_manager.global_step}"
        )


if __name__ == "__main__":
    unittest.main()
