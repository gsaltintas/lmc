import unittest

from lmc.experiment.perturb import PerturbedTrainingRunner
from tests.base import BaseTest

PERTURB_SCALE = 0.1


class TestConfig(BaseTest):
    def run_and_get_batch_hashes(self, model_dir, n_models=2, **kwargs):
        config = self.get_test_config(
            seed1=99,
            seed2=99,
            training_steps="4st",
            perturb_step="2st",
            warmup_ratio=0.5,
            same_steps_pperturb=False,
            lr_scheduler="triangle",
            use_wandb=True,
            model_dir=self.log_dir / model_dir,
            n_models=n_models,
            perturb_inds=[1],
            **kwargs,
        )
        exp = PerturbedTrainingRunner(config)
        exp.run_experiment()
        hashes = {}
        for i in range(1, n_models + 1):
            for value in ["x", "y"]:
                key = f"model{i}/{value}"
                hashes[key] = self.get_summary_value(
                    self.log_dir / model_dir, "data_hash/" + key
                )
        return hashes

    def test_batches_deterministic(self):
        for perturb_type in ["gaussian", "batch"]:
            with self.subTest(perturb_type):
                hashes = self.run_and_get_batch_hashes(
                    f"test-{perturb_type}",
                    2,
                    perturb_type=perturb_type,
                    perturb_seed1=11,
                    perturb_scale=0.2,
                )
                self.assertEqual(hashes["model1/x"], hashes["model2/x"])
                self.assertEqual(hashes["model1/y"], hashes["model2/y"])
                hashes_2 = self.run_and_get_batch_hashes(
                    f"test-{perturb_type}-2",
                    2,
                    perturb_type=perturb_type,
                    perturb_seed1=22,
                    perturb_scale=0.4,
                    evaluate_ckpt2=self.log_dir / f"test-{perturb_type}" / "model2",
                )
                self.assertEqual(hashes["model1/x"], hashes_2["model1/x"])
                self.assertEqual(hashes["model1/y"], hashes_2["model1/y"])


if __name__ == "__main__":
    unittest.main()
