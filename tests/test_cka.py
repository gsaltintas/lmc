import unittest

from lmc.experiment.perturb import PerturbedTrainingRunner
from tests.base import BaseTest


class TestCKA(BaseTest):

    def test_cka(
        self,
    ):
        config = self.get_test_config(experiment="perturb", model_name="resnet20-32", dataset="cifar10", n_models=2, cka_n_test=100, test_batch_size=100, training_steps="1st", perturb_step="2st", perturb_mode="batch", cka_include=".act2.out", eval_freq=None)
        exp_manager = PerturbedTrainingRunner(config)
        exp_manager.run_experiment()


if __name__ == "__main__":
    unittest.main()
