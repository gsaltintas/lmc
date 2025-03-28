import unittest

from lmc.experiment.perturb import PerturbedTrainingRunner
from tests.base import BaseTest


class TestCKA(BaseTest):

    def tearDown(self):
        pass

    def test_cka(
        self,
    ):
        config = self.get_test_config(experiment="perturb", n_models=2, cka_eval=True, eval_steps=["0"], perturb_seed1=99)
        exp_manager = PerturbedTrainingRunner(config)
        exp_manager.run_experiment()


if __name__ == "__main__":
    unittest.main()
