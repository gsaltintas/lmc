import unittest
import numpy as np

from sklearn.linear_model import LogisticRegression

from lmc.experiment.logreg import max_entropy_x
from lmc.utils.run import command_result_is_error, run_command
from tests.base import BaseTest


class TestTrainingRunner(BaseTest):
    def setUp(self):
        super().setUp()
        self.model = LogisticRegression(
            penalty=None, warm_start=True, solver="newton-cholesky"
        )

    def fit_logreg(self, labels, **kwargs):
        x, slope, intercept = max_entropy_x(
            [1, 2, 3, 3, 4, 5], labels, **kwargs, regression_model=self.model
        )
        identical_labels = all(i == labels[0] for i in labels)
        self.assertEqual(np.isnan(slope), identical_labels)
        self.assertEqual(np.isnan(intercept), identical_labels)
        return x

    def test_maxent_logistic_regression(self):
        x = self.fit_logreg([0, 0, 0, 1, 1, 1])
        self.assertAlmostEqual(x, 3, places=0)

        x = self.fit_logreg([0, 0, 0, 0, 0, 0])
        self.assertEqual(x, np.inf)

        x = self.fit_logreg([1, 1, 1, 1, 1, 1])
        self.assertEqual(x, 0)

        x = self.fit_logreg([0, 0, 0, 0, 0, 0], step_ratio=2)
        self.assertEqual(x, 10)

        x = self.fit_logreg([1, 1, 1, 1, 1, 1], step_ratio=10)
        self.assertEqual(x, 0.1)

    LOGREG_ARGS = """
            --perturb_debug_dummy_run {perturb_debug_dummy_run}  \
            --logreg_y {logreg_y}  \
            --logreg_threshold 0.5  \
            --logreg_n {logreg_n}  \
            --logreg_max_step_ratio 10  \
            --logreg_reseed_every_run true  \
    """

    def get_test_command(self, is_dummy_run, logreg_y, logreg_n, model_dir=None):
        perturb_debug_dummy_run = "true" if is_dummy_run else "false"
        args = str.format(
            self.LOGREG_ARGS,
            perturb_debug_dummy_run=perturb_debug_dummy_run,
            logreg_y=logreg_y,
            logreg_n=logreg_n,
        )
        return super().get_test_command(
            experiment="logreg",
            perturb_scale=0.1,
            perturb_step=390,
            use_wandb="true",
            lmc_on_train_end="true",
            model_dir=model_dir,
            args=args,
        )

    def test_dummy_logreg(self):
        # logic test: check that logistic regression over dummy values works
        def run_dummy_logreg(model_dir):
            command = self.get_test_command(
                is_dummy_run=True,
                logreg_y="test/dummyvalue",
                logreg_n=10,
                model_dir=self.log_dir / model_dir,
            )
            result = run_command(command, print_output=True)
            self.assertFalse(command_result_is_error(result))
            return self.get_summary_value("logreg/max_entropy_x")

        run_1 = run_dummy_logreg("test-logreg-1")
        # dummy runs draw logreg_y uniformly from [0, perturb_step)
        # so a threshold of 0.5 should result in predicted perturb_step near 1
        self.assertAlmostEqual(run_1, 1, 0)
        # repeat and check that the result is the same due to same seeds
        self.assertEqual(run_dummy_logreg("test-logreg-2"), run_1)

    def test_logreg(self):
        # integration test: check that running perturb works
        command = self.get_test_command(
            is_dummy_run=False,
            logreg_y="lmc-0-1/lmc/loss/weighted/increase_end0_train",
            logreg_n=1,
        )
        result = run_command(command, print_output=True)
        self.assertFalse(command_result_is_error(result))


if __name__ == "__main__":
    unittest.main()
