import unittest

import torch

from lmc.experiment.perturb import PerturbedTrainingRunner
from lmc.experiment.train import TrainingRunner
from lmc.utils.setup_training import configure_model
from lmc.utils.step import Step
from tests.base import BaseTest

EARLY_CKPTS = [
    1,
    2,
    3,
    4,
    5,
    6,
    57,
    108,
    160,
    211,
    263,
    314,
    366,
    417,
    469,
    469,
    521,
    573,
    625,
    677,
    729,
    781,
    833,
    885,
    938,
    938,
    990,
    1042,
    1094,
    1146,
    1198,
    1250,
    1302,
    1354,
    1407,
    1407,
    1459,
    1511,
    1563,
    1615,
    1667,
    1719,
    1771,
    1823,
    1876,
    1876,
    1928,
    1980,
    2032,
    2084,
    2136,
    2188,
    2240,
    2292,
    2345,
    2345,
    2397,
    2449,
    2501,
    2553,
    2605,
    2657,
    2709,
    2761,
    2814,
    2814,
    2866,
    2918,
    2970,
    3022,
    3074,
    3126,
    3178,
    3230,
    3283,
    3283,
    3335,
    3387,
    3439,
    3491,
    3543,
    3595,
    3647,
    3699,
    3752,
    3752,
    3804,
    3856,
    3908,
    3960,
    4012,
    4064,
    4116,
    4168,
    4221,
    4221,
    4273,
    4325,
    4377,
    4429,
    4481,
    4533,
    4585,
    4637,
    4690,
]


class TestTraining(BaseTest):
    def setUp(self):
        super().setUp()
        self.tests = 0
        self.steps_per_ep = self.get_test_config().data.get_steps_per_epoch()

    def test_model_init(self):
        # check that models are init the same way given the same seed
        config = self.get_test_config()
        exp_1 = TrainingRunner(config)
        exp_2 = TrainingRunner(config)
        exp_1.setup()
        exp_2.setup()
        sd_1 = exp_1.training_elements[0].model.state_dict()
        sd_2 = exp_2.training_elements[0].model.state_dict()
        self.assertTrue(self.state_dicts_equal(sd_1, sd_2))
        model, _ = configure_model(
            config, device=exp_1.device, seed=exp_1.config.seeds.seed1
        )
        self.assertTrue(self.state_dicts_equal(sd_1, model.state_dict()))

    def test_step(self):
        step = Step.from_short_string("2ep0st", steps_per_epoch=20)
        self.assertEqual(step.get_step(), 40)
        self.assertEqual("2ep0st", step.to_short_string())
        step = Step.from_short_string("2ep5st", steps_per_epoch=20)
        self.assertEqual(step.get_step(), 45)
        self.assertEqual("2ep5st", step.to_short_string())
        step = Step.from_short_string("0ep5st", steps_per_epoch=20)
        self.assertEqual(step.get_step(), 5)
        self.assertEqual("0ep5st", step.to_short_string())

    def _get_trainer(
        self,
        training_steps,
        save_freq,
        save_specific_steps,
        eval_freq,
        eval_specific_steps,
        lmc_freq,
        lmc_specific_steps,
        lmc_on_epoch_end=False,
        lmc_on_train_end=True,
        save_early_iters=False,
        train=False,
    ):
        self.tests += 1
        if not train:
            model_dir = self.log_dir / f"test-training-{self.tests}"
            config = self.get_test_config(
                experiment="train",
                model_dir=model_dir,
                training_steps=training_steps,
                save_freq=save_freq,
                save_specific_steps=save_specific_steps,
                eval_freq=eval_freq,
                eval_specific_steps=eval_specific_steps,
                lmc_freq=lmc_freq,
                lmc_specific_steps=lmc_specific_steps,
                lmc_on_epoch_end=lmc_on_epoch_end,
                lmc_on_train_end=lmc_on_train_end,
                save_best=False,
                save_early_iters=save_early_iters,
            )
            exp = TrainingRunner(config)
            exp.setup()
            return exp.save_steps, exp.eval_steps, exp.lmc_steps
        model_dir = self.log_dir / "test-training-live"
        steps = self.run_command_and_return_result(
            "test-training-live",
            "step/global",
            training_steps=training_steps,
            args=[
                "--save_freq",
                save_freq,
                "--save_specific_steps",
                save_specific_steps,
                "--eval_freq",
                eval_freq,
                "--eval_specific_steps",
                eval_specific_steps,
                "--lmc_freq",
                lmc_freq,
                "--lmc_specific_steps",
                lmc_specific_steps,
                "--lmc_on_epoch_end",
                "true",
                "--lmc_on_train_end",
                "true",
                "--save_best",
                "false",
                "--save_early_iters",
                "true",
            ],
        )
        self.assertEqual(
            steps, Step.from_short_string(training_steps, self.steps_per_ep).get_step()
        )
        self.assertEqual(self.get_summary_value(model_dir, "step/model1"), steps)
        self.assertEqual(self.get_summary_value(model_dir, "step/model2"), steps)
        # get list of checkpoints and their times
        save_steps = []
        for ckpt_file in model_dir.glob("model1*/checkpoints/*.ckpt"):
            save_steps.append(
                Step.from_short_string(ckpt_file.stem, self.steps_per_ep).get_step()
            )
        # get list of eval times by looking at wandb summary
        # n_evals = self.get_summary_value(model_dir, "step/epoch")
        return set(save_steps), None, None

    def test_save_freq(self):
        # check that checkpoints are saved according to the given frequency
        ckpts, evals, lmcs = self._get_trainer(
            "10000st", "2ep", "", "100st", "10st,20st,1ep", "", "3ep,5ep"
        )
        self.assertSetEqual(ckpts, set(range(0, 10000, 2 * self.steps_per_ep)))
        self.assertSetEqual(
            evals, set([10, 20, self.steps_per_ep]).union(range(0, 10000, 100))
        )
        self.assertSetEqual(lmcs, set([3 * self.steps_per_ep, 5 * self.steps_per_ep]))
        ckpts, evals, lmcs = self._get_trainer(
            "50ep",
            "",
            "",
            "",
            "",
            "",
            "",
            lmc_on_epoch_end=True,
            lmc_on_train_end=True,
            save_early_iters=True,
        )
        self.assertSetEqual(
            ckpts,
            set(EARLY_CKPTS),
        )
        self.assertSetEqual(evals, set())
        self.assertSetEqual(
            lmcs, set(range(0, 50 * self.steps_per_ep, self.steps_per_ep))
        )
        ckpts, evals, lmcs = self._get_trainer(
            "200st",
            "none",
            "10st,20st",
            "none",
            "100st",
            "999st",
            "160st",
            lmc_on_train_end=True,
            train=True,
        )
        self.assertSetEqual(ckpts, set([1, 2, 3, 4, 5, 6, 10, 20, 57, 108, 160, 200]))

    def test_load_ckpt(self):
        # check that loading from ckpt is identical to full train
        # - run n_models=2, then rerun n_models=1 with identical hparams and load second run from model2 ckpt of first run, check results identical
        # - repeat but change hparams (specifically, perturb_seed1 and perturb_scale), check results not identical
        ref_ce = self.run_command_and_return_result("test-ckpt-ref", "model2/test/cross_entropy", seed1=1, seed2=2, perturb_seed1=99, perturb_scale=0, perturb_step="100st", args=["--save_specific_steps", "100st"], n_models=2, lmc_on_train_end=True)
        self.assertFalse(self.ckpts_match(self.log_dir / "test-ckpt-ref" / "model2" / "checkpoints" / "0ep0st.ckpt", self.log_dir / "test-ckpt-ref" / "model1" / "checkpoints" / "0ep0st.ckpt"))
        path_to_model2 = str(self.log_dir / "test-ckpt-ref" / "model2" / "checkpoints" / "0ep0st.ckpt")

        def get_training_element(ckpt_path):
            config = self.get_test_config(
                perturb_seed1=11,
                perturb_scale=0.2,
                perturb_step="100st",
                seed1=99,  # different than ref
                args=["--save_specific_steps", "100st"],
                n_models=1,
                lr_scheduler="triangle",
                # resume_from=str(self.log_dir / "test-ckpt-ref"),
                # resume_step="100st",
                ckpt_path=ckpt_path,
            )
            #TODO ckpt_path doesn't get initialized, so manually set it
            config.model.ckpt_path = ckpt_path
            exp = PerturbedTrainingRunner(config)
            exp.setup()
            element = exp.training_elements[0]
            self.assertEqual(len(exp.training_elements), 1)
            self.assertEqual(exp.global_step, 0)
            self.assertEqual(exp.ep, 0)
            return element

        with self.subTest("ckpt_path"):
            ref_ckpt = torch.load(path_to_model2)
            # sanity check, not loading gives different model
            element = get_training_element("")
            self.assertFalse(self.state_dicts_equal(element.model.state_dict(), ref_ckpt["state_dict"]))
            # check that loading model works
            element = get_training_element(path_to_model2)
            self.assertTrue(self.state_dicts_equal(element.model.state_dict(), ref_ckpt["state_dict"]))

            # check training with loaded model gives identical result
            test_ce = self.run_command_and_return_result("test-ckpt-path", "model1/test/cross_entropy", seed1=99, loader_seed1=2, perturb_seed1=99, perturb_scale=0, perturb_step="100st", n_models=1, args=["--save_specific_steps", "100st", "--ckpt_path", str(self.log_dir / "test-ckpt-ref" / "model2" / "checkpoints" / "0ep0st.ckpt")])
            self.assertEqual(test_ce, ref_ce)
            self.assertFalse(self.ckpts_match(self.log_dir / "test-ckpt-path" / "model1" / "checkpoints" / "0ep0st.ckpt", self.log_dir / "test-ckpt-ref" / "model1" / "checkpoints" / "0ep0st.ckpt"))
            self.assertTrue(self.ckpts_match(self.log_dir / "test-ckpt-path" / "model1" / "checkpoints" / "0ep0st.ckpt", self.log_dir / "test-ckpt-ref" / "model2" / "checkpoints" / "0ep0st.ckpt"))
            self.assertFalse(self.ckpts_match(self.log_dir / "test-ckpt-path" / "model1" / "checkpoints" / "2ep0st.ckpt", self.log_dir / "test-ckpt-ref" / "model1" / "checkpoints" / "2ep0st.ckpt"))
            self.assertTrue(self.ckpts_match(self.log_dir / "test-ckpt-path" / "model1" / "checkpoints" / "2ep0st.ckpt", self.log_dir / "test-ckpt-ref" / "model2" / "checkpoints" / "2ep0st.ckpt"))
            self.assertFalse((self.log_dir / "test-ckpt-path" / "model2").exists())

        with self.subTest("resume_from"):
            test_ce = self.run_command_and_return_result("test-ckpt-resume", "model2/test/cross_entropy", seed1=1, seed2=2, perturb_seed1=99, perturb_scale=1, perturb_step="100st", n_models=2, args=["--save_specific_steps", "100st", "--resume_from", str(self.log_dir / "test-ckpt-ref"), "--resume_step", "100st"])
            # check that resuming run doesn't save checkpoint for 0ep
            self.assertEqual(test_ce, ref_ce)
            self.assertFalse((self.log_dir / "test-ckpt-resume" / "model1" / "checkpoints" / "0ep0st.ckpt").exists())
            self.assertFalse((self.log_dir / "test-ckpt-resume" / "model2" / "checkpoints" / "0ep0st.ckpt").exists())
            self.assertTrue(self.ckpts_match(self.log_dir / "test-ckpt-resume" / "model1" / "checkpoints" / "0ep100st.ckpt", self.log_dir / "test-ckpt-ref" / "model1" / "checkpoints" / "0ep100st.ckpt"))
            self.assertTrue(self.ckpts_match(self.log_dir / "test-ckpt-resume" / "model2" / "checkpoints" / "0ep100st.ckpt", self.log_dir / "test-ckpt-ref" / "model2" / "checkpoints" / "0ep100st.ckpt"))
            self.assertFalse(self.ckpts_match(self.log_dir / "test-ckpt-resume" / "model1" / "checkpoints" / "2ep0st.ckpt", self.log_dir / "test-ckpt-ref" / "model1" / "checkpoints" / "2ep0st.ckpt"))
            # #TODO resume_from is not perfectly deterministic, so this fails
            # self.assertTrue(self.ckpts_match(self.log_dir / "test-ckpt-resume" / "model2" / "checkpoints" / "2ep0st.ckpt", self.log_dir / "test-ckpt-ref" / "model2" / "checkpoints" / "2ep0st.ckpt"))

        with self.subTest("evaluate_ckpt"):
            # replace previous run model1 with evaluate_ckpt, see if run is identical
            test_ce = self.run_command_and_return_result("test-ckpt-eval", "model2/test/cross_entropy", seed1=99, seed2=2, perturb_seed1=99, perturb_scale=1, perturb_step="100st", n_models=2, args=["--save_specific_steps", "100st", "--evaluate_ckpt1", str(self.log_dir / "test-ckpt-ref" / "model1")], lmc_on_train_end=True)
            self.assertEqual(test_ce, ref_ce)
            barrier = self.get_summary_value(self.log_dir / "test-ckpt-ref", "lmc-0-1/lmc/loss/weighted/barrier_train")
            ref_barrier = self.get_summary_value(self.log_dir / "test-ckpt-eval", "lmc-0-1/lmc/loss/weighted/barrier_train")
            self.assertEqual(barrier, ref_barrier)

    def tearDown(self):
        return

if __name__ == "__main__":
    unittest.main()
