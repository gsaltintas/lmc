import unittest
from pathlib import Path


from lmc.experiment_config import PerturbedTrainer
from lmc.experiment.perturb import PerturbedTrainingRunner
from main import run_experiment
from tests.base import BaseTest


class TestConfig(BaseTest):
    CONFIG_YAML = Path(__file__).parent.joinpath("perturb.yaml").resolve().absolute()

    def test_build(self):
        conf = PerturbedTrainer.load_from_file(self.CONFIG_YAML)
        self.assertIsInstance(conf, PerturbedTrainer)
        print(conf.model)
        conf.save(self.log_dir, zip_code_base=False)
        new_yaml = self.log_dir / "config.yaml"
        self.assertTrue(self.text_files_equal(self.CONFIG_YAML, new_yaml))
        # check that we can load from the new yaml without issues
        PerturbedTrainer.load_from_file(new_yaml)

    def test_cmd_same_as_yaml(self):
        # check that loading from the config to file gives the same config
        with self.subTest("command line vs config file"):
            # run from command line
            command = self.get_test_command()
            self.run_command(command)
            cmd_run = self.get_last_created_in_dir(self.log_dir / "*")
            cmd_config = cmd_run / "config.yaml"
            cmd_ckpt_1, cmd_ckpt_2 = self.get_last_ckpts(cmd_run)

            # run from the config file instead
            self.run_command(f"python main.py perturb --config_file {cmd_config}")
            yaml_run = self.get_last_created_in_dir(self.log_dir / "*")
            yaml_config = yaml_run / "config.yaml"
            yaml_ckpt_1, yaml_ckpt_2 = self.get_last_ckpts(yaml_run)

            self.assertTrue(self.text_files_equal(cmd_config, yaml_config))
            self.assertTrue(self.ckpts_match(cmd_ckpt_1, yaml_ckpt_1))
            self.assertTrue(self.ckpts_match(cmd_ckpt_2, yaml_ckpt_2))

        # run programmatically
        with self.subTest("programmatic run"):
            exp = PerturbedTrainingRunner.create_from_file(self.CONFIG_YAML)
            run_experiment(exp)
            code_run = self.get_last_created_in_dir(self.log_dir / "*")
            code_config = code_run / "config.yaml"
            code_ckpt_1, code_ckpt_2 = self.get_last_ckpts(code_run)

            self.assertTrue(self.text_files_equal(code_config, yaml_config))
            self.assertTrue(self.ckpts_match(code_ckpt_1, yaml_ckpt_1))
            self.assertTrue(self.ckpts_match(code_ckpt_2, yaml_ckpt_2))

        # run programmatically again
        with self.subTest("identical run"):
            exp = PerturbedTrainingRunner.create_from_file(self.CONFIG_YAML)
            run_experiment(exp)
            second_run = self.get_last_created_in_dir(self.log_dir / "*")
            second_config = second_run / "config.yaml"
            second_ckpt_1, second_ckpt_2 = self.get_last_ckpts(second_run)

            self.assertTrue(self.text_files_equal(second_config, code_config))
            self.assertTrue(self.ckpts_match(second_ckpt_1, code_ckpt_1))
            self.assertTrue(self.ckpts_match(second_ckpt_2, code_ckpt_2))


if __name__ == "__main__":
    unittest.main()
