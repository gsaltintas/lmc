from copy import deepcopy
import unittest
from pathlib import Path
from typing import Any, Dict
import yaml

from lmc.experiment.perturb import PerturbedTrainingRunner
from lmc.experiment_config import PerturbedTrainer
from lmc.utils.run import command_result_is_error, run_command
from tests.base import BaseTest

PERTURB_SCALE = 0.1


class TestConfig(BaseTest):
    CONFIG_YAML = Path(__file__).parent.joinpath("perturb.yaml").resolve().absolute()

    def _check_nested_dict(self, d: Dict[str, Any], conf):
        for k, v in d.items():
            conf_val = getattr(conf, k)
            if hasattr(conf, k) and isinstance(v, dict):
                self._check_nested_dict(v, conf_val)
            elif isinstance(conf_val, Path):
                self.assertEqual(
                    Path(v).name,
                    conf_val.name,
                    f"Path names don't match {Path(v).name}, {conf_val.name}",
                )
            else:
                self.assertEqual(v, conf_val, f"mismatch at {k}")

    def tearDown(self):
        pass

    def test_build(self):
        conf = PerturbedTrainer.load_from_file(self.CONFIG_YAML)
        with open(self.CONFIG_YAML) as stream:
            file_dct = yaml.load(stream, Loader=yaml.Loader)

        self.assertIsInstance(conf, PerturbedTrainer)
        conf.save(self.log_dir, zip_code_base=False)
        new_yaml = self.log_dir / "config.yaml"

        with self.subTest("load from yaml"):
            self._check_nested_dict(file_dct, conf)
        with self.subTest("reload experiment"):
            conf2 = PerturbedTrainer.load_from_file(new_yaml)
            self._check_nested_dict(file_dct, conf2)
        # check that we can load from the new yaml without issues
        PerturbedTrainer.load_from_file(new_yaml)

    def test_cmd_same_as_yaml(self):
        # run from command line to generate config file first
        command = self.get_test_command(model_dir=self.log_dir / "test-config-base")
        result = run_command(command, print_output=True)
        self.assertFalse(command_result_is_error(result))
        base_run = self.log_dir / "test-config-base"
        base_config = PerturbedTrainer.load_from_file(base_run / "config.yaml")
        base_ckpt_1, base_ckpt_2 = self.get_last_ckpts(base_run)

        def copy_config(model_dir):
            config = deepcopy(base_config)
            config.model_dir = self.log_dir / model_dir
            return config

        def get_last_run_results(model_dir):
            run_dir = self.log_dir / model_dir
            self.assertEqual(run_dir.name, model_dir)
            ckpt_1, ckpt_2 = self.get_last_ckpts(run_dir)
            config_file = run_dir / "config.yaml"
            with open(config_file) as f:
                config_yaml = yaml.load(f, Loader=yaml.Loader)
            self.assertNotEqual(config_yaml["model_dir"], base_config.model_dir)
            config_yaml["model_dir"] = base_config.model_dir
            self._check_nested_dict(config_yaml, base_config)
            match_1 = self.ckpts_match(ckpt_1, base_ckpt_1)
            match_2 = self.ckpts_match(ckpt_2, base_ckpt_2)
            return match_1, match_2

        with self.subTest("cmd: run from command line passing config file"):
            config = copy_config("test-config-cmd")
            config.save(self.log_dir / "test-config-cmd-yaml", zip_code_base=False)
            result = run_command(
                f"python main.py perturb --config_file {self.log_dir / 'test-config-cmd-yaml' / 'config.yaml'}", print_output=True
            )
            self.assertFalse(command_result_is_error(result))
            ckpts_match_1, ckpts_match_2 = get_last_run_results("test-config-cmd")
            self.assertTrue(ckpts_match_1 and ckpts_match_2)

        with self.subTest("obj: run programmatically using config object, and change hparams"):
            base_config.seeds.seed1 += 1
            exp = PerturbedTrainingRunner(copy_config("test-config-obj"))
            self.assertTrue(exp.run_experiment())
            ckpts_match_1, ckpts_match_2 = get_last_run_results("test-config-obj")
            self.assertFalse(ckpts_match_1)
            self.assertTrue(ckpts_match_2)


if __name__ == "__main__":
    unittest.main()
