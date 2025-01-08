import unittest
from pathlib import Path
from typing import Any, Dict

import yaml

from lmc.experiment_config import PerturbedTrainer

config_yaml = Path(__file__).parent.joinpath("perturb.yaml").resolve().absolute()
import unittest
from pathlib import Path

from lmc.experiment.perturb import PerturbedTrainingRunner
from lmc.experiment_config import PerturbedTrainer
from main import run_experiment
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
                self.assertEqual(Path(v).name, conf_val.name, f"Path names don't match {Path(v).name}, {conf_val.name}")
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
        # check that loading from the config to file gives the same config
        # run from command line to generate config file first
        command = self.get_test_command()
        result = self.run_command(command, print_output=True)
        self.assertNotEqual(result, "error")
        cmd_run = self.get_last_created_in_dir(self.log_dir / "*")
        cmd_config = cmd_run / "config.yaml"
        cmd_ckpt_1, cmd_ckpt_2 = self.get_last_ckpts(cmd_run)

        # run programmatically
        with self.subTest("programmatic run"):
            exp = PerturbedTrainingRunner.create_from_file(cmd_config)
            config = exp.config  # compare all runs against this config
            self.assertTrue(run_experiment(exp))
            code_run = self.get_last_created_in_dir(self.log_dir / "*")
            code_ckpt_1, code_ckpt_2 = self.get_last_ckpts(code_run)

            with open(cmd_config) as f:
                self._check_nested_dict(yaml.load(f, Loader=yaml.Loader), config)
            self.assertTrue(self.ckpts_match(code_ckpt_1, cmd_ckpt_1))
            self.assertTrue(self.ckpts_match(code_ckpt_2, cmd_ckpt_2))

        # use config file from command line
        with self.subTest("command line config file"):
            result = self.run_command(f"python main.py perturb --config_file {cmd_config}", print_output=True)
            self.assertNotEqual(result, "error")
            yaml_run = self.get_last_created_in_dir(self.log_dir / "*")
            yaml_config = yaml_run / "config.yaml"
            yaml_ckpt_1, yaml_ckpt_2 = self.get_last_ckpts(yaml_run)

            with open(yaml_config) as f:
                self._check_nested_dict(yaml.load(f, Loader=yaml.Loader), config)
            self.assertTrue(self.ckpts_match(yaml_ckpt_1, cmd_ckpt_1))
            self.assertTrue(self.ckpts_match(yaml_ckpt_2, cmd_ckpt_2))

        # run programmatically again
        with self.subTest("identical run"):
            exp = PerturbedTrainingRunner.create_from_file(cmd_config)
            self.assertTrue(run_experiment(exp))
            second_run = self.get_last_created_in_dir(self.log_dir / "*")
            second_config = second_run / "config.yaml"
            second_ckpt_1, second_ckpt_2 = self.get_last_ckpts(second_run)

            with open(second_config) as f:
                self._check_nested_dict(yaml.load(f, Loader=yaml.Loader), config)
            self.assertTrue(self.ckpts_match(second_ckpt_1, cmd_ckpt_1))
            self.assertTrue(self.ckpts_match(second_ckpt_2, cmd_ckpt_2))


if __name__ == "__main__":
    unittest.main()
