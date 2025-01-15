import os
import unittest
from glob import glob
from pathlib import Path
from shutil import rmtree

import torch


class BaseTest(unittest.TestCase):
    TEST_COMMAND = """python main.py {experiment} {model_dir}  \
            --project test-project  \
                --run_name test-{experiment}  \
                --path {data_dir} \
                --log_dir {log_dir}  \
                --save_early_iters false  \
                --cleanup_after false  \
                --use_wandb {use_wandb}  \
                    --project {project} \
                    --run_name {run_name}
                --wandb_offline {wandb_offline}  \
                --zip_and_save_source false  \
            --model_name {model_name}  \
                --norm layernorm  \
            --dataset {dataset}  \
                --hflip true  \
                --random_rotation 10  \
                --random_crop false  \
            --optimizer sgd  \
                --training_steps 2ep  \
                --lr 0.1   \
                --lr_scheduler triangle  \
                --warmup_ratio 0.02  \
                --momentum 0.9  \
                --rewind_lr {rewind_lr}  \
            --n_models 2  \
                --perturb_mode gaussian  \
                --perturb_step {perturb_step}  \
                --perturb_inds {perturb_inds}  \
                --perturb_scale {perturb_scale}  \
                --same_steps_pperturb {same_steps_pperturb}  \
            --deterministic {deterministic}  \
                --seed1 {seed1}  \
                --seed2 {seed2}  \
                --loader_seed1 {seed1}  \
                --loader_seed2 {seed2}  \
                --perturb_seed1 {seed1}  \
                --perturb_seed2 {seed2}  \
            --lmc_check_perms false  \
                --lmc_on_epoch_end false  \
                --lmc_on_train_end {lmc_on_train_end}  \
                --n_points 3  \
    {args}"""

    SEED_1 = 42
    SEED_2 = 41

    def setUp(self):
        test_path = Path(
            os.path.relpath(os.path.dirname(os.path.realpath(__file__)), os.getcwd())
        )
        self.log_dir = test_path / "tmp"
        self.log_dir.mkdir(exist_ok=True)

        self.data_dir = test_path / "data"
        data_dir_env_var = os.environ.get("DATASET_DIR")
        if data_dir_env_var is None:
            raise ValueError("DATASET_DIR environment variable not set")
        if not self.data_dir.exists() and data_dir_env_var is not None:
            os.symlink(data_dir_env_var, self.data_dir, target_is_directory=True)

    def tearDown(self):
        rmtree(self.log_dir)

    def get_test_command(
        self,
        experiment="perturb",
        seed1=SEED_1,
        seed2=SEED_2,
        perturb_step=0,
        perturb_scale=0,
        deterministic=True,
        model_name="mlp/128x3",
        dataset="mnist",
        perturb_inds=[1],
        rewind_lr="false",
        use_wandb="false",
        wandb_offline="true",
        same_steps_pperturb="false",
        lmc_on_train_end="false",
        project="lmc-test",
        run_name=None,
        model_dir=None,
        args=[],
    ):
        command = str.format(
            self.TEST_COMMAND,
            experiment=experiment,
            seed1=seed1,
            seed2=seed2,
            perturb_step=perturb_step,
            perturb_scale=perturb_scale,
            deterministic=deterministic,
            log_dir=self.log_dir,
            data_dir=self.data_dir,
            model_name=model_name,
            dataset=dataset,
            perturb_inds=" ".join(str(x) for x in perturb_inds),
            rewind_lr=rewind_lr,
            use_wandb=use_wandb,
            wandb_offline=wandb_offline,
            same_steps_pperturb=same_steps_pperturb,
            lmc_on_train_end=lmc_on_train_end,
            project=project,
            run_name=run_name,
            model_dir="" if model_dir is None else f"--model_dir {model_dir}",
            args=args if isinstance(args, str) else " ".join(args),
        )
        return command

    @staticmethod
    def get_last_ckpts(exp_dir):
        model_1 = next(exp_dir.glob("model1*"))
        model_2 = next(exp_dir.glob("model2*"))
        ckpt_1 = BaseTest.get_last_created_in_dir(model_1 / "checkpoints" / "ep-*.ckpt")
        ckpt_2 = BaseTest.get_last_created_in_dir(model_2 / "checkpoints" / "ep-*.ckpt")
        return ckpt_1, ckpt_2

    @staticmethod
    def get_last_created_in_dir(path):
        children = glob(str(path))
        if len(children):
            return Path(max(children, key=os.path.getctime))
        return None

    @staticmethod
    def ckpts_match(ckpt_1, ckpt_2):
        sd_1 = torch.load(ckpt_1)["state_dict"]
        sd_2 = torch.load(ckpt_2)["state_dict"]
        if set(sd_1.keys()) != set(sd_2.keys()):
            return False
        for k, v in sd_1.items():
            if not torch.equal(v, sd_2[k]):
                return False
        return True

    @staticmethod
    def text_files_equal(file_1, file_2):
        with open(file_1, "r") as f1:
            with open(file_2, "r") as f2:
                for i, (l1, l2) in enumerate(zip(f1, f2)):
                    if l1 != l2:
                        print(f"Lines differ, {i}:\n{l1}\n{l2}")
                        return False
        return True
