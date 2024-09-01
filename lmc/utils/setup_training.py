import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from shutil import rmtree
from typing import Dict, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from lmc.experiment_config import Trainer

logger = logging.getLogger("setup")
@dataclass
class TrainingElement:
    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler
    train_loader: DataLoader
    train_eval_loader: DataLoader
    test_loader: DataLoader
    seed: int = 42
    loader_seed: int = 42
    aug_seed: int = 42
    optimal_acc: float = -1
    optimal_path: Path = None
    permutation = None
    prev_perm_wm = None
    prev_perm_am = None
    max_steps: int = None
    

def configure_model(config: Trainer) -> 'BaseModel':
    conf = config.model

def configure_optimizer(config: Trainer, model: 'BaseModel'):
    if config.trainer.optimizer.lower() == "sgd":
        opt = optim.SGD(model.parameters(), lr=config.trainer.lr, momentum=config.trainer.mom,
                            weight_decay=config.trainer.weight_decay)
        

def setup_model_dir(config:Trainer):
    if not config.logger.log_dir.exists():
        raise FileNotFoundError(f"Must provide an existing log_dir ({config.logger.log_dir})")
    hashname = config.hashname
    now = datetime.now()
    formatted_date = now.strftime("%d-%m-%y-%H-%M-%f")
    model_dir = Path(config.logger.log_dir, f"{hashname}-{formatted_date}")
    model_dir.mkdir(exist_ok=True)
    model_dir.joinpath("checkpoints").mkdir(exist_ok=True)
    
    # Save code as zip and config as yaml into the model directory.
    config.save(model_dir)

    config.model_dir = model_dir
    logger.info(f"Created model dir: {model_dir}")
    return model_dir

def setup_device():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}.")
    return device

def setup_experiment(config: Trainer) -> Tuple[Dict[int, TrainingElement], torch.device]:
    """Creates all necessary elements. models, datamodules, etc."""
    device = setup_device()
    model_dir = setup_model_dir(config)
    return
    config.model_dir.joinpath("barriers").mkdir(exist_ok=True)

    setup_wandb(config)
    training_elements: Dict[int, TrainingElement] = dict()
    assert config.n_models >= 1, f"n_models ({config.n_models}) should be >= 1."
    first_seed = config.seed1 if hasattr(config, "seed1") else config.trainer.seed
    for i in range(1, config.n_models + 1):
        suffix = str(i) if config.n_models > 1 else ""
        # suffix = str(i) if config.n_models > 1 else ""
        seed = config.trainer.seed
        loader_seed = seed
        aug_seed = seed
        if hasattr(config, f"seed{suffix}"):
            seed = getattr(config, f"seed{suffix}")
            loader_seed = getattr(config, f"loader_seed{suffix}")
            aug_seed = getattr(config, f"aug_seed{suffix}")
        c = config.data.export_dict() | {
            "seed": seed,
            "loader_seed": loader_seed,
            "aug_seed": aug_seed,
        }
        data_ = DataConfig(**c)
        dm = DataModule(data_)
        dm.prepare_data()
        dm.setup("fit")
        train_loader, test_loader = dm.train_dataloader(), dm.test_dataloader()
        dm.setup("test")
        train_eval_loader = dm.train_dataloader()
        dm.setup("fit")
        logger.info("Setup datamodule %d with loader_seed=%d, aug_seed=%d.", i, loader_seed, aug_seed)

        config.trainer.seed = seed
        # if config.n_models == 1:
        #     model_dir_ = model_dir
        # else:
        if True:
            model_dir_ = model_dir.joinpath(f"model{i}-seed_{seed}-ls_{loader_seed}-as_{aug_seed}")
        #  TODO: find a better way for multiple models
        seed_everything(seed)
        config.model.permute_from = None
        perms = None
        if hasattr(config, "resume_from") and (config.resume_from) and (config.resume_epoch > 0):
            config.model.ckpt_path = model_dir_.joinpath("checkpoints", f"epoch_{config.resume_epoch}").with_suffix(
                ".ckpt"
            )
            if not config.model.ckpt_path.exists():
                config.model.ckpt_path = model_dir_.joinpath("checkpoints", f"{config.resume_epoch}").with_suffix(
                    ".ckpt"
                )
            logger.info("Model will be loaded from %s.", config.model.ckpt_path)
            assert config.model.ckpt_path.exists(), f"Path {config.model.ckpt_path} doesn't exist."
            ckpt = torch.load(config.model.ckpt_path, map_location=device)

        if config.command == "permute-train-together" and i > 1:
            logger.info("Now setting up model %d, switching the seed to first seed (%d).", i, first_seed)
            config.trainer.seed = first_seed
            seed_everything(first_seed)
            config.model.permute_from = True
            # for now make sure to provide the same seed seed1=seed2
            model, perms = configure_model(config, model_dir, device, return_perms=True, model_ind=i)
            logger.info("Model %d setup complete, switching the seed to its own seed (%d).", i, seed)
            seed_everything(seed)

        elif config.command == "perturb-weights" and i > 1:
            # config.model.add_noise = True
            model = configure_model(config, model_dir, device, return_perms=False, model_ind=i)
        else:
            model = configure_model(config, model_dir, device, return_perms=False, model_ind=i)
        logger.info("Setup model %d with seed=%d.", i, seed)

        steps_per_epoch = len(train_loader)
        max_epochs = config.trainer.max_epochs
        max_steps = steps_per_epoch * max_epochs
        seed_everything(seed)
        opt = configure_optimizer(
            model,
            optimizer=config.optimizer.optimizer,
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            betas=config.optimizer.betas,
            momentum=config.optimizer.momentum,
            beta=config.optimizer.beta,
        )
        scheduler = configure_lr_scheduler(
            opt,
            opt_conf=config.optimizer,
            lr_scheduler=config.optimizer.lr_scheduler,
            lr_schedule=config.optimizer.lr_schedule,
            max_epochs=max_epochs,
            warmup_ratio=config.optimizer.warmup_ratio,
            steps_per_epoch=len(train_loader),
            lr=config.optimizer.lr,
        )
        if hasattr(config, "resume_from") and (config.resume_from) and (config.resume_epoch > 0):
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            logger.info("Optimizer loaded from ckpt")
            if scheduler is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                logger.info("Scheduler loaded from ckpt")
        # so that they have the same lr scheduler
        if (
            i > 1
            and (hasattr(config, "perturb_step") and config.perturb_step > 0)
            and (hasattr(config, "same_steps_post_perturb") and config.same_steps_post_perturb)
        ):
            max_steps += config.perturb_step
            max_epochs += config.perturb_step // steps_per_epoch
        scaler = None
        if config.trainer.use_scaler:
            scaler = GradScaler()
        if config.logger.print_optimizers:
            logger.info("Optimizer %d - %s", i, opt)

        # if not hasattr(config, f"model{i}_dir"):
        setattr(config, f"model{i}_dir", model_dir_)
        model_dir_.joinpath("checkpoints").mkdir(exist_ok=True, parents=True)
        training_elements[i] = TrainingElement(
            model=model,
            opt=opt,
            scheduler=scheduler,
            scaler=scaler,
            model_dir=model_dir_,
            seed=seed,
            loader_seed=loader_seed,
            aug_seed=aug_seed,
            train_loader=train_loader,
            train_eval_loader=train_eval_loader,
            test_loader=test_loader,
            permutation=perms,
            max_steps=max_steps,
        )
    seed_everything(first_seed)
    if config.command == "perturb-weights":
        with torch.no_grad():
            vec1, vec2 = parameters_to_vector(training_elements[1].model.parameters()), parameters_to_vector(
                training_elements[2].model.parameters()
            )
            dist = torch.linalg.norm(vec1 - vec2).detach()
            logger.info("Initial frobenius distance between two models: %f.", dist)
    logger.info("Model setups complete, switching seed to %d, which is passed to the first model.", first_seed)

    return training_elements, device

def cleanup(config: Trainer):
    if config.logger.cleanup_after:
        if not config.hashname in config.model_dir.as_posix():
            logger.info(f"Not sure if the model_dir is an outer directory, cleanup manually ({config.model_dir})")
            return
        logger.info("Deleteing %s", config.model_dir)
        rmtree(config.model_dir)