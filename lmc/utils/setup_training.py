import importlib
import logging
import math
import os
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from shutil import rmtree
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
)

import wandb
from lmc.config import DataConfig
from lmc.data.data_stats import DatasetRegistry, TaskType
from lmc.experiment_config import Experiment, Trainer
from lmc.models import MLP, ResNet
from lmc.models.bert import Bert
from lmc.models.t5 import T5
from lmc.models.utils import count_parameters
from lmc.utils.seeds import seed_everything, seed_worker
from lmc.utils.step import Step
from lmc.utils.training_element import Iterator, TrainingElement, TrainingElements

logger = logging.getLogger("setup")

WANDB_DIR = os.environ.get("SCRATCH", os.environ.get("TMPDIR"))


def load_model_from_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    ignore_mismatched_sizes: bool = True,
) -> None:
    """
    Load weights from a checkpoint into a model, gracefully handling missing or mismatched keys.
    Similar to HuggingFace's approach, where missing keys retain their random initialization.

    Args:
        model: PyTorch model to load weights into
        path: Path to checkpoint file
        ignore_mismatched_sizes: Whether to skip loading weights with mismatched sizes
    """
    # Load checkpoint
    model_state_dict = model.state_dict()

    # Track different types of keys
    missing_keys = []
    unexpected_keys = []
    mismatched_keys = []
    to_be_loaded = OrderedDict()

    # Load matching keys
    for k, v in state_dict.items():
        if k not in model_state_dict:
            unexpected_keys.append(k)
            continue

        if v.shape != model_state_dict[k].shape:
            if ignore_mismatched_sizes:
                print(k)
                mismatched_keys.append(k)
                continue
            else:
                raise ValueError(
                    f"Size mismatch for {k}: checkpoint has {v.shape}, model has {model_state_dict[k].shape}"
                )

        to_be_loaded[k] = v

    # Identify missing keys
    missing_keys = [k for k in model_state_dict.keys() if k not in to_be_loaded]

    # Load the matching weights
    model.load_state_dict(to_be_loaded, strict=False)

    # Log informative messages
    if unexpected_keys:
        logger.info("Keys in checkpoint but not in model: %s", unexpected_keys)

    if mismatched_keys:
        logger.info("Keys skipped due to size mismatch: %s", mismatched_keys)

    if missing_keys:
        keys_str = ", ".join(missing_keys)
        logger.warning(
            "Some weights are newly initialized and should be trained: %s\n"
            "You should TRAIN these layers to ensure good performance!",
            keys_str,
        )


def should_freeze_layer(layer_name: str, pattern: str) -> bool:
    """Check if layer should be frozen based on exact match or regex pattern"""
    try:
        # First try exact match
        if pattern == layer_name:
            return True
        # Then try as regex pattern
        if re.match(pattern, layer_name):
            return True
    except re.error:
        # If pattern is invalid regex, treat as normal string
        return pattern in layer_name
    return False


def freeze_layers(model, frozen_layers: List):
    """Freeze layers matching either exact names or regex patterns"""
    frozen_count = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += 1
        if any(should_freeze_layer(name, pattern) for pattern in frozen_layers):
            param.requires_grad = False
            frozen_count += 1
            logger.info(f"Frozen layer: {name}")

    if frozen_count > 0:
        logger.info(
            f"Froze {frozen_count}/{total_params} layers based on {len(frozen_layers)} patterns"
        )


def configure_nlp_model(config: Trainer, device: torch.device) -> torch.nn.Module:
    """Configure model for NLP tasks"""
    conf = config.model

    # Setup tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        config.data.tokenizer_name,
        cache_dir=str(config.data.path),
        model_max_length=config.data.max_seq_length,
        padding_side=config.data.padding_side,
        truncation_side=config.data.truncation_side,
    )
    if config.data.dataset not in DatasetRegistry.get_available_datasets():
        raise ValueError(f"Unkown output dimension for dataset {config.data.dataset}")
    out = (
        config.data.get_num_labels()
    )  # This will already raise an appropriate error if dataset not found

    if "t5" in conf.model_name.lower():
        model = T5(conf.model_name)
    elif "bert" in conf.model_name.lower():
        model = Bert(
            conf.model_name,
            output_dim=out,
            initialization_strategy=conf.initialization_strategy,
        )
    # Determine model type based on task
    elif config.data.dataset.startswith("glue"):
        model = AutoModelForSequenceClassification.from_pretrained(
            conf.model_name, num_labels=out, cache_dir=str(config.data.path)
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            conf.model_name, cache_dir=str(config.data.path)
        )

    model = model.to(device)
    logger.info(f"Created NLP model: {conf.model_name}")
    return model, tokenizer


def configure_vision_model(config: Trainer, device: torch.device) -> "BaseModel":
    """creates a model given the configuration"""
    conf = config.model
    out = (
        config.data.get_num_labels()
    )  # This will already raise an appropriate error if dataset not found
    # TODO: load checkpoints
    if "mlp" in conf.model_name:
        model = MLP.get_model_from_code(
            conf.model_name,
            output_dim=out,
            input_dim=config.data.get_num_in_channels()
            * config.data.get_default_res() ** 2,
            initialization_strategy=conf.initialization_strategy,
            norm=conf.norm,
            act=conf.act,
            hidden_dim=conf.width,
            depth=conf.num_hidden_layers + 1,
        )
    elif "resnet" in conf.model_name:
        if conf.model_name == "resnet":
            if config.data.dataset.lower() == "cifar10":
                conf.model_name = "resnet20"
        model = ResNet.get_model_from_code(
            model_code=conf.model_name,
            output_dim=out,
            initialization_strategy=conf.initialization_strategy,
            norm=conf.norm,
        )

    logger.info("Model created.")
    # logger.info
    model = model.to(device)
    logger.info(
        f"Total number of trainable parameters {count_parameters(model) / 1e6} (M)."
    )
    return model


def configure_model(
    config: Trainer,
    device: torch.device,
    seed: int = None,
    print_output=True,
    state_dict=None,
) -> Tuple["BaseModel", Optional[AutoTokenizer]]:
    seed_everything(seed)
    """ creates a model given the configuration """
    tokenizer = None
    if config.data.is_language_dataset():
        model, tokenizer = configure_nlp_model(config, device)
    else:
        model = configure_vision_model(config, device)
    # optionally load from ckpt
    if state_dict is None:  # will not be None if called from setup_experiment
        _, _, state_dict, _, _ = get_resume_state_dicts(config)
    if state_dict:
        load_model_from_state_dict(model, state_dict)
        logger.info("Model state dict updated")
    if print_output:
        print(model)
    return model, tokenizer


def configure_optimizer(config: Trainer, model: "BaseModel", state_dict=None):
    opt_conf = config.trainer.opt
    if opt_conf.optimizer.lower() == "sgd":
        opt = optim.SGD(
            model.parameters(),
            lr=opt_conf.lr,
            momentum=opt_conf.momentum,
            weight_decay=opt_conf.weight_decay,
        )
    elif opt_conf.optimizer.lower() == "adam":
        opt = optim.Adam(
            model.parameters(),
            lr=opt_conf.lr,
            betas=opt_conf.betas,
            weight_decay=opt_conf.weight_decay,
        )
    elif opt_conf.optimizer.lower() == "adamw":
        opt = optim.AdamW(
            model.parameters(),
            lr=opt_conf.lr,
            betas=opt_conf.betas,
            weight_decay=opt_conf.weight_decay,
        )
    else:
        raise NotImplementedError(f"Optimizer ({opt_conf.optimizer}) not implemented.")
    # optionally load from ckpt
    if state_dict is None:  # will not be None if called from setup_experiment
        __, _, _, state_dict, _ = get_resume_state_dicts(config)
    if state_dict:  # skipped if empty dict
        opt.load_state_dict(state_dict)
        logger.info("Optimizer state dict updated")
    return opt


def configure_lr_scheduler(
    optimizer: optim.Optimizer,
    training_steps: int,
    lr_scheduler: str = None,
    warmup_ratio: float = 0,
    lr_schedule: dict = None,
    global_step: int = 0,
    warmup_steps: int = None,
    state_dict=None,
):
    warmup_steps = (
        math.ceil(warmup_ratio * training_steps)
        if warmup_steps is None
        else warmup_steps
    )
    base_lr = optimizer.param_groups[0]["lr"]
    if lr_scheduler is None or lr_scheduler.lower() == "none":
        return None
    scheduler = None

    if lr_scheduler == "linear":
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=lr_schedule.get("start_factor", 1.0 / 3),
            end_factor=lr_schedule.get("end_factor", 1.0),
        )
    elif lr_scheduler == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=lr_schedule.get("gamma", 0.90)
        )
    elif lr_scheduler == "onecycle":  # cosine annealing with warmup
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=base_lr,
            total_steps=training_steps,
            anneal_strategy="cos",
            pct_start=warmup_ratio,
        )
    elif lr_scheduler == "flat":
        # Adjust the schedule to account for continuation
        start_ind = global_step if global_step < training_steps else 0
        # resetting lr scheduler, needs to be followed by reset_base_lrs
        if global_step > 0:
            schedule = np.interp(
                np.arange(0, training_steps + 1),
                [0, warmup_steps, global_step, training_steps],
                [0, 1, 1, 1],
            )[start_ind:]
        else:
            schedule = np.interp(
                np.arange(0, training_steps + 1),
                [0, warmup_steps, training_steps],
                [0, 1, 1],
            )[start_ind:]
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda x: schedule[x]
        )
    elif lr_scheduler == "triangle":
        # Adjust the schedule to account for continuation
        start_ind = global_step if global_step < training_steps else 0
        # resetting lr scheduler, needs to be followed by reset_base_lrs
        if global_step > 0:
            schedule = np.interp(
                np.arange(0, training_steps + 1),
                [0, warmup_steps, global_step, training_steps],
                [0, 1, 1, 0],
            )[start_ind:]
        else:
            schedule = np.interp(
                np.arange(0, training_steps + 1),
                [0, warmup_steps, training_steps],
                [0, 1, 0],
            )[start_ind:]
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda x: schedule[x]
        )
    elif lr_scheduler == "triangleold":
        start_ind = 1 if warmup_steps else 0
        schedule = np.interp(
            np.arange(training_steps + 2),
            [0, warmup_steps, training_steps + 1],
            [0, 1, 0],
        )[start_ind:]

        schedule = np.interp(
            np.arange(training_steps + 1), [0, warmup_steps, training_steps], [0, 1, 0]
        )[start_ind:]
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda x: schedule[x]
        )
    else:
        raise ValueError(f"Unkonwn lr_scheduler {lr_scheduler}")
    # optionally load from ckpt
    if scheduler is not None and state_dict:  # skipped if empty dict
        scheduler.load_state_dict(state_dict)
        logger.info("Scheduler state dict updated")
    return scheduler


def get_task_preprocessor(
    task_type: TaskType,
    tokenizer: PreTrainedTokenizer,
    data_conf: DataConfig,
    evaluate: bool,
) -> Callable:
    """Returns the appropriate preprocessing function for the given task type."""

    def preprocess_classification(examples):
        tokenizer_kwargs = {
            "padding": True if evaluate else False,
            "truncation": True,
            "max_length": data_conf.max_seq_length,
        }
        # Handle both single strings and lists of strings
        text = examples.get("text", examples.get("sentence", ""))
        if data_conf.lowercase:
            # Handle both single string and list of strings
            if isinstance(text, str):
                text = text.lower()
            else:
                text = [t.lower() for t in text]
        tokenized = tokenizer(text, **tokenizer_kwargs)
        # Add labels to the tokenized output
        labels = examples.get("label", examples.get("labels", None))
        if labels is not None:
            tokenized["labels"] = labels
        return tokenized

    def preprocess_sequence_pair(examples):
        tokenizer_kwargs = {
            "padding": True if evaluate else False,
            "truncation": True,
            "max_length": data_conf.max_seq_length,
        }
        # Make sure we get lists of strings
        text1 = examples.get(
            "sentence1", examples.get("question", examples.get("question1", []))
        )
        text2 = examples.get(
            "sentence2", examples.get("context", examples.get("question2", []))
        )

        if data_conf.lowercase:
            text1 = [t.lower() if t is not None else "" for t in text1]
            text2 = [t.lower() if t is not None else "" for t in text2]
        # Get labels

        labels = examples.get("label", examples.get("labels", None))
        # Filter out -1 labels or map them to a valid class
        labels = [
            l if l != -1 else 0 for l in labels
        ]  # Map -1 to 0 or handle as needed

        if labels is None:
            raise ValueError(
                "No labels found in dataset (looking for 'label' or 'labels' field)"
            )
        tokenized = tokenizer(text1, text2, **tokenizer_kwargs)

        if labels is not None:
            tokenized["labels"] = labels
        return tokenized

    def preprocess_generation(examples):
        tokenizer_kwargs = {
            "padding": True if evaluate else False,
            "truncation": True,
            "max_length": data_conf.max_seq_length,
        }
        # Make sure we get a list of strings
        if isinstance(examples["text"], str):
            text = [examples["text"]]
        else:
            text = examples["text"]

        if data_conf.lowercase:
            text = [t.lower() for t in text]
        return tokenizer(text, **tokenizer_kwargs)

    def preprocess_qa(examples):
        tokenizer_kwargs = {
            "padding": True if evaluate else False,
            "truncation": True,
            "max_length": data_conf.max_seq_length,
            "return_offsets_mapping": True,
        }
        # Make sure we get lists of strings
        questions = examples["question"]
        contexts = examples["context"]

        # Handle possible single string inputs
        if isinstance(questions, str):
            questions = [questions]
        if isinstance(contexts, str):
            contexts = [contexts]

        if data_conf.lowercase:
            questions = [q.lower() for q in questions]
            contexts = [c.lower() for c in contexts]
        return tokenizer(questions, contexts, **tokenizer_kwargs)

    def preprocess_nli(examples):
        tokenizer_kwargs = {
            "padding": True if evaluate else False,
            "truncation": True,
            "max_length": data_conf.max_seq_length,
        }

        # Get labels
        labels = examples.get("label", examples.get("labels", None))
        # Filter out -1 labels or map them to a valid class
        labels = [
            l if l != -1 else 0 for l in labels
        ]  # Map -1 to 0 or handle as needed

        if labels is None:
            raise ValueError(
                "No labels found in dataset (looking for 'label' or 'labels' field)"
            )
        # TODO: pass these keys to datasetinfo
        # NLI typically has premise and hypothesis
        premise = examples.get(
            "premise",
            examples.get(
                "sentence1", examples.get("question", examples.get("question1", []))
            ),
        )
        hypothesis = examples.get(
            "hypothesis",
            examples.get(
                "sentence2", examples.get("sentence", examples.get("question2", []))
            ),
        )

        if data_conf.lowercase:
            premise = [p.lower() for p in premise]
            hypothesis = [h.lower() for h in hypothesis]

        # Tokenize inputs
        tokenized = tokenizer(premise, hypothesis, **tokenizer_kwargs)

        # Add labels to the tokenized output
        tokenized["labels"] = labels

        return tokenized

    def preprocess_sequence_labeling(examples):
        tokenizer_kwargs = {
            "padding": True if evaluate else False,
            "truncation": True,
            "max_length": data_conf.max_seq_length,
            "return_offsets_mapping": True,
            "return_special_tokens_mask": True,
        }
        # Get the text and labels
        tokens = examples.get("tokens", examples.get("words", []))
        labels = examples.get("tags", examples.get("labels", []))

        if data_conf.lowercase:
            tokens = [[t.lower() for t in seq] for seq in tokens]

        # Tokenize the text
        tokenized = tokenizer(tokens, is_split_into_words=True, **tokenizer_kwargs)

        if labels is not None:
            # Align labels with wordpiece tokens
            aligned_labels = []
            for i, label in enumerate(labels):
                word_ids = tokenized.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(
                            -100 if not data_conf.label_all_tokens else label[word_idx]
                        )
                    previous_word_idx = word_idx
                aligned_labels.append(label_ids)
            tokenized["labels"] = aligned_labels

        return tokenized

    preprocessors = {
        TaskType.CLASSIFICATION: preprocess_classification,
        TaskType.SEQUENCE_PAIR: preprocess_sequence_pair,
        TaskType.GENERATION: preprocess_generation,
        TaskType.QUESTION_ANSWERING: preprocess_qa,
        TaskType.NATURAL_LANGUAGE_INFERENCE: preprocess_nli,
        TaskType.SEQUENCE_LABELING: preprocess_sequence_labeling,
        TaskType.REGRESSION: preprocess_sequence_pair,
    }

    return preprocessors[task_type]


def setup_nlp_loader(
    data_conf: DataConfig,
    train: bool,
    evaluate: bool,
    tokenizer: PreTrainedTokenizer,
    loader_seed: Optional[int] = None,
) -> DataLoader:
    """Setup data loader for NLP tasks"""
    if loader_seed is not None:
        torch.manual_seed(loader_seed)
        g = torch.Generator()
        g.manual_seed(loader_seed)

    dataset_conf = data_conf.dataset_info
    task_type = dataset_conf.task_type

    # Determine split based on dataset
    splits = dataset_conf.splits
    if train:
        split = splits["train"]
    else:
        # For evaluation, prefer validation split if it exists, otherwise use test
        if evaluate and "validation" in splits:
            split = splits["validation"]
        elif "test" in splits:
            split = splits["test"]
        else:
            logger.warning(
                f"Dataset {data_conf.dataset} has no validation/test split, using train split"
            )
            split = splits["train"]

    # Load dataset
    dataset_config = dataset_conf.hf_config
    load_kwargs = {
        "cache_dir": str(data_conf.path),
        "split": split,
        "name": dataset_config,
    }

    dataset = load_dataset(dataset_conf.hf_path, **load_kwargs)

    # Debug: print first example
    logger.debug(f"First example from dataset: {dataset[0]}")
    logger.debug(f"Dataset features: {dataset.features}")

    # Get appropriate preprocessor
    preprocessor = get_task_preprocessor(task_type, tokenizer, data_conf, evaluate)

    # Transform dataset
    tokenized_dataset = dataset.map(
        preprocessor, batched=True, remove_columns=dataset.column_names
    )

    # Setup data collator based on task
    if task_type == TaskType.GENERATION and data_conf.whole_word_masking:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=data_conf.masking_probability
        )
    else:
        data_collator = DataCollatorWithPadding(tokenizer)

    batch_size = data_conf.test_batch_size if evaluate else data_conf.batch_size
    loader_kwargs = dict()
    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=train and not evaluate,
        num_workers=data_conf.num_workers,
        collate_fn=data_collator,
        generator=g if loader_seed is not None else None,
        worker_init_fn=seed_worker,
        pin_memory=True,
        prefetch_factor=2 if data_conf.num_workers > 0 else None,
        persistent_workers=True if data_conf.num_workers > 0 else False,
    )


def setup_vision_loader(
    data_conf: DataConfig, train: bool, evaluate: bool, loader_seed: int = None
) -> DataLoader:
    dataset = data_conf.dataset
    dataset_conf = data_conf.dataset_info
    w = dataset_conf.resolution
    mean, std = list(dataset_conf.mean), list(dataset_conf.std)
    transforms_ = [transforms.Resize((w, w))]
    if not evaluate:
        if data_conf.hflip:
            transforms_.append(transforms.RandomHorizontalFlip())
        if rot := data_conf.random_rotation:
            transforms_.append(transforms.RandomRotation(rot))
        if data_conf.random_translate:
            t = data_conf.random_translate / w
            transforms_.append(
                transforms.RandomAffine(degrees=0, translate=(t, t), fill=mean)
            )
        if data_conf.gaussian_blur:
            transforms_.append(transforms.GaussianBlur(kernel_size=3))
    # put ToTensor here because RandomErasing needs it (can't move it earlier as that would break regression tests due to reordering transforms)
    transforms_.append(transforms.ToTensor())
    if not evaluate:
        if data_conf.cutout:
            scale = data_conf.cutout / w
            transforms_.append(
                transforms.RandomErasing(
                    p=0.5, scale=(0.02, scale), ratio=(0.3, 3.3), value=mean
                )
            )
        # TODO: do the cutmix/mixup
    transforms_.append(transforms.Normalize(mean, std))
    transforms_ = transforms.Compose(transforms_)
    dataset_cls = dataset_conf.torch_dataset
    if isinstance(dataset_cls, str):
        func = dataset_cls.split(".")[-1]
        module = ".".join(dataset_cls.split(".")[:-1])
        dataset_cls = getattr(importlib.import_module(module), func)

    dataset = dataset_cls(
        root=data_conf.path,
        train=train,
        transform=transforms_,
        download=data_conf.download,
    )

    batch_size = data_conf.batch_size if not evaluate else data_conf.test_batch_size

    g = torch.Generator()
    if loader_seed is not None:
        g.manual_seed(loader_seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=data_conf.num_workers,
        generator=g,
        worker_init_fn=seed_worker,
        pin_memory=True,
        prefetch_factor=2,
    )
    return loader


def setup_loader(
    data_conf: DataConfig,
    train: bool,
    evaluate: bool,
    loader_seed: int = None,
    tokenizer: AutoTokenizer = None,
) -> DataLoader:
    if data_conf.is_language_dataset():
        return setup_nlp_loader(
            data_conf, train, evaluate, tokenizer=tokenizer, loader_seed=loader_seed
        )
    return setup_vision_loader(data_conf, train, evaluate, loader_seed=loader_seed)


def setup_model_dir(config: Trainer) -> Path:
    """
    Set up the model directory for saving training artifacts.

    Raises:
        FileNotFoundError: If the specified `log_dir` does not exist.
    """
    if not config.logger.log_dir.exists():
        if config.logger.log_dir.parent.exists():
            config.logger.log_dir.mkdir()
        else:
            raise FileNotFoundError(
                f"Must provide an existing log_dir ({config.logger.log_dir})"
            )
    if config.model_dir is None:
        hashname = config.hashname[:5]
        now = datetime.now()
        formatted_date = now.strftime("%y-%m-%d")
        short_id = str(now.microsecond)[:5]
        config.model_dir = Path(
            config.logger.log_dir, f"{hashname}-{formatted_date}-{short_id}"
        )
        logger.info(f"Created model dir: {config.model_dir}")
    elif Path(config.model_dir).exists() and config.logger.enforce_new_model_dir:
        now = datetime.now()
        short_id = str(now.microsecond)[-4:]
        config.model_dir = Path(str(config.model_dir) + "_" + short_id)
        logger.info(
            f"enforce_new_model_dir True, creating a new model dir: {config.model_dir}"
        )
    config.model_dir = Path(config.model_dir)
    config.model_dir.mkdir(exist_ok=True)

    # Save code as zip and config as yaml into the model directory.
    config.save(config.model_dir, zip_code_base=config.zip_and_save_source)

    return config.model_dir


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_device(config: Experiment) -> torch.device:
    """
    Configure and initialize the computing device for PyTorch operations.
    """
    if config.deterministic:
        # set env variable for use_deterministic_algorithms
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = get_device()
    logger.info(f"Using device: {device}.")
    return device


def setup_wandb(config: Experiment) -> None:
    """given the configuration, sets up the wandb project"""
    if config.logger.use_wandb:
        conf_dct = config.wandb_dct()
        if config.log_to_same_experiment and config.resume_from:
            if config.resume_from.startswith("wandb:"):
                wandb_url = config.resume_from[6:]
            else:
                wandb_url = config.model_dir.joinpath("wandb.txt").read_text()
            url_parts = wandb_url.split("/")
            entity = url_parts[-3]
            project = url_parts[-2]
            run_id = url_parts[-1]
            wandb_kwargs = dict(entity=entity, project=project, dir=WANDB_DIR)
            wandb_kwargs["resume"] = "allow"
            wandb_kwargs["run_id"] = run_id
            run = wandb.init(**wandb_kwargs)
        else:
            run = wandb.init(
                name=config.logger.run_name,
                entity=config.logger.entity,
                group=config.logger.group,
                tags=config.logger.tags,
                project=config.logger.project,
                notes=config.logger.notes,
                config=conf_dct,
                dir=WANDB_DIR,
                id=config.logger.run_id,
                mode="offline" if config.logger.wandb_offline else "online",
            )
        if config.model_dir is not None:
            if run.url is None:
                url = f"https://wandb.ai/{run.entity}/{run.project}/runs/{run.id}"
            else:
                url = run.url
            Path(config.model_dir).joinpath("wandb.txt").write_text(url)


def get_ckpts_by_step(ckpt_dir: Path, steps_per_epoch: int) -> OrderedDict[int, Path]:
    ckpts = []
    for ckpt in ckpt_dir.glob("*.ckpt"):
        if ckpt.stem != "best":
            step = Step.from_short_string(ckpt.stem, steps_per_epoch).get_step()
            ckpts.append((step, ckpt))
    sorted_ckpt_dict = OrderedDict()
    for k, v in sorted(ckpts, key=lambda x: x[0]):
        sorted_ckpt_dict[k] = v
    return sorted_ckpt_dict


def get_last_ckpt(ckpt_dir: Path, steps_per_epoch: int) -> Path:
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt
    ckpts = get_ckpts_by_step(ckpt_dir, steps_per_epoch)
    last_step = list(ckpts.keys())[-1]
    return ckpts[last_step]


def get_resume_state_dicts(
    config, i: int=1
) -> Tuple[int, int, Dict, Dict, Dict]:
    """logic:
    if only ckpt_path is set, load the model and nothing else
    if resume_from is set, load the epoch, step, and optimizer states from the checkpoint
        the checkpoint is inferred from resume_from, resume_step, and i (model index)
    if both ckpt_path and resume_from are set, ckpt_path is used to initialize everything
    """
    device = get_device()
    model_sd = {}
    has_ckpt_path = hasattr(config.model, "ckpt_path") and (config.model.ckpt_path)
    has_resume_from = hasattr(config, "resume_from") and (config.resume_from)

    # don't load anything
    if not has_ckpt_path and not has_resume_from:
        return 0, 0, {}, {}, {}

    # load model only
    if has_ckpt_path and not has_resume_from:
        ckpt_path = Path(config.model.ckpt_path)
        if not ckpt_path.exists():
            raise ValueError(f"resume_from location does not exist: {ckpt_path}")
        model_sd = torch.load(ckpt_path, map_location=device)["state_dict"]
        logger.info(f"Model loaded from checkpoint {ckpt_path}")
        return 0, 0, model_sd, {}, {}

    if config.resume_from.startswith("wandb:"):
        raise NotImplementedError("Resuming from wandb is under development")

    # load everything from has_resume_from
    if not has_ckpt_path and has_resume_from:
        steps_per_epoch = config.data.get_steps_per_epoch()
        ckpt_dir = Path(config.resume_from) / f"model{i}" / "checkpoints"
        if not ckpt_dir.exists():
            raise ValueError(f"resume_from location does not exist: {ckpt_dir}")
        # resume_step is not set, or -1 or -1st means get last saved checkpoint
        if (
            not (hasattr(config, "resume_step") and (config.resume_step))
            or str(config.resume_step) == "-1"
            or str(config.resume_step) == "-1st"
        ):
            ckpt_file = get_last_ckpt(ckpt_dir, steps_per_epoch)
            logger.info(
                "resume_step is not set or is -1, resuming from last ckpt %s",
                ckpt_file,
            )
        # find checkpoint at resume_step
        else:
            step = Step.from_short_string(config.resume_step, steps_per_epoch).get_step()
            ckpts = get_ckpts_by_step(ckpt_dir, steps_per_epoch)
            if step not in ckpts:
                raise ValueError(
                    f"Could not find step {config.resume_step} in {ckpt_dir}"
                )
            ckpt_file = ckpts[step]

    # load everything from ckpt_path
    elif has_ckpt_path and has_resume_from:
        ckpt_file = config.model.ckpt_path

    # load state dicts
    ckpt_dict = torch.load(ckpt_file, map_location=device)

    epoch = ckpt_dict["epoch"]
    step = ckpt_dict["step"]
    model_sd = ckpt_dict["state_dict"]
    opt_sd = ckpt_dict["optimizer_state_dict"]
    schedule_sd = ckpt_dict["scheduler_state_dict"]
    logger.info(f"Model, optimizer, lr schedule loaded from {ckpt_file}, {epoch}ep {step}st.")

    return epoch, step, model_sd, opt_sd, schedule_sd


def setup_experiment(config: Trainer) -> Tuple[TrainingElements, torch.device, int]:
    config.logger.slurm_job_id = os.environ.get("SLURM_JOB_ID")
    """Creates all necessary elements. models, datamodules, etc."""
    device = setup_device(config)
    model_dir = setup_model_dir(config)
    steps_per_epoch = config.data.get_steps_per_epoch()
    # config.model_dir.joinpath("barriers").mkdir(exist_ok=True)

    setup_wandb(config)
    training_elements = TrainingElements()
    assert config.n_models >= 1, f"n_models ({config.n_models}) should be >= 1."
    first_seed = config.seeds.seed1
    for i in range(1, config.n_models + 1):
        suffix = str(i)
        seed = getattr(config.seeds, f"seed{suffix}")
        loader_seed = getattr(config.seeds, f"loader_seed{suffix}")
        perturb_seed = seed
        if hasattr(config, f"perturb_seed{suffix}"):
            perturb_seed = getattr(config.seeds, f"perturb_seed{suffix}")

        ## setup individual model dir
        model_dir_ = model_dir.joinpath(f"model{i}")
        setattr(config, f"model{i}_dir", model_dir_)
        model_dir_.joinpath("checkpoints").mkdir(exist_ok=True, parents=True)

        # Determine if using NLP or vision setup
        is_nlp_task = config.data.is_language_dataset()
        seed_everything(seed)

        # load state dicts if resuming
        epoch, start_step, model_sd, opt_sd, schedule_sd = get_resume_state_dicts(
            config, i
        )

        # model
        model, tokenizer = configure_model(
            config, device, seed=seed, state_dict=model_sd
        )
        #TODO init_model_vector is from current step, not 0, when resume_from starts at nonzero step
        init_model_vector = (
            nn.utils.parameters_to_vector(model.parameters()).detach().cpu()
        )
        if hasattr(config, "frozen_layers"):
            freeze_layers(model, config.frozen_layers)
        logger.info("Setup model %d with seed=%d.", i, seed)

        # data
        def get_train_eval_test_loaders(loader_fn, **kwargs):
            train_loader = loader_fn(train=True, evaluate=False, **kwargs)
            eval_loader = loader_fn(train=True, evaluate=True, **kwargs)
            test_loader = loader_fn(train=False, evaluate=True, **kwargs)
            return train_loader, eval_loader, test_loader

        if is_nlp_task:
            train_loader, train_eval_loader, test_loader = get_train_eval_test_loaders(
                setup_nlp_loader,
                data_conf=config.data,
                tokenizer=tokenizer,
                loader_seed=loader_seed,
            )
        else:
            train_loader, train_eval_loader, test_loader = get_train_eval_test_loaders(
                setup_vision_loader, data_conf=config.data, loader_seed=loader_seed
            )
        assert steps_per_epoch == len(train_loader)
        logger.info("Setup dataloaders of %d with loader_seed=%d", i, loader_seed)

        # optimizer
        seed_everything(seed)
        opt = configure_optimizer(config, model, state_dict=opt_sd)
        max_steps = config.trainer.training_steps.get_step(steps_per_epoch)
        if (ga := config.trainer.gradient_accumulation_steps) > 1:
            grad_steps = int(max_steps / ga)
            logger.info(
                "Gradient accumulation steps %d, modifying lr scheduler accordingly.",
                ga,
            )
        else:
            grad_steps = max_steps
        # scheduler
        scheduler = configure_lr_scheduler(
            opt,
            grad_steps,
            config.trainer.opt.lr_scheduler,
            config.trainer.opt.warmup_ratio,
            {},
            state_dict=schedule_sd,
        )
        if config.logger.print_optimizers:
            logger.info("Optimizer %d - %s", i, opt)
            logger.info("Scheduler %d - %s", i, scheduler)

        training_elements.add_element(
            TrainingElement(
                element_ind=i,
                model=model,
                opt=opt,
                scheduler=scheduler,
                model_dir=model_dir_,
                seed=seed,
                loader_seed=loader_seed,
                train_loader=train_loader,
                train_eval_loader=train_eval_loader,
                test_loader=test_loader,
                max_steps=Step(max_steps, steps_per_epoch),
                perturb_seed=perturb_seed,
                tokenizer=tokenizer,
                init_model_vector=init_model_vector,
            )
        )
    seed_everything(first_seed)
    if config.logger.use_tqdm:
        setup_iterators(training_elements, use_tqdm=True)
    logger.info(
        "Model setups complete, switching seed to %d, which is passed to the first model.",
        first_seed,
    )

    return training_elements, device, start_step


def cleanup(config: Experiment):
    """if script is called with cleanup_after, deletes the model_dir and all checkpoints created"""
    if config.logger.cleanup_after:
        if config.hashname not in config.model_dir.as_posix():
            logger.info(
                f"Not sure if the model_dir is an outer directory, cleanup manually ({config.model_dir})"
            )
            return

        logger.info("Deleting the experiment directory (%s)", config.model_dir)
        rmtree(config.model_dir)


# something wrong with the steps
def save_training(
    el: TrainingElement, path: Path, epoch: int = None, step: int = None
) -> None:
    """Given a training element, saves the model state, optimizer and scheduler state along with epoch."""
    torch.save(
        {
            "state_dict": el.model.state_dict(),
            "optimizer_state_dict": el.opt.state_dict(),
            "scheduler_state_dict": el.scheduler.state_dict()
            if el.scheduler is not None
            else None,
            "epoch": epoch,
            "step": step,
        },
        path,
        pickle_protocol=4,
    )


COLORS = ["#75507b", "#4f42b5", "#808080"]


def setup_iterators(
    training_elements: Dict[int, TrainingElement], use_tqdm: bool = True
):
    tqdm_cls = tqdm if use_tqdm else Iterator
    for i, el in enumerate(training_elements):
        color = COLORS[i % len(COLORS)]
        train_iterator = tqdm_cls(
            total=len(el.train_loader),
            desc=f"Training model {i} - epoch: ",
            position=2 * i,
            leave=True,
            # leave=False, disable=None,
            colour=color,
        )
        train_eval_iterator = tqdm_cls(
            total=len(el.train_loader),
            desc=f"Evaluating model {i} on train - epoch: ",
            position=2 + 2 * i,
            leave=True,
            # leave=False, disable=None,
            colour=color,
        )
        test_iterator = tqdm_cls(
            total=len(el.test_loader),
            desc=f"Evaluating model {i} - epoch: ",
            position=1 + 2 * i,
            leave=True,
            # leave=False, disable=None,
            colour=color,
        )
        el.extra_iterator = tqdm_cls(
            position=2 + 2 * i, desc="Extra iterator used for anything", colour="white"
        )
        el.train_iterator = train_iterator
        el.test_iterator = test_iterator
        el.train_eval_iterator = train_eval_iterator
        el.train_iterator = train_iterator
        el.test_iterator = test_iterator
        el.train_eval_iterator = train_eval_iterator
