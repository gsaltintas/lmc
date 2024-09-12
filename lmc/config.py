import argparse
import logging
import os
import re
from copy import deepcopy
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from pprint import pformat
from typing import (Dict, List, Literal, Optional, Tuple, Type, Union,
                    get_args, get_origin)

from rich.console import Console
from rich.table import Table

from lmc.models.type_declaration import (MODEL_NAME_PATTERNS, Activations,
                                         Inits, Norms)
from lmc.utils.utils import match_pattern

""" TODO: maybe omit defaults in the future versions """
logger = logging.getLogger("config")

def parse_bool(value):
    if value.lower() in ("true", "yes", "y") or value == "1":
        return True
    elif value.lower() in ("false", "no", "n") or value == "0":
        return False
    else:
        raise ValueError('Invalid input for bool config "%s"' % (value))


def maybe_get_arg(arg_name, positional=False, position=0, boolean_arg=False):
    parser = argparse.ArgumentParser(add_help=True)
    prefix = "" if positional else "--"
    if positional:
        for i in range(position):
            parser.add_argument(f"arg{i}")
    if boolean_arg:
        parser.add_argument(prefix + arg_name, action="store_true")
    else:
        parser.add_argument(prefix + arg_name, type=str, default=None)
    try:
        args = parser.parse_known_args()[0]
        return getattr(args, arg_name) if arg_name in args else None
    except:
        return None


def field_factory(
    maybe_arg: str, mapping: Dict[str, "Config"], default_val: str = None, from_args: bool = True
):
    # TODO: add here default factory from file/dict inits
    """
    Creates a factory function that returns a configuration object based on the provided argument.

    Args:
        maybe_arg (str): The name of the argument to be checked. This should be a key or a substring
                         that can be used to look up the correct configuration in the `mapping`.
        mapping (Dict[str, 'Config']): A dictionary mapping string keys to configuration classes.
                                       The key should match or be a substring of the provided argument.
        default_val (str, optional): A default value to use if the argument is not provided or is `None`.
                                     Defaults to `None`.

    Returns:
        Callable[[], 'Config']: A factory function that, when called, returns an instance of the
                                appropriate configuration class from the `mapping`.

    Raises:
        ValueError: If no matching configuration is found in the `mapping` for the provided argument.
    """

    def arg_fact():
        value = maybe_get_arg(maybe_arg)
        value = default_val if value is None else value
        config = None
        for key, conf in mapping.items():
            if key in value.lower():
                config = conf
        if config is None:
            raise ValueError(f"Not a valid name: for {maybe_arg} ({value})")

        return config
        
    return arg_fact


def add_basic_args(
    parser: argparse.ArgumentParser,
    cls: Type,
    defaults: "Config" = None,
    prefix: str = None,
    name: str = None,
    description: str = None,
    create_group: bool = True,
):
    """
    Add basic type arguments to the provided parser based on the fields of a dataclass.

    Args:
        parser (argparse.ArgumentParser): The argument parser to which arguments will be added.
        cls (Type): The dataclass type whose fields will be used to add arguments.
        defaults (Config, optional): An instance of the dataclass to use as defaults.
        prefix (str, optional): Prefix to add to argument names.
        name (str, optional): Name of the argument group.
        description (str, optional): Description for the argument group.
        create_group (bool, optional): Whether to create a new argument group. Default is True.
    """
    parser_ = parser
    if create_group:
        parser = parser.add_argument_group(
            name or cls._name, description or cls._description
        )

    for field_ in fields(cls):
        if field_.name.startswith("_"):
            continue
        typ = field_.type

        if get_origin(typ) is Union:
            if type(None) in get_args(typ):
                typ = get_args(typ)[0]
            else:
                typ = field_.default_factory()
        elif typ is USE_DEFAULT_FACTORY:
            typ = field_.default_factory()

        arg_name = f"--{field_.name}" if prefix is None else f"--{prefix}_{field_.name}"

        default = getattr(defaults, field_.name, None) if defaults else field_.default
        required = field_.default is MISSING and (not defaults or default is None)
        helptext = getattr(cls, f"_{field_.name}", "")
        if required:
            helptext = "(required: %(type)s) " + helptext
        elif default is not None:
            helptext = f"(default: {default}) " + helptext
        else:
            helptext = "(optional: %(type)s) " + helptext
        kwargs = dict(default=default, required=required, help=helptext)

        if get_origin(typ) is Literal:
            choices = get_args(typ)
            parser.add_argument(
                arg_name, type=type(choices[0]), choices=choices, **kwargs
            )
        elif typ is bool:

            def str2bool(v):
                if isinstance(v, bool):
                    return v
                if v.lower() in ("yes", "true", "t", "y", "1"):
                    return True
                elif v.lower() in ("no", "false", "f", "n", "0"):
                    return False
                else:
                    raise argparse.ArgumentTypeError("Boolean value expected.")

            parser.add_argument(arg_name, type=str2bool, **kwargs)
        elif typ in [str, float, int, Path]:
            parser.add_argument(arg_name, type=typ, **kwargs)
        elif isinstance(get_origin(typ), type) and issubclass(get_origin(typ), list):
            parser.add_argument(arg_name, nargs="*", **kwargs)
        elif isinstance(get_origin(typ), type) and issubclass(get_origin(typ), tuple):
            parser.add_argument(arg_name, nargs=len(get_args(typ)), **kwargs)
        elif isinstance(typ, type) and issubclass(typ, Config):
            add_basic_args(
                parser_,
                typ,
                defaults=defaults,
                prefix=(
                    typ._name
                    if hasattr(typ, "_add_prefix") and typ._add_prefix
                    else None
                ),
                create_group=create_group,
            )
        else:
            parser_.error(f"Invalid field type {typ} for hparams.")
    parser = parser_

@dataclass
class Config:
    """ base config class """
    _add_prefix: bool = field(init=False, default=False)  # Class-level default, set to true to parse arguments as --name.arg

    def __post_init__(self):
        if not hasattr(self, "_name"):
            raise ValueError(f"Must have field _name with string value.{self}")
        if not hasattr(self, "_description"):
            raise ValueError(f"Must have field _description with string value. {self}")

    def validate(self, strict: bool = True) -> None:
        props: List[str] = fields(self)
        for prop in props:
            if hasattr(self, f"_{prop}"):
                arg = getattr(self, f"_{prop}")
                val = getattr(self, prop)
                if arg.validator and not arg.validator(val):
                    if strict:
                        raise ValueError(
                            f"Argument doesn't comply with the specification: {arg.help_str()}"
                        )
                    else:
                        logger.warn(
                            f"Argument doesn't comply with the specification: {arg.help_str()}"
                        )

    @classmethod
    def add_args(
        cls,
        parser: argparse.ArgumentParser,
        defaults: "Config" = None,
        prefix: str = None,
        name: str = None,
        description: str = None,
        create_group: bool = True,
    ):
        add_basic_args(parser, cls, defaults, prefix, name, description, create_group)

    @classmethod
    def create_from_args(cls, args: argparse.Namespace, prefix: str = None) -> "Config":
        d = {}
        for _field in fields(cls):
            if _field.name.startswith("_"):
                continue
            typ = _field.type
            subprefix = prefix

            if get_origin(typ) is Union:
                if type(None) in get_args(typ):
                    typ = get_origin(get_args(typ)[0])
                else:
                    typ = _field.default_factory()
                    # subprefix = typ._name
            arg_name = (
                f"{_field.name}" if subprefix is None else f"{subprefix}_{_field.name}"
            )
            d[_field.name] = arg_name
            # Base types.
            if typ in [bool, str, float, int, Path] or (
                isinstance(typ, type) and issubclass(typ, list)
            ):
                if not hasattr(args, arg_name):
                    raise ValueError(f"Missing argument: {arg_name}.")
                d[_field.name] = getattr(args, arg_name)
            # Nested hparams.
            elif isinstance(typ, type) and issubclass(typ, Config):
                subprefix = typ._name if typ._add_prefix else None
                d[_field.name] = typ.create_from_args(args, subprefix)
            else:
                d[_field.name] = getattr(args, arg_name)

        return cls(**d)

    @property
    def display(self, only_modified: bool = True):
        console = Console()

        # Create a rich Table
        table = Table(title=self.__class__.__name__)

        # Add columns for the table
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        # Add rows for each non-default field
        for f in fields(self):
            if not f.name.startswith("_"):
                def_val = f.default
                value = getattr(self, f.name)
                if value == def_val and only_modified:
                    continue
                table.add_row(f.name, str(value))

        # Capture the table as a string using Console's capture method
        with console.capture() as capture:
            console.print(table)
        return capture.get()

    def __str__(self):
        fs = {}
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            if f.default is MISSING or (getattr(self, f.name) != f.default):
                value = getattr(self, f.name)
                if isinstance(value, str):
                    value = "'" + value + "'"
                if isinstance(value, Config):
                    value = str(value)
                if isinstance(value, Tuple):
                    value = "Tuple(" + ",".join(str(h) for h in value) + ")"
                fs[f.name] = value
        elements = [f"{name}={fs[name]}" for name in sorted(fs.keys())]
        return "Config(" + ", ".join(elements) + ")"

    def wandb_dct(self):
        d = dict()
        for field_ in fields(self):
            if field_.name.startswith("_"):
                continue
            val = getattr(self, field_.name)
            if isinstance(field_.type, type) and issubclass(field_.type, Config):
                val = val.wandb_dct()
            d[field_.name] = val
        return d
    
USE_DEFAULT_FACTORY = object()


@dataclass
class LrSchedule(Config):
    pass


@dataclass
class Optimizer(Config):
    lr: float
    optimizer: Literal["sgd", "adam", "adamw"]
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    gradient_clip_val: float = 0.0
    gradient_clip_algorithm: Literal["norm", "value", ""] = None
    lr_scheduler: Literal["linear", "exponential", "cosine", "triangle"] = None

    _lr = "Learning rate"
    _warmup_ratio = r"\rho * training_steps will be used for learning rate warmup"
    _gradient_clip_val = ""
    _gradient_clip_algorithm = ""
    _name = "optimizer"
    _description = "Configuration pertaining to optimizer hyper-parameters."


@dataclass
class _SGDConfig(Optimizer):
    momentum: float = 0.9
    _name = "sgd"
    _description = ""


@dataclass
class _AdamConfig(Optimizer):
    _name = "adam"
    _description = ""

    betas: Tuple[float, float] = (0.9, 0.999)


@dataclass
class TrainerConfig(Config):
    training_steps: str
    opt: Union[_SGDConfig, _AdamConfig] = field(
        init=True,
        default_factory=field_factory(
            maybe_arg="optimizer",
            mapping={"sgd": _SGDConfig, "adam": _AdamConfig, "adamw": _AdamConfig},
            default_val="sgd",
        ),
    )

    save_freq: str = "1ep"
    save_early_iters: bool = False
    save_best: bool = True
    use_scaler: bool = False
    label_smoothing: float = 0.0

    _training_steps = "Total number of steps (st) or epochs (ep), specify as Xep or Xst"
    _save_freq = "Frequency to save checkpoints during training in steps (st) or epochs (ep), specify as Xep or Xst"
    _save_best = "Pass to save the best model"
    _use_scaler = "Pass to use torch gradient scaling"

    _name = "trainer"
    _description = "Training hyper-parameters"

    @classmethod
    def create_from_args(cls, args: argparse.Namespace, prefix: str = None) -> Config:
        if args.training_steps.isnumeric():
            args.training_steps = f"{args.training_steps}st"
        if args.save_freq.isnumeric():
            args.save_freq = f"{args.save_freq}st"
        if not args.training_steps.endswith("ep") and not args.training_steps.endswith(
            "st"
        ):
            raise ValueError("Please specify training steps as either X | Xst or Xep.")
        if not args.save_freq.endswith("ep") and not args.save_freq.endswith(
            "st"
        ):
            raise ValueError("Please specify save frequency as either X | Xst or Xep.")
        return super().create_from_args(args, prefix)


@dataclass
class DataConfig(Config):
    dataset: Literal["cifar10", "mnist", "cifar100", "tiny-imagenet"]

    batch_size: int = 128
    test_batch_size: int = 1024
    path: Path = Path("./data")

    hflip: bool = True
    mixup: float = 0.0
    cutmix: float = 0.0
    gaussian_blur: bool = False
    random_rotation: float = 0.0
    random_crop: bool = False
    cutout: Optional[int] = None
    num_workers: int = 4

    _hflip: str = "Pass true to perform random horizontal flip with probability 0.5."
    _name = "data"
    _description = "Data and augmentations configuration"

    def __post_init__(self):
        self.path = Path(self.path).resolve().absolute()
        return super().__post_init__()


@dataclass
class LoggerConfig(Config):
    _name = "logger"
    _description = "Logger configuration"

    use_wandb: bool = False
    use_tqdm: bool = True
    print_summary: bool = True
    print_optimizers: bool = True

    project: str = None
    entity: str = None
    run_name: str = None
    group: str = None
    tags: Optional[List[str]] = None
    notes: str = None
    # resume_id: str = None
    experiment_id: str = None
    slurm_job_id: str = None
    log_dir: Path = Path("../experiments")
    cleanup_after: bool = False
    level: Literal["debug", "info", "warning", "error", "critical"] = "info"

    _cleanup_after: str = (
        "Pass true only if you want to delete the experiment directory after experiment is finished."
    )

    def __post_init__(self):
        s = self.slurm_job_id
        current_slurm_id = os.environ.get("SLURM_JOB_ID", None)
        if s:
            s = [s, current_slurm_id]
        else:
            s = current_slurm_id
        self.log_dir = Path(self.log_dir).resolve().absolute()
        return super().__post_init__()


@dataclass
class Model:
    def __str__(self):
        s = "("
        for f in fields(self):
            s += f.name + "=" + f.value
        return s + ")"


@dataclass
class ResNetConfig(Config):
    _name = "resnet"
    _description = ""
    _add_prefix = True

    width: int = 16
    depth: Literal[9, 18, 20, 50] = 20

    _width: str = "Output channels of the first convolution"


@dataclass
class MLPConfig(Config):
    _name = "mlp"
    _description = ""
    _add_prefix = True

    width: int = 1024
    num_hidden_layers: int = 4

    _num_hidden_layers: str = "Number of hidden layers, depth: (num_hidden_layers + 1)."


@dataclass
class ModelConfig(Config):
    _name = "model"
    _description = ""

    model_name: str
    model: Union[MLPConfig, ResNetConfig] = field(
        init=True,
        default_factory=field_factory(
            "model_name",
            mapping={"mlp": MLPConfig, "resnet": ResNetConfig},
            default_val="mlp",
        ),
    )
    norm: Norms = None
    act: Activations = "relu"
    initialization_strategy: Inits = "kaiming_normal"
    ckpt_path: Optional[Path] = None

    _model_name = "Name of the model e.g. mlp, resnet. Could also be model code resnet20-64, etc. Pass model name to see aditional arguments related to models"
    _initialization_strategy = "Initialization strategy for the model's layers"

    def __post_init__(self):
        correct_name = match_pattern(self.model_name, MODEL_NAME_PATTERNS)
        if not correct_name:
            raise ValueError(
                f"Model name not properly configured, try one of {pformat(MODEL_NAME_PATTERNS)}"
            )
        return super().__post_init__()


@dataclass
class SysConfig(Config):
    pass
