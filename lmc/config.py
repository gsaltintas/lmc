import argparse
import logging
import re
from copy import deepcopy
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from pprint import pformat
from typing import (Dict, List, Literal, Optional, Tuple, Union, get_args,
                    get_origin)

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

MODEL_NAME_PATTERNS = [
    r"vit[a-zA-Z]*/(\d+)-(\d+)-(\d+)x(\d+)-(\d+)",
    r"cnn[a-zA-Z]*/(\d+)xk=(\d+)-s=(\d+)-d=(\d+)-fc=(\d+)-p=(\d+)",
    r"cnn[a-zA-Z]*/(\d+)xk=(\d+)-s=(\d+)-d=(\d+)-fc=(\d+)",
    r"lcn[a-zA-Z]*/(\d+)xk=(\d+)-s=(\d+)-d=(\d+)-fc=(\d+)-p=(\d+)",
    r"lcn[a-zA-Z]*/(\d+)xk=(\d+)-s=(\d+)-d=(\d+)-fc=(\d+)",
    r"vgg[a-zA-Z]*(\d+)",
    r"resnet[a-zA-Z]*(\d+)-(\d+)",
    r"wideresnet[a-zA-Z]*/(\d+)xk=(\d+)-s=(\d+)-d=(\d+)-fc=(\d+)",
    r"ffcv-resnet[a-zA-Z]*-(\d+)" r"linear/(\d+)x(\d+)",
    r"mlp/(\d+)x(\d+)",
    r"mlp",
    r"resnet",
    r"simple-resnet",
]
FORMAT = "%(name)s - %(levelname)s: %(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

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
    maybe_arg: str, mapping: Dict[str, "Config"], default_val: str = None
):
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

    def _fact():
        value = maybe_get_arg(maybe_arg)
        value = default_val if value is None else value
        config = None
        for key, conf in mapping.items():
            if key in value.lower():
                config = conf
        if config is None:
            raise ValueError(f"Not a valid name: for {maybe_arg} ({value})")

        return config

    return _fact


@dataclass
class Config:
    _add_prefix: bool = field(init=False, default=False)  # Class-level default

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
        # if defaults and not isinstance(defaults, cls):
        #     raise ValueError(f"defaults must also be type {cls}.")

        if create_group:
            parser = parser.add_argument_group(
                name or cls._name, description or cls._description
            )

        for field_ in fields(cls):
            prefix_ = prefix
            if field_.name.startswith("_"):
                continue

            typ = field_.type
            subprefix = f"{prefix}_{field_.name}" if prefix else field_.name

            if get_origin(typ) is Union:
                if type(None) in get_args(typ):
                    typ = get_args(typ)[0]
                else:
                    typ = field_.default_factory()

            arg_name = (
                f"--{field_.name}" if prefix is None else f"--{prefix}_{field_.name}"
            )

            if defaults:
                default = deepcopy(getattr(defaults, field_.name, None))
            elif field_.default != MISSING:
                default = deepcopy(field_.default)
            else:
                default = None
            required = field_.default is MISSING and (
                not defaults or not getattr(defaults, field_.name)
            )

            def help_text(f):
                helptext = (
                    getattr(cls, f"_{f.name}") if hasattr(cls, f"_{f.name}") else ""
                )
                if required:
                    helptext = "(required: %(type)s) " + helptext
                elif default:
                    helptext = f"(default: {default}) " + helptext
                else:
                    helptext = "(optional: %(type)s) " + helptext
                return helptext

            helptext = help_text(field_)
            if get_origin(typ) is Literal:
                choices = get_args(typ)
                parser.add_argument(
                    arg_name,
                    type=type(choices[0]),
                    default=default,
                    required=required,
                    help=helptext,
                    choices=choices,
                )
            # todo: handle lists
            elif typ in [bool, str, float, int, Path]:
                parser.add_argument(
                    arg_name,
                    type=typ,
                    default=default,
                    required=required,
                    help=helptext,
                )
            elif isinstance(get_origin(typ), type) and issubclass(
                get_origin(typ), list
            ):
                parser.add_argument(
                    arg_name,
                    default=default,
                    required=required,
                    help=helptext,
                    nargs="*",
                )
            # # If it is a nested hparams, use the field name as the prefix and add all arguments.
            elif isinstance(typ, type) and issubclass(typ, Config):
                prefix = typ._name if typ._add_prefix else None
                typ.add_args(
                    parser,
                    defaults=default,
                    prefix=prefix,
                    create_group=False,
                    # parser, defaults=default, prefix=subprefix, create_group=False
                )

            else:
                logger.error(f"Invalid field type {typ} for hparams.")
            prefix = prefix_

    @classmethod
    def create_from_args(cls, args: argparse.Namespace, prefix: str = None) -> "Config":
        d = {}
        for _field in fields(cls):
            if _field.name.startswith("_"):
                continue
            typ = _field.type
            subprefix = prefix
            # subprefix = f"{prefix}_{_field.name}" if prefix else _field.name

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
    def display(self):
        console = Console()

        # Create a rich Table
        table = Table(title=self.__class__.__name__)

        # Add columns for the table
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        # Add rows for each non-default field
        for f in fields(self):
            if not f.name.startswith("_"):
                value = getattr(self, f.name)
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


@dataclass
class _SGDConfig(Config):
    momentum: float = 0.9
    _name = "sgd"
    _description = ""


@dataclass
class _AdamConfig(Config):
    # momentu1m: float = 0.9
    _name = "adam"
    _description = ""


@dataclass
class Optimizer(Config):
    lr: float
    optimizer: str
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    optimizer_: Union[_SGDConfig, _AdamConfig] = field(
        init=True,
        default_factory=field_factory(
            maybe_arg="optimizer",
            mapping={"sgd": _SGDConfig, "adam": _AdamConfig},
            default_val="sgd",
        ),
    )
    gradient_clip_val: float = 0.0
    gradient_clip_algorithm: Literal["norm", "value", ""] = None

    _lr = "Learning rate"
    _warmup_ratio = r"\rho * training_steps will be used for learning rate warmup"
    _gradient_clip_val = ""
    _gradient_clip_algorithm = ""
    _name = "optimizer"
    _description = "Configuration pertaining to optimizer hyper-parameters."


@dataclass
class TrainerConfig(Config):
    training_steps: str
    opt: Optimizer

    save_freq: int = 1
    save_best: bool = True
    use_scaler: bool = False

    _training_steps = "Total number of steps (st) or epochs (ep), specify as Xep or Xst"
    _save_freq = "Frequency to save checkpoints during training"
    _save_best = "Pass to save the best model"
    _use_scaler = "Pass to use torch gradient scaling"

    _name = "trainer"
    _description = "Training hyper-parameters"

    @classmethod
    def create_from_args(cls, args: argparse.Namespace, prefix: str = None) -> Config:
        if args.training_steps.isnumeric():
            args.training_steps = f"{args.training_steps}st"
        if not args.training_steps.endswith("ep") and not args.training_steps.endswith(
            "st"
        ):
            raise ValueError("Please specify training steps as either X | Xst or Yep.")
        return super().create_from_args(args, prefix)


@dataclass
class DataConfig(Config):
    dataset: Literal["cifar10", "mnist", "cifar100", "tiny-imagenet"]

    batch_size: int = 128
    test_batch_size: int = 1024
    hflip: bool = True
    path: Path = Path("./data")
    mixup: float = 0.0
    cutmix: float = 0.0
    gaussian_blur: bool = False
    random_rotation: float = 0.0
    cutout: int = 0

    _hflip: str = "Pass true to perform random horizontal flip with probability 0.5."
    _name = "data"
    _description = "Data and augmentations configuration"


@dataclass
class LoggerConfig(Config):
    _name = "logger"
    _description = "Logger configuration"

    use_wandb: bool = False
    use_tqdm: bool = True
    print_summary: bool = True

    project: str = None
    entity: str = None
    run_name: str = None
    group: str = None
    tags: Optional[List[str]] = None
    notes: str = None
    resume_id: str = None
    experiment_id: str = None
    slurm_job_id: str = None
    log_dir: Path = Path("../experiments")
    cleanup_after: bool = False

    _cleanup_after: str = "Pass true only if you want to delete the experiment directory after experiment is finished."


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

    width_multiplier: int = 1
    depth: Literal[9, 18, 20, 50] = 20


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
    norm: Literal["layernorm", "batchnorm", "groupnorm"] = "layernorm"
    act: Literal["relu", "linear", "id", "gelu", "elu"] = "relu"

    _model_name = "Pass model name to see aditional arguments related to models"

    def __post_init__(self):
        compiled_patterns = [re.compile(pattern) for pattern in MODEL_NAME_PATTERNS]
        correct_name = any(
            pattern.fullmatch(self.model_name) for pattern in compiled_patterns
        )
        if not correct_name:
            raise ValueError(
                f"Model name not properly configured, try one of {pformat(MODEL_NAME_PATTERNS)}"
            )
        return super().__post_init__()


@dataclass
class SysConfig(Config):
    num_workers: int = 4
