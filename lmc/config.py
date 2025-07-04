import argparse
import logging
import math
import os
from dataclasses import MISSING, dataclass, field, fields, make_dataclass
from pathlib import Path
from pprint import pformat
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)

from rich.console import Console
from rich.table import Table

from lmc.data.data_stats import (
    MAX_SEQ_LENGTH_DICT,
    TASK_MAPPING,
    DatasetRegistry,
    LanguageConfig,
    TaskType,
    VisionConfig,
)
from lmc.models.type_declaration import MODEL_NAME_PATTERNS, Activations, Inits, Norms
from lmc.utils.step import Step
from lmc.utils.utils import pattern_matched

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
    if boolean_arg:  # prob not true
        parser.add_argument(prefix + arg_name, action="store_true")
    else:
        parser.add_argument(prefix + arg_name, type=str, default=None)
    try:
        args = parser.parse_known_args()[0]
        return getattr(args, arg_name) if arg_name in args else None
    except:
        return None


def field_factory(
    maybe_arg: str,
    mapping: Dict[str, "Config"],
    default_val: str = None,
    from_args: bool = True,
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

    def arg_fact(**kwargs):
        value = maybe_get_arg(maybe_arg)
        if value is None:
            value = kwargs.get(maybe_arg, None)
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
    defaults: Union["Config", Dict[str, Any]] = None,
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

        default = None
        if defaults is not None:
            if isinstance(defaults, dict):
                flat_key = (
                    f"{field_.name}" if prefix is None else f"{prefix}_{field_.name}"
                )
                default = defaults.get(flat_key)
            else:
                default = getattr(defaults, field_.name, None)

        if default is None:
            if field_.default is not MISSING:
                default = field_.default
            elif field_.default_factory is not MISSING:
                default = field_.default_factory()
            else:
                default = None

        required = (
            field_.default is MISSING
            and field_.default_factory is MISSING
            and (not defaults or default is None)
        )

        helptext = getattr(cls, f"_{field_.name}", "")
        if required:
            helptext = f"(required: {typ.__name__}) " + helptext
        elif default is not None:
            helptext = f"(default: {default}) " + helptext
        else:
            helptext = f"(optional: {typ.__name__}) " + helptext
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
        elif typ == Step:
            parser.add_argument(arg_name, type=str, **kwargs)
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
        elif callable(typ):
            parser.add_argument(arg_name, type=typ, **kwargs)
        else:
            parser_.error(f"Invalid field type {typ} for config.")
    parser = parser_


@dataclass
class Config:
    """base config class"""

    _add_prefix: bool = field(
        init=False, default=False
    )  # Class-level default, set to true to parse arguments as --name.arg
    _name: str = field(
        init=False, default=""
    )  # Class-level default, set to true to parse arguments as --name.arg
    _description: str = field(
        init=False, default=""
    )  # Class-level default, set to true to parse arguments as --name.arg

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
        defaults: Union["Config", Dict[str, Any]] = None,
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
            elif typ == Step:
                arg_val = getattr(args, arg_name)
                if type(arg_val) is str:
                    arg_val = Step(arg_val)
                d[_field.name] = arg_val
            # Nested hparams.
            elif isinstance(typ, type) and issubclass(typ, Config):
                subprefix = typ._name if typ._add_prefix else None
                d[_field.name] = typ.create_from_args(args, subprefix)
            else:
                d[_field.name] = getattr(args, arg_name)

        return cls(**d)

    @property
    def display(self, only_modified: bool = False):
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
            typ = field_.type
            if get_origin(typ) is Union:
                if type(None) in get_args(typ):
                    typ = get_origin(get_args(typ)[0])
                else:
                    typ = field_.default_factory()
            val = getattr(self, field_.name)
            if isinstance(field_.type, type) and issubclass(field_.type, Step):
                # for steps only log step int
                if "st" in val.value:
                    val = int(val.value.replace("st", ""))
                else:
                    val = val.value
            elif isinstance(typ, type) and issubclass(typ, Config):
                val = val.wandb_dct()
            d[field_.name] = val
        return d


USE_DEFAULT_FACTORY = object()


@dataclass
class LrSchedule(Config):
    pass


@dataclass
class Optimizer(Config):
    lr: float = 0.1
    optimizer: Literal["sgd", "adam", "adamw"] = "sgd"
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    gradient_clip_val: float = 0.0
    gradient_clip_algorithm: Literal["norm", "value", ""] = None
    lr_scheduler: Literal[
        "linear", "exponential", "onecycle", "cosine", "triangle", "constant"
    ] = None

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
    momentum: float = None  # dummy variable
    _momentum: str = "Dummy variable for cross-compatibility with scripts that call sgd, this value is ignored"


@dataclass
class TrainerConfig(Config):
    training_steps: Step
    opt: Union[_SGDConfig, _AdamConfig] = field(
        init=True,
        default_factory=field_factory(
            maybe_arg="optimizer",
            mapping={"sgd": _SGDConfig, "adam": _AdamConfig, "adamw": _AdamConfig},
            default_val="sgd",
        ),
    )

    eval_freq: str = "1ep"
    eval_specific_steps: str = ""
    save_freq: str = "1ep"
    save_specific_steps: str = ""
    save_early_iters: bool = False
    save_best: bool = True
    use_scaler: bool = False
    label_smoothing: float = 0.0
    gradient_accumulation_steps: int = 1

    _training_steps = "Total number of steps (st) or epochs (ep), specify as Xep or Xst"
    _save_freq = "Frequency to save checkpoints during training in steps (st) or epochs (ep), specify as Xep or Xst, set to none or false to only save at end"
    _save_specific_steps = "Comma separated list of steps (st) or epochs (ep) to save, specify as Xep or Xst, is combined with save_freq and save_early_iters."
    _save_best = "Pass to save the best model"
    _use_scaler = "Pass to use torch gradient scaling"

    _name = "trainer"
    _description = "Training hyper-parameters"


@dataclass
class DataConfig(Config):
    # Dataset choices including both vision and language datasets
    dataset: Literal[
        # Vision datasets
        "cifar10",
        "cifar10easy",
        "cifar10hard",
        "mnist",
        "cifar100",
        "tiny-imagenet",
        "cinic10",
        "cinic10_wo_cifar10",
        "imagenet1k",
        "eurosat",
        "ade20k",
        # Language datasets - Text Classification/Regression (CR)
        "snli",
        "scitail",
        "cola",
        "sst2",
        "mrpc",
        "qqp",
        "mnli",
        "qnli",
        "rte",
        "wnli",
        "stsb",
        # "glue/cola", "glue/sst2", "glue/mrpc", "glue/qqp", "glue/mnli", "glue/qnli", "glue/rte", "glue/wnli", "glue/stsb",
        # Language datasets - Question Answering (QA)
        "squad_v1",
        "squad_v2",
        "newsqa",
        "hotpotqa",
        "duorc",
        "drop",
        "wikihop",
        "boolq",
        "comqa",
        # Language datasets - Sequence Labeling (SL)
        "conll2003",
        "ptb",
        "conj",
        # Language datasets - Language Modeling
        "wikitext-2",
        "wikitext-103",
        "cord",
        "cord-v2",
        "webtext",
        "c4",
        "pile",
        "bookcorpus",
        "gsm8k",
        "math",
        "mathqa",
        "asdiv",
    ]

    # General configurations
    batch_size: int = 128
    test_batch_size: int = 1024
    path: Path = Path("./data")
    download: bool = False
    num_workers: int = 4
    chat_template: Optional[str] = None

    # Vision-specific augmentations
    hflip: bool = True
    mixup: float = 0.0
    cutmix: float = 0.0
    gaussian_blur: bool = False
    random_rotation: float = 0.0
    random_crop: bool = False  # TODO this does nothing, only kept here to preserve backwards compatibility
    random_translate: float = 0.0
    cutout: int = 0
    resize: int = None
    pin_memory: bool = True

    # Language-specific configurations
    tokenizer_name: Optional[str] = None
    max_seq_length: int = 512
    min_seq_length: int = 4
    padding_side: str = "right"
    truncation_side: str = "right"
    pad_to_multiple_of: Optional[int] = 8

    # Text preprocessing flags
    lowercase: bool = True
    remove_punctuation: bool = False
    strip_accents: bool = True
    add_special_tokens: bool = True

    # Language augmentation settings
    token_dropout_prob: float = 0.0
    word_dropout_prob: float = 0.0
    whole_word_masking: bool = False
    masking_probability: float = 0.15
    span_length: int = 3
    enable_back_translation: bool = False
    translation_languages: List[str] = field(default_factory=lambda: ["de", "fr"])

    # Dataset splits
    validation_split: float = 0.1
    test_split: float = 0.1
    shuffle_dataset: bool = True
    random_labels: Optional[bool] = False
    _random_labels: str = "If True, labels will be shuffled"
    # Documentation fields
    _hflip: str = "Pass true to perform random horizontal flip with probability 0.5."
    _name = "data"
    _description = (
        "Data and augmentations configuration for both vision and language tasks"
    )

    def __post_init__(self):
        self.path = Path(self.path).resolve().absolute()

        # Validate language-specific configurations when using language datasets
        if self.is_language_dataset():
            if not self.tokenizer_name:
                raise ValueError("Must provide tokenizer_name for language datasets")

            if self.max_seq_length < self.min_seq_length:
                raise ValueError("max_seq_length must be greater than min_seq_length")

            if self.masking_probability > 1.0 or self.masking_probability < 0.0:
                raise ValueError("masking_probability must be between 0 and 1")

        self.max_seq_length = MAX_SEQ_LENGTH_DICT.get(self.dataset, 128)
        return super().__post_init__()

    def is_language_dataset(self) -> bool:
        """Check if the selected dataset is a language dataset"""
        return not isinstance(
            DatasetRegistry.get_dataset_info(self.dataset), VisionConfig
        )

    @property
    def task_type(self) -> TaskType:
        """Get the task type for the dataset"""
        return DatasetRegistry.get_dataset_info(self.dataset).task_type

    @property
    def max_gen_seq_length(self) -> int:
        return DatasetRegistry.get_dataset_info(self.dataset).max_gen_seq_length

    def is_sequence_labeling(self) -> bool:
        """Check if the current dataset is a sequence labeling task"""
        return self.task_type == TaskType.SEQUENCE_LABELING

    def is_question_answering(self) -> bool:
        """Check if the current dataset is a QA task"""
        return self.task_type == TaskType.QUESTION_ANSWERING

    def is_classification(self) -> bool:
        """Check if the current dataset is a classification task"""
        return self.task_type in [
            TaskType.CLASSIFICATION,
            TaskType.NATURAL_LANGUAGE_INFERENCE,
        ]

    def get_num_labels(self) -> int:
        """Get number of labels/classes for the dataset"""
        config: Union[VisionConfig, LanguageConfig] = DatasetRegistry.get_dataset_info(
            self.dataset
        )
        return config.classes

    def get_num_in_channels(self) -> int:
        """Get number of input channels for vision datasets."""
        if self.is_language_dataset():
            raise ValueError(
                f"Cannot get number of channels for language dataset {self.dataset}"
            )

        config: VisionConfig = DatasetRegistry.get_dataset_info(self.dataset)
        if not isinstance(config, VisionConfig):
            raise ValueError(f"Dataset {self.dataset} is not a vision dataset")

        return config.channels

    def get_default_res(self) -> int:
        """Get default resolution for vision datasets."""
        if self.is_language_dataset():
            raise ValueError(
                f"Cannot get resolution for language dataset {self.dataset}"
            )

        config: VisionConfig = DatasetRegistry.get_dataset_info(self.dataset)
        if not isinstance(config, VisionConfig):
            raise ValueError(f"Dataset {self.dataset} is not a vision dataset")

        return config.resolution

    def get_hf_path(self) -> str:
        config: LanguageConfig = DatasetRegistry.get_dataset_info(self.dataset)
        if not isinstance(config, VisionConfig):
            raise ValueError(f"Dataset {self.dataset} is not a language dataset")
        return config.hf_config

    @property
    def dataset_info(self) -> Union[VisionConfig, LanguageConfig]:
        """Get complete dataset configuration"""
        return DatasetRegistry.get_dataset_info(self.dataset)

    def get_steps_per_epoch(self) -> int:
        return math.ceil(self.dataset_info.samples / self.batch_size)


@dataclass
class LoggerConfig(Config):
    _name = "logger"
    _description = "Logger configuration"

    use_wandb: bool = False
    wandb_offline: bool = False
    use_tqdm: bool = True
    print_summary: bool = True
    print_optimizers: bool = True
    push_to_hub: bool = False
    hf_path: str = ""

    project: str = None
    entity: str = None
    run_name: str = None
    run_id: str = None
    group: str = None
    tags: Optional[List[str]] = None
    notes: str = None
    # resume_id: str = None
    experiment_id: str = None
    slurm_job_id: str = None
    log_dir: Path = Path("../experiments")
    cleanup_after: bool = False
    level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    profile: bool = False
    enforce_new_model_dir: bool = True
    report_permutation_stats: bool = True

    _cleanup_after: str = "Pass true only if you want to delete the experiment directory after experiment is finished."
    _enforce_new_model_dir: str = "If true, creates a new model dir if model_dir already exists (helpful for finetuning settings)."
    _report_permutation_stats: str = (
        "If true, collects and logs statistics about the found permutation."
    )

    def __post_init__(self):
        s = self.slurm_job_id
        current_slurm_id = os.environ.get("SLURM_JOB_ID", None)
        if s:
            s = [s, current_slurm_id]
        else:
            s = current_slurm_id
        self.log_dir = Path(self.log_dir).resolve().absolute()
        if self.hf_path == "":
            from huggingface_hub import whoami

            user = whoami()
            if "name" not in user:
                raise ValueError(
                    "Huggingface user unknown, please provide a valid hf_path to push the end model to the hub."
                )
            user = user["name"]
            self.hf_path = f"{user}/{self.run_name}-{current_slurm_id}"
        self.hf_path = self.hf_path.replace("--", "_").replace("..", "_")
        return super().__post_init__()


@dataclass
class Model:
    def __str__(self):
        s = "("
        for f in fields(self):
            s += f.name + "=" + f.value
        return s + ")"


@dataclass
class ModelConfig_(Config):
    # model_name: str = field(init=True, kw_only=True)
    model_name: str = "mlp/64x2"

    norm: Norms = None
    act: Activations = "relu"
    initialization_strategy: Inits = "kaiming_normal"
    ckpt_path: Optional[Path] = None
    gradient_checkpointing: bool = True
    revision: Optional[str] = None
    use_bfloat16: bool = False

    _model_name: str = "Name of the model e.g. mlp, resnet. Could also be model code resnet20-64, etc. Pass model name to see aditional arguments related to models"
    _initialization_strategy: str = "Initialization strategy for the model's layers"
    _gradient_checkpointing: str = (
        "Only implemented for HuggingFace models, disable by passing false"
    )
    _revision: str = "Pass revision for using earlier checkpoints from huggingface, e.g. '--model_name=allenai/OLMo-2-1124-7B --revision=step1000-tokens5B'"
    _use_bfloat_16: str = "If true, loads and trains the model with bfloat16 precision, only implemented for OLMo family"

    _name: str = "model"
    _description: str = ""
    _add_prefix: str = False

    def __post_init__(self):
        correct_name = pattern_matched(self.model_name, MODEL_NAME_PATTERNS)
        if not correct_name:
            raise ValueError(
                f"Model name not properly configured, try one of {pformat(MODEL_NAME_PATTERNS)}"
            )
        if self.revision and self.revision.lower() in ["null", "none"]:
            self.revision = None
        return super().__post_init__()


@dataclass
class ResNetConfig(ModelConfig_):
    _name = "resnet"
    _description = ""
    _add_prefix = True

    width: int = 16
    depth: Literal[9, 18, 20, 50] = 20

    _width: str = "Output channels of the first convolution"


@dataclass
class NLPModelConfig(ModelConfig_):
    pass


@dataclass
class MLPConfig(ModelConfig_):
    _name = "mlp"
    _description = ""
    _add_prefix = True

    width: int = 1024
    num_hidden_layers: int = 4

    _num_hidden_layers: str = "Number of hidden layers, depth: (num_hidden_layers + 1)."


def make_model_config(**kwargs) -> Type:
    model_name = maybe_get_arg("model_name")
    if model_name is None:
        model_name = kwargs.get("model_name", "mlp")
    model_name = str(model_name)
    model_cls = MLPConfig
    if "mlp" in model_name:
        model_cls = MLPConfig
    elif "resnet" in model_name:
        model_cls = ResNetConfig
    elif "bert" in model_name:
        model_cls = NLPModelConfig
    elif "t5" in model_name:
        model_cls = NLPModelConfig
    fields_ = [
        (f.name, f.type, f) for f in fields(model_cls) if not f.name.startswith("_")
    ] + [(f.name, f.type, f) for f in fields(model_cls) if f.name.startswith("_")]

    return make_dataclass("ModelConfig", fields_, bases=(model_cls,))


@dataclass
class ModelConfig(Config):
    pass


ModelConfig = make_model_config()


@dataclass
class SysConfig(Config):
    pass


@dataclass
class LMCConfig(Config):
    n_points: int = 11
    activation_matching_samples: int = 2
    lmc_check_perms: bool = True
    lmc_on_epoch_end: bool = False
    lmc_on_train_end: bool = True
    lmc_freq: str = ""
    lmc_specific_steps: str = ""

    _n_points: str = "Number of points to interpolate models."
    _activation_matching_samples: str = "Number of samples to match activations."
    _lmc_on_epoch_end: str = "Deprecated, same as setting lmc_freq=1ep"
