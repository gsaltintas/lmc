import argparse
import hashlib
import os
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields, is_dataclass, make_dataclass
from pathlib import Path
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

import yaml
from rich.console import Console
from rich.table import Table

from lmc.utils.step import Step
from lmc.utils.utils import flatten_dict

from .config import (
    USE_DEFAULT_FACTORY,
    Config,
    DataConfig,
    LMCConfig,
    LoggerConfig,
    ModelConfig,
    ModelConfig_,
    TrainerConfig,
    add_basic_args,
    make_model_config,
    maybe_get_arg,
)


def extract_candidate_keys(klass, d):
    """Recursively collect only the keys corresponding to `klass` (which may be a Union or Dataclass)."""
    if not isinstance(d, dict):
        return {}

    # If klass is a Union, accumulate keys from any matching candidate
    if get_origin(klass) is Union:
        merged = {}
        for candidate_type in get_args(klass):
            merged.update(extract_candidate_keys(candidate_type, d))
        return merged

    result = {}
    if is_dataclass(klass):
        for fld in fields(klass):
            if isinstance(fld.type, type) and issubclass(fld.type, Step):
                if fld.name in d:
                    result[fld.name] = d[fld.name]
            # If the field type is itself a Union or a dataclass, recurse
            elif get_origin(fld.type) is Union or is_dataclass(fld.type):
                result[fld.name] = extract_candidate_keys(fld.type, d)
            elif fld.name in d:
                result[fld.name] = d[fld.name]
    return result


def dataclass_from_dict(klass: Type[Any], d: dict[str, Any]) -> Any:
    """recursively populates a dataclass from a dictionary"""
    vals = {}
    n_models = d.get("n_models", 1)
    if n_models == 1 and hasattr(klass, "n_models"):
        n_models = getattr(klass, "n_models")
    model_name = d.get("model_name", "mlp/64x2")
    if model_name == "mlp/64x2" and hasattr(klass, "model_name"):
        model_name = getattr(klass, "model_name")

    for field_ in fields(klass):
        name, typ = field_.name, field_.type

        # Pull sub-dict if it exists, else gather relevant keys from the dict
        sub = d.get(name)
        if isinstance(typ, type) and issubclass(
            typ, Seeds
        ):  # and (sub is None or len(sub) == 0):
            typ = make_seeds_class(n_models)
        elif isinstance(typ, type) and issubclass(
            typ, PerturbSeeds
        ):  # and (sub is None or len(sub) == 0):
            typ = make_perturb_seeds_class(n_models)
        elif isinstance(typ, type) and issubclass(
            typ, ModelConfig_
        ):  # and (sub is None or len(sub) == 0):
            typ = make_model_config(model_name=model_name)

        # Check if the field type is a union
        if get_origin(typ) is Union and (
            sub is None or (isinstance(sub, dict) and len(sub) == 0)
        ):
            sub = extract_candidate_keys(typ, d)
        elif sub is not None and isinstance(typ, type) and issubclass(typ, Step):
            if isinstance(sub, dict):
                sub = Step(**sub)
            else:
                sub = Step(sub)
        elif (
            is_dataclass(typ)
            and issubclass(typ, (Experiment, Config))
            and (sub is None or len(sub) == 0)
        ):
            sub = extract_candidate_keys(typ, d)

        # Recursively build dataclass if needed
        if sub is None or (isinstance(sub, dict) and len(sub) == 0):
            continue
        elif get_origin(typ) is Union:
            if type(None) in get_args(typ):
                typ = get_args(typ)[0]
                vals[name] = sub
            else:
                typ = field_.default_factory(**sub)
                cls_ = dataclass_from_dict(typ, sub)
                vals[name] = cls_
        elif is_dataclass(typ) and isinstance(sub, dict):
            vals[name] = dataclass_from_dict(typ, sub)
        else:
            vals[name] = sub
    return klass(**vals)


def zip_and_save_source(target_base_dir: Union[str, Path]) -> None:
    """Write cached config file contents to target directory. Create a summary.md under configs for easy description of the experiment. Creates a zip of the source code.

    Args:
        config (Config):
        target_base_dir (Union[str, Path]): Directory to safe the configurations

    """
    target_base_dir = Path(target_base_dir)
    if not target_base_dir.exists():
        raise NotADirectoryError("Target directory doesn't exist.")

    # Copy source folder contents over
    target_path = target_base_dir.joinpath("src.zip").absolute()
    source_path = Path(__file__).parents[1].absolute()

    filter_files = lambda x: x.suffix in [".py", ".ipynb", ".yaml", ".yml", ".sh"]
    exclude_dirs = [
        "build",
        "experiments",
        ".ipynb_checkpoints",
        "tmp",
        "data",
        "wandb",
        "results",
    ]
    filter_dirs = lambda path: not any(
        [x in Path(path).parents._parts for x in exclude_dirs]
    )

    with zipfile.ZipFile(target_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(source_path):
            if not filter_dirs(root):
                continue
            for file_or_dir in files + dirs:
                full_path = Path(root, file_or_dir)
                if full_path.is_file() and filter_files(full_path):
                    zip_file.write(
                        os.path.join(root, file_or_dir),
                        os.path.relpath(
                            os.path.join(root, file_or_dir),
                            os.path.join(source_path, os.path.pardir),
                        ),
                    )


@dataclass
class Seeds(Config):
    _name = "seeds"
    _description = "Collection of seeds used during the experiment"


def make_seeds_class(n_models: int = None) -> Type:
    n_models = maybe_get_arg("n_models") if n_models is None else n_models
    if n_models is None:
        n_models = 1
    n_models = int(n_models)
    fields_ = [(f"seed{i}", int, 42) for i in range(1, 1 + n_models)]
    fields_ += [
        (f"_seed{i}", str, f"Seed used for training model {i}")
        for i in range(1, 1 + n_models)
    ]
    fields_ += [(f"loader_seed{i}", int, 42) for i in range(1, 1 + n_models)]
    fields_ += [
        (f"_loader_seed{i}", str, f"Seed used for data randomness of model {i}")
        for i in range(1, 1 + n_models)
    ]
    cls_ = make_dataclass("Seeds", fields_, bases=(Seeds,))
    return cls_


@dataclass(init=False)
class Experiment:
    """The bundle of hyperparameters necessary for a particular kind of job. Contains many config objects.

    Each config object should be a field of this dataclass.
    """

    trainer: TrainerConfig = None
    model: make_model_config() = field(
        init=True, default_factory=make_model_config
    )  # None
    data: DataConfig = None
    logger: LoggerConfig = None
    lmc: LMCConfig = None
    n_models: int = 1
    deterministic: bool = False
    zip_and_save_source: bool = True
    resume_from: Optional[str] = None
    resume_epoch: Optional[int] = -1
    log_to_same_experiment: Optional[bool] = False

    seeds: make_seeds_class() = field(init=False, default_factory=make_seeds_class)
    resume_from: str = None
    model_dir: Path = None

    _resume_from: str = "Pass the model_dir or wandb run (wandb:project/username/run_id) to continue training from, the following model dir must exist in the current file system."
    _log_to_same_experiment: str = "If true"
    _name_prefix: str = field(init=True, default="")
    _subconfigs: Tuple[str] = ("trainer", "model", "data", "logger")
    _description: str = field(init=True, default="")
    _deterministic: str = "If true, make CUDA exactly deterministic."
    _zip_and_save_source: str = "If true, copy code to output dir and zip compress"
    _resume_from: str = "Directory to load and resume checkpoint or wandb run by specifying wandb:ENTITY/PROJECT/PROJECT_ID"

    def __init__(self, *args, **kwargs):
        # Call Trainer's constructor so that 'model', 'data' etc. are set up
        self.trainer = kwargs.get("trainer") or dataclass_from_dict(
            TrainerConfig, kwargs
        )
        self.model = kwargs.get("model") or dataclass_from_dict(
            make_model_config(**kwargs), kwargs
        )
        self.data = kwargs.get("data") or dataclass_from_dict(DataConfig, kwargs)
        self.logger = kwargs.get("logger") or dataclass_from_dict(LoggerConfig, kwargs)
        self.lmc = kwargs.get("lmc") or dataclass_from_dict(LMCConfig, kwargs)
        self.n_models = kwargs.get("n_models", 1)
        self.model_dir = kwargs.get("model_dir", None)
        self.resume_from = kwargs.get("resume_from", None)
        self.deterministic = kwargs.get("deterministic", False)
        self.zip_and_save_source = kwargs.get("zip_and_save_source", True)

        # Dynamically build the Seeds class

        seeds_cls = make_seeds_class(self.n_models)
        seeds_kwargs = {k: kwargs.get(k) for k in seeds_cls.__annotations__}
        self.seeds = kwargs.get("seeds") or seeds_cls(**seeds_kwargs)
        self.__post_init__()

    @property
    def command(self) -> str:
        return maybe_get_arg("command", positional=True, position=0)

    def __post_init__(self):
        """The name under which experiments with these hyperparameters will be stored."""
        fields_dict = {f.name: getattr(self, f.name) for f in fields(self)}
        hparams_strs = [
            str(fields_dict[k])
            for k in sorted(fields_dict)
            if isinstance(fields_dict[k], Config)
        ]
        hash_str = hashlib.md5(";".join(hparams_strs).encode("utf-8")).hexdigest()
        self.hashname = f"{self._name_prefix}_{hash_str}"

    @classmethod
    def add_args(
        cls,
        parser: argparse.ArgumentParser,
        defaults: Union["Experiment", Dict[str, Any]] = None,
    ):
        add_basic_args(
            parser,
            cls,
            defaults,
            prefix=None,
            name=cls._name_prefix,
            description=cls._description,
            create_group=True,
        )

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> "Experiment":
        confs = defaultdict()
        for field_ in fields(cls):
            if field_.name.startswith("_"):
                continue
            conf = field_.type
            if conf is USE_DEFAULT_FACTORY:
                conf = field_.default_factory()
            if isinstance(conf, type) and issubclass(conf, Config):
                conf_ = conf.create_from_args(args)
            else:
                conf_ = getattr(args, field_.name)
            confs[field_.name] = conf_

        # Create the desc.
        return cls(**confs)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Experiment":
        return dataclass_from_dict(cls, d)

    def save(self, output_dir: Union[Path, str], zip_code_base: bool = True) -> None:
        """saves the configuration as a yaml file at output_dir"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        file_str = yaml.dump(
            asdict(
                self,
                dict_factory=lambda lst: dict(
                    [(key, val) for key, val in lst if not key.startswith("_")]
                ),
            )
        )
        output_dir.joinpath("config.yaml").write_text(file_str, encoding="utf-8")

        if zip_code_base:
            zip_and_save_source(output_dir)

    @classmethod
    def load_from_file(cls, file_path: Union[Path, str]) -> "Experiment":
        file_path = Path(file_path)
        assert file_path.exists()
        assert file_path.suffix in [".yaml", ".yml"]

        with open(file_path) as stream:
            dct = yaml.load(stream, Loader=yaml.Loader)

        # # Create the data class from loaded dict
        return dataclass_from_dict(cls, dct)

    @property
    def display(self) -> str:
        console = Console()
        s = ""

        # Create a rich Table
        table = Table(title=self.__class__.__name__)

        # Add columns for the table
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        # Add rows for each non-default field
        for field_ in fields(self):
            if field_.name.startswith("_"):
                continue
            typ = field_.type
            if isinstance(typ, type) and issubclass(typ, (Experiment, Config)):
                s += "\n" + getattr(self, field_.name).display + "\n"
            else:
                value = getattr(self, field_.name)
                table.add_row(field_.name, str(value))

        # Capture the table as a string using Console's capture method
        with console.capture() as capture:
            console.print(table)
        return capture.get() + "\n\n" + s

    def wandb_dct(self):
        d = dict()
        for field_ in fields(self):
            if field_.name.startswith("_"):
                continue
            val = getattr(self, field_.name)
            if isinstance(field_.type, type) and issubclass(field_.type, Step):
                # for steps only log step int
                if "st" in val.value:
                    val = int(val.value.replace("st", ""))
                else:
                    val = val.value
            if isinstance(field_.type, type) and issubclass(field_.type, Config):
                val = val.wandb_dct()
            d[field_.name] = val
        return d


@dataclass(init=False)
class Trainer(Experiment):
    _name_prefix: str = "trainer"
    _description: str = "Run a training script."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@dataclass(init=False)
class Finetuner(Trainer):
    _name_prefix: str = "finetuner"
    _description: str = "Run a finetuning script."

    # todo something wrong here
    frozen_layers: List[str] = field(init=True, default_factory=list)
    _frozen_layers: str = "List of frozen layers or regex patterns"

    def __init__(self, *args, **kwargs):
        self.frozen_layers = kwargs.get("frozen_layers", [])
        super().__init__(*args, **kwargs)


@dataclass
class PerturbSeeds(Config):
    _name = "perturb-seeds"
    _description = "Collection of seeds used during the perturbation of the models"


def make_perturb_seeds_class(n_models: int = None) -> Type:
    n_models = maybe_get_arg("n_models") if n_models is None else n_models
    if n_models is None:
        n_models = 1
    n_models = int(n_models)
    fields_ = [(f"perturb_seed{i}", int, None) for i in range(1, 1 + n_models)]
    fields_ += [
        (
            f"_perturb_seed{i}",
            str,
            f"Seed used for perturbing model {i}, defaults to none and uses the original randomness of the model setup.",
        )
        for i in range(1, 1 + n_models)
    ]
    cls_ = make_dataclass("PerturbSeeds", fields_, bases=(PerturbSeeds,))
    return cls_


@dataclass(init=False)
class PerturbedTrainer(Trainer):
    perturb_inds: List[int] = field(init=True, default_factory=lambda: [-1])
    perturb_step: Step
    perturb_mode: Literal["gaussian", "batch"] = "gaussian"
    perturb_scale: float = 0
    normalize_perturb: bool = False
    scale_to_init_if_normalized: bool = True
    same_steps_pperturb: bool = True
    rewind_lr: bool = False
    perturb_seeds: make_perturb_seeds_class() = field(
        default_factory=make_perturb_seeds_class, init=True
    )
    sample_noise_at: Literal["init", "perturb"] = (
        "perturb"  # TODO does nothing, always sample at perturb time, kept for backwards compatibility
    )
    dont_perturb_module_patterns: List[str] = field(
        init=True, default_factory=lambda: []
    )
    log_per_layer_l2: bool = False
    perturb_debug_dummy_run: bool = False

    _perturb_step: str = "Perturbation step either of the from Xst | X or Xep"
    _perturb_inds: str = "List of models to perturb"
    _perturb_mode: str = "Determines the perturbation mode,\n\tif gaussian, ϵ∼N(0,σ²)\n\tif batch, ϵ=∇L(x,y;θ₀), (x,y)∼D"
    _perturb_scale: str = "Scale to multiply the perturbation with"
    _normalize_perturb: str = (
        "If true, perturbation is normalized to have an l₂ norm of perturb_scale"
    )
    _same_steps_pperturb: str = (
        "If true, perturbed model is trained for 'training_steps' after perturbation"
    )
    _rewind_lr: str = "If true, learning rate is rewound back to the max learning rate"
    _sample_noise_at: str = "Sample noise at the given step, defaults to initialization"
    _name_prefix: str = "perturbed-trainer"
    _description: str = "Run a butterfly experiment."
    _dont_perturb_module_patterns: str = "List of regex patterns that match parameter names which should not be perturbed.\n If a parameter's name matches any pattern, it will receive zero noise instead of perturbation.\n Examples: ['.*\.bias$'] to skip bias terms, ['layer1\..*'] to skip layer1, ['.*\.norm2\..*'] for norm layers."

    def __init__(self, *args, **kwargs):
        self.perturb_inds = kwargs.get("perturb_inds", [-1])
        self.perturb_step = kwargs.get("perturb_step", 0)
        self.perturb_mode = kwargs.get("perturb_mode", "gaussian")
        self.perturb_scale = kwargs.get("perturb_scale", 0)
        self.normalize_perturb = kwargs.get("normalize_perturb", False)
        self.scale_to_init_if_normalized = kwargs.get(
            "scale_to_init_if_normalized", True
        )
        self.same_steps_pperturb = kwargs.get("same_steps_pperturb", True)
        self.log_per_layer_l2 = kwargs.get("log_per_layer_l2", False)
        self.rewind_lr = kwargs.get("rewind_lr", False)
        self.sample_noise_at = kwargs.get("sample_noise_at", "init")
        self.perturb_debug_dummy_run = kwargs.get("perturb_debug_dummy_run", False)
        self.dont_perturb_module_patterns = kwargs.get(
            "dont_perturb_module_patterns", []
        )

        n_models = kwargs.get("n_models", 1)
        # Dynamically build the Seeds class
        seeds_cls = make_perturb_seeds_class(n_models)
        seeds_kwargs = {k: kwargs.get(k) for k in seeds_cls.__annotations__}
        self.perturb_seeds = kwargs.get("perturb_seeds") or seeds_cls(**seeds_kwargs)
        super().__init__(*args, **kwargs)
        self.__post_init__()

    def __post_init__(self):
        # Convert negative indexes to 1-based from the end
        for i, p in enumerate(self.perturb_inds):
            p = int(p)
            if p < 0:
                p = self.n_models + p + 1  # 1-based indexing
            self.perturb_inds[i] = p

        if isinstance(self.perturb_step, int):
            self.perturb_step = Step(f"{self.perturb_step}st")
        elif isinstance(self.perturb_step, str):
            self.perturb_step = Step(self.perturb_step)

        return super().__post_init__()
