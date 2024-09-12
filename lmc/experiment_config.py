
import argparse
import hashlib
import os
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields, make_dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type, Union, get_args, get_origin

import yaml
from rich.console import Console
from rich.table import Table

from .config import (USE_DEFAULT_FACTORY, Config, DataConfig, LoggerConfig,
                     ModelConfig, TrainerConfig, add_basic_args, maybe_get_arg)


def dataclass_from_dict(klass: Type, d: Dict[str, Any]) -> dataclass:
    """ recursively creates a dataclass from dictionary """
    try:
        fieldtypes = dict()#{f.name: f.type for f in fields(klass)}
        for f in fields(klass):
            typ = f.type
            if get_origin(typ) is Union:
                if type(None) in get_args(typ):
                    typ = get_args(typ)[0]
                else:
                    typ = f.default_factory()
            elif typ is USE_DEFAULT_FACTORY:
                typ = f.default_factory()
            fieldtypes[f.name] = typ
            # todo: here handle better, doesn't parse the correct one yet for opt, but does it correctly for model
            # if f.name == "trainer" or f.name.startswith("opt"):
            #     print("Hello", f.type)
                #todo: handle union

        return klass(**{f:dataclass_from_dict(fieldtypes[f],d[f]) for f in d})
    except:
        return d # Not a dataclass field
    

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

def make_seeds_class() -> Type:
    n_models = maybe_get_arg("n_models")
    if n_models is None:
        n_models = 1
    n_models = int(n_models)
    fields_ = [(f"seed{i}", int, 42) for i in range(1, 1+n_models)]
    fields_ += [(f"_seed{i}", str, f"Seed used for training model {i}") for i in range(1, 1+n_models)]
    fields_ += [(f"loader_seed{i}", int, 42) for i in range(1, 1+n_models)]
    fields_ += [(f"_loader_seed{i}", str, f"Seed used for data randomness of model {i}") for i in range(1, 1+n_models)]
    cls_ = make_dataclass("Seeds", fields_, bases=(Seeds,))
    return cls_


@dataclass 
class Experiment:

    """The bundle of hyperparameters necessary for a particular kind of job. Contains many config objects.

    Each config object should be a field of this dataclass.
    """
    trainer: TrainerConfig = None
    model: ModelConfig = None
    data: DataConfig = None
    logger: LoggerConfig = None
    seeds: make_seeds_class() = field(init=True, default_factory=make_seeds_class)
    resume_from: str = None
    n_models: int = 1  
    model_dir: Path = None
    _resume_from: str = "Pass the model_dir or wandb run (wandb:project/username/run_id) to continue training from, the following model dir must exist in the current file system."
    _name_prefix: str = field(init=False, default="")
    _subconfigs: Tuple[str] = ("trainer", "model", "data", "logger")
    _description: str = field(init=False, default="")

    @property
    def command(self) -> str:
        return maybe_get_arg("command", positional=True, position=0)
    
    def __post_init__(self):
        """The name under which experiments with these hyperparameters will be stored."""
        fields_dict = {f.name: getattr(self, f.name) for f in fields(self)}
        hparams_strs = [str(fields_dict[k]) for k in sorted(fields_dict) if isinstance(fields_dict[k], Config)]
        hash_str = hashlib.md5(';'.join(hparams_strs).encode('utf-8')).hexdigest()
        self.hashname = f'{self._name_prefix}_{hash_str}'
    
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, defaults: 'Experiment' = None):
        add_basic_args(parser, cls, defaults, prefix=None, name=cls._name_prefix, description=cls._description, create_group=True)

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> 'Experiment':
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


    def save(self, output_dir: Union[Path, str], zip_code_base: bool = True) -> None:
        """ saves the configuration as a yaml file at output_dir """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        file_str = yaml.dump(asdict(self, dict_factory=lambda lst: dict([(key, val) for key, val in lst if not key.startswith("_")])))
        output_dir.joinpath("config.yaml").write_text(file_str, encoding="utf-8")

        if zip_code_base:
            zip_and_save_source(output_dir)


    @classmethod
    def load_from_file(cls, file_path: Union[Path, str]) -> 'Experiment':
        file_path = Path(file_path)
        assert file_path.exists()
        assert file_path.suffix in [".yaml", ".yml"]
        
        with open(file_path) as stream:
            dct = yaml.load(stream, Loader=yaml.Loader)
        # # Create the desc.
        return dataclass_from_dict(cls, dct)

    @property
    def display(self):
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
            if isinstance(typ, type) and issubclass(typ, Config):
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
            if isinstance(field_.type, type) and issubclass(field_.type, Config):
                val = val.wandb_dct()
            d[field_.name] = val
        return d

@dataclass
class Trainer(Experiment):
    subconfigs = [TrainerConfig, DataConfig, ModelConfig]
    _name_prefix = "trainer"
