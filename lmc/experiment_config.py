
import argparse
import hashlib
import os
import zipfile
from abc import abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Tuple, Type, Union

import yaml

from .config import (Config, DataConfig, LoggerConfig, ModelConfig,
                     TrainerConfig)

"""
"""

def dataclass_from_dict(klass: Type, d: Dict[str, Any]) -> dataclass:
    """ recursively creates a dataclass from dictionary """
    try:
        fieldtypes = {f.name: f.type for f in fields(klass)}
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
class Experiment:

    """The bundle of hyperparameters necessary for a particular kind of job. Contains many hparams objects.

    Each hparams object should be a field of this dataclass.
    """
    trainer: TrainerConfig = None
    model: ModelConfig = None
    data: DataConfig = None
    logger: LoggerConfig = None
    _name_prefix: str = field(init=False, default="")
    _subconfigs: Tuple[str] = ("trainer", "model", "data", "logger")

    @property
    def hashname(self) -> str:
        """The name under which experiments with these hyperparameters will be stored."""
        fields_dict = {f.name: getattr(self, f.name) for f in fields(self)}
        hparams_strs = [str(fields_dict[k]) for k in sorted(fields_dict) if isinstance(fields_dict[k], Config)]
        hash_str = hashlib.md5(';'.join(hparams_strs).encode('utf-8')).hexdigest()
        return f'{self._name_prefix}_{hash_str}'
    
    # @staticmethod
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, defaults: 'Experiment' = None):
        for f in fields(cls):
            if f.name.startswith("_"):
                continue
            typ = f.type
            if isinstance(typ, type) and issubclass(typ, Config):
                typ.add_args(parser, defaults=defaults)

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> 'Experiment':
        confs = defaultdict()
        for field_ in fields(cls):
            if field_.name.startswith("_"):
                continue
            conf = field_.type
            conf_ = conf.create_from_args(args)
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
        
        with open('config.yaml') as stream:
            dct = yaml.load(stream, Loader=yaml.Loader)
        # # Create the desc.
        return dataclass_from_dict(cls, dct)

    @property
    def display(self):
        s = ""
        for field_ in fields(self):
            if field_.name.startswith("_"):
                continue
            typ = field_.type
            if isinstance(typ, type) and issubclass(typ, Config):
                s += "\n" + getattr(self, field_.name).display + "\n"
            else:
                s += ""
        return s
    
    
@dataclass
class Trainer(Experiment):
    subconfigs = [TrainerConfig, DataConfig, ModelConfig]
    _name_prefix = "trainer"
