from abc import abstractmethod
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

"""base dataset class"""


import io
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import datasets
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DataMode:
    DISK = "disk"  # Load from disk on-the-fly
    MEMORY = "memory"  # Store in RAM as tensors
    MMAP = "mmap"  # Memory-mapped storage (Arrow/Parquet)


class MyDataset(Dataset, ABC):
    def __init__(
        self,
        root: Union[str, Path],
        transform=None,
        train: bool = True,
        download: bool = False,
        path_suffix: str = "",
        mode: str = DataMode.DISK,
        cache_size: int = 1000,
    ):
        self.root = Path(root).resolve()
        self.path = self.root.joinpath(path_suffix)
        self.transform = transform
        self.train = train
        self.mode = mode
        self.cache_size = cache_size

        # Will store data differently based on mode
        self.data = None
        self.labels = None
        self.hf_dataset = None

        if download:
            self._download()

        # Initialize storage based on mode
        self._initialize_storage()

    def _initialize_storage(self, arrow_path: str = "arrow"):
        """Initialize the appropriate storage mechanism based on mode"""
        if self.mode == DataMode.MMAP:
            arrow_path = self.path.joinpath(arrow_path)
            if not arrow_path.exists():
                self._create_arrow_dataset(arrow_path)
            split = "train" if self.train else "test"
            self.hf_dataset = datasets.load_from_disk(str(arrow_path))[split]

        elif self.mode == DataMode.MEMORY:
            self.data, self.labels = self._load_into_memory()

        elif self.mode == DataMode.DISK:
            self.data, self.labels = self._get_file_paths()

    @abstractmethod
    def _get_file_paths(self) -> tuple[list, list]:
        """Return lists of file paths and corresponding labels"""
        pass

    @abstractmethod
    def _create_arrow_dataset(self, arrow_path: Path):
        """Create an Arrow dataset at the specified path"""
        pass

    def _load_into_memory(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Load all images into memory as tensors"""
        paths, labels = self._get_file_paths()
        images = []

        for path in paths:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        return torch.stack(images), torch.tensor(labels)

    def __len__(self):
        if self.mode == DataMode.MMAP:
            return len(self.hf_dataset)
        return len(self.labels)

    def __getitem__(self, idx):
        if self.mode == DataMode.MMAP:
            item = self.hf_dataset[idx]
            img = Image.open(io.BytesIO(item["image"]["bytes"])).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, item["label"]

        elif self.mode == DataMode.MEMORY:
            img = self.data[idx]
            # No need to transform as it was done during loading
            return img, self.labels[idx]

        else:  # DISK mode
            img_path = self.data[idx]
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]

    @property
    @abstractmethod
    def classes(self):
        pass

    @abstractmethod
    def _download(self):
        pass
