import io
from pathlib import Path
from typing import Union

import datasets
import torch
from datasets import load_dataset
from PIL import Image

from .dataset import DataMode, MyDataset


class ImageNet(MyDataset):
    """ImageNet-1K dataset using HuggingFace's datasets"""

    def __init__(
        self,
        root: Union[str, Path],
        transform=None,
        train: bool = True,
        download: bool = False,  # Not used but kept for compatibility
        mode: str = DataMode.MMAP,
    ):
        # For HuggingFace version, we always want to use MMAP mode
        if mode != DataMode.MMAP:
            print(
                f"Warning: Forcing MMAP mode for HuggingFace ImageNet dataset instead of {mode}"
            )
        super().__init__(
            root, transform, train, download, path_suffix="imagenet", mode=DataMode.MMAP
        )

    def _initialize_storage(self):
        """Initialize the HuggingFace dataset"""
        split = "train" if self.train else "validation"
        print(f"Loading ImageNet-1K {split} split from HuggingFace...")
        self.hf_dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            # "imagenet-panda",  # Changed to more reliable version
            split=split,
            cache_dir=str(self.path),
            trust_remote_code=True,
        )

    def __getitem__(self, idx):
        """Get an image and its label"""
        item = self.hf_dataset[idx]

        # Convert image from bytes to PIL Image
        img = Image.open(io.BytesIO(item["image"]["bytes"])).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, item["label"]

    def __len__(self):
        return len(self.hf_dataset)

    @property
    def classes(self):
        """Get ImageNet class names"""
        return self.hf_dataset.features["label"].names

    def _get_file_paths(self):
        """Not used with HuggingFace dataset"""
        pass

    def _create_arrow_dataset(self, arrow_path: Path):
        """Not used with HuggingFace dataset"""
        pass

    def _download(self):
        """HuggingFace handles downloading"""
        pass
