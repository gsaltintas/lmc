"""implements CINIC10 dataset, for more information, refer to https://paperswithcode.com/dataset/cinic-10"""

import logging
import subprocess
from pathlib import Path
from typing import Union

import torch
from datasets import Dataset, DatasetDict, Image
from PIL import Image

from lmc.data.dataset import DataMode, MyDataset

logger = logging.getLogger(__name__)


class CINIC10(MyDataset):
    def __init__(
        self,
        root: Union[str, Path],
        transform=None,
        train: bool = True,
        download: bool = False,
        mix_valid_and_train: bool = False,
        mode: str = DataMode.MMAP,
    ):
        self.mix_valid_and_train = mix_valid_and_train
        super().__init__(
            root, transform, train, download, path_suffix="cinic10", mode=mode
        )

    def _get_file_paths(self):
        """Get paths and labels for all relevant images"""
        data = []
        labels = []
        glob_pattern = "*.png"

        if self.train:
            for label, cls in enumerate(self.classes):
                # Add training files
                train_files = list(self.path.joinpath("train", cls).glob(glob_pattern))
                data.extend(train_files)
                labels.extend([label] * len(train_files))

                # Add validation files if requested
                if self.mix_valid_and_train:
                    valid_files = list(
                        self.path.joinpath("valid", cls).glob(glob_pattern)
                    )
                    data.extend(valid_files)
                    labels.extend([label] * len(valid_files))
        else:
            for label, cls in enumerate(self.classes):
                test_files = list(self.path.joinpath("test", cls).glob(glob_pattern))
                data.extend(test_files)
                labels.extend([label] * len(test_files))

        return data, labels

    def _create_arrow_dataset(self, arrow_path: Path):
        """Create a memory-mapped Arrow dataset"""

        def load_image(example):
            """Load image binary data for Arrow storage"""
            with open(example["image_path"], "rb") as f:
                binary = f.read()
            return {"image": {"bytes": binary, "path": example["image_path"]}}

        # Collect data for each split
        splits_data = {"train": [], "valid": [], "test": []}

        for label, cls in enumerate(self.classes):
            for split in splits_data.keys():
                img_files = list(self.path.joinpath(split, cls).glob("*.png"))
                if img_files:  # Only add if files exist
                    splits_data[split].append(
                        {
                            "image_path": [str(p) for p in img_files],
                            "label": [label] * len(img_files),
                        }
                    )

        # Combine data for each split
        dataset_dict = {}
        for split, split_data in splits_data.items():
            if not split_data:  # Skip empty splits
                continue

            # Combine all classes for this split
            paths = []
            labels = []
            for d in split_data:
                paths.extend(d["image_path"])
                labels.extend(d["label"])

            # Create dataset and convert images to binary
            dataset = Dataset.from_dict({"image_path": paths, "label": labels})
            dataset = dataset.map(
                load_image,
                num_proc=4,
                remove_columns=["image_path"],
                desc=f"Processing {split} split",
            )
            dataset_dict[split] = dataset

        # Save to disk
        arrow_dataset = DatasetDict(dataset_dict)
        arrow_dataset.save_to_disk(str(arrow_path))

    def _load_into_memory(self):
        """Load all images into memory as tensors"""
        paths, labels = self._get_file_paths()
        images = []

        for path in paths:
            try:
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except Exception as e:
                logging.error(f"Error loading image {path}: {e}")
                continue

        if not images:
            raise RuntimeError("No valid images found in the dataset")

        return torch.stack(images), torch.tensor(labels)

    @property
    def classes(self):
        """Get list of class names"""
        return sorted([p.name for p in (self.path.joinpath("train").iterdir())])

    def _download(self):
        """Download and extract the CINIC10 dataset"""
        if not self.path.exists():
            self.path.mkdir(parents=True)

            # Download the dataset
            url = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
            download_path = str(self.root.joinpath("CINIC-10.tar.gz"))

            logging.info(f"Downloading CINIC10 dataset to {download_path}")
            subprocess.run(["wget", "-P", str(self.root), url], check=True)

            logging.info("Extracting dataset...")
            subprocess.run(
                ["tar", "-xvf", download_path, "--xform=s|^|cinic10/|S"], check=True
            )

            # Clean up downloaded archive
            Path(download_path).unlink()
            logging.info("Dataset downloaded and extracted successfully")

    def __repr__(self):
        return (
            f"CINIC10(train={self.train}, "
            f"mix_valid_and_train={self.mix_valid_and_train}, "
            f"mode={self.mode}, "
            f"transform={self.transform})"
        )


class CINIC10_WO_CIFAR10(CINIC10):
    """CINIC10 dataset without CIFAR10 images"""

    def _get_file_paths(self):
        """Override to exclude CIFAR10 images"""
        data = []
        labels = []
        glob_pattern = "[!cifar10]*.png"  # Exclude files containing 'cifar10'

        if self.train:
            for label, cls in enumerate(self.classes):
                train_files = list(self.path.joinpath("train", cls).glob(glob_pattern))
                data.extend(train_files)
                labels.extend([label] * len(train_files))

                if self.mix_valid_and_train:
                    valid_files = list(
                        self.path.joinpath("valid", cls).glob(glob_pattern)
                    )
                    data.extend(valid_files)
                    labels.extend([label] * len(valid_files))
        else:
            for label, cls in enumerate(self.classes):
                test_files = list(self.path.joinpath("test", cls).glob(glob_pattern))
                data.extend(test_files)
                labels.extend([label] * len(test_files))

        return data, labels

    def _initialize_storage(self, arrow_path: str = "arrow_cinic10_wo_cifar10"):
        super()._initialize_storage(arrow_path)
