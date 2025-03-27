import os
from pathlib import Path
from typing import Any, Tuple
import numpy as np

from PIL import Image
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset


class CIFAR10_SPLIT(Dataset):
    def __init__(self,
        score_file,
        easy_split,
        root,
        train=True,
        transform=None,
        download=False
    ):
        root = root.parent / "cifar10"
        dataset = CIFAR10(root=root, train=train, transform=transform, download=download)
        self.root = dataset.root
        self.transform = dataset.transform
        self.target_transform = dataset.target_transform
        self.train = dataset.train
        self.download = dataset.download
        if self.train:
            idx = self._get_difficulty_argsort(score_file)
        else:
            idx = np.arange(len(dataset.targets))

        split_point = len(dataset) // 2
        split = idx[:split_point] if easy_split else idx[split_point:]

        self.data = dataset.data[split]
        self.targets = [dataset.targets[i] for i in split]

    @staticmethod
    def _get_difficulty_argsort(score_file):
        file_path = Path(os.path.realpath(__file__))
        difficulty_stats_path = file_path.parent / "difficulty_scores" / score_file
        scores = np.load(difficulty_stats_path)
        if not isinstance(scores, np.ndarray):  # npz file is key:value based, take "arr_0" as key
            scores = scores["arr_0"]
        return np.argsort(scores)

    """Following is copied from torchvision.datasets.CIFAR10"""

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class CIFAR10_EL2N_EASY(CIFAR10_SPLIT):
    def __init__(self, root, train=True, transform=None, download=False):
        super().__init__("el2n_20ep0it.npz", True, root, train, transform, download)


class CIFAR10_EL2N_HARD(CIFAR10_SPLIT):
    def __init__(self, root, train=True, transform=None, download=False):
        super().__init__("el2n_20ep0it.npz", False, root, train, transform, download)
