import os
import subprocess
from pathlib import Path
from typing import Union

from PIL import Image
from torch.utils.data import Dataset


def download_tinyimagenet(root: Union[str, Path] = "data"):
    root = Path(root).resolve()

    # Download data from Stanford source
    subprocess.run(["wget", "http://cs231n.stanford.edu/tiny-imagenet-200.zip"])

    # Unzip the downloaded file
    subprocess.run(["unzip", "-qq", "tiny-imagenet-200.zip"])

    # Define data directories
    DATA_DIR = Path("tiny-imagenet-200")
    TRAIN_DIR = DATA_DIR.joinpath("train")
    VALID_DIR = DATA_DIR.joinpath("val")
   


class TinyImageNet(Dataset):
    def __init__(self, root, transform=None, train=True, download=False):
        self.root = Path(root).resolve()
        p = self.root.joinpath("tinyimagenet")
        if p.exists():
            self.path = p
        else:
            self.path = self.root.joinpath("tiny-imagenet-200")
        self.transform = transform
        self.train = train
        self.data = []
        self.labels = []
        
        if download:
            self._download()
        self.classes = [p.name for p in (self.path.joinpath("train").iterdir())]

        if self.train:
            self.data_dir = self.path.joinpath("train")
            for label, cls in enumerate(self.classes):
                img_dir = os.path.join(self.data_dir, cls, "images")
                img_files = os.listdir(img_dir)
                for img_file in img_files:
                    img_path = os.path.join(img_dir, img_file)
                    self.data.append(img_path)
                    self.labels.append(label)
        else:
            self.data_dir = self.path.joinpath("val")
            with open(self.path.joinpath("val", "val_annotations.txt"), "r") as f:
                for line in f:
                    img_file, cls = line.split("\t")[:2]
                    img_path = self.data_dir.joinpath("images", img_file)
                    label = self.classes.index(cls)
                    self.data.append(img_path)
                    self.labels.append(label)
                    
    def _download(self):
        if not self.path.exists():
            self.path.mkdir(parents=True)
            subprocess.run(["wget", "-P", self.root, "http://cs231n.stanford.edu/tiny-imagenet-200.zip"])
            subprocess.run(["unzip", "-qq", "-d", self.root, self.root.joinpath("tiny-imagenet-200.zip")])
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
