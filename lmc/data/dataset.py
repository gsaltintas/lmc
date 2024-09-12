from abc import abstractmethod
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

"""base dataset class"""

class MyDataset(Dataset):
    def __init__(self, root, transform=None, train=True, download=False, path_suffix: str = ""):
        self.root = Path(root).resolve()
        self.path = self.root.joinpath(path_suffix)
        self.transform = transform
        self.train = train
        self.data = []
        self.labels = []
        
        if download:
            self._download()
    
    @property
    @abstractmethod
    def classes(self):
        pass
    
    @abstractmethod
    def _download(self):
        raise NotImplementedError("Must be implemented by subclasses")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
