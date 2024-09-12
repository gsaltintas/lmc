import subprocess

from lmc.data.dataset import MyDataset

""" implements CINIC10 dataset, for more information, refer to https://paperswithcode.com/dataset/cinic-10 """

class CINIC10(MyDataset):
    def __init__(self, root, transform=None, train=True, download=False, mix_valid_and_train:bool=False):
        super().__init__(root, transform, train, download, path_suffix="cinic10")
        self.mix_valid_and_train = mix_valid_and_train
        glob_pattern = "*.png"
        
        if self.train:
            for label, cls in enumerate(self.classes):
                img_files = list(self.path.joinpath("train", cls).glob(glob_pattern))
                if self.mix_valid_and_train:
                    img_files += list(self.path.joinpath("valid", cls).glob(glob_pattern))
                self.data.extend(img_files)
                self.labels.extend([label] * len(img_files))
        else:
            for label, cls in enumerate(self.classes):
                img_files = list(self.path.joinpath("test", cls).glob(glob_pattern))
                self.data.extend(img_files)
                self.labels.extend([label] * len(img_files))
    
    @property
    def classes(self):
        return [p.name for p in (self.path.joinpath("train").iterdir())]
    
    def _download(self):
        if not self.path.exists():
            self.path.mkdir(parents=True)
            subprocess.run(["wget", "-P", self.root, "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"])
            subprocess.run(["tar", "-xvf", self.root.joinpath("CINIC-10.tar.gz"), " --xform='s|^|cinic10/|S'" ])
                    

class CINIC10_WO_CIFAR10(CINIC10):
    
    # The original CIFAR-10 data was processed into image format (.png) and stored as follows:  [$set$/$class_name$/cifar-10-$origin$-$index$.png]
    
    def __init__(self, root, transform=None, train=True, download=False, mix_valid_and_train: bool = False):
        super().__init__(root, transform, train, download, mix_valid_and_train)
        glob_pattern = "[!cifar10]*.png"
        
        if self.train:
            for label, cls in enumerate(self.classes):
                img_files = list(self.path.joinpath("train", cls).glob(glob_pattern))
                if self.mix_valid_and_train:
                    img_files += list(self.path.joinpath("valid", cls).glob(glob_pattern))
                self.data.extend(img_files)
                self.labels.extend([label] * len(img_files))
        else:
            for label, cls in enumerate(self.classes):
                img_files = list(self.path.joinpath("test", cls).glob(glob_pattern))
                self.data.extend(img_files)
                self.labels.extend([label] * len(img_files))