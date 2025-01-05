import os
import unittest
from pathlib import Path
from shutil import rmtree

import torch
from torch.utils.data import DataLoader, Dataset

from lmc.config import DataConfig
from lmc.experiment_config import Trainer
from lmc.utils.setup_training import setup_loader


# Dummy dataset
class DummyDataset(Dataset):
    def __init__(self, size):
        self.data = torch.arange(size)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class TestDataLoaderConsistency(unittest.TestCase):
    @classmethod
    def setUp(cls):
        DATA_PATH_ENV_VAR = "DATASET_DIR"
        data_dir = os.environ.get(DATA_PATH_ENV_VAR)
        if data_dir is None:
            raise ValueError(
                f"Need to set the environment variable {DATA_PATH_ENV_VAR}"
            )
        cls.conf = DataConfig(dataset="cifar10", path=data_dir + "/cifar10", hflip=True, random_rotation=10)
        cls.loader_seed = 43

    # @classmethod
    # def tearDown(cls) -> None:
    #     rmtree(cls.conf.path)

    def test_dataloader_consistency(self):
        self.conf.num_workers = 1
        dataloader1 = setup_loader(self.conf, train=True, evaluate=False, loader_seed=self.loader_seed)
        dataloader2 = setup_loader(self.conf, train=True, evaluate=False, loader_seed=self.loader_seed)
        dataloader3 = setup_loader(self.conf, train=True, evaluate=False, loader_seed=self.loader_seed)
        
        # Compare the data from all DataLoader instances
        for ind, ((x1, y1), (x2, y2), (x3, y3)) in enumerate(zip(dataloader1, dataloader2, dataloader3)):
            self.assertTrue(torch.equal(x1, x2), "DataLoader outputs differ")
            self.assertTrue(torch.equal(x1, x3), "DataLoader outputs differ")
            self.assertTrue(torch.equal(y1, y2), "DataLoader outputs differ")
            self.assertTrue(torch.equal(y1, y3), "DataLoader outputs differ")

            if ind > 4:
                break

    def test_dataloader_consistency_num_workers(self):
        self.conf.num_workers = 4
        dataloader1 = setup_loader(self.conf, train=True, evaluate=False, loader_seed=self.loader_seed)
        dataloader2 = setup_loader(self.conf, train=True, evaluate=False, loader_seed=self.loader_seed)
        dataloader3 = setup_loader(self.conf, train=True, evaluate=False, loader_seed=self.loader_seed)
        
        # Compare the data from all DataLoader instances
        for ind, ((x1, y1), (x2, y2), (x3, y3)) in enumerate(zip(dataloader1, dataloader2, dataloader3)):
            self.assertTrue(torch.equal(x1, x2), "DataLoader outputs differ")
            self.assertTrue(torch.equal(x1, x3), "DataLoader outputs differ")
            self.assertTrue(torch.equal(y1, y2), "DataLoader outputs differ")
            self.assertTrue(torch.equal(y1, y3), "DataLoader outputs differ")

            if ind > 4:
                break

if __name__ == "__main__":
    unittest.main()
