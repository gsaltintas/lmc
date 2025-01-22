import unittest
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset

from lmc.config import DataConfig
from lmc.experiment_config import Trainer
from lmc.utils.lmc_utils import repair
from lmc.utils.setup_training import configure_model, setup_loader
from tests.base import BaseTest


# Dummy dataset
class DummyDataset(Dataset):
    def __init__(self, size):
        self.data = torch.arange(size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestDataloader(BaseTest):
    def setUp(self):
        super().setUp()

    def _get_dataloader(
        self,
        hflip=False,
        random_rotation=0,
        random_translate=0,
        cutout=0,
        seed=99,
        num_workers=1,
    ):
        config = DataConfig(
            dataset="cifar10",
            path=self.data_dir / "cifar10",
            hflip=hflip,
            random_rotation=random_rotation,
            random_translate=random_translate,
            cutout=cutout,
            num_workers=num_workers,
        )
        dataloader = setup_loader(config, train=True, evaluate=False, loader_seed=seed)
        return dataloader

    def test_dataloader_consistency(self):
        def dataloaders_equal(**kwargs):
            dataloader1 = self._get_dataloader(**kwargs)
            dataloader2 = self._get_dataloader(**kwargs)
            dataloader3 = self._get_dataloader(**kwargs)
            # Compare the data from all DataLoader instances
            for ind, ((x1, y1), (x2, y2), (x3, y3)) in enumerate(
                zip(dataloader1, dataloader2, dataloader3)
            ):
                self.assertTrue(torch.equal(x1, x2), "DataLoader outputs differ")
                self.assertTrue(torch.equal(x1, x3), "DataLoader outputs differ")
                self.assertTrue(torch.equal(y1, y2), "DataLoader outputs differ")
                self.assertTrue(torch.equal(y1, y3), "DataLoader outputs differ")
                if ind > 4:
                    break

        with self.subTest("num_workers = 1"):
            dataloaders_equal(num_workers=1)
        with self.subTest("num_workers = 4"):
            dataloaders_equal(num_workers=4)

    def _batch_allclose_to_ref(self, dataloader):
        images, labels = next(iter(dataloader))
        # compare with saved images
        ref = np.load(Path("tests") / "data_batch.npz")
        ref_images, ref_labels = ref["images"], ref["labels"]
        return np.allclose(images, ref_images) and np.allclose(labels, ref_labels)

    def test_data_transforms(self):
        # check that none of these fail
        next(iter(self._get_dataloader()))
        next(iter(self._get_dataloader(hflip=True)))
        next(iter(self._get_dataloader(random_rotation=10)))
        next(iter(self._get_dataloader(random_translate=4)))
        next(iter(self._get_dataloader(cutout=2)))
        # check that data is same as stored reference batch
        self.assertTrue(
            self._batch_allclose_to_ref(
                self._get_dataloader(
                    hflip=True, random_rotation=10, random_translate=4, cutout=2
                )
            )
        )
        self.assertFalse(
            self._batch_allclose_to_ref(
                self._get_dataloader(
                    hflip=True,
                    random_rotation=10,
                    random_translate=4,
                    cutout=2,
                    seed=98,
                )
            )
        )

    def _get_running_stats(self, model):
        stats = {}
        for k, v in model.state_dict().items():
            if "running_mean" in k or "running_var" in k:
                stats[k] = v.detach().clone()
        return stats

    def test_repair(self):
        config = Trainer.from_dict(
            dict(
                n_models=1,
                training_steps="1ep",
                model_name="resnet20-8",
                dataset="cifar10",
                path=self.data_dir / "cifar10",
                norm="batchnorm",
            )
        )
        config.model.norm = "batchnorm"  # TODO hack as from_dict doesn't set this, see tests in test_config.py
        model, _ = configure_model(config, device="cpu", seed=41)
        init_stats = self._get_running_stats(model)
        default_stats = {
            k: torch.zeros_like(v) if "mean" in k else torch.ones_like(v)
            for k, v in init_stats.items()
        }
        self.assertTrue(self.state_dicts_equal(init_stats, default_stats))
        # check that repair changes the running stats
        model = repair(model, self._get_dataloader())
        ref_stats = self._get_running_stats(model)
        self.assertFalse(self.state_dicts_equal(init_stats, ref_stats))
        # check that running repair twice gives same stats
        model = repair(model, self._get_dataloader())
        stats = self._get_running_stats(model)
        self.assertTrue(self.state_dicts_equal(ref_stats, stats))

        # check first norm layer's outputs
        model.eval()
        (_, conv), (_m, norm) = list(model.named_modules())[2:4]
        data = torch.cat([x for x, _ in self._get_dataloader()], dim=0)
        pre_activation = norm(conv(data))
        pre_activation = pre_activation.transpose(1, 0).reshape(
            pre_activation.shape[1], -1
        )
        norm_mean, norm_std = (
            torch.mean(pre_activation, dim=-1),
            torch.std(pre_activation, dim=-1),
        )
        for m, s in zip(norm_mean, norm_std):
            self.assertAlmostEqual(m.item(), 0, places=1)
            self.assertAlmostEqual(s.item(), 1, places=1)


if __name__ == "__main__":
    unittest.main()
