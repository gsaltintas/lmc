import unittest
import numpy as np
from pathlib import Path
from lmc.experiment_config import Trainer
from lmc.utils.setup_training import setup_loader
from tests.base import BaseTest


class TestDataTransforms(BaseTest):

    def _get_batch(self, seed=99):
        config = Trainer.from_dict(
            dict(
                path=self.data_dir / "cifar10",
                n_models=1,
                training_steps="1ep",
                model_name="resnet20-32",
                dataset="cifar10",
                hflip=True,
                random_rotation=10,
                random_translate=4,
                cutout=2,
            )
        )
        dl = setup_loader(config.data, train=True, evaluate=False, loader_seed=seed)
        images, labels = next(iter(dl))
        # compare with saved images
        ref = np.load(Path("tests") / "data_batch.npz")
        ref_images, ref_labels = ref["images"], ref["labels"]
        return np.allclose(images, ref_images) and np.allclose(labels, ref_labels)

    def test_data_transforms(self):
        self.assertTrue(self._get_batch())
        self.assertFalse(self._get_batch(seed=98))


if __name__ == "__main__":
    unittest.main()
