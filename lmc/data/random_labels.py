from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from lmc.utils.seeds import temp_seed


class RandomLabelDataset(Dataset):
    """Wrapper dataset that randomizes labels of the base dataset."""

    def __init__(self, base_dataset: Dataset, random_seed: Optional[int] = None):
        self.base_dataset = base_dataset
        self.random_seed = random_seed

        # Store original attributes
        self.transform = getattr(base_dataset, "transform", None)
        self.targets = getattr(base_dataset, "targets", None)

        if self.targets is not None:
            with temp_seed(self.random_seed):
                # Get number of classes from the targets
                num_classes = len(set(self.targets))

                # Create random permutation for label mapping
                self.label_mapping = np.random.permutation(num_classes)

            # Apply permutation to targets
            self.shuffled_targets = [
                self.label_mapping[target] for target in self.targets
            ]
        else:
            raise ValueError("Dataset does not have 'targets' attribute")

    def __getitem__(self, index):
        img, _ = self.base_dataset[index]
        return img, self.shuffled_targets[index]

    def __len__(self):
        return len(self.base_dataset)
