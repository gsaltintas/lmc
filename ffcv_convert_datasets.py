import os
from torchvision.datasets import CIFAR10, CIFAR100
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField


data_dir = os.environ.get("DATASET_DIR")

# Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs
for name, data_cls in [("cifar10", CIFAR10), ("cifar100", CIFAR100)]:
    for split in ["train", "test"]:
    my_dataset = data_cls(f"{data_dir}/{name}", train=(split=="train"), download=False)
    os.makedirs(f"{data_dir}/ffcv/{name}", exist_ok=True)
    write_path = f"{data_dir}/ffcv/{name}/{split}.beton"

    # Pass a type for each data field
    writer = DatasetWriter(write_path, {
        # Tune options to optimize dataset size, throughput at train-time
        'image': RGBImageField(max_resolution=256),
        'label': IntField()
    })

    # Write dataset
    writer.from_indexed_dataset(my_dataset)
