"""
Modified from Gregor Bachmann (Original: https://github.com/gregorbachmann/scaling_mlps/blob/main/data_utils/data_stats.py)
Date: August 1, 2023
"""

from enum import Enum
from typing import Callable, Dict

import numpy as np
from torchvision import datasets as D

from lmc.data.utils import TaskType

# Define all the relevant stats for the datasets to look up

# Number of samples
SAMPLE_DICT = {
    "imagenet21": 11801680,
    "imagenet": 1281167,
    "tinyimagenet": 100000,
    "cifar10": 50000,
    "cifar100": 50000,
    "mnist": 60000,
    "stl10": 5000,
    "cinic10": 180000,
    "cinic10_wo_cifar10": 130000, #todo:dk
}

# Number of classes
CLASS_DICT = {
    "imagenet21": 11230,
    "imagenet": 1000,
    "tinyimagenet": 200,
    "cifar10": 10,
    "cifar100": 100,
    "mnist": 10,
    "stl10": 10,
    "cinic10": 10,
    "cinic10_wo_cifar10": 10,
}

# Number of channels
CHANNELS_DICT = {
    "imagenet21": 3,
    "imagenet": 3,
    "tinyimagenet": 3,
    "cifar10": 3,
    "cifar100": 3,
    "mnist": 1,
    "stl10": 3,
    "cinic10": 3,
    "cinic10_wo_cifar10": 3,
}

# Image resolutions
DEFAULT_RES_DICT = {
    "imagenet21": 64,
    "imagenet": 64,
    "tinyimagenet": 64,
    "cifar10": 32,
    "cifar100": 32,
    "mnist": 28,
    "stl10": 64,
    "cinic10": 32,
    "cinic10_wo_cifar10": 32,
}


# Parent directory name
DATA_DICT = {
    "imagenet21": "imagenet21",
    "imagenet": "imagenet",
    "tinyimagenet": "tiny-imagenet-200",
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "mnist": "mnist",
    "stl10": "stl10",
    "cinic10": "cinic10",
    "cinic10_wo_cifar10": "cinic10",
}

MODE_DICT = {
    "imagenet21": "test",
    "imagenet": "val",
    "tinyimagenet": "val",
    "cifar10": "val",
    "cifar100": "test",
    "mnist": "val",
    "stl10": "val",
    "cinic10": "test",
    "cinic10_wo_cifar10": "test",
}

# Standardization statistics
MEAN_DICT = {
    "imagenet21": np.array([0.485, 0.456, 0.406]),
    "imagenet": np.array([0.485, 0.456, 0.406]),
    "tinyimagenet": np.array([0.485, 0.456, 0.406]),
    "cifar10": np.array([0.49139968, 0.48215827, 0.44653124]),
    "cifar100": np.array([0.49139968, 0.48215827, 0.44653124]),
    "mnist": np.array([0.1307]),
    "stl10": np.array([0.4914, 0.4822, 0.4465]),
    "cinic10": np.array([0.47889522, 0.47227842, 0.43047404]),
    "cinic10_wo_cifar10": np.array([0.47889522, 0.47227842, 0.43047404]),
}


STD_DICT = {
    "imagenet21": np.array([0.229, 0.224, 0.225]),
    "imagenet": np.array([0.229, 0.224, 0.225]),
    "tinyimagenet": np.array([0.229, 0.224, 0.225]),
    "cifar10": np.array([0.24703233, 0.24348505, 0.26158768]), 
    "cifar100": np.array([0.24703233, 0.24348505, 0.26158768]),
    "mnist": np.array([0.3081]),
    "stl10": np.array([0.2471, 0.2435, 0.2616]),
    "cinic10": np.array([0.24205776, 0.23828046, 0.25874835]),
    "cinic10_wo_cifar10": np.array([0.24205776, 0.23828046, 0.25874835]),
}

# Whether dataset can be cached in memory, available in torch
OS_CACHED_DICT = {
    "imagenet21": False,
    "imagenet": False,
    "tinyimagenet": False,
    "cifar10": True,
    "cifar100": False,
    "mnist": True,
    "stl10": True,
    "pets": True,
    "cinic10": False,
    "cinic10_wo_cifar10": False
}


TORCH_DICT = {
    "imagenet": D.ImageNet,
    "tinyimagenet": "lmc.data.TinyImageNet",
    "cifar10": D.CIFAR10,
    "cifar100": D.CIFAR100,
    "mnist": D.MNIST,
    "stl10": D.STL10,
    "pets": D.OxfordIIITPet,
    "cinic10": "lmc.data.CINIC10",
    "cinic10_wo_cifar10": "lmc.data.CINIC10_WO_CIFAR10"
}

IS_GENERATION_TASK = {
    # Vision tasks (all False)
    "imagenet21": False,
    "imagenet": False,
    "tinyimagenet": False,
    "cifar10": False,
    "cifar100": False,
    "mnist": False,
    "stl10": False,
    "cinic10": False,
    "cinic10_wo_cifar10": False,
}

TASK_MAPPING = {
    ## vision
     "imagenet21": TaskType.CLASSIFICATION,
    "imagenet": TaskType.CLASSIFICATION,
    "tinyimagenet": TaskType.CLASSIFICATION,
    "cifar10": TaskType.CLASSIFICATION,
    "cifar100": TaskType.CLASSIFICATION,
    "mnist": TaskType.CLASSIFICATION,
    "stl10": TaskType.CLASSIFICATION,
    "cinic10": TaskType.CLASSIFICATION,
    "cinic10_wo_cifar10": TaskType.CLASSIFICATION,
    
}