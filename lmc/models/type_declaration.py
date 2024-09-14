""" defines patterns for different model codes """
from typing import Literal, Optional

import numpy as np

PATTERNS = {
    "vit": [r"vit[a-zA-Z]*/(\d+)-(\d+)-(\d+)x(\d+)-(\d+)"],
    "cnn": [
        r"cnn[a-zA-Z]*/(\d+)xk=(\d+)-s=(\d+)-d=(\d+)-fc=(\d+)-p=(\d+)",
        r"cnn[a-zA-Z]*/(\d+)xk=(\d+)-s=(\d+)-d=(\d+)-fc=(\d+)",
    ],
    "lcn": [
        r"lcn[a-zA-Z]*/(\d+)xk=(\d+)-s=(\d+)-d=(\d+)-fc=(\d+)-p=(\d+)",
        r"lcn[a-zA-Z]*/(\d+)xk=(\d+)-s=(\d+)-d=(\d+)-fc=(\d+)",
    ],
    "vgg": [r"vgg[a-zA-Z]*(\d+)"],
    "mlp": [r"mlp/(\d+)x(\d+)", r"mlp"],
    "resnet": [
        r"resnet",
        r"simple-resnet",
        r"ffcv-resnet[a-zA-Z]*-(\d+)" r"linear/(\d+)x(\d+)",
        r"resnet[a-zA-Z]*(\d+)-(\d+)",
        r"wideresnet[a-zA-Z]*/(\d+)xk=(\d+)-s=(\d+)-d=(\d+)-fc=(\d+)",
    ],
}
MODEL_NAME_PATTERNS = np.concatenate(list(PATTERNS.values()))
Inits = Optional[
    Literal[
        "xavier_uniform",
        "glorot_uniform",
        "xavier_normal",
        "glorot_normal",
        "kaiming_uniform",
        "he_uniform",
        "kaiming_normal",
        "he_normal",
    ]
]
Norms = Optional[Literal["layernorm", "batchnorm", "groupnorm"]]
Activations = Literal["relu", "linear", "elu", "gelu", "leaky_relu"]
