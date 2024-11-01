import sys
from pathlib import Path

import torch

sys.path.insert(0,Path(__file__).parents[1].resolve().absolute().as_posix())

from lmc.models import ResNet

if __name__ == "__main__":
    model = ResNet.get_model_from_code(model_code="resnet20-32", output_dim=10, norm="layernorm", initialization_strategy="kaiming_normal")
    ps = model.permutation_spec()
    perms = model.get_random_permutation()

    print(ps)

    model = model.to(torch.float64)
    model_ = model._permute(perms, inplace=False)
    x= torch.randn(33, 3, 32, 32, dtype=torch.float64)
    with torch.no_grad():
        assert torch.allclose(model(x), model_(x))
