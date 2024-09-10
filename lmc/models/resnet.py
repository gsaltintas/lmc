from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Literal

from torch import nn
from torch.nn import functional as F

from .base_model import INIT_STRATEGIES, BaseModel
from .layers import LayerNorm2d, norm_layer

plan_mapping: Dict[int, List[int]] = {
    18: [2,2,2,2],
    34: [3,4,6,3],
    20: [3, 3, 3],
}

class ResNet(BaseModel):
    _name: str = "resnet"

    """A residual neural network as originally designed for CIFAR-10."""

    class Block(nn.Module):
        """A ResNet block."""

        def __init__(self, f_in: int, f_out: int, downsample=False, norm:str = "layernorm" ):
            super(ResNet.Block, self).__init__()

            stride = 2 if downsample else 1
            self.conv1 = nn.Conv2d(f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = norm_layer(norm, f_out)
            self.conv2 = nn.Conv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = norm_layer(norm, f_out)

            # No parameters for shortcut connections.
            if downsample or f_in != f_out:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                    norm_layer(norm, f_out)
                )
            else:
                self.shortcut = nn.Sequential()

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return F.relu(out)


    def __init__(self, plan, output_dim: int = ..., initialization_strategy: str = ..., norm: str = "layernorm", width: int = 1) -> None:
        super().__init__(output_dim, initialization_strategy, act="relu", norm=norm)
        # Initial convolution.
        plan = [(width * 2**(i), n) for i, n in enumerate(plan)]
        current_filters = plan[0][0]
        self.conv = nn.Conv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = norm_layer(norm, current_filters)

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (out_filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(ResNet.Block(current_filters, out_filters, downsample, norm=norm))
                current_filters = out_filters

        self.blocks = nn.Sequential(*blocks)

        self.fc = nn.Linear(current_filters, output_dim)
        self.reset_parameters(initialization_strategy)

    @staticmethod
    def get_model_from_code(model_code: str, output_dim: int = ..., initialization_strategy: str = ..., norm: str = "layernorm") -> 'ResNet':
        model_code = model_code.lower()
        if not ResNet.is_valid_model_code(model_code):
            raise ValueError(f"{model_code} invalid.")
        
        dw = model_code.split("-")
        depth = int(dw[0][6:])
        width = 16 if len(dw) == 1 else int(dw[1])
        plan = plan_mapping.get(depth, [])
        return ResNet(plan=plan, output_dim=output_dim, initialization_strategy=initialization_strategy, width=width, norm=norm)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = self.fc(out.flatten(1))
        return out
    
    def permutation_spec(self, **kwargs):
        # TODO
        return super().permutation_spec(**kwargs)
