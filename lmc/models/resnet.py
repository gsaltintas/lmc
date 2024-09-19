from collections import OrderedDict
from typing import Dict, List

from torch import nn
from torch.nn import functional as F

from lmc.utils.permutations import PermSpec

from .base_model import BaseModel
from .layers import norm_layer

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
            self.norm1 = norm_layer(norm, f_out)
            self.act1 = nn.ReLU()
            self.conv2 = nn.Conv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False)
            self.norm2 = norm_layer(norm, f_out)
            self.act2 = nn.ReLU()

            # No parameters for shortcut connections.
            if downsample or f_in != f_out:
                self.shortcut = nn.Sequential(
                    OrderedDict(conv=nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                    # make sure this is named as norm
                    norm=norm_layer(norm, f_out))
                )
            else:
                self.shortcut = nn.Sequential()

        def forward(self, x):
            out = self.act1(self.norm1(self.conv1(x)))
            out = self.norm2(self.conv2(out))
            out += self.shortcut(x)
            return self.act2(out)


    def __init__(self, plan, output_dim: int = ..., initialization_strategy: str = ..., norm: str = "layernorm", width: int = 1) -> None:
        super().__init__(output_dim, initialization_strategy, act="act", norm=norm)
        # Initial convolution.
        plan = [(width * 2**(i), n) for i, n in enumerate(plan)]
        self.plan = plan
        current_filters = plan[0][0]
        conv = nn.Conv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        norm = norm_layer(norm, current_filters)

        # The subsequent blocks of the ResNet.
        blocks = OrderedDict()
        for segment_index, (out_filters, num_blocks) in enumerate(plan):
            block = []
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                block.append(ResNet.Block(current_filters, out_filters, downsample, norm=norm))
                current_filters = out_filters
            blocks[f"block{segment_index}"] = nn.Sequential(*block)

        blocks = nn.Sequential(blocks)

        fc = nn.Linear(current_filters, output_dim)
        self.model = nn.Sequential(OrderedDict(conv=conv, norm=norm, act = nn.ReLU(), blocks=blocks, fc=fc))
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
        out = self.model.act(self.model.norm(self.model.conv(x)))
        out = self.model.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = self.model.fc(out.flatten(1))
        return out
    
    def _acts_to_perms(self):
        acts_to_perms = {"act": "P_0"}
        conv_block = lambda block_ind, ind, act_ind, perm: {
            f"blocks.block{block_ind}.{ind}.act{act_ind}": perm,
        }
        prev_perm: int = 0
        next_perm: int = 1
        for segment_index, (out_filters, num_blocks) in enumerate(self.plan):
            for block_index in range(num_blocks):
                acts_to_perms.update(conv_block(segment_index, block_index, 1, f"P_{next_perm}"))
                downsample = segment_index > 0 and block_index == 0
                if downsample:
                    next_perm += 1
                    acts_to_perms.update(conv_block(segment_index, block_index, 2, f"P_{next_perm}"))
                    prev_perm = next_perm
                else:
                    acts_to_perms.update(conv_block(segment_index, block_index, 2, f"P_{prev_perm}"))
                next_perm += 1

        return acts_to_perms
    
    def permutation_spec(self, **kwargs) -> PermSpec:
        names_to_perms = {"conv.weight": ("P_0",), "norm.weight": ("P_0", ),  "norm.bias": ("P_0", )}
        conv_block1 = lambda block_ind, ind, perm, inv_perm: {
            f"blocks.block{block_ind}.{ind}.conv1.weight": (perm, inv_perm),
            f"blocks.block{block_ind}.{ind}.norm1.weight": (perm, ),
            f"blocks.block{block_ind}.{ind}.norm1.bias": (perm, ),
        }
        conv_block2 = lambda block_ind, ind, perm, inv_perm: {
            f"blocks.block{block_ind}.{ind}.conv2.weight": (perm, inv_perm),
            f"blocks.block{block_ind}.{ind}.norm2.weight": (perm, ),
            f"blocks.block{block_ind}.{ind}.norm2.bias": (perm, ),
        }
        def shortcut_perms(block_ind, ind, perm, inv_perm): 
            if block_ind != 0 and ind == 0:
                return { 
                    f"blocks.block{block_ind}.{ind}.shortcut.conv.weight": (perm, inv_perm) ,
                    # make sure this goese to norm
                    f"blocks.block{block_ind}.{ind}.shortcut.norm.weight": (perm, ) ,
                    f"blocks.block{block_ind}.{ind}.shortcut.norm.bias": (perm, ) 
            # f""
        }
        prev_perm: int = 0
        next_perm: int = 1
        for segment_index, (out_filters, num_blocks) in enumerate(self.plan):
            for block_index in range(num_blocks):
                names_to_perms.update(conv_block1(segment_index, block_index, f"P_{next_perm}", f"P_{prev_perm}"))
                downsample = segment_index > 0 and block_index == 0
                if downsample:
                    next_perm += 1
                    names_to_perms.update(shortcut_perms(segment_index, block_index, f"P_{next_perm}", f"P_{prev_perm}"))
                    names_to_perms.update(conv_block2(segment_index, block_index, f"P_{next_perm}", f"P_{next_perm-1}"))
                    prev_perm = next_perm
                else:
                    names_to_perms.update(conv_block2(segment_index, block_index, f"P_{prev_perm}", f"P_{next_perm}"))
                next_perm += 1

        names_to_perms["fc.weight"] = (None, f"P_{prev_perm}")
        names_to_perms["fc.bias"] = (None, )
        return PermSpec(names_to_perms=names_to_perms, acts_to_perms=self._acts_to_perms())


    # def permutation_spec(self, permute_residuals: bool = False) -> Permutations:
    #     first_params = {
    #         "conv1.weight": ("P_0", None),
    #         "conv1.bias": ("P_0",),
    #         "norm1.weight": ("P_0",),
    #         "norm1.bias": ("P_0",),
    #     }
    #     conv_block = lambda layer, ind, perm, inv_perm, block_ind: {
    #         f"layer{layer}.{ind}.conv{block_ind}.weight": (perm, inv_perm),
    #         f"layer{layer}.{ind}.conv{block_ind}.bias": (perm,),
    #         f"layer{layer}.{ind}.norm{block_ind}.weight": (perm,),
    #         f"layer{layer}.{ind}.norm{block_ind}.bias": (perm,),
    #     }

    #     def shortcut_block(layer, shortcut_ind, perm, inv_perm):
    #         if shortcut_ind == 0 and layer != 1:
    #             return {
    #                 f"layer{layer}.0.shortcut.0.weight": (perm, inv_perm),
    #                 f"layer{layer}.0.shortcut.0.bias": (perm,),
    #                 f"layer{layer}.0.shortcut.1.weight": (perm,),
    #                 f"layer{layer}.0.shortcut.1.bias": (perm,),
    #             }
    #         else:
    #             if permute_residuals:
    #                 d = {
    #                     # f"layer{layer}.{shortcut_ind}.shortcut.0.indices": (perm, inv_perm) ,
    #                     f"layer{layer}.{shortcut_ind}.shortcut.0.input_indices": (inv_perm,),
    #                     f"layer{layer}.{shortcut_ind}.shortcut.0.output_indices": (perm,),
    #                 }
    #                 return d
    #             else:
    #                 # return { f"layer{layer}.{shortcut_ind}.shortcut.0.indices": (None,) }
    #                 d = {
    #                     # f"layer{layer}.{shortcut_ind}.shortcut.0.indices": (perm, inv_perm) ,
    #                     f"layer{layer}.{shortcut_ind}.shortcut.0.input_indices": (None,),
    #                     f"layer{layer}.{shortcut_ind}.shortcut.0.output_indices": (None,),
    #                 }
    #                 return d

    #     if self.config.layers:
    #         d = self.config.layers
    #         depth = len(d)
    #     else:
    #         d = [(self.config.depth - 2) // 6] * 3
    #         depth = 3
    #     layer_perms = {}
    #     curr_perm = 0
    #     block_inv_perm = curr_perm
    #     for i in range(depth):
    #         # block_perm = curr_perm + d[i] - 1
    #         for j in range(d[i]):
    #             curr_perm += 1
    #             if i > 0 and j == 0:
    #                 # later layers, later blocks, shortcut is a conv-norm layer
    #                 # todo: shortcut is a conv
    #                 layer_perms.update(conv_block(i + 1, j, f"P_{curr_perm}", f"P_{block_inv_perm}", 1))
    #                 if permute_residuals:
    #                     layer_perms.update(shortcut_block(i + 1, j, f"P_{curr_perm + 1}", f"P_{block_inv_perm}"))
    #                     block_inv_perm = curr_perm + 1
    #                     layer_perms.update(conv_block(i + 1, j, f"P_{block_inv_perm}", f"P_{curr_perm}", 2))
    #                     curr_perm += 1
    #                 else:
    #                     layer_perms.update(shortcut_block(i + 1, j, f"P_{curr_perm + 1}", f"P_{block_inv_perm}"))
    #                     block_inv_perm = curr_perm + 1
    #                     layer_perms.update(conv_block(i + 1, j, f"P_{block_inv_perm}", f"P_{curr_perm}", 2))
    #                     curr_perm += 1

    #             else:
    #                 # first layer, first block
    #                 layer_perms.update(conv_block(i + 1, j, f"P_{curr_perm}", f"P_{block_inv_perm}", 1))
    #                 # last block of any layer
    #                 if permute_residuals:
    #                     layer_perms.update(conv_block(i + 1, j, f"P_{curr_perm+1}", f"P_{curr_perm}", 2))
    #                     layer_perms.update(shortcut_block(i + 1, j, f"P_{curr_perm+1}", f"P_{block_inv_perm}"))
    #                     # layer_perms.update(shortcut_block(i + 1, j, f"P_{curr_perm+1}", None))
    #                     curr_perm += 1
    #                     block_inv_perm = curr_perm
    #                 else:
    #                     layer_perms.update(conv_block(i + 1, j, f"P_{block_inv_perm}", f"P_{curr_perm}", 2))
    #                     layer_perms.update(shortcut_block(i + 1, j, None, None))
    #     curr_perm = block_inv_perm
    #     names_to_perms = (
    #         first_params
    #         | layer_perms
    #         | {
    #             "fc.weight": (None, f"P_{curr_perm}"),
    #             "fc.bias": (None,),
    #         }
    #     )

    #     assert (
    #         len((s := set(names_to_perms)) - set(k := dict(self.model.named_parameters()).keys())) == 0
    #     ), f"Following keys appear in the permutations but not in the model parameters: {s - k}"
    #     assert len(k - s) == 0, f"Following keys were not encountered in the permutations: {k - s}"
    #     # names_to_perms
    #     return permutation_spec_from_names_to_perms(names_to_perms)
