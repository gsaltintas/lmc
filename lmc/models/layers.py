""" defines auxilary layers """

import torch
from torch import nn
from torch.nn.modules.batchnorm import _NormBase


class LayerNorm2d(nn.Module):
    """ 2d layernorm implementation for image data """
    def __init__(self, nchan, eps: float = 1e-7):
        # super().__init__(num_features=nchan, eps=eps)
        super().__init__()  
        self.channels = nchan
        self.weight = nn.Parameter(torch.ones(nchan), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(nchan), requires_grad=True)
        self.eps = eps

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        x = x - x.mean(1, keepdim=True)
        x = x / (x.std(1, keepdim=True, unbiased=False) + self.eps)
        x = x * self.weight.view(1, -1, 1, 1)
        x = x + self.bias.reshape(1, -1, 1, 1)
        return x
    
    def __repr__(self):
        return f"LayerNorm2d({self.channels})"

def norm_layer(norm: str, out_channels: int) -> nn.Module:
    """ given a norm string, returns the corresponding norm layer, pass norm=None for no norm, 
    out_channels specifies the dimension of the axis normalization occurs"""
    if "batch" in str(norm).lower():
        return nn.BatchNorm2d(out_channels)
    elif "group" in str(norm).lower():
        return nn.GroupNorm(1, out_channels)
    elif "layer" in str(norm).lower():
        return LayerNorm2d(out_channels)
    return nn.Sequential()

def is_norm_layer(layer: nn.Module) -> bool:
    """ checks if a layer is a norm layer """
    return isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm, LayerNorm2d))

def norm_layer_1d(norm: str, out_channels: int) -> nn.Module:
    """ given a norm string, returns the corresponding 1D norm layer, pass norm=None for no norm, 
    out_channels specifies the dimension of the axis normalization occurs"""
    if "batch" in str(norm).lower():
        return nn.BatchNorm1d(out_channels)
    elif "group" in str(norm).lower():
        return nn.GroupNorm(1, out_channels)
    elif "layer" in str(norm).lower():
        return nn.LayerNorm(out_channels)
    return nn.Sequential()


def get_layer(layer_name: str) -> nn.Module:
    """given a layer_name string returns the corresponding layer class, pass activations, etc."""
    layer_name_ = layer_name.lower()

    if layer_name_ == "linear":
        return nn.Linear
    elif layer_name_ == "relu":
        return nn.ReLU
    elif layer_name_ == "convd2":
        return nn.Conv2d
    elif layer_name_ == "layernorm":
        return LayerNorm2d
    elif layer_name_ == "batchnorm":
        return nn.BatchNorm2d
    elif layer_name_ == "groupnorm":
        return nn.GroupNorm
    elif layer_name_ == "dropout":
        return nn.Dropout
    else:
        try:
            return getattr(nn, layer_name)
        except:
            raise ValueError(f"Layer {layer_name} not supported currently")
