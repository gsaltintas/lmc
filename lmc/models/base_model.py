from collections import OrderedDict
import logging
from abc import abstractmethod
from copy import deepcopy
from typing import Dict, Optional, get_args

import numpy as np
import torch
from torch import nn

from lmc.permutations import (PermSpec, PermType, get_permutation_sizes,
                              get_random_permutation_with_fixed_points,
                              permute_model, permute_state_dct)
from lmc.utils.utils import pattern_matched

from .layers import LayerNorm2d
from .type_declaration import PATTERNS, Activations, Inits, Norms

__all__ = ["BaseModel"]



INIT_STRATEGIES = get_args(get_args(Inits)[0])

class BaseModel(nn.Module):
    _name: str = None
    is_language_model: bool = False

    def __init__(
        self,
        output_dim: int = ...,
        initialization_strategy: Inits = ...,
        act: Activations = "relu",
        norm: Norms = None,
    ) -> None:
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model: Optional[nn.Module] = None
        if (
            initialization_strategy is not None
            and initialization_strategy not in INIT_STRATEGIES
        ):
            raise ValueError(
                f"Invalid initialization strategy {initialization_strategy}. Choose from {INIT_STRATEGIES}."
            )
        self.initialization_strategy = initialization_strategy
        self.output_dim = output_dim
        self.act = act
        self.norm = norm

    @classmethod
    @abstractmethod
    def is_valid_model_code(cls, model_code: str) -> bool:
        """Is the model name string a valid name for models in this class?"""
        match = pattern_matched(model_code, PATTERNS.get(cls._name, []))
        return match

    @staticmethod
    @abstractmethod
    def get_model_from_code(model_code: str, output_dim: int, **kwargs) -> "BaseModel":
        """Returns an instance of this class as described by the model_code string."""
        pass

    def reset_parameters(self, init_strategy: str = None):
        init_strategy = (
            self.initialization_strategy if init_strategy is None else init_strategy
        )
        if init_strategy == "pretrained":
            return
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                if init_strategy in ("xavier_uniform", "glorot_uniform"):
                    nn.init.xavier_uniform_(m.weight)
                elif init_strategy in ("xavier_normal", "glorot_normal"):
                    nn.init.xavier_normal_(m.weight)
                elif init_strategy in ("kaiming_uniform", "he_uniform"):
                    nn.init.kaiming_uniform_(m.weight)
                elif init_strategy in ("kaiming_normal", "he_normal"):
                    nn.init.kaiming_normal_(m.weight)
                else:
                    init_strategy = "default"
                    m.reset_parameters()
                self.logger.debug("%s initialized with %s", name, init_strategy)

            elif isinstance(m, LayerNorm2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise ValueError(".model should be set in the subclasses.")
        return self.model(x)

    def _permute(self, perms: PermType, inplace: bool = True, **kwargs):
        """permutes the parameters of the model according to the perms dict and its permutation_spec, if inplace is True, the model is modified, otherwise a new model is returned"""
        permuted_dct = permute_state_dct(self.model.state_dict(), self.permutation_spec(), perms=perms)
        model_ = self
        if not inplace:
            model_ = deepcopy(self)
        model_.model.load_state_dict(permuted_dct, strict=not self.is_language_model)
        return model_
        return permute_model(self.model, self.permutation_spec(), perms, inplace)

    @abstractmethod
    def permutation_spec(self, **kwargs) -> PermSpec:
        raise NotImplementedError("Must be implemented by subclasses.")

    def get_random_permutation(
        self, fixed_points_fraction: Optional[float] = 0, **permutation_spec_kwargs
    ) -> PermType:
        """returns a random permutation for each parameter of the model according to its permutation_spec, the permutation is a dict of numpy arrays, each array is a permutation of the corresponding parameter"""
        spec = self.permutation_spec(**permutation_spec_kwargs)
        if fixed_points_fraction > 0:
            perms = self._get_random_permutation_with_fixed_points(fixed_points_fraction)
        else:
            sizes = get_permutation_sizes(self.model.state_dict(), spec)
            perms = {p: np.random.permutation(n) for p, n in sizes.items()}
        return perms
    
    def _get_random_permutation_with_fixed_points(
        self, fixed_points_fraction: float, **permutation_spec_kwargs
    ) -> Dict[str, np.array]:
        # TODO: maybe just discontinue
        spec = self.permutation_spec(**permutation_spec_kwargs)
        sizes = get_permutation_sizes(self.model.state_dict(), spec)
        perms = {
            p: get_random_permutation_with_fixed_points(n, fixed_points_fraction)
            for p, n in sizes.items()
        }
        return perms

    @property
    def device(self):
        return next(self.parameters()).device

    @abstractmethod
    def get_init_stds(self, include_constant_params=False) -> OrderedDict[str, float]:
        """Returns dict of per-layer standard deviations for each parameter in the layer at initialization.
        The expected L2 norm of each layer is thus standard deviation * sqrt(number of parameters)"""
        # heuristic approach based on pytorch defaults
        std = OrderedDict()
        # work backwards so that we have std of next layer (assuming parameters are in order)
        last_randinit_layer = None
        last_randinit_inputs = 0
        for k, v in reversed(list(self.named_parameters())):
            # bias or norm weight/bias
            if k.endswith(".bias") or (k.endswith(".weight") and v.dim() == 1):
                # if we need to assign some noise to a constant layer, use the same scaling as the next non-constant layer
                # if this is the last layer, use 1/sqrt(n) as scaling instead
                if include_constant_params:
                    if last_randinit_layer is None:
                        self.logger.info("{k} has constant init but is not followed by another layer, assume 1/sqrt(n) std")
                        std[k] = 1 / v.shape[0]**0.5
                    else:
                        # check that output dims matches input dims of weight layer
                        assert last_randinit_inputs == v.shape[0]
                        std[k] = std[last_randinit_layer]
                else:
                    std[k] = 0.0
            # conv or linear weight
            elif k.endswith(".weight") and v.dim() > 1:
                if self.initialization_strategy in ("xavier_uniform", "glorot_uniform"):
                    std[k] = xavier_uniform_std(v)
                elif self.initialization_strategy in ("xavier_normal", "glorot_normal"):
                    std[k] = xavier_normal_std(v)
                elif self.initialization_strategy in ("kaiming_uniform", "he_uniform"):
                    std[k] = kaiming_uniform_std(v)
                elif self.initialization_strategy in ("kaiming_normal", "he_normal"):
                    std[k] = kaiming_normal_std(v)
                # ignore shortcut layers when saving next input layer
                if "shortcut" not in k:
                    last_randinit_layer = k
                    last_randinit_inputs = v.shape[1]
            else:
                raise ValueError(f"Unknown layer type and std: name={k} shape={v.shape}")
        reversed_std = OrderedDict()
        for k, v in reversed(std.items()):
            reversed_std[k] = v
        return reversed_std


def fan_in(tensor):
    k = 1
    # multiply everything except output size, which is first dim
    for d in tensor.shape[1:]:
        k *= d
    return k


def xavier_uniform_std(tensor) -> float:
    raise NotImplementedError()


def xavier_normal_std(tensor) -> float:
    raise NotImplementedError()


def kaiming_uniform_std(tensor) -> float:
    # kaiming uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
    # std is 1/sqrt(12)*(b-a), where b-a = 2/sqrt(k)
    return 2 / (12 * fan_in(tensor))**0.5


def kaiming_normal_std(tensor) -> float:
    return (2 / fan_in(tensor))**0.5
