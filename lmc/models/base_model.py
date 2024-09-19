import logging
from abc import abstractmethod
from copy import deepcopy
from typing import Dict, Optional, get_args

import numpy as np
import torch
from torch import nn

from lmc.models.layers import LayerNorm2d
from lmc.models.type_declaration import Activations, Inits, Norms
from lmc.utils.permutations import (PermSpec, PermType, get_permutation_sizes,
                                    get_random_permutation_with_fixed_points,
                                    permute_model, permute_state_dct)
from lmc.utils.utils import pattern_matched

from .type_declaration import PATTERNS

__all__ = ["BaseModel"]



INIT_STRATEGIES = get_args(get_args(Inits)[0])

class BaseModel(nn.Module):
    _name: str = None

    def __init__(
        self,
        output_dim: int = ...,
        initialization_strategy: Inits = ...,
        act: Activations = "relu",
        norm: Norms = None
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
        model_.model.load_state_dict(permuted_dct)
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