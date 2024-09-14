from collections import OrderedDict

import torch
from torch import nn

from .base_model import BaseModel
from .layers import get_layer, norm_layer_1d
from .type_declaration import Activations, Inits, Norms


class MLP(BaseModel):
    _name: str = "mlp"

    def __init__(
        self,
        hidden_dim: int,
        num_hidden_layers: int,
        input_dim: int,
        output_dim: int = ...,
        initialization_strategy: Inits = ...,
        act: Activations = ...,
        norm: Norms = ...,
    ) -> None:
        super().__init__(output_dim, initialization_strategy, act, norm)
        in_dims = [input_dim] + [hidden_dim] * num_hidden_layers
        out_dims = [hidden_dim] * num_hidden_layers + [output_dim]
        layers = OrderedDict()
        for i, (in_, out_) in enumerate(zip(in_dims, out_dims)):
            layers[f"linear{i}"] = nn.Linear(in_, out_)
            if i < num_hidden_layers:
                layers[f"act{i}"] = get_layer(act)()
                if norm is not None:
                    layers[f"norm{i}"] = norm_layer_1d(norm, out_)

        self.model = nn.Sequential(layers)

        # Initialize.
        self.reset_parameters(initialization_strategy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x.flatten(1))
        return y

    @staticmethod
    def get_model_from_code(
        model_code: str,
        output_dim: int,
        input_dim: int,
        initialization_strategy: str = ...,
        act: Activations = ...,
        norm: Norms = ...,
        hidden_dim: int = ...,
        depth: int = ...,
    ) -> "MLP":
        model_code = model_code.lower()
        if not MLP.is_valid_model_code(model_code):
            raise ValueError(f"{model_code} invalid.")
        try:
            hidden_dim, depth = model_code[4:].split("x")
        except ValueError:
            pass
        return MLP(
            hidden_dim=int(hidden_dim),
            num_hidden_layers=int(depth) - 1,
            input_dim=input_dim,
            output_dim=output_dim,
            initialization_strategy=initialization_strategy,
            act=act,
            norm=norm,
        )

    def permutation_spec(self, **kwargs):
        return super().permutation_spec(**kwargs)

