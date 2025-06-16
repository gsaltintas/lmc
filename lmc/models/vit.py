import os
from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn
from transformers import ViTConfig, ViTForImageClassification

from lmc.permutations import PermSpec, PermType, permute_state_dct

from .base_model import BaseModel
from .type_declaration import PATTERNS


class VIT(BaseModel):
    _name: str = "VIT"
    is_language_model: bool = (
        False  # Changed from True to False as ViT is a vision model
    )
    model_name: str = None

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        output_dim: int = ...,
        initialization_strategy: str = "pretrained",
        norm: str = "layernorm",
        gradient_checkpointing: bool = True,
    ):
        self.model_name = model_name
        super().__init__(output_dim, initialization_strategy, act="act", norm=norm)
        kwargs = dict(token=os.environ.get("HF_TOKEN", None))
        if model_name in [
            "google/vit-base-patch16-224",
            "google/vit-large-patch16-224",
        ]:
            kwargs["from_flax"] = True
        else:
            kwargs["num_labels"] = output_dim
        if initialization_strategy == "pretrained":
            model = ViTForImageClassification.from_pretrained(
                model_name,
                **kwargs,
            )
            if output_dim != model.num_labels:
                new_classifier_head = nn.Linear(
                    in_features=model.classifier.in_features, out_features=output_dim
                )
                model.classifier = new_classifier_head
                import gc

                gc.collect()
                self.logger.warn(
                    "Model's classifier head reset, you probably need to finetune this model."
                )
            self.model = model
        else:
            raise ValueError(f"Vit from scratch is not implemented yet.")
            # Create a custom configuration
            config = ViTConfig.from_pretrained(model_name, num_labels=output_dim)
            self.model = ViTForImageClassification(
                config,
            )
            self.logger.info(
                f"ViT model completely initialized from scratch, if that's not the desired behavior pass --initialization_strategy=pretrained."
            )
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.d_head = (
            self.model.config.hidden_size // self.model.config.num_attention_heads
        )

    @classmethod
    def is_valid_model_code(cls, model_code: str) -> bool:
        return model_code.lower() in PATTERNS[cls._name]

    @staticmethod
    def get_model_from_code(model_code: str, output_dim: int, **kwargs) -> "BaseModel":
        return VIT(model_name=model_code, output_dim=output_dim, **kwargs)

    def forward(
        self,
        pixel_values: torch.Tensor,  # Changed from input_ids to pixel_values
        attention_mask: torch.Tensor = None,  # Optional for ViT
        labels: torch.Tensor = None,
        **kwargs,
    ):
        out = self.model(
            pixel_values=pixel_values[:128],  # Changed from input_ids to pixel_values
            head_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        return out.logits

    def permutation_spec(self, prefix: str = "", **kwargs) -> PermSpec:
        names_to_perms = OrderedDict()
        P0 = "P_0"

        # ViT embeddings
        names_to_perms["vit.embeddings.patch_embeddings.projection.weight"] = (P0, None)
        names_to_perms["vit.embeddings.patch_embeddings.projection.bias"] = (P0,)
        names_to_perms["vit.embeddings.cls_token"] = (None, None, P0)
        names_to_perms["vit.embeddings.position_embeddings"] = (None, None, P0)
        names_to_perms["vit.layernorm.weight"] = (P0,)
        names_to_perms["vit.layernorm.bias"] = (P0,)

        # For each transformer layer
        for i in range(self.num_layers):
            base = f"vit.encoder.layer.{i}"
            P_in = P0  # Input permutation

            # Head-level permutations
            P_head = f"P_head_{i}"  # Permute attention heads
            P_dhead = f"P_dhead_{i}"  # Permute within head dimension

            # Layer normalization before attention
            names_to_perms.update(
                {
                    f"{base}.layernorm_before.weight": (P_in,),
                    f"{base}.layernorm_before.bias": (P_in,),
                }
            )

            # Self-attention with head permutations
            attention = f"{base}.attention"

            # For query, key, value weights, handle both head and d_head dimensions
            names_to_perms.update(
                {
                    f"{attention}.attention.query.weight": ((P_head, P_dhead), P_in),
                    f"{attention}.attention.query.bias": ((P_head, P_dhead),),
                    f"{attention}.attention.key.weight": ((P_head, P_dhead), P_in),
                    f"{attention}.attention.key.bias": ((P_head, P_dhead),),
                    f"{attention}.attention.value.weight": ((P_head, P_dhead), P_in),
                    f"{attention}.attention.value.bias": ((P_head, P_dhead),),
                }
            )

            # Attention output combining head permutations
            names_to_perms.update(
                {
                    f"{attention}.output.dense.weight": (P_in, (P_head, P_dhead)),
                    f"{attention}.output.dense.bias": (P_in,),
                }
            )

            # Layer normalization before MLP
            names_to_perms.update(
                {
                    f"{base}.layernorm_after.weight": (P_in,),
                    f"{base}.layernorm_after.bias": (P_in,),
                }
            )

            # MLP layers
            names_to_perms.update(
                {
                    f"{base}.intermediate.dense.weight": (f"P_ff_{i}", P_in),
                    f"{base}.intermediate.dense.bias": (f"P_ff_{i}",),
                    f"{base}.output.dense.weight": (P_in, f"P_ff_{i}"),
                    f"{base}.output.dense.bias": (P_in,),
                }
            )

        # Output head
        P_pool = "P_pooler"
        names_to_perms.update(
            {
                # "vit.pooler.dense.weight": (P_pool, P_in),
                # "vit.pooler.dense.bias": (P_pool,),
                "classifier.weight": (None, P_in),
                "classifier.bias": (None,),
            }
        )

        if prefix:
            names_to_perms = OrderedDict(
                (f"{prefix}.{name}", val) for (name, val) in names_to_perms.items()
            )

        acts_to_perms = None  # Handle activation permutations if needed
        return PermSpec(
            names_to_perms=names_to_perms,
            acts_to_perms=acts_to_perms,
            model_name="ViT",  # Changed from "Bert" to "ViT"
            num_heads=self.num_heads,
            d_head=self.d_head,
        )

    def _permute(self, perms: PermType, inplace: bool = True, **kwargs) -> "BaseModel":
        """
        Permutes the parameters of the model according to the perms dict and its permutation_spec.

        Args:
            perms: Dictionary of permutations
            inplace: If True, modify the model in place; if False, return a new model
            **kwargs: Additional arguments passed to permute_state_dct

        Returns:
            Permuted model
        """
        # Get permutation spec
        perm_spec = self.permutation_spec()

        # Permute state dict
        permuted_dct = permute_state_dct(
            model_state_dct=self.model.state_dict(),
            perm_spec=perm_spec,
            perms=perms,
            inplace=inplace,
            **kwargs,
        )

        # Create new model if not inplace
        model_ = self if inplace else deepcopy(self)

        try:
            # Load permuted state dict
            model_.model.load_state_dict(permuted_dct)
        except Exception as e:
            raise ValueError(
                f"Failed to load permuted state dict: {str(e)}\n"
                f"Original shapes: {[p.shape for p in self.model.state_dict().values()]}\n"
                f"Permuted shapes: {[p.shape for p in permuted_dct.values()]}"
            )

        return model_

    def get_init_stds(self, include_constant_params=False):
        std = OrderedDict()
        default_std = 0.02
        for k, v in self.named_parameters():
            # everything has same std
            if include_constant_params:
                std[k] = default_std
            # 0 std for bias or norm weight/bias if include_constant_params=False
            elif k.endswith(".bias") or (k.endswith(".weight") and v.dim() == 1):
                std[k] = 0
            else:
                std[k] = default_std
        return std
