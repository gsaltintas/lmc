from collections import OrderedDict

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from lmc.permutations.utils import PermSpec

from .base_model import BaseModel
from .type_declaration import PATTERNS


class T5(BaseModel):
    _name: str = "T5"
    is_language_model: bool = True

    def __init__(
        self,
        model_name: str = "t5-small",
        initialization_strategy: str = "pretrained",
        norm: str = "layernorm",
    ):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        output_dim = self.tokenizer.vocab_size
        super().__init__(output_dim, initialization_strategy, act="act", norm=norm)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.num_layers = self.model.config.num_layers
        self.num_heads = self.model.config.num_heads
        self.d_head = self.model.config.d_model // self.model.config.num_heads

    @classmethod
    def is_valid_model_code(cls, model_code: str) -> bool:
        return model_code.lower() in PATTERNS[cls._name]

    @staticmethod
    def get_model_from_code(model_code: str, output_dim: int, **kwargs) -> "BaseModel":
        return T5(model_name=model_code, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        decoder_input_ids: torch.Tensor = None,
    ):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, decoder_input_ids=decoder_input_ids
        )

    def _acts_to_perms(self):
        return None

    def permutation_spec(self, prefix: str = "", **kwargs) -> PermSpec:
        """Generate permutation specification for T5 model.
        Maps between parameter names and their associated permutations."""
        names_to_perms = OrderedDict()
        P0 = "P_0"

        # Shared embeddings
        names_to_perms["shared.weight"] = (None, P0)  # Don't permute vocab dimension

        # Helper function for transformer block permutations
        def add_layer_perms(base: str, layer_idx: int, P_in: str):
            # For each layer, we need permutations for:
            # 1. Self attention (Q, K, V)
            # 2. Layer norm
            # 3. Dense layers
            P_qk = f"P_qk_{layer_idx}"
            P_v = f"P_v_{layer_idx}"
            P_ff = f"P_ff_{layer_idx}"

            # Self attention
            names_to_perms.update({
                # Q, K projections share same permutation for attention computation
                f"{base}.{layer_idx}.layer.0.SelfAttention.q.weight": ((P_qk, f"P_dhead_{layer_idx}"), P_in),
                f"{base}.{layer_idx}.layer.0.SelfAttention.q.bias": ((P_qk, f"P_dhead_{layer_idx}"),),
                f"{base}.{layer_idx}.layer.0.SelfAttention.k.weight": ((P_qk, f"P_dhead_{layer_idx}"), P_in),
                f"{base}.{layer_idx}.layer.0.SelfAttention.k.bias": ((P_qk, f"P_dhead_{layer_idx}"),),
                f"{base}.{layer_idx}.layer.0.SelfAttention.v.weight": ((P_v, f"P_dhead_{layer_idx}"), P_in),
                f"{base}.{layer_idx}.layer.0.SelfAttention.v.bias": ((P_v, f"P_dhead_{layer_idx}"),),
                f"{base}.{layer_idx}.layer.0.SelfAttention.o.weight": (P_in, (P_v, f"P_dhead_{layer_idx}")),
                f"{base}.{layer_idx}.layer.0.SelfAttention.o.bias": (P_in,),
                
                # Layer norms
                f"{base}.{layer_idx}.layer.0.layer_norm.weight": (P_in,),
                f"{base}.{layer_idx}.layer.0.layer_norm.bias": (P_in,),
                
                # Feed-forward
                f"{base}.{layer_idx}.layer.1.DenseReluDense.wi.weight": (P_ff, P_in),
                f"{base}.{layer_idx}.layer.1.DenseReluDense.wi.bias": (P_ff,),
                f"{base}.{layer_idx}.layer.1.DenseReluDense.wo.weight": (P_in, P_ff),
                f"{base}.{layer_idx}.layer.1.DenseReluDense.wo.bias": (P_in,),
                f"{base}.{layer_idx}.layer.1.layer_norm.weight": (P_in,),
                f"{base}.{layer_idx}.layer.1.layer_norm.bias": (P_in,),
            })

            # For decoder, add cross-attention
            if base == "decoder.block":
                names_to_perms.update({
                    f"{base}.{layer_idx}.layer.1.EncDecAttention.q.weight": ((f"P_qk_cross_{layer_idx}", f"P_dhead_cross_{layer_idx}"), P_in),
                    f"{base}.{layer_idx}.layer.1.EncDecAttention.q.bias": ((f"P_qk_cross_{layer_idx}", f"P_dhead_cross_{layer_idx}"),),
                    f"{base}.{layer_idx}.layer.1.EncDecAttention.k.weight": ((f"P_qk_cross_{layer_idx}", f"P_dhead_cross_{layer_idx}"), P_in),
                    f"{base}.{layer_idx}.layer.1.EncDecAttention.k.bias": ((f"P_qk_cross_{layer_idx}", f"P_dhead_cross_{layer_idx}"),),
                    f"{base}.{layer_idx}.layer.1.EncDecAttention.v.weight": ((f"P_v_cross_{layer_idx}", f"P_dhead_cross_{layer_idx}"), P_in),
                    f"{base}.{layer_idx}.layer.1.EncDecAttention.v.bias": ((f"P_v_cross_{layer_idx}", f"P_dhead_cross_{layer_idx}"),),
                    f"{base}.{layer_idx}.layer.1.EncDecAttention.o.weight": (P_in, (f"P_v_cross_{layer_idx}", f"P_dhead_cross_{layer_idx}")),
                    f"{base}.{layer_idx}.layer.1.EncDecAttention.o.bias": (P_in,),
                    # Additional layer norm for cross-attention
                    f"{base}.{layer_idx}.layer.1.layer_norm.weight": (P_in,),
                    f"{base}.{layer_idx}.layer.1.layer_norm.bias": (P_in,),
                })

        # Encoder layers
        for i in range(self.num_layers):
            add_layer_perms("encoder.block", i, P0)

        # Decoder layers
        for i in range(self.num_layers):
            add_layer_perms("decoder.block", i, P0)

        # Final layer norms
        names_to_perms.update({
            "encoder.final_layer_norm.weight": (P0,),
            "encoder.final_layer_norm.bias": (P0,),
            "decoder.final_layer_norm.weight": (P0,),
            "decoder.final_layer_norm.bias": (P0,),
        })

        if prefix:
            names_to_perms = OrderedDict(
                (f"{prefix}.{name}", val) for (name, val) in names_to_perms.items()
            )        
        
        perm_spec = PermSpec(
            names_to_perms=names_to_perms,
            acts_to_perms=self._acts_to_perms(),
            model_name="T5",
            num_heads=self.num_heads,
            d_head=self.d_head
        )
        return perm_spec