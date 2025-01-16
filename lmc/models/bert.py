from collections import OrderedDict
from copy import deepcopy

import torch
from transformers import (BertConfig, BertForSequenceClassification,
                          BertTokenizer)

from lmc.permutations import PermSpec, PermType, permute_state_dct

from .base_model import BaseModel
from .type_declaration import PATTERNS


class Bert(BaseModel):
    _name = "BERT"
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        output_dim: int = ...,
        initialization_strategy: str = "pretrained",
        norm: str = "layernorm",
        gradient_checkpointing: bool = True
    ):
        super().__init__(output_dim, initialization_strategy, act="act", norm=norm)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        if initialization_strategy == "pretrained":
            self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=output_dim)
        else:
            # Create a custom configuration
            config = BertConfig.from_pretrained(model_name, num_labels=output_dim)
            self.model = BertForSequenceClassification(config,)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.num_layers = self.model.config.num_hidden_layers

        self.num_heads = self.model.config.num_attention_heads
        self.d_head = self.model.config.hidden_size // self.model.config.num_attention_heads
        
    @classmethod
    def is_valid_model_code(cls, model_code: str) -> bool:
        return model_code.lower() in PATTERNS[cls._name]

    @staticmethod
    def get_model_from_code(model_code: str, output_dim: int, **kwargs) -> "BaseModel":
        return Bert(model_name=model_code, output_dim=output_dim, **kwargs)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        **kwargs  # This will capture any additional arguments
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs  # Pass them along to the model
        )
        
    def permutation_spec(self, prefix: str = "", **kwargs) -> PermSpec:
        names_to_perms = OrderedDict()
        P0 = "P_0"
        
        # Embeddings remain the same...
        names_to_perms["bert.embeddings.word_embeddings.weight"] = (None, P0)
        names_to_perms["bert.embeddings.position_embeddings.weight"] = (None, P0)
        names_to_perms["bert.embeddings.token_type_embeddings.weight"] = (None, P0)
        names_to_perms["bert.embeddings.LayerNorm.weight"] = (P0,)
        names_to_perms["bert.embeddings.LayerNorm.bias"] = (P0,)

        # For each transformer layer
        for i in range(self.num_layers):
            base = f"bert.encoder.layer.{i}"
            P_in = P0  # Input permutation
            
            # Head-level permutations
            P_head = f"P_head_{i}"    # Permute attention heads
            P_dhead = f"P_dhead_{i}"  # Permute within head dimension
            
            # Self-attention with head permutations
            attention = f"{base}.attention.self"
            
            # For query, key, value weights, we need to handle both head and d_head dimensions
            # Shape is (hidden_size, num_heads * d_head) = (768, 12 * 64)
            names_to_perms.update({
                # Split output dimension into (num_heads, d_head)
                f"{attention}.query.weight": ((P_head, P_dhead), P_in),  # Tuple for head dims
                f"{attention}.query.bias": ((P_head, P_dhead),),
                f"{attention}.key.weight": ((P_head, P_dhead), P_in),
                f"{attention}.key.bias": ((P_head, P_dhead),),
                f"{attention}.value.weight": ((P_head, P_dhead), P_in),
                f"{attention}.value.bias": ((P_head, P_dhead),),
            })
            
            # Attention output needs to combine head permutations
            names_to_perms.update({
                # Combine head dimensions back to hidden_size
                f"{base}.attention.output.dense.weight": (P_in, (P_head, P_dhead)),
                f"{base}.attention.output.dense.bias": (P_in,),
                f"{base}.attention.output.LayerNorm.weight": (P_in,),
                f"{base}.attention.output.LayerNorm.bias": (P_in,),
            })
            
            # FFN layers remain the same...
            names_to_perms.update({
                f"{base}.intermediate.dense.weight": (f"P_ff_{i}", P_in),
                f"{base}.intermediate.dense.bias": (f"P_ff_{i}",),
                f"{base}.output.dense.weight": (P_in, f"P_ff_{i}"),
                f"{base}.output.dense.bias": (P_in,),
                f"{base}.output.LayerNorm.weight": (P_in,),
                f"{base}.output.LayerNorm.bias": (P_in,),
            })

        # Output head remains the same...
        P_pool = "P_pooler"
        names_to_perms.update({
            "bert.pooler.dense.weight": (P_pool, P_in),
            "bert.pooler.dense.bias": (P_pool,),
            "classifier.weight": (None, P_pool),
            "classifier.bias": (None,),
        })

        if prefix:
            names_to_perms = OrderedDict(
                (f"{prefix}.{name}", val) for (name, val) in names_to_perms.items()
            )

        acts_to_perms = None  # Handle activation permutations if needed
        return PermSpec(names_to_perms=names_to_perms, acts_to_perms=acts_to_perms, model_name="Bert", num_heads=self.num_heads, d_head=self.d_head)

    def _permute(
        self, 
        perms: PermType, 
        inplace: bool = True, 
        **kwargs
    ) -> "BaseModel":
        """
        Permutes the parameters of the model according to the perms dict and its permutation_spec.
        
        Args:
            perms: Dictionary of permutations
            inplace: If True, modify the model in place; if False, return a new model
            **kwargs: Additional arguments passed to permute_state_dct
            
        Returns:
            Permuted model
        """
        # Get head dimensions from model config if not provided
            
        # Get permutation spec
        perm_spec = self.permutation_spec()
        
        # Permute state dict
        permuted_dct = permute_state_dct(
            model_state_dct=self.model.state_dict(),
            perm_spec=perm_spec,
            perms=perms,
            inplace=inplace,
            **kwargs
        )
        
        # Create new model if not inplace
        model_ = self if inplace else deepcopy(self)
        
        try:
            # Load permuted state dict
            model_.model.load_state_dict(permuted_dct)
        except Exception as e:
            raise ValueError(f"Failed to load permuted state dict: {str(e)}\n"
                            f"Original shapes: {[p.shape for p in self.model.state_dict().values()]}\n"
                            f"Permuted shapes: {[p.shape for p in permuted_dct.values()]}")
        
        return model_