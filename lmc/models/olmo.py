from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import torch
import transformers
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    OlmoConfig,
    OlmoForCausalLM,
)

from lmc.data.data_stats import CHAT_TEMPLATES
from lmc.permutations import PermSpec, PermType, permute_state_dct

from .base_model import BaseModel
from .type_declaration import PATTERNS


class OLMo(BaseModel):
    _name: str = "OLMO"
    is_language_model: bool = True
    model_name: str = None

    def __init__(
        self,
        model_name: str = "allenai/OLMo-2-1124-7B",
        output_dim: Optional[int] = None,
        initialization_strategy: str = "pretrained",
        norm: str = "layernorm",
        gradient_checkpointing: bool = True,
        task_type: str = "generation",
        revision: Optional[str] = None,
        use_bfloat16: bool = True,
        chat_template: str = None,
    ):
        """
        Initialize OLMo model for either generation or classification tasks.

        Args:
            model_name: Name/path of the OLMo model to use
            output_dim: Number of output classes for classification (None for generation)
            initialization_strategy: Whether to use pretrained weights
            norm: Normalization type
            gradient_checkpointing: Whether to use gradient checkpointing
            task_type: Either "generation" or "classification"
        """
        if "allenai" not in model_name:
            model_name = f"allenai/{model_name}"
        self.model_name = model_name
        self.task_type = task_type
        super().__init__(output_dim, initialization_strategy, act="act", norm=norm)
        kwargs = dict(device_map="auto")
        if use_bfloat16:
            kwargs["torch_dtype"] = torch.bfloat16

        if initialization_strategy == "pretrained":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, revision=revision, **kwargs
            )
            # if task_type == "classification":
            #     if output_dim is None:
            #         raise ValueError(
            #             "output_dim must be specified for classification tasks"
            #         )
            #     # self.classifier = nn.Linear(self.model.config.hidden_size, output_dim)
        else:
            # Create a custom configuration
            config = OlmoConfig.from_pretrained(model_name, trust_remote_code=True)
            if task_type == "classification":
                if output_dim is None:
                    raise ValueError(
                        "output_dim must be specified for classification tasks"
                    )
                config.num_labels = output_dim
                ## error, doesn't work
                self.model = AutoModelForSequenceClassification(
                    config, trust_remote_code=True
                )
            else:  # generation
                self.model = AutoModelForCausalLM(
                    config, trust_remote_code=True, revision=revision, **kwargs
                )

            self.logger.info(
                f"OLMo model initialized from scratch, if that's not the desired behavior pass --initialization_strategy=pretrained."
            )
        print(f"Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            revision=revision,
            chat_template=CHAT_TEMPLATES.get(chat_template, None),
            padding_side="right",
            config=self.model.config,
        )
        self.generation_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            # chat_template=chat_template,
            chat_template=CHAT_TEMPLATES.get(chat_template, None),
            revision=revision,
            padding_side="left",  # Default, but being explicit
            config=self.model.config,
        )
        # Set padding token if not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.generation_tokenizer.pad_token = self.tokenizer.eos_token

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Store architecture details for permutation
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.d_head = (
            self.model.config.hidden_size // self.model.config.num_attention_heads
        )
        print(
            self.num_layers,
            self.num_heads,
            self.d_head,
            self.model.config.hidden_size,
            self.model.config.num_attention_heads,
        )
        self.use_bfloat16 = use_bfloat16
        # exit(0)

    @classmethod
    def is_valid_model_code(cls, model_code: str) -> bool:
        return model_code.lower() in PATTERNS[cls._name]

    @staticmethod
    def get_model_from_code(
        model_code: str,
        output_dim: Optional[int] = None,
        task_type: str = "generation",
        **kwargs,
    ) -> "BaseModel":
        return OLMo(
            model_name=model_code, output_dim=output_dim, task_type=task_type, **kwargs
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        """
        Forward pass handling both generation and classification.
        For generation, labels are the shifted input_ids for causal LM.
        For classification, labels are the class indices.
        """
        if self.use_bfloat16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Your existing forward logic here
                return self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    # **kwargs,
                )
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            # **kwargs,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        # max_length: int = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate text using the model.
        Only available when task_type is "generation".
        """
        if self.task_type != "generation":
            raise ValueError("Generation is only available for generation task type")

        if self.use_bfloat16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                return self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    # # max_length=max_length,
                    repetition_penalty=1.1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs,
                )
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # max_length=max_length,
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

    def permutation_spec(self, prefix: str = "", **kwargs) -> PermSpec:
        """
        Define permutation specification for OLMo architecture.

        OLMo uses non-parametric layer norm so it is missing in the perm_spec
        """
        # ..Missing {'model.transformer.ln_f.bias', 'lm_head.weight', 'model.transformer.ln_f.weight'}
        names_to_perms = OrderedDict()
        P0 = "P_0"

        # Word embeddings
        names_to_perms["model.embed_tokens.weight"] = (None, P0)
        # names_to_perms["model.transformer.ln_f.weight"] = (P0,)
        # names_to_perms["model.transformer.ln_f.bias"] = (P0,)

        # For each transformer block
        for i in range(self.num_layers):
            base = f"model.layers.{i}"
            P_in = P0
            P_head = f"P_head_{i}"
            P_dhead = f"P_dhead_{i}"

            # Attention
            names_to_perms.update(
                {
                    f"{base}.self_attn.q_proj.weight": ((P_head, P_dhead), P_in),
                    f"{base}.self_attn.k_proj.weight": ((P_head, P_dhead), P_in),
                    f"{base}.self_attn.v_proj.weight": ((P_head, P_dhead), P_in),
                    f"{base}.self_attn.o_proj.weight": (P_in, (P_head, P_dhead)),
                }
            )
            if "7b" in self.model_name.lower():
                names_to_perms.update(
                    {  # For RMSNorm in attention
                        f"{base}.self_attn.q_norm.weight": (
                            (P_head, P_dhead),
                        ),  # Only apply head permutation
                        f"{base}.self_attn.k_norm.weight": ((P_head, P_dhead),),
                        f"{base}.post_attention_layernorm.weight": (P_in,),
                        f"{base}.post_feedforward_layernorm.weight": (P_in,),
                        # f"{base}.self_attn.q_norm.weight": (P_in,),
                        # f"{base}.self_attn.k_norm.weight": (P_in,),
                    }
                )
            # FFN
            names_to_perms.update(
                {
                    f"{base}.mlp.gate_proj.weight": (f"P_ff_{i}", P_in),
                    f"{base}.mlp.up_proj.weight": (f"P_ff_{i}", P_in),
                    f"{base}.mlp.down_proj.weight": (P_in, f"P_ff_{i}"),
                }
            )
        names_to_perms.update({"model.norm.weight": (P0,)})
        # Output head depends on task type
        if self.task_type == "classification":
            names_to_perms.update(
                {
                    "classifier.weight": (None, P0),
                    "classifier.bias": (None,),
                }
            )
        else:  # generation
            pass
            names_to_perms.update(
                {
                    "lm_head.weight": (None, P0),  # Usually tied with embeddings
                }
            )

        if prefix:
            names_to_perms = OrderedDict(
                (f"{prefix}.{name}", val) for (name, val) in names_to_perms.items()
            )

        return PermSpec(
            names_to_perms=names_to_perms,
            acts_to_perms=None,  # Handle activation permutations if needed
            model_name="OLMo",
            num_heads=self.num_heads,
            d_head=self.d_head,
        )

    def permutation_spec1(self, prefix: str = "", **kwargs) -> PermSpec:
        """
        Define permutation specification for OLMo architecture.
        Similar to BERT but adapted for OLMo's architecture.
        """
        names_to_perms = OrderedDict()
        P0 = "P_0"

        # Embeddings
        names_to_perms["embeddings.word_embeddings.weight"] = (None, P0)
        if hasattr(self.model, "embeddings.position_embeddings"):
            names_to_perms["embeddings.position_embeddings.weight"] = (None, P0)
        names_to_perms["embeddings.LayerNorm.weight"] = (P0,)
        names_to_perms["embeddings.LayerNorm.bias"] = (P0,)

        # For each transformer layer
        for i in range(self.num_layers):
            base = f"encoder.layer.{i}"
            P_in = P0
            P_head = f"P_head_{i}"
            P_dhead = f"P_dhead_{i}"

            # Self-attention
            attention = f"{base}.attention.self"
            names_to_perms.update(
                {
                    f"{attention}.query.weight": ((P_head, P_dhead), P_in),
                    f"{attention}.query.bias": ((P_head, P_dhead),),
                    f"{attention}.key.weight": ((P_head, P_dhead), P_in),
                    f"{attention}.key.bias": ((P_head, P_dhead),),
                    f"{attention}.value.weight": ((P_head, P_dhead), P_in),
                    f"{attention}.value.bias": ((P_head, P_dhead),),
                }
            )

            # Attention output
            names_to_perms.update(
                {
                    f"{base}.attention.output.dense.weight": (P_in, (P_head, P_dhead)),
                    f"{base}.attention.output.dense.bias": (P_in,),
                    f"{base}.attention.output.LayerNorm.weight": (P_in,),
                    f"{base}.attention.output.LayerNorm.bias": (P_in,),
                }
            )

            # FFN
            names_to_perms.update(
                {
                    f"{base}.intermediate.dense.weight": (f"P_ff_{i}", P_in),
                    f"{base}.intermediate.dense.bias": (f"P_ff_{i}",),
                    f"{base}.output.dense.weight": (P_in, f"P_ff_{i}"),
                    f"{base}.output.dense.bias": (P_in,),
                    f"{base}.output.LayerNorm.weight": (P_in,),
                    f"{base}.output.LayerNorm.bias": (P_in,),
                }
            )

        # Output head depends on task type
        if self.task_type == "classification":
            names_to_perms.update(
                {
                    "classifier.weight": (None, P0),
                    "classifier.bias": (None,),
                }
            )
        else:  # generation
            names_to_perms.update(
                {
                    "lm_head.weight": (None, P0),  # Usually tied with embeddings
                }
            )

        if prefix:
            names_to_perms = OrderedDict(
                (f"{prefix}.{name}", val) for (name, val) in names_to_perms.items()
            )

        return PermSpec(
            names_to_perms=names_to_perms,
            acts_to_perms=None,  # Handle activation permutations if needed
            model_name="OLMo",
            num_heads=self.num_heads,
            d_head=self.d_head,
        )

    def _permute(self, perms: PermType, inplace: bool = True, **kwargs) -> "BaseModel":
        """
        Permutes the parameters of the model according to the perms dict and its permutation_spec.
        """
        perm_spec = self.permutation_spec()

        permuted_dct = permute_state_dct(
            model_state_dct=self.model.state_dict(),
            perm_spec=perm_spec,
            perms=perms,
            inplace=inplace,
            **kwargs,
        )

        model_ = self if inplace else deepcopy(self)

        try:
            model_.model.load_state_dict(permuted_dct)
        except Exception as e:
            raise ValueError(
                f"Failed to load permuted state dict: {str(e)}\n"
                f"Original shapes: {[p.shape for p in self.model.state_dict().values()]}\n"
                f"Permuted shapes: {[p.shape for p in permuted_dct.values()]}"
            )

        return model_

    def get_init_stds(self, include_constant_params: bool = False) -> Dict[str, float]:
        """
        Get initialization standard deviations for model parameters.
        """
        std = OrderedDict()
        default_std = 0.02  # Common default for transformer models

        for k, v in self.named_parameters():
            if include_constant_params:
                std[k] = default_std
            elif k.endswith(".bias") or (k.endswith(".weight") and v.dim() == 1):
                std[k] = 0
            else:
                std[k] = default_std

        return std
        return std
