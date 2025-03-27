from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from lmc.permutations import PermSpec, PermType, permute_state_dct

from .base_model import BaseModel
from .type_declaration import PATTERNS

## TODO:/ WIP


class SEGFORMER(BaseModel):
    _name: str = "SEGFORMER"
    is_language_model: bool = False
    model_name: str = None

    def __init__(
        self,
        model_name: str = "nvidia/mit-b0",
        # model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",  # Pre-trained on ADE20K
        output_dim: int = 150,  # Default for ADE20K
        initialization_strategy: str = "pretrained",
        norm: str = "layernorm",
    ):
        self.model_name = model_name
        super().__init__(output_dim, initialization_strategy, act="act", norm=norm)

        if initialization_strategy == "pretrained":
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            import code

            # code.interact(local=locals() | globals())
            # Check if we need to replace the segmentation head
            # if output_dim != self.model.config.num_labels:
            #     # Create a new decoder head with the right output dimension
            #     self.model.decode_head.classifier = nn.Conv2d(
            #         self.model.decode_head.classifier.in_channels,
            #         output_dim,
            #         kernel_size=1,
            #     )
            #     self.logger.warn(
            #         "Model's classifier head reset, you probably need to finetune this model."
            #     )
        else:
            raise ValueError(f"SegFormer from scratch is not implemented yet.")

        # Store architecture details for permutation specification
        self.num_encoder_blocks = len(self.model.config.hidden_sizes)
        self.num_layers = sum(self.model.config.depths)
        self.hidden_sizes = self.model.config.hidden_sizes

        # Create image processor for preprocessing
        try:
            self.processor = SegformerImageProcessor.from_pretrained(model_name)
        except:
            self.processor = None
            self.logger.warning(
                "Could not load image processor for model. Make sure to preprocess inputs correctly."
            )

    @classmethod
    def is_valid_model_code(cls, model_code: str) -> bool:
        return model_code.lower() in PATTERNS[cls._name]

    @staticmethod
    def get_model_from_code(model_code: str, output_dim: int, **kwargs) -> "BaseModel":
        return SEGFORMER(model_name=model_code, output_dim=output_dim, **kwargs)

    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        return_dict: bool = True,
        **kwargs,
    ):
        # Handle potential preprocessing if needed
        if (
            self.processor is not None
            and hasattr(pixel_values, "shape")
            and len(pixel_values.shape) == 4
        ):
            # Check if pixel values need normalization
            if pixel_values.max() > 1.0:
                # Assume [0, 255] range and normalize
                pixel_values = pixel_values / 255.0

        # Forward pass through the model
        outputs = self.model(
            pixel_values=pixel_values,
            labels=labels,
            return_dict=return_dict,
            **kwargs,
        )

        if not return_dict:
            return outputs

        # For consistency with your API, rename logits attribute if needed
        if hasattr(outputs, "logits") and not hasattr(outputs, "logits"):
            outputs.logits = outputs.logits

        return outputs.logits

    def permutation_spec(self, prefix: str = "", **kwargs) -> PermSpec:
        """
        Define permutation specification for SegFormer.

        SegFormer has a hierarchical encoder with 4 stages, with each stage having
        multiple MixFFN and attention blocks.
        """
        names_to_perms = OrderedDict()
        P0 = "P_0"

        # Get all layer counts and dimensions
        depths = self.model.config.depths
        hidden_sizes = self.model.config.hidden_sizes

        # SegFormer patch embeddings at each stage
        for i in range(len(hidden_sizes)):
            if i == 0:
                # First stage has a convolutional patch embedding
                names_to_perms[
                    f"segformer.encoder.patch_embeddings.{i}.proj.weight"
                ] = (P0, None, None, None)
                names_to_perms[f"segformer.encoder.patch_embeddings.{i}.proj.bias"] = (
                    P0,
                )
                names_to_perms[
                    f"segformer.encoder.patch_embeddings.{i}.layer_norm.weight"
                ] = (P0,)
                names_to_perms[
                    f"segformer.encoder.patch_embeddings.{i}.layer_norm.bias"
                ] = (P0,)
            else:
                # Later stages have patch embeddings with strided convolutions
                names_to_perms[
                    f"segformer.encoder.patch_embeddings.{i}.proj.weight"
                ] = (None, P0, None, None)
                names_to_perms[f"segformer.encoder.patch_embeddings.{i}.proj.bias"] = (
                    None,
                )
                names_to_perms[
                    f"segformer.encoder.patch_embeddings.{i}.layer_norm.weight"
                ] = (None,)
                names_to_perms[
                    f"segformer.encoder.patch_embeddings.{i}.layer_norm.bias"
                ] = (None,)

        # Track layer index across stages
        layer_idx = 0

        # Process each stage
        for stage_idx, (depth, hidden_size) in enumerate(zip(depths, hidden_sizes)):
            # For each block in the stage
            for block_idx in range(depth):
                block_prefix = f"segformer.encoder.block.{stage_idx}.{block_idx}"

                # Layer normalization
                names_to_perms[f"{block_prefix}.layer_norm_1.weight"] = (P0,)
                names_to_perms[f"{block_prefix}.layer_norm_1.bias"] = (P0,)

                # Attention
                P_head = f"P_head_{layer_idx}"

                # SegFormer uses efficient attention with different implementation
                # We'll provide a simplified permutation spec
                names_to_perms[f"{block_prefix}.attention.query.weight"] = (P_head, P0)
                names_to_perms[f"{block_prefix}.attention.query.bias"] = (P_head,)
                names_to_perms[f"{block_prefix}.attention.key.weight"] = (P_head, P0)
                names_to_perms[f"{block_prefix}.attention.key.bias"] = (P_head,)
                names_to_perms[f"{block_prefix}.attention.value.weight"] = (P_head, P0)
                names_to_perms[f"{block_prefix}.attention.value.bias"] = (P_head,)
                names_to_perms[f"{block_prefix}.attention.proj.weight"] = (P0, P_head)
                names_to_perms[f"{block_prefix}.attention.proj.bias"] = (P0,)

                # Layer normalization
                names_to_perms[f"{block_prefix}.layer_norm_2.weight"] = (P0,)
                names_to_perms[f"{block_prefix}.layer_norm_2.bias"] = (P0,)

                # MixFFN
                P_ff = f"P_ff_{layer_idx}"
                names_to_perms[f"{block_prefix}.mlp.dense_1.weight"] = (P_ff, P0)
                names_to_perms[f"{block_prefix}.mlp.dense_1.bias"] = (P_ff,)
                names_to_perms[f"{block_prefix}.mlp.dense_2.weight"] = (P0, P_ff)
                names_to_perms[f"{block_prefix}.mlp.dense_2.bias"] = (P0,)

                # Increment layer counter
                layer_idx += 1

        # Decoder head
        # This is simplified as SegFormer has a complex MLP decoder
        names_to_perms["decode_head.linear_c.0.weight"] = (None, P0)
        names_to_perms["decode_head.linear_c.1.weight"] = (None, None)
        names_to_perms["decode_head.linear_c.2.weight"] = (None, None)
        names_to_perms["decode_head.linear_c.3.weight"] = (None, None)
        names_to_perms["decode_head.classifier.weight"] = (None, None)
        names_to_perms["decode_head.classifier.bias"] = (None,)

        if prefix:
            names_to_perms = OrderedDict(
                (f"{prefix}.{name}", val) for (name, val) in names_to_perms.items()
            )

        # Get values for the PermSpec
        # Use the first stage values as approximation
        num_heads = (
            self.model.config.num_attention_heads[0]
            if isinstance(self.model.config.num_attention_heads, list)
            else self.model.config.num_attention_heads
        )
        d_head = hidden_sizes[0] // num_heads

        acts_to_perms = None  # Handle activation permutations if needed
        return PermSpec(
            names_to_perms=names_to_perms,
            acts_to_perms=acts_to_perms,
            model_name="SegFormer",
            num_heads=num_heads,
            d_head=d_head,
        )

    def _permute(self, perms: PermType, inplace: bool = True, **kwargs) -> "BaseModel":
        """
        Permutes the parameters of the model according to the perms dict and its permutation_spec.
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
        default_std = 0.02  # Common initialization std for transformers

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

    def preprocess_image(self, image):
        """
        Preprocess an image for the model using the model's processor.

        Args:
            image: PIL Image or numpy array

        Returns:
            Tensor ready for model input
        """
        if self.processor is None:
            raise ValueError(
                "Image processor not available. Initialize the model with a valid model_name."
            )

        inputs = self.processor(images=image, return_tensors="pt")
        return inputs.pixel_values
