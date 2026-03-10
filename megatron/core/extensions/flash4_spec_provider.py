# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FA4 spec provider: TE linear layers + FA4 attention kernels with Ulysses CP."""

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.transformer.flash4_attention import FA4DotProductAttention


class FA4SpecProvider(TESpecProvider):
    """Extends TESpecProvider to use FA4 for attention while keeping TE for linear layers."""

    def __init__(self):
        super().__init__(fallback_to_eager_attn=False)

    def core_attention(self) -> type:
        """Use FA4 attention instead of TE attention."""
        return FA4DotProductAttention
