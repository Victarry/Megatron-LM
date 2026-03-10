# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Flash Attention 4 (FA4) attention backend with Ulysses/A2A context parallelism.

FA4 from Dao-AILab uses CuTeDSL kernels for Hopper+ GPUs. This module bypasses
Transformer Engine for attention while still using TE for linear layers.

Ulysses CP: All-to-all scatter heads across CP ranks, compute full-sequence
attention per rank, all-to-all gather back. Requires num_query_groups % cp_size == 0.
"""

import math
from typing import Optional

import torch
from torch import Tensor

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import all_to_all
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide


class FA4DotProductAttention(MegatronModule):
    """Core attention using Flash Attention 4 kernels with optional Ulysses A2A CP.

    Same constructor signature as DotProductAttention. Supports SBHD layout only.

    Args:
        config: Transformer config
        layer_number: Layer index (1-based internally)
        attn_mask_type: Attention mask type
        attention_type: "self" or "cross"
        attention_dropout: Override dropout rate
        softmax_scale: Override softmax scale
        cp_comm_type: Context parallelism communication type (must be "a2a" if CP > 1)
        pg_collection: Process group collection
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)

        self.config = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values (mirrors DotProductAttention)
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])
        else:
            assert hasattr(pg_collection, 'tp'), (
                "FA4DotProductAttention pg_collection must have tp process group"
            )
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp

        world_size = pg_collection.tp.size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(config.num_query_groups, world_size)

        # Softmax scale
        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            self.softmax_scale = softmax_scale

        if config.apply_query_key_layer_scaling:
            self.softmax_scale /= self.layer_number

        # FA4 does not support attention dropout
        dropout_p = config.attention_dropout if attention_dropout is None else attention_dropout
        assert dropout_p == 0.0, (
            f"FA4 does not support attention dropout, got dropout_p={dropout_p}"
        )

        # Context parallelism setup
        self.cp_size = config.context_parallel_size
        self.cp_comm_type = cp_comm_type

        if self.cp_size > 1:
            assert cp_comm_type == "a2a", (
                f"FA4 only supports 'a2a' (Ulysses) context parallelism, got {cp_comm_type}"
            )
            assert config.num_query_groups % self.cp_size == 0, (
                f"num_query_groups ({config.num_query_groups}) must be divisible by "
                f"cp_size ({self.cp_size}) for Ulysses CP"
            )
            assert config.num_attention_heads % self.cp_size == 0, (
                f"num_attention_heads ({config.num_attention_heads}) must be divisible by "
                f"cp_size ({self.cp_size}) for Ulysses CP"
            )
            assert pg_collection is not None and hasattr(pg_collection, 'cp'), (
                "pg_collection must have cp process group for context parallelism"
            )
            self.cp_group = pg_collection.cp

    def _a2a_cp2hp(self, t: Tensor) -> Tensor:
        """Ulysses scatter: [S/cp, B, H, d] -> [S, B, H/cp, d] via All-to-All.

        Splits along head dim into cp chunks, concatenates along seq dim,
        then performs all-to-all to redistribute.
        """
        s, b, h, d = t.shape
        # Reshape to 2D for all_to_all: [S/cp * B, H * d]
        t_2d = t.reshape(s * b, h * d)
        # Split hidden into cp chunks and concat along token dim
        chunks = torch.split(t_2d, h * d // self.cp_size, dim=1)
        t_2d = torch.cat(chunks, dim=0)  # [S/cp * B * cp, H*d/cp] = [S*B, H*d/cp]
        # All-to-all: each rank sends its seq portion, receives head portion
        t_2d = all_to_all(self.cp_group, t_2d)
        # Reshape back: [S, B, H/cp, d]
        return t_2d.reshape(s * self.cp_size, b, h // self.cp_size, d)

    def _a2a_hp2cp(self, t: Tensor) -> Tensor:
        """Ulysses gather: [S, B, H/cp, d] -> [S/cp, B, H, d] via All-to-All (reverse).

        Performs all-to-all then splits along token dim, concatenates along head dim.
        """
        s, b, h_local, d = t.shape
        # Reshape to 2D: [S * B, H_local * d]
        t_2d = t.reshape(s * b, h_local * d)
        # All-to-all: each rank sends its head portion, receives seq portion
        t_2d = all_to_all(self.cp_group, t_2d)
        # Split token dim into cp chunks and concat along hidden dim
        s_local = s // self.cp_size
        chunks = torch.split(t_2d, s_local * b, dim=0)
        t_2d = torch.cat(chunks, dim=1)  # [S_local * B, H_local * cp * d] = [S/cp*B, H*d]
        # Reshape back: [S/cp, B, H, d]
        return t_2d.reshape(s_local, b, h_local * self.cp_size, d)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        attn_mask_type: Optional[AttnMaskType] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        """Forward pass using FA4 kernels.

        Args:
            query: [S(/cp), B, Hq, d]
            key: [S(/cp), B, Hk, d]
            value: [S(/cp), B, Hk, d]
            attention_mask: Unused by FA4 (uses causal flag instead)
            attn_mask_type: Override mask type
            attention_bias: Not supported
            packed_seq_params: Not supported (FA4 uses BSHD only)

        Returns:
            Tensor: [S(/cp), B, Hp] where Hp = num_heads_per_partition * head_dim
        """
        assert packed_seq_params is None, (
            "Packed sequence (THD) is not supported by FA4DotProductAttention. "
            "FA4 uses BSHD format only."
        )
        assert attention_bias is None, (
            "Attention bias is not supported by FA4DotProductAttention."
        )

        # Lazy import FA4 to avoid import errors when flash_attn is not installed
        try:
            from flash_attn.cute import flash_attn_func as fa4_flash_attn_func
        except ImportError:
            raise ImportError(
                "Flash Attention 4 (flash_attn.cute) is required for the flash4 backend. "
                "Install from https://github.com/Dao-AILab/flash-attention"
            )

        # Determine causal flag
        if attn_mask_type is not None:
            causal = attn_mask_type in (AttnMaskType.causal, AttnMaskType.padding_causal)
        else:
            causal = self.attn_mask_type in (AttnMaskType.causal, AttnMaskType.padding_causal)

        # Ulysses A2A CP: scatter heads, gather full sequence
        if self.cp_size > 1:
            query = self._a2a_cp2hp(query)   # [S/cp,B,Hq,d] -> [S,B,Hq/cp,d]
            key = self._a2a_cp2hp(key)       # [S/cp,B,Hk,d] -> [S,B,Hk/cp,d]
            value = self._a2a_cp2hp(value)   # [S/cp,B,Hk,d] -> [S,B,Hk/cp,d]

        # SBHD -> BSHD for FA4
        q = query.permute(1, 0, 2, 3).contiguous()
        k = key.permute(1, 0, 2, 3).contiguous()
        v = value.permute(1, 0, 2, 3).contiguous()

        # FA4 kernel call (FA4 does not support attention dropout)
        # FA4 returns (output, lse) tuple; we only need the output
        output, _lse = fa4_flash_attn_func(
            q, k, v,
            softmax_scale=self.softmax_scale,
            causal=causal,
        )

        # BSHD -> SBHD
        output = output.permute(1, 0, 2, 3).contiguous()

        # Ulysses A2A CP: gather heads, scatter sequence back
        if self.cp_size > 1:
            output = self._a2a_hp2cp(output)  # [S,B,Hq/cp,d] -> [S/cp,B,Hq,d]

        # Merge heads: [S(/cp), B, Hq, d] -> [S(/cp), B, Hp]
        new_shape = output.size()[:-2] + (self.hidden_size_per_partition,)
        return output.view(*new_shape)
