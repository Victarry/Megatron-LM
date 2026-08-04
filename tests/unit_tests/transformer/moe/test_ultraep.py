# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.transformer.moe import ultraep_manager
from megatron.core.transformer.moe.experts import TEGroupedMLP, _parse_te_expert_idx
from megatron.core.transformer.moe.ultraep_manager import UltraEPManager
from megatron.core.transformer.transformer_config import TransformerConfig


def _ultraep_config_kwargs(**overrides):
    kwargs = dict(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_ffn_hidden_size=128,
        expert_model_parallel_size=2,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type='alltoall',
        gradient_accumulation_fusion=True,
        params_dtype=torch.bfloat16,
        gated_linear_unit=True,
        add_bias_linear=False,
        moe_enable_ultraep=True,
        moe_num_redundant_experts_per_rank=1,
    )
    kwargs.update(overrides)
    return kwargs


def test_ultraep_transformer_config_accepts_supported_configuration():
    config = TransformerConfig(**_ultraep_config_kwargs())

    assert config.moe_enable_ultraep
    assert config.moe_num_redundant_experts_per_rank == 1


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"moe_num_redundant_experts_per_rank": 0}, "must be greater than zero"),
        ({"moe_token_dispatcher_type": "allgather"}, "token_dispatcher_type"),
        ({"moe_grouped_gemm": False}, "moe_grouped_gemm"),
        ({"add_bias_linear": True}, "add_bias_linear"),
    ],
)
def test_ultraep_transformer_config_rejects_unsupported_configuration(overrides, match):
    with pytest.raises(ValueError, match=match):
        TransformerConfig(**_ultraep_config_kwargs(**overrides))


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("linear_fc1.weight0", 0),
        ("linear_fc1.bias12", 12),
        ("linear_fc1._extra_state", None),
        ("linear_fc1._extra_state3", None),
        ("linear_fc2.weight1", None),
        ("linear_fc1.weight", None),
    ],
)
def test_parse_te_expert_idx(key, expected):
    assert _parse_te_expert_idx(key, "linear_fc1") == expected


def test_ultraep_checkpoint_filter_drops_replica_entries_in_singleton_mode():
    experts = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(experts)
    experts.ep_group = SimpleNamespace(size=lambda: 2, rank=lambda: 1)
    master_value = object()
    replica_value = object()
    extra_state = object()

    filtered = experts._ultraep_filter_replica_checkpoint_entries(
        {
            "linear_fc1.weight0": master_value,
            "linear_fc1.weight1": replica_value,
            "linear_fc1._extra_state": extra_state,
        },
        "linear_fc1",
        ep_axis=0,
        num_local_master_experts=1,
        fix_metadata=False,
    )

    assert filtered == {"linear_fc1.weight0": master_value, "linear_fc1._extra_state": extra_state}


def test_ultraep_checkpoint_filter_restores_logical_expert_metadata():
    experts = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(experts)
    experts.ep_group = SimpleNamespace(size=lambda: 2, rank=lambda: 1)
    master = ShardedTensor.from_rank_offsets(
        "linear_fc2.weight0", torch.zeros(2, 3), (0, 3, 6), prepend_axis_num=1
    )
    replica = ShardedTensor.from_rank_offsets(
        "linear_fc2.weight1", torch.zeros(2, 3), (0, 4, 6), prepend_axis_num=1
    )
    extra_state = ShardedObject("linear_fc2._extra_state", object(), (1,), (0,))

    filtered = experts._ultraep_filter_replica_checkpoint_entries(
        {
            "linear_fc2.weight0": master,
            "linear_fc2.weight1": replica,
            "linear_fc2._extra_state": extra_state,
        },
        "linear_fc2",
        ep_axis=0,
        num_local_master_experts=1,
        fix_metadata=True,
    )

    assert set(filtered) == {"linear_fc2.weight0", "linear_fc2._extra_state"}
    assert filtered["linear_fc2.weight0"].global_shape == (2, 2, 3)
    assert filtered["linear_fc2.weight0"].global_offset == (1, 0, 0)
    assert filtered["linear_fc2.weight0"].axis_fragmentations == (2, 1, 1)
    assert filtered["linear_fc2._extra_state"] is extra_state


def test_ultraep_manager_preserves_one_based_mcore_layer_ids():
    manager = UltraEPManager.__new__(UltraEPManager)
    manager.num_layers = 2

    assert manager.layer_id(1) == 1
    assert manager.layer_id(2) == 2
    with pytest.raises(ValueError, match="must be in"):
        manager.layer_id(0)


def test_destroy_ultraep_managers_closes_and_clears_registry():
    manager_a = SimpleNamespace(close=Mock())
    manager_b = SimpleNamespace(close=Mock())
    ultraep_manager._ULTRAEP_MANAGER_REGISTRY.update({1: manager_a, 2: manager_b})

    ultraep_manager.destroy_ultraep_managers()

    manager_a.close.assert_called_once_with()
    manager_b.close.assert_called_once_with()
    assert ultraep_manager._ULTRAEP_MANAGER_REGISTRY == {}


def test_ultraep_manager_close_is_idempotent():
    manager = UltraEPManager.__new__(UltraEPManager)
    manager.runtime = SimpleNamespace(destroy=Mock())
    manager._closed = False

    manager.close()
    manager.close()

    manager.runtime.destroy.assert_called_once_with()
    assert manager._closed
