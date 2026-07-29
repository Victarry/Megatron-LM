# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

import pytest
import torch

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


def _require_distributed_cuda(world_size=4):
    if not torch.cuda.is_available():
        pytest.skip("BalancedMoELayer state-dict tests require CUDA.")
    if int(os.environ.get("WORLD_SIZE", "1")) < world_size:
        pytest.skip(f"Run with torchrun --nproc-per-node={world_size}.")


def _make_config(*, balanced, grouped):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_ffn_hidden_size=32,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=grouped,
        expert_model_parallel_size=4,
        tensor_model_parallel_size=1,
        add_bias_linear=False,
        params_dtype=torch.bfloat16,
        bf16=True,
    )
    if balanced:
        config.moe_use_balanced_layer = True
        config.moe_num_spare_experts = 4
        config.moe_balance_assignment_algorithm = "approx_bin_packing"
        config.moe_balance_enable_debug_stats = True
    return config


def _submodules(grouped):
    if grouped:
        if not HAVE_TE or not is_te_min_version("1.9.0.dev0"):
            pytest.skip("TEGroupedMLP state-dict coverage requires Transformer Engine >= 1.9.")
        spec = get_gpt_layer_with_transformer_engine_submodules(
            num_experts=8, moe_grouped_gemm=True
        ).mlp
    else:
        spec = get_gpt_layer_local_submodules(num_experts=8, moe_grouped_gemm=False).mlp
    submodules = get_submodules(spec)
    assert isinstance(submodules, MoESubmodules)
    return submodules


def _sharded_metadata(state_dict):
    metadata = {}
    for key, value in state_dict.items():
        if hasattr(value, "global_shape"):
            metadata[key] = (
                value.key,
                value.global_shape,
                value.global_offset,
                getattr(value, "axis_fragmentations", None),
                value.replica_id,
            )
        else:
            metadata[key] = type(value).__name__
    return metadata


@pytest.mark.internal
@pytest.mark.parametrize("grouped", [False, True])
def test_balanced_moe_state_dict_matches_regular_moe(grouped):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        from megatron.core.transformer.moe.balanced_moe_layer import BalancedMoELayer

        _set_random_seed(seed_=1234, data_parallel_random_init=False)
        baseline = MoELayer(_make_config(balanced=False, grouped=grouped), _submodules(grouped))
        balanced = BalancedMoELayer(
            _make_config(balanced=True, grouped=grouped), _submodules(grouped)
        )
        baseline.cuda()
        balanced.cuda()

        baseline_state = baseline.state_dict()
        balanced_state = balanced.state_dict()
        assert baseline_state.keys() == balanced_state.keys()
        assert not any("local_experts.2." in key for key in balanced_state)
        assert not any("weight2" in key for key in balanced_state)

        balanced.load_state_dict(baseline_state, strict=True)
        baseline.load_state_dict(balanced_state, strict=True)

        baseline_sharded = _sharded_metadata(baseline.sharded_state_dict())
        balanced_sharded = _sharded_metadata(balanced.sharded_state_dict())
        assert baseline_sharded.keys() == balanced_sharded.keys()
        assert balanced_sharded == baseline_sharded
    finally:
        Utils.destroy_model_parallel()
