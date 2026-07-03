# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib
import os

import pytest
import torch

from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _require_distributed_cuda(world_size=4):
    if not torch.cuda.is_available():
        pytest.skip("Balanced expert dispatcher tests require CUDA.")
    if int(os.environ.get("WORLD_SIZE", "1")) < world_size:
        pytest.skip(f"Run with torchrun --nproc-per-node={world_size}.")


def _require_expert_dispatcher():
    try:
        module = importlib.import_module("megatron.core.transformer.moe.expert_weight_dispatcher")
    except ModuleNotFoundError:
        pytest.fail(
            "megatron.core.transformer.moe.expert_weight_dispatcher is required for "
            "BalancedMoELayer expert dispatcher tests."
        )
    try:
        return module.AllToAllExpertWeightDispatcher
    except AttributeError:
        pytest.fail("AllToAllExpertWeightDispatcher is required in expert_weight_dispatcher.py.")


def _make_config(dtype):
    return TransformerConfig(
        num_layers=1,
        hidden_size=6,
        num_attention_heads=2,
        num_moe_experts=8,
        moe_ffn_hidden_size=10,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        moe_token_dispatcher_type="alltoall",
        expert_model_parallel_size=4,
        tensor_model_parallel_size=1,
        add_bias_linear=False,
        params_dtype=dtype,
        bf16=dtype is torch.bfloat16,
        fp16=dtype is torch.float16,
    )


def _make_home_weight(global_home_id, shape, dtype, device):
    values = torch.arange(shape.numel(), dtype=torch.float32, device=device).reshape(shape)
    values = values + global_home_id * 100000
    return values.to(dtype).detach().requires_grad_(True)


def _gather_home_weights(local_home_weights, ep_group):
    local_stack = torch.stack([weight.detach() for weight in local_home_weights], dim=0)
    gathered = [torch.empty_like(local_stack) for _ in range(torch.distributed.get_world_size(ep_group))]
    torch.distributed.all_gather(gathered, local_stack, group=ep_group)
    return torch.cat(gathered, dim=0)


def _expert_offloading_map(device):
    expert_map = torch.zeros(8, 8, dtype=torch.bool, device=device)
    expert_map[3, 0] = True
    expert_map[3, 1] = True
    expert_map[6, 2] = True
    expert_map[0, 4] = True
    expert_map[7, 7] = True
    return expert_map


def _expected_local_spare_weights(expert_map, gathered_home_weights, ep_rank, spare_per_rank):
    expected = []
    for local_spare_idx in range(spare_per_rank):
        spare_global = ep_rank * spare_per_rank + local_spare_idx
        home_indices = torch.where(expert_map[:, spare_global])[0]
        if home_indices.numel() == 0:
            expected.append(None)
        else:
            assert home_indices.numel() == 1
            expected.append(gathered_home_weights[home_indices.item()])
    return expected


def _assert_inactive_or_expected(dispatched_weight, expected_weight, reference_shape, *, dtype):
    if expected_weight is None:
        assert dispatched_weight.shape == reference_shape
        assert dispatched_weight.dtype == dtype
        torch.testing.assert_close(dispatched_weight, torch.zeros_like(dispatched_weight), rtol=0, atol=0)
        return
    torch.testing.assert_close(dispatched_weight, expected_weight, rtol=0, atol=0)
    assert dispatched_weight.dtype == dtype


def _expected_home_grads(expert_map, weight_shape, dtype, device):
    expected = torch.zeros((8, *weight_shape), dtype=dtype, device=device)
    for home_idx, spare_idx in zip(*torch.where(expert_map)):
        scale = float(spare_idx.item() + 1)
        expected[home_idx] += torch.ones(weight_shape, dtype=dtype, device=device) * scale
    return expected


@pytest.mark.internal
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_expert_weight_dispatch_values(dtype):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        dispatcher_cls = _require_expert_dispatcher()
        pg_collection = get_default_pg_collection()
        ep_rank = pg_collection.ep.rank()
        device = torch.device("cuda", torch.cuda.current_device())
        weight_shape = torch.Size([5, 6])
        local_home_weights = [
            _make_home_weight(ep_rank * 2 + local_idx, weight_shape, dtype, device)
            for local_idx in range(2)
        ]
        all_home_weights = _gather_home_weights(local_home_weights, pg_collection.ep)
        expert_map = _expert_offloading_map(device)

        dispatcher = dispatcher_cls(
            config=_make_config(dtype),
            ep_group=pg_collection.ep,
            num_home_experts=8,
            num_spare_experts=8,
        )
        metadata = dispatcher.preprocess(expert_map)
        dispatched = dispatcher.dispatch(metadata, *local_home_weights)

        assert len(dispatched) == 2
        expected = _expected_local_spare_weights(expert_map, all_home_weights, ep_rank, 2)
        for dispatched_weight, expected_weight in zip(dispatched, expected):
            _assert_inactive_or_expected(
                dispatched_weight, expected_weight, weight_shape, dtype=dtype
            )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_expert_weight_dispatch_grad_foldback(dtype):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        dispatcher_cls = _require_expert_dispatcher()
        pg_collection = get_default_pg_collection()
        ep_rank = pg_collection.ep.rank()
        device = torch.device("cuda", torch.cuda.current_device())
        weight_shape = torch.Size([5, 6])
        local_home_weights = [
            _make_home_weight(ep_rank * 2 + local_idx, weight_shape, dtype, device)
            for local_idx in range(2)
        ]
        expert_map = _expert_offloading_map(device)

        dispatcher = dispatcher_cls(
            config=_make_config(dtype),
            ep_group=pg_collection.ep,
            num_home_experts=8,
            num_spare_experts=8,
        )
        metadata = dispatcher.preprocess(expert_map)
        dispatched = dispatcher.dispatch(metadata, *local_home_weights)

        loss = torch.zeros((), dtype=dtype, device=device)
        for local_spare_idx, dispatched_weight in enumerate(dispatched):
            spare_global = ep_rank * 2 + local_spare_idx
            if not expert_map[:, spare_global].any():
                continue
            loss = loss + dispatched_weight.sum() * float(spare_global + 1)
        loss.backward()

        expected_global_grads = _expected_home_grads(expert_map, weight_shape, dtype, device)
        expected_local_grads = expected_global_grads[ep_rank * 2 : (ep_rank + 1) * 2]
        for weight, expected_grad in zip(local_home_weights, expected_local_grads):
            assert weight.grad is not None
            if dtype is torch.bfloat16:
                torch.testing.assert_close(weight.grad, expected_grad, rtol=2e-2, atol=2e-2)
            elif dtype is torch.float32:
                torch.testing.assert_close(weight.grad, expected_grad, rtol=1e-5, atol=1e-6)
            else:
                torch.testing.assert_close(weight.grad, expected_grad, rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()
