# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib
import math
import os
import time
from dataclasses import dataclass

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_BENCHMARK_WARMUP_ITERS = 5
_BENCHMARK_ITERS = 20
_BENCHMARK_WEIGHT_CASES = [
    ("representative_fc1_proxy", torch.Size([4096, 2048])),
    ("representative_fc2_proxy", torch.Size([2048, 4096])),
]


@dataclass(frozen=True)
class _RoutePayloadStats:
    normalized_remote_payload_bytes: int
    busiest_send_payload_bytes: int
    busiest_recv_payload_bytes: int
    busiest_send_plus_recv_payload_bytes: int
    logical_payload_bytes: int


def _require_distributed_cuda(world_size=4):
    if not torch.cuda.is_available():
        pytest.skip("Balanced expert dispatcher tests require CUDA.")
    if int(os.environ.get("WORLD_SIZE", "1")) < world_size:
        pytest.skip(f"Run with torchrun --nproc-per-node={world_size}.")


def _cuda_event_latency_ms(fn, *, warmup_iters=_BENCHMARK_WARMUP_ITERS, iters=_BENCHMARK_ITERS):
    result = None
    for _ in range(warmup_iters):
        result = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        result = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, result


def _cuda_event_latency_and_peak_delta_mb(
    fn, device, *, warmup_iters=_BENCHMARK_WARMUP_ITERS, iters=_BENCHMARK_ITERS
):
    result = None
    for _ in range(warmup_iters):
        result = fn()
    torch.cuda.synchronize()

    baseline_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    wall_start = time.perf_counter()
    for _ in range(iters):
        result = fn()
    end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0 / iters
    peak_delta_mb = max(torch.cuda.max_memory_allocated(device) - baseline_allocated, 0) / (
        1024 * 1024
    )
    return start.elapsed_time(end) / iters, wall_ms, peak_delta_mb, result


def _cuda_event_backward_latency_and_peak_delta_mb(
    make_loss, device, *, warmup_iters=_BENCHMARK_WARMUP_ITERS, iters=_BENCHMARK_ITERS
):
    result = None
    for _ in range(warmup_iters):
        loss, result = make_loss()
        loss.backward()
    torch.cuda.synchronize()

    baseline_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    event_ms_total = 0.0
    wall_ms_total = 0.0
    for _ in range(iters):
        loss, result = make_loss()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        wall_start = time.perf_counter()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        wall_ms_total += (time.perf_counter() - wall_start) * 1000.0
        event_ms_total += start.elapsed_time(end)
    peak_delta_mb = max(torch.cuda.max_memory_allocated(device) - baseline_allocated, 0) / (
        1024 * 1024
    )
    return event_ms_total / iters, wall_ms_total / iters, peak_delta_mb, result


def _distributed_float_stats(local_value, device):
    local = torch.tensor(local_value, dtype=torch.float64, device=device)
    max_value = local.clone()
    sum_value = local.clone()
    torch.distributed.all_reduce(max_value, op=torch.distributed.ReduceOp.MAX)
    torch.distributed.all_reduce(sum_value, op=torch.distributed.ReduceOp.SUM)
    mean_value = sum_value / torch.distributed.get_world_size()
    return max_value.item(), mean_value.item()


def _distributed_int_stats(local_value, device, group=None):
    local = torch.tensor(local_value, dtype=torch.int64, device=device)
    max_value = local.clone()
    sum_value = local.clone()
    torch.distributed.all_reduce(max_value, op=torch.distributed.ReduceOp.MAX, group=group)
    torch.distributed.all_reduce(sum_value, op=torch.distributed.ReduceOp.SUM, group=group)
    return int(max_value.item()), int(sum_value.item())


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


def _require_symmetric_memory_expert_dispatcher():
    try:
        module = importlib.import_module("megatron.core.transformer.moe.expert_weight_dispatcher")
    except ModuleNotFoundError:
        pytest.fail(
            "megatron.core.transformer.moe.expert_weight_dispatcher is required for "
            "BalancedMoELayer expert dispatcher tests."
        )
    try:
        dispatcher_cls = module.SymmetricMemoryExpertWeightDispatcher
    except AttributeError:
        pytest.fail(
            "SymmetricMemoryExpertWeightDispatcher is required in expert_weight_dispatcher.py."
        )
    availability_error = dispatcher_cls.availability_error()
    if availability_error is not None:
        pytest.skip(f"Symmetric Memory expert dispatch is unavailable: {availability_error}")
    return dispatcher_cls


def _require_hybridep_expert_dispatcher():
    try:
        module = importlib.import_module("megatron.core.transformer.moe.expert_weight_dispatcher")
    except ModuleNotFoundError:
        pytest.fail(
            "megatron.core.transformer.moe.expert_weight_dispatcher is required for "
            "BalancedMoELayer HybridEP dispatcher tests."
        )
    try:
        dispatcher_cls = module.HybridEPExpertWeightDispatcher
    except AttributeError:
        pytest.fail("HybridEPExpertWeightDispatcher is required in expert_weight_dispatcher.py.")
    availability_error = dispatcher_cls.availability_error()
    if availability_error is not None:
        pytest.skip(f"HybridEP expert dispatch is unavailable: {availability_error}")
    return dispatcher_cls


def _make_config(dtype, *, grad_combine_dtype="fp32"):
    config = TransformerConfig(
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
    config.moe_balance_expert_weight_grad_combine_dtype = grad_combine_dtype
    return config


def _make_home_weight(global_home_id, shape, dtype, device):
    values = torch.arange(shape.numel(), dtype=torch.float32, device=device).reshape(shape)
    values = values + global_home_id * 100000
    return values.to(dtype).detach().requires_grad_(True)


def _gather_home_weights(local_home_weights, ep_group):
    local_stack = torch.stack([weight.detach() for weight in local_home_weights], dim=0)
    gathered = [
        torch.empty_like(local_stack) for _ in range(torch.distributed.get_world_size(ep_group))
    ]
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


def _alternate_expert_offloading_map(device):
    expert_map = torch.zeros(8, 8, dtype=torch.bool, device=device)
    expert_map[1, 0] = True
    expert_map[4, 1] = True
    expert_map[2, 3] = True
    expert_map[7, 4] = True
    expert_map[5, 6] = True
    return expert_map


def _single_home_all_spares_offloading_map(device):
    expert_map = torch.zeros(8, 8, dtype=torch.bool, device=device)
    expert_map[0, :] = True
    return expert_map


def _fp32_combine_probe_spare_grad(spare_idx, weight_shape, dtype, device):
    value = 256.0 if spare_idx == 0 else 1.0
    return torch.full(weight_shape, value, dtype=dtype, device=device)


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
        torch.testing.assert_close(
            dispatched_weight, torch.zeros_like(dispatched_weight), rtol=0, atol=0
        )
        return
    torch.testing.assert_close(dispatched_weight, expected_weight, rtol=0, atol=0)
    assert dispatched_weight.dtype == dtype


def _expected_home_grads(expert_map, weight_shape, dtype, device):
    expected = torch.zeros((8, *weight_shape), dtype=dtype, device=device)
    for home_idx, spare_idx in zip(*torch.where(expert_map)):
        scale = float(spare_idx.item() + 1)
        expected[home_idx] += torch.ones(weight_shape, dtype=dtype, device=device) * scale
    return expected


def _expected_home_grads_from_spare_grads(expert_map, weight_shape, device):
    expected_fp32 = torch.zeros((8, *weight_shape), dtype=torch.float32, device=device)
    expected_bf16_sequential = torch.zeros(
        (8, *weight_shape), dtype=torch.bfloat16, device=device
    )
    for home_idx, spare_idx in zip(*torch.where(expert_map)):
        spare_idx = int(spare_idx.item())
        home_idx = int(home_idx.item())
        spare_grad = _fp32_combine_probe_spare_grad(
            spare_idx, weight_shape, torch.bfloat16, device
        )
        expected_fp32[home_idx].add_(spare_grad.float())
        expected_bf16_sequential[home_idx].add_(spare_grad)
    return expected_fp32, expected_bf16_sequential


def _run_dispatch_loss(dispatched, expert_map, ep_rank, spare_per_rank, dtype, device):
    loss = torch.zeros((), dtype=dtype, device=device)
    for local_spare_idx, dispatched_weight in enumerate(dispatched):
        spare_global = ep_rank * spare_per_rank + local_spare_idx
        if not expert_map[:, spare_global].any():
            continue
        loss = loss + dispatched_weight.sum() * float(spare_global + 1)
    return loss


def _weight_payload_bytes(weight_shape, dtype, device):
    weight_numel = int(torch.Size(weight_shape).numel())
    element_size = torch.empty((), dtype=dtype, device=device).element_size()
    return weight_numel * element_size


def _global_payload_bytes(local_pair_count, bytes_per_pair, device):
    global_pair_count = torch.tensor(local_pair_count, dtype=torch.int64, device=device)
    torch.distributed.all_reduce(global_pair_count, op=torch.distributed.ReduceOp.SUM)
    return int(global_pair_count.item()) * bytes_per_pair


def _max_payload_bytes(local_pair_count, bytes_per_pair, device):
    max_pair_count, _ = _distributed_int_stats(local_pair_count, device)
    return max_pair_count * bytes_per_pair


def _unique_remote_send_pair_count(metadata, ep_rank):
    """Count unique source-side remote (local_home, destination rank) routes."""

    local_home_routes = metadata.local_to_global_routing_map
    count = 0
    for local_home_idx in range(local_home_routes.shape[0]):
        for dest_rank in range(local_home_routes.shape[1]):
            if dest_rank == ep_rank:
                continue
            if local_home_routes[local_home_idx, dest_rank].any().item():
                count += 1
    return count


def _symm_backward_remote_pair_count(metadata, ep_rank):
    return sum(
        1
        for dest_rank, _, _, _ in metadata.backward_grad_sources
        if dest_rank != ep_rank
    )


def _remote_recv_pair_count(metadata, ep_rank):
    return sum(count for rank, count in enumerate(metadata.output_splits) if rank != ep_rank)


def _hybridep_padded_weight_payload_bytes(dispatcher, reference_weight):
    chunk_size = dispatcher._weight_chunk_size(reference_weight)
    chunks_per_weight = math.ceil(reference_weight.numel() / chunk_size)
    return chunks_per_weight * chunk_size * reference_weight.element_size()


def _expert_dispatch_payload_stats(metadata, ep_rank, weight_shape, dtype, device):
    weight_bytes = _weight_payload_bytes(weight_shape, dtype, device)
    local_remote_send_pairs = sum(
        count for rank, count in enumerate(metadata.input_splits) if rank != ep_rank
    )
    local_remote_recv_pairs = sum(
        count for rank, count in enumerate(metadata.output_splits) if rank != ep_rank
    )

    total_remote_send_pairs = torch.tensor(local_remote_send_pairs, dtype=torch.int64, device=device)
    busiest_send_pairs = torch.tensor(local_remote_send_pairs, dtype=torch.int64, device=device)
    busiest_recv_pairs = torch.tensor(local_remote_recv_pairs, dtype=torch.int64, device=device)
    busiest_send_plus_recv_pairs = torch.tensor(
        local_remote_send_pairs + local_remote_recv_pairs, dtype=torch.int64, device=device
    )
    torch.distributed.all_reduce(total_remote_send_pairs, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(busiest_send_pairs, op=torch.distributed.ReduceOp.MAX)
    torch.distributed.all_reduce(busiest_recv_pairs, op=torch.distributed.ReduceOp.MAX)
    torch.distributed.all_reduce(
        busiest_send_plus_recv_pairs, op=torch.distributed.ReduceOp.MAX
    )

    normalized_remote_payload_bytes = int(total_remote_send_pairs.item()) * weight_bytes
    busiest_send_payload_bytes = int(busiest_send_pairs.item()) * weight_bytes
    busiest_recv_payload_bytes = int(busiest_recv_pairs.item()) * weight_bytes
    busiest_send_plus_recv_payload_bytes = (
        int(busiest_send_plus_recv_pairs.item()) * weight_bytes
    )
    logical_payload_bytes = int(metadata.global_routing_map.sum().item()) * weight_bytes
    return _RoutePayloadStats(
        normalized_remote_payload_bytes=normalized_remote_payload_bytes,
        busiest_send_payload_bytes=busiest_send_payload_bytes,
        busiest_recv_payload_bytes=busiest_recv_payload_bytes,
        busiest_send_plus_recv_payload_bytes=busiest_send_plus_recv_payload_bytes,
        logical_payload_bytes=logical_payload_bytes,
    )


def _gib_per_second(num_bytes, latency_ms):
    if latency_ms <= 0.0:
        return 0.0
    return num_bytes / (latency_ms / 1000.0) / (1024**3)


def _print_benchmark_line(
    *,
    case_name,
    phase,
    backend,
    dtype,
    weight_shape,
    event_mean_ms,
    event_max_ms,
    wall_mean_ms,
    wall_max_ms,
    mean_peak_mb,
    max_peak_mb,
    route_payload_stats,
    actual_remote_payload_bytes,
    actual_busiest_sender_payload_bytes,
    extra_fields=None,
):
    extra_text = ""
    if extra_fields:
        extra_text = " " + " ".join(f"{key}={value}" for key, value in extra_fields.items())
    print(
        "BENCHMARK balanced_moe_expert_weight_dispatch "
        f"case={case_name} phase={phase} backend={backend} "
        f"dtype={dtype} weight_shape={tuple(weight_shape)} "
        f"warmup_iters={_BENCHMARK_WARMUP_ITERS} iters={_BENCHMARK_ITERS} "
        f"event_mean_rank_ms={event_mean_ms:.4f} event_max_rank_ms={event_max_ms:.4f} "
        f"wall_mean_rank_ms={wall_mean_ms:.4f} wall_max_rank_ms={wall_max_ms:.4f} "
        f"mean_peak_delta_mb={mean_peak_mb:.2f} max_peak_delta_mb={max_peak_mb:.2f} "
        f"normalized_remote_payload_mib="
        f"{route_payload_stats.normalized_remote_payload_bytes / (1024**2):.2f} "
        f"normalized_remote_payload_algo_event_gib_s="
        f"{_gib_per_second(route_payload_stats.normalized_remote_payload_bytes, event_max_ms):.2f} "
        f"actual_remote_payload_mib={actual_remote_payload_bytes / (1024**2):.2f} "
        f"actual_remote_payload_algo_event_gib_s="
        f"{_gib_per_second(actual_remote_payload_bytes, event_max_ms):.2f} "
        f"actual_busiest_sender_payload_mib="
        f"{actual_busiest_sender_payload_bytes / (1024**2):.2f} "
        f"actual_bus_bw_event_gib_s="
        f"{_gib_per_second(actual_busiest_sender_payload_bytes, event_max_ms):.2f} "
        f"normalized_busiest_rank_send_mib="
        f"{route_payload_stats.busiest_send_payload_bytes / (1024**2):.2f} "
        f"normalized_busiest_rank_send_event_gib_s="
        f"{_gib_per_second(route_payload_stats.busiest_send_payload_bytes, event_max_ms):.2f} "
        f"normalized_busiest_rank_recv_mib="
        f"{route_payload_stats.busiest_recv_payload_bytes / (1024**2):.2f} "
        f"normalized_busiest_rank_recv_event_gib_s="
        f"{_gib_per_second(route_payload_stats.busiest_recv_payload_bytes, event_max_ms):.2f} "
        f"normalized_busiest_rank_send_plus_recv_mib="
        f"{route_payload_stats.busiest_send_plus_recv_payload_bytes / (1024**2):.2f} "
        f"normalized_busiest_rank_send_plus_recv_event_gib_s="
        f"{_gib_per_second(route_payload_stats.busiest_send_plus_recv_payload_bytes, event_max_ms):.2f} "
        f"logical_dispatch_payload_mib="
        f"{route_payload_stats.logical_payload_bytes / (1024**2):.2f} "
        f"logical_dispatch_payload_algo_event_gib_s="
        f"{_gib_per_second(route_payload_stats.logical_payload_bytes, event_max_ms):.2f}"
        f"{extra_text}",
        flush=True,
    )


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


@pytest.mark.internal
@pytest.mark.parametrize("backend", ["all_to_all", "symmetric_memory"])
def test_bf16_expert_weight_dispatch_combines_weight_grads_in_fp32(backend):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        if backend == "symmetric_memory":
            dispatcher_cls = _require_symmetric_memory_expert_dispatcher()
        else:
            dispatcher_cls = _require_expert_dispatcher()

        pg_collection = get_default_pg_collection()
        ep_rank = pg_collection.ep.rank()
        device = torch.device("cuda", torch.cuda.current_device())
        dtype = torch.bfloat16
        weight_shape = torch.Size([5, 6])
        local_home_weights = [
            _make_home_weight(ep_rank * 2 + local_idx, weight_shape, dtype, device)
            for local_idx in range(2)
        ]
        expert_map = _single_home_all_spares_offloading_map(device)

        dispatcher = dispatcher_cls(
            config=_make_config(dtype),
            ep_group=pg_collection.ep,
            num_home_experts=8,
            num_spare_experts=8,
        )
        metadata = dispatcher.preprocess(expert_map)
        dispatched = dispatcher.dispatch(metadata, *local_home_weights)
        grad_outputs = []
        for local_spare_idx, dispatched_weight in enumerate(dispatched):
            spare_global = ep_rank * 2 + local_spare_idx
            grad_outputs.append(
                _fp32_combine_probe_spare_grad(spare_global, weight_shape, dtype, device)
                if expert_map[:, spare_global].any()
                else torch.zeros_like(dispatched_weight)
            )

        torch.autograd.backward(dispatched, grad_outputs)

        expected_fp32, expected_bf16_sequential = _expected_home_grads_from_spare_grads(
            expert_map, weight_shape, device
        )
        expected_local_fp32 = expected_fp32[ep_rank * 2 : (ep_rank + 1) * 2]
        expected_local_bf16_sequential = expected_bf16_sequential[
            ep_rank * 2 : (ep_rank + 1) * 2
        ]

        for weight, expected_grad in zip(local_home_weights, expected_local_fp32):
            assert weight.grad is not None
            torch.testing.assert_close(
                weight.grad, expected_grad.to(dtype=weight.grad.dtype), rtol=0, atol=0
            )

        if ep_rank == 0:
            assert not torch.equal(
                expected_local_bf16_sequential[0], expected_local_fp32[0].to(torch.bfloat16)
            )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize("backend", ["all_to_all", "symmetric_memory"])
def test_bf16_expert_weight_dispatch_grad_combine_dtype_is_configurable(backend):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        if backend == "symmetric_memory":
            dispatcher_cls = _require_symmetric_memory_expert_dispatcher()
        else:
            dispatcher_cls = _require_expert_dispatcher()

        pg_collection = get_default_pg_collection()
        ep_rank = pg_collection.ep.rank()
        device = torch.device("cuda", torch.cuda.current_device())
        dtype = torch.bfloat16
        weight_shape = torch.Size([5, 6])
        local_home_weights = [
            _make_home_weight(ep_rank * 2 + local_idx, weight_shape, dtype, device)
            for local_idx in range(2)
        ]
        expert_map = _single_home_all_spares_offloading_map(device)

        dispatcher = dispatcher_cls(
            config=_make_config(dtype, grad_combine_dtype="bf16"),
            ep_group=pg_collection.ep,
            num_home_experts=8,
            num_spare_experts=8,
        )
        metadata = dispatcher.preprocess(expert_map)
        dispatched = dispatcher.dispatch(metadata, *local_home_weights)
        grad_outputs = []
        for local_spare_idx, dispatched_weight in enumerate(dispatched):
            spare_global = ep_rank * 2 + local_spare_idx
            grad_outputs.append(
                _fp32_combine_probe_spare_grad(spare_global, weight_shape, dtype, device)
                if expert_map[:, spare_global].any()
                else torch.zeros_like(dispatched_weight)
            )

        torch.autograd.backward(dispatched, grad_outputs)

        expected_fp32, expected_bf16_sequential = _expected_home_grads_from_spare_grads(
            expert_map, weight_shape, device
        )
        expected_local_fp32 = expected_fp32[ep_rank * 2 : (ep_rank + 1) * 2]
        expected_local_bf16_sequential = expected_bf16_sequential[
            ep_rank * 2 : (ep_rank + 1) * 2
        ]

        for weight, expected_grad in zip(local_home_weights, expected_local_bf16_sequential):
            assert weight.grad is not None
            torch.testing.assert_close(weight.grad, expected_grad, rtol=0, atol=0)

        if ep_rank == 0:
            assert not torch.equal(
                expected_local_bf16_sequential[0], expected_local_fp32[0].to(dtype)
            )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize("backend", ["all_to_all", "symmetric_memory"])
def test_bf16_runtime_expert_weight_grad_edge_is_fp32(backend):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        if backend == "symmetric_memory":
            dispatcher_cls = _require_symmetric_memory_expert_dispatcher()
        else:
            dispatcher_cls = _require_expert_dispatcher()

        balanced_module = importlib.import_module(
            "megatron.core.transformer.moe.balanced_moe_layer"
        )
        cast_runtime_weights = balanced_module._cast_runtime_expert_weights_for_forward

        pg_collection = get_default_pg_collection()
        ep_rank = pg_collection.ep.rank()
        device = torch.device("cuda", torch.cuda.current_device())
        forward_dtype = torch.bfloat16
        weight_shape = torch.Size([5, 6])
        local_home_weights = [
            _make_home_weight(ep_rank * 2 + local_idx, weight_shape, forward_dtype, device)
            for local_idx in range(2)
        ]
        expert_map = _single_home_all_spares_offloading_map(device)

        dispatcher = dispatcher_cls(
            config=_make_config(forward_dtype),
            ep_group=pg_collection.ep,
            num_home_experts=8,
            num_spare_experts=8,
        )
        metadata = dispatcher.preprocess(expert_map)
        dispatched = dispatcher.dispatch(
            metadata, *local_home_weights, runtime_weight_dtype=torch.float32
        )

        observed_grad_dtypes = []
        hook_handles = []
        for local_spare_idx, dispatched_weight in enumerate(dispatched):
            spare_global = ep_rank * 2 + local_spare_idx
            if not expert_map[:, spare_global].any():
                continue
            assert dispatched_weight.dtype is torch.float32

            def record_grad_dtype(grad):
                observed_grad_dtypes.append(grad.dtype)
                return grad

            hook_handles.append(dispatched_weight.register_hook(record_grad_dtype))

        runtime_weights = cast_runtime_weights(dispatched, forward_dtype)
        grad_outputs = []
        for local_spare_idx, runtime_weight in enumerate(runtime_weights):
            assert runtime_weight.dtype is forward_dtype
            spare_global = ep_rank * 2 + local_spare_idx
            grad_outputs.append(
                _fp32_combine_probe_spare_grad(
                    spare_global, weight_shape, forward_dtype, device
                )
                if expert_map[:, spare_global].any()
                else torch.zeros_like(runtime_weight)
            )

        torch.autograd.backward(runtime_weights, grad_outputs)
        for handle in hook_handles:
            handle.remove()

        assert observed_grad_dtypes
        assert all(dtype is torch.float32 for dtype in observed_grad_dtypes)

        expected_fp32, _ = _expected_home_grads_from_spare_grads(
            expert_map, weight_shape, device
        )
        expected_local_fp32 = expected_fp32[ep_rank * 2 : (ep_rank + 1) * 2]
        for weight, expected_grad in zip(local_home_weights, expected_local_fp32):
            assert weight.grad is not None
            torch.testing.assert_close(
                weight.grad, expected_grad.to(dtype=weight.grad.dtype), rtol=0, atol=0
            )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
def test_expert_weight_dispatch_uses_all_to_all_not_full_collectives(monkeypatch):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        dispatcher_cls = _require_expert_dispatcher()
        pg_collection = get_default_pg_collection()
        ep_rank = pg_collection.ep.rank()
        device = torch.device("cuda", torch.cuda.current_device())
        dtype = torch.float32
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
        local_home_slice = expert_map[ep_rank * 2 : (ep_rank + 1) * 2]
        local_spare_slice = expert_map[:, ep_rank * 2 : (ep_rank + 1) * 2]
        assert sum(metadata.input_splits) == int(local_home_slice.sum().item())
        assert sum(metadata.output_splits) == int(local_spare_slice.sum().item())

        calls = {"all_to_all_single": 0}
        original_all_to_all_single = torch.distributed.all_to_all_single

        def counted_all_to_all_single(*args, **kwargs):
            calls["all_to_all_single"] += 1
            return original_all_to_all_single(*args, **kwargs)

        def forbidden_collective(*args, **kwargs):
            raise AssertionError("expert weight dispatch must not use full collectives")

        monkeypatch.setattr(torch.distributed, "all_to_all_single", counted_all_to_all_single)
        monkeypatch.setattr(torch.distributed, "all_gather", forbidden_collective)
        monkeypatch.setattr(torch.distributed, "all_reduce", forbidden_collective)

        dispatched = dispatcher.dispatch(metadata, *local_home_weights)
        expected = _expected_local_spare_weights(expert_map, all_home_weights, ep_rank, 2)
        for dispatched_weight, expected_weight in zip(dispatched, expected):
            _assert_inactive_or_expected(
                dispatched_weight, expected_weight, weight_shape, dtype=dtype
            )

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
            torch.testing.assert_close(weight.grad, expected_grad, rtol=1e-5, atol=1e-6)
        assert calls["all_to_all_single"] == 2
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_symmetric_memory_expert_weight_dispatch_matches_native(dtype):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        native_dispatcher_cls = _require_expert_dispatcher()
        symm_dispatcher_cls = _require_symmetric_memory_expert_dispatcher()
        pg_collection = get_default_pg_collection()
        ep_rank = pg_collection.ep.rank()
        device = torch.device("cuda", torch.cuda.current_device())
        weight_shape = torch.Size([5, 6])
        local_home_weights = [
            _make_home_weight(ep_rank * 2 + local_idx, weight_shape, dtype, device)
            for local_idx in range(2)
        ]
        expert_map = _expert_offloading_map(device)

        config = _make_config(dtype)
        native_dispatcher = native_dispatcher_cls(
            config=config, ep_group=pg_collection.ep, num_home_experts=8, num_spare_experts=8
        )
        symm_dispatcher = symm_dispatcher_cls(
            config=config, ep_group=pg_collection.ep, num_home_experts=8, num_spare_experts=8
        )
        native_dispatched = native_dispatcher.dispatch(
            native_dispatcher.preprocess(expert_map), *local_home_weights
        )
        symm_dispatched = symm_dispatcher.dispatch(
            symm_dispatcher.preprocess(expert_map), *local_home_weights
        )

        for native_weight, symm_weight in zip(native_dispatched, symm_dispatched):
            torch.testing.assert_close(symm_weight, native_weight, rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_symmetric_memory_expert_weight_dispatch_grad_matches_native(dtype):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        native_dispatcher_cls = _require_expert_dispatcher()
        symm_dispatcher_cls = _require_symmetric_memory_expert_dispatcher()
        pg_collection = get_default_pg_collection()
        ep_rank = pg_collection.ep.rank()
        device = torch.device("cuda", torch.cuda.current_device())
        weight_shape = torch.Size([5, 6])
        native_home_weights = [
            _make_home_weight(ep_rank * 2 + local_idx, weight_shape, dtype, device)
            for local_idx in range(2)
        ]
        symm_home_weights = [
            weight.detach().clone().requires_grad_(True) for weight in native_home_weights
        ]
        expert_map = _expert_offloading_map(device)

        config = _make_config(dtype)
        native_dispatcher = native_dispatcher_cls(
            config=config, ep_group=pg_collection.ep, num_home_experts=8, num_spare_experts=8
        )
        symm_dispatcher = symm_dispatcher_cls(
            config=config, ep_group=pg_collection.ep, num_home_experts=8, num_spare_experts=8
        )
        native_dispatched = native_dispatcher.dispatch(
            native_dispatcher.preprocess(expert_map), *native_home_weights
        )
        symm_dispatched = symm_dispatcher.dispatch(
            symm_dispatcher.preprocess(expert_map), *symm_home_weights
        )

        native_loss = _run_dispatch_loss(native_dispatched, expert_map, ep_rank, 2, dtype, device)
        symm_loss = _run_dispatch_loss(symm_dispatched, expert_map, ep_rank, 2, dtype, device)
        native_loss.backward()
        symm_loss.backward()

        for native_weight, symm_weight in zip(native_home_weights, symm_home_weights):
            assert native_weight.grad is not None
            assert symm_weight.grad is not None
            if dtype is torch.bfloat16:
                torch.testing.assert_close(
                    symm_weight.grad, native_weight.grad, rtol=2e-2, atol=2e-2
                )
            elif dtype is torch.float32:
                torch.testing.assert_close(
                    symm_weight.grad, native_weight.grad, rtol=1e-5, atol=1e-6
                )
            else:
                torch.testing.assert_close(symm_weight.grad, native_weight.grad, rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
def test_symmetric_memory_expert_weight_dispatch_active_staging_and_remote_read_dedup():
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        symm_dispatcher_cls = _require_symmetric_memory_expert_dispatcher()
        pg_collection = get_default_pg_collection()
        ep_rank = pg_collection.ep.rank()
        device = torch.device("cuda", torch.cuda.current_device())
        dtype = torch.float32
        weight_shape = torch.Size([5, 6])
        local_home_weights = [
            _make_home_weight(ep_rank * 2 + local_idx, weight_shape, dtype, device)
            for local_idx in range(2)
        ]
        all_home_weights = _gather_home_weights(local_home_weights, pg_collection.ep)
        expert_map = _expert_offloading_map(device)

        dispatcher = symm_dispatcher_cls(
            config=_make_config(dtype),
            ep_group=pg_collection.ep,
            num_home_experts=8,
            num_spare_experts=8,
        )
        metadata = dispatcher.preprocess(expert_map)
        dispatched = dispatcher.dispatch(metadata, *local_home_weights)

        expected = _expected_local_spare_weights(expert_map, all_home_weights, ep_rank, 2)
        for dispatched_weight, expected_weight in zip(dispatched, expected):
            _assert_inactive_or_expected(
                dispatched_weight, expected_weight, weight_shape, dtype=dtype
            )

        expected_duplicate_reads_avoided = sum(
            1
            for local_spare_idx, alias in enumerate(metadata.remote_spare_aliases)
            if alias is not None and alias != local_spare_idx
        )
        assert dispatcher._debug_staged_home_experts == len(metadata.active_local_home_indices)
        assert dispatcher._debug_unique_remote_sources == len(metadata.unique_remote_source_indices)
        assert dispatcher._debug_duplicate_remote_reads_avoided == expected_duplicate_reads_avoided
        assert dispatcher._debug_low_level_get_calls == len(metadata.unique_remote_source_indices)

        _, total_staged = _distributed_int_stats(
            dispatcher._debug_staged_home_experts, device, group=pg_collection.ep
        )
        _, total_unique_remote_sources = _distributed_int_stats(
            dispatcher._debug_unique_remote_sources, device, group=pg_collection.ep
        )
        _, total_duplicate_reads_avoided = _distributed_int_stats(
            dispatcher._debug_duplicate_remote_reads_avoided, device, group=pg_collection.ep
        )
        _, total_get_calls = _distributed_int_stats(
            dispatcher._debug_low_level_get_calls, device, group=pg_collection.ep
        )
        assert total_staged == 3
        assert total_unique_remote_sources == 3
        assert total_duplicate_reads_avoided == 1
        assert total_get_calls == 3

        loss = _run_dispatch_loss(dispatched, expert_map, ep_rank, 2, dtype, device)
        loss.backward()

        expected_global_grads = _expected_home_grads(expert_map, weight_shape, dtype, device)
        expected_local_grads = expected_global_grads[ep_rank * 2 : (ep_rank + 1) * 2]
        for weight, expected_grad in zip(local_home_weights, expected_local_grads):
            assert weight.grad is not None
            torch.testing.assert_close(weight.grad, expected_grad, rtol=1e-5, atol=1e-6)
        assert dispatcher._debug_backward_schedule_entries == len(metadata.backward_grad_sources)
        _, total_backward_schedule_entries = _distributed_int_stats(
            dispatcher._debug_backward_schedule_entries, device, group=pg_collection.ep
        )
        assert total_backward_schedule_entries == 5
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
def test_symmetric_memory_expert_weight_dispatch_rejects_malformed_backward_schedule():
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        symm_dispatcher_cls = _require_symmetric_memory_expert_dispatcher()
        pg_collection = get_default_pg_collection()
        ep_rank = pg_collection.ep.rank()
        device = torch.device("cuda", torch.cuda.current_device())
        dtype = torch.float32
        weight_shape = torch.Size([5, 6])
        local_home_weights = [
            _make_home_weight(ep_rank * 2 + local_idx, weight_shape, dtype, device)
            for local_idx in range(2)
        ]
        expert_map = _expert_offloading_map(device)

        dispatcher = symm_dispatcher_cls(
            config=_make_config(dtype),
            ep_group=pg_collection.ep,
            num_home_experts=8,
            num_spare_experts=8,
        )
        metadata = dispatcher.preprocess(expert_map)
        dispatched = dispatcher.dispatch(metadata, *local_home_weights)
        metadata.backward_grad_sources = [(0, dispatcher.num_local_spare_experts, 0, 0)]
        loss = _run_dispatch_loss(dispatched, expert_map, ep_rank, 2, dtype, device)

        with pytest.raises(RuntimeError, match="invalid local spare index"):
            loss.backward()
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
def test_symmetric_memory_expert_weight_dispatch_reuses_workspace_without_stale_data():
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        symm_dispatcher_cls = _require_symmetric_memory_expert_dispatcher()
        pg_collection = get_default_pg_collection()
        ep_rank = pg_collection.ep.rank()
        device = torch.device("cuda", torch.cuda.current_device())
        dtype = torch.float32
        weight_shape = torch.Size([5, 6])
        expert_map_a = _expert_offloading_map(device)
        expert_map_b = _alternate_expert_offloading_map(device)

        dispatcher = symm_dispatcher_cls(
            config=_make_config(dtype),
            ep_group=pg_collection.ep,
            num_home_experts=8,
            num_spare_experts=8,
        )

        def make_shifted_home_weights(offset):
            return [
                (
                    _make_home_weight(ep_rank * 2 + local_idx, weight_shape, dtype, device)
                    + offset
                )
                .detach()
                .requires_grad_(True)
                for local_idx in range(2)
            ]

        for expert_map, offset in (
            (expert_map_a, 0.0),
            (expert_map_b, 17.0),
            (expert_map_a, 29.0),
        ):
            local_home_weights = make_shifted_home_weights(offset)
            all_home_weights = _gather_home_weights(local_home_weights, pg_collection.ep)
            metadata = dispatcher.preprocess(expert_map)
            dispatched = dispatcher.dispatch(metadata, *local_home_weights)
            expected = _expected_local_spare_weights(expert_map, all_home_weights, ep_rank, 2)
            for dispatched_weight, expected_weight in zip(dispatched, expected):
                _assert_inactive_or_expected(
                    dispatched_weight, expected_weight, weight_shape, dtype=dtype
                )

        assert dispatcher._debug_home_weight_barriers == 1
        assert dispatcher._debug_workspace_sync_mode == "double_buffered_stage_barrier"
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_hybridep_expert_weight_dispatch_matches_native(dtype):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        native_dispatcher_cls = _require_expert_dispatcher()
        hybridep_dispatcher_cls = _require_hybridep_expert_dispatcher()
        pg_collection = get_default_pg_collection()
        ep_rank = pg_collection.ep.rank()
        device = torch.device("cuda", torch.cuda.current_device())
        weight_shape = torch.Size([5, 6])
        local_home_weights = [
            _make_home_weight(ep_rank * 2 + local_idx, weight_shape, dtype, device)
            for local_idx in range(2)
        ]
        expert_map = _expert_offloading_map(device)

        config = _make_config(dtype)
        native_dispatcher = native_dispatcher_cls(
            config=config, ep_group=pg_collection.ep, num_home_experts=8, num_spare_experts=8
        )
        hybridep_dispatcher = hybridep_dispatcher_cls(
            config=config, ep_group=pg_collection.ep, num_home_experts=8, num_spare_experts=8
        )
        native_dispatched = native_dispatcher.dispatch(
            native_dispatcher.preprocess(expert_map), *local_home_weights
        )
        hybridep_dispatched = hybridep_dispatcher.dispatch(
            hybridep_dispatcher.preprocess(expert_map), *local_home_weights
        )

        for native_weight, hybridep_weight in zip(native_dispatched, hybridep_dispatched):
            torch.testing.assert_close(hybridep_weight, native_weight, rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_hybridep_expert_weight_dispatch_grad_matches_native(dtype):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        native_dispatcher_cls = _require_expert_dispatcher()
        hybridep_dispatcher_cls = _require_hybridep_expert_dispatcher()
        pg_collection = get_default_pg_collection()
        ep_rank = pg_collection.ep.rank()
        device = torch.device("cuda", torch.cuda.current_device())
        weight_shape = torch.Size([5, 6])
        native_home_weights = [
            _make_home_weight(ep_rank * 2 + local_idx, weight_shape, dtype, device)
            for local_idx in range(2)
        ]
        hybridep_home_weights = [
            weight.detach().clone().requires_grad_(True) for weight in native_home_weights
        ]
        expert_map = _expert_offloading_map(device)

        config = _make_config(dtype)
        native_dispatcher = native_dispatcher_cls(
            config=config, ep_group=pg_collection.ep, num_home_experts=8, num_spare_experts=8
        )
        hybridep_dispatcher = hybridep_dispatcher_cls(
            config=config, ep_group=pg_collection.ep, num_home_experts=8, num_spare_experts=8
        )
        native_dispatched = native_dispatcher.dispatch(
            native_dispatcher.preprocess(expert_map), *native_home_weights
        )
        hybridep_dispatched = hybridep_dispatcher.dispatch(
            hybridep_dispatcher.preprocess(expert_map), *hybridep_home_weights
        )

        native_loss = _run_dispatch_loss(native_dispatched, expert_map, ep_rank, 2, dtype, device)
        hybridep_loss = _run_dispatch_loss(
            hybridep_dispatched, expert_map, ep_rank, 2, dtype, device
        )
        native_loss.backward()
        hybridep_loss.backward()

        for native_weight, hybridep_weight in zip(native_home_weights, hybridep_home_weights):
            assert native_weight.grad is not None
            assert hybridep_weight.grad is not None
            if dtype is torch.bfloat16:
                torch.testing.assert_close(
                    hybridep_weight.grad, native_weight.grad, rtol=2e-2, atol=2e-2
                )
            else:
                torch.testing.assert_close(
                    hybridep_weight.grad, native_weight.grad, rtol=1e-5, atol=1e-6
                )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_hybridep_expert_weight_dispatch_recompute_matches_native(monkeypatch, dtype):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        native_dispatcher_cls = _require_expert_dispatcher()
        hybridep_dispatcher_cls = _require_hybridep_expert_dispatcher()
        pg_collection = get_default_pg_collection()
        ep_rank = pg_collection.ep.rank()
        device = torch.device("cuda", torch.cuda.current_device())
        weight_shape = torch.Size([5, 6])
        native_home_weights = [
            _make_home_weight(ep_rank * 2 + local_idx, weight_shape, dtype, device)
            for local_idx in range(2)
        ]
        hybridep_home_weights = [
            weight.detach().clone().requires_grad_(True) for weight in native_home_weights
        ]
        expert_map = _expert_offloading_map(device)
        config = _make_config(dtype)

        native_dispatcher = native_dispatcher_cls(
            config=config, ep_group=pg_collection.ep, num_home_experts=8, num_spare_experts=8
        )
        native_metadata = native_dispatcher.preprocess(expert_map)
        native_dispatched = native_dispatcher.dispatch(native_metadata, *native_home_weights)
        native_loss = _run_dispatch_loss(native_dispatched, expert_map, ep_rank, 2, dtype, device)
        native_loss.backward()

        hybridep_dispatcher = hybridep_dispatcher_cls(
            config=config, ep_group=pg_collection.ep, num_home_experts=8, num_spare_experts=8
        )
        hybridep_metadata = hybridep_dispatcher.preprocess(expert_map)
        dispatch_calls = {"no_grad": 0, "grad": 0}
        original_dispatch = hybridep_dispatcher_cls.dispatch

        def counted_dispatch(self, metadata, *expert_weights):
            if torch.is_grad_enabled():
                dispatch_calls["grad"] += 1
            else:
                dispatch_calls["no_grad"] += 1
            return original_dispatch(self, metadata, *expert_weights)

        monkeypatch.setattr(hybridep_dispatcher_cls, "dispatch", counted_dispatch)
        checkpoint = tensor_parallel.CheckpointWithoutOutput()

        def checkpointed_dispatch(*weights):
            return tuple(hybridep_dispatcher.dispatch(hybridep_metadata, *weights))

        hybridep_dispatched = list(
            checkpoint.checkpoint(checkpointed_dispatch, *hybridep_home_weights)
        )
        for native_weight, hybridep_weight in zip(native_dispatched, hybridep_dispatched):
            torch.testing.assert_close(hybridep_weight, native_weight, rtol=0, atol=0)

        hybridep_loss = _run_dispatch_loss(
            hybridep_dispatched, expert_map, ep_rank, 2, dtype, device
        )
        checkpoint.discard_output_and_register_recompute(hybridep_loss)
        hybridep_loss.backward()

        for native_weight, hybridep_weight in zip(native_home_weights, hybridep_home_weights):
            assert native_weight.grad is not None
            assert hybridep_weight.grad is not None
            if dtype is torch.bfloat16:
                torch.testing.assert_close(
                    hybridep_weight.grad, native_weight.grad, rtol=2e-2, atol=2e-2
                )
            else:
                torch.testing.assert_close(
                    hybridep_weight.grad, native_weight.grad, rtol=1e-5, atol=1e-6
                )
        assert dispatch_calls["no_grad"] == 1
        assert dispatch_calls["grad"] == 1
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_symmetric_memory_expert_weight_dispatch_recompute_matches_native(monkeypatch, dtype):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        native_dispatcher_cls = _require_expert_dispatcher()
        symm_dispatcher_cls = _require_symmetric_memory_expert_dispatcher()
        pg_collection = get_default_pg_collection()
        ep_rank = pg_collection.ep.rank()
        device = torch.device("cuda", torch.cuda.current_device())
        weight_shape = torch.Size([5, 6])
        native_home_weights = [
            _make_home_weight(ep_rank * 2 + local_idx, weight_shape, dtype, device)
            for local_idx in range(2)
        ]
        symm_home_weights = [
            weight.detach().clone().requires_grad_(True) for weight in native_home_weights
        ]
        expert_map = _expert_offloading_map(device)
        config = _make_config(dtype)

        native_dispatcher = native_dispatcher_cls(
            config=config, ep_group=pg_collection.ep, num_home_experts=8, num_spare_experts=8
        )
        native_metadata = native_dispatcher.preprocess(expert_map)
        native_dispatched = native_dispatcher.dispatch(native_metadata, *native_home_weights)
        native_loss = _run_dispatch_loss(native_dispatched, expert_map, ep_rank, 2, dtype, device)
        native_loss.backward()

        symm_dispatcher = symm_dispatcher_cls(
            config=config, ep_group=pg_collection.ep, num_home_experts=8, num_spare_experts=8
        )
        symm_metadata = symm_dispatcher.preprocess(expert_map)
        symm_dispatcher._get_workspace(symm_home_weights[0])

        dispatch_calls = {"no_grad": 0, "grad": 0}
        original_dispatch = symm_dispatcher_cls.dispatch

        def counted_dispatch(self, metadata, *expert_weights):
            if torch.is_grad_enabled():
                dispatch_calls["grad"] += 1
            else:
                dispatch_calls["no_grad"] += 1
            return original_dispatch(self, metadata, *expert_weights)

        monkeypatch.setattr(symm_dispatcher_cls, "dispatch", counted_dispatch)

        symm_mem = importlib.import_module("torch.distributed._symmetric_memory")
        original_get = getattr(symm_mem, "get", None)
        get_calls = {"count": 0}

        def counted_get(*args, **kwargs):
            get_calls["count"] += 1
            return original_get(*args, **kwargs)

        def forbidden_collective(*args, **kwargs):
            raise AssertionError("Symmetric Memory recompute dispatch must use low-level get.")

        if original_get is not None:
            monkeypatch.setattr(symm_mem, "get", counted_get)
        monkeypatch.setattr(torch.distributed, "all_to_all_single", forbidden_collective)
        monkeypatch.setattr(torch.distributed, "all_gather", forbidden_collective)
        monkeypatch.setattr(torch.distributed, "all_reduce", forbidden_collective)
        if hasattr(torch.ops, "symm_mem"):
            monkeypatch.setattr(
                torch.ops.symm_mem, "all_to_all_vdev", forbidden_collective, raising=False
            )
            monkeypatch.setattr(
                torch.ops.symm_mem, "all_to_all_vdev_2d", forbidden_collective, raising=False
            )
            monkeypatch.setattr(
                torch.ops.symm_mem, "all_to_all_vdev_2d_offset", forbidden_collective, raising=False
            )

        checkpoint = tensor_parallel.CheckpointWithoutOutput()

        def checkpointed_dispatch(*weights):
            return tuple(symm_dispatcher.dispatch(symm_metadata, *weights))

        symm_dispatched = list(checkpoint.checkpoint(checkpointed_dispatch, *symm_home_weights))
        for native_weight, symm_weight in zip(native_dispatched, symm_dispatched):
            torch.testing.assert_close(symm_weight, native_weight, rtol=0, atol=0)

        symm_loss = _run_dispatch_loss(symm_dispatched, expert_map, ep_rank, 2, dtype, device)
        checkpoint.discard_output_and_register_recompute(symm_loss)
        symm_loss.backward()

        for native_weight, symm_weight in zip(native_home_weights, symm_home_weights):
            assert native_weight.grad is not None
            assert symm_weight.grad is not None
            if dtype is torch.bfloat16:
                torch.testing.assert_close(
                    symm_weight.grad, native_weight.grad, rtol=2e-2, atol=2e-2
                )
            else:
                torch.testing.assert_close(
                    symm_weight.grad, native_weight.grad, rtol=1e-5, atol=1e-6
                )

        assert dispatch_calls["no_grad"] == 1
        assert dispatch_calls["grad"] == 1
        assert symm_dispatcher._debug_low_level_get_calls > 0
        if original_get is not None:
            assert get_calls["count"] > 0
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
def test_symmetric_memory_expert_weight_dispatch_uses_get_not_a2a(monkeypatch):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        symm_dispatcher_cls = _require_symmetric_memory_expert_dispatcher()
        symm_mem = importlib.import_module("torch.distributed._symmetric_memory")
        pg_collection = get_default_pg_collection()
        ep_rank = pg_collection.ep.rank()
        device = torch.device("cuda", torch.cuda.current_device())
        dtype = torch.float32
        weight_shape = torch.Size([5, 6])
        local_home_weights = [
            _make_home_weight(ep_rank * 2 + local_idx, weight_shape, dtype, device)
            for local_idx in range(2)
        ]
        all_home_weights = _gather_home_weights(local_home_weights, pg_collection.ep)
        expert_map = _expert_offloading_map(device)
        dispatcher = symm_dispatcher_cls(
            config=_make_config(dtype),
            ep_group=pg_collection.ep,
            num_home_experts=8,
            num_spare_experts=8,
        )
        dispatcher._get_workspace(local_home_weights[0])

        get_calls = {"count": 0}
        original_get = getattr(symm_mem, "get", None)

        def counted_get(*args, **kwargs):
            get_calls["count"] += 1
            return original_get(*args, **kwargs)

        def forbidden_collective(*args, **kwargs):
            raise AssertionError("Symmetric Memory expert dispatch must use low-level get.")

        if original_get is not None:
            monkeypatch.setattr(symm_mem, "get", counted_get)
        monkeypatch.setattr(torch.distributed, "all_to_all_single", forbidden_collective)
        monkeypatch.setattr(torch.distributed, "all_gather", forbidden_collective)
        monkeypatch.setattr(torch.distributed, "all_reduce", forbidden_collective)
        if hasattr(torch.ops, "symm_mem"):
            monkeypatch.setattr(
                torch.ops.symm_mem, "all_to_all_vdev", forbidden_collective, raising=False
            )
            monkeypatch.setattr(
                torch.ops.symm_mem, "all_to_all_vdev_2d", forbidden_collective, raising=False
            )
            monkeypatch.setattr(
                torch.ops.symm_mem, "all_to_all_vdev_2d_offset", forbidden_collective, raising=False
            )

        metadata = dispatcher.preprocess(expert_map)
        dispatched = dispatcher.dispatch(metadata, *local_home_weights)
        expected = _expected_local_spare_weights(expert_map, all_home_weights, ep_rank, 2)
        for dispatched_weight, expected_weight in zip(dispatched, expected):
            _assert_inactive_or_expected(
                dispatched_weight, expected_weight, weight_shape, dtype=dtype
            )

        loss = _run_dispatch_loss(dispatched, expert_map, ep_rank, 2, dtype, device)
        loss.backward()

        expected_global_grads = _expected_home_grads(expert_map, weight_shape, dtype, device)
        expected_local_grads = expected_global_grads[ep_rank * 2 : (ep_rank + 1) * 2]
        for weight, expected_grad in zip(local_home_weights, expected_local_grads):
            assert weight.grad is not None
            torch.testing.assert_close(weight.grad, expected_grad, rtol=1e-5, atol=1e-6)
        assert dispatcher._debug_low_level_get_calls > 0
        if original_get is not None:
            assert get_calls["count"] > 0
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize("case_name, weight_shape", _BENCHMARK_WEIGHT_CASES)
def test_expert_weight_dispatch_latency_benchmark(case_name, weight_shape):
    _require_distributed_cuda()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        dispatcher_module = importlib.import_module(
            "megatron.core.transformer.moe.expert_weight_dispatcher"
        )
        native_dispatcher_cls = dispatcher_module.AllToAllExpertWeightDispatcher
        symm_dispatcher_cls = getattr(
            dispatcher_module, "SymmetricMemoryExpertWeightDispatcher", None
        )
        hybridep_dispatcher_cls = getattr(dispatcher_module, "HybridEPExpertWeightDispatcher", None)
        pg_collection = get_default_pg_collection()
        ep_rank = pg_collection.ep.rank()
        global_rank = torch.distributed.get_rank()
        device = torch.device("cuda", torch.cuda.current_device())
        dtype = torch.bfloat16
        expert_map = _expert_offloading_map(device)
        config = _make_config(dtype)
        weight_bytes = _weight_payload_bytes(weight_shape, dtype, device)

        def make_home_weights():
            return [
                _make_home_weight(ep_rank * 2 + local_idx, weight_shape, dtype, device)
                for local_idx in range(2)
            ]

        local_home_weights = make_home_weights()

        def make_dispatch_loss(dispatcher, metadata):
            weights = make_home_weights()
            dispatched = dispatcher.dispatch(metadata, *weights)
            loss = _run_dispatch_loss(dispatched, expert_map, ep_rank, 2, dtype, device)
            return loss, weights

        def dispatch_gradients(dispatcher, metadata):
            loss, weights = make_dispatch_loss(dispatcher, metadata)
            loss.backward()
            return [weight.grad.detach().clone() for weight in weights]

        def assert_gradients_match(candidate_dispatcher, candidate_metadata):
            native_grads = dispatch_gradients(native_dispatcher, native_metadata)
            candidate_grads = dispatch_gradients(candidate_dispatcher, candidate_metadata)
            for native_grad, candidate_grad in zip(native_grads, candidate_grads):
                torch.testing.assert_close(candidate_grad, native_grad, rtol=2e-2, atol=2e-2)

        def distributed_timing_stats(event_ms, wall_ms, peak_delta_mb):
            event_max_ms, event_mean_ms = _distributed_float_stats(event_ms, device)
            wall_max_ms, wall_mean_ms = _distributed_float_stats(wall_ms, device)
            max_peak_mb, mean_peak_mb = _distributed_float_stats(peak_delta_mb, device)
            return (
                event_mean_ms,
                event_max_ms,
                wall_mean_ms,
                wall_max_ms,
                mean_peak_mb,
                max_peak_mb,
            )

        native_dispatcher = native_dispatcher_cls(
            config=config, ep_group=pg_collection.ep, num_home_experts=8, num_spare_experts=8
        )
        native_metadata = native_dispatcher.preprocess(expert_map)
        route_payload_stats = _expert_dispatch_payload_stats(
            native_metadata, ep_rank, weight_shape, dtype, device
        )

        def run_native_dispatch():
            with torch.no_grad():
                return native_dispatcher.dispatch(native_metadata, *local_home_weights)

        (
            native_event_latency_ms,
            native_wall_latency_ms,
            native_peak_delta_mb,
            native_dispatched,
        ) = (
            _cuda_event_latency_and_peak_delta_mb(run_native_dispatch, device)
        )
        assert native_event_latency_ms > 0.0
        assert native_wall_latency_ms > 0.0
        assert len(native_dispatched) == 2
        (
            native_event_mean_ms,
            native_event_max_ms,
            native_wall_mean_ms,
            native_wall_max_ms,
            native_mean_peak_mb,
            native_max_peak_mb,
        ) = distributed_timing_stats(
            native_event_latency_ms, native_wall_latency_ms, native_peak_delta_mb
        )

        if global_rank == 0:
            _print_benchmark_line(
                case_name=case_name,
                phase="forward",
                backend="native_all_to_all",
                dtype=dtype,
                weight_shape=weight_shape,
                event_mean_ms=native_event_mean_ms,
                event_max_ms=native_event_max_ms,
                wall_mean_ms=native_wall_mean_ms,
                wall_max_ms=native_wall_max_ms,
                mean_peak_mb=native_mean_peak_mb,
                max_peak_mb=native_max_peak_mb,
                route_payload_stats=route_payload_stats,
                actual_remote_payload_bytes=route_payload_stats.normalized_remote_payload_bytes,
                actual_busiest_sender_payload_bytes=route_payload_stats.busiest_send_payload_bytes,
            )

        def make_native_loss():
            return make_dispatch_loss(native_dispatcher, native_metadata)

        (
            native_backward_event_ms,
            native_backward_wall_ms,
            native_backward_peak_delta_mb,
            _,
        ) = _cuda_event_backward_latency_and_peak_delta_mb(make_native_loss, device)
        assert native_backward_event_ms > 0.0
        assert native_backward_wall_ms > 0.0
        (
            native_backward_event_mean_ms,
            native_backward_event_max_ms,
            native_backward_wall_mean_ms,
            native_backward_wall_max_ms,
            native_backward_mean_peak_mb,
            native_backward_max_peak_mb,
        ) = distributed_timing_stats(
            native_backward_event_ms, native_backward_wall_ms, native_backward_peak_delta_mb
        )

        if global_rank == 0:
            _print_benchmark_line(
                case_name=case_name,
                phase="backward",
                backend="native_all_to_all",
                dtype=dtype,
                weight_shape=weight_shape,
                event_mean_ms=native_backward_event_mean_ms,
                event_max_ms=native_backward_event_max_ms,
                wall_mean_ms=native_backward_wall_mean_ms,
                wall_max_ms=native_backward_wall_max_ms,
                mean_peak_mb=native_backward_mean_peak_mb,
                max_peak_mb=native_backward_max_peak_mb,
                route_payload_stats=route_payload_stats,
                actual_remote_payload_bytes=route_payload_stats.normalized_remote_payload_bytes,
                actual_busiest_sender_payload_bytes=route_payload_stats.busiest_recv_payload_bytes,
            )

        symm_error = None
        symm_dispatched = None
        if symm_dispatcher_cls is None:
            symm_error = "SymmetricMemoryExpertWeightDispatcher is unavailable"
        else:
            symm_error = symm_dispatcher_cls.availability_error()

        if symm_error is None:
            symm_dispatcher = symm_dispatcher_cls(
                config=config, ep_group=pg_collection.ep, num_home_experts=8, num_spare_experts=8
            )
            symm_metadata = symm_dispatcher.preprocess(expert_map)
            symm_dispatcher._get_workspace(local_home_weights[0])
            symm_forward_actual_payload_bytes = _global_payload_bytes(
                len(symm_metadata.unique_remote_source_indices), weight_bytes, device
            )
            symm_forward_busiest_sender_payload_bytes = _max_payload_bytes(
                _unique_remote_send_pair_count(symm_metadata, ep_rank), weight_bytes, device
            )
            symm_backward_actual_payload_bytes = _global_payload_bytes(
                _symm_backward_remote_pair_count(symm_metadata, ep_rank), weight_bytes, device
            )
            symm_backward_busiest_sender_payload_bytes = _max_payload_bytes(
                _remote_recv_pair_count(symm_metadata, ep_rank), weight_bytes, device
            )

            def run_symm_dispatch():
                with torch.no_grad():
                    return symm_dispatcher.dispatch(symm_metadata, *local_home_weights)

            (
                symm_event_latency_ms,
                symm_wall_latency_ms,
                symm_peak_delta_mb,
                symm_dispatched,
            ) = (
                _cuda_event_latency_and_peak_delta_mb(run_symm_dispatch, device)
            )
            assert symm_event_latency_ms > 0.0
            assert symm_wall_latency_ms > 0.0
            _, total_get_calls = _distributed_int_stats(
                symm_dispatcher._debug_low_level_get_calls, device
            )
            _, total_staged_homes = _distributed_int_stats(
                symm_dispatcher._debug_staged_home_experts, device
            )
            _, total_unique_remote_sources = _distributed_int_stats(
                symm_dispatcher._debug_unique_remote_sources, device
            )
            _, total_duplicate_reads_avoided = _distributed_int_stats(
                symm_dispatcher._debug_duplicate_remote_reads_avoided, device
            )
            max_home_weight_barriers, _ = _distributed_int_stats(
                symm_dispatcher._debug_home_weight_barriers, device
            )
            assert total_get_calls > 0
            symm_extra_fields = {
                "symm_low_level_get_calls": total_get_calls,
                "symm_staged_home_experts": total_staged_homes,
                "symm_unique_remote_sources": total_unique_remote_sources,
                "symm_duplicate_remote_reads_avoided": total_duplicate_reads_avoided,
                "symm_home_weight_barriers_per_rank": max_home_weight_barriers,
                "symm_workspace_sync_mode": symm_dispatcher._debug_workspace_sync_mode,
            }
            assert {
                "symm_low_level_get_calls",
                "symm_staged_home_experts",
                "symm_unique_remote_sources",
                "symm_duplicate_remote_reads_avoided",
                "symm_home_weight_barriers_per_rank",
                "symm_workspace_sync_mode",
            } <= symm_extra_fields.keys()
            (
                symm_event_mean_ms,
                symm_event_max_ms,
                symm_wall_mean_ms,
                symm_wall_max_ms,
                symm_mean_peak_mb,
                symm_max_peak_mb,
            ) = distributed_timing_stats(
                symm_event_latency_ms, symm_wall_latency_ms, symm_peak_delta_mb
            )

            for native_weight, symm_weight in zip(native_dispatched, symm_dispatched):
                torch.testing.assert_close(symm_weight, native_weight, rtol=0, atol=0)
            assert_gradients_match(symm_dispatcher, symm_metadata)

            if global_rank == 0:
                _print_benchmark_line(
                    case_name=case_name,
                    phase="forward",
                    backend="symmetric_memory",
                    dtype=dtype,
                    weight_shape=weight_shape,
                    event_mean_ms=symm_event_mean_ms,
                    event_max_ms=symm_event_max_ms,
                    wall_mean_ms=symm_wall_mean_ms,
                    wall_max_ms=symm_wall_max_ms,
                    mean_peak_mb=symm_mean_peak_mb,
                    max_peak_mb=symm_max_peak_mb,
                    route_payload_stats=route_payload_stats,
                    actual_remote_payload_bytes=symm_forward_actual_payload_bytes,
                    actual_busiest_sender_payload_bytes=symm_forward_busiest_sender_payload_bytes,
                    extra_fields=symm_extra_fields,
                )

            def make_symm_loss():
                return make_dispatch_loss(symm_dispatcher, symm_metadata)

            (
                symm_backward_event_ms,
                symm_backward_wall_ms,
                symm_backward_peak_delta_mb,
                _,
            ) = _cuda_event_backward_latency_and_peak_delta_mb(make_symm_loss, device)
            assert symm_backward_event_ms > 0.0
            assert symm_backward_wall_ms > 0.0
            _, total_backward_schedule_entries = _distributed_int_stats(
                symm_dispatcher._debug_backward_schedule_entries, device
            )
            symm_backward_extra_fields = {
                "symm_backward_schedule_entries": total_backward_schedule_entries,
                "symm_workspace_sync_mode": symm_dispatcher._debug_workspace_sync_mode,
            }
            (
                symm_backward_event_mean_ms,
                symm_backward_event_max_ms,
                symm_backward_wall_mean_ms,
                symm_backward_wall_max_ms,
                symm_backward_mean_peak_mb,
                symm_backward_max_peak_mb,
            ) = distributed_timing_stats(
                symm_backward_event_ms, symm_backward_wall_ms, symm_backward_peak_delta_mb
            )

            if global_rank == 0:
                _print_benchmark_line(
                    case_name=case_name,
                    phase="backward",
                    backend="symmetric_memory",
                    dtype=dtype,
                    weight_shape=weight_shape,
                    event_mean_ms=symm_backward_event_mean_ms,
                    event_max_ms=symm_backward_event_max_ms,
                    wall_mean_ms=symm_backward_wall_mean_ms,
                    wall_max_ms=symm_backward_wall_max_ms,
                    mean_peak_mb=symm_backward_mean_peak_mb,
                    max_peak_mb=symm_backward_max_peak_mb,
                    route_payload_stats=route_payload_stats,
                    actual_remote_payload_bytes=symm_backward_actual_payload_bytes,
                    actual_busiest_sender_payload_bytes=symm_backward_busiest_sender_payload_bytes,
                    extra_fields=symm_backward_extra_fields,
                )
        elif global_rank == 0:
            for phase in ("forward", "backward"):
                print(
                    "BENCHMARK balanced_moe_expert_weight_dispatch "
                    f"case={case_name} phase={phase} backend=symmetric_memory "
                    f"skipped reason={symm_error!r}",
                    flush=True,
                )

        hybridep_error = None
        if hybridep_dispatcher_cls is None:
            hybridep_error = "HybridEPExpertWeightDispatcher is unavailable"
        else:
            hybridep_error = hybridep_dispatcher_cls.availability_error()

        if hybridep_error is None:
            hybridep_dispatcher = hybridep_dispatcher_cls(
                config=config, ep_group=pg_collection.ep, num_home_experts=8, num_spare_experts=8
            )
            hybridep_metadata = hybridep_dispatcher.preprocess(expert_map)
            hybridep_pair_payload_bytes = _hybridep_padded_weight_payload_bytes(
                hybridep_dispatcher, local_home_weights[0]
            )
            hybridep_actual_payload_bytes = _global_payload_bytes(
                _unique_remote_send_pair_count(hybridep_metadata, ep_rank),
                hybridep_pair_payload_bytes,
                device,
            )
            hybridep_forward_busiest_sender_payload_bytes = _max_payload_bytes(
                _unique_remote_send_pair_count(hybridep_metadata, ep_rank),
                hybridep_pair_payload_bytes,
                device,
            )
            hybridep_backward_busiest_sender_payload_bytes = _max_payload_bytes(
                len(hybridep_metadata.unique_remote_source_indices),
                hybridep_pair_payload_bytes,
                device,
            )

            def run_hybridep_dispatch():
                with torch.no_grad():
                    return hybridep_dispatcher.dispatch(hybridep_metadata, *local_home_weights)

            (
                hybridep_event_latency_ms,
                hybridep_wall_latency_ms,
                hybridep_peak_delta_mb,
                hybridep_dispatched,
            ) = (
                _cuda_event_latency_and_peak_delta_mb(run_hybridep_dispatch, device)
            )
            assert hybridep_event_latency_ms > 0.0
            assert hybridep_wall_latency_ms > 0.0
            (
                hybridep_event_mean_ms,
                hybridep_event_max_ms,
                hybridep_wall_mean_ms,
                hybridep_wall_max_ms,
                hybridep_mean_peak_mb,
                hybridep_max_peak_mb,
            ) = distributed_timing_stats(
                hybridep_event_latency_ms, hybridep_wall_latency_ms, hybridep_peak_delta_mb
            )

            for native_weight, hybridep_weight in zip(native_dispatched, hybridep_dispatched):
                torch.testing.assert_close(hybridep_weight, native_weight, rtol=0, atol=0)
            assert_gradients_match(hybridep_dispatcher, hybridep_metadata)

            hybridep_extra_fields = {
                "hybridep_unique_remote_pairs": int(
                    hybridep_actual_payload_bytes / hybridep_pair_payload_bytes
                ),
                "hybridep_padded_pair_payload_mib": f"{hybridep_pair_payload_bytes / (1024**2):.2f}",
            }

            if global_rank == 0:
                _print_benchmark_line(
                    case_name=case_name,
                    phase="forward",
                    backend="hybridep",
                    dtype=dtype,
                    weight_shape=weight_shape,
                    event_mean_ms=hybridep_event_mean_ms,
                    event_max_ms=hybridep_event_max_ms,
                    wall_mean_ms=hybridep_wall_mean_ms,
                    wall_max_ms=hybridep_wall_max_ms,
                    mean_peak_mb=hybridep_mean_peak_mb,
                    max_peak_mb=hybridep_max_peak_mb,
                    route_payload_stats=route_payload_stats,
                    actual_remote_payload_bytes=hybridep_actual_payload_bytes,
                    actual_busiest_sender_payload_bytes=hybridep_forward_busiest_sender_payload_bytes,
                    extra_fields=hybridep_extra_fields,
                )

            def make_hybridep_loss():
                return make_dispatch_loss(hybridep_dispatcher, hybridep_metadata)

            (
                hybridep_backward_event_ms,
                hybridep_backward_wall_ms,
                hybridep_backward_peak_delta_mb,
                _,
            ) = _cuda_event_backward_latency_and_peak_delta_mb(make_hybridep_loss, device)
            assert hybridep_backward_event_ms > 0.0
            assert hybridep_backward_wall_ms > 0.0
            (
                hybridep_backward_event_mean_ms,
                hybridep_backward_event_max_ms,
                hybridep_backward_wall_mean_ms,
                hybridep_backward_wall_max_ms,
                hybridep_backward_mean_peak_mb,
                hybridep_backward_max_peak_mb,
            ) = distributed_timing_stats(
                hybridep_backward_event_ms,
                hybridep_backward_wall_ms,
                hybridep_backward_peak_delta_mb,
            )

            if global_rank == 0:
                _print_benchmark_line(
                    case_name=case_name,
                    phase="backward",
                    backend="hybridep",
                    dtype=dtype,
                    weight_shape=weight_shape,
                    event_mean_ms=hybridep_backward_event_mean_ms,
                    event_max_ms=hybridep_backward_event_max_ms,
                    wall_mean_ms=hybridep_backward_wall_mean_ms,
                    wall_max_ms=hybridep_backward_wall_max_ms,
                    mean_peak_mb=hybridep_backward_mean_peak_mb,
                    max_peak_mb=hybridep_backward_max_peak_mb,
                    route_payload_stats=route_payload_stats,
                    actual_remote_payload_bytes=hybridep_actual_payload_bytes,
                    actual_busiest_sender_payload_bytes=hybridep_backward_busiest_sender_payload_bytes,
                    extra_fields=hybridep_extra_fields,
                )
        elif global_rank == 0:
            for phase in ("forward", "backward"):
                print(
                    "BENCHMARK balanced_moe_expert_weight_dispatch "
                    f"case={case_name} phase={phase} backend=hybridep "
                    f"skipped reason={hybridep_error!r}",
                    flush=True,
                )
    finally:
        Utils.destroy_model_parallel()
