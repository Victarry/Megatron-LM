# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Critical GPU parity tests for the BalancedMoELayer MoonEP data plane."""

import os
import time

import pytest
import torch
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.moe.balanced_moe_layer import BalancedMoELayer
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.moonep_backend import adapt_mcore_route
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


def _require_moonep(world_size=4):
    if not torch.cuda.is_available():
        pytest.skip("MoonEP tests require CUDA.")
    if int(os.environ.get("WORLD_SIZE", "1")) < world_size:
        pytest.skip(f"Run with torchrun --nproc-per-node={world_size}.")
    try:
        import moonep
    except ImportError as exc:
        pytest.fail(f"MoonEP GPU runtime is not importable: {exc}")
    return moonep


def _submodules(num_experts, shared_expert=False):
    spec = get_gpt_layer_with_transformer_engine_submodules(
        num_experts=num_experts,
        moe_grouped_gemm=True,
    ).mlp
    submodules = get_submodules(spec)
    assert isinstance(submodules, MoESubmodules)
    if not shared_expert:
        submodules.shared_experts = None
    return submodules


def _config(num_experts, *, moonep, shared_expert=False):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=512,
        num_attention_heads=8,
        num_moe_experts=num_experts,
        moe_ffn_hidden_size=512,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        moe_router_dtype="fp32",
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        expert_model_parallel_size=4,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        add_bias_linear=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        fp16=False,
        activation_func=F.silu,
        gated_linear_unit=True,
        use_transformer_engine_op_fuser=True,
        moe_mlp_glu_interleave_size=32,
    )
    if shared_expert:
        config.moe_shared_expert_intermediate_size = 512
        config.moe_shared_expert_overlap = False
    if moonep:
        config.moe_use_balanced_layer = True
        config.moe_balance_backend = "moonep"
        config.moe_balance_moonep_token_padding = 128
    return config


def _grad(param):
    main_grad = getattr(param, "main_grad", None)
    return main_grad if main_grad is not None else param.grad


def _zero_layer_grads(layer):
    for param in layer.parameters():
        if param.grad is not None:
            param.grad = None
        main_grad = getattr(param, "main_grad", None)
        if main_grad is not None:
            main_grad.zero_()
        if hasattr(param, "grad_added_to_main_grad"):
            param.grad_added_to_main_grad = False


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


def _benchmark_phase(fn, *, warmup=5, iterations=20):
    for _ in range(warmup):
        _, event = fn()
        torch.cuda.current_stream().wait_event(event)
        torch.cuda.current_stream().synchronize()
    torch.distributed.barrier()

    event_latencies = []
    wall_latencies = []
    result = None
    for _ in range(iterations):
        torch.distributed.barrier()
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        wall_start = time.perf_counter()
        result, event = fn()
        torch.cuda.current_stream().wait_event(event)
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        end.synchronize()
        wall_latencies.append((time.perf_counter() - wall_start) * 1000.0)
        event_latencies.append(start.elapsed_time(end))
    event_ms = torch.tensor(
        sum(event_latencies) / iterations, dtype=torch.float64, device="cuda"
    )
    wall_ms = torch.tensor(
        sum(wall_latencies) / iterations, dtype=torch.float64, device="cuda"
    )
    torch.distributed.all_reduce(event_ms, op=torch.distributed.ReduceOp.MAX)
    torch.distributed.all_reduce(wall_ms, op=torch.distributed.ReduceOp.MAX)
    return result, float(event_ms.item()), float(wall_ms.item())


def _print_moonep_benchmark(phase, event_ms, wall_ms, local_payload_bytes):
    payload = torch.tensor(local_payload_bytes, dtype=torch.int64, device="cuda")
    torch.distributed.all_reduce(payload, op=torch.distributed.ReduceOp.MAX)
    busiest_payload = int(payload.item())
    if torch.distributed.get_rank() == 0:
        bus_bw = busiest_payload / (event_ms / 1000.0) / (1024**3)
        print(
            "BENCHMARK balanced_moe_moonep "
            f"phase={phase} backend=moonep "
            f"event_max_rank_ms={event_ms:.4f} wall_max_rank_ms={wall_ms:.4f} "
            f"busiest_rank_actual_remote_payload_mib={busiest_payload / (1024**2):.2f} "
            f"actual_bus_bw_event_gib_s={bus_bw:.2f}",
            flush=True,
        )


@pytest.mark.internal
def test_moonep_route_adapter_dispatch_combine_forward_backward_parity():
    moonep = _require_moonep()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    buffer = None
    try:
        group = get_default_pg_collection().ep
        rank = group.rank()
        world_size = group.size()
        device = torch.device("cuda", torch.cuda.current_device())
        num_tokens, hidden_size, topk = 96, 128, 2
        num_experts = world_size * 4
        num_spares = num_experts // world_size

        generator = torch.Generator(device=device).manual_seed(20260728 + rank)
        hidden = torch.randn(
            num_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
            requires_grad=True,
        )
        probs = torch.rand(
            num_tokens,
            num_experts,
            dtype=torch.float32,
            device=device,
            generator=generator,
            requires_grad=True,
        )
        token_ids = torch.arange(num_tokens, dtype=torch.int64, device=device)
        expert_ids = torch.stack(
            (
                token_ids % num_experts,
                (token_ids * 7 + 3) % num_experts,
            ),
            dim=1,
        )
        routing_map = torch.zeros(
            num_tokens, num_experts, dtype=torch.bool, device=device
        ).scatter_(1, expert_ids, True)
        routing_map[:3].zero_()

        route_weights, topk_ids, tokens_per_expert = adapt_mcore_route(
            probs, routing_map, topk
        )
        buffer = moonep.Buffer(
            S=num_tokens,
            H=hidden_size,
            K=topk,
            E=num_experts,
            B=num_spares,
            num_ep_ranks=world_size,
            group=group,
        )
        dispatched, dispatched_weights, cu_seqlens, plan = buffer.dispatch(
            hidden, route_weights, topk_ids, tokens_per_expert
        )

        group_experts = torch.arange(
            num_experts + num_spares, dtype=torch.int32, device=device
        )
        group_experts[num_experts:] = plan.experts_to_copy[rank].clamp_min(0)
        starts = torch.cat(
            (torch.zeros(1, dtype=cu_seqlens.dtype, device=device), cu_seqlens[:-1])
        )
        scales = torch.ones(dispatched.shape[0], dtype=torch.bfloat16, device=device)
        for group_index, (start, end) in enumerate(zip(starts.tolist(), cu_seqlens.tolist())):
            scales[start:end] = (int(group_experts[group_index]) + 1) / num_experts
        base = (dispatched * scales[:, None]).to(torch.bfloat16)
        expert_output = (base.float() * dispatched_weights[:, None]).to(torch.bfloat16)
        actual_output, _, _ = buffer.combine(plan=plan, hidden_nvsh=expert_output)

        hidden_ref = hidden.detach().clone().requires_grad_(True)
        probs_ref = probs.detach().clone().requires_grad_(True)
        valid = routing_map.gather(1, topk_ids.long())
        dense_weights = probs_ref.gather(1, topk_ids.long()).masked_fill(~valid, 0.0)
        reference_output = torch.zeros_like(hidden_ref)
        for slot in range(topk):
            scale = ((topk_ids[:, slot].float() + 1.0) / num_experts).to(torch.bfloat16)
            slot_base = (hidden_ref * scale[:, None]).to(torch.bfloat16)
            reference_output.add_(
                (slot_base.float() * dense_weights[:, slot, None]).to(torch.bfloat16)
            )
        torch.testing.assert_close(actual_output, reference_output)

        output_grad = torch.randn(
            actual_output.shape,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        grad_expert, _, _, _ = buffer.dispatch(output_grad, plan=plan)
        grad_dispatched = (
            grad_expert.float() * dispatched_weights[:, None] * scales[:, None].float()
        ).to(torch.bfloat16)
        grad_dispatched_weights = (grad_expert.float() * base.float()).sum(dim=-1)
        grad_hidden, grad_route_weights, _ = buffer.combine(
            plan=plan,
            hidden_nvsh=grad_dispatched,
            route_weights_nvs=grad_dispatched_weights,
        )
        route_weights.backward(grad_route_weights)
        reference_output.backward(output_grad)

        torch.testing.assert_close(grad_hidden, hidden_ref.grad)
        torch.testing.assert_close(probs.grad, probs_ref.grad)
    finally:
        if buffer is not None:
            buffer.destroy()
        Utils.destroy_model_parallel()


@pytest.mark.internal
def test_moonep_vmm_shadow_prefetch_and_fanout_fp32_grad_reduce():
    moonep = _require_moonep()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    buffer = None
    table = None
    try:
        group = get_default_pg_collection().ep
        rank = group.rank()
        world_size = group.size()
        device = torch.device("cuda", torch.cuda.current_device())
        num_experts = world_size
        num_spares = 1
        projection_shape = (1024, 1024)
        table = moonep.DistributedExpertTable(
            num_experts,
            num_spares,
            [projection_shape],
            rank=rank,
            world_size=world_size,
            group=group,
        )
        home_weight = torch.full(
            projection_shape,
            rank + 1,
            dtype=torch.bfloat16,
            device=device,
        )
        table.refresh_home_weights([[home_weight]])

        buffer = moonep.Buffer(
            S=128,
            H=128,
            K=1,
            E=num_experts,
            B=num_spares,
            num_ep_ranks=world_size,
            group=group,
        )
        hidden = torch.zeros(128, 128, dtype=torch.bfloat16, device=device)
        weights = torch.ones(128, 1, dtype=torch.float32, device=device)
        ids = torch.zeros(128, 1, dtype=torch.int32, device=device)
        counts = torch.bincount(ids.flatten(), minlength=num_experts).to(torch.int32)
        _, _, _, plan = buffer.dispatch(hidden, weights, ids, counts)
        plan.experts_to_copy.fill_(0)

        buffer.prefetch_weights(plan, table.full_weights)
        torch.testing.assert_close(
            table.full_weights[0][num_experts],
            table.full_weights[0][0],
            rtol=0,
            atol=0,
        )

        table.full_grads[0].zero_()
        table.full_grads[0][num_experts].fill_(rank + 1)
        buffer.reduce_grads(plan, table.full_grads, table.reduce_buffers)
        local_grad = table.full_grads[0][table.local_home_slice][0]
        expected = sum(range(1, world_size + 1)) if rank == 0 else 0
        torch.testing.assert_close(
            local_grad,
            torch.full_like(local_grad, expected),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            table.full_grads[0][num_experts],
            torch.zeros_like(table.full_grads[0][num_experts]),
            rtol=0,
            atol=0,
        )
    finally:
        if table is not None:
            table.close()
        if buffer is not None:
            buffer.destroy()
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize(
    "case_name,projection_shape",
    [
        ("representative_fc1_proxy", (4096, 2048)),
        ("representative_fc2_proxy", (2048, 4096)),
    ],
)
def test_moonep_data_plane_latency_and_actual_payload_bandwidth(
    case_name, projection_shape
):
    del case_name
    moonep = _require_moonep()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    buffer = None
    table = None
    try:
        group = get_default_pg_collection().ep
        rank = group.rank()
        world_size = group.size()
        device = torch.device("cuda", torch.cuda.current_device())
        num_experts = world_size * 2
        num_spares = 2
        num_tokens, hidden_size = 1024, 2048
        table = moonep.DistributedExpertTable(
            num_experts,
            num_spares,
            [projection_shape],
            rank=rank,
            world_size=world_size,
            group=group,
        )
        local_weights = [
            torch.full(
                projection_shape,
                rank * num_spares + index + 1,
                dtype=torch.bfloat16,
                device=device,
            )
            for index in range(num_spares)
        ]
        table.refresh_home_weights([local_weights])
        buffer = moonep.Buffer(
            S=num_tokens,
            H=hidden_size,
            K=1,
            E=num_experts,
            B=num_spares,
            num_ep_ranks=world_size,
            group=group,
        )
        hidden = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        weights = torch.ones(num_tokens, 1, dtype=torch.float32, device=device)
        ids = (
            torch.arange(num_tokens, dtype=torch.int32, device=device) % num_experts
        ).view(-1, 1)
        counts = torch.bincount(ids.flatten(), minlength=num_experts).to(torch.int32)
        dispatched, _, _, plan = buffer.dispatch(hidden, weights, ids, counts)

        # Deterministic ring fan-out: every rank prefetches both experts owned by
        # its next peer, so the busiest-rank payload is unambiguous.
        for destination in range(world_size):
            owner = (destination + 1) % world_size
            plan.experts_to_copy[destination].copy_(
                torch.arange(
                    owner * num_spares,
                    (owner + 1) * num_spares,
                    dtype=torch.int32,
                    device=device,
                )
            )

        projection_numels = (int(torch.Size(projection_shape).numel()),)
        payload = moonep.calculate_payload_bytes(
            plan,
            rank=rank,
            hidden_size=hidden_size,
            projection_numels=projection_numels,
            group=group,
        )

        def dispatch_phase():
            result = buffer.dispatch(
                hidden,
                weights,
                ids,
                counts,
                async_finish=True,
                zero_copy=False,
            )
            return result, result[-1]

        dispatch_result, event_ms, wall_ms = _benchmark_phase(dispatch_phase)
        dispatched, dispatched_weights, _, measured_plan, _ = dispatch_result
        measured_payload = moonep.calculate_payload_bytes(
            measured_plan,
            rank=rank,
            hidden_size=hidden_size,
            projection_numels=projection_numels,
            group=group,
        )
        _print_moonep_benchmark(
            "planning_token_dispatch_forward",
            event_ms,
            wall_ms,
            measured_payload.dispatch,
        )

        def prefetch_phase():
            event = buffer.prefetch_weights(
                plan, table.full_weights, async_finish=True
            )
            return None, event

        _, event_ms, wall_ms = _benchmark_phase(prefetch_phase)
        _print_moonep_benchmark(
            "expert_weight_prefetch_forward", event_ms, wall_ms, payload.prefetch
        )

        def combine_phase():
            result = buffer.combine(
                plan=measured_plan,
                hidden_nvsh=dispatched,
                async_finish=True,
                zero_copy=False,
            )
            return result, result[-1]

        _, event_ms, wall_ms = _benchmark_phase(combine_phase)
        _print_moonep_benchmark(
            "token_combine_forward", event_ms, wall_ms, measured_payload.combine
        )

        def redispatch_phase():
            result = buffer.dispatch(
                hidden,
                plan=measured_plan,
                async_finish=True,
                zero_copy=False,
            )
            return result, result[-1]

        redispatch_result, event_ms, wall_ms = _benchmark_phase(redispatch_phase)
        grad_dispatched = redispatch_result[0]
        _print_moonep_benchmark(
            "output_grad_redispatch_backward",
            event_ms,
            wall_ms,
            measured_payload.redispatch,
        )

        def hidden_combine_phase():
            result = buffer.combine(
                plan=measured_plan,
                hidden_nvsh=grad_dispatched,
                route_weights_nvs=dispatched_weights,
                async_finish=True,
                zero_copy=False,
            )
            return result, result[-1]

        _, event_ms, wall_ms = _benchmark_phase(hidden_combine_phase)
        _print_moonep_benchmark(
            "hidden_router_grad_combine_backward",
            event_ms,
            wall_ms,
            measured_payload.combine_with_route_weights,
        )

        def grad_reduce_phase():
            table.full_grads[0][num_experts:].fill_(rank + 1)
            torch.cuda.current_stream().synchronize()
            event = buffer.reduce_grads(
                plan,
                table.full_grads,
                table.reduce_buffers,
                async_finish=True,
            )
            return None, event

        _, event_ms, wall_ms = _benchmark_phase(grad_reduce_phase)
        _print_moonep_benchmark(
            "expert_weight_grad_reduce_backward",
            event_ms,
            wall_ms,
            payload.grad_reduce,
        )
    finally:
        if table is not None:
            table.close()
        if buffer is not None:
            buffer.destroy()
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_TE, reason="MoonEP full-layer parity requires Transformer Engine.")
@pytest.mark.skipif(
    not is_te_min_version("2.14.0"),
    reason="MoonEPGroupedMLP requires Transformer Engine 2.14 or later.",
)
def test_balanced_moe_moonep_full_layer_parity_refresh_and_state_dict():
    _require_moonep()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    try:
        num_experts = 32
        _set_random_seed(seed_=1234, data_parallel_random_init=False)
        baseline = MoELayer(
            _config(num_experts, moonep=False, shared_expert=True),
            _submodules(num_experts, shared_expert=True),
            layer_number=1,
        )
        balanced = BalancedMoELayer(
            _config(num_experts, moonep=True, shared_expert=True),
            _submodules(num_experts, shared_expert=True),
            layer_number=1,
        )
        baseline.cuda().to(dtype=torch.bfloat16)
        balanced.cuda().to(dtype=torch.bfloat16)
        baseline.experts.enable_runtime_weight_main_grad_accumulation()
        assert baseline.state_dict().keys() == balanced.state_dict().keys()
        balanced.load_state_dict(baseline.state_dict(), strict=True)
        baseline_sharded = _sharded_metadata(baseline.sharded_state_dict())
        balanced_sharded = _sharded_metadata(balanced.sharded_state_dict())
        assert balanced_sharded == baseline_sharded

        for iteration in range(2):
            _zero_layer_grads(baseline)
            _zero_layer_grads(balanced)
            if iteration == 1:
                for baseline_param, balanced_param in zip(
                    baseline.experts.parameters(), balanced.experts.parameters()
                ):
                    baseline_param.data.add_(0.015625)
                    balanced_param.data.add_(0.015625)
                balanced.is_first_microbatch = True

            generator = torch.Generator(device="cuda").manual_seed(9000 + iteration)
            hidden = torch.randn(
                8,
                4,
                512,
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
                requires_grad=True,
            )
            balanced_hidden = hidden.detach().clone().requires_grad_(True)
            output_grad = torch.randn(
                hidden.shape,
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            padding_mask = torch.zeros(4, 8, dtype=torch.bool, device="cuda")
            padding_mask[:, -1] = True

            # Complete each independent reference graph before starting the
            # candidate so parameter/main_grad observations cannot overlap.
            baseline_output, _ = baseline(hidden, padding_mask=padding_mask)
            baseline_output.backward(output_grad)

            balanced_output, _ = balanced(balanced_hidden, padding_mask=padding_mask)
            balanced_output.backward(output_grad)

            torch.testing.assert_close(balanced_output, baseline_output)
            torch.testing.assert_close(balanced_hidden.grad, hidden.grad)
            for baseline_param, balanced_param in zip(
                baseline.router.parameters(), balanced.router.parameters()
            ):
                torch.testing.assert_close(_grad(balanced_param), _grad(baseline_param))
            for baseline_param, balanced_param in zip(
                baseline.experts.parameters(), balanced.experts.parameters()
            ):
                assert _grad(balanced_param).dtype == torch.float32
                torch.testing.assert_close(
                    _grad(balanced_param),
                    _grad(baseline_param).float(),
                )
            for baseline_param, balanced_param in zip(
                baseline.shared_experts.parameters(), balanced.shared_experts.parameters()
            ):
                torch.testing.assert_close(
                    _grad(balanced_param),
                    _grad(baseline_param),
                )
            assert baseline.state_dict().keys() == balanced.state_dict().keys()
    finally:
        Utils.destroy_model_parallel()
