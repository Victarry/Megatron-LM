# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib
import os

import pytest
import torch

_BENCHMARK_WARMUP_ITERS = 5
_BENCHMARK_ITERS = 20


def _require_offloading_planner():
    try:
        return importlib.import_module("megatron.core.transformer.moe.offloading_planner")
    except ModuleNotFoundError as exc:
        pytest.fail(
            "megatron.core.transformer.moe.offloading_planner is required for "
            "BalancedMoELayer planner parity tests."
        )


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


def _current_cuda_device():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    return torch.device("cuda", torch.cuda.current_device())


def _effective_column_for_home(home_idx, ep_size, home_per_rank, spare_per_rank):
    ep_rank = home_idx // home_per_rank
    local_home_idx = home_idx % home_per_rank
    return ep_rank * (home_per_rank + spare_per_rank) + local_home_idx


def _effective_column_for_spare(spare_idx, ep_size, home_per_rank, spare_per_rank):
    ep_rank = spare_idx // spare_per_rank
    local_spare_idx = spare_idx % spare_per_rank
    return ep_rank * (home_per_rank + spare_per_rank) + home_per_rank + local_spare_idx


def undo_reroute(
    rerouting_map: torch.Tensor,
    rerouted_probs: torch.Tensor,
    expert_offloading_map: torch.Tensor,
    *,
    ep_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold spare columns back into their mapped home columns."""

    num_home_experts, num_spare_experts = expert_offloading_map.shape
    assert num_home_experts % ep_size == 0
    assert num_spare_experts % ep_size == 0
    home_per_rank = num_home_experts // ep_size
    spare_per_rank = num_spare_experts // ep_size

    home_map = torch.zeros(
        rerouting_map.shape[0], num_home_experts, dtype=rerouting_map.dtype, device=rerouting_map.device
    )
    home_probs = torch.zeros(
        rerouted_probs.shape[0], num_home_experts, dtype=rerouted_probs.dtype, device=rerouted_probs.device
    )

    for home_idx in range(num_home_experts):
        effective_col = _effective_column_for_home(home_idx, ep_size, home_per_rank, spare_per_rank)
        home_map[:, home_idx] |= rerouting_map[:, effective_col]
        home_probs[:, home_idx] += rerouted_probs[:, effective_col]

    home_indices, spare_indices = torch.where(expert_offloading_map)
    for home_idx, spare_idx in zip(home_indices.tolist(), spare_indices.tolist()):
        effective_col = _effective_column_for_spare(spare_idx, ep_size, home_per_rank, spare_per_rank)
        home_map[:, home_idx] |= rerouting_map[:, effective_col]
        home_probs[:, home_idx] += rerouted_probs[:, effective_col]

    return home_map, home_probs


def _make_local_routing(ep_size: int, topk: int, dtype: torch.dtype, device: torch.device):
    num_home_experts = 8
    tokens_per_rank = 12
    routing = torch.zeros(ep_size, tokens_per_rank, num_home_experts, dtype=torch.bool, device=device)
    probs = torch.zeros(ep_size, tokens_per_rank, num_home_experts, dtype=dtype, device=device)

    for ep_rank in range(ep_size):
        for token_idx in range(tokens_per_rank):
            if token_idx < 8:
                first = 0
            else:
                first = (ep_rank * 2 + token_idx) % num_home_experts
            experts = [first]
            if topk == 2:
                second = (first + 3 + ep_rank) % num_home_experts
                if second == first:
                    second = (second + 1) % num_home_experts
                experts.append(second)
            values = [1.0] if topk == 1 else [0.625, 0.375]
            for expert_idx, prob in zip(experts, values):
                routing[ep_rank, token_idx, expert_idx] = True
                probs[ep_rank, token_idx, expert_idx] = prob

    return routing, probs


def _make_routing_from_counts(
    counts: torch.Tensor, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    ep_size, num_home_experts = counts.shape
    tokens_per_rank = int(counts.sum(dim=1).max().item())
    routing = torch.zeros(
        ep_size, tokens_per_rank, num_home_experts, dtype=torch.bool, device=device
    )
    probs = torch.zeros(ep_size, tokens_per_rank, num_home_experts, dtype=dtype, device=device)

    for ep_rank in range(ep_size):
        offset = 0
        for expert_idx, count in enumerate(counts[ep_rank].tolist()):
            end = offset + int(count)
            if end > offset:
                routing[ep_rank, offset:end, expert_idx] = True
                probs[ep_rank, offset:end, expert_idx] = 1.0
            offset = end

    return routing, probs


def _assert_equivalent_plan(
    original_map,
    original_probs,
    rerouting_map,
    rerouted_probs,
    expert_offloading_map,
    ep_size,
):
    restored_map, restored_probs = undo_reroute(
        rerouting_map, rerouted_probs, expert_offloading_map, ep_size=ep_size
    )
    assert torch.equal(restored_map, original_map)
    torch.testing.assert_close(restored_probs, original_probs, rtol=0, atol=0)
    assert expert_offloading_map.sum(dim=0).max().item() <= 1
    assert torch.equal(rerouting_map.sum(dim=1), original_map.sum(dim=1))
    torch.testing.assert_close(rerouted_probs.sum(dim=1), original_probs.sum(dim=1), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offloading planner uses CUDA kernels")
@pytest.mark.parametrize(
    "topk,assignment_algorithm,spare_per_ep",
    [
        (1, "approx_bin_packing", 1),
        (2, "one_shot_greedy", 2),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_offloading_planner_equivalence(topk, assignment_algorithm, spare_per_ep, dtype):
    planner = _require_offloading_planner()
    device = torch.device("cuda")
    ep_size = 4
    routing, probs = _make_local_routing(ep_size, topk, dtype, device)
    counts = routing.sum(dim=1).to(torch.int32)

    rerouting_by_rank = []
    offloading_maps = []
    for ep_rank in range(ep_size):
        rerouting_map, rerouted_probs, expert_offloading_map = planner.gen_offloading_plan(
            routing[ep_rank],
            probs[ep_rank],
            counts,
            ep_rank,
            num_ep_ranks=ep_size,
            num_spare_experts_per_ep_rank=spare_per_ep,
            assignment_algorithm=assignment_algorithm,
        )
        _assert_equivalent_plan(
            routing[ep_rank],
            probs[ep_rank],
            rerouting_map,
            rerouted_probs,
            expert_offloading_map,
            ep_size,
        )
        rerouting_by_rank.append(rerouting_map)
        offloading_maps.append(expert_offloading_map)

    assert all(torch.equal(offloading_maps[0], m) for m in offloading_maps[1:])

    before_rank_load = routing.reshape(-1, routing.shape[-1]).sum(dim=0).reshape(ep_size, -1).sum(dim=1)
    effective = torch.cat(rerouting_by_rank, dim=0)
    effective_per_rank = routing.shape[-1] // ep_size + spare_per_ep
    after_rank_load = effective.sum(dim=0).reshape(ep_size, effective_per_rank).sum(dim=1)
    assert after_rank_load.max() <= before_rank_load.max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offloading planner uses CUDA kernels")
def test_approx_bin_packing_keeps_one_home_per_spare_slot():
    planner = _require_offloading_planner()
    device = torch.device("cuda")
    ep_size = 8
    num_home_experts = 32
    counts = torch.zeros(ep_size, num_home_experts, dtype=torch.int32, device=device)
    counts[0, 0:4] = torch.tensor([80, 90, 110, 120], dtype=torch.int32, device=device)
    counts[0, 8:32] = 50
    routing, probs = _make_routing_from_counts(counts, torch.float32, device)

    rerouting_map, rerouted_probs, expert_offloading_map = planner.gen_offloading_plan(
        routing[0],
        probs[0],
        counts,
        ep_rank=0,
        num_ep_ranks=ep_size,
        num_spare_experts_per_ep_rank=1,
        assignment_algorithm="approx_bin_packing",
    )

    _assert_equivalent_plan(
        routing[0], probs[0], rerouting_map, rerouted_probs, expert_offloading_map, ep_size
    )
    assert expert_offloading_map.any(dim=0).sum().item() == 1
    assert expert_offloading_map[:, 1].sum().item() == 1
    assert expert_offloading_map[3, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offloading planner uses CUDA kernels")
def test_random_offloading_equivalence():
    planner = _require_offloading_planner()
    device = torch.device("cuda")
    ep_size = 4
    topk = 2
    routing, probs = _make_local_routing(ep_size, topk, torch.float32, device)
    counts = routing.sum(dim=1).to(torch.int32)

    rerouting_map, rerouted_probs, expert_offloading_map = planner.gen_random_offloading_plan(
        routing[0],
        probs[0],
        counts,
        ep_rank=0,
        ep=ep_size,
        spare_expert_per_ep_rank=1,
    )
    _assert_equivalent_plan(
        routing[0], probs[0], rerouting_map, rerouted_probs, expert_offloading_map, ep_size
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offloading planner uses CUDA kernels")
def test_approx_bin_packing_rejects_multiple_spares_per_ep():
    planner = _require_offloading_planner()
    device = torch.device("cuda")
    ep_size = 4
    routing, probs = _make_local_routing(ep_size, topk=1, dtype=torch.float32, device=device)
    counts = routing.sum(dim=1).to(torch.int32)

    with pytest.raises(ValueError, match="approx_bin_packing"):
        planner.gen_offloading_plan(
            routing[0],
            probs[0],
            counts,
            ep_rank=0,
            num_ep_ranks=ep_size,
            num_spare_experts_per_ep_rank=2,
            assignment_algorithm="approx_bin_packing",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offloading planner uses CUDA kernels")
def test_offloading_planner_kernel_latency_benchmark():
    if os.getenv("MEGATRON_BALANCED_MOE_PLANNER_EAGER") == "1":
        pytest.skip("Planner latency benchmark targets the production compiled path.")

    planner = _require_offloading_planner()
    device = _current_cuda_device()
    ep_size = 8
    num_home_experts = 32
    counts = torch.full((ep_size, num_home_experts), 8, dtype=torch.int32, device=device)
    counts[0, 0:4] = torch.tensor([512, 480, 448, 416], dtype=torch.int32, device=device)
    counts[0, 4:] = 16
    routing, probs = _make_routing_from_counts(counts, torch.float32, device)

    def run_planner():
        return planner.gen_offloading_plan(
            routing[0],
            probs[0],
            counts,
            ep_rank=0,
            num_ep_ranks=ep_size,
            num_spare_experts_per_ep_rank=1,
            assignment_algorithm="approx_bin_packing",
        )

    latency_ms, (rerouting_map, rerouted_probs, expert_offloading_map) = _cuda_event_latency_ms(
        run_planner
    )
    assert latency_ms > 0.0
    assert torch.isfinite(torch.tensor(latency_ms, device=device))
    _assert_equivalent_plan(
        routing[0], probs[0], rerouting_map, rerouted_probs, expert_offloading_map, ep_size
    )
    print(
        "BENCHMARK balanced_moe_planner_kernel "
        f"algorithm=approx_bin_packing ep_size={ep_size} "
        f"num_home_experts={num_home_experts} local_tokens={routing.shape[1]} "
        f"warmup_iters={_BENCHMARK_WARMUP_ITERS} iters={_BENCHMARK_ITERS} "
        f"mean_ms={latency_ms:.4f}",
        flush=True,
    )
